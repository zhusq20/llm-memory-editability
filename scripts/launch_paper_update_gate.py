#!/usr/bin/env python3
"""Run the two authorized update-gate arms on two independently idle GPUs.

The user explicitly requested direct training without short training or reload
audits. This controller therefore runs train, then an optional R1--4 evaluation.
It never calls preflight, smoke, audit, grid-audit, or canonical cloud reporting.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from launch_dual_gate import docker, read, sha, snapshot, utc, write  # noqa: E402

from llm_memory_editability.paper_update_contract import validate_config  # noqa: E402

DOCKER = ["docker", "--context", "lm-memory"]
DATA_TARGET = (
    "/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability"
)


def credential_archive(path):
    """Install private credentials without placing their content in argv or logs."""
    path = Path(path)
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("A private W&B credential file with mode 0600 or 0400 is required")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(info)
        value = path.read_bytes().strip()
        if not value:
            raise ValueError("Empty tracking credential")
        info = tarfile.TarInfo("run/secrets/wandb")
        info.size, info.mode, info.uid, info.gid = len(value), 0o400, os.getuid(), os.getgid()
        archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


def admit(gpus, rows, blocked):
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Exactly two distinct GPUs are required")
    selected = {}
    for gpu in gpus:
        row = rows.get(gpu)
        if row is None or blocked.get(gpu) or row["used"] > 512 or row["utilization"] > 5:
            raise ValueError(f"GPU {gpu} is occupied or reserved; no task was changed")
        if row["free"] < 18000:
            raise ValueError(f"GPU {gpu} has less than 18000 MiB available")
        selected[gpu] = row["uuid"]
    return selected


def verify_frozen(base, config):
    manifest = read(base / "freeze.json")
    if manifest["image"] != config["runtime"]["image"]:
        raise ValueError("Frozen runtime image differs from the selected configuration")
    for relative, expected in manifest["files"].items():
        if sha(base / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")
    if (
        json.loads(docker("image", "inspect", manifest["image"]).stdout)[0]["Id"]
        != manifest["image"]
    ):
        raise ValueError("Docker image identity differs")


def archive_source(base, mirror):
    """Persist frozen source through one file instead of thousands of NFS writes."""
    archive = base / "source.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        bundle.add(base / "source", arcname="source")
    destination = mirror / archive.name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(archive, destination)
    expected = sha(archive)
    if sha(destination) != expected:
        raise ValueError("Mirrored frozen source archive differs")
    return {"file": archive.name, "sha256": expected, "bytes": archive.stat().st_size}


class Controller:
    def __init__(self, config, base, mirror, config_relative, gpus, token_file, skip_grid=False):
        self.config, self.base, self.mirror = config, Path(base), Path(mirror)
        self.output = self.base / "output"
        self.config_relative = str(config_relative)
        self.gpus, self.token_file, self.skip_grid = gpus, token_file, skip_grid
        self.mutex, self.mirror_mutex = threading.RLock(), threading.Lock()
        self.last_mirror = 0.0
        self.state = {
            "batch": config["batch"],
            "state": "preparing_direct_training",
            "independent_audits": "not requested by user",
            "arms": {name: {"state": "pending"} for name in config["arms"]},
        }
        allowed = sorted(os.sched_getaffinity(0))
        first = config["runtime"].get("cpu_first")
        self.cpus = list(range(first, first + 4)) if first is not None else allowed[:4]
        if len(self.cpus) != 4 or not set(self.cpus) <= set(allowed):
            raise ValueError("Four available CPU affinity slots are required")

    def update(self, name=None, **values):
        with self.mutex:
            target = self.state if name is None else self.state["arms"][name]
            target.update(values)
            self.state["updated_utc"] = utc()
            write(self.output / "controller-state.json", self.state)

    def mirror_output(self, force=False):
        with self.mirror_mutex:
            if not force and time.monotonic() - self.last_mirror < 60:
                return
            self.mirror.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delay-updates",
                    "--exclude=*.tmp*",
                    "--exclude=*.lock",
                    str(self.output) + "/",
                    str(self.mirror) + "/",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            self.last_mirror = time.monotonic()

    def command(self, name, stage, gpu, uuid):
        position = list(self.config["arms"]).index(name)
        cpus = self.cpus[2 * position : 2 * position + 2]
        label = f"{self.config['batch']}-{sha(self.base / 'freeze.json')[:8]}-{name}-{stage}"
        rt = self.config["runtime"]
        args = [
            *DOCKER,
            "create",
            "--name",
            label,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpuset-cpus",
            ",".join(map(str, cpus)),
            "--cpus",
            "2",
            "--memory",
            f"{rt['memory_gib']}g",
            "--memory-swap",
            f"{rt['memory_gib']}g",
            "--shm-size",
            "2g",
            "--tmpfs",
            "/tmp:rw,size=4g",
            "--network",
            "host",
            "--workdir",
            "/project",
            "--gpus",
            f"device={uuid}",
            "--env",
            "PYTHONPATH=/project/src:/project/scripts",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "OMP_NUM_THREADS=2",
            "--env",
            "MKL_NUM_THREADS=2",
            "--env",
            f"WANDB_DOCKER={rt['image']}",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "XDG_CACHE_HOME=/tmp/cache",
            "--env",
            "WANDB_CACHE_DIR=/tmp/wandb-cache",
            "--env",
            "WANDB_CONSOLE=off",
            "--env",
            "WANDB_SILENT=true",
            "--env",
            "WANDB_DISABLE_CODE=true",
            "--mount",
            f"type=bind,source={self.base / 'source'},target=/project,readonly",
            "--mount",
            f"type=bind,source={self.base / 'data'},target={DATA_TARGET},readonly",
            "--mount",
            f"type=bind,source={self.output},target=/output",
        ]
        bootstrap = "import os,sys;os.umask(0o022);"
        if stage == "train":
            bootstrap += (
                "from pathlib import Path;"
                "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
            )
        bootstrap += "os.execvp(sys.argv[1],sys.argv[1:])"
        args += [
            rt["image"],
            "python",
            "-c",
            bootstrap,
            "python",
            "-u",
            "scripts/run_paper_update_gate.py",
            stage,
            "--config",
            self.config_relative,
            "--base",
            "/output",
            "--arm",
            name,
        ]
        return label, args, cpus

    def execute(self, name, stage, gpu, uuid):
        verify_frozen(self.base, self.config)
        rows, blocked = snapshot()
        row = rows.get(gpu)
        if (
            row is None
            or row["uuid"] != uuid
            or blocked.get(gpu)
            or row["used"] > 512
            or row["utilization"] > 5
            or row["free"] < 18000
        ):
            raise ValueError(f"Selected GPU {gpu} is no longer independently idle")
        folder = self.output / "execution" / name / stage
        folder.mkdir(parents=True, exist_ok=False)  # Failed or completed stages are never retried.
        label, args, cpus = self.command(name, stage, gpu, uuid)
        write(folder / "command.json", {"argv": args, "utc": utc()})
        created = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
        if stage == "train":
            subprocess.run(
                [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
                input=credential_archive(self.token_file),
                check=True,
                capture_output=True,
                timeout=60,
            )
        write(
            self.output / "runtime" / f"{name}-{stage}.json",
            {
                "gpu": gpu,
                "gpu_uuid": uuid,
                "cpus": cpus,
                "container": label,
                "gpu_memory_fraction": self.config["runtime"]["gpu_memory_fraction"],
                "image": self.config["runtime"]["image"],
                "config_sha256": sha(self.base / "source" / self.config_relative),
                "freeze_sha256": sha(self.base / "freeze.json"),
                "exclusive_gpu": True,
                "utc": utc(),
            },
        )
        docker("start", label)
        write(
            folder / "launch.json", {"container": label, "id": created.stdout.strip(), "utc": utc()}
        )
        self.update(name, state="running", stage=stage, gpu=gpu, gpu_uuid=uuid, container=label)
        started = time.monotonic()
        timeout = self.config["runtime"].get("stage_timeout_hours", 72) * 3600
        with (folder / "stdout.log").open("x") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                while True:
                    state = json.loads(docker("inspect", label).stdout)[0]["State"]
                    write(folder / "heartbeat.json", {"state": state, "utc": utc()})
                    resource = (
                        subprocess.run(
                            [
                                "nvidia-smi",
                                f"--id={gpu}",
                                "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        .stdout.strip()
                        .split(",")
                    )
                    write(
                        self.output / "resources" / f"{name}.json",
                        {
                            "gpu": gpu,
                            "gpu_uuid": uuid,
                            "used_mib": int(resource[0]),
                            "utilization": int(resource[1]),
                            "utc": utc(),
                        },
                    )
                    self.mirror_output()
                    if not state["Running"]:
                        write(folder / "exit.json", {"code": state["ExitCode"], "utc": utc()})
                        if state["ExitCode"]:
                            raise RuntimeError(f"{name}/{stage} exited {state['ExitCode']}")
                        break
                    if time.monotonic() - started > timeout:
                        write(folder / "timeout.json", {"own_container": label, "utc": utc()})
                        docker("stop", "--time", "30", label)
                        raise TimeoutError(f"Own container {label} exceeded stage timeout")
                    time.sleep(self.config["runtime"].get("poll_seconds", 15))
            finally:
                try:
                    follower.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    follower.terminate()
                    follower.wait(timeout=10)
        self.mirror_output(force=True)

    def worker(self, name, gpu, uuid):
        try:
            for stage in ("train",) if self.skip_grid else ("train", "grid"):
                self.execute(name, stage, gpu, uuid)
            self.update(
                name, state="training_complete_audit_not_requested", grid=not self.skip_grid
            )
        except BaseException as error:
            self.update(
                name, state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            raise

    def run(self):
        self.update(state="scientific_training")
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {
                pool.submit(self.worker, name, gpu, self.gpus[gpu]): name
                for name, gpu in zip(self.config["arms"], self.gpus, strict=True)
            }
            for future in concurrent.futures.as_completed(jobs):
                try:
                    future.result()
                except BaseException as error:
                    errors.append({"arm": jobs[future], "error": repr(error)})
        self.update(
            state="needs_attention"
            if errors
            else (
                "training_complete_independent_audit_not_requested"
                if self.skip_grid
                else "training_and_grid_complete_independent_audit_not_requested"
            ),
            errors=errors,
            grid_evaluated=not self.skip_grid,
        )
        self.mirror_output(force=True)
        return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--config", default="configs/paper-update-gate-multihop-v1.json")
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    config_path = (base / "source" / args.config).resolve()
    relative = config_path.relative_to(base / "source")
    config = read(config_path)
    validate_config(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "batch": config["batch"],
                    "arms": config["arms"],
                    "stages": ["train"] if args.skip_grid else ["train", "grid"],
                }
            )
        )
        return
    verify_frozen(base, config)
    credential_archive(
        args.wandb_token_file
    )  # Check private availability before starting either arm.
    output = base / "output"
    output.mkdir(exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Preserve this controller attempt; no automatic restart")
        rows, blocked = snapshot()
        gpus = admit(args.gpus, rows, blocked)
        args.mirror.mkdir(parents=True, exist_ok=True)
        source_archive = archive_source(base, args.mirror)
        write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "freeze_sha256": sha(base / "freeze.json"),
                "config_sha256": sha(config_path),
                "gpu_uuids": gpus,
                "direct_training_user_requested": True,
                "independent_audits_requested": False,
                "source_archive": source_archive,
            },
        )
        write(args.mirror / "freeze.json", read(base / "freeze.json"))
        controller = Controller(
            config, base, args.mirror, relative, gpus, args.wandb_token_file, args.skip_grid
        )
        try:
            passed = controller.run()
        except BaseException as error:
            controller.update(
                state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            controller.mirror_output(force=True)
            raise
        write(output / "controller-exit.json", {"code": 0 if passed else 1, "utc": utc()})
        controller.mirror_output(force=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
