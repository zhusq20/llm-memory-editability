#!/usr/bin/env python3
"""Prepare shared data, then run four independently admitted LM Docker lanes.

Default is a read-only plan. --execute is a one-time submission, with no automatic
retries or takeover of existing containers. Scientific runner arguments are stored
in the launch configuration and must be reviewed before enabling execution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from launch_paper_update_gate import credential_archive  # noqa: E402

DOCKER = ["docker", "--context", "lm-memory"]
DEFAULT_PLAN = ROOT / "configs/knowledge-method-diagnosis-launch-v1.json"
ARMS = ("pd_lora", "sft_lora", "sft_full", "sft_down")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def call(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kwargs)


def reserved_devices(requests, rows):
    """Unknown/generic GPU requests reserve every device; never override them."""
    by_uuid = {row["uuid"]: index for index, row in rows.items()}
    result = set()
    for request in requests or []:
        if not any("gpu" in caps for caps in request.get("Capabilities", [])):
            continue
        devices = request.get("DeviceIDs") or []
        if not devices:
            result.update(rows)
        for device in devices:
            if device == "all":
                result.update(rows)
            elif device.isdigit():
                # An explicit absent index cannot reserve a different physical GPU.
                if int(device) in rows:
                    result.add(int(device))
            elif device in by_uuid:
                result.add(by_uuid[device])
            else:
                result.update(rows)
    return result


def snapshot():
    rows = {}
    for line in call(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    ).stdout.splitlines():
        index, uuid, used, free, utilization = [x.strip() for x in line.split(",")]
        rows[int(index)] = {
            "uuid": uuid,
            "used": int(used),
            "free": int(free),
            "utilization": int(utilization),
        }
    blocked = {}
    by_uuid = {row["uuid"]: index for index, row in rows.items()}
    for line in call(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    ).stdout.splitlines():
        uuid, pid = [x.strip() for x in line.split(",")]
        if uuid in by_uuid:
            blocked.setdefault(by_uuid[uuid], []).append(f"active CUDA process {pid}")
    for context in ("default", "lm-memory"):
        args = ["docker", "--context", context, "ps", "-a", "-q"]
        for status in ("created", "running", "paused", "restarting"):
            args += ["--filter", f"status={status}"]
        identifiers = call(args, timeout=30).stdout.split()
        if identifiers:
            # Only inspect scheduling fields: environment variables can hold credentials.
            fmt = "{{json .Name}} {{json .HostConfig.DeviceRequests}}"
            output = call(
                ["docker", "--context", context, "inspect", "--format", fmt, *identifiers],
                timeout=30,
            ).stdout
            for line in output.splitlines():
                name, end = json.JSONDecoder().raw_decode(line)
                requests = json.loads(line[end:].strip())
                for gpu in reserved_devices(requests, rows):
                    blocked.setdefault(gpu, []).append(f"{context}:{name}")
    return rows, blocked


def idle(row, uuid, blockers, runtime):
    return bool(
        row
        and row["uuid"] == uuid
        and not blockers
        and row["used"] <= runtime["maximum_idle_gpu_used_mib"]
        and row["utilization"] <= 5
        and row["free"] >= runtime["minimum_free_gpu_mib"]
    )


def acquire_gpu(lane, runtime, update, deadline, lock_directory=None):
    directory = lock_directory or Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}")
    directory.mkdir(mode=0o700, exist_ok=True)
    index, uuid = lane["gpu"], lane["gpu_uuid"]
    while time.monotonic() < deadline:
        rows, blocked = snapshot()
        if idle(rows.get(index), uuid, blocked.get(index), runtime):
            lease = (directory / f"gpu-{index}.lock").open("a")
            acquired = False
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Re-read after taking the shared project lock to close admission races.
                rows, blocked = snapshot()
                if idle(rows.get(index), uuid, blocked.get(index), runtime):
                    acquired = True
                    return lease
            except BlockingIOError:
                pass
            finally:
                if not acquired:
                    lease.close()
        update(state="waiting_for_gpu", gpu=index, blockers=blocked.get(index, []))
        time.sleep(runtime.get("poll_seconds", 30))
    raise TimeoutError(f"GPU {index} did not become independently idle")


def runner_command(plan, stage, arm):
    values = {"config": plan["scientific_config"], "base": "/batch", "arm": arm}
    return [part.format(**values) for part in plan["commands"][stage]]


def container_command(plan, stage, arm, lane):
    runtime = plan["runtime"]
    label = f"kmd-v1-{arm}-{stage}"
    boot = (
        "import os,sys;os.umask(0o022);from pathlib import Path;"
        "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
        "os.execvp(sys.argv[1],sys.argv[1:])"
    )
    argv = [
        *DOCKER,
        "create",
        "--name",
        label,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpuset-cpus",
        lane["cpus"],
        "--cpus",
        "10",
        "--memory",
        "128g",
        "--memory-swap",
        "128g",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,size=16g",
        "--network",
        "host",
        "--workdir",
        "/project",
        "--gpus",
        f"device={lane['gpu_uuid']}",
        "--label",
        f"lm-memory.batch={plan['batch']}",
        "--label",
        f"lm-memory.arm={arm}",
        "--label",
        f"lm-memory.stage={stage}",
    ]
    env = {
        "PYTHONPATH": "/batch/deps:/project/src:/project/scripts",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "10",
        "MKL_NUM_THREADS": "10",
        "HF_HOME": "/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_CACHE": "/batch/data-cache",
        "XDG_CACHE_HOME": "/tmp/cache",
        "WANDB_CACHE_DIR": "/tmp/wandb-cache",
        "WANDB_CONSOLE": "off",
        "WANDB_SILENT": "true",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_ENTITY": "zsqzz",
        "WANDB_PROJECT": "llm-memory-editability",
        "WANDB_DOCKER": runtime["image"],
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for key, value in env.items():
        argv += ["--env", f"{key}={value}"]
    for source, target, readonly in (
        (str(Path(plan["output"]) / "source"), "/project", True),
        (runtime["hf_cache"], "/hf", True),
        (plan["output"], "/batch", False),
        (str(Path(plan["output"]) / "deps"), "/batch/deps", True),
        (str(Path(plan["output"]) / "upstream"), "/batch/upstream", True),
    ):
        argv += [
            "--mount",
            f"type=bind,source={source},target={target}" + (",readonly" if readonly else ""),
        ]
    argv += [runtime["image"], "python", "-c", boot, *runner_command(plan, stage, arm)]
    return label, argv


def validate(plan, execute=False):
    if plan["batch"] != "knowledge-method-diagnosis-development-v1":
        raise ValueError("Unexpected batch")
    if set(plan["lanes"]) != set(ARMS):
        raise ValueError("Exactly four requested arms are required")
    lanes = list(plan["lanes"].values())
    gpu_mapping, uuid_mapping = {}, {}
    for lane in lanes:
        mapping = (lane["gpu_uuid"], lane["cpus"])
        if lane["gpu"] in gpu_mapping and gpu_mapping[lane["gpu"]] != mapping:
            raise ValueError("Arms sharing a GPU require identical UUID and CPU mapping")
        if lane["gpu_uuid"] in uuid_mapping and uuid_mapping[lane["gpu_uuid"]] != lane["gpu"]:
            raise ValueError("One physical GPU UUID cannot map to different indices")
        gpu_mapping[lane["gpu"]] = mapping
        uuid_mapping[lane["gpu_uuid"]] = lane["gpu"]
    if set(plan["commands"]) != {"prepare", "train", "evaluate", "audit"}:
        raise ValueError("The ordered stages are prepare, train, evaluate, audit")
    if plan["prepare_lane"] != "pd_lora":
        raise ValueError("Shared preparation must use the GPU 3 lane")
    if execute and not plan["cli_reviewed"]:
        raise ValueError("Runner CLI must be reviewed before execution")
    if execute and socket.gethostname() != plan["host"]:
        raise ValueError("Host differs from the reviewed GPU UUID mapping")


def prepared(output):
    marker = output / "shared/prepare-complete.json"
    if not marker.is_file():
        return False
    value = json.loads(marker.read_text())
    return (
        isinstance(value, dict)
        and bool(value)
        and value.get("passed", True) is not False
        and value.get("complete", True) is not False
        and value.get("state", "complete") not in {"failed", "running", "needs_attention"}
    )


class Controller:
    def __init__(self, plan):
        self.plan = plan
        self.output = Path(plan["output"])
        self.mutex = threading.Lock()
        self.state = {
            "batch": plan["batch"],
            "state": "preparing",
            "jobs": {
                "shared": {"state": "pending"},
                **{arm: {"state": "waiting_for_prepare"} for arm in ARMS},
            },
        }

    def update(self, name, **values):
        with self.mutex:
            self.state["jobs"][name].update(values)
            self.state["updated_utc"] = utc()
            write(self.output / "controller-state.json", self.state)

    def run_stage(self, arm, stage, lane):
        folder = self.output / "execution" / arm / stage
        if folder.exists():
            raise FileExistsError(f"Existing attempt is preserved: {folder}")
        label, argv = container_command(self.plan, stage, arm, lane)
        existing = call(
            [*DOCKER, "ps", "-a", "--filter", f"name=^/{label}$", "-q"], timeout=30
        ).stdout.strip()
        if existing:
            raise FileExistsError(f"Container {label} already exists; no takeover")
        write(folder / "command.json", {"argv": argv, "utc": utc()})
        call(argv, timeout=120)
        # Credential bytes are only sent over stdin and never enter stored argv/logs.
        subprocess.run(
            [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
            input=credential_archive(self.plan["wandb_token_file"]),
            check=True,
            capture_output=True,
            timeout=120,
        )
        call([*DOCKER, "start", label], timeout=120)
        self.update(
            arm,
            state="running",
            stage=stage,
            container=label,
            gpu=lane["gpu"],
            gpu_uuid=lane["gpu_uuid"],
            started_utc=utc(),
        )
        with (folder / "stdout.log").open("w") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                code = int(call([*DOCKER, "wait", label]).stdout.strip())
            finally:
                try:
                    follower.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    follower.terminate()
                    follower.wait(timeout=10)
        write(folder / "exit.json", {"code": code, "utc": utc()})
        if code:
            raise RuntimeError(f"{stage} exited {code}; no retry")

    def worker(self, arm, stages, lane):
        lease = None

        def update(**kwargs):
            self.update(arm, **kwargs)

        deadline = time.monotonic() + self.plan["runtime"]["admission_timeout_hours"] * 3600
        try:
            lease = acquire_gpu(lane, self.plan["runtime"], update, deadline)
            for stage in stages:
                # Recheck admission between containers while retaining the shared GPU lease.
                while True:
                    rows, blocked = snapshot()
                    if idle(
                        rows.get(lane["gpu"]),
                        lane["gpu_uuid"],
                        blocked.get(lane["gpu"]),
                        self.plan["runtime"],
                    ):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("GPU became occupied between stages")
                    update(
                        state="waiting_for_gpu", stage=stage, blockers=blocked.get(lane["gpu"], [])
                    )
                    time.sleep(self.plan["runtime"].get("poll_seconds", 30))
                self.run_stage(arm, stage, lane)
            self.update(arm, state="complete", stages=stages, finished_utc=utc())
            return True
        except Exception as error:  # noqa: BLE001 - retain failures, never silently retry
            self.update(arm, state="needs_attention", error=repr(error), finished_utc=utc())
            return False
        finally:
            if lease is not None:
                lease.close()

    def run(self):
        lane = self.plan["lanes"][self.plan["prepare_lane"]]
        success = self.worker("shared", ["prepare"], lane)
        if not success or not prepared(self.output):
            self.update(
                "shared", state="needs_attention", reason="Shared preparation did not finish"
            )
            for arm in ARMS:
                self.update(arm, state="blocked_by_prepare")
            self.state["state"] = "needs_attention"
        else:
            self.state["state"] = "running"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(
                        self.worker, arm, ["train", "evaluate", "audit"], self.plan["lanes"][arm]
                    )
                    for arm in ARMS
                ]
                outcomes = [future.result() for future in futures]
            self.state["state"] = "complete" if all(outcomes) else "needs_attention"
            try:
                call(
                    [
                        sys.executable,
                        str(self.output / "source/scripts/summarize_knowledge_method_diagnosis.py"),
                        "--batch",
                        str(self.output),
                    ],
                    timeout=120,
                )
            except Exception as error:  # Preserve training outcomes if reporting fails.
                self.state["state"] = "needs_attention"
                self.state["summary_error"] = repr(error)
        write(self.output / "controller-state.json", self.state)
        write(self.output / "controller-exit.json", {**self.state, "finished_utc": utc()})
        return 0 if self.state["state"] == "complete" else 1


def freeze_source(plan, launch_path):
    destination = Path(plan["output"]) / "source"
    if destination.exists():
        raise FileExistsError("Existing source snapshot must be preserved")
    sources = list((ROOT / "src/llm_memory_editability").rglob("*.py"))
    sources += list((ROOT / "scripts").glob("*.py"))
    sources += list((ROOT / "tests").glob("test_knowledge_method*.py"))
    sources += list((ROOT / "results/knowledge-method-diagnosis-20261021").glob("*.json"))
    protocol = (
        ROOT / "docs/development-artifacts/knowledge-method-diagnosis-development-v1/protocol.md"
    )
    if protocol.is_file():
        sources.append(protocol)
    sources += [ROOT / plan["scientific_config"], launch_path.resolve(), ROOT / "pyproject.toml"]
    manifest = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(
        Path(plan["output"]) / "source-manifest.json",
        {
            "utc": utc(),
            "source_sha256": manifest,
            "policy": "Snapshot mounted read-only at /project in every scientific container",
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate(plan, args.execute)
    if not args.execute:
        commands = {
            "shared/prepare": container_command(
                plan, "prepare", "shared", plan["lanes"][plan["prepare_lane"]]
            )[1]
        }
        for arm in ARMS:
            for stage in ("train", "evaluate", "audit"):
                commands[f"{arm}/{stage}"] = container_command(
                    plan, stage, arm, plan["lanes"][arm]
                )[1]
        print(json.dumps({"execute": False, "plan": plan, "commands": commands}, indent=2))
        return 0
    output = Path(plan["output"])
    dependency_check = json.loads((output / "dependency-check/verification.json").read_text())
    if not dependency_check.get("passed") or not dependency_check.get("base_packages_unchanged"):
        raise ValueError("Isolated dependencies must pass the reviewed engineering check")
    credential_archive(plan["wandb_token_file"])
    call([*DOCKER, "image", "inspect", plan["runtime"]["image"]], timeout=30)
    with (output / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists() or prepared(output):
            raise FileExistsError("Batch already submitted or prepared; inspect existing records")
        manifest = freeze_source(plan, args.plan)
        write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "utc": utc(),
                "plan": plan,
                "source_sha256": manifest,
                "policy": (
                    "One submission; prepare must finish before four arms; "
                    "no retries, takeover or preemption"
                ),
            },
        )
        return Controller(plan).run()


if __name__ == "__main__":
    os.umask(0o022)
    raise SystemExit(main())
