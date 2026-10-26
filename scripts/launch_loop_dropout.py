#!/usr/bin/env python3
"""Freeze one dropout ablation and wait for an unreserved GPU within the quota."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from launch_loop_transfer import blockers, credential_archive  # noqa: E402

BATCH = "loop-dropout-development-v1"
CONFIG = f"configs/{BATCH}.json"
BASELINE = "loop-state-handoff-development-v1"
DOCKER = ["docker", "--context", "lm-memory"]
STAGES = ("environment", "smoke", "smoke-grid", "train", "grid", "cloud")
TESTS = [
    "tests/test_loop_dropout_runner.py",
    "tests/test_loop_dropout_launcher.py",
    "tests/test_loop_state_runner.py",
    "tests/test_loop_state_handoff.py",
    "tests/test_loop_native_handoff.py",
]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, timeout=120, **kwargs)


def validate(config, original):
    if config["batch"] != BATCH or not config["status"].startswith("authorized"):
        raise ValueError("Only the authorized dropout ablation may be submitted")
    if config["execution_stop_step"] != 500000 or config["training"]["steps"] != 1500000:
        raise ValueError("Stop at 500000 without shortening the original LR horizon")
    for key in (
        "data",
        "training",
        "handoff",
        "arms",
        "dropout_seed",
        "paired_recipe",
        "evaluation_rounds",
        "evaluation_batch_size",
        "grid_rounds",
        "grid_joint_rounds",
        "heartbeat_steps",
        "smoke_steps",
        "score_design",
        "input_root",
        "input_origin",
    ):
        if key != "paired_recipe" and config[key] != original[key]:
            raise ValueError(f"Paired recipe changed: {key}")
    model = dict(original["model"])
    model.update(attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
    if config["model"] != model:
        raise ValueError("Only the three dropout probabilities may change")
    if config["evaluation_nodes"] != [n for n in original["evaluation_nodes"] if n <= 500000]:
        raise ValueError("Use every original evaluation node through the execution cap")
    if config["budget"]["new_scientific_updates"] != 500000:
        raise ValueError("Exactly one 500000-update scientific run is authorized")
    if config["tracking"] != original["tracking"]:
        raise ValueError("Preserve online tracking to zsqzz/llm-memory-editability")
    if config["runtime"]["image"] != original["runtime"]["image"]:
        raise ValueError("The fixed image must remain unchanged")
    if config["runtime"]["docker_context"] != "lm-memory":
        raise ValueError("Use the LM Docker context")
    if config["runtime"]["maximum_local_user_gpus"] != 4:
        raise ValueError("The host-wide user GPU limit is four")
    if config["runtime"]["automatic_retry"] or config["runtime"]["jobs_per_gpu"] != 1:
        raise ValueError("No automatic retry or GPU sharing is authorized")


def freeze(base):
    config = json.loads((ROOT / CONFIG).read_text())
    old = ROOT / "results" / BASELINE / "runtime" / "source"
    old_config = old / "configs" / f"{BASELINE}.json"
    if digest(old_config) != config["reused_comparators"]["state-mlp"]["baseline_config_sha256"]:
        raise ValueError("Original frozen comparator configuration changed")
    validate(config, json.loads(old_config.read_text()))
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    shutil.copytree(old, source)
    overlays = [
        CONFIG,
        "scripts/run_loop_dropout.py",
        "scripts/launch_loop_dropout.py",
        "tests/test_loop_dropout_runner.py",
        "tests/test_loop_dropout_launcher.py",
        f"docs/development-artifacts/{BATCH}/protocol.md",
    ]
    for name in overlays:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            path.chmod(0o644)
            files[str(path.relative_to(base))] = digest(path)
        elif path.is_dir():
            path.chmod(0o755)
    unchanged = {}
    for path in old.rglob("*"):
        if path.is_file():
            name = str(path.relative_to(old))
            if name not in overlays:
                expected = digest(path)
                if digest(source / name) != expected:
                    raise ValueError(f"Original source changed: {name}")
                unchanged[name] = expected
    inputs = {
        str(Path(config["input_root"]) / n): digest(Path(config["input_root"]) / n)
        for n in ("world.npz", "panels.npz", "complete.json")
    }
    for name, key in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        if inputs[str(Path(config["input_root"]) / name)] != config["input_origin"][key]:
            raise ValueError(f"Original paired data changed: {name}")
    run = Path(config["reused_comparators"]["state-mlp"]["run"])
    comparator = {str(run / n): digest(run / n) for n in ("run.json", "checkpoint-0000000.pt")}
    for name in ("artifacts", "engineering"):
        (base / name).mkdir(mode=0o755)
    base.chmod(0o755)
    write(
        base / "freeze.json",
        {
            "batch": BATCH,
            "utc": utc(),
            "image": config["runtime"]["image"],
            "files": files,
            "inputs": inputs,
            "unchanged_original_files": unchanged,
            "original_source": str(old.resolve()),
            "comparator_initialization": comparator,
            "scientific_updates": 500000,
            "learning_rate_horizon_steps": 1500000,
            "new_control_training_updates": 0,
        },
    )
    return {"base": str(base), "files": len(files), "unchanged_files": len(unchanged)}


def verify_freeze(base):
    manifest = json.loads((base / "freeze.json").read_text())
    for section in ("files", "inputs", "comparator_initialization"):
        for name, expected in manifest[section].items():
            if digest(base / name) != expected:
                raise ValueError(f"Frozen artifact changed: {name}")
    return manifest


def resource_snapshot():
    """Keep owner/device/identity fields only; never persist container environments."""
    rows = {}
    raw = command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).stdout
    for line in raw.splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(",")]
        rows[int(index)] = dict(
            uuid=uuid, used=int(used), free=int(free), utilization=int(utilization)
        )
    raw = command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    ).stdout
    processes, own_processes = [], []
    for line in raw.splitlines():
        if not line.strip():
            continue
        uuid, pid = [v.strip() for v in line.split(",")]
        processes.append(uuid)
        try:
            if Path(f"/proc/{pid}").stat().st_uid == os.getuid():
                own_processes.append(uuid)
        except FileNotFoundError:
            pass
    containers = []
    for context in ("default", "lm-memory"):
        ids = command(
            [
                "docker",
                "--context",
                context,
                "ps",
                "-aq",
                "--filter",
                "status=running",
                "--filter",
                "status=paused",
                "--filter",
                "status=restarting",
                "--filter",
                "status=created",
            ],
            text=True,
        ).stdout.split()
        if not ids:
            continue
        for c in json.loads(
            command(["docker", "--context", context, "inspect", *ids], text=True).stdout
        ):
            labels = c["Config"].get("Labels") or {}
            containers.append(
                {
                    "id": c["Id"],
                    "name": c["Name"],
                    "context": context,
                    "state": c["State"],
                    "requests": c["HostConfig"].get("DeviceRequests"),
                    "restart": c["HostConfig"]["RestartPolicy"],
                    "owner": labels.get("io.gpu-quota.owner"),
                    "user": c["Config"].get("User"),
                    "cpu_set": c["HostConfig"].get("CpusetCpus", ""),
                }
            )
    return {
        "utc": utc(),
        "rows": rows,
        "processes": processes,
        "own_processes": own_processes,
        "containers": containers,
    }


def owned_gpus(snap, runtime):
    owned = {gpu for gpu, row in snap["rows"].items() if row["uuid"] in snap["own_processes"]}
    for c in snap["containers"]:
        if c["owner"] != runtime["owner"] and (c.get("user") or "").split(":")[0] != str(
            os.getuid()
        ):
            continue
        for request in c["requests"] or []:
            for gpu, row in snap["rows"].items():
                if any(d in (str(gpu), row["uuid"]) for d in request.get("DeviceIDs") or []):
                    owned.add(gpu)
            if request.get("Count") == -1:
                # Unqualified visibility on a user container makes ownership ambiguous.
                owned.update(snap["rows"])
    return owned


def priority_reasons(runtime, gpu):
    reasons = []
    for priority in runtime["priority_batches"]:
        try:
            state = json.loads(Path(priority["controller_state"]).read_text())
            if state.get("state") == "complete" and all(
                job.get("state") == "complete" for job in state.get("jobs", {}).values()
            ):
                continue
            pool = priority.get("fixed_gpu_ids", priority.get("fallback_gpu_ids", []))
            if "schedule_amendment" in priority:
                amendment = json.loads(Path(priority["schedule_amendment"]).read_text())
                pool = amendment["effective_gpu_pool"]
            if gpu in pool:
                reasons.append(
                    "Reserved for the existing queue through its final audit/cloud stage"
                )
        except (OSError, ValueError, KeyError):
            reasons.append("Priority queue state unavailable; admission fails closed")
    return reasons


def available(snap, runtime, gpu):
    if gpu not in snap["rows"] or gpu in runtime["excluded_gpu_ids"]:
        return ["GPU unavailable or explicitly excluded"]
    reasons = priority_reasons(runtime, gpu) + blockers(snap, gpu, [])
    owned = owned_gpus(snap, runtime)
    if len(owned | {gpu}) > runtime["maximum_local_user_gpus"]:
        reasons.append("Host-wide four-GPU user quota is full")
    return reasons + cpu_reasons(snap, runtime)


def cpu_reasons(snap, runtime):
    def expand(value):
        cpus = set()
        for part in filter(None, value.split(",")):
            endpoints = [int(v) for v in part.split("-")]
            cpus.update(range(endpoints[0], endpoints[-1] + 1))
        return cpus

    selected = expand(runtime["cpu_set"])
    return [
        f"CPU reservation overlaps {c['name']}"
        for c in snap["containers"]
        if selected & expand(c.get("cpu_set", ""))
    ]


def code_archive(base):
    """Create-copy-start avoids NFS private-parent bind access problems."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in [base / "source", *sorted((base / "source").rglob("*"))]:
            if path.is_symlink():
                raise ValueError("Frozen code must not contain symlinks")
            name = str(Path("project") / path.relative_to(base / "source"))
            item = archive.gettarinfo(str(path), arcname=name)
            item.uid = item.gid = 0
            item.uname = item.gname = "root"
            item.mode = 0o555 if path.is_dir() else 0o444
            with path.open("rb") if path.is_file() else io.BytesIO() as stream:
                archive.addfile(item, stream if path.is_file() else None)
    return buffer.getvalue()


def stage_command(config, base, stage, uuid=None):
    runtime = config["runtime"]
    name = f"ld0-1022-state-mlp-v2-{stage}"
    boot = "import os,sys;os.umask(0o022);"
    if stage in ("train", "cloud"):
        boot += f"os.chown('/run/secrets/wandb',{os.getuid()},{os.getgid()});"
    boot += f"os.setgroups([]);os.setgid({os.getgid()});os.setuid({os.getuid()});"
    if stage in ("train", "cloud"):
        boot += (
            "from pathlib import Path;"
            "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
        )
    boot += "os.execvp(sys.argv[1],sys.argv[1:])"
    argv = [
        *DOCKER,
        "create",
        "--name",
        name,
        "--user",
        "0:0",
        "--cpuset-cpus",
        runtime["cpu_set"],
        "--cpus",
        "2",
        "--memory",
        "16g",
        "--memory-swap",
        "16g",
        "--shm-size",
        "2g",
        "--tmpfs",
        "/tmp:rw,size=4g",
        "--network",
        "none" if stage == "contracts" else "host",
        "--workdir",
        "/project",
        "--label",
        f"io.gpu-quota.owner={runtime['owner']}",
        "--label",
        f"lm-memory.batch={BATCH}",
    ]
    if stage not in ("contracts", "cloud"):
        if uuid is None:
            raise ValueError("A specific admitted GPU UUID is required")
        argv += ["--gpus", f"device={uuid}"]
    environment = {
        "PYTHONPATH": "/project/src:/project/scripts",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/cache",
        "WANDB_CACHE_DIR": "/tmp/wandb-cache",
        "WANDB_CONSOLE": "off",
        "WANDB_SILENT": "true",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_DOCKER": runtime["image"],
        "RUFF_CACHE_DIR": "/tmp/ruff-cache",
    }
    for key, value in environment.items():
        argv += ["--env", f"{key}={value}"]
    for src, dst, suffix in [
        (runtime["persistent_root"], runtime["persistent_root"], ",readonly"),
        (base / "artifacts", "/output", ""),
    ]:
        argv += ["--mount", f"type=bind,source={src},target={dst}{suffix}"]
    if stage == "contracts":
        new_files = [
            "scripts/run_loop_dropout.py",
            "scripts/launch_loop_dropout.py",
            "tests/test_loop_dropout_runner.py",
            "tests/test_loop_dropout_launcher.py",
        ]
        code = (
            "import pathlib,subprocess;"
            "p=pathlib.Path('/project/scripts/run_loop_dropout.py');"
            "assert p.stat().st_uid==0 and p.stat().st_mode&0o777==0o444;"
            f"subprocess.run(['python','-m','ruff','check',*{new_files!r}],check=True);"
            f"subprocess.run(['python','-m','ruff','format','--check',*{new_files!r}],check=True);"
            f"subprocess.run(['python','-m','pytest','-q','-p','no:cacheprovider',*{TESTS!r}],check=True)"
        )
        job = ["python", "-c", code]
    else:
        job = [
            "python",
            "-u",
            "scripts/run_loop_dropout.py",
            stage,
            "--config",
            CONFIG,
            "--base",
            "/output",
            "--arm",
            "state-mlp",
        ]
    return name, [*argv, runtime["image"], "python", "-c", boot, *job]


def release_unstarted(identifier, folder):
    """Release only our never-started container so a failed prepare cannot reserve a GPU."""
    item = json.loads(command([*DOCKER, "inspect", identifier], text=True).stdout)[0]
    state = item["State"]
    labels = item["Config"].get("Labels") or {}
    safe = (
        item["Id"] == identifier
        and state["Status"] == "created"
        and state["Pid"] == 0
        and not state["Running"]
        and state["StartedAt"] == "0001-01-01T00:00:00Z"
        and item["HostConfig"]["RestartPolicy"]["Name"] == "no"
        and labels.get("lm-memory.batch") == BATCH
    )
    write(
        folder / "unstarted-cleanup.json",
        {"id": identifier, "eligible": safe, "state": state, "utc": utc()},
    )
    if safe:
        command([*DOCKER, "rm", identifier])
        write(
            folder / "unstarted-cleanup.json",
            {"id": identifier, "eligible": True, "removed": True, "utc": utc()},
        )


def is_unstarted_receipt(container, identifier):
    state = container["state"]
    return (
        container["id"] == identifier
        and state["Status"] == "created"
        and state["Pid"] == 0
        and not state["Running"]
        and state["StartedAt"] == "0001-01-01T00:00:00Z"
        and container["restart"]["Name"] == "no"
    )


class Controller:
    def __init__(self, base, token=None):
        self.base, self.token = base, token
        self.config = json.loads((base / "source" / CONFIG).read_text())
        original = json.loads((base / "source/configs" / f"{BASELINE}.json").read_text())
        validate(self.config, original)
        self.state = {
            "batch": BATCH,
            "state": "submitted",
            "stage": "environment",
            "scientific_updates_completed": 0,
            "scientific_update_budget": 500000,
            "learning_rate_horizon_steps": 1500000,
            "tracking_project": "https://wandb.ai/zsqzz/llm-memory-editability",
        }

    def update(self, **values):
        self.state.update(values, updated_utc=utc())
        write(self.base / "controller-state.json", self.state)

    def admit(self, selected=None):
        runtime = self.config["runtime"]
        confirmations = {}
        while True:
            snap = resource_snapshot()
            reasons = {}
            for gpu in runtime["gpu_candidates"] if selected is None else [selected]:
                reasons[gpu] = available(snap, runtime, gpu)
                confirmations[gpu] = 0 if reasons[gpu] else confirmations.get(gpu, 0) + 1
                if confirmations[gpu] >= runtime["idle_confirmations"]:
                    write(self.base / "resource-admission" / f"{self.state['stage']}.json", snap)
                    return gpu, snap["rows"][gpu]["uuid"]
            self.update(
                state="waiting_resource",
                waiting_reasons=reasons,
                current_user_gpus=sorted(owned_gpus(snap, runtime)),
            )
            write(self.base / "resource-latest.json", snap)
            time.sleep(runtime["poll_seconds"])

    def execute(self, stage, gpu=None, uuid=None):
        folder = self.base / "execution" / stage
        folder.mkdir(parents=True, exist_ok=False)
        name, argv = stage_command(self.config, self.base, stage, uuid)
        if stage in ("contracts", "cloud"):
            if reasons := cpu_reasons(resource_snapshot(), self.config["runtime"]):
                raise RuntimeError(f"Engineering CPU reservation conflict: {reasons}")
        if stage not in ("contracts", "cloud"):
            fresh = resource_snapshot()
            reasons = available(fresh, self.config["runtime"], gpu)
            if reasons or fresh["rows"][gpu]["uuid"] != uuid:
                raise RuntimeError(f"Resources changed immediately before create: {reasons}")
        write(folder / "command.json", {"argv": argv, "gpu": gpu, "utc": utc()})
        identifier = None
        try:
            result = command(argv, text=True)
            identifier = result.stdout.strip()
            write(folder / "container.json", {"id": identifier, "name": name})
            command([*DOCKER, "cp", "--archive", "-", f"{name}:/"], input=code_archive(self.base))
            if stage in ("train", "cloud"):
                command(
                    [*DOCKER, "cp", "--archive", "-", f"{name}:/"],
                    input=credential_archive(self.token),
                )
            fresh = resource_snapshot()
            # Ignore only this exact not-yet-started container, never another reservation.
            fresh["containers"] = [
                c for c in fresh["containers"] if not is_unstarted_receipt(c, identifier)
            ]
            if stage not in ("contracts", "cloud"):
                reasons = available(fresh, self.config["runtime"], gpu)
                if fresh["rows"][gpu]["uuid"] != uuid:
                    reasons.append("GPU identity changed before start")
            else:
                reasons = cpu_reasons(fresh, self.config["runtime"])
            if reasons:
                raise RuntimeError(f"Resources changed before start: {reasons}")
            command([*DOCKER, "start", name])
        except BaseException:
            if identifier is not None:
                release_unstarted(identifier, folder)
            raise
        self.update(state="running", stage=stage, gpu=gpu, gpu_uuid=uuid, container=name)
        with (folder / "stdout.log").open("w") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", name], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                while True:
                    info = json.loads(
                        command(
                            [*DOCKER, "inspect", "--format", "{{json .State}}", name], text=True
                        ).stdout
                    )
                    if info["Status"] == "exited":
                        break
                    self.update(last_container_check_utc=utc())
                    time.sleep(self.config["runtime"]["poll_seconds"])
            finally:
                if follower.poll() is None:
                    follower.terminate()
                follower.wait(timeout=30)
        write(
            folder / "exit.json",
            {"code": info["ExitCode"], "oom_killed": info["OOMKilled"], "utc": utc()},
        )
        if info["ExitCode"]:
            raise RuntimeError(f"{name} exited {info['ExitCode']}; no automatic retry")

    def run(self):
        with (self.base / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.base / "controller-process.json").exists():
                raise FileExistsError("Already submitted; do not restart a historical entry")
            verify_freeze(self.base)
            checks = json.loads((self.base / "engineering/cpu-checks.json").read_text())
            if not checks.get("passed") or checks["freeze_sha256"] != digest(
                self.base / "freeze.json"
            ):
                raise ValueError("CPU contracts and lint must cover the exact frozen source")
            image = command(
                [
                    *DOCKER,
                    "image",
                    "inspect",
                    self.config["runtime"]["image"],
                    "--format",
                    "{{.Id}}",
                ],
                text=True,
            ).stdout.strip()
            if image != checks["image"] or image != self.config["runtime"]["image"]:
                raise ValueError("Fixed runtime image changed")
            credential_archive(self.token)
            write(
                self.base / "controller-process.json",
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "utc": utc(),
                    "controller_sha256": digest(__file__),
                    "freeze_sha256": digest(self.base / "freeze.json"),
                    "config_sha256": digest(self.base / "source" / CONFIG),
                },
            )
            self.update()
            gpu = uuid = None
            try:
                for stage in STAGES:
                    self.update(stage=stage)
                    if stage != "cloud":
                        gpu, found_uuid = self.admit(gpu)
                        if uuid is not None and found_uuid != uuid:
                            raise RuntimeError("GPU UUID changed between stages")
                        uuid = found_uuid
                    verify_freeze(self.base)
                    self.execute(stage, gpu, uuid)
                self.update(state="complete", scientific_updates_completed=500000)
            except BaseException as error:
                self.update(state="failed", error=repr(error))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "contracts", "run"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    if args.action == "freeze":
        print(json.dumps(freeze(base)))
    elif args.action == "contracts":
        verify_freeze(base)
        Controller(base).execute("contracts")
    else:
        if args.wandb_token_file is None:
            parser.error("run requires --wandb-token-file")
        Controller(base, args.wandb_token_file).run()


if __name__ == "__main__":
    os.umask(0o022)
    main()
