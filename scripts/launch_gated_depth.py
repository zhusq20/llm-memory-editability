#!/usr/bin/env python3
"""Freeze and queue the authorized six-arm depth experiment without preemption.

All execution uses the LM daemon. A durable claim names one immutable container;
recovery attaches to that container and never repeats a claimed training stage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "gated-loop-depth-extrapolation-v1"
ARMS = ("N-Full", "N-Fixed", "N-Gate", "Q-Full", "Q-Fixed", "Q-Gate")
DOCKER = ("docker", "--context", "lm-memory")
TESTS = (
    "tests/test_whole_loop_gate.py",
    "tests/test_gated_depth_data.py",
    "tests/test_gated_depth_runner.py",
    "tests/test_gated_depth_controller.py",
)
CHECK_FILES = (
    "src/llm_memory_editability/whole_loop_gate.py",
    "src/llm_memory_editability/gated_depth_data.py",
    "scripts/run_gated_depth.py",
    "scripts/analyze_gated_depth.py",
    "scripts/launch_gated_depth.py",
    *TESTS,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60).stdout


def docker(*args, context="lm-memory"):
    return checked(["docker", "--context", context, *args])


def validate(config):
    rt = config["runtime"]
    if config["batch"] != BATCH or set(config["arms"]) != set(ARMS):
        raise ValueError("The authorized six arms are required")
    if config["seeds"] != [11, 22, 33]:
        raise ValueError("The authorized formal matrix uses three paired seeds")
    if (
        config["smoke"] != {"seed": 101, "steps": 32}
        or config["pilot"]["seed"] != 101
        or config["pilot"]["steps"] != 20000
        or config["training"]["steps"] != 100000
        or config["training"]["loops"] != 5
    ):
        raise ValueError("Frozen smoke, pilot, and formal budgets cannot expand automatically")
    if rt["docker_context"] != "lm-memory" or rt["tracking"] != "online":
        raise ValueError("LM Docker and online tracking are required")
    if rt["maximum_parallel"] != 2 or len(rt["cpu_sets"]) != 2:
        raise ValueError("Exactly two single-GPU worker lanes are required")
    if not rt["image"].startswith("sha256:") or len(rt["image"]) != 71:
        raise ValueError("The existing image must be pinned by its full identity")
    if len(set(rt["gpu_candidates"])) != len(rt["gpu_candidates"]):
        raise ValueError("GPU candidates must be distinct")
    cpus = [cpu for lane in rt["cpu_sets"] for cpu in lane]
    if not all(rt["cpu_sets"]) or len(set(cpus)) != len(cpus):
        raise ValueError("CPU worker lanes must be nonempty and disjoint")
    if not 1 <= rt["poll_seconds"] <= 60:
        raise ValueError("Bound resource/heartbeat polls to at most 60 seconds")


def freeze(config_path, base):
    config_path, base = Path(config_path).resolve(), Path(base).resolve()
    config = read(config_path)
    validate(config)
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    source.mkdir()
    # Preserve dependencies as they exist, including uncommitted code. Never
    # import a live checkout when a durable controller subsequently resumes.
    for folder in ("src", "scripts", "tests"):
        shutil.copytree(
            ROOT / folder,
            source / folder,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    (source / "configs").mkdir()
    shutil.copy2(config_path, source / "configs" / f"{BATCH}.json")
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    documents = ROOT / "docs/development-artifacts" / BATCH
    if documents.exists():
        shutil.copytree(documents, source / "docs/development-artifacts" / BATCH)
    semantics = source / "docs/loop_semantics_audit.md"
    semantics.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "docs/loop_semantics_audit.md", semantics)
    shutil.copytree(ROOT / "containers/gated-depth", source / "containers/gated-depth")
    for path in (base, source, *source.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    write(
        base / "freeze.json",
        {
            "batch": BATCH,
            "utc": utc(),
            "host": socket.gethostname(),
            "image": config["runtime"]["image"],
            "repository": str(ROOT),
            "commit": checked(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
            "credentials_included": False,
            "files": {
                str(path.relative_to(base)): sha(path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            },
        },
    )
    return base


def verify(base, *, image=True):
    base = Path(base).resolve()
    manifest = read(base / "freeze.json")
    for relative, expected in manifest["files"].items():
        path = base / relative
        if not path.resolve().is_relative_to(base) or sha(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    config = read(base / "source/configs" / f"{BATCH}.json")
    validate(config)
    if (
        image
        and json.loads(docker("image", "inspect", manifest["image"]))[0]["Id"] != manifest["image"]
    ):
        raise ValueError("Pinned image identity changed")
    return config


def predecessor_controller_active(prior):
    """Check the recorded process identity; an unrelated reused PID is not a queue."""
    root = Path(prior["root"])
    try:
        pid = int(read(root / "controller-process.json")["pid"])
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    argv = [part.decode(errors="replace") for part in command if part]
    process_root = str(Path(prior.get("process_root", root.parent)).resolve())
    return any(process_root in value for value in argv) and any(
        value.endswith(".py")
        and any(word in Path(value).name for word in ("launch", "resume", "controller"))
        for value in argv
    )


def previous_queue_blockers(runtime):
    """Protect each old lane through training, independent replay, and cloud audit."""
    root = Path(runtime["previous_queue_root"])
    reasons = {}
    for gpu, jobs in runtime["previous_queue_lanes"].items():
        try:
            lane = read(root / "handoff" / f"lane-{gpu}.json")
            if lane["state"] != "finished" or set(lane["outcomes"]) != set(jobs):
                raise ValueError("Previous lane still has scheduled work")
            for job in jobs:
                status = read(root / "handoff" / f"{job}-status.json")
                if (
                    lane["outcomes"][job] != 0
                    or status.get("state") != "complete"
                    or not status.get("independent_audit_passed")
                    or not status.get("cloud_verified")
                ):
                    raise ValueError("Previous lane has not completed all required audits")
                for stage in ("train", "audit", "cloud-audit"):
                    if read(root / "execution" / job / stage / "exit.json")["code"] != 0:
                        raise ValueError("Previous stage did not exit successfully")
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons[int(gpu)] = str(error)
    for prior in runtime.get("reserved_predecessors", []):
        root = Path(prior["root"])
        try:
            if prior.get(
                "policy"
            ) == "active_controller_only" and not predecessor_controller_active(prior):
                continue
            state = read(root / "controller-state.json")
            if state.get("state") != "complete":
                raise ValueError(f"Prior controller state: {state.get('state', 'unknown')}")
            if read(root / "completion-manifest.json").get("passed") is not True:
                raise ValueError("Prior batch completion manifest has not passed")
        except (OSError, ValueError, KeyError, TypeError) as error:
            for gpu in prior["gpus"]:
                reasons[int(gpu)] = f"{prior['name']}: {error}"
    return reasons


def resource_snapshot():
    rows = {}
    raw = checked(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    for line in raw.splitlines():
        index, uuid, used, free, utilization = [value.strip() for value in line.split(",")]
        rows[int(index)] = {
            "uuid": uuid,
            "used": int(used),
            "free": int(free),
            "utilization": int(utilization),
        }
    processes = checked(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    by_uuid = {row["uuid"]: index for index, row in rows.items()}
    blocked = {index: [] for index in rows}
    for line in processes.splitlines():
        uuid, pid = [value.strip() for value in line.split(",")]
        if uuid in by_uuid:
            blocked[by_uuid[uuid]].append(f"CUDA process {pid}")
    for context in ("default", "lm-memory"):
        identifiers = docker("ps", "-q", context=context).split()
        if not identifiers:
            continue
        for item in json.loads(docker("inspect", *identifiers, context=context)):
            for request in item["HostConfig"].get("DeviceRequests") or []:
                if request.get("Count") == -1:
                    for index in blocked:
                        blocked[index].append(f"{context}:{item['Name']} reserves all GPUs")
                for device in request.get("DeviceIDs") or []:
                    index = int(device) if device.isdigit() else by_uuid.get(device)
                    if index in blocked:
                        blocked[index].append(f"{context}:{item['Name']}")
    return rows, blocked


def idle_candidates(runtime, rows, blocked, reserved=()):
    return [
        index
        for index in runtime["gpu_candidates"]
        if index in rows
        and index not in reserved
        and not blocked.get(index)
        and rows[index]["used"] <= runtime["maximum_idle_gpu_used_mib"]
        and rows[index]["utilization"] <= runtime["maximum_idle_gpu_utilization"]
        and rows[index]["free"] >= runtime["minimum_free_gpu_mib"]
    ]


def validate_stage_artifact(base, stage, arm=None, seed=None):
    """A successful process exit is separate from the required scientific receipt."""
    if stage in ("contracts", "prepare", "pilot-gate"):
        return
    if stage == "analyze":
        receipt = read(Path(base) / "analysis.json")
        if receipt.get("passed") is not True:
            raise ValueError("Required aggregate analysis did not pass")
        return receipt
    phase = (
        "smoke" if stage.startswith("smoke") else "pilot" if stage.startswith("pilot") else "train"
    )
    filenames = {
        "smoke": "complete.json",
        "pilot": "complete.json",
        "train": "complete.json",
        "smoke-audit": "audit.json",
        "pilot-audit": "audit.json",
        "audit": "audit.json",
        "evaluate": "evaluation.json",
        "cloud-audit": "cloud-audit.json",
    }
    path = Path(base) / phase / f"{arm}-seed{seed}" / filenames[stage]
    receipt = read(path)
    if receipt.get("passed") is not True:
        raise ValueError(f"Required stage artifact did not pass: {path}")
    if stage in ("smoke", "pilot", "train"):
        steps = {"smoke": 32, "pilot": 20000, "train": 100000}[stage]
        if (
            receipt.get("steps") != steps
            or receipt.get("arm") != arm
            or receipt.get("seed") != seed
            or receipt.get("phase") != stage
        ):
            raise ValueError(f"Completed training identity or budget differs: {path}")
    return receipt


def secret_archive(token):
    """Send one private credential directly into an owned, stopped container."""
    token = Path(token)
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("A private existing W&B credential file is required")
    content = token.read_bytes()
    if not content.strip():
        raise ValueError("W&B credential file is empty")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            if name == "run/secrets":
                info.uid, info.gid = os.getuid(), os.getgid()
            archive.addfile(info)
        info = tarfile.TarInfo("run/secrets/wandb")
        info.size, info.mode, info.uid, info.gid = (
            len(content),
            0o400,
            os.getuid(),
            os.getgid(),
        )
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def runner_command(stage, arm=None, seed=None):
    if stage == "contracts":
        checks = repr(list(CHECK_FILES))
        tests = repr(list(TESTS))
        return [
            "python",
            "-c",
            "import subprocess;"
            f"subprocess.run(['python','-m','pytest','-q','-p','no:cacheprovider',*{tests},"
            "'--junitxml=/output/contracts.xml'],check=True);"
            f"subprocess.run(['python','-m','ruff','check',*{checks}],check=True);"
            f"subprocess.run(['python','-m','ruff','format','--check',*{checks}],check=True)",
        ]
    args = [
        "python",
        "-u",
        "scripts/run_gated_depth.py",
        stage,
        "--config",
        f"configs/{BATCH}.json",
        "--base",
        "/output",
    ]
    if arm is not None:
        args += ["--arm", arm, "--seed", str(seed)]
    return args


def container_command(config, base, job, stage, lane, row=None, arm=None, seed=None):
    rt = config["runtime"]
    cpus = rt["cpu_sets"][lane]
    name = f"{BATCH}-{job}-{stage}".lower()
    args = [
        *DOCKER,
        "create",
        "--name",
        name,
        "--label",
        f"lm-memory.batch={BATCH}",
        "--label",
        f"lm-memory.freeze={sha(base / 'freeze.json')}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--cpus",
        str(len(cpus)),
        "--memory",
        f"{rt['memory_gib']}g",
        "--memory-swap",
        f"{rt['memory_gib']}g",
        "--shm-size",
        "4g",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
        "--mount",
        f"type=bind,source={base / 'source'},target=/frozen,readonly",
        "--mount",
        f"type=bind,source={base},target=/output",
    ]
    if row is not None:
        args += ["--gpus", f"device={row['uuid']}"]
    for value in (
        "PYTHONPATH=/frozen/src:/frozen/scripts",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "XDG_CACHE_HOME=/tmp/gated-depth-cache",
        "WANDB_CONFIG_DIR=/tmp/gated-depth-config",
        "WANDB_DISABLE_CODE=true",
        "WANDB_CONSOLE=off",
        "WANDB_SILENT=true",
        "WANDB_MODE=online",
        f"WANDB_DOCKER={rt['image']}",
        "RUFF_CACHE_DIR=/tmp/ruff-cache",
        f"OMP_NUM_THREADS={len(cpus)}",
        f"MKL_NUM_THREADS={len(cpus)}",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    ):
        args += ["--env", value]
    bootstrap = "import os,sys;os.umask(0o022);"
    if stage in ("smoke", "pilot", "train", "cloud-audit"):
        bootstrap += (
            "from pathlib import Path;"
            "secret=Path('/run/secrets/wandb');"
            "os.environ['WANDB_API_KEY']=secret.read_text().strip();secret.unlink();"
        )
    bootstrap += "os.execvp(sys.argv[1],sys.argv[1:])"
    args += [rt["image"], "python", "-c", bootstrap, *runner_command(stage, arm, seed)]
    return name, args


class Controller:
    def __init__(self, config, base, token):
        self.config, self.base, self.token = config, Path(base), Path(token)
        self.mutex = threading.RLock()
        self.leases = {}
        self.stop = threading.Event()
        state = self.base / "controller-state.json"
        self.state = read(state) if state.exists() else {"batch": BATCH, "jobs": {}}

    def update(self, job=None, **values):
        with self.mutex:
            target = self.state if job is None else self.state["jobs"].setdefault(job, {})
            target.update(values)
            self.state["updated_utc"] = utc()
            write(self.base / "controller-state.json", self.state)
            write(self.base / "science-schedule.json", self.state["jobs"])

    def stage_directory(self, job, stage):
        return self.base / "execution" / job / stage

    def finished(self, job, stage):
        receipt = self.stage_directory(job, stage) / "exit.json"
        if not receipt.exists():
            return False
        if read(receipt)["code"] != 0:
            raise RuntimeError(f"Failed stage preserved without retry: {job}/{stage}")
        return True

    def recovery_row(self, job, stages):
        for stage in stages:
            folder = self.stage_directory(job, stage)
            if self.finished(job, stage):
                continue
            if folder.exists():
                claim = read(folder / "claim.json")
                if claim.get("admission") is None:
                    return None
                return claim["admission"]
            return None
        return None

    def acquire(self, job, stages):
        rt = self.config["runtime"]
        existing = self.recovery_row(job, stages)
        lock_root = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
        lock_root.mkdir(exist_ok=True)
        while not self.stop.is_set():
            try:
                with self.mutex:
                    rows, blocked = resource_snapshot()
                    for index, reason in previous_queue_blockers(rt).items():
                        blocked.setdefault(index, []).append(reason)
                    candidates = idle_candidates(rt, rows, blocked, self.leases)
                    if existing is not None:
                        index = existing["gpu"]
                        if index in self.leases:
                            candidates = []
                        elif index not in rows or rows[index]["uuid"] != existing["uuid"]:
                            raise ValueError("The claimed GPU identity changed")
                        else:
                            candidates = [index]
                    for index in candidates:
                        held = (lock_root / f"gpu-{index}.lock").open("a")
                        try:
                            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            held.close()
                            continue
                        self.leases[index] = held
                        row = {"gpu": index, **rows[index]}
                        self.update(job, state="admitted", admission=row)
                        return row
                    self.update(
                        job,
                        state="waiting_for_idle_resources",
                        blocked=blocked,
                        rows=rows,
                    )
            except (subprocess.SubprocessError, OSError) as error:
                self.update(job, state="waiting_resource_probe", resource_error=str(error))
            self.stop.wait(rt["poll_seconds"])
        raise RuntimeError("A sibling failure stopped new admissions")

    def release(self, row):
        with self.mutex:
            self.leases.pop(row["gpu"]).close()

    def execute(self, job, stage, lane, row=None, arm=None, seed=None):
        if self.finished(job, stage):
            validate_stage_artifact(self.base, stage, arm, seed)
            return
        verify(self.base)
        folder = self.stage_directory(job, stage)
        name, command = container_command(self.config, self.base, job, stage, lane, row, arm, seed)
        if folder.exists():
            claim = read(folder / "claim.json")
            if claim["container"] != name:
                raise ValueError("Recovery container identity changed")
            item = json.loads(docker("inspect", name))[0]
            labels = item["Config"].get("Labels") or {}
            if (
                item["Config"]["Image"] != self.config["runtime"]["image"]
                or labels.get("lm-memory.freeze") != sha(self.base / "freeze.json")
                or item["State"]["Status"] == "created"
            ):
                raise ValueError("Claimed container was not safely started; preserve for review")
            requested = [
                device
                for request in item["HostConfig"].get("DeviceRequests") or []
                for device in request.get("DeviceIDs") or []
            ]
            if row is not None and requested != [row["uuid"]]:
                raise ValueError("Existing container GPU differs from its durable claim")
        else:
            folder.mkdir(parents=True, exist_ok=False)
            write(
                folder / "claim.json",
                {"container": name, "utc": utc(), "pid": os.getpid(), "admission": row},
            )
            write(folder / "command.json", {"argv": command, "utc": utc()})
            created = checked(command).strip()
            if stage in ("smoke", "pilot", "train", "cloud-audit"):
                subprocess.run(
                    [*DOCKER, "cp", "--archive", "-", f"{name}:/"],
                    input=secret_archive(self.token),
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
            docker("start", name)
            write(folder / "launch.json", {"container": name, "id": created, "utc": utc()})
        while True:
            item = json.loads(docker("inspect", name))[0]
            status = item["State"]
            self.update(job, state="running", stage=stage, container=name, admission=row)
            if not status["Running"]:
                with (folder / "stdout.log").open("w") as log:
                    subprocess.run(
                        [*DOCKER, "logs", name],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                write(
                    folder / "exit.json",
                    {"code": status["ExitCode"], "state": status, "utc": utc()},
                )
                if status["ExitCode"] != 0:
                    raise RuntimeError(f"Failed container preserved without retry: {name}")
                validate_stage_artifact(self.base, stage, arm, seed)
                self.update(job, state="stage_complete", stage=stage)
                return
            time.sleep(self.config["runtime"]["poll_seconds"])

    def pipeline(self, phase, arm, seed, stages, lane):
        job = f"{phase}-{arm.lower()}-seed{seed}"
        if all(self.finished(job, stage) for stage in stages):
            for stage in stages:
                validate_stage_artifact(self.base, stage, arm, seed)
            self.update(job, state="complete", completed_stages=list(stages))
            return
        row = self.acquire(job, stages)
        try:
            for stage in stages:
                self.execute(job, stage, lane, row, arm, seed)
            self.update(job, state="complete", completed_stages=list(stages))
        finally:
            self.release(row)

    def phase(self, phase, seeds, stages):
        self.update(state="running_or_waiting", phase=phase)
        tasks = iter((arm, seed) for seed in seeds for arm in ARMS)

        def worker(lane):
            while not self.stop.is_set():
                with self.mutex:
                    task = next(tasks, None)
                if task is None:
                    return
                arm, seed = task
                try:
                    self.pipeline(phase, arm, seed, stages, lane)
                except Exception as error:
                    self.stop.set()
                    self.update(
                        f"{phase}-{arm.lower()}-seed{seed}",
                        state="failed",
                        error=str(error),
                        traceback=traceback.format_exc(),
                    )
                    raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, lane) for lane in range(2)]
            for future in futures:
                future.result()

    def run(self):
        self.execute("setup", "contracts", 0)
        self.execute("setup", "prepare", 0)
        self.phase("smoke", [self.config["smoke"]["seed"]], ("smoke", "smoke-audit"))
        self.phase("pilot", [self.config["pilot"]["seed"]], ("pilot", "pilot-audit"))
        self.execute("setup", "pilot-gate", 0)
        gate = read(self.base / "pilot-gate.json")
        if gate.get("passed") is not True:
            self.update(state="pilot_no_go", phase="pilot", pilot_gate=gate)
            return
        self.phase("train", self.config["seeds"], ("train", "audit", "evaluate", "cloud-audit"))
        self.execute("setup", "analyze", 0)
        self.update(state="complete", phase="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "plan", "run"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / f"{BATCH}.json")
    parser.add_argument("--wandb-token-file", type=Path)
    args = parser.parse_args()
    os.umask(0o022)
    base = args.base.resolve()
    if args.action == "freeze":
        freeze(args.config, base)
        print(json.dumps({"batch": BATCH, "base": str(base), "frozen": True}))
        return
    config = verify(base)
    if args.action == "verify":
        print(json.dumps({"batch": BATCH, "verified": True}))
        return
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "base": str(base),
                    "arms": ARMS,
                    "seeds": config["seeds"],
                    "runtime": config["runtime"],
                },
                indent=2,
            )
        )
        return
    if socket.gethostname() != config["runtime"]["host"]:
        raise ValueError("Frozen runtime host differs")
    if Path(__file__).resolve() != base / "source/scripts/launch_gated_depth.py":
        raise ValueError("Run the immutable launcher from the frozen source directory")
    if args.wandb_token_file is None:
        raise ValueError("Provide the existing private W&B credential file")
    secret_archive(args.wandb_token_file)
    cpus = {cpu for lane in config["runtime"]["cpu_sets"] for cpu in lane}
    if not cpus <= os.sched_getaffinity(0):
        raise ValueError("Configured CPU affinity is unavailable")
    with (base / "controller.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "utc": utc(),
            "base": str(base),
            "freeze_sha256": sha(base / "freeze.json"),
            "automatic_retry": False,
            "maximum_parallel": 2,
        }
        write(base / "controller-process.json", receipt)
        write(
            base / "controller-attempts" / f"{os.getpid()}-{time.time_ns()}.json",
            receipt,
        )
        controller = Controller(config, base, args.wandb_token_file.resolve())
        exit_code = 1
        try:
            controller.run()
            exit_code = 0
        except Exception as error:
            controller.update(
                state="needs_attention",
                error=str(error),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            write(
                base / "controller-exit.json",
                {"code": exit_code, "state": controller.state.get("state"), "utc": utc()},
            )


if __name__ == "__main__":
    main()
