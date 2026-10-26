#!/usr/bin/env python3
"""Queue the authorized depth/support extension after the current batch."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_loop_hop_depth as parent_launcher  # noqa: E402

from llm_memory_editability.loop_hop_support import expand_support  # noqa: E402

BATCH = "loop-hop-depth-support-v1"
DOCKER = parent_launcher.DOCKER
ENGINEERING, SCIENCE = parent_launcher.ENGINEERING, parent_launcher.SCIENCE
read, write, sha, utc = (
    parent_launcher.read,
    parent_launcher.write,
    parent_launcher.sha,
    parent_launcher.utc,
)
docker, secret_archive = parent_launcher.docker, parent_launcher.secret_archive
snapshot = parent_launcher.snapshot


def validate(config):
    expected = {
        f"{arch}-l{layers}-r{rounds}-base": {
            "architecture": "loop" if arch == "loop" else "discoloop",
            "unique_layers": layers,
            "repeats": rounds,
            "data_condition": "base",
        }
        for layers, rounds in ((2, 3), (2, 4), (3, 2), (3, 3), (3, 4))
        for arch in ("loop", "disco")
    }
    expected.update(
        {
            f"{arch}-l{layers}-r2-more": {
                "architecture": "loop" if arch == "loop" else "discoloop",
                "unique_layers": layers,
                "repeats": 2,
                "data_condition": "more",
            }
            for layers in (2, 3)
            for arch in ("loop", "disco")
        }
    )
    if config["batch"] != BATCH or config["arms"] != expected:
        raise ValueError("Only ten new structural and four support-control runs are authorized")
    if config["supports"] != {"base": 10000, "more": 30000}:
        raise ValueError("The registered nested support levels changed")
    if config["training"]["steps"] != 300000 or config["smoke_steps"] != 32:
        raise ValueError("The fixed scientific/engineering budget changed")
    rt = config["runtime"]
    if rt["gpu_candidates"] != [9] or rt["docker_context"] != "lm-memory":
        raise ValueError("Use only physical cuda9 through LM Docker")
    if rt["tracking"] != "online" or rt["maximum_parallel"] != 8:
        raise ValueError("Tracking or bounded parallelism changed")
    if rt["gpu_memory_fraction"] * 8 > 0.75:
        raise ValueError("GPU allocation caps exceed the registered bound")


def freeze(config, base):
    parent = Path(config["parent"]["root"])
    original = read(parent / "freeze.json")
    if sha(parent / "freeze.json") != config["parent"]["freeze_sha256"]:
        raise ValueError("Parent freeze identity differs")
    for rel, expected in original["files"].items():
        if sha(parent / rel) != expected:
            raise ValueError(f"Parent source/input changed: {rel}")
    base.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "source", base / "source")
    for rel in (
        f"configs/{BATCH}.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
        f"docs/development-artifacts/{BATCH}/history-review.md",
        "scripts/launch_loop_hop_support.py",
        "scripts/run_loop_hop_support.py",
        "src/llm_memory_editability/loop_hop_support.py",
        "tests/test_loop_hop_support.py",
    ):
        destination = base / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    (base / "input").mkdir()
    shutil.copytree(parent / "input", base / "input/base")
    expand_support(
        base / "input/base",
        base / "input/more",
        multiplier=3,
        seed=config["support_expansion_seed"],
    )
    for folder in (base / "source", base / "input"):
        for path in [folder, *folder.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
    base.chmod(0o755)
    write(
        base / "freeze.json",
        {
            "image": config["runtime"]["image"],
            "utc": utc(),
            "preparation_python": sys.version,
            "preparation_numpy": __import__("numpy").__version__,
            "parent_freeze_sha256": sha(parent / "freeze.json"),
            "files": {
                str(p.relative_to(base)): sha(p)
                for folder in (base / "source", base / "input")
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            },
        },
    )


def verify(base):
    parent_launcher.verify(base)


class Controller(parent_launcher.Controller):
    def __init__(self, config, base):
        super().__init__(config, base)
        self.state["batch"] = BATCH

    def acquire(self, name):
        started = time.monotonic()
        rt = self.config["runtime"]
        while True:
            with self.mutex:
                rows, blocked = snapshot(f"{BATCH}-")
                row = rows.get(9)
                members = self.leases.get(9, set())
                idle_ok = bool(members) or (row and row["used"] <= 512 and row["utilization"] <= 5)
                if (
                    row
                    and not blocked[9]
                    and idle_ok
                    and len(members) < rt["maximum_parallel"]
                    and row["free"] >= rt["minimum_free_gpu_mib"]
                ):
                    self.leases.setdefault(9, set()).add(name)
                    self.update(name, state="admitted", gpu=9, gpu_uuid=row["uuid"])
                    return 9, row["uuid"]
                self.update(name, state="waiting_resources", blocked=blocked)
            if time.monotonic() - started > rt["waiting_timeout_hours"] * 3600:
                raise TimeoutError("Resource wait expired")
            time.sleep(rt["poll_seconds"])

    def execute(self, name, stage, gpu=None):
        verify(self.base)
        folder = self.base / "execution" / name / stage
        if (folder / "exit.json").exists():
            if read(folder / "exit.json")["code"] == 0:
                return
            raise RuntimeError(f"Preserve failed {name}/{stage}; no automatic retry")
        folder.mkdir(parents=True, exist_ok=False)
        label = f"{BATCH}-{name}-{stage}"
        rt = self.config["runtime"]
        position = (
            list(self.config["arms"]).index(name)
            if name in self.config["arms"]
            else len(self.config["arms"])
        )
        cpus = [rt["cpu_first"] + 2 * position + i for i in range(2)]
        if not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("Requested CPU affinity is unavailable")
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
            "16g",
            "--memory-swap",
            "16g",
            "--shm-size",
            "2g",
            "--tmpfs",
            "/tmp:rw,size=4g",
            "--network",
            "host",
            "--workdir",
            "/frozen",
            "--env",
            "PYTHONPATH=/frozen/src:/frozen/scripts",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "RUFF_CACHE_DIR=/tmp/ruff-cache",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "WANDB_CONSOLE=off",
            "--env",
            "WANDB_SILENT=true",
            "--env",
            "WANDB_DISABLE_CODE=true",
            "--mount",
            f"type=bind,source={self.base / 'source'},target=/frozen,readonly",
            "--mount",
            f"type=bind,source={self.base},target=/output",
        ]
        condition = self.config["arms"].get(name, {}).get("data_condition", "base")
        for origin, destination in (
            (self.base / "input" / condition, "/inputs"),
            (self.base / "input" / "base", "/base-inputs"),
            (self.base / "input" / "more", "/expanded-inputs"),
            (Path(self.config["parent"]["root"]), "/parent"),
        ):
            args += ["--mount", f"type=bind,source={origin},target={destination},readonly"]
        if gpu is not None:
            args += ["--gpus", f"device={gpu[1]}"]
        if stage == "contracts":
            entry = [
                "python",
                "-c",
                (
                    "import json,subprocess;from pathlib import Path;"
                    "from llm_memory_editability.loop_hop_data import load,audit_data;"
                    f"c=json.loads(Path('configs/{BATCH}.json').read_text());"
                    "w,p,m=load('/inputs');audit_data(w,p,m);"
                    "assert all(m[k]==v for k,v in c['data'].items());"
                    "subprocess.run(['python','-m','pytest','-q','tests/test_loop_hop_support.py',"
                    "'--junitxml=/output/contracts.xml'],check=True);"
                    "subprocess.run(['python','-m','ruff','check','scripts/run_loop_hop_support.py',"
                    "'scripts/launch_loop_hop_support.py','tests/test_loop_hop_support.py'],check=True)"
                ),
            ]
        else:
            entry = [
                "python",
                "-u",
                "scripts/run_loop_hop_support.py",
                stage,
                "--config",
                f"configs/{BATCH}.json",
                "--base",
                "/output",
                "--arm",
                name,
            ]
        tracked = stage in ("train", "cloud")
        bootstrap = "import os,sys;os.umask(0o022);"
        if tracked:
            bootstrap += (
                "from pathlib import Path;"
                "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
            )
        bootstrap += "os.execvp(sys.argv[1],sys.argv[1:])"
        args += [rt["image"], "python", "-c", bootstrap, *entry]
        write(
            folder / "command.json",
            {"argv": args, "utc": utc(), "freeze_sha256": sha(self.base / "freeze.json")},
        )
        created = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
        if tracked:
            subprocess.run(
                [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
                input=secret_archive(),
                check=True,
                capture_output=True,
                timeout=60,
            )
        docker("start", label)
        write(
            folder / "launch.json", {"container": label, "id": created.stdout.strip(), "utc": utc()}
        )
        if name in self.config["arms"]:
            write(
                self.base / "runtime" / f"{name}-{stage}.json",
                {
                    "gpu": gpu[0] if gpu else None,
                    "gpu_uuid": gpu[1] if gpu else None,
                    "cpus": cpus,
                    "container": label,
                    "gpu_memory_fraction": rt["gpu_memory_fraction"],
                    "shared_only_with_own_batch": True,
                    "utc": utc(),
                },
            )
            self.update(name, state="running", stage=stage, container=label)
        started = time.monotonic()
        with (folder / "stdout.log").open("x") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                while True:
                    state = json.loads(docker("inspect", label).stdout)[0]["State"]
                    write(folder / "heartbeat.json", {"state": state, "utc": utc()})
                    if not state["Running"]:
                        write(folder / "exit.json", {"code": state["ExitCode"], "utc": utc()})
                        if state["ExitCode"]:
                            raise RuntimeError(f"{name}/{stage} exited {state['ExitCode']}")
                        break
                    if time.monotonic() - started > rt["stage_timeout_hours"] * 3600:
                        write(folder / "timeout.json", {"own_container": label, "utc": utc()})
                        docker("stop", "--time", "30", label)
                        raise TimeoutError("Own stage hard timeout")
                    time.sleep(rt["poll_seconds"])
            finally:
                try:
                    follower.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    follower.terminate()
                    follower.wait(timeout=10)

    def wait_parent(self):
        rt = self.config["runtime"]
        self.global_lock = (Path(rt["persistent_root"]) / "loop-hop-execution.lock").open("a")
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(self.global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                self.update(
                    state="waiting_parent_completion",
                    parent=self.config["parent"]["root"],
                    message="CPU contracts passed; no GPU/container slot requested",
                )
                if time.monotonic() - started > rt["waiting_timeout_hours"] * 3600:
                    raise TimeoutError("Parent completion wait expired") from None
                time.sleep(rt["poll_seconds"])
        parent = Path(self.config["parent"]["root"])
        if not (parent / "completion-manifest.json").exists():
            raise RuntimeError("Parent lock released without its verified completion manifest")
        if not read(parent / "completion-manifest.json")["passed"]:
            raise RuntimeError("Parent completion did not pass")
        reused = {}
        for name in ("loop-l2-r2", "disco-l2-r2"):
            folder = parent / "runs" / name
            complete = read(folder / "complete.json")
            if complete["step"] != 300000:
                raise ValueError("Reused baseline is not the fixed endpoint")
            checkpoint = folder / "checkpoint-0300000.pt"
            if sha(checkpoint) != complete["final_checkpoint_sha256"]:
                raise ValueError("Reused baseline checkpoint bytes changed")
            for filename in ("audit.json", "grid-audit.json", "cloud-audit.json"):
                if not read(folder / filename)["passed"]:
                    raise ValueError(f"Reused baseline audit failed: {name}/{filename}")
            reused[name] = {
                "root": str(folder),
                "step": complete["step"],
                "files": {
                    p: sha(folder / p)
                    for p in (
                        "complete.json",
                        "audit.json",
                        "grid.json",
                        "grid-audit.json",
                        "cloud-audit.json",
                    )
                },
                "checkpoint_sha256": complete["final_checkpoint_sha256"],
            }
        write(self.base / "reused-baselines.json", reused)
        write(
            Path(rt["persistent_root"]) / "loop-hop-current.json",
            {
                "batch": BATCH,
                "gpu": 9,
                "pid": os.getpid(),
                "parent_batch": "loop-hop-unified-v1",
                "utc": utc(),
            },
        )

    def run(self):
        self.execute("all", "contracts")
        write(self.base / "cpu-contracts-complete.json", {"passed": True, "utc": utc()})
        self.wait_parent()
        for engineering in (True, False):
            self.update(state="engineering" if engineering else "scientific_training")
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                jobs = {
                    pool.submit(self.worker, name, engineering): name
                    for name in self.config["arms"]
                }
                for future in concurrent.futures.as_completed(jobs):
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append({"arm": jobs[future], "error": repr(error)})
            if errors:
                self.update(state="needs_attention", errors=errors)
                return
            if engineering:
                write(
                    self.base / "engineering-complete.json",
                    {"passed": True, "arms": list(self.config["arms"]), "utc": utc()},
                )
        write(
            self.base / "completion-manifest.json",
            {
                "passed": True,
                "new_scientific_updates": len(self.config["arms"]) * 300000,
                "reused_baselines": read(self.base / "reused-baselines.json"),
                "arms": {
                    name: read(self.base / "runs" / name / "cloud-audit.json")
                    for name in self.config["arms"]
                },
                "utc": utc(),
            },
        )
        self.update(state="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read(ROOT / "configs" / f"{BATCH}.json")
    validate(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "batch": BATCH,
                    "new_arms": len(config["arms"]),
                    "reused_arms": 2,
                    "new_updates": len(config["arms"]) * 300000,
                }
            )
        )
        return
    base = Path(config["runtime"]["persistent_root"]) / BATCH
    if args.resume:
        verify(base)
    else:
        freeze(config, base)
    with (base / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(
            base / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "argv": sys.argv,
                "freeze_sha256": sha(base / "freeze.json"),
            },
        )
        controller = Controller(config, base)
        try:
            controller.run()
        except BaseException as error:
            controller.update(
                state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            raise


if __name__ == "__main__":
    main()
