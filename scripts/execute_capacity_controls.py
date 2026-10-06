"""Launch the bounded storage/recall controls on the released LM GPU 0/1."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import execute_parametric_architecture as scheduler
from execute_capacity_scaling import devices_free, predecessor_status

read, write, digest = scheduler.read, scheduler.write, scheduler.digest


def freeze(path):
    config = read(path)
    repo = Path.cwd().resolve()
    artifact = repo / "docs/development-artifacts" / config["batch"]
    source = artifact / "source"
    if source.exists():
        raise FileExistsError(source)
    assert config["gpus"] == [0, 1]
    assert config["runtime"]["docker_context"] == "lm-memory"
    config.update(
        repository=str(repo),
        source_root=str(source),
        source_files={},
        execution_lock=str(artifact / "execution-lock.json"),
        frozen_utc=scheduler.now(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    for spec in config["runs"]:
        if spec.get("parent"):
            parent = Path(spec["parent"])
            assert read(parent / "audit.json")["passed"]
            spec["parent_checkpoint_sha256"] = digest(parent / "latest.pt")
    files = sorted(
        set(
            list(Path("src/llm_memory_editability").glob("*.py"))
            + [path.relative_to(repo), Path("configs/experiment-tracking-defaults.json")]
            + [Path(p) for p in config["additional_sources"]]
        )
    )
    for filename in files:
        target = source / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename, target)
        config["source_files"][str(target)] = digest(target)
    frozen = artifact / "frozen-config.json"
    write(frozen, config)
    write(config["execution_lock"], {"config_sha256": digest(frozen), "utc": scheduler.now()})
    print(frozen)


def command_builder(config, config_path, spec, out, gpu, attempt):
    _, command = scheduler.container_command(config, config_path, spec, out, gpu, attempt)
    old = str(Path(config["source_root"]) / "scripts/execute_parametric_architecture.py")
    command[command.index(old)] = str(
        Path(config["source_root"]) / "scripts/execute_capacity_controls.py"
    )
    name = (
        "lm-capctrl-"
        + hashlib.sha256(f"{config['batch']}:{spec['name']}:{attempt}".encode()).hexdigest()[:20]
    )
    command[command.index("--name") + 1] = name
    return name, command


def worker(path, name):
    config = read(path)
    spec = next(s for s in config["runs"] if s["name"] == name)
    filename = (
        "evaluate_capacity_recall.py" if spec.get("evaluation_only") else "run_capacity_controls.py"
    )
    script = Path(config["source_root"]) / "scripts" / filename
    out = Path(config["results_root"]) / "runs" / name
    try:
        for action, timeout in (("train", spec["max_wall_seconds"]), ("audit", 1200)):
            subprocess.run(
                [sys.executable, "-u", str(script), action, "--config", str(path), "--run", name],
                check=True,
                timeout=timeout,
            )
    except BaseException:
        write(out / "failure.json", {"error": traceback.format_exc(), "utc": scheduler.now()})
        raise


def controller(path):
    config = read(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while config.get("predecessors"):
            ready, evidence = predecessor_status(config["predecessors"])
            if ready:
                break
            write(
                root / "controller-state.json",
                {
                    "state": "waiting_for_predecessor",
                    "evidence": evidence,
                    "updated_utc": scheduler.now(),
                },
            )
            time.sleep(20)
        while not devices_free(config["gpus"]):
            write(root / "controller-state.json", {"state": "waiting_for_gpu_0_1"})
            time.sleep(20)
        scheduler.control_locked(config, path, root, command_builder=command_builder)
    subprocess.run(
        [
            sys.executable,
            str(Path(config["source_root"]) / "scripts/report_capacity_controls.py"),
            "--root",
            str(root),
        ],
        check=True,
    )


def launch(path):
    config = read(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    script = Path(config["source_root"]) / "scripts/execute_capacity_controls.py"
    cmd = [sys.executable, "-u", str(script), "controller", "--config", str(path)]
    with (root / "controller.log").open("a") as log:
        proc = subprocess.Popen(
            cmd,
            env=scheduler.host_env(config["source_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(
        root / "controller-process.json", {"pid": proc.pid, "command": cmd, "utc": scheduler.now()}
    )
    print(proc.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "launch", "controller", "container-worker"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run")
    args = parser.parse_args()
    if args.action == "container-worker":
        worker(args.config, args.run)
    else:
        {"freeze": freeze, "launch": launch, "controller": controller}[args.action](
            args.config.resolve()
        )
