"""Freeze and execute sequential research stages on explicitly assigned GPUs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(source):
    env = dict(os.environ)
    if not env.get("MEMORY_INTERFACE_CONTAINER"):
        env["LD_LIBRARY_PATH"] = "/lib64" + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(source / "src"))
    return env


def freeze(input_path):
    config = json.loads(input_path.read_text())
    root = Path.cwd().resolve()
    artifact = root / "docs/development-artifacts" / config["batch"]
    source = artifact / "source"
    if source.exists():
        raise FileExistsError(source)
    assert len(config["gpus"]) == 4 and set(config["gpus"]) == {2, 3, 4, 5}
    assert config["runtime"]["image"].startswith("sha256:")
    assert config["runtime"]["docker_host"].startswith("unix://")
    assert config["runtime"]["docker_context"] == "lm-memory", "Use only the LM project's daemon"
    assert config["runtime"]["docker_host"] == "unix:///run/docker-lm-memory/docker.sock"
    names = [s["name"] for s in config["specs"]]
    assert len(names) == len(set(names)) and names
    files = sorted(
        list(Path("src/llm_memory_editability").glob("*.py"))
        + [Path(__file__).relative_to(root), Path("configs/experiment-tracking-defaults.json")]
        + [Path(p) for p in config.get("additional_sources", [])]
    )
    config["source"] = {}
    for path in files:
        dest = source / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        config["source"][str(path)] = digest(path)
    config.update(
        frozen_utc=now(),
        repository=str(root),
        source_root=str(source),
        results_root=str(root / "results" / config["batch"]),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    for spec in config["specs"]:
        if "parent_dir" in spec:
            parent = (root / spec["parent_dir"]).resolve()
            spec["parent_dir"] = str(parent)
            spec["parent_checkpoint_sha256"] = digest(parent / "model.pt")
    path = artifact / "frozen-config.json"
    write(path, config)
    print(json.dumps({"config": str(path), "runs": len(names), "source_files": len(files)}))


def launch(config_path):
    config = json.loads(config_path.read_text())
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    saved_config = root / "frozen-config.json"
    if saved_config.exists() and saved_config.read_bytes() != config_path.read_bytes():
        raise RuntimeError("The result directory contains a different frozen configuration")
    shutil.copyfile(config_path, saved_config)
    state = root / "controller-state.json"
    if state.exists() and json.loads(state.read_text()).get("state") == "running":
        raise RuntimeError("A running controller already owns this stage")
    script = Path(config["source_root"]) / "scripts" / Path(__file__).name
    command = [sys.executable, "-u", str(script), "controller", "--config", str(config_path)]
    with (root / "controller.log").open("a") as log:
        proc = subprocess.Popen(
            command,
            env=environment(Path(config["source_root"])),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(root / "controller-process.json", {"pid": proc.pid, "command": command})
    print(json.dumps({"pid": proc.pid, "root": str(root)}))


def controller(config_path):
    config = json.loads(config_path.read_text())
    root, source = Path(config["results_root"]), Path(config["source_root"])
    for path, expected in config["source"].items():
        assert digest(source / path) == expected, f"Frozen source changed: {path}"
    state = {
        "state": "running",
        "started_utc": now(),
        "gpus": config["gpus"],
        "completed": [],
        "failed": [],
        "active": {},
    }
    write(root / "controller-state.json", state)
    env = environment(source)
    tracking_python = str(Path(config["repository"]) / ".venv-wandb/bin/python")
    tracker_command = [
        tracking_python,
        "-u",
        "-m",
        "llm_memory_editability.interface_tracking",
        "--root",
        str(root),
        "--runs-dir",
        str(root / "runs"),
        "--defaults",
        str(source / "configs/experiment-tracking-defaults.json"),
    ]
    tracking_log = (root / "tracking.log").open("a")
    tracker = subprocess.Popen(
        tracker_command, env=env, stdout=tracking_log, stderr=subprocess.STDOUT
    )
    write(root / "tracking-process.json", {"pid": tracker.pid, "command": tracker_command})
    queue = list(range(len(config["specs"])))
    active = {}
    try:
        deadline = time.monotonic() + 30
        while not (root / "tracking-ready.json").exists() and tracker.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)
        if tracker.poll() is not None or not (root / "tracking-ready.json").exists():
            state["tracking_failure"] = "Tracking process did not become ready"
            state["state"] = "finished_with_failures"
            write(root / "controller-state.json", state)
            raise RuntimeError(state["tracking_failure"])
        while queue or active:
            if tracker.poll() is not None:
                state["tracking_failure"] = f"Tracking process exited early: {tracker.returncode}"
            for gpu, (proc, index, log) in list(active.items()):
                result = proc.poll()
                if result is None:
                    continue
                log.close()
                name = config["specs"][index]["name"]
                state["completed" if result == 0 else "failed"].append(name)
                del active[gpu]
            # Stop scheduling after a failed contract rather than silently run an invalid matrix.
            if state["failed"] or state.get("tracking_failure"):
                queue.clear()
            for gpu in config["gpus"]:
                if gpu in active or not queue:
                    continue
                index = queue.pop(0)
                spec = config["specs"][index]
                out = root / "runs" / spec["name"]
                if (
                    (out / "complete.json").exists()
                    and (out / "audit.json").exists()
                    and json.loads((out / "audit.json").read_text()).get("passed") is True
                ):
                    state["completed"].append(spec["name"])
                    continue
                out.mkdir(parents=True, exist_ok=True)
                log = (out / "worker.log").open("a")
                command = [
                    sys.executable,
                    "-u",
                    __file__,
                    "worker",
                    "--config",
                    str(config_path),
                    "--index",
                    str(index),
                    "--gpu",
                    str(gpu),
                ]
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (proc, index, log)
            state["active"] = {
                str(gpu): {"pid": p.pid, "name": config["specs"][i]["name"]}
                for gpu, (p, i, _) in active.items()
            }
            state["updated_utc"] = now()
            write(root / "controller-state.json", state)
            time.sleep(2)
        state["state"] = (
            "complete"
            if not state["failed"] and not state.get("tracking_failure")
            else "finished_with_failures"
        )
        state["finished_utc"] = now()
        write(root / "controller-state.json", state)
        try:
            tracker.wait(timeout=120)
        except subprocess.TimeoutExpired:
            write(root / "tracking-pending.json", {"pid": tracker.pid, "reason": "sync pending"})
        write(root / "tracking-exit.json", {"returncode": tracker.returncode})
    finally:
        tracking_log.close()


def container_work(config_path, index, gpu):
    """One isolated container per experiment, including its fresh-process audit."""
    config = json.loads(config_path.read_text())
    spec, runtime = config["specs"][index], config["runtime"]
    out = Path(config["results_root"]) / "runs" / spec["name"]
    docker = (
        ["docker", "--context", runtime["docker_context"]]
        if runtime.get("docker_context")
        else ["docker", "--host", runtime["docker_host"]]
    )
    name = (
        "lm-interface-" + hashlib.sha256((config["batch"] + spec["name"]).encode()).hexdigest()[:20]
    )
    command = docker + [
        "run",
        "--name",
        name,
        "--label",
        "project=llm-memory-editability",
        "--label",
        "batch=" + config["batch"],
        "--network=none",
        "--gpus",
        f"device={gpu}",
        "--cpus",
        str(runtime.get("cpus", 4)),
        "--memory",
        runtime.get("memory", "16g"),
        "--shm-size=1g",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--mount",
        f"type=bind,src={config['repository']},dst={config['repository']},readonly",
        "--mount",
        f"type=bind,src={out},dst={out}",
        "--workdir",
        config["repository"],
        "--env",
        "MEMORY_INTERFACE_CONTAINER=1",
        "--env",
        f"PHYSICAL_GPU={gpu}",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "PYTHONPATH=" + str(Path(config["source_root"]) / "src"),
    ]
    for key, value in runtime.get("env", {}).items():
        command += ["--env", f"{key}={value}"]
    command += [
        runtime["image"],
        runtime["python"],
        "-u",
        __file__,
        "container-worker",
        "--config",
        str(config_path),
        "--index",
        str(index),
        "--gpu",
        "0",
    ]
    write(out / "container-command.json", {"name": name, "physical_gpu": gpu, "command": command})
    process = subprocess.Popen(command)
    for _ in range(50):
        snapshot = subprocess.run(docker + ["inspect", name], capture_output=True, text=True)
        if snapshot.returncode == 0:
            info = json.loads(snapshot.stdout)[0]
            if info["State"].get("Running"):
                write(
                    out / "container-runtime.json",
                    {
                        "physical_gpu": gpu,
                        "host_pid": info["State"]["Pid"],
                        "container_id": info["Id"],
                        "image": info["Image"],
                        "cpus": runtime.get("cpus", 4),
                        "memory": runtime.get("memory", "16g"),
                    },
                )
                break
        if process.poll() is not None:
            break
        time.sleep(0.2)
    result = process.wait()
    inspected = subprocess.run(docker + ["inspect", name], capture_output=True, text=True)
    if inspected.returncode == 0:
        write(out / "container-inspect.json", json.loads(inspected.stdout))
    if result:
        raise RuntimeError(f"Experiment container exited {result}: {name}")


def work(config_path, index, gpu, audit_only=False):
    import torch

    config = json.loads(config_path.read_text())
    spec = config["specs"][index]
    out = Path(config["results_root"]) / "runs" / spec["name"]
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{gpu}")
    if "parent_dir" in spec:
        assert digest(Path(spec["parent_dir"]) / "model.pt") == spec["parent_checkpoint_sha256"]
    module = importlib.import_module("llm_memory_editability." + config["module"])
    if audit_only:
        module.audit(out, device)
        return
    if not (out / "complete.json").exists():
        if config["module"] == "alignment_coverage":
            module.train(spec, out, config["source"], device)
        else:
            module.run(spec, out, device)
    # Deliberately reload in a fresh Python process, not the writer's model object.
    subprocess.run(
        [
            sys.executable,
            "-u",
            __file__,
            "audit",
            "--config",
            str(config_path),
            "--index",
            str(index),
            "--gpu",
            str(gpu),
        ],
        check=True,
        env=environment(Path(config["source_root"])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["freeze", "launch", "controller", "worker", "container-worker", "audit"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    path = args.config.resolve()
    if args.action == "freeze":
        freeze(path)
    elif args.action == "launch":
        launch(path)
    elif args.action == "controller":
        controller(path)
    else:
        config = json.loads(path.read_text())
        out = Path(config["results_root"]) / "runs" / config["specs"][args.index]["name"]
        try:
            if args.action == "worker":
                with open(f"/tmp/llm-memory-interface-gpu-{args.gpu}.lock", "w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    container_work(path, args.index, args.gpu)
            elif args.action == "container-worker":
                assert os.environ.get("MEMORY_INTERFACE_CONTAINER") == "1"
                work(path, args.index, args.gpu)
            else:
                work(path, args.index, args.gpu, True)
        except Exception:
            write(
                out / "failure.json",
                {"action": args.action, "utc": now(), "traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    main()
