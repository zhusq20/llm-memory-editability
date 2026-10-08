"""Freeze and run new learning/use experiments without touching historical batches."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run_memory_interface_next import environment, now, write

PROJECT = Path("/ossfs/workspace/llm-memory-editability")
SCRIPT = "scripts/execute_learning_use.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def freeze(path):
    config = read(path)
    assert config["runtime"]["docker_context"] == "lm-memory"
    assert config["runtime"]["image"].startswith("sha256:")
    assert set(config["gpus"]) <= set(range(8)) and config["gpus"]
    names = [spec["name"] for spec in config["specs"]]
    assert names and len(names) == len(set(names))
    assert all(Path(name).name == name for name in names)
    artifact = PROJECT / "docs/development-artifacts" / config["batch"]
    source = artifact / "source"
    if source.exists():
        raise FileExistsError("Frozen source already exists; use a new revision")
    assert not (PROJECT / "results" / config["batch"] / "controller-state.json").exists()
    inputs = {}
    for spec in config["specs"]:
        for key in ("data_file",):
            if key in spec:
                target = (PROJECT / spec[key]).resolve()
                spec[key] = str(target)
                spec[key + "_sha256"] = sha(target)
                if key == "data_file":
                    assert spec.get("data_sha256", sha(target)) == sha(target)
                    spec["data_sha256"] = sha(target)
                inputs[str(target)] = sha(target)
        if "parent_dir" in spec:
            parent = (PROJECT / spec["parent_dir"]).resolve()
            spec["parent_dir"] = str(parent)
            spec["parent_checkpoint_sha256"] = sha(parent / "model.pt")
            inputs[str(parent / "model.pt")] = spec["parent_checkpoint_sha256"]
            if (parent / "run.json").exists():
                inputs[str(parent / "run.json")] = sha(parent / "run.json")
        if "parent_run_dir" in spec:
            parent = (PROJECT / spec["parent_run_dir"]).resolve()
            spec["parent_run_dir"] = str(parent)
            parent_run_sha256 = sha(parent / "run.json")
            assert spec.get("parent_run_sha256", parent_run_sha256) == parent_run_sha256
            spec["parent_run_sha256"] = parent_run_sha256
            for filename in ("run.json", "sampling-plan.json", "audit.json"):
                target = parent / filename
                if target.exists():
                    inputs[str(target)] = sha(target)
            if "original_stage_a_steps" in spec:
                target = parent / f"predictions-{spec['original_stage_a_steps']:07d}.json"
                inputs[str(target)] = sha(target)
        if "parent_checkpoint" in spec:
            target = (PROJECT / spec["parent_checkpoint"]).resolve()
            spec["parent_checkpoint"] = str(target)
            checkpoint_sha256 = sha(target)
            assert spec.get("parent_checkpoint_sha256", checkpoint_sha256) == checkpoint_sha256
            spec["parent_checkpoint_sha256"] = checkpoint_sha256
            inputs[str(target)] = checkpoint_sha256
        if "tokenizer" in spec:
            tokenizer = (PROJECT / spec["tokenizer"]).resolve()
            spec["tokenizer"] = str(tokenizer)
            for token_file in tokenizer.iterdir():
                if token_file.is_file():
                    inputs[str(token_file)] = sha(token_file)
    files = list((PROJECT / "src/llm_memory_editability").glob("*.py"))
    files += [PROJECT / SCRIPT, PROJECT / "scripts/run_memory_interface_next.py"]
    files += [PROJECT / "configs/experiment-tracking-defaults.json", path.resolve()]
    files += [PROJECT / name for name in config.get("additional_sources", [])]
    hashes = {}
    for file in sorted(set(files)):
        relative = file.relative_to(PROJECT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        hashes[str(relative)] = sha(target)
    config.update(
        frozen_utc=now(),
        repository=str(PROJECT),
        source_root=str(source),
        results_root=str(PROJECT / "results" / config["batch"]),
        source=hashes,
        input_files=inputs,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
    )
    destination = artifact / "frozen-config.json"
    write(destination, config)
    write(artifact / "execution-lock.json", {"config_sha256": sha(destination)})
    print(json.dumps({"config": str(destination), "runs": len(names)}))


def checked(path):
    assert sha(path) == read(path.with_name("execution-lock.json"))["config_sha256"]
    config = read(path)
    for relative, expected in config["source"].items():
        assert sha(Path(config["source_root"]) / relative) == expected, relative
    for target, expected in config["input_files"].items():
        assert sha(target) == expected, target
    return config


def gpu_available(gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    memory, utilization = [int(value.strip()) for value in result.stdout.strip().split(",")]
    if memory >= 256 or utilization >= 5:
        return False
    # A container can reserve a GPU before its first CUDA allocation.
    for context in ("default", "d157", "lm-memory"):
        identifiers = subprocess.check_output(
            ["docker", "--context", context, "ps", "-q"], text=True
        ).split()
        if not identifiers:
            continue
        snapshots = json.loads(
            subprocess.check_output(
                ["docker", "--context", context, "inspect", *identifiers], text=True
            )
        )
        for snapshot in snapshots:
            for request in snapshot["HostConfig"].get("DeviceRequests") or []:
                if request.get("Count") == -1 or str(gpu) in (request.get("DeviceIDs") or []):
                    return False
    return True


def claim_gpu(gpu):
    lock = open(f"/tmp/lm-learning-use-gpu-{gpu}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not gpu_available(gpu):
            lock.close()
            return None
        return lock
    except BlockingIOError:
        lock.close()
        return None


def start_tracker(config, root):
    command = [
        str(PROJECT / ".venv-wandb/bin/python"),
        "-u",
        "-m",
        "llm_memory_editability.interface_tracking",
        "--root",
        str(root),
        "--runs-dir",
        str(root / "runs"),
        "--defaults",
        config["source_root"] + "/configs/experiment-tracking-defaults.json",
    ]
    with (root / "tracking.log").open("a") as log:
        process = subprocess.Popen(
            command,
            env=environment(Path(config["source_root"])),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    write(root / "tracking-process.json", {"pid": process.pid, "command": command})
    deadline = time.monotonic() + 45
    while not (root / "tracking-ready.json").exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("W&B sidecar failed to initialize; see tracking.log")
        time.sleep(0.25)
    return process


def controller(path):
    config = checked(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "controller-state.json").exists():
        raise FileExistsError(
            "Existing history must be explicitly recovered, never silently restarted"
        )
    shutil.copy2(path, root / "frozen-config.json")
    state = dict(
        state="starting",
        started_utc=now(),
        completed=[],
        failed=[],
        active={},
        queued=[spec["name"] for spec in config["specs"]],
        gpus=config["gpus"],
    )
    write(root / "controller-state.json", state)
    active, pending = {}, list(enumerate(config["specs"]))
    tracker = None
    try:
        tracker = start_tracker(config, root)
        state["state"] = "running"
        while pending or active:
            for gpu, job in list(active.items()):
                if job["process"].poll() is None:
                    continue
                job["log"].close()
                out = job["out"]
                audit = read(out / "audit.json") if (out / "audit.json").exists() else {}
                passed = job["process"].returncode == 0 and audit.get("passed") is True
                state["completed" if passed else "failed"].append(job["spec"]["name"])
                write(
                    out / "process-status.json",
                    {
                        "returncode": job["process"].returncode,
                        "passed": passed,
                        "utc": now(),
                    },
                )
                if not passed and not (out / "failure.json").exists():
                    write(out / "failure.json", {"reason": "worker_or_independent_audit_failed"})
                job["gpu_lock"].close()
                del active[gpu]
            if tracker.poll() is not None:
                state["tracking_failure"] = tracker.returncode
            # A failed contract pauses further submissions; already running jobs finish normally.
            can_schedule = not state["failed"] and "tracking_failure" not in state
            if can_schedule:
                for gpu in config["gpus"]:
                    if gpu in active or not pending:
                        continue
                    gpu_lock = claim_gpu(gpu)
                    if gpu_lock is None:
                        continue
                    index, spec = pending.pop(0)
                    out = root / "runs" / spec["name"]
                    out.mkdir(parents=True, exist_ok=False)
                    command = [
                        sys.executable,
                        "-u",
                        config["source_root"] + "/scripts/run_memory_interface_next.py",
                        "worker",
                        "--config",
                        str(path),
                        "--index",
                        str(index),
                        "--gpu",
                        str(gpu),
                    ]
                    log = (out / "worker.log").open("a")
                    process = subprocess.Popen(
                        command,
                        env=environment(Path(config["source_root"])),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    active[gpu] = dict(
                        process=process, log=log, out=out, spec=spec, gpu_lock=gpu_lock
                    )
                    write(out / "host-worker.json", {"pid": process.pid, "command": command})
            state.update(
                active={
                    str(gpu): {"name": job["spec"]["name"], "pid": job["process"].pid}
                    for gpu, job in active.items()
                },
                queued=[spec["name"] for _, spec in pending],
                updated_utc=now(),
            )
            write(root / "controller-state.json", state)
            if not can_schedule and not active:
                break
            time.sleep(2)
        state.update(
            state="complete"
            if not pending and not state["failed"] and "tracking_failure" not in state
            else "finished_with_failures",
            finished_utc=now(),
        )
        write(root / "controller-state.json", state)
    except BaseException:
        state.update(
            state="finished_with_failures", error=traceback.format_exc(), updated_utc=now()
        )
        write(root / "controller-state.json", state)
        raise
    finally:
        # Do not kill scientific jobs on a controller error. Preserve ownership and evidence.
        for job in active.values():
            job["log"].close()
        if tracker is not None and tracker.poll() is not None:
            write(root / "tracking-exit.json", {"returncode": tracker.returncode})


def launch(path):
    config = checked(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "controller-process.json").exists():
        raise FileExistsError("Controller has already been launched")
    command = [
        sys.executable,
        "-u",
        config["source_root"] + "/" + SCRIPT,
        "controller",
        "--config",
        str(path),
    ]
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            command,
            env=environment(Path(config["source_root"])),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(root / "controller-process.json", {"pid": process.pid, "command": command, "utc": now()})
    print(
        json.dumps(
            {
                "pid": process.pid,
                "results": str(root),
                "wandb": "https://wandb.ai/zhusq20/llm-memory-editability",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "launch", "controller"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "launch": launch, "controller": controller}[args.action](
        args.config.resolve()
    )
