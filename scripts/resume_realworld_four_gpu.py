"""Resume the frozen TwoWiki matrix on physical GPUs 2–5, with nine slots."""

from __future__ import annotations

import fcntl
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/realworld-composition-confirmation-v1.json"
# Start the untrained run immediately; distribute the eight reloads around it.
ASSIGNMENTS = [
    (2, "confirmation-standard4-init3"),
    (2, "confirmation-standard8-init1"),
    (3, "confirmation-standard8-init2"),
    (3, "confirmation-loop4x2-init1"),
    (4, "confirmation-standard8-init3"),
    (4, "confirmation-loop4x2-init2"),
    (5, "confirmation-loop4x2-init3"),
    (5, "confirmation-standard4-init1"),
    (5, "confirmation-standard4-init2"),
]


def main():
    assert os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] == "/lib64"
    assert not os.environ.get("CUDA_VISIBLE_DEVICES"), "Use physical GPU indices"
    config = json.loads(CONFIG.read_text())
    source = Path(config["source_snapshot"])
    sys.path.insert(0, str(source / "src"))
    frozen = runpy.run_path(str(source / "scripts/run_realworld_composition.py"))
    frozen["verify"](config)
    write = frozen["write_json"]
    utc = frozen["utc"]
    root = Path(config["results_root"])
    art = Path(config["artifact_root"])
    specs = {s["name"]: s for s in config["runs"]}
    assert set(specs) == {name for _, name in ASSIGNMENTS}
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A surviving worker must never share its result directory with another writer.
    for proc in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = proc.read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if str(CONFIG).encode() in command and any(
            mode in command for mode in (b"worker", b"audit", b"controller")
        ):
            raise RuntimeError(f"Existing experiment process: {proc.parent.name}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    recovery = root / "recovery" / stamp
    recovery.mkdir(parents=True)
    for base, names in [
        (root, ["controller.json", "controller-state.json"]),
        (art, ["confirmation-completion.json", "confirmation-summary.json"]),
    ]:
        for name in names:
            if (base / name).exists():
                shutil.copy2(base / name, recovery / name)
    for _, name in ASSIGNMENTS:
        out = root / "confirmation" / name
        saved = recovery / name
        saved.mkdir()
        for filename in ["run.json", "status.json", "trained.json"]:
            if (out / filename).exists():
                shutil.copy2(out / filename, saved / filename)
        for filename in ["failure.json", "attempt-failure.json"]:
            if (out / filename).exists():
                shutil.move(str(out / filename), saved / filename)
    write(
        recovery / "manifest.json",
        {
            "created_utc": utc(),
            "pid": os.getpid(),
            "config": str(CONFIG),
            "config_sha256": frozen["sha256"](CONFIG),
            "assignments": ASSIGNMENTS,
            "driver_library_prefix": "/lib64",
            "reason": "User requested recovery on four idle GPUs after CUDA error 803",
            "scientific_configuration_unchanged": True,
        },
    )
    write(root / "controller.json", {"pid": os.getpid(), "recovery": str(recovery)})
    environment = {**os.environ, "PYTHONPATH": str(source / "src")}
    active, completed, failed = {}, [], []
    retries = {}

    def state(phase="running"):
        result = {
            "state": phase,
            "pid": os.getpid(),
            "updated_utc": utc(),
            "recovery": str(recovery),
            "completed": completed,
            "failed": failed,
            "active": {
                name: {"gpu": gpu, "pid": child.pid, "mode": mode}
                for name, (gpu, child, mode) in active.items()
            },
            "queued": [],
            "retries": retries,
        }
        write(root / "controller-state.json", result)
        return result

    def launch(gpu, name):
        out = root / "confirmation" / name
        out.mkdir(parents=True, exist_ok=True)
        mode = "audit" if (out / "trained.json").exists() else "worker"
        with (recovery / name / f"{mode}.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(source / "scripts/run_realworld_composition.py"),
                    mode,
                    "--config",
                    str(CONFIG),
                    "--name",
                    name,
                    "--gpu",
                    str(gpu),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        if mode == "audit" and (out / "run.json").exists():
            metadata = json.loads((out / "run.json").read_text())
            metadata.update(pid=child.pid, gpu=gpu, recovery_mode="audit")
            write(out / "run.json", metadata)
        active[name] = (gpu, child, mode)
        print(json.dumps({"name": name, "gpu": gpu, "pid": child.pid, "mode": mode}), flush=True)

    state()
    for gpu, name in ASSIGNMENTS:
        if (root / "confirmation" / name / "complete.json").exists():
            completed.append(name)
        else:
            launch(gpu, name)
    state()
    tracking_command = [
        str(PROJECT / ".venv-wandb/bin/python"),
        str(source / "scripts/track_experiment_wandb.py"),
        "--root",
        str(root),
        "--runs-dir",
        str(root / "confirmation"),
        "--defaults",
        str(PROJECT / "configs/experiment-tracking-defaults.json"),
        "--detach",
    ]

    def ensure_tracker():
        metadata = root / "tracking-wandb/process.json"
        if metadata.exists() and frozen["process_alive"](json.loads(metadata.read_text())["pid"]):
            return
        subprocess.run(tracking_command, check=True, env=environment)

    next_tracking_check = 0.0
    while active:
        if time.monotonic() >= next_tracking_check:
            next_tracking_check = time.monotonic() + 60
            try:
                ensure_tracker()
            except Exception:
                with (recovery / "tracking-errors.log").open("a") as log:
                    log.write(traceback.format_exc())
        for name, (gpu, child, mode) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            del active[name]
            if code:
                out = root / "confirmation" / name
                path = out / "attempt-failure.json"
                detail = (
                    json.loads(path.read_text()) if path.exists() else {"error": "Process exited"}
                )
                failure = {
                    **detail,
                    "name": name,
                    "mode": mode,
                    "exit_code": code,
                    "attempt": retries.get(name, 0),
                    "created_utc": utc(),
                }
                write(recovery / name / f"failure-{mode}-{retries.get(name, 0)}.json", failure)
                transient = code < 0 or any(
                    s in detail["error"]
                    for s in ["CUDA out of memory", "Input/output error", "Connection reset"]
                )
                if transient and retries.get(name, 0) < config.get("transient_retries", 0):
                    retries[name] = retries.get(name, 0) + 1
                    launch(gpu, name)
                else:
                    failed.append(failure)
                    write(out / "failure.json", failure)
            elif mode == "worker":
                launch(gpu, name)
            else:
                completed.append(name)
        state()
        if active:
            time.sleep(5)
    if not failed:
        state("finalizing")
        try:
            frozen["verify"](config)
            frozen["finalize_confirmation"](config)
            frozen["analyze_confirmation"](config)
        except Exception:
            failed.append({"mode": "independent_recount", "error": traceback.format_exc()})
            write(recovery / "finalization-failure.json", failed[-1])
    final = state("finished_with_failures" if failed else "complete")
    final.update(
        formal_training_started=True,
        independent_recount_passed=not failed,
        scientific_updates=sum(specs[n]["steps"] for n in completed),
    )
    write(root / "controller-state.json", final)
    write(art / "confirmation-completion.json", final)
    write(
        art / "confirmation-summary.json",
        {
            name: json.loads((root / "confirmation" / name / "endpoint.json").read_text())
            for name in completed
        },
    )
    write(recovery / "completion.json", final)


if __name__ == "__main__":
    main()
