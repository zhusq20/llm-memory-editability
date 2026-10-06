#!/usr/bin/env python3
"""Freeze and execute development, support curves, then prerequisite-gated load curves."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.composition_curves import (  # noqa: E402
    anchors_pass,
    calibrated_choice,
    confirmation_specs,
    load_specs,
    sha256,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def environment(source):
    return {
        **os.environ,
        "PYTHONPATH": str(Path(source) / "src"),
        "LD_LIBRARY_PATH": "/lib64"
        + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
    }


def freeze(path):
    config = read(path)
    artifact = ROOT / "docs/development-artifacts" / config["batch"]
    source = artifact / "source"
    if source.exists():
        raise FileExistsError("This batch already has a frozen source snapshot")
    assert config["runtime"]["docker_context"] == "lm-memory"
    assert config["gpus"] == [0, 1]
    preflight = Path(config["preflight"])
    assert read(preflight)["passed"] is True
    files = [
        *Path("src/llm_memory_editability").glob("*.py"),
        Path("scripts/run_composition_curves.py"),
        Path("scripts/execute_composition_curves.py"),
        Path("scripts/report_composition_curves.py"),
        Path("tests/test_composition_curves.py"),
        Path("configs/experiment-tracking-defaults.json"),
        path.relative_to(ROOT),
        Path("docs/development-artifacts") / config["batch"] / "protocol.md",
    ]
    hashes = {}
    for file in files:
        target = source / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / file, target)
        hashes[str(target)] = sha256(target)
    config.update(
        repository=str(ROOT),
        source_root=str(source),
        source_files=hashes,
        preflight_sha256=sha256(preflight),
        frozen_utc=utc(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    frozen = artifact / "frozen-config.json"
    write_json(frozen, config)
    write_json(artifact / "execution-lock.json", {"config_sha256": sha256(frozen), "utc": utc()})
    print(json.dumps({"config": str(frozen), "sources": len(hashes)}))


def container_command(config, path, spec, out, gpu, attempt):
    name = (
        "lm-curves-"
        + hashlib.sha256(f"{config['batch']}:{spec['name']}:{attempt}".encode()).hexdigest()[:20]
    )
    runtime = config["runtime"]
    repository = Path(config["repository"])
    command = [
        "docker",
        "--context",
        "lm-memory",
        "run",
        "--name",
        name,
        "--label",
        "project=llm-memory-editability",
        "--label",
        "batch=" + config["batch"],
        "--label",
        "run=" + spec["name"],
        "--network=none",
        "--gpus",
        f"device={gpu}",
        "--cpus",
        str(runtime["cpus"]),
        "--memory",
        runtime["memory"],
        "--shm-size=2g",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--mount",
        f"type=bind,src={repository},dst={repository},readonly",
        "--mount",
        f"type=bind,src={out},dst={out}",
        "--workdir",
        str(repository),
    ]
    for key, value in {
        "PHYSICAL_GPU": gpu,
        "PYTHONDONTWRITEBYTECODE": 1,
        "OMP_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
        "PYTHONPATH": str(Path(config["source_root"]) / "src"),
        "XDG_CACHE_HOME": "/tmp/cache",
    }.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend(
        [
            runtime["image"],
            runtime["python"],
            "-u",
            str(Path(config["source_root"]) / "scripts/execute_composition_curves.py"),
            "worker",
            "--config",
            str(path),
            "--run",
            spec["name"],
        ]
    )
    return name, command


def worker(path, name):
    config = read(path)
    label = config["phase"]
    assert sha256(path) == read(path.with_name(label + "-execution-lock.json"))["config_sha256"]
    spec = next(item for item in config["runs"] if item["name"] == name)
    out = Path(config["results_root"]) / "runs" / name
    script = Path(config["source_root"]) / "scripts/run_composition_curves.py"
    arguments = ["--config", str(path), "--run", name, "--out", str(out)]
    begun = time.monotonic()
    try:
        train = [sys.executable, "-u", str(script), "train", *arguments]
        if (out / "latest.pt").exists():
            train.append("--resume")
        if not (out / "complete.json").exists():
            subprocess.run(train, check=True, timeout=spec["max_wall_seconds"])
        subprocess.run(
            [sys.executable, "-u", str(script), "audit", *arguments], check=True, timeout=1800
        )
        write_json(
            out / "worker-completion.json",
            {"passed": True, "wall_seconds": time.monotonic() - begun},
        )
    except BaseException:
        write_json(out / "failure.json", {"utc": utc(), "traceback": traceback.format_exc()})
        raise


def gpu_free(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.splitlines()}
    if memory[gpu] > 256:
        return False
    for context in ("default", "lm-memory", "d157"):
        output = subprocess.run(
            ["docker", "--context", context, "ps", "-q"], capture_output=True, text=True
        )
        if output.returncode:
            return False
        if not output.stdout.strip():
            continue
        containers = json.loads(
            subprocess.check_output(
                ["docker", "--context", context, "inspect", *output.stdout.split()], text=True
            )
        )
        for container in containers:
            for request in container["HostConfig"].get("DeviceRequests") or []:
                if request.get("Count") == -1 or str(gpu) in (request.get("DeviceIDs") or []):
                    return False
    return True


def start_tracking(config, root):
    previous = root / "tracking-process.json"
    if previous.exists():
        old = read(previous)
        command_file = Path(f"/proc/{old['pid']}/cmdline")
        if command_file.exists():
            actual = command_file.read_bytes().replace(b"\x00", b" ").decode()
            if "llm_memory_editability.curve_tracking" in actual and str(root) in actual:
                write_json(root / f"tracking-recovery-{time.time_ns()}.json", old)
                os.kill(old["pid"], signal.SIGTERM)
                time.sleep(1)
    command = [
        str(Path(config["repository"]) / ".venv-wandb/bin/python"),
        "-u",
        "-m",
        "llm_memory_editability.curve_tracking",
        "--root",
        str(root),
        "--runs-dir",
        str(root / "runs"),
        "--defaults",
        str(Path(config["source_root"]) / "configs/experiment-tracking-defaults.json"),
    ]
    log = (root / "tracking.log").open("a")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=environment(config["source_root"])
    )
    write_json(root / "tracking-process.json", {"pid": process.pid, "command": command})
    return process, log


def recover_active(config, state):
    """Adopt surviving workers after a controller interruption; never double launch."""
    root = Path(config["results_root"])
    names = subprocess.check_output(
        [
            "docker",
            "--context",
            "lm-memory",
            "ps",
            "--filter",
            "label=batch=" + config["batch"],
            "--format",
            "{{.Names}}",
        ],
        text=True,
    ).split()
    for name in names:
        info = json.loads(
            subprocess.check_output(
                ["docker", "--context", "lm-memory", "inspect", name], text=True
            )
        )[0]
        ids = info["HostConfig"]["DeviceRequests"][0]["DeviceIDs"]
        if len(ids) != 1 or int(ids[0]) not in config["gpus"]:
            raise RuntimeError("Recovered worker has an unexpected GPU assignment")
        gpu = ids[0]
        state["active"][gpu] = {"name": info["Config"]["Labels"]["run"], "container": name}
    while state["active"]:
        for gpu, item in list(state["active"].items()):
            info = json.loads(
                subprocess.check_output(
                    ["docker", "--context", "lm-memory", "inspect", item["container"]], text=True
                )
            )[0]
            if info["Config"]["Labels"]["batch"] != config["batch"]:
                raise RuntimeError("Recovered container belongs to another batch")
            if info["Image"] != config["runtime"]["image"]:
                raise RuntimeError("Recovered image differs from the frozen image")
            if info["State"].get("Running"):
                continue
            out = root / "runs" / item["name"]
            passed = info["State"]["ExitCode"] == 0 and (out / "audit.json").exists()
            passed = passed and read(out / "audit.json").get("passed") is True
            field = "completed" if passed else "failed"
            if item["name"] not in state[field]:
                state[field].append(item["name"])
            write_json(out / "recovered-container-inspect.json", info)
            del state["active"][gpu]
        state.update(updated_utc=utc(), state="recovering" if state["active"] else "running")
        write_json(root / "controller-state.json", state)
        if state["active"]:
            time.sleep(3)


def stage(config, specs, label, state):
    root = Path(config["results_root"])
    path = Path(config["source_root"]).parent / f"{label}-config.json"
    materialized = {**config, "runs": specs, "phase": label}
    if path.exists():
        if read(path) != materialized:
            raise ValueError("Attempted to change a materialized stage")
    else:
        write_json(path, materialized)
        write_json(
            path.with_name(label + "-execution-lock.json"),
            {"config_sha256": sha256(path), "utc": utc()},
        )
    queue = [spec for spec in specs if spec["name"] not in state["completed"]]
    if state.get("active"):
        raise RuntimeError("Existing active containers must be adopted before resuming")
    active = {}
    state.update(stage=label, state="running", stage_runs=[spec["name"] for spec in specs])
    while queue or active:
        for gpu, item in list(active.items()):
            if item["process"].poll() is None:
                continue
            item["log"].close()
            out = root / "runs" / item["spec"]["name"]
            info = json.loads(
                subprocess.check_output(
                    ["docker", "--context", "lm-memory", "inspect", item["container"]], text=True
                )
            )
            write_json(out / "container-inspect.json", info)
            passed = item["process"].returncode == 0 and (out / "audit.json").exists()
            passed = passed and read(out / "audit.json").get("passed") is True
            state["completed" if passed else "failed"].append(item["spec"]["name"])
            del active[gpu]
        if state["failed"]:
            state.update(queued=[spec["name"] for spec in queue], state="scientific_failure")
            write_json(root / "controller-state.json", state)
            if not active:
                raise RuntimeError("A registered scientific job failed; remaining queue preserved")
        spent = sum(
            read(file)["wall_seconds"] for file in (root / "runs").glob("*/worker-completion.json")
        )
        if spent > config["budget"]["maximum_gpu_hours"] * 3600:
            raise RuntimeError("Frozen cumulative GPU time budget reached")
        for gpu in config["gpus"]:
            if gpu in active or not queue or state["failed"] or not gpu_free(gpu):
                continue
            spec = queue.pop(0)
            out = root / "runs" / spec["name"]
            if (out / "failure.json").exists():
                raise RuntimeError("Recorded failure must not be silently overwritten")
            if (out / "audit.json").exists() and read(out / "audit.json").get("passed"):
                state["completed"].append(spec["name"])
                continue
            out.mkdir(parents=True, exist_ok=True)
            attempt = len(list(out.glob("container-command-*.json"))) + 1
            name, command = container_command(config, path, spec, out, gpu, attempt)
            write_json(
                out / f"container-command-{attempt}.json", {"name": name, "command": command}
            )
            log = (out / f"worker-{attempt}.log").open("a")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = {"process": process, "log": log, "container": name, "spec": spec}
            for _ in range(60):
                result = subprocess.run(
                    ["docker", "--context", "lm-memory", "inspect", name],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    info = json.loads(result.stdout)[0]
                    if info["State"].get("Running"):
                        write_json(
                            out / "container-runtime.json",
                            {
                                "physical_gpu": gpu,
                                "host_pid": info["State"]["Pid"],
                                "container_id": info["Id"],
                                "image": info["Image"],
                                "cpus": config["runtime"]["cpus"],
                                "memory": config["runtime"]["memory"],
                            },
                        )
                        break
                if process.poll() is not None:
                    break
                time.sleep(0.2)
        state.update(
            active={
                str(gpu): {
                    "name": item["spec"]["name"],
                    "container": item["container"],
                    "pid": item["process"].pid,
                }
                for gpu, item in active.items()
            },
            queued=[spec["name"] for spec in queue],
            updated_utc=utc(),
            consumed_gpu_hours=spent / 3600,
        )
        write_json(root / "controller-state.json", state)
        time.sleep(3)


def report(config, root):
    subprocess.run(
        [
            sys.executable,
            str(Path(config["source_root"]) / "scripts/report_composition_curves.py"),
            "--root",
            str(root),
        ],
        env=environment(config["source_root"]),
        check=True,
    )


def controller(path):
    config = read(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert sha256(path) == read(path.parent / "execution-lock.json")["config_sha256"]
        for file, expected in config["source_files"].items():
            assert sha256(file) == expected, file
        assert sha256(config["preflight"]) == config["preflight_sha256"]
        old = (
            read(root / "controller-state.json")
            if (root / "controller-state.json").exists()
            else {}
        )
        if old.get("state") == "complete":
            raise RuntimeError("This batch is already complete")
        state = {
            "state": "running",
            "started_utc": utc(),
            "completed": [],
            "failed": [],
            "active": {},
            "gpus": config["gpus"],
            **old,
        }
        write_json(root / "frozen-config.json", config)
        write_json(root / "controller-state.json", state)
        tracker, tracking_log = start_tracking(config, root)
        try:
            recover_active(config, state)
            selected = None
            choice_path = root / "development-selection.json"
            if choice_path.exists():
                selected = read(choice_path)["selected"]
            else:
                for steps in config["calibration_steps"]:
                    specs = [
                        {
                            **config["base_spec"],
                            "layers": layers,
                            "world_seed": config["development_world"],
                            "initialization": config["initialization"],
                            "stream_seed": config["stream_seed"],
                            "phi": config["anchor_phi"],
                            "phase": "development",
                            "fixed_steps": steps,
                            "name": f"dev-l{layers}-s{steps}",
                        }
                        for layers in config["calibration_layers"]
                    ]
                    stage(config, specs, f"calibration-{steps}", state)
                    records = [
                        (
                            spec,
                            read(root / "runs" / spec["name"] / "complete.json")["endpoint"][
                                "metrics"
                            ],
                        )
                        for spec in specs
                    ]
                    choice = calibrated_choice(records, config["calibration_thresholds"])
                    report(config, root)
                    if choice is not None:
                        selected = {**choice}
                        selected.pop("fixed_steps")
                        selected.pop("name")
                        if steps > config["calibration_steps"][0]:
                            selected["target_exposures"] = config["budget"][
                                "extension_target_exposures"
                            ]
                        break
                write_json(
                    choice_path,
                    {"selected": selected, "rule": config["selection_rule"], "utc": utc()},
                )
            if selected is None:
                state.update(
                    state="complete",
                    conclusion="development_prerequisite_not_met",
                    load_curve_started=False,
                    finished_utc=utc(),
                )
                write_json(root / "controller-state.json", state)
            else:
                support = confirmation_specs(config, selected)
                stage(config, support, "support", state)
                report(config, root)
                gate, anchor_records = [], []
                for spec in support:
                    if spec["phi"] != config["anchor_phi"]:
                        continue
                    out = root / "runs" / spec["name"]
                    meta, history = read(out / "run.json"), read(out / "learning.json")
                    row = next(
                        record
                        for record in history
                        if record["step"] == meta["budgets"]["exposure"]
                    )
                    passed = all(
                        row["metrics"][key]["accuracy"] >= value
                        for key, value in config["load_gate"].items()
                    )
                    gate.append({"name": spec["name"], "passed": passed, "metrics": row["metrics"]})
                    anchor_records.append((spec, row["metrics"]))
                allowed = anchors_pass(
                    anchor_records, config["confirmation_worlds"], config["load_gate"]
                )
                write_json(
                    root / "load-gate.json", {"passed": allowed, "anchors": gate, "utc": utc()}
                )
                if allowed:
                    stage(config, load_specs(config, selected), "load", state)
                    report(config, root)
                state.update(
                    state="complete",
                    load_curve_started=allowed,
                    conclusion="registered_curves_complete"
                    if allowed
                    else "confirmation_anchor_not_met",
                    finished_utc=utc(),
                )
                write_json(root / "controller-state.json", state)
            tracker.wait(timeout=1800)
            if tracker.returncode != 0 or not (root / "tracking-completion.json").exists():
                raise RuntimeError("Science finished but W&B synchronization is incomplete")
            report(config, root)
        except BaseException:
            state.update(state="controller_error", error=traceback.format_exc(), updated_utc=utc())
            write_json(root / "controller-state.json", state)
            raise
        finally:
            tracking_log.close()


def launch(path):
    config = read(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "controller-process.json").exists():
        previous = read(root / "controller-process.json")
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Controller process already exists")
    script = Path(config["source_root"]) / "scripts/execute_composition_curves.py"
    command = [sys.executable, "-u", str(script), "controller", "--config", str(path)]
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment(config["source_root"]),
        )
    write_json(root / "controller-process.json", {"pid": process.pid, "command": command})
    print(json.dumps({"pid": process.pid, "root": str(root)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "launch", "controller", "worker"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run")
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.config.resolve(), args.run)
    else:
        globals()[args.action](args.config.resolve())


if __name__ == "__main__":
    main()
