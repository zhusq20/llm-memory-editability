"""Freeze and run the staged architecture comparison in four isolated GPU slots."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def host_env(source):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/lib64" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env.update(PYTHONPATH=str(Path(source) / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def freeze(input_path):
    config = read(input_path)
    repo = Path.cwd().resolve()
    artifact = repo / "docs/development-artifacts" / config["batch"]
    source = artifact / "source"
    if source.exists():
        raise FileExistsError(source)
    assert config["runtime"]["docker_context"] == "lm-memory"
    assert config["gpus"] == [2, 3, 4, 5]
    files = sorted(
        list(Path("src/llm_memory_editability").glob("*.py"))
        + [Path(__file__).relative_to(repo), Path("scripts/run_parametric_architecture.py")]
        + [Path("configs/experiment-tracking-defaults.json")]
        + [Path(p) for p in config.get("additional_sources", [])]
    )
    config.update(
        repository=str(repo),
        source_root=str(source),
        frozen_utc=now(),
        source_files={},
        execution_lock=str(artifact / "execution-lock.json"),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    for path in files:
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        config["source_files"][str(target)] = digest(target)
    config["tokenizer_sha256"] = {
        str(p.relative_to(config["tokenizer"])): digest(p)
        for p in sorted(Path(config["tokenizer"]).rglob("*"))
        if p.is_file()
    }
    assert digest(config["data_file"]) == config["data_sha256"]
    path = artifact / "frozen-config.json"
    write(path, config)
    write(
        config["execution_lock"],
        {
            "config_sha256": digest(path),
            "frozen_utc": now(),
            "environment": {"torch": "2.6.0+cu126", "transformers": "4.40.0"},
        },
    )
    print(json.dumps({"config": str(path), "runs": len(config["runs"])}))


def launch(config_path):
    config = read(config_path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    script = Path(config["source_root"]) / "scripts" / Path(__file__).name
    command = [sys.executable, "-u", str(script), "controller", "--config", str(config_path)]
    with (root / "controller.log").open("a") as log:
        proc = subprocess.Popen(
            command,
            env=host_env(config["source_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(root / "controller-process.json", {"pid": proc.pid, "command": command})
    print(json.dumps({"pid": proc.pid, "results": str(root)}))


def container_command(config, config_path, spec, out, gpu, attempt):
    runtime = config["runtime"]
    identity = f"{config['batch']}:{spec['name']}:{attempt}"
    name = "lm-architecture-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
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
        "/tmp:rw,size=4g",
        "--mount",
        f"type=bind,src={config['repository']},dst={config['repository']},readonly",
        "--mount",
        f"type=bind,src={out},dst={out}",
        "--workdir",
        config["repository"],
    ]
    env = {
        "PHYSICAL_GPU": str(gpu),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONPATH": str(Path(config["source_root"]) / "src"),
        "XDG_CACHE_HOME": "/tmp/cache",
        "TORCH_HOME": "/tmp/torch",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
        "TRITON_CACHE_DIR": "/tmp/triton",
    }
    for key, value in env.items():
        command += ["--env", f"{key}={value}"]
    command += [
        runtime["image"],
        runtime["python"],
        "-u",
        str(Path(config["source_root"]) / "scripts" / Path(__file__).name),
        "container-worker",
        "--config",
        str(config_path),
        "--run",
        spec["name"],
    ]
    return name, command


def container_worker(config_path, run_name):
    config = read(config_path)
    out = Path(config["results_root"]) / "runs" / run_name
    cli = Path(config["source_root"]) / "scripts/run_parametric_architecture.py"
    base = [
        "--config",
        str(config_path),
        "--run",
        run_name,
        "--out",
        str(out),
        "--device",
        "cuda:0",
    ]
    try:
        subprocess.run([sys.executable, "-u", str(cli), "train", *base], check=True)
        subprocess.run([sys.executable, "-u", str(cli), "audit", *base], check=True)
        assert read(out / "audit.json")["passed"] is True
    except BaseException:
        write(out / "failure.json", {"traceback": traceback.format_exc(), "utc": now()})
        raise


def controller(config_path):
    config = read(config_path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return control_locked(config, config_path, root)


def control_locked(config, config_path, root):
    assert digest(config_path) == read(config["execution_lock"])["config_sha256"]
    for path, expected in config["source_files"].items():
        assert digest(path) == expected, f"Frozen source changed: {path}"
    existing = subprocess.check_output(
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
    ).strip()
    if existing:
        raise RuntimeError(
            f"Existing containers still own this batch; do not duplicate: {existing}"
        )
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in devices.splitlines():
        index, memory = (int(v.strip()) for v in line.split(","))
        if index in config["gpus"] and memory > 1024:
            raise RuntimeError(f"GPU {index} is already occupied ({memory} MiB)")
    state = {
        "state": "running",
        "started_utc": now(),
        "gpus": config["gpus"],
        "completed": [],
        "failed": [],
        "active": {},
        "phase": config["phase"],
    }
    write(root / "controller-state.json", state)
    shutil.copyfile(config_path, root / "frozen-config.json")
    tracking = [
        str(Path(config["repository"]) / ".venv-wandb/bin/python"),
        "-u",
        "-m",
        "llm_memory_editability.interface_tracking",
        "--root",
        str(root),
        "--defaults",
        str(Path(config["source_root"]) / "configs/experiment-tracking-defaults.json"),
    ]
    tracking_log = (root / "tracking.log").open("a")
    tracker = subprocess.Popen(
        tracking, env=host_env(config["source_root"]), stdout=tracking_log, stderr=subprocess.STDOUT
    )
    write(root / "tracking-process.json", {"pid": tracker.pid, "command": tracking})
    queue, active = list(config["runs"]), {}
    try:
        deadline = time.monotonic() + 45
        while not (root / "tracking-ready.json").exists() and tracker.poll() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("W&B sidecar did not become ready")
            time.sleep(0.2)
        while queue or active:
            if tracker.poll() is not None:
                state["tracking_failure"] = f"W&B sidecar exited {tracker.returncode}"
            for gpu, item in list(active.items()):
                if item["process"].poll() is None:
                    continue
                item["log"].close()
                out = root / "runs" / item["name"]
                result = subprocess.run(
                    ["docker", "--context", "lm-memory", "inspect", item["container"]],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    write(out / "container-inspect.json", json.loads(result.stdout))
                passed = item["process"].returncode == 0 and (out / "audit.json").exists()
                passed = passed and read(out / "audit.json").get("passed") is True
                state["completed" if passed else "failed"].append(item["name"])
                del active[gpu]
            if state["failed"] or state.get("tracking_failure"):
                queue.clear()
            for gpu in config["gpus"]:
                if gpu in active or not queue:
                    continue
                spec = queue.pop(0)
                out = root / "runs" / spec["name"]
                if (out / "audit.json").exists() and read(out / "audit.json").get("passed"):
                    state["completed"].append(spec["name"])
                    continue
                if (out / "failure.json").exists():
                    raise RuntimeError(f"Review recorded failure before resuming {spec['name']}")
                out.mkdir(parents=True, exist_ok=True)
                attempt = len(list(out.glob("container-command-*.json"))) + 1
                name, command = container_command(config, config_path, spec, out, gpu, attempt)
                write(out / f"container-command-{attempt}.json", {"name": name, "command": command})
                log = (out / "worker.log").open("a")
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {"process": proc, "name": spec["name"], "container": name, "log": log}
                for _ in range(60):
                    result = subprocess.run(
                        ["docker", "--context", "lm-memory", "inspect", name],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        info = json.loads(result.stdout)[0]
                        if info["State"].get("Running"):
                            write(
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
                    if proc.poll() is not None:
                        break
                    time.sleep(0.2)
            state.update(
                active={
                    str(g): {
                        "name": x["name"],
                        "container": x["container"],
                        "pid": x["process"].pid,
                    }
                    for g, x in active.items()
                },
                queued=[s["name"] for s in queue],
                updated_utc=now(),
            )
            write(root / "controller-state.json", state)
            time.sleep(3)
        if not state["failed"] and not state.get("tracking_failure") and config["phase"] == "main":
            subprocess.run(
                [
                    sys.executable,
                    str(Path(config["source_root"]) / "scripts/report_parametric_architecture.py"),
                    "--config",
                    str(config_path),
                    "--out",
                    str(root / "report"),
                ],
                env=host_env(config["source_root"]),
                check=True,
            )
        state.update(
            state="finished_with_failures"
            if state["failed"] or state.get("tracking_failure")
            else "complete",
            finished_utc=now(),
        )
        write(root / "controller-state.json", state)
        if state["state"] == "complete" and config.get("transition"):
            next_path = select_main(config, root)
            launch(next_path)
    except BaseException:
        state.update(state="controller_error", error=traceback.format_exc(), updated_utc=now())
        write(root / "controller-state.json", state)
        raise
    finally:
        tracking_log.close()


def select_main(config, root):
    """Select only by the predeclared development fit score, retaining every candidate."""
    transition = config["transition"]
    choices = {}
    candidates = []
    expected = {(a, lr) for a in ["M8", "W8", "IHC8", "HC8"] for lr in [5e-5, 1e-4]}
    expected.add(("D8", 1e-4))
    observed = [(s["architecture"], s["learning_rate"]) for s in config["runs"]]
    assert len(observed) == len(expected) and set(observed) == expected
    for spec in config["runs"]:
        out = root / "runs" / spec["name"]
        assert read(out / "audit.json")["passed"]
        record = read(out / "learning.json")[-1]
        assert record["step"] == spec["steps"] == 32000
        metrics = record["metrics"]
        score = 0.5 * metrics["atomic"]["nll"] + 0.5 * metrics["train_composition"]["nll"]
        assert math.isfinite(score)
        candidates.append(
            {
                "architecture": spec["architecture"],
                "learning_rate": spec["learning_rate"],
                "score": score,
                "run": spec["name"],
            }
        )
    historical = transition["historical_development"]
    assert digest(historical["path"]) == historical["sha256"]
    historical_node = next(r for r in read(historical["path"]) if r["step"] == 32000)
    historical_metrics = historical_node["metrics"]
    historical_score = (
        historical_metrics["atomic"]["nll"] + historical_metrics["train_composition"]["nll"]
    ) / 2
    assert math.isfinite(historical_score)
    assert historical_score == historical["selection_score"]
    candidates.append(
        {
            "architecture": "D8",
            "learning_rate": 5e-5,
            "score": historical_score,
            "run": "D8_hist_development",
        }
    )
    for architecture in ["M8", "W8", "IHC8", "HC8", "D8"]:
        rows = [r for r in candidates if r["architecture"] == architecture]
        choices[architecture] = min(rows, key=lambda r: (r["score"], r["learning_rate"]))
    write(
        root / "development-selection.json",
        {"rule": transition["selection_rule"], "candidates": candidates, "choices": choices},
    )
    main = dict(config)
    main.pop("transition")
    main.update(
        batch=transition["batch"],
        phase="main",
        data_file=transition["data_file"],
        data_sha256=transition["data_sha256"],
        results_root=transition["results_root"],
        evaluation_nodes=[0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000, 300000],
        development_selection=str(root / "development-selection.json"),
        tracking={**config["tracking"], "group": transition["batch"]},
        runs=[],
    )
    main["reporting"] = {
        **transition.get("reporting", {}),
        "historical_reuse_verified": transition["historical_reuse_verified"],
    }
    arms = ["M8", "W8", "IHC8", "HC8"]
    if choices["D8"]["learning_rate"] != 5e-5 or not transition["historical_reuse_verified"]:
        arms.append("D8")
    for seed_index in range(3):
        for architecture in arms:
            selected = choices[architecture]
            spec = dict(
                next(
                    s
                    for s in config["runs"]
                    if s["name"] == selected["run"]
                    or (selected["run"] == "D8_hist_development" and s["architecture"] == "D8")
                )
            )
            spec.update(
                name=f"{architecture.lower()}-init{seed_index + 1}",
                steps=300000,
                initialization=811201 + seed_index,
                sampling_seed=812201 + seed_index,
                dropout_seed=813201 + seed_index,
                learning_rate=choices[architecture]["learning_rate"],
            )
            main["runs"].append(spec)
    artifact = Path(config["repository"]) / "docs/development-artifacts" / transition["batch"]
    main["execution_lock"] = str(artifact / "execution-lock.json")
    path = artifact / "frozen-config.json"
    if path.exists():
        raise FileExistsError("Main stage has already been frozen")
    assert digest(main["data_file"]) == main["data_sha256"]
    write(path, main)
    write(
        main["execution_lock"],
        {
            "config_sha256": digest(path),
            "frozen_utc": now(),
            "development_config_sha256": digest(root / "frozen-config.json"),
            "environment": {"torch": "2.6.0+cu126", "transformers": "4.40.0"},
        },
    )
    write(root / "next-stage.json", {"config": str(path), "runs": len(main["runs"])})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "launch", "controller", "container-worker"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run")
    args = parser.parse_args()
    if args.action == "container-worker":
        container_worker(args.config.resolve(), args.run)
    else:
        globals()[args.action](args.config.resolve())


if __name__ == "__main__":
    main()
