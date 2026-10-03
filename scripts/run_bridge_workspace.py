"""Freeze and concurrently execute causal bridge interventions on existing weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from llm_memory_editability.bridge_workspace import audit_run, run
from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import file_hash

ROOT = Path("results/bridge-workspace-v1")
ARTIFACTS = Path("docs/development-artifacts/bridge-workspace-v1")


def specifications(phase):
    base = Path("results/representation-alignment-v1/runs")
    if phase in {"development", "direction-calibration"}:
        return [
            {
                "name": p.name,
                "phase": phase,
                "family": "alignment",
                "checkpoint": str(p / "model.pt"),
            }
            for p in sorted(base.glob("development-v2-w*"))
        ]
    reproduction = json.loads(Path("configs/grokking-reproduction-development-v1.json").read_text())
    if phase == "engineering":
        return [
            {
                "name": name + "-engineering",
                "phase": phase,
                "family": "reproduction",
                "checkpoint": str(
                    Path("results/grokking-reproduction-v1/development")
                    / name
                    / "checkpoint-0700000.pt"
                ),
                "data": "results/grokking-reproduction-v1/data",
                "checkpoint_step": 700000,
            }
            for name in ("standard8-phi7.2-wd0.1", "loop4x2-phi7.2-wd0.1")
        ]
    specs = [
        {
            "name": p.name,
            "phase": "existing-world-validation",
            "family": "alignment",
            "checkpoint": str(p / "model.pt"),
        }
        for p in sorted(base.glob("confirmation-w*"))
    ]
    for step in (700000, 1000000, 1500000):
        for original in reproduction["runs"]:
            name = original["name"]
            specs.append(
                {
                    "name": f"{name}-s{step}",
                    "phase": "paired-architecture-exploration",
                    "family": "reproduction",
                    "checkpoint": str(
                        Path("results/grokking-reproduction-v1/development")
                        / name
                        / f"checkpoint-{step:07d}.pt"
                    ),
                    "data": "results/grokking-reproduction-v1/data",
                    "checkpoint_step": step,
                }
            )
    return specs


def freeze(phase):
    config_path = Path(f"configs/bridge-workspace-{phase}-v1.json")
    if config_path.exists():
        raise FileExistsError(config_path)
    directory = ARTIFACTS / phase
    sources = sorted(str(p) for p in Path("src/llm_memory_editability").glob("*.py")) + [
        "scripts/run_bridge_workspace.py",
        "scripts/report_bridge_workspace.py",
        "tests/test_bridge_workspace.py",
        "configs/experiment-tracking-defaults.json",
        str(ARTIFACTS / "design.md"),
    ]
    specs = specifications(phase)
    assert len(specs) == (2 if phase == "engineering" else 42 if phase == "comparison" else 4)
    hashes = {}
    for spec in specs:
        checkpoint = Path(spec["checkpoint"])
        if checkpoint.exists() and time.time() - checkpoint.stat().st_mtime > 20:
            hashes[str(checkpoint)] = file_hash(checkpoint)
        world = Path(spec.get("data", checkpoint.parent)) / "world.npz"
        hashes[str(world)] = file_hash(world)
        if spec["family"] == "reproduction":
            for name in ("complete.json", "panels.npz"):
                path = Path(spec["data"]) / name
                hashes[str(path)] = file_hash(path)
    config = {
        "created_utc": utc(),
        "phase": phase,
        "new_training_updates": 0,
        "settings": {
            "selection_seed": 801101,
            "calibration_seed": 802101,
            "rotation_seed": 803101,
            "axis_families": ["prefix_jacobian", "input_embedding"],
            "prefixes_per_family": 8
            if phase == "engineering"
            else 64
            if phase == "comparison"
            else 32,
            "native_per_stratum": 16
            if phase == "engineering"
            else 256
            if phase == "comparison"
            else 128,
            "calibration_prompts": 4 if phase == "engineering" else 32,
            "jacobian_chunk": 32,
            "batch_size": 64,
        },
        "gpus": list(range(8)),
        "max_new_processes_per_gpu": 2 if phase == "comparison" else 1,
        "minimum_free_memory_mib": 12000,
        "specs": specs,
        "input_hashes": hashes,
        "source": {p: file_hash(p) for p in sources},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "transformers", "pytest")
        },
        "units": (
            "queries, unique first facts, original worlds, paired initializations; "
            "layers and relations are repeated measurements"
        ),
        "interpretation": (
            "development calibration / frozen post-hoc existing-world validation / "
            "one-world architecture exploration"
        ),
    }
    write_json(config_path, config)
    for source in sources:
        destination = directory / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(config_path, directory / "frozen-config.json")
    write_json(
        directory / "freeze.json",
        {
            "config": str(config_path),
            "config_sha256": file_hash(config_path),
            "source": config["source"],
            "created_utc": utc(),
        },
    )
    print(json.dumps({"config": str(config_path), "runs": len(specs), "new_training_updates": 0}))


def gpu_resources():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    return [
        dict(
            zip(
                ("gpu", "memory_free_mib", "memory_used_mib", "utilization_percent"),
                map(int, row.split(",")),
                strict=True,
            )
        )
        for row in subprocess.check_output(command, text=True).strip().splitlines()
    ]


def process_alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


class AdoptedProcess:
    """Observe an unchanged worker owned by the previous controller."""

    def __init__(self, pid, out):
        self.pid, self.out = pid, Path(out)

    def poll(self):
        if process_alive(self.pid):
            return None
        return 0 if (self.out / "interventions-complete.json").exists() else 1


def amend(config_path, label="audit-loader-amendment-v1"):
    """Version a loader/controller correction while retaining matrix and raw results."""
    import signal

    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    root = ROOT / config["phase"]
    previous = json.loads((root / "controller-state.json").read_text())
    process = json.loads((root / "process.json").read_text())
    directory = ARTIFACTS / config["phase"] / label
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    write_json(directory / "previous-controller-state.json", previous)
    original_source = dict(config["source"])
    for source in original_source:
        destination = directory / "source" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config["source"] = {p: file_hash(p) for p in original_source}
    config["source_root"] = str((directory / "source").resolve())
    config["resume_active"] = previous["active"]
    config["resume_auditing"] = previous.get("auditing", [])
    config["amendment"] = {
        "created_utc": utc(),
        "original_config_sha256": file_hash(config_path),
        "reason": (
            "Support alignment checkpoint spec.steps during reload; resume saved forwards "
            "and adopt active workers. Matrix, data, directions, deltas and scoring unchanged."
        ),
        "original_source": original_source,
    }
    if label.startswith("numerical-controls"):
        config["amendment"]["reason"] = (
            "Keep exact reload prediction checks; record inverse/restore numerical "
            "outcomes separately without changing interventions, data or scores. "
            "Preserve failures and avoid mechanism claims below the control variation."
        )
    amended = config_path.with_name(config_path.stem + "-" + label + ".json")
    write_json(amended, config)
    shutil.copy2(amended, directory / "frozen-config.json")
    # Stop only this batch's coordinator and recorder. Workers have their own
    # sessions and continue; the original eight trainers are not addressed.
    for pid in (process["pid"], previous["tracker_pid"]):
        if process_alive(pid):
            os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        if not process_alive(process["pid"]):
            break
        time.sleep(0.1)
    launch(amended)


def launch(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["phase"]
    root.mkdir(parents=True, exist_ok=True)
    state = root / "process.json"
    if state.exists():
        pid = json.loads(state.read_text())["pid"]
        try:
            if process_alive(pid):
                raise RuntimeError(f"Controller already active: {pid}")
        except ProcessLookupError:
            pass
    frozen = Path(config.get("source_root", ARTIFACTS / config["phase"] / "source")).resolve()
    for source, expected in config["source"].items():
        assert file_hash(frozen / source) == expected, f"Frozen source changed: {source}"
    env = {
        **os.environ,
        "PYTHONPATH": str(frozen / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = [
        sys.executable,
        "-u",
        str(frozen / "scripts/run_bridge_workspace.py"),
        "controller",
        "--config",
        str(config_path),
    ]
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    write_json(state, {"pid": process.pid, "command": command, "started_utc": utc()})
    print(
        json.dumps(
            {"controller_pid": process.pid, "root": str(root), "log": str(root / "controller.log")}
        )
    )


def controller(config_path):
    config = json.loads(Path(config_path).read_text())
    root = ROOT / config["phase"]
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    tracker_command = [
        str(Path(".venv-wandb/bin/python").absolute()),
        "-u",
        "-m",
        "llm_memory_editability.experiment_tracking",
        "--root",
        str(ROOT if config["phase"] == "comparison" else root),
        "--runs-dir",
        str(runs),
    ]
    with (root / "tracking.log").open("a") as log:
        tracker = subprocess.Popen(
            tracker_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    write_json(root / "tracking-process.json", {"pid": tracker.pid})
    pending, active, failed = list(config["specs"]), {}, []
    auditing, reported_count, report_process = set(), 0, None
    completed = [s["name"] for s in pending if (runs / s["name"] / "complete.json").exists()]
    pending = [s for s in pending if s["name"] not in completed]
    for name, info in config.get("resume_active", {}).items():
        if name in completed:
            continue
        active[name] = (AdoptedProcess(info["pid"], runs / name), info["gpu"])
        pending = [s for s in pending if s["name"] != name]
        if name in config.get("resume_auditing", []):
            auditing.add(name)
    while pending or active:
        for name, (process, gpu) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            if (
                code == 0
                and name not in auditing
                and (runs / name / "interventions-complete.json").exists()
                and not (runs / name / "complete.json").exists()
            ):
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "audit",
                    "--run-dir",
                    str(runs / name),
                    "--gpu",
                    str(gpu),
                ]
                with (runs / name / "reload-audit.log").open("a") as log:
                    auditor = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                active[name] = (auditor, gpu)
                auditing.add(name)
                continue
            if code == 0 and (runs / name / "complete.json").exists():
                completed.append(name)
            else:
                failed.append({"name": name, "exit_code": code, "gpu": gpu})
                write_json(
                    runs / name / "failure.json",
                    {
                        "exit_code": code,
                        "stage": "reload_audit" if name in auditing else "worker",
                        "finished_utc": utc(),
                    },
                )
            del active[name]
        resources = gpu_resources()
        for resource in resources:
            gpu = resource["gpu"]
            capacity = config["max_new_processes_per_gpu"] - sum(
                g == gpu for _, g in active.values()
            )
            if (
                gpu not in config["gpus"]
                or resource["memory_free_mib"] < config["minimum_free_memory_mib"]
            ):
                continue
            for _ in range(capacity):
                ready = [
                    s
                    for s in pending
                    if Path(s["checkpoint"]).exists()
                    and time.time() - Path(s["checkpoint"]).stat().st_mtime > 20
                ]
                if not ready:
                    break
                # Launch available architecture jobs ahead of the tiny validation jobs;
                # each card can host both, alongside its existing trainer.
                has_large = any(g == gpu and "phi" in name for name, (_, g) in active.items())
                priority = "alignment" if has_large else "reproduction"
                spec = next((s for s in ready if s["family"] == priority), ready[0])
                out = runs / spec["name"]
                out.mkdir(parents=True, exist_ok=True)
                for path, expected in config["input_hashes"].items():
                    if path == spec["checkpoint"]:
                        assert file_hash(path) == expected, f"Checkpoint changed: {path}"
                write_json(
                    out / "input-pin.json",
                    {"checkpoint_sha256": file_hash(spec["checkpoint"]), "pinned_utc": utc()},
                )
                audit_only = (out / "interventions-complete.json").exists()
                command = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "audit" if audit_only else "worker",
                    *(
                        ["--run-dir", str(out)]
                        if audit_only
                        else [
                            "--config",
                            str(Path(config_path).resolve()),
                            "--name",
                            spec["name"],
                        ]
                    ),
                    "--gpu",
                    str(gpu),
                ]
                with (out / "worker.log").open("a") as log:
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                active[spec["name"]] = (process, gpu)
                if audit_only:
                    auditing.add(spec["name"])
                pending.remove(spec)
        if (
            not active
            and len(completed) > reported_count
            and (report_process is None or report_process.poll() is not None)
        ):
            report_out = ARTIFACTS / config["phase"] / "report"
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("report_bridge_workspace.py")),
                "--root",
                str(root),
                "--out",
                str(report_out),
            ]
            with (root / "report.log").open("a") as log:
                report_process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            reported_count = len(completed)
        state = {
            "state": "running",
            "updated_utc": utc(),
            "completed": completed,
            "failed": failed,
            "active": {name: {"pid": p.pid, "gpu": g} for name, (p, g) in active.items()},
            "queued": [
                {"name": s["name"], "waiting_for_checkpoint": not Path(s["checkpoint"]).exists()}
                for s in pending
            ],
            "resources": resources,
            "tracker_pid": tracker.pid,
            "auditing": sorted(auditing & set(active)),
            "report_pid": report_process.pid if report_process is not None else None,
        }
        write_json(root / "controller-state.json", state)
        if config["phase"] == "comparison":
            write_json(ROOT / "controller-state.json", state)
        if tracker.poll() is not None and (pending or active):
            write_json(
                root / "tracking-failure.json",
                {"exit_code": tracker.returncode, "updated_utc": utc()},
            )
        time.sleep(5)
    state.update(
        state="finished_with_failures" if failed else "complete",
        updated_utc=utc(),
        active={},
        queued=[],
    )
    write_json(root / "controller-state.json", state)
    if config["phase"] == "comparison":
        write_json(ROOT / "controller-state.json", state)


def worker(config_path, name, gpu):
    config = json.loads(Path(config_path).read_text())
    spec = next(s for s in config["specs"] if s["name"] == name)
    out = ROOT / config["phase"] / "runs" / name
    try:
        world = Path(spec.get("data", Path(spec["checkpoint"]).parent)) / "world.npz"
        assert file_hash(world) == config["input_hashes"][str(world)], "Frozen world changed"
        if spec["family"] == "reproduction":
            for name in ("complete.json", "panels.npz"):
                path = Path(spec["data"]) / name
                assert file_hash(path) == config["input_hashes"][str(path)], "Frozen data changed"
        pin = json.loads((out / "input-pin.json").read_text())
        assert file_hash(spec["checkpoint"]) == pin["checkpoint_sha256"], (
            "Pinned checkpoint changed"
        )
        run(spec, config["settings"], out, gpu)
    except Exception:
        write_json(
            out / "failure.json", {"finished_utc": utc(), "traceback": traceback.format_exc()}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["freeze", "amend", "launch", "controller", "worker", "audit", "status"]
    )
    parser.add_argument(
        "--phase",
        choices=["development", "direction-calibration", "engineering", "comparison"],
        default="development",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--amend-label", default="audit-loader-amendment-v1")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.phase)
    elif args.action == "amend":
        amend(args.config, args.amend_label)
    elif args.action == "launch":
        launch(args.config)
    elif args.action == "controller":
        controller(args.config)
    elif args.action == "worker":
        worker(args.config, args.name, args.gpu)
    elif args.action == "audit":
        audit_run(args.run_dir, args.gpu)
    else:
        print((ROOT / args.phase / "controller-state.json").read_text())


if __name__ == "__main__":
    main()
