"""Queue a bounded capacity-development pipeline behind both existing LM batches."""

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

import execute_parametric_architecture as scheduler

from llm_memory_editability.capacity_scaling import grid_capacity, prerequisite_pass

read, write, now, digest = scheduler.read, scheduler.write, scheduler.now, scheduler.digest


def freeze(path):
    config = read(path)
    repo = Path.cwd().resolve()
    artifact = repo / "docs/development-artifacts" / config["batch"]
    source = artifact / "source"
    if source.exists():
        raise FileExistsError(source)
    assert config["runtime"]["docker_context"] == "lm-memory"
    assert config["gpus"] == [2, 3]
    config.update(
        repository=str(repo),
        source_root=str(source),
        source_files={},
        frozen_utc=now(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    files = sorted(
        set(
            list(Path("src/llm_memory_editability").glob("*.py"))
            + [
                Path("scripts/execute_parametric_architecture.py"),
                Path("configs/experiment-tracking-defaults.json"),
                path.relative_to(repo),
            ]
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
    write(artifact / "execution-lock.json", {"config_sha256": digest(frozen), "utc": now()})
    print(json.dumps({"frozen_config": str(frozen), "sources": len(files)}))


def predecessor_status(paths):
    """Missing/failed/partially finished predecessors never release the barrier."""
    records = []
    for root in map(Path, paths):
        file = root / "controller-state.json"
        if not file.exists():
            records.append({"root": str(root), "ready": False, "reason": "missing state"})
            continue
        state = read(file)
        expected = read(root / "frozen-config.json").get("runs", [])
        expected_names = {s["name"] for s in expected}
        complete = set(state.get("completed", []))
        ready = bool(expected_names) and state.get("state") == "complete"
        ready = ready and expected_names <= complete
        ready = ready and not any(state.get(k) for k in ("active", "queued", "failed"))
        if ready:
            ready = all(
                (root / "runs" / n / "audit.json").exists()
                and read(root / "runs" / n / "audit.json").get("passed") is True
                for n in expected_names
            )
        records.append(
            {
                "root": str(root),
                "ready": ready,
                "state": state.get("state"),
                "completed": len(complete),
                "expected": len(expected_names),
            }
        )
    return all(r["ready"] for r in records), records


def devices_free(gpus):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    observed = {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}
    if not all(g in observed and observed[g] < 256 for g in gpus):
        return False
    # A newly started but not-yet-allocated container also owns its GPU.
    for context in ("default", "lm-memory", "d157"):
        result = subprocess.run(
            ["docker", "--context", context, "ps", "-q"], capture_output=True, text=True
        )
        if result.returncode:
            # An unreachable unrelated daemon is not evidence that a GPU is free.
            return False
        if result.stdout.strip():
            info = json.loads(
                subprocess.check_output(
                    ["docker", "--context", context, "inspect", *result.stdout.split()], text=True
                )
            )
            for container in info:
                for req in container["HostConfig"].get("DeviceRequests") or []:
                    ids = req.get("DeviceIDs") or []
                    if req.get("Count") == -1 or any(str(g) in ids for g in gpus):
                        return False
    return True


def command_builder(config, config_path, spec, out, gpu, attempt):
    name, command = scheduler.container_command(config, config_path, spec, out, gpu, attempt)
    old = str(Path(config["source_root"]) / "scripts/execute_parametric_architecture.py")
    new = str(Path(config["source_root"]) / "scripts/execute_capacity_scaling.py")
    command[command.index(old)] = new
    # Docker names use a short stable digest, retaining each failed attempt.
    name = (
        "lm-capacity-"
        + hashlib.sha256(f"{config['batch']}:{spec['name']}:{attempt}".encode()).hexdigest()[:20]
    )
    command[command.index("--name") + 1] = name
    return name, command


def worker(config_path, name):
    config = read(config_path)
    script = Path(config["source_root"]) / "scripts/run_capacity_scaling.py"
    base = ["--config", str(config_path), "--run", name]
    spec = next(s for s in config["runs"] if s["name"] == name)
    subprocess.run(
        [sys.executable, "-u", str(script), "train", *base],
        check=True,
        timeout=spec["max_wall_seconds"],
    )
    subprocess.run([sys.executable, "-u", str(script), "audit", *base], check=True, timeout=1200)


def make_spec(config, name, **changes):
    return {**config["base_spec"], "name": name, **changes}


def run_stage(config, stage, specs):
    root = Path(config["results_root"]) / stage
    path = Path(config["source_root"]).parent / (stage + "-config.json")
    execution_lock = path.with_name(stage + "-execution-lock.json")
    stage_config = {
        **config,
        "batch": config["batch"],
        "results_root": str(root),
        "runs": specs,
        "phase": "development",
        "execution_lock": str(execution_lock),
    }
    if path.exists():
        assert read(path) == stage_config, "Do not mutate a previously materialized stage"
    else:
        write(path, stage_config)
    if execution_lock.exists():
        assert read(execution_lock)["config_sha256"] == digest(path)
    else:
        write(execution_lock, {"config_sha256": digest(path), "utc": now()})
    root.mkdir(parents=True, exist_ok=True)
    if (root / "controller-state.json").exists():
        state = read(root / "controller-state.json")
        if state.get("state") == "complete":
            assert {s["name"] for s in specs} == set(state["completed"])
            for spec in specs:
                assert read(root / "runs" / spec["name"] / "audit.json")["passed"]
            assert read(root / "tracking-completion.json")["passed"]
            return {s["name"]: root / "runs" / s["name"] for s in specs}
        if state.get("state") in {"finished_with_failures", "controller_error"}:
            raise RuntimeError(f"Review failed stage before resuming: {root}")
    while not devices_free(config["gpus"]):
        time.sleep(20)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scheduler.control_locked(stage_config, path, root, command_builder=command_builder)
    assert read(root / "controller-state.json")["state"] == "complete", stage
    deadline = time.monotonic() + 300
    while not (root / "tracking-completion.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"W&B did not drain for {stage}")
        time.sleep(3)
    assert read(root / "tracking-completion.json")["passed"]
    return {s["name"]: root / "runs" / s["name"] for s in specs}


def report(config, paths, selected, recipe):
    rows = []
    for path in paths:
        meta, audit = read(path / "run.json"), read(path / "audit.json")
        spec, m = meta["spec"], audit["metrics"]
        rows.append(
            {
                "path": str(path),
                "name": spec["name"],
                "heads_n": spec["heads_n"],
                "fixed_target": spec.get("support_heads_n") == config["fixed_target_support_heads"],
                "bits": m["information"]["world_bits"],
                "atomic": m["atomic"]["accuracy"],
                "II": m["II"]["accuracy"],
                "OO": m["OO"]["accuracy"],
                "metrics": m,
            }
        )
    curve = [
        r
        for r in rows
        if not r["fixed_target"]
        and (r["heads_n"] != config["load_heads"][0] or r["name"] == selected.name)
    ]
    thresholds = config["capacity_thresholds"]
    summary = {
        "phase": "development",
        "independent_worlds": 1,
        "initializations": 1,
        "recipe": recipe,
        "runs": rows,
        "C1": grid_capacity(curve, "atomic", thresholds["atomic"], thresholds["atomic"]),
        "C2_II": grid_capacity(curve, "II", thresholds["II"], thresholds["atomic"]),
        "sensitivity": {},
        "fixed_target_paired": [],
        "limitations": [
            "No scaling exponent or population confidence interval from one world",
            "Background adds both knowledge and optimizer updates",
            "II/OO conclusions are separate; composition is not new entropy",
        ],
    }
    for a in config["sensitivity_thresholds"]["atomic"]:
        for b in config["sensitivity_thresholds"]["II"]:
            summary["sensitivity"][f"atomic-{a}-II-{b}"] = grid_capacity(curve, "II", b, a)
    base = np_load(selected / f"predictions-e{recipe['epochs']:05d}.npz")
    for row in rows:
        if not row["fixed_target"]:
            continue
        other = np_load(Path(row["path"]) / f"predictions-e{recipe['epochs']:05d}.npz")
        import numpy as np

        from llm_memory_editability.capacity_scaling import pack

        world = np_load(Path(row["path"]) / "world.npz")
        base_world = np_load(selected / "world.npz")
        for pool in ("II", "IO", "OI", "OO"):
            assert np.array_equal(world[pool][:, :5], base_world[pool][:, :5])
            shared = base[pool + "_both_atomic"] & other[pool + "_both_atomic"]
            labels = pack(world[pool])[2]

            def accuracy(pred, shared=shared, labels=labels):
                return (
                    float(np.all(pred[shared] == labels[shared], axis=1).mean())
                    if shared.any()
                    else None
                )

            summary["fixed_target_paired"].append(
                {
                    "run": row["name"],
                    "pool": pool,
                    "common_n": int(shared.sum()),
                    "all_n": len(shared),
                    "baseline_accuracy": accuracy(base[pool + "_predictions"]),
                    "loaded_accuracy": accuracy(other[pool + "_predictions"]),
                }
            )
    write(Path(config["results_root"]) / "summary.json", summary)
    return summary


def np_load(path):
    import numpy as np

    with np.load(path) as data:
        return dict(data)


def controller(config_path):
    config = read(config_path)
    assert digest(config_path) == read(config_path.parent / "execution-lock.json")["config_sha256"]
    for filename, expected in config["source_files"].items():
        assert digest(filename) == expected, filename
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                ready, records = predecessor_status(config["predecessors"])
                write(
                    root / "controller-state.json",
                    {
                        "state": "waiting_for_predecessors",
                        "predecessors": records,
                        "gpus": config["gpus"],
                        "updated_utc": now(),
                    },
                )
                if ready and devices_free(config["gpus"]):
                    break
                time.sleep(20)
            write(root / "controller-state.json", {"state": "calibrating", "updated_utc": now()})
            calibration = [
                make_spec(config, f"calibration-lr{lr:g}", learning_rate=lr)
                for lr in config["calibration_learning_rates"]
            ]
            paths = run_stage(config, "calibration", calibration)
            candidates = []
            for name, path in paths.items():
                m = read(path / "audit.json")["metrics"]
                s = read(path / "run.json")["spec"]
                candidates.append(
                    {
                        "name": name,
                        "path": str(path),
                        "learning_rate": s["learning_rate"],
                        "score": (m["atomic"]["answer_nll"] + m["train_composition"]["answer_nll"])
                        / 2,
                    }
                )
            choice = min(candidates, key=lambda c: (c["score"], c["learning_rate"]))
            selected = Path(choice["path"])
            recipe = {
                "learning_rate": choice["learning_rate"],
                "epochs": config["base_spec"]["epochs"],
            }
            write(
                root / "development-selection.json",
                {"rule": config["selection_rule"], "candidates": candidates, "choice": choice},
            )
            if not prerequisite_pass(
                read(selected / "audit.json")["metrics"], config["prerequisite_thresholds"]
            ):
                recipe["epochs"] = config["long_calibration_epochs"]
                longer = make_spec(config, "calibration-long", **recipe)
                extra = run_stage(config, "calibration-long", [longer])
                paths.update(extra)
                selected = extra[longer["name"]]
            passed = prerequisite_pass(
                read(selected / "audit.json")["metrics"], config["prerequisite_thresholds"]
            )
            write(
                root / "calibration-decision.json",
                {
                    "passed": passed,
                    "selected": str(selected),
                    "recipe": recipe,
                    "thresholds": config["prerequisite_thresholds"],
                },
            )
            if not passed:
                write(
                    root / "controller-state.json",
                    {
                        "state": "development_prerequisite_not_met",
                        "updated_utc": now(),
                        "reason": "Low-load operation not established; no capacity-gap attribution",
                        "completed": list(paths),
                        "training_runs": len(paths),
                    },
                )
                return
            write(root / "controller-state.json", {"state": "load_scan", "updated_utc": now()})
            scan = [
                make_spec(config, f"load-h{h}", heads_n=h, **recipe)
                for h in config["load_heads"][1:]
            ]
            paths.update(run_stage(config, "load-scan", scan))
            write(root / "controller-state.json", {"state": "fixed_target", "updated_utc": now()})
            control = [
                make_spec(
                    config,
                    f"background-h{h}",
                    heads_n=h,
                    support_heads_n=config["fixed_target_support_heads"],
                    **recipe,
                )
                for h in config["load_heads"][1:]
            ]
            paths.update(run_stage(config, "fixed-target", control))
            report(config, list(paths.values()), selected, recipe)
            write(
                root / "controller-state.json",
                {
                    "state": "complete",
                    "updated_utc": now(),
                    "completed": list(paths),
                    "training_runs": len(paths),
                    "next": "Review boundaries before width/architecture/confirmation",
                },
            )
        except BaseException:
            write(
                root / "controller-state.json",
                {
                    "state": "controller_error",
                    "error": traceback.format_exc(),
                    "updated_utc": now(),
                },
            )
            raise


def launch(path):
    config = read(path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    script = Path(config["source_root"]) / "scripts/execute_capacity_scaling.py"
    command = [sys.executable, "-u", str(script), "controller", "--config", str(path)]
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            command,
            env=scheduler.host_env(config["source_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(root / "controller-process.json", {"pid": process.pid, "command": command, "utc": now()})
    print(json.dumps({"pid": process.pid, "results": str(root)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "launch", "controller", "container-worker"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.action == "container-worker":
        worker(path, args.run)
    else:
        {"freeze": freeze, "launch": launch, "controller": controller}[args.action](path)
