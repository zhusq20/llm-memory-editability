"""Prepare, execute and independently audit a frozen recurrence-use follow-up."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.depth_step import EVALUATION_SPLITS
from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.multihop_scaling import build_world, construct, run_name
from llm_memory_editability.recurrence_use import TASKS, evaluate, select_donors
from llm_memory_editability.storage_composition import file_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/development-artifacts/recurrence-use-v1"
RESULTS = ROOT / "results/recurrence-use-v1"
FILES = (
    "src/llm_memory_editability/recurrence_use.py",
    "scripts/run_recurrence_use.py",
    "scripts/report_recurrence_use.py",
    "tests/test_recurrence_use.py",
    "docs/development-artifacts/recurrence-use-v1/design.md",
)


def config_path(phase):
    return ROOT / f"configs/recurrence-use-{phase}-v1.json"


def prepare(phase):
    destination = config_path(phase)
    if destination.exists():
        raise FileExistsError(destination)
    if phase == "followup":
        development = json.loads(config_path("development").read_text())
        for run in development["runs"]:
            folder = RESULTS / "development" / run["name"]
            if not (folder / "audit.json").exists():
                raise ValueError("Development reload audits must pass before follow-up freeze")
    original_phase = "confirmation" if phase == "followup" else "development"
    original = ROOT / f"configs/multihop-scaling-{original_phase}-v1.json"
    main = json.loads(original.read_text())
    specs = [s for s in main["specs"] if (s["width"], s["layers"], s["repeats"]) == (256, 1, 4)]
    if len(specs) != (12 if phase == "followup" else 2):
        raise ValueError("Unexpected registered matrix")
    source = {name: file_hash(ROOT / name) for name in FILES}
    source.update(main["source"])
    for name, digest in source.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Historical dependency changed: " + name)
    snapshot = ARTIFACT / (phase + "-source")
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in source:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    runs, worlds, selections = [], {}, {}
    for spec in specs:
        world = build_world(spec)
        worlds[spec["world_seed"], spec["phi"]] = world
    for seed in sorted({s["world_seed"] for s in specs}):
        selection = select_donors(worlds[seed, 1.0], worlds[seed, 4.0])
        path = ARTIFACT / "selections" / f"world-{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **selection)
        selections[seed] = str(path.relative_to(ROOT))
    for spec in specs:
        name = run_name(spec)
        parent = ROOT / "results/multihop-scaling-v1" / original_phase / name
        recorded = json.loads((parent / "spec.json").read_text())
        if recorded["spec"] != spec or recorded["source"] != main["source"]:
            raise ValueError("Historical specification/source mismatch")
        files = (
            "model-064000.pt",
            "world.npz",
            "world-metadata.json",
            "spec.json",
            "predictions-064000.npz",
            "exposures-064000.npz",
            "audit.json",
        )
        hashes = {name: file_hash(parent / name) for name in files}
        with np.load(parent / "exposures-064000.npz") as actual:
            for task in ("train_2", "train_3", "train_4"):
                if not (actual[task] > 0).all():
                    raise ValueError("Training-table roles lack actual sampled exposure")
        selection = selections[spec["world_seed"]]
        runs.append(
            {
                "name": name,
                "parent": str(parent.relative_to(ROOT)),
                "spec": spec,
                "input_sha256": hashes,
                "selection": selection,
                "selection_sha256": file_hash(ROOT / selection),
            }
        )
    config = {
        "created_utc": utc(),
        "phase": phase,
        "training_updates": 0,
        "status": "Frozen follow-up on previously observed worlds/checkpoints",
        "test_repeats": [2, 4, 6],
        "original_config": str(original.relative_to(ROOT)),
        "original_config_sha256": file_hash(original),
        "source": source,
        "runs": runs,
        "precision": "float32, TF32 enabled, no mixed precision",
        "planned_runs": len(runs),
        "planned_model_budgets": len(runs) * 3,
    }
    write_json(destination, config)
    shutil.copy2(destination, snapshot / destination.name)
    print(
        json.dumps(
            {
                "config": str(destination),
                "runs": len(runs),
                "same_bridge_common_queries": {
                    str(s): int(np.load(ROOT / p)["common"].sum()) for s, p in selections.items()
                },
            }
        ),
        flush=True,
    )


def validate_sources(config):
    for name, digest in config["source"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("Frozen source changed: " + name)


def load_run(config, entry, device):
    validate_sources(config)
    parent = ROOT / entry["parent"]
    for name, digest in entry["input_sha256"].items():
        if file_hash(parent / name) != digest:
            raise ValueError("Historical input changed: " + name)
    if file_hash(ROOT / entry["selection"]) != entry["selection_sha256"]:
        raise ValueError("Frozen graph selection changed")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world = build_world(entry["spec"])
    with np.load(parent / "world.npz") as archive:
        for task in EVALUATION_SPLITS:
            np.testing.assert_array_equal(archive[task], world[task])
    state = torch.load(parent / "model-064000.pt", map_location=device, weights_only=False)
    if state["spec"] != entry["spec"] or state["step"] != 64000:
        raise ValueError("Checkpoint identity differs")
    model = construct(entry["spec"], device).eval()
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    with np.load(ROOT / entry["selection"]) as archive:
        selection = {key: archive[key] for key in archive.files}
    return model, world, selection


def worker(phase, index, device, audit=False):
    path = config_path(phase)
    config = json.loads(path.read_text())
    entry = config["runs"][index]
    folder = RESULTS / phase / entry["name"]
    done = folder / ("audit.json" if audit else "complete.json")
    if done.exists():
        saved = json.loads(done.read_text())
        if saved["config_sha256"] != file_hash(path) or not saved["passed"]:
            raise ValueError("Existing output differs from this freeze")
        print(json.dumps({"run": entry["name"], "reused_verified": str(done)}), flush=True)
        return
    model, world, selection = load_run(config, entry, device)
    started = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    checks = []
    for repeats in config["test_repeats"]:
        arrays, summary = evaluate(model, world, selection, repeats)
        prediction_path = folder / f"r{repeats}.npz"
        summary_path = folder / f"r{repeats}.json"
        if repeats == 4:
            with np.load(ROOT / entry["parent"] / "predictions-064000.npz") as archive:
                for task in TASKS:
                    np.testing.assert_array_equal(
                        arrays[task + "_baseline_generated"], archive[task + "_generated"]
                    )
            summary["historical_R4_generation_exact"] = True
        if audit:
            with np.load(prediction_path) as saved:
                if set(saved.files) != set(arrays):
                    raise AssertionError("Audit output matrix differs")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, saved[key], err_msg=key)
            registered = json.loads(summary_path.read_text())
            for key in ("conditions", "engineering", "model_sha256", "weights_unchanged"):
                if summary[key] != registered[key]:
                    raise AssertionError("Audit summary differs: " + key)
        else:
            if prediction_path.exists() or summary_path.exists():
                raise FileExistsError("Partial output requires inspection: " + str(prediction_path))
            np.savez_compressed(prediction_path, **arrays)
            write_json(summary_path, summary)
        checks.append(
            {
                "R": repeats,
                "arrays": len(arrays),
                "conditions": len(summary["conditions"]),
                "prediction_sha256": file_hash(prediction_path),
                "summary_sha256": file_hash(summary_path),
            }
        )
        print(
            json.dumps(
                {
                    "run": entry["name"],
                    "R": repeats,
                    "audit": audit,
                    "seconds": summary["seconds"],
                    "arrays": len(arrays),
                }
            ),
            flush=True,
        )
    write_json(
        done,
        {
            "passed": True,
            "run": entry["name"],
            "config_sha256": file_hash(path),
            "checks": checks,
            "training_updates": 0,
            "independent_reload": audit,
            "finished_utc": utc(),
            "seconds": time.perf_counter() - started,
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
        },
    )


def execute(phase, devices, audit=False):
    config = json.loads(config_path(phase).read_text())
    validate_sources(config)
    log_folder = RESULTS / phase / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    def batch(slot, gpu):
        failures = []
        for index in range(slot, len(config["runs"]), len(devices)):
            name = config["runs"][index]["name"]
            log = log_folder / f"{name}-{'audit' if audit else 'run'}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--phase",
                phase,
                "--index",
                str(index),
                "--device",
                gpu,
            ]
            if audit:
                command += ["--audit"]
            with log.open("a") as output:
                completed = subprocess.run(
                    command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT
                )
            print(
                json.dumps(
                    {
                        "run": name,
                        "audit": audit,
                        "exit_code": completed.returncode,
                        "log": str(log.relative_to(ROOT)),
                    }
                ),
                flush=True,
            )
            if completed.returncode:
                failures.append(name)
        return failures

    failures = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(batch, i, gpu) for i, gpu in enumerate(devices)]
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        write_json(
            ARTIFACT / f"{phase}-{'audit' if audit else 'run'}-failures.json",
            {"failed_runs": failures, "recorded_utc": utc()},
        )
        raise RuntimeError("Failed runs: " + str(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit", "worker"))
    parser.add_argument("--phase", choices=("development", "followup"), required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--index", type=int)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.phase)
    elif args.mode == "worker":
        worker(args.phase, args.index, args.device, args.audit)
    else:
        execute(args.phase, args.devices, args.mode == "audit")


if __name__ == "__main__":
    main()
