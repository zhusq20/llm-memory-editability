#!/usr/bin/env python3
"""Independently audit frozen small-transformer runs and reload their endpoints.

Historical metadata and copied source/configuration determine every check. The current
configuration and current training implementation are not used to reconstruct a run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("atomic", "train_composite", "test_composite", "ood_composite")
DATA_FIELDS = (
    "world_seed",
    "entities",
    "relations",
    "degree",
    "phi",
    "id_fraction",
    "id_test_fraction",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_modules(out, metadata):
    """Hash each copied input and import training/data modules from that exact snapshot."""
    snapshot = out / "source"
    copied = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    verified = {}
    for original, expected in metadata["files"].items():
        matches = [
            p for rel, p in copied.items() if original == rel or original.endswith("/" + rel)
        ]
        require(len(matches) == 1, f"missing or ambiguous source copy: {original}")
        path = matches[0]
        actual = digest(path)
        require(actual == expected, f"source snapshot hash mismatch: {path}")
        verified[path.relative_to(snapshot).as_posix()] = actual
    configurations = [snapshot / p for p in verified if p.startswith("configs/")]
    require(len(configurations) == 1, "expected one copied run configuration")
    frozen_config = read_json(configurations[0])
    expected_spec = {**frozen_config["base"], **frozen_config["runs"][out.name]}
    require(
        expected_spec == metadata["spec"], "metadata spec differs from its copied configuration"
    )
    package = "_grok_audit_" + hashlib.sha256(str(out).encode()).hexdigest()[:16]
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(snapshot / "src/llm_memory_editability")]
    sys.modules[package] = namespace
    training = importlib.import_module(package + ".grok_depth")
    data = importlib.import_module(package + ".grok_depth_data")
    return training, data, frozen_config, verified


def registered_phase_runs(frozen, phase, execute_wide=False):
    """Conditional controls become required only after the saved execution decision."""
    phase_runs = {
        name
        for name, item in frozen["runs"].items()
        if item.get("phase", frozen["base"].get("phase")) == phase
    }
    if phase != "confirmation" or "primary_runs" not in frozen:
        return sorted(phase_runs)
    primary = set(frozen["primary_runs"])
    conditional = set(frozen.get("conditional_wide_runs", []))
    require(not primary & conditional, "primary and conditional run lists overlap")
    require(primary | conditional == phase_runs, "registered matrix lists differ from phase runs")
    require(len(primary) == len(frozen["primary_runs"]), "duplicate primary run registration")
    require(
        len(conditional) == len(frozen.get("conditional_wide_runs", [])),
        "duplicate conditional run registration",
    )
    return sorted(primary | conditional if execute_wide else primary)


def scores_from_arrays(pred, true_rows):
    n = len(true_rows)
    if not n:
        require(not pred, "unexpected prediction arrays for an empty split")
        return {"n": 0, "answer_accuracy": None, "accuracy": None, "nll": None}
    require(set(pred) == {"answer", "stop", "nll", "target"}, "unexpected prediction columns")
    for key in ("answer", "stop", "target"):
        require(pred[key].shape == (n,), f"incorrect {key} shape")
    require(pred["nll"].shape == (n, 2), "NLL must include answer and EOS")
    require(np.array_equal(pred["target"], true_rows[:, -1]), "saved targets differ from world")
    require(np.isfinite(pred["nll"]).all(), "nonfinite saved NLL")
    answer_correct = pred["answer"] == true_rows[:, -1]
    return {
        "n": n,
        "answer_accuracy": float(answer_correct.mean()),
        "accuracy": float((answer_correct & (pred["stop"] == 1)).mean()),
        "nll": float(pred["nll"].mean()),
    }


def check_scores(actual, expected, name, atol, rtol):
    for key in ("n", "answer_accuracy", "accuracy", "nll"):
        if actual[key] is None or expected[key] is None:
            require(actual[key] is expected[key], f"{name}.{key}: inconsistent empty split")
        elif key == "nll":
            require(
                np.isclose(actual[key], expected[key], atol=atol, rtol=rtol),
                f"{name}.nll differs: {actual[key]} versus {expected[key]}",
            )
        else:
            require(actual[key] == expected[key], f"{name}.{key} differs")


def estimated_flops_per_step(spec):
    """Independent shape calculation: dense/attention matmuls, forward plus 2x backward."""
    batch, width, layers = spec["batch_size"], spec["width"], spec["layers"]
    sequence, output_positions = 4, 2
    vocab = 2 + spec["entities"] + spec["relations"]
    forward = layers * (24 * batch * sequence * width**2 + 4 * batch * sequence**2 * width)
    forward += 2 * batch * output_positions * width * vocab
    return 3 * forward


def audit_exposure(training, spec, world, rows, checkpoint, parameters):
    nodes = spec["nodes"]
    require(
        nodes == sorted(set(nodes)) and nodes[0] == 0 and nodes[-1] == spec["steps"],
        "invalid registered evaluation nodes",
    )
    require([r["step"] for r in rows] == nodes, "recorded nodes differ from historical spec")
    stream = training.EpochStream(
        len(world["atomic"]) + len(world["train_composite"]), spec["stream_seed"]
    )
    counts = {"atomic": 0, "composite": 0}
    prev = 0
    for row in rows:
        left = (row["step"] - prev) * spec["batch_size"]
        while left:
            n = min(left, 131072)
            ix = stream.take(n)
            atomic = int((ix < len(world["atomic"])).sum())
            counts["atomic"] += atomic
            counts["composite"] += n - atomic
            left -= n
        require(row["counts"] == counts, f"exposure mismatch at step {row['step']}")
        require(row["examples"] == row["step"] * spec["batch_size"], "example count mismatch")
        require(row["supervised_tokens"] == 2 * row["examples"], "supervision count mismatch")
        require(
            row["effective_input_tokens"] == 3 * counts["atomic"] + 4 * counts["composite"],
            "effective input token count mismatch",
        )
        require(row["parameters"] == parameters, "reported parameter count mismatch")
        require(
            row["estimated_training_flops"] == row["step"] * estimated_flops_per_step(spec),
            f"estimated FLOPs mismatch at step {row['step']}",
        )
        prev = row["step"]
    require(checkpoint["counts"] == counts, "checkpoint exposure differs")
    saved_stream = training.EpochStream(stream.size, 0)
    saved_stream.load_state_dict(checkpoint["stream"])
    require(
        np.array_equal(stream.take(4099), saved_stream.take(4099)),
        "checkpoint data stream cannot reproduce the registered sequence",
    )
    return {
        "all_nodes_checked": len(nodes),
        "examples": spec["steps"] * spec["batch_size"],
        "counts": counts,
        "checkpoint_stream_matches": True,
    }


def audit_run(out, device, atol, rtol, execute_wide=False):
    metadata = read_json(out / "metadata.json")
    spec = metadata["spec"]
    require(spec["phase"] == out.parent.name, "run phase differs from metadata")
    training, data, frozen, verified = source_modules(out, metadata)
    expected_runs = registered_phase_runs(frozen, spec["phase"], execute_wide)
    base = {
        "run_id": out.name,
        "spec": spec,
        "source_hashes": verified,
        "expected_phase_runs_from_frozen_config": expected_runs,
    }
    if not (out / "complete.json").exists():
        return {**base, "state": "incomplete", "status": read_json(out / "status.json")}
    complete = read_json(out / "complete.json")
    checkpoint = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    require(checkpoint["spec"] == complete["spec"] == spec, "checkpoint/complete specs differ")
    require(checkpoint["step"] == spec["steps"], "checkpoint has not reached registered budget")
    status = read_json(out / "status.json")
    require(
        status["state"] == "complete" and status["step"] == spec["steps"],
        "status does not mark completed budget",
    )
    with np.load(out / "world.npz", allow_pickle=False) as stored:
        world = {key: stored[key].copy() for key in stored.files}
    world["metadata"] = read_json(out / "world-metadata.json")
    world_audit = data.audit_world(world)
    require(world_audit == read_json(out / "data-audit.json"), "world audit changed")
    rebuilt = data.build_world(spec["world_seed"], **{key: spec[key] for key in DATA_FIELDS[1:]})
    require(rebuilt["metadata"] == world["metadata"], "world metadata regeneration differs")
    for key in world:
        if key != "metadata":
            require(np.array_equal(world[key], rebuilt[key]), f"regenerated world differs: {key}")
    width, layers = spec["width"], spec["layers"]
    vocab = 2 + spec["entities"] + spec["relations"]
    cfg = training.ModelConfig(
        vocab_size=vocab, width=width, layers=layers, heads=spec["heads"], context=8
    )
    model = training.SmallGPT(cfg, spec["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    require(
        parameters == vocab * width + 8 * width + layers * (12 * width**2 + 13 * width) + 2 * width,
        "parameter count differs from independent GPT-2 formula",
    )
    require(complete["parameters"] == parameters, "complete parameter count mismatch")
    rows = read_json(out / "learning.json")
    require(complete["endpoint"] == rows[-1], "complete endpoint differs from learning log")
    exposure = audit_exposure(training, spec, world, rows, checkpoint, parameters)
    for step in spec["nodes"]:
        require((out / f"predictions-{step:07d}.npz").exists(), f"missing predictions at {step}")
    for step in spec["weight_nodes"]:
        require((out / f"weights-{step:07d}.pt").exists(), f"missing weights at {step}")
    for state in checkpoint["optimizer"]["state"].values():
        require(int(state["step"].item()) == spec["steps"], "optimizer step mismatch")
    step = spec["steps"]
    with np.load(out / f"predictions-{step:07d}.npz", allow_pickle=False) as arrays:
        stored_predictions = {key: arrays[key].copy() for key in arrays.files}
    expected_columns = {
        split + "_" + key
        for split in SPLITS
        if len(world[split])
        for key in ("answer", "stop", "nll", "target")
    }
    require(set(stored_predictions) == expected_columns, "endpoint prediction columns differ")
    comparisons = {}
    for split in SPLITS:
        saved = (
            {
                key: stored_predictions[split + "_" + key]
                for key in ("answer", "stop", "nll", "target")
            }
            if len(world[split])
            else {}
        )
        independent = scores_from_arrays(saved, world[split])
        check_scores(independent, rows[-1][split], split + ".saved_scores", 1e-7, 1e-7)
        measured, fresh = training.evaluate_rows(model, world[split], device)
        check_scores(measured, independent, split + ".reloaded_scores", atol, rtol)
        delta = 0.0
        if saved:
            for key in ("answer", "stop", "target"):
                require(np.array_equal(saved[key], fresh[key]), f"reloaded {split}.{key} differs")
            delta = float(np.max(np.abs(saved["nll"] - fresh["nll"])))
            require(
                np.allclose(saved["nll"], fresh["nll"], atol=atol, rtol=rtol),
                f"reloaded {split}.nll exceeds tolerance; max abs delta {delta}",
            )
        comparisons[split] = {
            "scores": measured,
            "prediction_arrays_identical": True,
            "nll_max_abs_delta": delta,
            "eos_generated_after_model_answer": True,
        }
    two_calls = training.two_calls(model, world["test_composite"], device)
    check_scores(two_calls, rows[-1]["two_calls"], "two_calls", atol, rtol)
    return {
        **base,
        "state": "passed",
        "checkpoint_step": step,
        "parameters": parameters,
        "world_audit": world_audit,
        "exposure": exposure,
        "flops_audit": {
            "all_nodes_checked": len(rows),
            "estimated_per_step": estimated_flops_per_step(spec),
            "estimated_endpoint": spec["steps"] * estimated_flops_per_step(spec),
            "convention": (
                "dense/attention multiply-adds count as two; backward 2x forward; "
                "excludes elementwise operations"
            ),
        },
        "endpoint": comparisons,
        "two_calls": two_calls,
        "checkpoint_sha256": digest(out / "latest.pt"),
    }


def pairing_audit(runs):
    """Verify data/exposure pairing for runs differing only in model architecture."""
    ignored = {"layers", "width", "heads"}
    groups = {}
    for row in runs:
        if row.get("state") != "passed":
            continue
        key = json.dumps({k: v for k, v in row["spec"].items() if k not in ignored}, sort_keys=True)
        groups.setdefault(key, []).append(row)
    pairs = []
    for group in groups.values():
        for a, b in combinations(group, 2):
            same_data = a["world_audit"]["dataset_sha256"] == b["world_audit"]["dataset_sha256"]
            same_exposure = a["exposure"] == b["exposure"]
            require(same_data and same_exposure, "architecture pair data/exposure mismatch")
            pairs.append(
                {
                    "runs": [a["run_id"], b["run_id"]],
                    "dataset_identical": same_data,
                    "exposure_identical": same_exposure,
                    "world_seed": a["spec"]["world_seed"],
                    "initialization_seed": a["spec"]["initialization"],
                    "stream_seed": a["spec"]["stream_seed"],
                    "same_seed_does_not_imply_identical_cross_depth_weights": True,
                }
            )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # CPU and CUDA implement SDPA/GEMM differently. Discrete predictions must always
    # match exactly; NLL allows bounded numerical rounding, stated in the artifact.
    atol, rtol = (1e-3, 2e-3) if device.type == "cpu" else (1e-5, 1e-5)
    decision_path = ROOT / "docs/development-artifacts/grok-depth-v1/wide-control-decision.json"
    decision = read_json(decision_path) if decision_path.exists() else None
    if decision is not None:
        require(
            type(decision.get("execute")) is bool, "wide control decision requires boolean execute"
        )
    execute_wide = decision is not None and decision["execute"]
    root = ROOT / "results/grok-depth-v1" / args.phase
    paths = sorted(p.parent for p in root.glob("*/metadata.json"))
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "device": str(device),
        "torch": torch.__version__,
        "auditor_sha256": digest(Path(__file__)),
        "nll_tolerance": {"atol": atol, "rtol": rtol},
        "discrete_prediction_tolerance": "exact equality",
        "runs": [],
    }
    if args.phase == "confirmation":
        result["conditional_wide_control"] = {
            "execute": execute_wide,
            "decision": decision,
            "decision_sha256": digest(decision_path) if decision is not None else None,
            "decision_status": "recorded" if decision is not None else "pending",
        }
    for out in paths:
        print(f"Auditing {out.name} on {device}", flush=True)
        try:
            row = audit_run(out, device, atol, rtol, execute_wide)
        except Exception as exc:
            row = {"run_id": out.name, "state": "failed", "error": repr(exc)}
        result["runs"].append(row)
        print(row["run_id"], row["state"], row.get("error", ""), flush=True)
    expected = set().union(
        *(set(r.get("expected_phase_runs_from_frozen_config", [])) for r in result["runs"])
    )
    observed = {r["run_id"] for r in result["runs"]}
    result["matrix"] = {
        "expected_from_frozen_configs": sorted(expected),
        "observed": sorted(observed),
        "missing": sorted(expected - observed),
        "unexpected": sorted(observed - expected),
    }
    try:
        result["architecture_pairs"] = pairing_audit(result["runs"])
    except Exception as exc:
        result["pairing_error"] = repr(exc)
    result["passed"] = (
        bool(paths)
        and all(r["state"] == "passed" for r in result["runs"])
        and not (expected - observed)
        and not (observed - expected)
        and "pairing_error" not in result
    )
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output = ROOT / f"docs/development-artifacts/grok-depth-v1/endpoint-audit-{args.phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(output)
    print(output, flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
