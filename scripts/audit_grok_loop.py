#!/usr/bin/env python3
"""Recount looped-model endpoints and optionally reload them at execution precision."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import source_hash, utc, write_json
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.grok_multihop import autonomous_calls, evaluate_rows
from llm_memory_editability.grok_multihop_data import audit_world, path_details

ROOT = Path(__file__).resolve().parents[1]
SCORING_SPLITS = (
    "atomic",
    "id_atomic",
    "ood_atomic",
    "train_composite",
    "test_composite",
    "ood_composite",
)


def independent_flops_per_step(spec):
    batch, length, width = spec["batch_size"], spec["hops"] + 2, spec["width"]
    depth = spec["layers"] * spec["repeats"]
    vocab = 2 + spec["entities"] + spec["relations"]
    return 3 * (
        depth * (24 * batch * length * width**2 + 4 * batch * length**2 * width)
        + 4 * batch * width * vocab
    )


def audit_node(row, predictions, world, spec):
    """Recount every registered learning node, including all EOS conditions."""
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    step, batch, atomic = row["step"], spec["batch_size"], spec["n_atomic_per_batch"]
    length, width, vocab = spec["hops"] + 2, spec["width"], 2 + spec["entities"] + spec["relations"]
    parameters = (vocab + 10) * width + spec["layers"] * (12 * width**2 + 13 * width)
    check("fixed_32_atomic_224_composite_batch", atomic == 32 and batch - atomic == 224)
    check("atomic_exposure", row["counts"]["atomic"] == step * atomic)
    check("composite_exposure", row["counts"]["composite"] == step * (batch - atomic))
    check("example_count", row["examples"] == step * batch)
    check("supervised_tokens", row["supervised_tokens"] == step * batch * 2)
    check(
        "effective_input_tokens",
        row["effective_input_tokens"] == step * (atomic * 3 + (batch - atomic) * length),
    )
    check(
        "FLOPs_independent_formula",
        row["estimated_training_flops"] == step * independent_flops_per_step(spec),
    )
    check("parameters_independent_formula", row["parameters"] == parameters)
    for name, expected in (
        ("hops", spec["hops"]),
        ("unique_layers", spec["layers"]),
        ("repeats", spec["repeats"]),
        ("effective_depth", spec["layers"] * spec["repeats"]),
    ):
        check(name, row[name] == expected)
    names = SCORING_SPLITS + (("test_full_composite",) if step == spec["steps"] else ())
    for name in names:
        target = world[name][:, -1]
        metric = row[name]
        check(name + ":denominator", metric["n"] == len(target))
        if not len(target):
            check(
                name + ":empty",
                all(metric[key] is None for key in ("accuracy", "answer_accuracy", "nll")),
            )
            continue
        answer, stop, nll = (
            predictions[name + "_" + suffix] for suffix in ("answer", "stop", "nll")
        )
        valid = answer.shape == stop.shape == target.shape and nll.shape == (len(target), 2)
        check(name + ":prediction_shapes", valid)
        if not valid:
            continue
        check(name + ":targets", np.array_equal(predictions[name + "_target"], target))
        check(name + ":answer", metric["answer_accuracy"] == (answer == target).mean())
        check(name + ":answer_eos", metric["accuracy"] == ((answer == target) & (stop == 1)).mean())
        check(name + ":nll_finite", np.isfinite(nll).all())
        check(name + ":nll", metric["nll"] == nll.mean())
    rows = world["test_composite"]
    calls = row["autonomous_calls"]
    check("autonomous_calls:denominator", calls["n"] == len(rows))
    if not len(rows):
        check(
            "autonomous_calls:empty", calls["accuracy"] is None and calls["answer_accuracy"] is None
        )
        return checks
    nodes, _ = path_details(rows, world["atomic"], spec["entities"], spec["relations"])
    correct, eos = [], []
    for j in range(1, spec["hops"] + 1):
        answer, stop = (
            predictions[f"autonomous_hop{j}_answer"],
            predictions[f"autonomous_hop{j}_stop"],
        )
        correct.append(answer == nodes[:, j])
        eos.append(stop == 1)
        check(
            f"autonomous_calls:hop{j}",
            calls["hop_accuracy"][j - 1] == (correct[-1] & eos[-1]).mean(),
        )
    all_eos = np.stack(eos).all(0)
    check("autonomous_calls:calls", calls["calls"] == spec["hops"])
    check("autonomous_calls:answer", calls["answer_accuracy"] == correct[-1].mean())
    check("autonomous_calls:accuracy", calls["accuracy"] == (correct[-1] & all_eos).mean())
    check(
        "autonomous_calls:all_intermediates",
        calls["all_intermediate_answers_and_eos_correct"]
        == (np.stack(correct).all(0) & all_eos).mean(),
    )
    return checks


def stream_state_hash(state):
    serializable = json.dumps(state, sort_keys=True, default=lambda value: value.tolist())
    return hashlib.sha256(serializable.encode()).hexdigest()


def audit_sampling_pairs(runs):
    """Check paired terminal sampling states and total exposure across architectures."""
    groups = defaultdict(list)
    for run in runs:
        groups[json.dumps(run["sampling_condition"], sort_keys=True)].append(run)
    results = []
    for condition, group in groups.items():
        checks = []
        for name in ("dataset_sha256", "terminal_stream_sha256", "endpoint_counts"):
            values = [json.dumps(run[name], sort_keys=True) for run in group]
            checks.append({"name": name, "passed": len(set(values)) == 1})
        results.append(
            {
                "condition": json.loads(condition),
                "runs": [run["run"] for run in group],
                "comparisons_available": len(group) > 1,
                "passed": all(check["passed"] for check in checks),
                "checks": checks,
            }
        )
    return results


def audit_run(directory, device=None):
    meta = json.loads((directory / "metadata.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    spec, endpoint = complete["spec"], complete["endpoint"]
    world = dict(np.load(directory / "world.npz"))
    world["metadata"] = json.loads((directory / "world-metadata.json").read_text())
    predictions = dict(np.load(directory / f"predictions-{spec['steps']:07d}.npz"))
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    check("spec_identity", meta["spec"] == spec)
    check("fixed_budget_complete", endpoint["step"] == spec["steps"])
    data_audit = audit_world(world)
    check("dataset_hash", data_audit["dataset_sha256"] == world["metadata"]["dataset_sha256"])
    for filename, expected in meta["files"].items():
        snapshot = directory / "source" / Path(filename).relative_to(ROOT)
        check(
            f"execution_source:{snapshot.name}", source_hash([snapshot])[str(snapshot)] == expected
        )
        current = Path(filename)
        current_matches = source_hash([current])[str(current)] == expected
        check(f"current_source:{current.name}", current_matches)
        if device and current.suffix == ".py" and not current_matches:
            raise ValueError(f"Current reload source differs from execution: {current}")
    history = json.loads((directory / "learning.json").read_text())
    check("registered_learning_nodes", [row["step"] for row in history] == spec["nodes"])
    check("endpoint_matches_history", bool(history) and endpoint == history[-1])
    for row in history:
        path = directory / f"predictions-{row['step']:07d}.npz"
        check(f"node{row['step']}:predictions_exist", path.exists())
        if not path.exists():
            continue
        with np.load(path) as archive:
            for item in audit_node(row, dict(archive), world, spec):
                check(f"node{row['step']}:{item['name']}", item["passed"])
    latest = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
    check("latest_spec", latest["spec"] == spec)
    check("latest_step", latest["step"] == spec["steps"])
    check("latest_counts", latest["counts"] == endpoint["counts"])
    check("latest_stream_batches", latest["stream"]["batches"] == spec["steps"])
    check("latest_stream_atomic_batch", latest["stream"]["n_atomic"] == spec["n_atomic_per_batch"])
    check("latest_stream_batch_size", latest["stream"]["batch_size"] == spec["batch_size"])
    check("latest_stream_atomic_pool", latest["stream"]["atomic_size"] == len(world["atomic"]))
    check(
        "latest_stream_composite_pool",
        latest["stream"]["composite_size"] == len(world["train_composite"]),
    )
    terminal_stream_hash = stream_state_hash(latest["stream"])
    del latest
    names = SCORING_SPLITS + ("test_full_composite",)
    length, width = spec["hops"] + 2, spec["width"]
    vocab = 2 + spec["entities"] + spec["relations"]
    reload_error = None
    if device:
        state = torch.load(
            directory / f"weights-{spec['steps']:07d}.pt", map_location=device, weights_only=False
        )
        check("checkpoint_spec", state["spec"] == spec)
        cfg = ModelConfig(
            vocab_size=vocab, width=width, layers=spec["layers"], heads=spec["heads"], context=8
        )
        model = (
            LoopGPT(
                cfg,
                repeats=spec["repeats"],
                dropout=spec["dropout"],
                initialization=spec["init_scheme"],
            )
            .to(device)
            .eval()
        )
        model.load_state_dict(state["model"])
        check(
            "parameter_count", sum(p.numel() for p in model.parameters()) == complete["parameters"]
        )
        reload_error = 0.0
        for name in names:
            _, reloaded = evaluate_rows(model, world[name], device, length)
            if not len(world[name]):
                continue
            for key in ("answer", "stop", "target"):
                check(
                    name + ":reload_" + key,
                    np.array_equal(reloaded[key], predictions[name + "_" + key]),
                )
            error = float(np.max(np.abs(reloaded["nll"] - predictions[name + "_nll"])))
            reload_error = max(reload_error, error)
            check(name + ":reload_nll", error <= 1e-5)
        _, call_reload = autonomous_calls(model, world["test_composite"], world, device, length)
        for key, value in call_reload.items():
            check(
                "autonomous_reload:" + key, np.array_equal(value, predictions["autonomous_" + key])
            )
        del model
    return {
        "run": directory.name,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "reload_max_nll_absolute_error": reload_error,
        "counts": data_audit["counts"],
        "learning_nodes_checked": len(history),
        "dataset_sha256": data_audit["dataset_sha256"],
        "terminal_stream_sha256": terminal_stream_hash,
        "endpoint_counts": endpoint["counts"],
        "sampling_condition": {
            key: value
            for key, value in spec.items()
            if key not in {"architecture", "layers", "repeats"}
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-loop-development-v1.json")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out", default="docs/development-artifacts/grok-loop-v1/endpoint-audit.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.device:
        torch.cuda.set_device(torch.device(args.device))
    cfg = json.loads((ROOT / args.config).read_text())
    result = {
        "started_utc": utc(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device else None,
        "torch": torch.__version__,
        "tf32": True,
        "auditor_source": source_hash([Path(__file__).resolve()]),
        "runs": [],
    }
    for run_id, item in cfg["runs"].items():
        spec = {**cfg["base"], **item}
        directory = ROOT / cfg["output_root"] / spec["phase"] / run_id
        if (directory / "complete.json").exists():
            result["runs"].append(audit_run(directory, args.device))
    result["paired_sampling"] = audit_sampling_pairs(result["runs"])
    result["finished_utc"] = utc()
    result["passed"] = (
        len(result["runs"]) == len(cfg["runs"])
        and all(run["passed"] for run in result["runs"])
        and all(pair["passed"] for pair in result["paired_sampling"])
    )
    write_json(ROOT / args.out, result)
    print(json.dumps({"passed": result["passed"], "runs": len(result["runs"])}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
