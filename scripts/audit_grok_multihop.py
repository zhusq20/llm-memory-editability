#!/usr/bin/env python3
"""Recount longer-path endpoints and optionally reload them at execution precision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import SmallGPT, source_hash, utc, write_json
from llm_memory_editability.grok_multihop import autonomous_calls, evaluate_rows
from llm_memory_editability.grok_multihop_data import audit_world

ROOT = Path(__file__).resolve().parents[1]


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
    names = (
        "atomic",
        "id_atomic",
        "ood_atomic",
        "train_composite",
        "test_composite",
        "ood_composite",
        "test_full_composite",
    )
    for name in names:
        target = world[name][:, -1]
        check(name + ":denominator", endpoint[name]["n"] == len(target))
        if not len(target):
            check(name + ":empty", endpoint[name]["accuracy"] is None)
            continue
        answer, stop = (predictions[name + "_" + suffix] for suffix in ("answer", "stop"))
        check(name + ":targets", np.array_equal(predictions[name + "_target"], target))
        check(name + ":answer", endpoint[name]["answer_accuracy"] == (answer == target).mean())
        check(
            name + ":answer_eos",
            endpoint[name]["accuracy"] == ((answer == target) & (stop == 1)).mean(),
        )
        check(name + ":nll", endpoint[name]["nll"] == predictions[name + "_nll"].mean())
    call_eos = np.stack(
        [predictions[f"autonomous_hop{j}_stop"] == 1 for j in range(1, spec["hops"] + 1)]
    ).all(0)
    call_answer = predictions[f"autonomous_hop{spec['hops']}_answer"]
    call_correct = (call_answer == world["test_composite"][:, -1]) & call_eos
    check(
        "autonomous_calls:accuracy", endpoint["autonomous_calls"]["accuracy"] == call_correct.mean()
    )
    batch, length, width, layers = (
        spec["batch_size"],
        spec["hops"] + 2,
        spec["width"],
        spec["layers"],
    )
    vocab = 2 + spec["entities"] + spec["relations"]
    flops = 3 * (
        layers * (24 * batch * length * width**2 + 4 * batch * length**2 * width)
        + 4 * batch * width * vocab
    )
    check(
        "FLOPs_independent_formula", endpoint["estimated_training_flops"] == flops * spec["steps"]
    )
    check("example_count", sum(endpoint["counts"].values()) == spec["steps"] * batch)
    check(
        "effective_token_count",
        endpoint["effective_input_tokens"]
        == endpoint["counts"]["atomic"] * 3 + endpoint["counts"]["composite"] * length,
    )
    reload_error = None
    if device:
        state = torch.load(
            directory / f"weights-{spec['steps']:07d}.pt", map_location=device, weights_only=False
        )
        check("checkpoint_spec", state["spec"] == spec)
        cfg = ModelConfig(
            vocab_size=vocab, width=width, layers=layers, heads=spec["heads"], context=8
        )
        model = SmallGPT(cfg, spec["dropout"]).to(device).eval()
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-multihop-development-v1.json")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out", default="docs/development-artifacts/grok-multihop-v1/endpoint-audit.json"
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
    result["finished_utc"] = utc()
    result["passed"] = len(result["runs"]) == len(cfg["runs"]) and all(
        run["passed"] for run in result["runs"]
    )
    write_json(ROOT / args.out, result)
    print(json.dumps({"passed": result["passed"], "runs": len(result["runs"])}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
