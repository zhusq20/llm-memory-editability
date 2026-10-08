"""Sparse versus dense sequential learning, using the audited existing trainers.

The ordinary four-layer path is unchanged, so compatible completed ordinary
runs can be reused. MoE uses the already tested whole-batch balancing update;
wide dense uses its actual FFN dimensions in the compute ledger.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from . import parametric_architecture as architecture
from . import parametric_architecture_train as architecture_train
from . import sequential_transfer as st
from .grok_depth import utc
from .grokking_reproduction import model_digest
from .loop_learning_train import training_plan
from .realworld_composition_data import sha256, write_json

ARMS = ("D4", "M4", "W4")


def construct(config, spec, device):
    arm = spec.get("architecture", "D4" if spec.get("arm") == "standard" else None)
    if arm not in ARMS:
        raise ValueError(f"Unknown sequential architecture: {arm}")
    if spec.get("unique_layers", 4) != 4 or spec.get("repeats", 1) != 1:
        raise ValueError("This comparison executes four independent layers once")
    return architecture.construct(config, {**spec, "architecture": arm}, device)


def manifest(model):
    ledger = architecture.parameter_ledger(model)
    return {
        **ledger,
        "unique_parameters": ledger["total_parameters"],
        "trainable_unique_parameters": ledger["trainable_parameters"],
        "active_unique_parameters": ledger["nominal_parameters_selected_per_token"],
        "hidden_size": model.config.n_embd,
        "attention_heads": model.config.n_head,
        "vocab_size": model.config.vocab_size,
        "tied_input_output_embedding": True,
        "initialization": architecture.initialization_manifest(model),
        "initial_model_sha256": model_digest(model),
        "scope": "same GPT-2 backbone; dense 4d, eight 2d experts top-2, wide 16d FFN",
        "flops_scope": "actual padded attention/readout and valid expert tokens; 3x forward",
    }


@contextmanager
def adapter(spec, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ("construct", "training_plan", "update", "save_checkpoint")
    originals = {key: getattr(st, key) for key in keys}
    last_update = {}

    def build(config, current_spec, device):
        model = construct(config, current_spec, device)
        target = out / "architecture.json"
        if not target.exists():
            write_json(target, manifest(model))
        return model

    def update(model, optimizer, records, current_spec, device, pad):
        if model.kind == "dense":
            result = originals["update"](model, optimizer, records, current_spec, device, pad)
        else:
            result = architecture_train.update(model, optimizer, records, current_spec, device, pad)
        last_update.clear()
        last_update.update(result)
        step = int(optimizer.state[next(model.parameters())]["step"])
        selected = set(current_spec.get("evaluation_nodes", []))
        selected |= {
            current_spec["stage_a_steps"],
            current_spec["stage_a_steps"] + current_spec["stage_b_steps"],
        }
        if step % current_spec.get("status_interval", 50) == 0 or step in selected:
            with (out / "update-details.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **result}, allow_nan=False) + "\n")
        return result

    def save(path, model, optimizer, step, counts, counters, history, sampling):
        if history and history[-1]["step"] == step and last_update:
            history[-1].update(
                answer_ce=last_update["loss"],
                balance_loss=last_update.get("balance_loss", 0.0),
                objective=last_update.get("objective", last_update["loss"]),
                router=architecture_train._routing_metrics(last_update.get("router")),
            )
            router = last_update.get("router", {})
            for layer, entropy in enumerate(router.get("router_entropy_per_layer", [])):
                history[-1][f"router_entropy_layer_{layer}"] = entropy
                for expert, value in enumerate(router["assignment_fractions"][layer]):
                    history[-1][f"router_fraction_layer_{layer}_expert_{expert}"] = value
            write_json(out / "learning.json", history)
        originals["save_checkpoint"](
            path, model, optimizer, step, counts, counters, history, sampling
        )

    st.construct, st.training_plan = build, training_plan
    st.update, st.save_checkpoint = update, save
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(st, key, value)


def run(spec, out, device):
    with adapter(spec, out):
        return st.run(spec, out, device)


def audit(out, device):
    spec = json.loads((Path(out) / "run.json").read_text())["spec"]
    with adapter(spec, out):
        return st.audit(out, device)


def preflight(spec, out, device):
    with adapter(spec, out):
        return st.preflight(spec, out, device)


def verify_reused(config, out, device):
    """Read-only re-evaluation of old weights under the new ordinary builder."""
    results = []
    for entry in config["reused_runs"]:
        root = Path(entry["run_dir"])
        metadata = json.loads((root / "run.json").read_text())
        spec = metadata["spec"]
        if spec["arm"] != "standard":
            raise ValueError("Reused baseline must have independent ordinary layers")
        if any(sha256(root / name) != value for name, value in entry["files"].items()):
            raise ValueError("A reused baseline artifact has changed")
        if not json.loads((root / "audit.json").read_text())["passed"]:
            raise ValueError("A reused baseline lacks its independent reload audit")
        with adapter(spec, Path(out) / entry["name"]):
            data, tokenizer, model, actual_device = st._setup(spec, device)
            records, plan = training_plan(data, spec)
            sampling = st.plan_manifest(records, plan, spec)
            expected_sampling = json.loads((root / "sampling-plan.json").read_text())
            if sampling != expected_sampling:
                raise AssertionError("Reused ordinary sample stream/exposure differs")
            if model_digest(model) != metadata["initial_model_sha256"]:
                raise AssertionError("Reused ordinary initial weights differ")
            for filename, step in (
                ("stage-a.pt", spec["stage_a_steps"]),
                ("latest.pt", spec["stage_a_steps"] + spec["stage_b_steps"]),
            ):
                state = torch.load(root / filename, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"])
                if state["step"] != step or state["plan_sha256"] != sampling["plan_sha256"]:
                    raise AssertionError("Reused checkpoint step or sample stream differs")
                metrics, predictions = st.evaluate_transfer(
                    model,
                    data,
                    tokenizer,
                    actual_device,
                    full=True,
                    autonomous=True,
                    batch=spec["evaluation_batch_size"],
                )
                expected = json.loads((root / f"predictions-{step:07d}.json").read_text())
                history = json.loads((root / "learning.json").read_text())
                expected_metrics = next(row["metrics"] for row in history if row["step"] == step)
                if predictions != expected or metrics != expected_metrics:
                    raise AssertionError("Reused endpoint generations/scores differ")
                results.append(
                    {
                        "name": entry["name"],
                        "step": step,
                        "passed": True,
                        "predictions_recomputed": sum(map(len, predictions.values())),
                    }
                )
    result = {
        "passed": bool(results) and all(row["passed"] for row in results),
        "nodes": results,
        "old_artifacts_modified": False,
        "created_utc": utc(),
    }
    write_json(Path(out) / "reuse-validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--verify-reused", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit(args.out, args.device)
    else:
        config = json.loads(args.config.read_text())
        if args.verify_reused:
            result = verify_reused(config, args.out, args.device)
        else:
            spec = next(row for row in config["specs"] if row["name"] == args.run)
            result = (preflight if args.preflight else run)(spec, args.out, args.device)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
