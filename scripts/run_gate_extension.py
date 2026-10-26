#!/usr/bin/env python3
"""Run registered gate extensions through the existing native training machinery.

The sampler, answer/EOS loss, CUDA graph, checkpoints and online tracker are
inherited. This scoped adapter supplies the new data families and the explicitly
registered frozen-zero positional embedding for length extrapolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_paper_update_gate as update  # noqa: E402

from llm_memory_editability import gate_extension_data as data  # noqa: E402
from llm_memory_editability.dual_gate import DualGateGPT  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402

native, depth = update.native, update.depth
_native_graph_values = native.graph_values
STAGES = ("preflight", "train", "smoke", "audit", "grid", "grid-audit")


def validate_config(config):
    if config["family"] not in {"confirmation", "wang", "depth"}:
        raise ValueError("Unknown extension family")
    position = config.get("position_encoding", "learned")
    if position not in {"learned", "none"}:
        raise ValueError("position_encoding must be learned or none")
    if (config["family"] == "depth") != (position == "none"):
        raise ValueError("Frozen-zero NoPE is registered only and always for depth")
    for arm in config["arms"].values():
        if arm["architecture"] != "discoloop" or arm["update_mode"] not in {
            "full",
            "fixed",
            "token",
        }:
            raise ValueError("Only the registered paper/fixed/token gate arms can run")
        if arm["initial_update"] != 0.95:
            raise ValueError("The extension freezes the selected initial update at 0.95")
    if len(config["arms"]) != 3 or {arm["update_mode"] for arm in config["arms"].values()} != {
        "full",
        "fixed",
        "token",
    }:
        raise ValueError("Each case requires one complete full/fixed/token comparison")
    if len({(arm["unique_layers"], arm["repeats"]) for arm in config["arms"].values()}) != 1:
        raise ValueError("All three arms must share the same block and loop counts")
    nodes = config["evaluation_nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != config["training"]["steps"]:
        raise ValueError("Evaluation nodes must uniquely span the complete training budget")
    repeats = config["inference_repeats"]
    if not repeats or repeats != sorted(set(repeats)) or repeats[0] < 1:
        raise ValueError("Inference budgets must be a nonempty increasing positive list")


def load(root):
    world, panels, metadata = data.load(root)
    # Wang's original two atomic panels are samples. Premise indices refer to
    # all 40k facts, so retain a separately named full-table evaluation.
    if metadata["kind"] == "wang":
        panels = {**panels, "atomic_all": world["atoms"]}
    return world, panels, metadata


def verify_input(config):
    """Reject mismounted data before allocating a model or consuming a GPU."""
    _, _, metadata = load(config["input_root"])
    expected_kind = {
        "confirmation": "permutation_confirmation",
        "wang": "wang",
        "depth": "depth_extrapolation",
    }[config["family"]]
    if metadata["kind"] != expected_kind or config["data"]["kind"] != expected_kind:
        raise ValueError("Dataset kind disagrees with the registered experiment family")
    for key, expected in config.get("input_origin", {}).items():
        if key in {"world_sha256", "panels_sha256"} and metadata[key] != expected:
            raise ValueError(f"Frozen input identity changed: {key}")
    for key in ("world_seed", "entities_per_graph", "relations"):
        if key in config["data"] and metadata.get(key) != config["data"][key]:
            raise ValueError(f"Mounted data differs from the registered specification: {key}")
    return metadata


def initialize(config, name, metadata, device):
    model, spec, arm, _ = update.initialize(config, name, metadata, device)
    position = config.get("position_encoding", "learned")
    if position == "none":
        with torch.no_grad():
            model.transformer.wpe.weight.zero_()
        model.transformer.wpe.weight.requires_grad_(False)
    model.position_encoding = position
    arm = {**arm, "position_encoding": position, "family": config["family"]}
    return model, spec, arm, model_digest(model)


def optimizer_for(model, lr, decay):
    """Keep the existing AdamW groups, excluding explicitly frozen parameters."""
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    regular = [(n, p) for n, p in parameters if not n.startswith("gate_")]
    return torch.optim.AdamW(
        [
            {
                "params": [p for n, p in regular if not any(s in n for s in ("bias", "ln"))],
                "weight_decay": decay,
            },
            {
                "params": [p for n, p in regular if any(s in n for s in ("bias", "ln"))]
                + [p for n, p in parameters if n.startswith("gate_")],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
        capturable=True,
    )


def accounting(model, spec, arm, metadata):
    result = update.accounting(model, spec, arm, metadata)
    result.update(
        family=arm["family"],
        position_encoding=arm["position_encoding"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        primary_score=metadata["primary_score"],
        flops_note=f"Dense padded T={metadata['sequence_length']}, two supervised output slots; "
        "training factor3; includes both differentiable vocabulary projections per channel. "
        "Excludes softmax, normalization, dropout, reductions and memory traffic.",
    )
    return result


def graph_values(graph, coefficient):
    result = _native_graph_values(graph, coefficient)
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError(f"Nonfinite captured training loss or gradient: {result}")
    return result


def premise_metrics(rows, prediction, atomic_correct, world, metadata):
    nodes, edges = data.path_details(rows, metadata, world)
    eligible = atomic_correct[edges].all(axis=1)
    answer = prediction["answer"] == rows[:, -1] + metadata["entity_offset"]
    exact = answer & (prediction["stop"] == metadata["end_marker"])
    nonrepeat = np.array([len(set(path)) == len(path) for path in nodes])
    return {
        "premise_correct_n": int(eligible.sum()),
        "premise_coverage": float(eligible.mean()),
        "answer_accuracy_given_premises": float(answer[eligible].mean())
        if eligible.any()
        else None,
        "accuracy_given_premises": float(exact[eligible].mean()) if eligible.any() else None,
        "nonrepeated_n": int(nonrepeat.sum()),
        "nonrepeated_answer_accuracy": float(answer[nonrepeat].mean()) if nonrepeat.any() else None,
    }


def atomic_correctness(predictions, panels, metadata):
    names = ("atomic_all",) if "atomic_all" in panels else ("atomic_id", "atomic_ood")
    # These conditional subsets concern whether the prerequisite FACT can be
    # answered. EOS formatting remains an independently reported exact score.
    return np.concatenate(
        [
            predictions[name]["answer"] == panels[name][:, -1] + metadata["entity_offset"]
            for name in names
        ]
    )


def measure_panels(model, world, panels, metadata, device, batch_size):
    metrics, predictions = {}, {}
    for name, rows in panels.items():
        metrics[name], predictions[name] = native.evaluate(
            model, rows, metadata, device, batch_size
        )
    atomic = atomic_correctness(predictions, panels, metadata)
    for name, rows in panels.items():
        if not name.startswith("atomic_"):
            metrics[name].update(premise_metrics(rows, predictions[name], atomic, world, metadata))
    return metrics, predictions


@torch.no_grad()
def sweep(model, world, panels, metadata, device, batch_size, repeats):
    original_repeats, original_mode = model.repeats, model.training
    before = model_digest(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    metrics, predictions = {}, {}
    try:
        for count in repeats:
            model.repeats = count
            metrics[str(count)], predictions[str(count)] = measure_panels(
                model, world, panels, metadata, device, batch_size
            )
    finally:
        model.repeats = original_repeats
        model.train(original_mode)
    if model_digest(model) != before or not torch.equal(cpu_rng, torch.get_rng_state()):
        raise AssertionError("Endpoint evaluation changed weights or CPU RNG")
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise AssertionError("Endpoint evaluation consumed training CUDA RNG")
    common = np.logical_and.reduce(
        [atomic_correctness(predictions[str(r)], panels, metadata) for r in repeats]
    )
    for count in metrics:
        for name, rows in panels.items():
            if name.startswith("atomic_"):
                continue
            conditional = premise_metrics(rows, predictions[count][name], common, world, metadata)
            metrics[count][name].update(
                common_premise_n=conditional["premise_correct_n"],
                common_premise_coverage=conditional["premise_coverage"],
                answer_accuracy_given_common_premises=conditional["answer_accuracy_given_premises"],
                accuracy_given_common_premises=conditional["accuracy_given_premises"],
            )
    return metrics, predictions


def preflight(config, name, base):
    """Discarded actual CUDA/BF16 forwards, backward, capture and one update."""
    destination = base / "preflight" / f"{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    world, _, metadata = load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    table = data.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    index = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)

    def matching(mode):
        configured = {
            **config,
            "arms": {**config["arms"], name: {**config["arms"][name], "update_mode": mode}},
        }
        return initialize(configured, name, metadata, device)[0]

    full = matching("full")
    reference = DualGateGPT(full.config, spec["repeats"], gate_variant="paper", discoloop=True)
    reference = reference.to(device)
    reference.load_state_dict(full.state_dict(), strict=True)
    if config["position_encoding"] == "none":
        reference.transformer.wpe.weight.requires_grad_(False)
    findings = {
        f"full_P_{key}": value
        for key, value in update.exact_pair(
            reference, full, table, index, config["dropout_seed"]
        ).items()
    }
    del reference, full
    fixed, token = matching("fixed"), matching("token")
    findings["fixed_token_initial_hashes_equal"] = model_digest(fixed) == model_digest(token)
    findings.update(
        {
            f"fixed_token_{key}": value
            for key, value in update.exact_pair(
                fixed, token, table, index, config["dropout_seed"]
            ).items()
        }
    )
    del fixed, token
    output, gradients = update.forward_gradients(model, table, index, config["dropout_seed"])
    findings["actual_forward_all_gradients_finite"] = bool(torch.isfinite(output).all()) and all(
        bool(torch.isfinite(g).all()) for g in gradients.values()
    )
    findings["all_trainable_parameters_receive_gradients"] = set(gradients) == {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    findings["all_gate_gradients_nonzero"] = all(
        bool(torch.count_nonzero(g)) for n, g in gradients.items() if n.startswith("gate_")
    )
    del output, gradients
    model.zero_grad(set_to_none=True)
    # Restore the configured scientific RNG before checking capture restoration.
    torch.manual_seed(config["dropout_seed"])
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    findings["optimizer_contains_all_and_only_trainable_parameters"] = optimized == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    gate_ids = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
    findings["all_gates_zero_decay"] = all(
        group["weight_decay"] == 0
        for group in optimizer.param_groups
        if any(id(p) in gate_ids for p in group["params"])
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    graph = native.AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    findings.update(
        capture_restores_weights=model_digest(model) == initial,
        capture_restores_cpu_rng=torch.equal(cpu_rng, torch.get_rng_state()),
        capture_restores_cuda_rng=torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        capture_restores_adam=all(
            not bool(torch.count_nonzero(v))
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ),
    )
    saved = {n: p.detach().clone() for n, p in model.state_dict().items()}
    lr.fill_(spec["lr"])
    graph(index)
    torch.cuda.synchronize()
    graph_values(graph, 0.0)
    findings["captured_update_nonzero"] = model_digest(model) != initial
    if config["position_encoding"] == "none":
        findings["frozen_positions_remain_zero_after_real_update"] = not bool(
            torch.count_nonzero(model.transformer.wpe.weight)
        )
    model.load_state_dict(saved, strict=True)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    lr.zero_()
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    findings["discarded_engineering_update_restored"] = model_digest(model) == initial
    result = {
        "passed": all(findings.values()),
        "findings": findings,
        "engineering_only": True,
        "engineering_updates_executed": 1,
        "engineering_updates_retained": 0,
        "arm": name,
        "config_sha256": config["_checked_config_sha256"],
        "world_sha256": metadata["world_sha256"],
        "accounting": accounting(model, spec, arm, metadata),
        "utc": utc(),
    }
    write_json(destination, result)
    if not result["passed"]:
        raise AssertionError(findings)


@contextmanager
def installed(config):
    """Scope all adaptations, including nested calls and exceptional exits."""
    _, panels, _ = load(config["input_root"])
    replacements = {
        "initialize": initialize,
        "accounting": accounting,
        "optimizer_for": optimizer_for,
        "preflight": preflight,
        "load": load,
        "pack_rows": data.pack_rows,
        "make_table": data.make_table,
        "exposure_summary": data.exposure_summary,
        "measure_panels": measure_panels,
        "premise_metrics": premise_metrics,
        "graph_values": graph_values,
        "panel_names": lambda: tuple(panels),
    }
    saved = {key: getattr(native, key) for key in replacements}
    saved_depth = {key: getattr(depth, key) for key in ("load", "panel_names", "sweep")}
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        depth.load = load
        depth.panel_names = replacements["panel_names"]
        depth.sweep = lambda *args: sweep(*args, repeats=config["inference_repeats"])
        yield
    finally:
        for key, value in saved.items():
            setattr(native, key, value)
        for key, value in saved_depth.items():
            setattr(depth, key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positional_stage", nargs="?", choices=STAGES)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", "--base", dest="base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    stage = args.stage or args.positional_stage
    if stage is None or (
        args.stage and args.positional_stage and args.stage != args.positional_stage
    ):
        parser.error("Specify exactly one unambiguous stage")
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.arm not in config["arms"]:
        raise ValueError("Arm is absent from the frozen configuration")
    verify_input(config)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = args.base / ("smoke" if stage == "smoke" else "runs") / args.arm
    try:
        with installed(config):
            if stage == "preflight":
                preflight(
                    {**config, "_checked_config_sha256": digest(args.config)}, args.arm, args.base
                )
            elif stage == "smoke":
                native.train(config, args.arm, out, True, args.config, args.base)
            elif stage == "train":
                depth.tracked_train(config, args.arm, args.base, args.config)
            elif stage == "audit":
                native.audit(config, args.arm, out, args.config)
            else:
                depth.endpoint(
                    config, args.arm, args.base, args.config, audit=stage == "grid-audit"
                )
    except BaseException as error:
        write_json(
            args.base / "failures" / f"{args.arm}-{stage}.json",
            {
                "utc": utc(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
