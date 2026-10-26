#!/usr/bin/env python3
"""Two injection ablations using the unchanged multi-hop training machinery.

Only initialization, architecture accounting, and a discarded preflight are
adapted. The ordinary CE, optimizer policy, sampler, CUDA graph, checkpointing,
evaluation, and W&B learning curves use the existing runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_dual_gate as original  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402
from llm_memory_editability.paper_update_contract import verify_input  # noqa: E402
from llm_memory_editability.residual_gate_injection import ResidualGateInjectionGPT  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402

BATCH = "residual-gate-injection-ablation-v1"
native, depth = original.native, original.depth
optimizer_for = original.optimizer_for


def validate_config(config, reference):
    if config.get("batch") != BATCH:
        raise ValueError("Unexpected residual-gate injection batch")
    expected = {
        name: {
            "architecture": "loop" if injection == "none" else "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "injection": injection,
            "update_mode": "token",
            "initial_update": 0.95,
        }
        for name, injection in (("only-u", "none"), ("fixed-q-plus-u", "fixed1"))
    }
    if config.get("arms") != expected:
        raise ValueError("Exactly the two authorized token-gate injection ablations are required")
    for field in ("data", "model", "dropout_seed", "evaluation_batch_size", "input_origin"):
        if config[field] != reference[field]:
            raise ValueError(f"Paired recipe changed: {field}")
    if config["training"] != {**reference["training"], "steps": 1500000}:
        raise ValueError("Preserve the existing recipe, with 1500000 steps from initialization")
    nodes = reference["evaluation_nodes"] + list(range(400000, 1500001, 100000))
    if config["evaluation_nodes"] != nodes:
        raise ValueError("Preserve every old evaluation node and the 100k continuation nodes")
    if config["budget"]["new_scientific_updates"] != 3000000:
        raise ValueError("The two-arm scientific budget must be 3000000 updates")
    if config["gate_scope"] != "continuous_update_only_nonfinal_all_tokens":
        raise ValueError("Wrong gate placement")
    for relative, expected_hash in config["paired_recipe"]["unchanged_source_sha256"].items():
        if digest(ROOT / relative) != expected_hash:
            raise ValueError(f"Inherited scientific implementation changed: {relative}")


def initialize(config, name, metadata, device):
    spec, arm = native.arm_spec(config, name)
    options = {
        key: config["arms"][name][key] for key in ("injection", "update_mode", "initial_update")
    }
    arm = {**arm, **options, "gate_scope": config["gate_scope"]}
    torch.manual_seed(spec["initialization"])
    hf = GPT2Config(
        **config["model"],
        n_layer=spec["unique_layers"],
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    model = ResidualGateInjectionGPT(
        hf, spec["repeats"], **options, **native.architecture_flags(arm)
    ).to(device)
    initial = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    return model, spec, arm, initial


def accounting(model, spec, arm, metadata):
    return {
        **original.original_accounting(model, spec, arm, metadata),
        "injection": model.injection,
        "update_mode": model.update_mode,
        "initial_update": model.initial_update,
        "paper_gate_parameters": 0,
        "update_gate_parameters": model.config.n_embd + 1,
        "gate_scope": "continuous_update_only_nonfinal_all_tokens",
        "boundary": "X + u(H) * (H-X)" + (" + q(H)" if model.discoloop else ""),
        "final_readout": "raw H; no final boundary",
        "pointwise_cost_note": "One shared token-scalar u gate and residual correction per "
        "nonfinal boundary. These O(B*T*d*(R-1)) operations are excluded from matrix FLOPs. "
        "Only fixed-q-plus-u computes the two vocabulary projections; no early exit.",
    }


def forward_gradients(model, table, index, seed):
    model.zero_grad(set_to_none=True)
    model.train()
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(table[0][index], table[1][index])
        loss = F.cross_entropy(logits.flatten(0, 1), table[2][index].flatten())
    loss.backward()
    return logits.detach(), {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def preflight(config, name, base):
    """CUDA/BF16 initialization and capture checks; one discarded update, no smoke run."""
    destination = base / "preflight" / f"{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    world, _, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    backbone = SupervisedAlignmentGPT(
        model.config, spec["repeats"], **native.architecture_flags(arm)
    ).to(device)
    backbone.load_state_dict(
        {n: value for n, value in model.state_dict().items() if not n.startswith("gate_")},
        strict=True,
    )
    findings = {
        "historical_initial_backbone_hash_matches": model_digest(backbone)
        == config["paired_recipe"]["initial_backbone_sha256"],
        "no_learned_injection_gate": not any(
            n.startswith("gate_paper_") for n in model.state_dict()
        ),
    }
    del backbone
    fixed = ResidualGateInjectionGPT(
        model.config,
        spec["repeats"],
        injection=model.injection,
        update_mode="fixed",
        initial_update=model.initial_update,
    ).to(device)
    fixed.load_state_dict(model.state_dict(), strict=True)
    table = native.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    index = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
    first, grad_first = forward_gradients(fixed, table, index, config["dropout_seed"])
    second, grad_second = forward_gradients(model, table, index, config["dropout_seed"])
    gate_gradients = {n: g for n, g in grad_second.items() if n.startswith("gate_update_")}
    findings.update(
        fixed_token_initial_forward_exact=torch.equal(first, second),
        fixed_token_all_backbone_gradients_exact=all(
            torch.equal(value, grad_second[name]) for name, value in grad_first.items()
        ),
        update_gate_gradients_finite_and_nonzero=len(gate_gradients) == 2
        and all(torch.isfinite(value).all().item() for value in gate_gradients.values())
        and all(torch.count_nonzero(value).item() > 0 for value in gate_gradients.values()),
    )
    del fixed, first, second, grad_first, grad_second, gate_gradients
    model.zero_grad(set_to_none=True)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    gate_ids = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
    findings["all_gate_parameters_have_zero_decay"] = all(
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
            not torch.count_nonzero(value)
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        ),
    )
    saved = {n: p.detach().clone() for n, p in model.state_dict().items()}
    lr.fill_(spec["lr"])
    graph(index)
    torch.cuda.synchronize()
    findings["captured_update_finite"] = all(
        np.isfinite(value) for value in native.graph_values(graph, 0).values()
    )
    findings["captured_update_nonzero"] = model_digest(model) != initial
    model.load_state_dict(saved, strict=True)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    lr.zero_()
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    findings["engineering_update_restored"] = model_digest(model) == initial
    write_json(
        destination,
        {
            "passed": all(findings.values()),
            "findings": findings,
            "engineering_only": True,
            "engineering_updates_executed": 1,
            "engineering_updates_retained": 0,
            "config_sha256": config["_checked_config_sha256"],
            "accounting": accounting(model, spec, arm, metadata),
            "utc": utc(),
        },
    )
    if not all(findings.values()):
        raise AssertionError(findings)


@contextmanager
def installed():
    replacements = {
        "initialize": initialize,
        "accounting": accounting,
        "optimizer_for": optimizer_for,
        "preflight": preflight,
    }
    saved = {key: getattr(native, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(native, key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "grid"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=("only-u", "fixed-q-plus-u"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference = ROOT / config["paired_recipe"]["config"]
    if digest(reference) != config["paired_recipe"]["sha256"]:
        raise ValueError("Pinned reference recipe changed")
    validate_config(config, json.loads(reference.read_text()))
    verify_input(config)
    with installed():

        def checked_preflight(value, name, base):
            return preflight({**value, "_checked_config_sha256": digest(args.config)}, name, base)

        native.preflight = checked_preflight
        depth.main()


if __name__ == "__main__":
    os.umask(0o022)
    main()
