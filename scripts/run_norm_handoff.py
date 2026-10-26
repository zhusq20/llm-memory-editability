#!/usr/bin/env python3
"""Adapters for the R/N/A/C comparison, invoked explicitly on the target host."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_dual_gate as gates  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402
from llm_memory_editability.norm_handoff import NormHandoffGPT, gate_parameter_count  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402

native, depth = gates.native, gates.depth
original_accounting = native.accounting
optimizer_for = gates.optimizer_for


def initialize(config, name, metadata, device):
    spec, arm = native.arm_spec(config, name)
    change = {key: config["arms"][name][key] for key in ("handoff", "residual_scale")}
    arm.update(change)
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
    model = NormHandoffGPT(hf, spec["repeats"], **change, **native.architecture_flags(arm)).to(
        device
    )
    initial = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    return model, spec, arm, initial


def accounting(model, spec, arm, metadata):
    return {
        **original_accounting(model, spec, arm, metadata),
        "handoff": model.handoff,
        "residual_scale": model.residual_scale,
        "gate_parameters": gate_parameter_count(model.handoff, model.config.n_embd),
        "pointwise_cost_note": "Residual scaling, RMSNorm and scalar gate O(B*T*d) work "
        "is excluded from matrix FLOPs; no additional vocabulary projections.",
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
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }


def paired_contract(reference, candidate, table, index, seed):
    a, ga = forward_gradients(reference, table, index, seed)
    b, gb = forward_gradients(candidate, table, index, seed)
    backbone = [n for n in ga if not n.startswith("gate_")]
    return {
        "forward_exact": torch.equal(a, b),
        "all_backbone_gradients_exact": all(torch.equal(ga[n], gb[n]) for n in backbone),
    }


def preflight(config, name, base):
    world, _, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    backbone = {n: p for n, p in model.state_dict().items() if not n.startswith("gate_")}
    reference = SupervisedAlignmentGPT(model.config, spec["repeats"], discoloop=True).to(device)
    reference.load_state_dict(backbone, strict=True)
    findings = {
        "baseline_step0_backbone_hash_matches": model_digest(reference)
        == config["baseline"]["initial_model_sha256"]
    }
    table = native.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    index = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
    # Check the gamma=1/native-boundary path without changing scientific weights.
    mode, scale = model.handoff, model.residual_scale
    try:
        model.handoff, model.residual_scale = "fixed", 1.0
        findings.update(
            {
                f"native_{k}": v
                for k, v in paired_contract(
                    reference, model, table, index, config["dropout_seed"]
                ).items()
            }
        )
    finally:
        model.handoff, model.residual_scale = mode, scale
    del reference, backbone
    if mode in ("additive", "complementary"):
        peer = NormHandoffGPT(
            model.config,
            spec["repeats"],
            handoff="complementary" if mode == "additive" else "additive",
            discoloop=True,
        ).to(device)
        peer.load_state_dict(model.state_dict(), strict=True)
        findings.update(
            {
                f"A_C_{k}": v
                for k, v in paired_contract(
                    peer, model, table, index, config["dropout_seed"]
                ).items()
            }
        )
        del peer
    # Test the actual arm, including its gate gradients, before graph capture.
    output, grads = forward_gradients(model, table, index, config["dropout_seed"])
    findings["actual_arm_output_gradients_finite"] = bool(torch.isfinite(output).all()) and all(
        bool(torch.isfinite(v).all()) for v in grads.values()
    )
    findings["all_parameters_receive_gradients"] = len(grads) == len(list(model.parameters()))
    if mode in ("additive", "complementary"):
        findings["gate_gradients_nonzero"] = all(
            bool(torch.count_nonzero(v)) for n, v in grads.items() if n.startswith("gate_")
        )
    del output, grads
    model.zero_grad(set_to_none=True)
    write_json(base / "preflight" / f"{name}-forward.json", {"findings": findings, "utc": utc()})
    if not all(findings.values()):
        raise AssertionError(findings)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    gate_ids = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
    findings["gate_decay_zero"] = all(
        g["weight_decay"] == 0
        for g in optimizer.param_groups
        if any(id(p) in gate_ids for p in g["params"])
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    graph = native.AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    findings.update(
        capture_restores_weights=model_digest(model) == initial,
        capture_restores_cpu_rng=torch.equal(cpu_rng, torch.get_rng_state()),
        capture_restores_cuda_rng=torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        capture_restores_adam=all(
            not torch.count_nonzero(v)
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ),
    )
    saved = {n: p.detach().clone() for n, p in model.state_dict().items()}
    lr.fill_(spec["lr"])
    graph(index)
    torch.cuda.synchronize()
    findings["captured_update_finite"] = all(
        np.isfinite(v) for v in native.graph_values(graph, 0).values()
    )
    findings["captured_update_nonzero"] = model_digest(model) != initial
    model.load_state_dict(saved, strict=True)
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                v.zero_()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    findings["engineering_update_restored"] = model_digest(model) == initial
    write_json(
        base / "preflight" / f"{name}.json",
        {
            "passed": all(findings.values()),
            "findings": findings,
            "engineering_only": True,
            "engineering_updates_executed": 1,
            "engineering_updates_retained": 0,
            "accounting": accounting(model, spec, arm, metadata),
            "utc": utc(),
        },
    )
    if not all(findings.values()):
        raise AssertionError(findings)


def install():
    native.initialize = initialize
    native.accounting = accounting
    native.optimizer_for = optimizer_for
    native.preflight = preflight
    depth.cloud = gates.cloud


if __name__ == "__main__":
    os.umask(0o022)
    install()
    # The launcher enforces the all-arm engineering barrier before any train.
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        base = Path(sys.argv[sys.argv.index("--base") + 1])
        path = Path(sys.argv[sys.argv.index("--config") + 1])
        receipt = json.loads((base / "engineering-complete.json").read_text())
        if not receipt["passed"] or receipt["config_sha256"] != digest(path):
            raise ValueError("All-arm target-machine engineering checks must pass first")
    depth.main()
