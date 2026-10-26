#!/usr/bin/env python3
"""New gate adapters around the frozen native multi-hop training/evaluation code."""

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
import run_loop_hop_depth as depth  # noqa: E402

from llm_memory_editability.dual_gate import DualGateGPT, gate_parameter_count  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402

native = depth.native
original_accounting = native.accounting


def optimizer_for(model, lr, decay):
    parameters = list(model.named_parameters())
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


def initialize(config, name, metadata, device):
    spec, arm = native.arm_spec(config, name)
    variant = config["arms"][name]["gate_variant"]
    arm = {**arm, "gate_variant": variant}
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
    model = DualGateGPT(
        hf, spec["repeats"], gate_variant=variant, **native.architecture_flags(arm)
    ).to(device)
    initial = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    return model, spec, arm, initial


def accounting(model, spec, arm, metadata):
    result = original_accounting(model, spec, arm, metadata)
    result.update(
        gate_variant=model.gate_variant,
        gate_parameters=gate_parameter_count(model.gate_variant, model.config.n_embd),
        gate_operations_note="O(B*T*d*(R-1)) pointwise/reduction operations; "
        "reported separately, excluded from matrix-only FLOPs. No extra vocabulary projection.",
    )
    return result


def preflight(config, name, base):
    world, _, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    reference = SupervisedAlignmentGPT(model.config, spec["repeats"], discoloop=True).to(device)
    backbone = {n: p for n, p in model.state_dict().items() if not n.startswith("gate_")}
    reference.load_state_dict(backbone, strict=True)
    if model.gate_variant == "paper":
        # P starts at .99, not F. Preserve the same AMP operation order in its
        # constant-gate reference; the original boundary scales after casting.
        from types import MethodType

        from llm_memory_editability.self_decode_alignment import _loss_dtype, _readout

        initial_gate = torch.sigmoid(model.gate_paper_bias.detach()).clone()

        def paper_reference(current, state):
            probability = _loss_dtype(_readout(current, state)).softmax(-1)
            value = _loss_dtype(F.linear(probability, current.transformer.wte.weight.T))
            channel = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            return state + (initial_gate * channel).to(state.dtype)

        reference._discoloop_boundary = MethodType(paper_reference, reference)
    table = native.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    index = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
    outputs, grads = [], []
    for current in (reference, model):
        current.train()
        torch.manual_seed(config["dropout_seed"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = current(table[0][index], table[1][index])
            loss = F.cross_entropy(logits.flatten(0, 1), table[2][index].flatten())
        loss.backward()
        outputs.append(logits.detach())
        grads.append({n: p.grad.detach().clone() for n, p in current.named_parameters()})
    findings = {
        "initial_backbone_forward_exact": torch.equal(*outputs),
        "initial_backbone_all_gradients_exact": all(
            torch.equal(grads[0][n], grads[1][n]) for n in grads[0]
        ),
        "gate_gradients_finite_and_nonzero": all(
            torch.isfinite(v).all().item() for n, v in grads[1].items() if n.startswith("gate_")
        )
        and any(
            torch.count_nonzero(v).item() for n, v in grads[1].items() if n.startswith("gate_")
        ),
        "baseline_step0_backbone_hash_matches": model_digest(reference)
        == config["baseline"]["initial_model_sha256"],
    }
    # Do not keep default-stream autograd graphs alive during side-stream
    # capture. This includes loop locals, not only the saved comparisons.
    del reference, outputs, grads, current, logits, loss
    model.zero_grad(set_to_none=True)
    write_json(base / "preflight" / f"{name}-forward.json", {"findings": findings, "utc": utc()})
    lr = torch.tensor(0.0, device=device)
    optimizer = native.optimizer_for(model, lr, spec["weight_decay"])
    findings["all_gate_parameters_have_zero_decay"] = all(
        group["weight_decay"] == 0
        for group in optimizer.param_groups
        if any(
            id(p) in {id(v) for n, v in model.named_parameters() if n.startswith("gate_")}
            for p in group["params"]
        )
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
    # One real engineering update, then restore model and optimizer before exit.
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
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
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


def cloud(config, name, base):
    import wandb

    out = base / "runs" / name
    tracking = json.loads((out / "tracking.json").read_text())
    paths = ("complete.json", "audit.json", "grid.json", "grid-audit.json")
    payload = {p: json.loads((out / p).read_text()) for p in paths}
    if not all(payload[p]["passed"] for p in ("audit.json", "grid-audit.json")):
        raise ValueError("Independent audits must pass before cloud certification")
    canonical = out / "canonical-result.json"
    write_json(canonical, payload)
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        id=tracking["id"],
        resume="must",
        dir=str(out),
        mode="online",
        save_code=False,
        settings=wandb.Settings(disable_git=True),
    )
    run.summary.update(
        {
            "independently_reloaded": True,
            "canonical_result_sha256": digest(canonical),
            "scientific_final_step": payload["complete.json"]["step"],
        }
    )
    run.save(str(canonical), base_path=str(out), policy="now")
    run.finish()
    remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{tracking['id']}")
    downloaded = remote.file(canonical.name).download(
        root=str(out / "cloud-download"), replace=False
    )
    passed = (
        digest(downloaded.name) == digest(canonical)
        and remote.summary.get("canonical_result_sha256") == digest(canonical)
        and remote.summary.get("scientific_final_step") == config["training"]["steps"]
        and remote.state == "finished"
    )
    write_json(
        out / "cloud-audit.json",
        {
            "passed": passed,
            "url": tracking["url"],
            "canonical_sha256": digest(canonical),
            "utc": utc(),
        },
    )
    if not passed:
        raise AssertionError("Complete canonical JSON cloud verification failed")


def install():
    native.initialize = initialize
    native.accounting = accounting
    native.preflight = preflight
    native.optimizer_for = optimizer_for
    depth.cloud = cloud


if __name__ == "__main__":
    os.umask(0o022)
    install()
    depth.main()
