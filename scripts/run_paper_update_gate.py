#!/usr/bin/env python3
"""Train/evaluate two update gates with the existing paper-gate data and runner.

No training loop, loss, sampler, optimizer, scorer or inference sweep is replaced.
The adapter is installed only for the requested stage and restored afterwards.
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
import run_dual_gate as paper  # noqa: E402

from llm_memory_editability.dual_gate import DualGateGPT  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402
from llm_memory_editability.paper_update_contract import (  # noqa: E402
    PANELS,
    validate_config,
    verify_baseline,
    verify_input,
)
from llm_memory_editability.paper_update_gate import PaperUpdateGateGPT  # noqa: E402

native, depth = paper.native, paper.depth
original_accounting = paper.original_accounting
optimizer_for = paper.optimizer_for
STAGES = (
    "verify-baseline",
    "preflight",
    "smoke",
    "smoke-audit",
    "smoke-grid",
    "smoke-grid-audit",
    "check-engineering",
    "train",
    "audit",
    "grid",
    "grid-audit",
    "cloud",
)


def initialize(config, name, metadata, device):
    spec, arm = native.arm_spec(config, name)
    changes = {k: config["arms"][name][k] for k in ("update_mode", "initial_update")}
    arm = {**arm, "gate_variant": "paper", **changes}
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
    model = PaperUpdateGateGPT(hf, spec["repeats"], **changes, **native.architecture_flags(arm)).to(
        device
    )
    initial = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    return model, spec, arm, initial


def accounting(model, spec, arm, metadata):
    width = model.config.n_embd
    return {
        **original_accounting(model, spec, arm, metadata),
        "gate_variant": "paper",
        "update_mode": model.update_mode,
        "initial_update": model.initial_update,
        "paper_gate_parameters": width + 1,
        "update_gate_parameters": width + 1 if model.update_mode == "token" else 0,
        "fixed_update_buffer_elements": width + 1 if model.update_mode == "fixed" else 0,
        "pointwise_cost_note": "One token-scalar update gate and residual correction at each "
        "nonfinal boundary; O(B*T*d*(R-1)) work excluded from matrix-only FLOPs. "
        "The paper vocabulary channel is evaluated once, unchanged. No early exit.",
    }


@contextmanager
def installed():
    """Keep imports and other experiment runners free of persistent mutation."""
    replacements = {
        "initialize": initialize,
        "accounting": accounting,
        "optimizer_for": optimizer_for,
        "preflight": preflight,
    }
    saved = {key: getattr(native, key) for key in replacements}
    old_cloud = depth.cloud
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        depth.cloud = paper.cloud
        yield
    finally:
        for key, value in saved.items():
            setattr(native, key, value)
        depth.cloud = old_cloud


def forward_gradients(model, table, index, seed):
    model.zero_grad(set_to_none=True)
    model.train()
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(table[0][index], table[1][index])
        loss = F.cross_entropy(logits.flatten(0, 1), table[2][index].flatten())
    loss.backward()
    return logits.detach(), {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }


def exact_pair(first, second, table, index, seed):
    a, ga = forward_gradients(first, table, index, seed)
    b, gb = forward_gradients(second, table, index, seed)
    return {
        "forward_exact": torch.equal(a, b),
        "all_common_gradients_exact": set(ga).issubset(gb)
        and all(torch.equal(ga[name], gb[name]) for name in ga),
    }


def preflight(config, name, base):
    """Actual CUDA/BF16 contracts, then one restored engineering update."""
    destination = base / "preflight" / f"{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    world, _, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    table = native.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    index = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
    state = {n: v for n, v in model.state_dict().items() if not n.startswith("gate_update_")}
    reference = DualGateGPT(model.config, spec["repeats"], gate_variant="paper", discoloop=True)
    reference = reference.to(device)
    reference.load_state_dict(state, strict=True)
    full = PaperUpdateGateGPT(model.config, spec["repeats"], update_mode="full").to(device)
    full.load_state_dict(state, strict=True)
    findings = {
        "original_P_initial_hash_matches": model_digest(reference)
        == config["baseline"]["initial_model_sha256"],
        **{
            f"full_P_{key}": value
            for key, value in exact_pair(
                reference, full, table, index, config["dropout_seed"]
            ).items()
        },
    }
    del reference, full, state
    fixed = PaperUpdateGateGPT(
        model.config, spec["repeats"], update_mode="fixed", initial_update=model.initial_update
    ).to(device)
    token = PaperUpdateGateGPT(
        model.config, spec["repeats"], update_mode="token", initial_update=model.initial_update
    ).to(device)
    fixed.load_state_dict(model.state_dict(), strict=True)
    token.load_state_dict(model.state_dict(), strict=True)
    findings.update(
        {
            f"fixed_token_{key}": value
            for key, value in exact_pair(fixed, token, table, index, config["dropout_seed"]).items()
        }
    )
    del fixed, token
    output, grads = forward_gradients(model, table, index, config["dropout_seed"])
    findings["actual_output_and_gradients_finite"] = bool(torch.isfinite(output).all()) and all(
        bool(torch.isfinite(v).all()) for v in grads.values()
    )
    findings["all_trainable_parameters_receive_gradients"] = len(grads) == len(
        list(model.parameters())
    )
    findings["paper_gate_gradients_nonzero"] = all(
        bool(torch.count_nonzero(v)) for n, v in grads.items() if n.startswith("gate_paper_")
    )
    if model.update_mode == "token":
        findings["update_gate_gradients_nonzero"] = all(
            bool(torch.count_nonzero(v)) for n, v in grads.items() if n.startswith("gate_update_")
        )
    del output, grads
    model.zero_grad(set_to_none=True)
    if not all(findings.values()):
        write_json(base / "preflight" / f"{name}-forward.json", {"findings": findings})
        raise AssertionError(findings)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    gate_ids = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
    findings["all_gate_parameters_zero_decay"] = all(
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
            not torch.count_nonzero(v)
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ),
    )
    saved = {n: p.detach().clone() for n, p in model.state_dict().items()}
    saved_adam = {
        p: {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in state.items()}
        for p, state in optimizer.state.items()
    }
    lr.fill_(spec["lr"])
    graph(index)
    torch.cuda.synchronize()
    findings["captured_update_finite"] = all(
        np.isfinite(v) for v in native.graph_values(graph, 0).values()
    )
    findings["captured_update_nonzero"] = model_digest(model) != initial
    model.load_state_dict(saved, strict=True)
    for p, state in saved_adam.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                optimizer.state[p][key].copy_(value)
            else:
                optimizer.state[p][key] = value
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


def engineering_receipt(config, path, base):
    """Bind all-arm checks to this configuration before manual training."""
    hashes = {}
    config_sha = digest(path)
    for name in config["arms"]:
        folder = base / "smoke" / name
        files = [base / "preflight" / f"{name}.json"] + [
            folder / f for f in ("complete.json", "audit.json", "grid.json", "grid-audit.json")
        ]
        pre, complete, audit, grid, replay = [json.loads(f.read_text()) for f in files]
        checkpoint = folder / f"checkpoint-{config['smoke_steps']:07d}.pt"
        findings = {
            "preflight": pre["passed"] is True and pre["config_sha256"] == config_sha,
            "complete": complete["engineering_only"] is True
            and complete["arm"] == name
            and complete["state"] == "complete"
            and complete["step"] == complete["budget"] == config["smoke_steps"]
            and complete["config_sha256"] == config_sha
            and complete["world_sha256"] == config["input_origin"]["world_sha256"]
            and digest(checkpoint) == complete["final_checkpoint_sha256"],
            "audit": audit["passed"] is True
            and audit["exposure_match"] is True
            and audit["arm"] == name
            and audit["step"] == complete["step"]
            and set(audit["panels"]) == set(PANELS)
            and all(
                p["passed"] and p["answer_and_eos_ids_exact"] for p in audit["panels"].values()
            ),
            "grid": grid["engineering_only"] is True
            and grid["arm"] == name
            and grid["step"] == complete["step"]
            and grid["config_sha256"] == config_sha
            and grid["model_sha256"] == complete["model_sha256"]
            and grid["world_sha256"] == config["input_origin"]["world_sha256"]
            and grid["panels_sha256"] == config["input_origin"]["panels_sha256"]
            and set(grid["metrics"]) == {"1", "2", "3", "4"}
            and all(set(m) == set(PANELS) for m in grid["metrics"].values()),
            "grid_audit": replay["passed"] is True
            and replay["model_sha256"] == complete["model_sha256"]
            and replay["grid_sha256"] == digest(folder / "grid.json")
            and set(replay["findings"]) == {f"r{r}/{p}" for r in range(1, 5) for p in PANELS}
            and all(v is True for v in replay["findings"].values()),
        }
        if not all(findings.values()):
            raise ValueError(f"Engineering check failed or config changed: {name}: {findings}")
        for filename in [*files, checkpoint]:
            hashes[str(filename.relative_to(base))] = digest(filename)
    return {"passed": True, "config_sha256": config_sha, "files": hashes, "utc": utc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.stage == "verify-baseline":
        receipt = {"baseline": verify_baseline(config), "input": verify_input(config), "utc": utc()}
        destination = args.base / "baseline-reuse.json"
        if destination.exists():
            raise FileExistsError(destination)
        write_json(destination, receipt)
        print(json.dumps(receipt))
        return
    if args.stage == "check-engineering":
        receipt = engineering_receipt(config, args.config, args.base)
        destination = args.base / "engineering-complete.json"
        if destination.exists():
            raise FileExistsError(destination)
        write_json(destination, receipt)
        return
    if args.arm not in config["arms"]:
        raise ValueError("Only the two new arms can execute; paper baseline is read-only")
    verify_input(config)
    if args.stage in {"preflight", "train"}:
        verify_baseline(config)
    # The user explicitly requested direct 300k training without short training
    # or reload gates. Optional engineering stages remain callable, but are not
    # silently inserted into the authorized training path.
    # The imported CLI reads the same file; attach the digest only inside the
    # preflight closure, not to scientific configuration or checkpoint identity.
    with installed():
        original = native.preflight

        def checked_preflight(value, name, base):
            original({**value, "_checked_config_sha256": digest(args.config)}, name, base)

        native.preflight = checked_preflight
        depth.main()


if __name__ == "__main__":
    os.umask(0o022)
    main()
