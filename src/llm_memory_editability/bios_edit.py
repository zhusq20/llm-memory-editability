"""Fixed-budget development edits with held-out retention and full-answer scoring."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_branch import attach_branch, build_bases, remove_branch
from .bios_data import N_BASE, N_QUERIES, load_world, paired_edit, rng_for, write_json
from .bios_model import CausalLM, ModelConfig, matmul_flops, select_parameters
from .bios_train import code_fingerprint, evaluate, precision, tensor_data

EDIT_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def score_edit(world, pair, target, arrays, old_correct):
    correct = arrays["correct"]
    e, d = pair["E"], pair["D"]
    root = e[world.relation[e] == 1]
    members = e[world.relation[e] == 2]
    unseen = np.zeros(N_QUERIES, dtype=bool)
    unseen[pair["heldout"]] = True
    retention, full_retention = {}, {}
    for g in range(4):
        ids = (pair["strata"] == g) & old_correct
        full_retention[str(g)] = {
            "known": int(ids.sum()),
            "broken": int((ids & ~correct).sum()),
            "rate": float((~correct[ids]).mean()) if ids.any() else None,
        }
        held = ids & unseen
        retention[str(g)] = {
            "known": int(held.sum()),
            "broken": int((held & ~correct).sum()),
            "rate": float((~correct[held]).mean()) if held.any() else None,
        }
    for name, base in (("2_base", True), ("2_derived", False)):
        ids = (
            (pair["strata"] == 2) & old_correct & unseen & ((np.arange(N_QUERIES) < N_BASE) == base)
        )
        retention[name] = {
            "known": int(ids.sum()),
            "broken": int((ids & ~correct).sum()),
            "rate": float((~correct[ids]).mean()) if ids.any() else None,
        }
    rates = [v["rate"] for v in retention.values()]
    root_acc, member_acc, d_acc = (float(correct[ids].mean()) for ids in (root, members, d))
    joint = (
        None
        if any(rate is None for rate in rates)
        else bool(
            root_acc == 1
            and member_acc >= 0.95
            and d_acc >= 0.95
            and all(rate <= 0.01 for rate in rates)
        )
    )
    return {
        "E": float(correct[e].mean()),
        "E_root": root_acc,
        "E_member": member_acc,
        "D": d_acc,
        "U_accuracy": float(correct[pair["strata"] >= 0].mean()),
        "U_heldout_destruction": retention,
        "U_full_destruction": full_retention,
        "joint_pass": joint,
        "E_target_value_nll": float(arrays["value_nll"][e].mean()),
    }


def reference_logits(model, data, ids, device):
    result = []
    with torch.no_grad(), precision(device):
        for begin in range(0, len(ids), 256):
            subset = torch.as_tensor(ids[begin : begin + 256], device=device)
            result.append(model(data["tokens"][subset], data["positions"][subset]).detach())
    return torch.cat(result)


@torch.no_grad()
def capture_down(model, tokens, positions, layer):
    captured = {}

    def capture(module, inputs, output):
        captured.update(
            z=inputs[0].float().clone(),
            h=output.float().clone(),
            weight=module.weight.detach().float().clone(),
        )

    handle = model.blocks[layer].mlp.down.register_forward_hook(capture)
    try:
        model(tokens, positions)
    finally:
        handle.remove()
    return captured


def edit_case(
    model,
    baseline,
    world,
    data_old,
    pair,
    kind,
    scope,
    args,
    old_correct,
    branch_kind=None,
    basis=None,
):
    suffix = "" if branch_kind is None else f"-{branch_kind}-branch"
    out = Path(args.output) / f"support-{pair['support']}-{kind}-{scope}{suffix}"
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    device = data_old["tokens"].device
    model.load_state_dict(baseline)
    selected = select_parameters(model, scope, args.window)
    branch, handles, zero_error = None, [], None
    if basis is not None:
        probe = torch.as_tensor(pair["E"][:32], device=device)
        with torch.no_grad(), precision(device):
            before = model(data_old["tokens"][probe], data_old["positions"][probe])
        branch, handles = attach_branch(model, basis, layer=args.window + 1)
        selected.extend(branch.parameters())
        with torch.no_grad(), precision(device):
            after = model(data_old["tokens"][probe], data_old["positions"][probe])
        zero_error = (before - after).abs().max().item()
        if zero_error != 0:
            raise ValueError("Branch was not function-preserving at initialization")
    target = pair[kind]
    data_new = tensor_data(world, device, target)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_start = time.perf_counter()
    calibration = None
    if scope == "down":
        calibration_ids = torch.as_tensor(
            np.concatenate([pair["E"][:32], pair["replay"][:32]]), device=device
        )
        calibration = capture_down(
            model,
            data_old["tokens"][calibration_ids],
            data_old["positions"][calibration_ids],
            args.window + 1,
        )
    old_logits = reference_logits(model, data_old, pair["replay"], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    basis_seconds = getattr(args, "branch_preparation_seconds", 0.0) if basis else 0.0
    setup_flops = matmul_flops(model.config, len(pair["replay"]), backward=False)
    optimizer = torch.optim.AdamW(
        selected, lr=args.lr, weight_decay=0.1, fused=device.type == "cuda"
    )
    sample_rng = rng_for(world.seed, 41, pair["support"])
    e_choices = sample_rng.integers(0, len(pair["E"]), size=(args.steps, 128))
    r_choices = sample_rng.integers(0, len(pair["replay"]), size=(args.steps, 128))
    timeline, train_seconds = [], 0.0
    case_config = {
        "scope": scope,
        "branch": branch_kind,
        "branch_parameters": sum(p.numel() for p in branch.parameters()) if branch else 0,
        "branch_zero_initialization_max_logit_difference": zero_error,
        "kind": kind,
        "support": pair["support"],
        "selection": pair["selection"],
        "steps": args.steps,
        "lr": args.lr,
        "retention_kl_weight": args.retention,
        "window_zero_based": list(range(args.window, args.window + 3)),
        "trainable_parameters": sum(p.numel() for p in selected),
        "setup_seconds": setup_seconds,
        "shared_basis_preparation_seconds": basis_seconds,
        "preparation_accounting": (
            "Basis construction is executed once per old checkpoint. Standalone cold-start time "
            "below charges that entire common construction; do not sum it across reused branches."
        ),
        "setup_matmul_flops_estimate": setup_flops,
        "flops_limitation": (
            "backward dense matmul upper approximation; not exact frozen-scope operation count"
        ),
    }
    write_json(out / "config.json", case_config)
    np.savez_compressed(
        out / "sets.npz",
        E=pair["E"],
        D=pair["D"],
        strata=pair["strata"],
        replay=pair["replay"],
        heldout=pair["heldout"],
        target=target,
        old_correct=old_correct,
        edit_sampling=e_choices,
        replay_sampling=r_choices,
    )
    checkpoints = sorted({s for s in EDIT_CHECKPOINTS if s <= args.steps} | {args.steps})
    for step in range(args.steps + 1):
        if step in checkpoints:
            eval_start = time.perf_counter()
            _, arrays = evaluate(model, data_old, world, answers=target)
            metrics = score_edit(world, pair, target, arrays, old_correct)
            np.savez_compressed(out / f"predictions-{step}.npz", **arrays)
            point = {
                "step": step,
                **metrics,
                "edit_seconds": train_seconds,
                "setup_plus_edit_seconds": setup_seconds + train_seconds,
                "cold_start_including_basis_seconds": basis_seconds + setup_seconds + train_seconds,
                "setup_plus_edit_matmul_flops_estimate": setup_flops
                + step * matmul_flops(model.config, 256),
                "target_supervised_tokens": step * 256,
                "replay_supervised_tokens": step * 256,
                "eval_seconds": time.perf_counter() - eval_start,
            }
            if calibration is not None:
                current = capture_down(
                    model,
                    data_old["tokens"][calibration_ids],
                    data_old["positions"][calibration_ids],
                    args.window + 1,
                )
                observed = current["h"] - calibration["h"]
                predicted = F.linear(calibration["z"], current["weight"] - calibration["weight"])
                point["down_calibration"] = {
                    "feature_max_change": (current["z"] - calibration["z"]).abs().max().item(),
                    "delta_h_max_error": (observed - predicted).abs().max().item(),
                    "delta_h_rms_error": (observed - predicted).square().mean().sqrt().item(),
                    "fixed_prefix_tokens": int(
                        calibration["z"].shape[0] * calibration["z"].shape[1]
                    ),
                }
            timeline.append(point)
            write_json(out / "trajectory.json", timeline)
            print(
                json.dumps(
                    {
                        "event": "edit",
                        "case": out.name,
                        "step": step,
                        "E": metrics["E"],
                        "D": metrics["D"],
                        "joint": metrics["joint_pass"],
                    }
                ),
                flush=True,
            )
        if step == args.steps:
            break
        model.train()
        ei = torch.as_tensor(pair["E"][e_choices[step]], device=device)
        ri = torch.as_tensor(pair["replay"][r_choices[step]], device=device)
        reference = old_logits[torch.as_tensor(r_choices[step], device=device)].float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            logits_e = model(data_new["tokens"][ei], data_new["positions"][ei]).float()
            logits_r = model(data_old["tokens"][ri], data_old["positions"][ri]).float()
            ce = F.cross_entropy(
                logits_e.reshape(-1, world.vocab_size), data_new["labels"][ei].reshape(-1)
            )
            kl = (
                F.kl_div(
                    F.log_softmax(logits_r, dim=-1), F.softmax(reference, dim=-1), reduction="none"
                )
                .sum(-1)
                .mean()
            )
            loss = ce + args.retention * kl
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(selected, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            write_json(out / "failure.json", {"step": step, "reason": "nonfinite loss or gradient"})
            raise FloatingPointError("Nonfinite edit")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - start
    passed = [p for p in timeline if p["joint_pass"] is True]
    result = {
        **case_config,
        "status": "complete",
        "final": timeline[-1],
        "first_observed_joint_step": passed[0]["step"] if passed else None,
        "budget_censored": (not bool(passed) if timeline[-1]["joint_pass"] is not None else None),
        "joint_evaluable": timeline[-1]["joint_pass"] is not None,
    }
    write_json(out / "complete.json", result)
    if handles:
        remove_branch(model, handles)
    return result


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    world = load_world(args.world)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CausalLM(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    data = tensor_data(world, device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "run.json", {**vars(args), "world_step": checkpoint["step"], **code_fingerprint()}
    )
    _, old_arrays = evaluate(model, data, world)
    bases = None
    if args.branches:
        bases, operation_checks = build_bases(model, world, data, layer=args.window + 1)
        write_json(out / "branch-operation-checks.json", operation_checks)
        args.branch_preparation_seconds = operation_checks.get("construction_seconds", 0.0)
    results = []
    for support in args.supports:
        pair = paired_edit(world, support=support, k=1)
        scopes = getattr(args, "scopes", None)
        if scopes is None:
            scopes = ["mlp", "all"] + (["down"] if support == 0 else [])
        for scope in scopes:
            for kind in ("coherent", "exception"):
                results.append(
                    edit_case(
                        model,
                        checkpoint["model"],
                        world,
                        data,
                        pair,
                        kind,
                        scope,
                        args,
                        old_arrays["correct"],
                    )
                )
        if support == 0 and bases is not None:
            for branch_kind, basis in bases.items():
                for kind in ("coherent", "exception"):
                    results.append(
                        edit_case(
                            model,
                            checkpoint["model"],
                            world,
                            data,
                            pair,
                            kind,
                            "mlp",
                            args,
                            old_arrays["correct"],
                            branch_kind,
                            basis,
                        )
                    )
    write_json(out / "complete.json", {"status": "complete", "cases": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--supports", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--scopes", choices=["mlp", "all", "down"], nargs="+")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--retention", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--branches", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
