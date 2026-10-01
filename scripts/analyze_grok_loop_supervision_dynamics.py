#!/usr/bin/env python3
"""Finite-horizon state diagnostics on every OOD query, without answer input."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grok_loop_supervision import (
    digest,
    evaluate,
    from_state,
    load_world,
)


@torch.no_grad()
def measure(checkpoint, rows, out, reference, repeats=64):
    state = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    model = from_state(state, "cuda:0").eval()
    # Preserve native length-four attention shape; mask the answer input.
    tokens = torch.zeros((len(rows), 4), device="cuda:0", dtype=torch.long)
    tokens[:, :3] = torch.as_tensor(rows[:, :3], device="cuda:0")
    target = torch.as_tensor(rows[:, -1], device="cuda:0")
    positions = torch.tensor([[2, 3]], device="cuda:0").expand(len(rows), -1)
    previous = model.token(tokens) + model.position(torch.arange(4, device="cuda:0"))
    previous_ln = model.ln_final(previous)
    saved, aggregate, audits = [], [], []
    anchor = None
    with np.load(reference) as ref:
        for r, x in enumerate(model.states(tokens, repeats), 1):
            z = model.readout(x, positions)[:, 0]
            ln = model.ln_final(x)
            # Three causal prefix positions only; padded suffix is excluded.
            raw_step = (x[:, :3] - previous[:, :3]).square().mean(-1).sqrt()
            raw_norm = x[:, :3].square().mean(-1).sqrt()
            ln_step = (ln[:, :3] - previous_ln[:, :3]).square().mean(-1).sqrt()
            centered = x[:, :3] - x[:, :3].mean(-1, keepdim=True)
            old_centered = previous[:, :3] - previous[:, :3].mean(-1, keepdim=True)
            unit = torch.nn.functional.normalize(centered, dim=-1)
            old_unit = torch.nn.functional.normalize(old_centered, dim=-1)
            unit_step = (unit - old_unit).square().sum(-1).sqrt()
            target_score = z.gather(1, target[:, None]).squeeze(1)
            alternatives = z.clone().scatter_(1, target[:, None], -torch.inf)
            margin = target_score - alternatives.max(-1).values
            pred = z.argmax(-1)
            datum = {
                "raw_step_by_position": raw_step.cpu().numpy(),
                "raw_norm_by_position": raw_norm.cpu().numpy(),
                "relative_step_by_position": (raw_step / raw_norm.clamp_min(1e-12)).cpu().numpy(),
                "ln_step_by_position": ln_step.cpu().numpy(),
                "unit_step_by_position": unit_step.cpu().numpy(),
                "answer": pred.cpu().numpy(),
                "margin": margin.cpu().numpy(),
            }
            saved.append(datum)
            raw_all = raw_step.square().mean(-1).sqrt()
            norm_all = raw_norm.square().mean(-1).sqrt()
            aggregate.append(
                {
                    "repeat": r,
                    "n": len(rows),
                    "raw_step_rms_mean": raw_all.mean().item(),
                    "raw_step_rms_min": raw_all.min().item(),
                    "raw_step_rms_max": raw_all.max().item(),
                    "raw_norm_rms_mean": norm_all.mean().item(),
                    "relative_step_mean": (raw_all / norm_all).mean().item(),
                    "ln_step_rms_mean": ln_step.square().mean(-1).sqrt().mean().item(),
                    "unit_step_mean": unit_step.mean().item(),
                    "answer_accuracy": (pred == target).float().mean().item(),
                    "margin_mean": margin.mean().item(),
                }
            )
            if r <= 16:
                prefix = f"ood_composite_r{r}_"
                if not np.array_equal(datum["answer"], ref[prefix + "answer"]):
                    raise ValueError(f"Causal-prefix answer mismatch R{r}")
                error = float(np.abs(datum["margin"] - ref[prefix + "margin"]).max())
                if error > 2e-5:
                    raise ValueError(f"Prefix margin mismatch R{r}: {error}")
                audits.append({"repeat": r, "answer_equal": True, "max_margin_error": error})
            if r == 32:
                anchor = x[:, :3].clone()
            previous, previous_ln = x, ln
    if anchor is None:
        raise ValueError("Expected at least 32 loops")
    drift = (x[:, :3] - anchor).square().mean((1, 2)).sqrt().cpu().numpy()
    np.savez_compressed(
        out.with_suffix(".npz"),
        **{k: np.stack([s[k] for s in saved]) for k in saved[0]},
        query=rows[:, :3],
        target=rows[:, -1],
        raw_drift_32_to_64=drift,
    )
    complete_scores = {}
    for r in (32, 64):
        scores, pred = evaluate(model, rows, "cuda:0", r)
        if not np.array_equal(pred["answer"], saved[r - 1]["answer"]):
            raise ValueError(f"Native answer mismatch R{r}")
        np.savez_compressed(out.with_name(out.name + f"-generation-r{r}.npz"), **pred)
        complete_scores[str(r)] = scores
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "reference": str(reference),
        "reference_sha256": digest(reference),
        "created_utc": utc(),
        "source_sha256": digest(__file__),
        "finite_horizon_only": True,
        "prefix_positions": [0, 1, 2],
        "raw_drift_32_to_64_mean": float(drift.mean()),
        "native_audit": audits,
        "full_generation": complete_scores,
        "trajectory": aggregate,
    }
    write_json(out.with_suffix(".json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, endpoints, sources = [], [], set()
    for run_id, spec in cfg["runs"].items():
        directory = Path(cfg["output_root"]) / run_id
        jobs = [
            (
                spec["arm"],
                directory / "latest.pt",
                directory / f"predictions-{cfg['base']['steps']:07d}.npz",
            )
        ]
        if spec["source"] not in sources:
            sources.add(spec["source"])
            jobs.insert(
                0,
                (
                    "source",
                    Path(spec["source"]) / "latest.pt",
                    directory / "predictions-0000000.npz",
                ),
            )
        rows = load_world(spec["source"])["ood_composite"]
        for arm, checkpoint, reference in jobs:
            name = f"w{spec['world']}-i{spec['initialization']}-{arm}"
            out = output / name
            if out.with_suffix(".json").exists():
                result = json.loads(out.with_suffix(".json").read_text())
                if result["source_sha256"] != digest(__file__) or result[
                    "checkpoint_sha256"
                ] != digest(checkpoint):
                    raise ValueError("Changed analysis; use a new output directory")
            else:
                result = measure(checkpoint, rows, out, reference)
            tag = {"world": spec["world"], "initialization": spec["initialization"], "arm": arm}
            all_rows.extend([{**tag, **r} for r in result["trajectory"]])
            endpoints.append(
                {
                    **tag,
                    **result["trajectory"][-1],
                    "raw_drift_32_to_64_mean": result["raw_drift_32_to_64_mean"],
                    "full_generation": result["full_generation"],
                }
            )
            print(json.dumps({"model": name, "state": "complete"}), flush=True)
    with (output / "trajectory.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(output / "summary.json", {"endpoints": endpoints, "models": len(endpoints)})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    fields = ["raw_step_rms_mean", "raw_norm_rms_mean", "ln_step_rms_mean", "answer_accuracy"]
    titles = [
        "Absolute residual step (RMS)",
        "Residual magnitude (RMS)",
        "LayerNorm output step (RMS)",
        "OOD answer accuracy",
    ]
    for arm, color in (("source", "#777777"), ("single", "#b54a42"), ("multi", "#2673aa")):
        for ax, field in zip(axes, fields, strict=True):
            values = [
                np.mean([r[field] for r in all_rows if r["arm"] == arm and r["repeat"] == t])
                for t in range(1, 65)
            ]
            ax.plot(range(1, 65), values, color=color, label=arm)
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Evaluation loops")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[2].set_yscale("log")
    axes[3].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output / "dynamics.png", dpi=180)
    fig.savefig(output / "dynamics.pdf")


if __name__ == "__main__":
    main()
