#!/usr/bin/env python3
"""Aggregate and independently audit every depth trajectory and intervention."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from llm_memory_editability.twohop_depth import (
    ART,
    DATA,
    ENTITY,
    EOS,
    RESULTS,
    ROOT,
    audit_world,
    build_model,
    digest,
    evaluate,
    read,
    write,
)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate():
    lock = read(ART / "lock.json")
    cfg = lock["config"]
    curves, flow = [], []
    for job in lock["jobs"]:
        out = RESULTS / job["name"]
        w = dict(np.load(DATA / f"world-{job['world']}.npz"))
        for step in cfg["nodes"]:
            if not (out / f"node-{step}.json").exists():
                continue
            node = read(out / f"node-{step}.json")
            common = {
                "run": job["name"],
                "world": job["world"],
                "seed": job["seed"],
                "arch": job["arch"]["name"],
                "step": step,
            }
            curves.append(
                {
                    **common,
                    "parameters": lock["parameters"][common["arch"]],
                    **node["metrics"],
                    "training_seconds": node["training_seconds"],
                    "training_flops_estimate": node["training_flops_estimate"],
                }
            )
            p = np.load(out / f"predictions-{step}.npz")
            d = np.load(out / f"diagnostics-{step}.npz")
            cases = w["cases"]
            atom_ok = (p["atomic_pred"] == w["atomic_y"]) & (p["atomic_eos"] == EOS)
            r = cfg["relations"]
            known = np.ones(len(cases), dtype=bool)
            for subjects, rels in (
                (cases[:, 1], cases[:, 2]),
                (cases[:, 4], cases[:, 3]),
                (cases[:, 6], cases[:, 2]),
                (cases[:, 7], cases[:, 3]),
            ):
                known &= atom_ok[subjects * r + rels]
            clean_ok = d["clean_pred"] == d["old_y"]
            eligible = known & clean_ok
            for layer in range(job["arch"]["layers"] + 1):
                for kind in ("bridge", "same_bridge", "random", "source", "prefix", "identity"):
                    pred = d[f"{kind}_{layer}_pred"]
                    alt = pred == d["new_y"]
                    gain = alt.astype(float) - (d["clean_pred"] == d["new_y"]).astype(float)
                    flow.append(
                        {
                            **common,
                            "layer": layer,
                            "kind": kind,
                            "n": len(cases),
                            "clean_accuracy": float(clean_ok.mean()),
                            "known_n": int(known.sum()),
                            "eligible_n": int(eligible.sum()),
                            "counterfactual_accuracy": float(alt.mean()),
                            "counterfactual_gain": float(gain.mean()),
                            "counterfactual_given_known": float(alt[known].mean())
                            if known.any()
                            else None,
                            "counterfactual_given_eligible": float(alt[eligible].mean())
                            if eligible.any()
                            else None,
                            "old_retention": float((pred == d["old_y"]).mean()),
                            "wrong_relation_accuracy": float((pred == d["wrong_y"]).mean()),
                            "margin_gain": float(
                                (d[f"{kind}_{layer}_margin"] - d["clean_margin"]).mean()
                            ),
                            "bridge_lens_pos2": float(
                                (d[f"lens_bridge_pos2_{layer}"] == ENTITY + cases[:, 4]).mean()
                            ),
                            "bridge_lens_pos3": float(
                                (d[f"lens_query_pos3_{layer}"] == ENTITY + cases[:, 4]).mean()
                            ),
                            "answer_lens_pos3": float(
                                (d[f"lens_query_pos3_{layer}"] == d["old_y"]).mean()
                            ),
                        }
                    )
    save_csv(ART / "learning.csv", curves)
    save_csv(ART / "interventions.csv", flow)
    summary = []
    for step in (cfg["primary_step"], cfg["steps"]):
        for arch in cfg["architectures"]:
            rows = [row for row in curves if row["step"] == step and row["arch"] == arch["name"]]
            if not rows:
                continue
            summary.append(
                {
                    "step": step,
                    "arch": arch["name"],
                    "runs": len(rows),
                    "parameters": lock["parameters"][arch["name"]],
                    **{
                        k: float(np.mean([row[k] for row in rows]))
                        for k in (
                            "atomic_exact",
                            "train_exact",
                            "test_exact",
                            "two_call_exact",
                            "test_both_atoms_n",
                        )
                    },
                    "test_min": min(row["test_exact"] for row in rows),
                    "test_max": max(row["test_exact"] for row in rows),
                    "test_by_run": {row["run"]: row["test_exact"] for row in rows},
                }
            )
    write(
        ART / "summary.json",
        {"curves": len(curves), "intervention_rows": len(flow), "summary": summary},
    )
    plot(curves, flow, cfg)
    print(__import__("json").dumps(summary, ensure_ascii=False, indent=2))


def plot(curves, flow, cfg):
    os.environ.setdefault("MPLCONFIGDIR", str(ART / "mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, step in zip(axes, (cfg["primary_step"], cfg["steps"]), strict=True):
        for key, label in (
            ("atomic_exact", "One hop"),
            ("test_exact", "Held-out two hop"),
            ("two_call_exact", "Two calls"),
        ):
            points = []
            for depth in (1, 2, 3, 4, 6):
                vals = [
                    v[key] for v in curves if v["arch"] == f"d{depth}w128" and v["step"] == step
                ]
                points.append(np.mean(vals) if vals else np.nan)
            ax.plot([1, 2, 3, 4, 6], points, marker="o", label=label)
        ax.set(title=f"{step} optimizer steps", xlabel="Transformer blocks", ylim=(-0.02, 1.03))
        ax.set_xticks([1, 2, 3, 4, 6])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Exact answer + EOS accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ART / "depth-accuracy.png", dpi=180)
    fig.savefig(ART / "depth-accuracy.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, depth in zip(axes, (2, 4, 6), strict=True):
        matrix = np.zeros((depth + 1, len(cfg["nodes"])))
        for layer in range(depth + 1):
            for col, step in enumerate(cfg["nodes"]):
                vals = [
                    row["counterfactual_gain"]
                    for row in flow
                    if row["arch"] == f"d{depth}w128"
                    and row["step"] == step
                    and row["layer"] == layer
                    and row["kind"] == "bridge"
                ]
                matrix[layer, col] = np.mean(vals) if vals else np.nan
        im = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set(title=f"{depth} blocks", xlabel="Training step")
        ax.set_xticks(range(len(cfg["nodes"])), cfg["nodes"], rotation=60)
    axes[0].set_ylabel("Patch after block (0 = embeddings)")
    fig.colorbar(im, ax=axes, label="Counterfactual answer gain", shrink=0.8)
    fig.savefig(ART / "bridge-formation.png", dpi=180, bbox_inches="tight")
    fig.savefig(ART / "bridge-formation.pdf", bbox_inches="tight")
    plt.close(fig)


def audit(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    lock = read(ART / "lock.json")
    cfg = lock["config"]
    checks, reloads = [], []
    for path, expected in lock["files"].items():
        frozen = ROOT / path if path.startswith("data/") else RESULTS / "execution-source" / path
        assert digest(frozen) == expected, path
        checks.append(path)
    for job in lock["jobs"]:
        out = RESULTS / job["name"]
        complete = read(out / "complete.json")
        assert complete["step"] == cfg["steps"] and complete["nodes"] == cfg["nodes"]
        w = dict(np.load(DATA / f"world-{job['world']}.npz"))
        audit_world(w, cfg)
        for filename, expected in complete["files"].items():
            assert digest(out / filename) == expected
            checks.append(job["name"] + "/" + filename)
        for step in cfg["nodes"]:
            p = np.load(out / f"predictions-{step}.npz")
            d = np.load(out / f"diagnostics-{step}.npz")
            metrics = read(out / f"node-{step}.json")["metrics"]
            for name, key, mask in (
                ("atomic", "atomic", slice(None)),
                ("train", "composite", w["train_mask"]),
                ("test", "composite", ~w["train_mask"]),
            ):
                acc = np.mean(
                    (p[name + "_pred"] == w[key + "_y"][mask]) & (p[name + "_eos"] == EOS)
                )
                assert acc == metrics[name + "_exact"]
            for layer in range(job["arch"]["layers"] + 1):
                assert np.max(d[f"identity_{layer}_max_delta"]) == 0
            for kind in ("bridge", "same_bridge", "random", "source", "prefix"):
                assert np.max(d[f"{kind}_{job['arch']['layers']}_max_delta"]) == 0
        ckpt = torch.load(out / f"model-{cfg['steps']}.pt", map_location=device, weights_only=False)
        model = build_model(job["arch"], cfg).to(device).eval()
        model.load_state_dict(ckpt["model"])
        actual_metrics, actual = evaluate(model, w, cfg, device)
        stored = np.load(out / f"predictions-{cfg['steps']}.npz")
        for key in actual:
            if "logp" not in key:
                np.testing.assert_array_equal(actual[key], stored[key])
        reloads.append(
            {"run": job["name"], "metrics": actual_metrics, "predictions_identical": True}
        )
        print(f"Reload verified: {job['name']}", flush=True)
        del model
    write(
        ART / "audit.json",
        {
            "complete": True,
            "runs": len(reloads),
            "nodes": len(reloads) * len(cfg["nodes"]),
            "hashed_files": len(checks),
            "all_identity_and_terminal_controls_exact": True,
            "endpoint_reloads": reloads,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["aggregate", "audit"])
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    if args.stage == "aggregate":
        aggregate()
    else:
        audit(args.device)
