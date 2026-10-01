#!/usr/bin/env python3
"""Held-out prediction and paired-world summaries for the exploratory matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np

from llm_memory_editability.hebbian_future import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    auc,
    bootstrap_mean,
    digest,
    read,
    ridge_predict,
    unit,
    write,
)


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def real_report():
    cfg = read(CONFIG)
    facts = read(DATA / "real-facts.json")
    rows = read(RESULTS / "real/measurements.json")
    n = len(facts)
    assert [(r["fact"], r["view"]) for r in rows] == list(itertools.product(range(n), range(4)))
    refs = rows[::4]
    dev = np.array([r["role"] == "dev" for r in facts])
    test = ~dev
    query_dev = np.repeat(dev, 3)
    query_test = ~query_dev
    true_tokens = np.array([r["true_token"] for r in refs])
    ref_tokens = np.array([r["prediction_token"] for r in refs])
    query_rows = [r for r in rows if r["view"]]
    query_tokens = np.array([r["prediction_token"] for r in query_rows])
    failure = np.array([1 - r["strict_correct"] for r in query_rows], dtype=float)
    token_failure = (query_tokens != np.repeat(true_tokens, 3)).astype(float)
    relations = sorted({r["relation_id"] for r in facts})
    base = np.array(
        [
            [
                float(r["strict_correct"]),
                r["margin"],
                r["true_probability"],
                r["top_probabilities"][0],
                r["entropy"],
                len(r["prompt"]),
                *[float(r["relation"] == rel) for rel in relations],
            ]
            for r in refs
        ]
    )
    base = np.repeat(base, 3, axis=0)
    base = np.c_[base, [len(r["prompt"]) for r in query_rows]]
    outcomes = {"strict_generation_failure": failure, "first_token_failure": token_failure}
    base_preds = {
        label: ridge_predict(base[query_dev], y[query_dev], base, cfg["real"]["prediction_ridge"])
        for label, y in outcomes.items()
    }
    results, predictions, behavior = [], {}, []
    for label, y in outcomes.items():
        p = base_preds[label]
        results.append(
            {
                "method": "canonical_confidence",
                "outcome": label,
                "eval_brier": float(np.mean((p[query_test] - y[query_test]) ** 2)),
                "dev_brier": float(np.mean((p[query_dev] - y[query_dev]) ** 2)),
                "eval_auc": auc(y[query_test], p[query_test]),
                "brier_gain": 0.0,
            }
        )
        predictions[label + "-baseline"] = p.reshape(n, 3)
    unique = sorted(set(ref_tokens))
    lookup = {token: i for i, token in enumerate(unique)}
    onehot = np.zeros((n, len(unique)))
    onehot[np.arange(n), [lookup[t] for t in ref_tokens]] = 1
    features = np.load(RESULTS / "real/features.npz")
    own = np.repeat(np.arange(n), 3)

    def process(name, kernel):
        self_score = kernel[np.arange(n * 3), own]
        other = kernel.copy()
        other[np.arange(n * 3), own] = -np.inf
        best_other = other.max(1)
        chosen = kernel.argmax(1)
        estimate = kernel @ onehot
        pred_tokens = np.array(unique)[estimate.argmax(1)]
        excluded = kernel.copy()
        excluded[np.arange(n * 3), own] = 0
        excluded_tokens = np.array(unique)[(excluded @ onehot).argmax(1)]
        stats = np.c_[
            self_score,
            best_other,
            self_score - best_other,
            (chosen == own).astype(float),
            estimate.max(1),
            (pred_tokens == np.repeat(true_tokens, 3)).astype(float),
            (pred_tokens == np.repeat(ref_tokens, 3)).astype(float),
        ]
        for label, y in outcomes.items():
            design = np.c_[base, stats]
            p = ridge_predict(
                design[query_dev], y[query_dev], design, cfg["real"]["prediction_ridge"]
            )
            error = ((base_preds[label] - y) ** 2 - (p - y) ** 2).reshape(n, 3).mean(1)
            low, high = bootstrap_mean(error[test], cfg["seed"], cfg["real"]["bootstrap"])
            results.append(
                {
                    "method": name,
                    "outcome": label,
                    "dev_brier": float(np.mean((p[query_dev] - y[query_dev]) ** 2)),
                    "eval_brier": float(np.mean((p[query_test] - y[query_test]) ** 2)),
                    "eval_auc": auc(y[query_test], p[query_test]),
                    "brier_gain": float(error[test].mean()),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                }
            )
            predictions[label + "-" + name] = p.reshape(n, 3)
        errors = query_test & (token_failure == 1)
        behavior.append(
            {
                "method": name,
                "self_retrieval": float((chosen[query_test] == own[query_test]).mean()),
                "next_token_agreement": float(
                    (pred_tokens[query_test] == query_tokens[query_test]).mean()
                ),
                "wrong_token_agreement": float(
                    (pred_tokens[errors] == query_tokens[errors]).mean()
                ),
                "wrong_token_agreement_without_self": float(
                    (excluded_tokens[errors] == query_tokens[errors]).mean()
                ),
                "reference_copy_wrong_agreement": float(
                    (np.repeat(ref_tokens, 3)[errors] == query_tokens[errors]).mean()
                ),
                "reference_token_support_coverage": float(
                    np.isin(query_tokens[query_test], unique).mean()
                ),
            }
        )

    for layer in cfg["real"]["layers"]:
        for kind in ("x", "phi"):
            f = features[f"{kind}-{layer}"]
            reference, query = unit(f[:, 0]), unit(f[:, 1:].reshape(n * 3, -1))
            cross, gram = query @ reference.T, reference @ reference.T
            process(f"{kind}-L{layer}-cosine", cross)
            if kind == "phi":
                eigen, vectors = np.linalg.eigh(gram)
                for ridge in cfg["real"]["kernel_ridges"]:
                    kernel = (cross @ vectors / (np.maximum(eigen, 0) + ridge)) @ vectors.T
                    process(f"phi-L{layer}-ridge{ridge}", kernel)
        print("analyzed real layer", layer, flush=True)
    by_view = []
    for view in range(4):
        vr = [r for r in rows if r["view"] == view and r["role"] == "eval"]
        by_view.append(
            {
                "view": view,
                "n": len(vr),
                "strict_accuracy": float(np.mean([r["strict_correct"] for r in vr])),
                "first_token_accuracy": float(
                    np.mean([r["prediction_token"] == r["true_token"] for r in vr])
                ),
            }
        )
    csv_file(ART / "real-prediction.csv", results)
    csv_file(ART / "real-token-agreement.csv", behavior)
    csv_file(ART / "real-views.csv", by_view)
    np.savez_compressed(RESULTS / "real/predictions.npz", **predictions)
    write(
        ART / "real-summary.json",
        {
            "rows": results,
            "behavior": behavior,
            "by_view": by_view,
            "scope": (
                "All 20 kernel variants are exploratory; canonical memories of eval subjects "
                "are available, query outcomes are held out. CIs are per-fact, unadjusted; "
                "no eval-selected winner is confirmatory."
            ),
        },
    )


def chain_report():
    cfg = read(CONFIG)
    summaries, noise_rows, rehearsal_rows, paired = [], [], [], []
    complete = []
    for directory in sorted((RESULTS / "chains").glob("w*-r*-d*")):
        if not (directory / "complete.json").exists():
            continue
        meta = read(directory / "complete.json")
        complete.append(meta)
        key = {k: meta[k] for k in ("world", "rho", "depth")}
        for row in read(directory / "learning.json"):
            summaries.append({**key, "step": row["step"], **row["summary"]})
        for row in read(directory / "robustness.json"):
            noise_rows.append(
                {**key, "noise": row["noise"], "noise_seed": row["noise_seed"], **row["summary"]}
            )
        for arm in read(directory / "rehearsals.json"):
            for row in arm["trajectory"]:
                rehearsal_rows.append(
                    {**key, "arm": arm["arm"], "step": row["step"], **row["summary"]}
                )
    endpoints = [r for r in summaries if r["step"] == cfg["chains"]["steps"]]
    metrics = [
        "member_accuracy",
        "root_accuracy",
        "home_accuracy",
        "held_composite",
        "held_conflict_composite",
        "held_common_conflict_composite",
        "held_two_step",
        "held_both_known_composite",
        "held_conflict_home_copy",
    ]
    aggregates = []
    for depth, rho in itertools.product(cfg["chains"]["layers"], cfg["chains"]["correlations"]):
        group = [r for r in endpoints if r["depth"] == depth and r["rho"] == rho]
        if group:
            aggregates.append(
                {
                    "depth": depth,
                    "rho": rho,
                    "worlds": len(group),
                    **{
                        k: float(np.mean([r[k] for r in group if r[k] is not None]))
                        if any(r[k] is not None for r in group)
                        else None
                        for k in metrics
                    },
                }
            )
    for depth in cfg["chains"]["layers"]:
        for metric in metrics:
            effects = []
            for world in cfg["chains"]["worlds"]:
                group = {
                    r["rho"]: r for r in endpoints if r["world"] == world and r["depth"] == depth
                }
                if (
                    0.25 in group
                    and 0.75 in group
                    and all(group[r][metric] is not None for r in (0.25, 0.75))
                ):
                    effects.append(group[0.25][metric] - group[0.75][metric])
            if effects:
                paired.append(
                    {
                        "depth": depth,
                        "metric": metric,
                        "contrast": "rho0.25-rho0.75",
                        "mean": float(np.mean(effects)),
                        "world_effects": effects,
                    }
                )
    csv_file(ART / "chain-learning.csv", summaries)
    csv_file(ART / "chain-robustness.csv", noise_rows)
    csv_file(ART / "chain-rehearsal.csv", rehearsal_rows)
    csv_file(ART / "chain-endpoints.csv", aggregates)
    write(
        ART / "chain-summary.json",
        {
            "completed": len(complete),
            "expected": 24,
            "aggregates": aggregates,
            "paired_effects": paired,
            "runtime_seconds_sum": sum(m["seconds"] for m in complete),
            "training_input_tokens": sum(m["training_input_tokens"] for m in complete),
            "rehearsal_input_tokens": sum(m["rehearsal_input_tokens"] for m in complete),
            "training_flops_estimate": sum(m["training_flops_estimate"] for m in complete),
        },
    )


def plot():
    os.environ.setdefault("MPLCONFIGDIR", str(ART / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    real = read(ART / "real-summary.json")
    selected = [
        r
        for r in real["rows"]
        if r["outcome"] == "strict_generation_failure" and r["method"] != "canonical_confidence"
    ]
    axes[0].barh(range(len(selected)), [r["brier_gain"] for r in selected])
    axes[0].set_yticks(range(len(selected)), [r["method"] for r in selected], fontsize=6)
    axes[0].axvline(0, color="black", lw=0.7)
    axes[0].set_title("Held-out Brier improvement")
    chain = read(ART / "chain-summary.json")
    for depth in read(CONFIG)["chains"]["layers"]:
        rows = [r for r in chain["aggregates"] if r["depth"] == depth]
        for axis, metric in zip(
            axes[1:], ["held_composite", "held_common_conflict_composite"], strict=True
        ):
            axis.plot(
                [r["rho"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                label=f"{depth} layers",
            )
            axis.set_ylim(0, 1)
            axis.set_xlabel("Training home/HQ agreement")
            axis.legend()
    axes[1].set_title("All held-out composition")
    axes[2].set_title("Same always-conflicting people")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ART / ("overview." + suffix), dpi=180)
    plt.close(fig)


def audit():
    lock = read(ART / "lock.json")
    checked = {}
    for name, value in lock["files"].items():
        # The immutable execution snapshot remains available even as live docs evolve.
        checked[name] = digest(ART / "source" / name) == value
    jobs = list((RESULTS / "chains").glob("w*-r*-d*/complete.json"))
    groups = {}
    for path in jobs:
        meta = read(path)
        groups.setdefault((meta["world"], meta["depth"]), set()).add(meta["initial_hash"])
        for name, value in meta["files"].items():
            checked[str(path.parent / name)] = digest(path.parent / name) == value
    checked["all_24_chain_runs"] = len(jobs) == 24
    checked["paired_initial_weights"] = all(len(hashes) == 1 for hashes in groups.values())
    for name, value in read(RESULTS / "real/complete.json")["files"].items():
        checked["real/" + name] = digest(RESULTS / "real" / name) == value
    write(ART / "audit.json", {"passed": all(checked.values()), "checks": checked})
    assert all(checked.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["real", "chains", "plot", "audit"])
    args = parser.parse_args()
    {"real": real_report, "chains": chain_report, "plot": plot, "audit": audit}[args.stage]()
