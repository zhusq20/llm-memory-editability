#!/usr/bin/env python3
"""Predeclared paired summaries, conditional subsets, bootstrap and four figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_gated_depth import write_csv  # noqa: E402

from llm_memory_editability.gated_depth_data import load  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402


def transitions(correct, loops):
    old = correct[:, loops.index(5)]
    return {
        "retain": np.array(
            [correct[old, j].mean() if old.any() else np.nan for j in range(len(loops))]
        ),
        "recover": np.array(
            [correct[~old, j].mean() if (~old).any() else np.nan for j in range(len(loops))]
        ),
    }


def template_bootstrap(values, rows, replicates, seed):
    """Cluster by complete relation sequence, conditional on this graph and model."""
    unique = sorted({tuple(row["relation_ids"]) for row in rows})
    groups = np.asarray(
        [
            np.mean(
                [
                    value
                    for value, row in zip(values, rows, strict=True)
                    if tuple(row["relation_ids"]) == key
                ]
            )
            for key in unique
        ]
    )
    rng = np.random.default_rng(seed)
    means = groups[rng.integers(len(groups), size=(replicates, len(groups)))].mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def common_known_mask(rows, atoms_correct, fact_index):
    mask = []
    for row in rows:
        edges = [
            fact_index[head, relation]
            for head, relation in zip(
                row["intermediate_entities"][:-1], row["relation_ids"], strict=True
            )
        ]
        mask.append(bool(atoms_correct[edges].all()))
    return np.array(mask)


def analyze(config, base):
    pools, _, metadata = load(base / "data")
    fact_index = {
        (row["start_entity"], row["relation_ids"][0]): i
        for i, row in enumerate(pools["facts_train"])
    }
    arms, seeds = list(config["arms"]), config["seeds"]
    loops = config["evaluation"]["loops"]
    robust = [loops.index(r) for r in config["evaluation"]["robust_loops"]]
    out = base / "analysis"
    out.mkdir(exist_ok=True)
    scores, differences, conditioned, runs = [], [], [], []
    cache = {}
    for seed in seeds:
        initialization, exposure = set(), set()
        for arm in arms:
            run = base / "train" / f"{arm}-seed{seed}"
            for name in ("complete", "audit", "evaluation", "cloud-audit"):
                assert json.loads((run / f"{name}.json").read_text())["passed"]
            complete = json.loads((run / "complete.json").read_text())
            initialization.add(complete["initial_backbone_sha256"])
            exposure.add(complete["stream_sha256"])
            runs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "steps": complete["steps"],
                    "best_step": complete["best_step"],
                    "parameters": complete["parameters"],
                    "input_tokens": complete["input_tokens"],
                    "matrix_training_flops": complete["training_matrix_flops_estimate"],
                    "wall_seconds": complete["wall_seconds"],
                    "peak_cuda_bytes": complete["peak_cuda_bytes"],
                }
            )
            for checkpoint in ("best", "last"):
                for name in (
                    "facts_train",
                    "eval_A_short",
                    "eval_A_long",
                    "eval_B_short",
                    "eval_B_long",
                ):
                    values = dict(np.load(run / f"evaluation-{checkpoint}" / f"{name}.npz"))
                    correct = values["prediction"] == values["labels"][:, None]
                    cache[seed, arm, checkpoint, name] = correct
                    rows = pools[name]
                    for k in sorted({row["hop_count"] for row in rows}):
                        mask = np.array([row["hop_count"] == k for row in rows])
                        selected = [row for row in rows if row["hop_count"] == k]
                        for j, r in enumerate(loops):
                            ci = template_bootstrap(
                                correct[mask, j],
                                selected,
                                config["evaluation"]["bootstrap_replicates"],
                                config["evaluation"]["bootstrap_seed"],
                            )
                            scores.append(
                                {
                                    "seed": seed,
                                    "arm": arm,
                                    "checkpoint": checkpoint,
                                    "split": name,
                                    "hop": k,
                                    "metric": f"accuracy_R{r}",
                                    "value": float(correct[mask, j].mean()),
                                    "ci_low": ci[0],
                                    "ci_high": ci[1],
                                }
                            )
                        gain = (
                            correct[mask, loops.index(20)].astype(float)
                            - correct[mask, loops.index(5)]
                        )
                        ci = template_bootstrap(
                            gain,
                            selected,
                            config["evaluation"]["bootstrap_replicates"],
                            config["evaluation"]["bootstrap_seed"],
                        )
                        scores.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "checkpoint": checkpoint,
                                "split": name,
                                "hop": k,
                                "metric": "gain_R20_minus_R5",
                                "value": float(gain.mean()),
                                "ci_low": ci[0],
                                "ci_high": ci[1],
                            }
                        )
                    if name.endswith("short"):
                        value = correct[:, robust].mean(1)
                        ci = template_bootstrap(
                            value,
                            rows,
                            config["evaluation"]["bootstrap_replicates"],
                            config["evaluation"]["bootstrap_seed"],
                        )
                        scores.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "checkpoint": checkpoint,
                                "split": name,
                                "hop": "1-5",
                                "metric": "RobustScore",
                                "value": float(value.mean()),
                                "ci_low": ci[0],
                                "ci_high": ci[1],
                            }
                        )
                # Only report observed discrete hops meeting threshold, no interpolation.
                long = cache[seed, arm, checkpoint, "eval_A_long"]
                rows = pools["eval_A_long"]
                threshold = {}
                for r in (5, 10, 20):
                    passed = [
                        k
                        for k in metadata["long_hops"]
                        if long[
                            np.array([row["hop_count"] == k for row in rows]), loops.index(r)
                        ].mean()
                        >= 0.8
                    ]
                    threshold[str(r)] = max(passed) if passed else None
                write_json(
                    run / f"evaluation-{checkpoint}" / "max_observed_long_hop_at80.json", threshold
                )
        assert len(initialization) == 1 and len(exposure) == 1, "Unpaired initialization or samples"
        for checkpoint in ("best", "last"):
            common_atoms = np.logical_and.reduce(
                [cache[seed, arm, checkpoint, "facts_train"][:, loops.index(5)] for arm in arms]
            )
            for name in ("eval_A_short", "eval_A_long", "eval_B_short", "eval_B_long"):
                rows = pools[name]
                masks = {
                    "all_models_know_every_atom_R5": common_known_mask(
                        rows, common_atoms, fact_index
                    )
                }
                if name.endswith("short"):
                    masks["all_models_correct_R5"] = np.logical_and.reduce(
                        [cache[seed, arm, checkpoint, name][:, loops.index(5)] for arm in arms]
                    )
                for condition, mask in masks.items():
                    for arm in arms:
                        correct = cache[seed, arm, checkpoint, name]
                        for k in sorted({row["hop_count"] for row in rows}):
                            hop_mask = np.array([row["hop_count"] == k for row in rows])
                            subset = mask & hop_mask
                            for j, r in enumerate(loops):
                                conditioned.append(
                                    {
                                        "seed": seed,
                                        "arm": arm,
                                        "checkpoint": checkpoint,
                                        "split": name,
                                        "condition": condition,
                                        "hop": k,
                                        "loop": r,
                                        "n": int(subset.sum()),
                                        "full_n": int(hop_mask.sum()),
                                        "coverage": float(subset.sum() / hop_mask.sum()),
                                        "accuracy": float(correct[subset, j].mean())
                                        if subset.any()
                                        else None,
                                    }
                                )
            for prefix in ("N", "Q"):
                for name in ("eval_A_short", "eval_A_long", "eval_B_short", "eval_B_long"):
                    gate = cache[seed, f"{prefix}-Gate", checkpoint, name]
                    fixed = cache[seed, f"{prefix}-Fixed", checkpoint, name]
                    rows = pools[name]
                    metrics = {
                        f"accuracy_R{r}": (
                            gate[:, loops.index(r)].astype(float) - fixed[:, loops.index(r)]
                        )
                        for r in (5, 10, 20)
                    }
                    if name.endswith("short"):
                        metrics["RobustScore"] = gate[:, robust].mean(1) - fixed[:, robust].mean(1)
                    metrics["gain_R20_minus_R5"] = (
                        gate[:, loops.index(20)].astype(float)
                        - gate[:, loops.index(5)]
                        - fixed[:, loops.index(20)]
                        + fixed[:, loops.index(5)]
                    )
                    for metric, values in metrics.items():
                        for k in ["all", *sorted({row["hop_count"] for row in rows})]:
                            mask = np.array([k == "all" or row["hop_count"] == k for row in rows])
                            ci = template_bootstrap(
                                values[mask],
                                [row for row, take in zip(rows, mask, strict=True) if take],
                                config["evaluation"]["bootstrap_replicates"],
                                config["evaluation"]["bootstrap_seed"],
                            )
                            differences.append(
                                {
                                    "seed": seed,
                                    "comparison": f"{prefix}-Gate_minus_{prefix}-Fixed",
                                    "checkpoint": checkpoint,
                                    "split": name,
                                    "metric": metric,
                                    "hop": k,
                                    "difference": float(values[mask].mean()),
                                    "ci_low": ci[0],
                                    "ci_high": ci[1],
                                }
                            )
    write_csv(out / "runs.csv", runs)
    write_csv(out / "scores_by_seed.csv", scores)
    write_csv(out / "paired_differences.csv", differences)
    write_csv(out / "common_subsets.csv", conditioned)
    aggregates = []
    groups = {}
    for row in scores:
        key = tuple(row[k] for k in ("arm", "checkpoint", "split", "hop", "metric"))
        groups.setdefault(key, []).append(row["value"])
    for key, values in groups.items():
        aggregates.append(
            dict(zip(("arm", "checkpoint", "split", "hop", "metric"), key, strict=True))
            | {
                "seeds": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
            }
        )
    write_csv(out / "seed_mean_sd.csv", aggregates)
    paired_groups = {}
    fields = ("comparison", "checkpoint", "split", "hop", "metric")
    for row in differences:
        paired_groups.setdefault(tuple(row[k] for k in fields), []).append(row["difference"])
    write_csv(
        out / "paired_mean_sd.csv",
        [
            dict(zip(fields, key, strict=True))
            | {
                "seeds": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "all_seeds_positive": all(v > 0 for v in values),
            }
            for key, values in paired_groups.items()
        ],
    )
    figures(config, base, out, cache, pools)
    result = {
        "passed": True,
        "trajectories": len(runs),
        "utc": utc(),
        "bootstrap_unit": "relation-sequence clusters, conditional on one fixed graph/model; "
        "few templates imply coarse intervals",
        "seed_variance": "Three paired training seeds; not independent graph replication",
        "interpretation": "Retain/Recover and long-chain gain are separate; "
        "no best test loop or seed selection",
        "checkpoint_policy": "best by A-short validation R5 only; "
        "last fixed-budget reported separately",
    }
    write_json(base / "analysis.json", result)
    return result


def figures(config, base, out, cache, pools):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms, seeds, loops = list(config["arms"]), config["seeds"], config["evaluation"]["loops"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm in arms:
        values = np.stack([cache[s, arm, "best", "eval_A_short"].mean(0) for s in seeds])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax.plot(loops, mean, label=arm)
        ax.fill_between(loops, mean - sd, mean + sd, alpha=0.12)
    ax.set(xlabel="Inference loops", ylabel="A-short accuracy", ylim=(0, 1))
    ax.axvline(5, color="gray", linestyle="--")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_loops.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    hops = config["data"]["long_hops"]
    rows = pools["eval_A_long"]
    for ax, arm in zip(axes.flat, arms, strict=True):
        means = np.stack([cache[s, arm, "best", "eval_A_long"] for s in seeds]).mean(0)
        values = np.stack(
            [means[np.array([row["hop_count"] == k for row in rows])].mean(0) for k in hops]
        )
        image = ax.imshow(values, vmin=0, vmax=1, aspect="auto")
        ax.set(title=arm, xlabel="Inference loops", ylabel="A-long hop")
        ax.set_xticks(range(len(loops)), loops, rotation=45)
        ax.set_yticks(range(len(hops)), hops)
    fig.colorbar(image, ax=axes.ravel().tolist())
    fig.savefig(out / "accuracy_heatmap.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm in arms:
        stats = [transitions(cache[s, arm, "best", "eval_A_short"], loops) for s in seeds]
        for ax, metric in zip(axes, ("retain", "recover"), strict=True):
            ax.plot(loops, np.mean([r[metric] for r in stats], axis=0), label=arm)
            ax.set(xlabel="Inference loops", ylabel=metric, ylim=(0, 1))
    axes[0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "retention_recovery.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm in arms:
        stats = []
        for seed in seeds:
            path = (
                base / "train" / f"{arm}-seed{seed}" / "evaluation-best/eval_A_short-dynamics.csv"
            )
            with path.open() as handle:
                stats.extend(list(csv.DictReader(handle)))
        for ax, metric in zip(axes, ("alpha_mean", "update_ratio"), strict=True):
            values = [
                np.mean([float(row[metric]) for row in stats if int(row["loop"]) == r])
                for r in range(1, 33)
            ]
            ax.plot(range(1, 33), values, label=arm)
            ax.set(xlabel="Inference loops", ylabel=metric)
    axes[0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "gate_and_state_dynamics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    analyze(json.loads(args.config.read_text()), args.base)


if __name__ == "__main__":
    main()
