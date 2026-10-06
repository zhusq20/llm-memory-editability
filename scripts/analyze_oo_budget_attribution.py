"""Recount archived OO predictions and compare measured training budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATCH = "oo-budget-attribution-20261005"
STEPS = (32000, 64000, 128000, 200000, 300000, 500000, 700000, 1000000, 1500000)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    sources, flat, endpoints, recounts = {}, [], [], []

    def register(path):
        sources[str(path.relative_to(ROOT))] = sha(path)

    meta_path = ROOT / "results/grokking-reproduction-v1/data/complete.json"
    metadata = read(meta_path)
    register(meta_path)
    synthetic = ROOT / "results/grokking-reproduction-v1/development"
    real = ROOT / "results/realworld-composition-confirmation-v1/confirmation"
    for dataset, parent in [("synthetic", synthetic), ("real_native", real)]:
        for run in sorted(parent.iterdir()):
            if not (run / "learning.json").exists():
                continue
            runmeta = read(run / "run.json")
            spec = runmeta["spec"]
            register(run / "run.json")
            register(run / "learning.json")
            history = read(run / "learning.json")
            architecture = "loop4x2" if "loop4x2" in run.name else "standard8"
            if "standard4" in run.name:
                architecture = "standard4"
            local = []
            for point in history:
                metrics = point["metrics"]
                row = {
                    "dataset": dataset,
                    "run": run.name,
                    "architecture": architecture,
                    "initialization": spec["initialization"],
                    "learning_rate": spec.get("lr", spec.get("learning_rate")),
                    "weight_decay": spec["weight_decay"],
                    "phi": spec.get("phi"),
                    "step": point["step"],
                    "oo_n": metrics["test_oo"]["n"],
                    "oo_percent": 100
                    * metrics["test_oo"]["accuracy" if dataset == "synthetic" else "alias_em"],
                    "evaluation_scope": "fixed_full_OO_pool"
                    if dataset == "synthetic"
                    else "fixed_panel",
                    "training_hours": point["training_seconds"] / 3600,
                    "estimated_matmul_training_flops": point["estimated_matmul_training_flops"],
                    "executed_input_tokens": point["executed_input_tokens"],
                    "examples": point["examples"],
                    "atomic_epochs": point["atomic_epochs"],
                    "composition_epochs": point["composition_epochs"],
                }
                if dataset == "synthetic" and point["step"] in STEPS:
                    path = run / f"predictions-{point['step']:07d}-test_oo.npz"
                    with np.load(path) as raw:
                        correct = raw["answer"] == raw["rows"][:, -1] + metadata["entity_offset"]
                        correct &= raw["stop"] == metadata["end_marker"]
                    assert len(correct) == row["oo_n"]
                    assert abs(float(correct.mean()) * 100 - row["oo_percent"]) < 1e-10
                    register(path)
                    recounts.append(
                        {
                            "run": run.name,
                            "step": point["step"],
                            "n": len(correct),
                            "correct": int(correct.sum()),
                        }
                    )
                local.append(row)
            final = dict(local[-1])
            if dataset == "real_native":
                path = run / "endpoint-predictions.json"
                raw = [r for r in read(path)["test_all"] if r["role"] == "OO"]
                expected = read(run / "endpoint.json")["test_oo"]
                assert len(raw) == expected["n"] == 1283
                correct = sum(r["alias_em"] for r in raw)
                assert abs(correct / len(raw) - expected["alias_em"]) < 1e-12
                final.update(
                    oo_n=len(raw),
                    oo_percent=100 * correct / len(raw),
                    evaluation_scope="full_endpoint",
                )
                register(path)
                register(run / "endpoint.json")
                recounts.append(
                    {"run": run.name, "step": final["step"], "n": len(raw), "correct": int(correct)}
                )
            endpoints.append(final)
            flat.extend(local)

    synthetic_300k = [r for r in flat if r["dataset"] == "synthetic" and r["step"] == 300000]
    real_loop = [
        r for r in endpoints if r["dataset"] == "real_native" and r["architecture"] == "loop4x2"
    ]
    synthetic_loop = [r for r in endpoints if r["run"] == "loop4x2-phi7.2-wd0.3"][0]
    real_summary = {
        key: statistics.mean(r[key] for r in real_loop)
        for key in [
            "oo_percent",
            "training_hours",
            "estimated_matmul_training_flops",
            "executed_input_tokens",
            "examples",
            "atomic_epochs",
            "composition_epochs",
        ]
    }
    real_summary.update(
        n_initializations=3,
        oo_n_each=1283,
        oo_sample_sd=statistics.stdev(r["oo_percent"] for r in real_loop),
    )
    temporal_brackets = []
    for endpoint in real_loop:
        for key in [
            "training_hours",
            "estimated_matmul_training_flops",
            "examples",
            "atomic_epochs",
        ]:
            target = endpoint[key]
            candidates = [r for r in flat if r["run"] == synthetic_loop["run"]]
            lower = [r for r in candidates if r[key] <= target]
            upper = [r for r in candidates if r[key] >= target]
            temporal_brackets.append(
                {
                    "real_run": endpoint["run"],
                    "budget": key,
                    "real_value": target,
                    "synthetic_below": max(lower, key=lambda r: r[key]) if lower else None,
                    "synthetic_above": min(upper, key=lambda r: r[key]) if upper else None,
                    "extrapolated": False,
                }
            )
    summary = {
        "analysis": "retrospective_read_only_recount",
        "same_300k_updates": synthetic_300k,
        "real_loop_300k_full_endpoint_mean": real_summary,
        "synthetic_loop_wd03_1500k": synthetic_loop,
        "real_300k_over_synthetic_1500k": {
            key: real_summary[key] / synthetic_loop[key]
            for key in [
                "training_hours",
                "estimated_matmul_training_flops",
                "executed_input_tokens",
                "examples",
                "atomic_epochs",
            ]
        },
        "limits": [
            "Same optimizer steps are not the same FLOPs, token exposure, epochs or GPU time.",
            "FLOPs share a dense-matmul/attention convention, approximate backward as 2x forward, "
            "and exclude elementwise ops, optimizer, evaluation and system overhead.",
            "Recorded GPU time comes from separate runs and execution implementations; "
            "it is an engineering budget, not a causal control.",
            "Synthetic is one graph/initialization; real is one graph with three initializations. "
            "Neither query count nor seeds are independent worlds.",
            "Real intermediate points use 64 OO queries; only real 300k uses the full 1283 pool. "
            "Synthetic uses its fixed 2004-query OO pool throughout.",
            "Synthetic strict token+stop accuracy and real official-alias EM "
            "are different scoring protocols.",
            "Dataset, entity representation, support, sampling, initialization, LR and WD "
            "remain confounded across historical batches.",
            "Retrospective threshold/cross-budget views do not change the frozen endpoints. "
            "No interpolated accuracy or extrapolated 3M synthetic score is reported.",
        ],
    }
    write(out / "summary.json", summary)
    write(out / "budget-brackets.json", temporal_brackets)
    write(
        out / "recount-audit.json",
        {
            "passed": True,
            "raw_pools_recounted": len(recounts),
            "records": recounts,
            "sources_sha256": sources,
        },
    )
    with (out / "learning-budgets.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (out / "endpoints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    plot(flat, real_loop, out)
    print(
        json.dumps(
            {
                "real_loop": real_summary,
                "synthetic_loop_wd03": synthetic_loop,
                "ratios": summary["real_300k_over_synthetic_1500k"],
                "recounted_pools": len(recounts),
            },
            indent=2,
        )
    )


def plot(rows, real_loop, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout="constrained")
    colors = {0.1: "#3679b5", 0.3: "#d3544b"}
    for decay, color in colors.items():
        run = f"loop4x2-phi7.2-wd{decay}"
        values = [r for r in rows if r["run"] == run]
        for ax, key, scale, label in zip(
            axes,
            ["step", "training_hours", "estimated_matmul_training_flops"],
            [1000, 1, 1e18],
            [
                "Updates (thousands)",
                "Measured training GPU hours",
                "Estimated matmul training FLOPs (1e18)",
            ],
            strict=True,
        ):
            ax.plot(
                [r[key] / scale for r in values],
                [r["oo_percent"] for r in values],
                color=color,
                lw=1.2,
                label=f"Synthetic Loop, wd={decay}",
            )
            ax.set_xlabel(label)
    for ax, key, scale in zip(
        axes,
        ["step", "training_hours", "estimated_matmul_training_flops"],
        [1000, 1, 1e18],
        strict=True,
    ):
        ax.scatter(
            [r[key] / scale for r in real_loop],
            [r["oo_percent"] for r in real_loop],
            marker="D",
            color="#282828",
            s=35,
            label="Real names, 300k: 3 seeds",
            zorder=5,
        )
        ax.set_ylim(-2, 103)
        ax.set_ylabel("Strict OO / official-alias EM (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Historical budget comparison: different datasets / LR / scoring; no causal ranking",
        fontsize=12,
    )
    fig.savefig(out / "budget-comparison.png", dpi=170)
    fig.savefig(out / "budget-comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/development-artifacts" / BATCH)
    main(parser.parse_args().out)
