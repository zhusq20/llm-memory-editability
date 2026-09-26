"""Summarize all development trajectories without selecting successful cases."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read(path):
    return json.loads(path.read_text())


def run(args):
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trajectories, rows, edits = {}, [], []
    for directory in sorted(root.glob("world-*-seed-*-*")):
        if not (directory / "learning.json").exists():
            continue
        config = read(directory / "config.json")
        learning = read(directory / "learning.json")
        key = (config["world_seed"], config["seed"], config["order"])
        trajectories[key] = learning
        final = learning[-1]
        reached = [p for p in learning if p["learning_threshold_passed"]]
        row = {
            "run": directory.name,
            "world": key[0],
            "seed": key[1],
            "order": key[2],
            "complete": (directory / "complete.json").exists(),
            "step": final["step"],
            "base_accuracy": final["base_accuracy"],
            "derived_accuracy": final["derived_accuracy"],
            "old_exception_accuracy": final["strata"]["actual_old_exception"],
            "first_observed_learning_step": reached[0]["step"] if reached else None,
            "first_observed_learning_matmul_flops_estimate": reached[0][
                "train_matmul_flops_estimate"
            ]
            if reached
            else None,
            "train_seconds": final["train_seconds"],
            "train_matmul_flops_estimate": final["train_matmul_flops_estimate"],
        }
        rows.append(row)
        for result in sorted(directory.glob("edits/*/complete.json")):
            case = read(result)
            edits.append(
                {
                    "run": directory.name,
                    "world": key[0],
                    "seed": key[1],
                    "order": key[2],
                    "directory": str(result.parent),
                    **case,
                }
            )
    with (out / "learning-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for col, (field, title) in enumerate(
        (
            ("base_accuracy", "Atomic facts"),
            ("derived_accuracy", "Derived queries"),
            ("old_exception", "Old personal exceptions"),
        )
    ):
        ax = axes[col]
        for order, color, label in (
            ("SA", "#2874a6", "Structure first"),
            ("AS", "#d35400", "Facts first"),
        ):
            series = [v for k, v in trajectories.items() if k[2] == order]
            by_step = defaultdict(list)
            for trajectory in series:
                for p in trajectory:
                    y = (
                        p["strata"]["actual_old_exception"]
                        if field == "old_exception"
                        else p[field]
                    )
                    by_step[p["step"]].append((p["train_matmul_flops_estimate"] / 1e15, y * 100))
            values = [(s, np.array(v)) for s, v in sorted(by_step.items())]
            x = [v[:, 0].mean() for _, v in values]
            mean = [v[:, 1].mean() for _, v in values]
            low, high = [v[:, 1].min() for _, v in values], [v[:, 1].max() for _, v in values]
            ax.plot(x, mean, "o-", markersize=3, color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.12)
        ref = next(iter(trajectories.values()))
        for step in (2640, 5280):
            p = next(p for p in ref if p["step"] == step)
            ax.axvline(
                p["train_matmul_flops_estimate"] / 1e15, color="gray", linewidth=0.8, linestyle=":"
            )
        ax.set(
            title=title,
            xlabel="Training matrix FLOPs estimate (PFLOP)",
            ylabel="Complete-answer accuracy (%)",
            ylim=(-2, 102),
        )
        ax.grid(alpha=0.15)
    axes[0].legend(loc="lower right", fontsize=9)
    n_worlds = len({key[0] for key in trajectories})
    n_seeds = len({key[1] for key in trajectories})
    fig.suptitle(
        f"bioS-Work symbolic development: {n_worlds} worlds × {n_seeds} initializations; "
        "shading = observed range"
    )
    fig.savefig(out / "learning-curves.png", dpi=180)
    fig.savefig(out / "learning-curves.pdf")
    plt.close(fig)
    paired_learning = []
    for world in (0, 1):
        for seed in (0, 1):
            if (world, seed, "SA") not in trajectories or (world, seed, "AS") not in trajectories:
                continue
            sa, ass = trajectories[world, seed, "SA"][-1], trajectories[world, seed, "AS"][-1]
            if sa["step"] != ass["step"]:
                continue
            paired_learning.append(
                {
                    "world": world,
                    "seed": seed,
                    "step": sa["step"],
                    "base_SA_minus_AS_pp": 100 * (sa["base_accuracy"] - ass["base_accuracy"]),
                    "derived_SA_minus_AS_pp": 100
                    * (sa["derived_accuracy"] - ass["derived_accuracy"]),
                    "quality_subset_passed": bool(
                        min(
                            sa["base_accuracy"],
                            ass["base_accuracy"],
                            sa["derived_accuracy"],
                            ass["derived_accuracy"],
                        )
                        >= 0.99
                        and abs(sa["base_accuracy"] - ass["base_accuracy"]) <= 0.005
                        and abs(sa["derived_accuracy"] - ass["derived_accuracy"]) <= 0.005
                        and abs(sa["value_nll"] - ass["value_nll"]) <= 0.1
                    ),
                }
            )
    baseline_edits = [e for e in edits if e.get("branch") is None]
    edit_groups = defaultdict(list)
    for case in edits:
        edit_groups[(case["scope"], case.get("branch"), case["order"], case["kind"])].append(case)
    aggregates = []
    for (scope, branch, order, kind), cases in edit_groups.items():
        final = [c["final"] for c in cases]
        aggregates.append(
            {
                "scope": scope,
                "branch": branch,
                "order": order,
                "kind": kind,
                "n": len(cases),
                "mean_E": float(np.mean([f["E"] for f in final])),
                "mean_D": float(np.mean([f["D"] for f in final])),
                "joint_ever_passed": sum(c["first_observed_joint_step"] is not None for c in cases),
                "joint_final_passed": sum(f["joint_pass"] is True for f in final),
                "joint_evaluable": sum(c["joint_evaluable"] for c in cases),
                "mean_U_heldout_destruction": {
                    str(g): float(
                        np.mean(
                            [
                                f["U_heldout_destruction"][str(g)]["rate"]
                                for f in final
                                if f["U_heldout_destruction"][str(g)]["rate"] is not None
                            ]
                        )
                    )
                    for g in range(4)
                },
            }
        )
    if baseline_edits:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
        for row, scope in enumerate(("mlp", "all", "down")):
            for order, color in (("SA", "#2874a6"), ("AS", "#d35400")):
                for kind, style in (("coherent", "-"), ("exception", "--")):
                    group = [
                        e
                        for e in baseline_edits
                        if (e["scope"], e["order"], e["kind"]) == (scope, order, kind)
                    ]
                    points = defaultdict(list)
                    for case in group:
                        for point in read(Path(case["directory"]) / "trajectory.json"):
                            rates = [
                                v["rate"]
                                for v in point["U_heldout_destruction"].values()
                                if v["rate"] is not None
                            ]
                            points[point["step"]].append(
                                [point["E"], point["D"], max(rates) if rates else np.nan]
                            )
                    for col in range(3):
                        x = sorted(points)
                        y = [100 * np.mean(np.array(points[s])[:, col]) for s in x]
                        axes[row, col].plot(x, y, style, color=color, label=f"{order} {kind}")
                        axes[row, col].set_xscale("symlog", linthresh=1)
                        axes[row, col].set(
                            xlabel="Edit optimization steps",
                            ylabel="Percent",
                            title=(
                                f"FT-{scope.upper()} · "
                                + ["E success", "D propagation", "Worst stratum damage"][col]
                            ),
                        )
                        axes[row, col].grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Middle-window development edits: all fixed cases; no target-NLL filtering")
        fig.savefig(out / "editing-curves.png", dpi=180)
        fig.savefig(out / "editing-curves.pdf")
        plt.close(fig)
    # Paired common-known retention, independent of each arm's individual denominator.
    common_retention, interactions = [], []
    lookup = {
        (e["world"], e["seed"], e["support"], e["scope"], e.get("branch"), e["order"], e["kind"]): e
        for e in edits
    }
    for key, sa in lookup.items():
        world, seed, support, scope, branch, order, kind = key
        if order != "SA":
            continue
        ass = lookup.get((world, seed, support, scope, branch, "AS", kind))
        if ass is None:
            continue
        s_dir, a_dir = Path(sa["directory"]), Path(ass["directory"])
        s_set, a_set = np.load(s_dir / "sets.npz"), np.load(a_dir / "sets.npz")
        shared = s_set["old_correct"] & a_set["old_correct"]
        unseen = np.zeros(len(shared), dtype=bool)
        unseen[s_set["heldout"]] = True
        s_pred, a_pred = (
            np.load(s_dir / "predictions-512.npz"),
            np.load(a_dir / "predictions-512.npz"),
        )
        damage = {}
        for g in range(4):
            mask = shared & unseen & (s_set["strata"] == g)
            damage[str(g)] = {
                "known": int(mask.sum()),
                "SA": float((~s_pred["correct"][mask]).mean()) if mask.any() else None,
                "AS": float((~a_pred["correct"][mask]).mean()) if mask.any() else None,
            }
        common_retention.append(
            {
                "world": world,
                "seed": seed,
                "support": support,
                "scope": scope,
                "branch": branch,
                "kind": kind,
                "damage": damage,
            }
        )
        if kind == "coherent":
            se = lookup.get((world, seed, support, scope, branch, "SA", "exception"))
            ae = lookup.get((world, seed, support, scope, branch, "AS", "exception"))
            if se and ae:
                interactions.append(
                    {
                        "world": world,
                        "seed": seed,
                        "support": support,
                        "scope": scope,
                        "branch": branch,
                        "E_error_difference_in_differences": (sa["final"]["E"] - se["final"]["E"])
                        - (ass["final"]["E"] - ae["final"]["E"]),
                        "D_error_difference_in_differences": (sa["final"]["D"] - se["final"]["D"])
                        - (ass["final"]["D"] - ae["final"]["D"]),
                    }
                )
    organization = []
    for path in sorted(root.glob("world-*-seed-*-*/organization-*/measurements.json")):
        report = read(path)
        grouped = defaultdict(list)
        for probe in report["probes"]:
            grouped[(probe["layer"], probe["intervention"])].append(probe)
        organization.append(
            {
                "run": path.parent.parent.name,
                "step": report["step"],
                "device": report.get("device"),
                "precision": report.get("precision"),
                "geometry": report["geometry"],
                "mean_selectivity": [
                    {
                        "layer": layer,
                        "intervention": intervention,
                        "score": float(np.mean([v["selective_shared_score"] for v in values])),
                        "same_company_nonexception": float(
                            np.mean([v["same_company_nonexception"] for v in values])
                        ),
                        "independent_attribute": float(
                            np.mean([v["independent_attribute"] for v in values])
                        ),
                        "unrelated_company": float(
                            np.mean([v["unrelated_company"] for v in values])
                        ),
                    }
                    for (layer, intervention), values in grouped.items()
                ],
            }
        )
    if organization:
        # CPU and GPU are precision checks of the same checkpoints, not extra seeds.
        endpoint_precision = (
            "BF16 autocast"
            if any(r["step"] == 13280 and r["precision"] == "BF16 autocast" for r in organization)
            else "FP32 CPU diagnostic"
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
        fields = ["same_company_nonexception", "independent_attribute", "unrelated_company"]
        titles = [
            "Same-company nonexceptions",
            "Independent attribute (birth city)",
            "Unrelated company",
        ]
        for order, color, label in (
            ("SA", "#2874a6", "Structure first"),
            ("AS", "#d35400", "Facts first"),
        ):
            reports = [
                r
                for r in organization
                if r["run"].endswith(order)
                and r["step"] == 13280
                and r["precision"] == endpoint_precision
            ]
            for col, field in enumerate(fields):
                values = (
                    np.array(
                        [
                            [
                                next(
                                    v[field]
                                    for v in r["mean_selectivity"]
                                    if v["layer"] == layer and v["intervention"] == "donor"
                                )
                                for layer in range(8)
                            ]
                            for r in reports
                        ]
                    )
                    * 100
                )
                if len(values):
                    axes[col].plot(range(1, 9), values.mean(0), "o-", color=color, label=label)
                    axes[col].fill_between(
                        range(1, 9), values.min(0), values.max(0), color=color, alpha=0.12
                    )
                axes[col].set(
                    title=titles[col],
                    xlabel="MLP layer (one-based)",
                    ylabel="Donor-answer probability change (pp)",
                )
                axes[col].axhline(0, color="gray", linewidth=0.7)
                axes[col].grid(alpha=0.15)
        axes[0].legend(fontsize=9)
        fig.suptitle(
            f"Candidate company-mean replacement · {endpoint_precision} · observed run range"
        )
        fig.savefig(out / "organization-diagnostics.png", dpi=180)
        fig.savefig(out / "organization-diagnostics.pdf")
        plt.close(fig)
    dump(
        out / "summary.json",
        {
            "phase": "exploratory symbolic development; not confirmatory inference",
            "expected_learning_runs": 8,
            "complete_learning_runs": sum(r["complete"] for r in rows),
            "completed_edit_cases": sum(
                len(read(p)["cases"])
                for pattern in (
                    "world-*/edits/complete.json",
                    "world-*/edits-window0/complete.json",
                )
                for p in root.glob(pattern)
            ),
            "completed_middle_window_edit_cases": len(edits),
            "edit_aggregate_scope": (
                "middle window only; early-window analysis is in failure-analysis-20260926"
            ),
            "interrupted_attempts": (
                read(root / "interruption-ledger.json")
                if (root / "interruption-ledger.json").exists()
                else []
            ),
            "learning": rows,
            "paired_learning": paired_learning,
            "edit_aggregates": aggregates,
            "common_known_retention": common_retention,
            "edit_interactions": interactions,
            "organization": organization,
            "limitations": [
                "Two worlds/two initializations planned; completions and interruptions separated.",
                "Symbolic atomic-token answers; natural-language validation pending.",
                "No target-NLL matching or hyperparameter/window selection search.",
                "FLOPs are matrix-operation estimates; editing plots use steps within each scope.",
            ],
        },
    )
    print(json.dumps({"learning_runs": len(rows), "edit_cases": len(edits), "output": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-dev-v1")
    parser.add_argument("--output", default="results/bios-dev-v1/report")
    run(parser.parse_args())
