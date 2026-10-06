#!/usr/bin/env python3
"""Export every trajectory and the separate compute/exposure and load curves."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def collect(root):
    config = read(root / "frozen-config.json")
    endpoints, learning = [], []
    for path in sorted((root / "runs").glob("*")):
        if not (path / "run.json").exists() or not (path / "learning.json").exists():
            continue
        meta, history = read(path / "run.json"), read(path / "learning.json")
        spec = meta["spec"]
        identity = {
            "run": path.name,
            "phase": spec["phase"],
            "world_seed": spec["world_seed"],
            "entities": spec["entities"],
            "phi": spec["phi"],
            "parameters": meta["parameters"],
            "knowledge_bits": meta["knowledge_bits"],
            "bits_per_parameter": meta["knowledge_bits"] / meta["parameters"],
            "composition_examples": meta["composition_examples"],
            "atomic_examples": meta["atomic_examples"],
            "initial_model_sha256": meta["initial_model_sha256"],
            "world_sha256": meta["world_sha256"],
        }
        for row in history:
            flat = {
                **identity,
                "step": row["step"],
                "atomic_epochs": row["atomic_epochs"],
                "composition_epochs": row["composition_epochs"],
                "flops": row["estimated_matmul_training_flops"],
                "training_seconds": row["training_seconds"],
            }
            for name in ("atomic", "atomic_id", "atomic_ood", "train_composition", "II", "OO"):
                flat[name] = row["metrics"][name]["accuracy"]
                flat[name + "_n"] = row["metrics"][name]["n"]
                flat[name + "_answer"] = row["metrics"][name]["answer_accuracy"]
            learning.append(flat)
            for budget, step in meta["budgets"].items():
                if row["step"] == step:
                    endpoints.append(
                        {
                            **flat,
                            "budget": budget,
                            "audited": (path / "audit.json").exists()
                            and read(path / "audit.json")["passed"] is True,
                        }
                    )
    return config, endpoints, learning


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(axis, rows, xkey, metric, label, color):
    grouped = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            grouped[row[xkey]].append(row[metric])
    xs = sorted(grouped)
    if not xs:
        return
    averages = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    axis.plot(xs, [100 * value for value in averages], "o-", label=label, color=color)
    if any(len(grouped[x]) > 1 for x in xs):
        axis.fill_between(
            xs,
            [100 * min(grouped[x]) for x in xs],
            [100 * max(grouped[x]) for x in xs],
            color=color,
            alpha=0.12,
        )


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config, endpoints, learning = collect(root)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "budget-points.csv", endpoints)
    write_csv(out / "all-learning-nodes.csv", learning)
    state = read(root / "controller-state.json")
    write = {
        "state": state,
        "runs_observed": sorted({row["run"] for row in learning}),
        "budget_points": endpoints,
        "independent_confirmation_worlds": len(config["confirmation_worlds"]),
        "limits": [
            "Partial curves retain observed points and do not imply complete registration",
            "Shading is the range across observed worlds, not a confidence interval",
            "Knowledge bits condition on graph keys; composition adds no independent entropy",
            "Varying N creates closed graphs with different truths, not nested knowledge",
            "Native answer plus generated EOS differs from external decomposed calls",
        ],
    }
    (out / "summary.json").write_text(json.dumps(write, indent=2, ensure_ascii=False) + "\n")
    formal = [row for row in endpoints if row["phase"] == "confirmation" and row["audited"]]
    support = [row for row in formal if row["run"].startswith("support-")]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("atomic", "II", "OO"),
        ("Atomic recall", "Unseen ID composition", "OOD composition"),
        strict=True,
    ):
        for budget, color, label in [
            ("compute", "#2563eb", "128k updates: equal compute"),
            ("exposure", "#d97706", "Equal mean record exposures"),
        ]:
            plot_curve(
                axis,
                [row for row in support if row["budget"] == budget],
                "composition_examples",
                metric,
                label,
                color,
            )
        axis.set(
            xlabel="Distinct composition training examples",
            ylabel="Answer + generated EOS (%)",
            title=title,
            ylim=(-2, 102),
        )
        axis.set_xscale("symlog", linthresh=1000)
        axis.grid(alpha=0.2)
    if support:
        axes[0].legend(fontsize=8)
    figure.suptitle("Registered support curve — observed audited points only")
    for extension in ("png", "pdf"):
        figure.savefig(out / ("support-curves." + extension), dpi=180)
    plt.close(figure)
    load = [
        row
        for row in formal
        if row["budget"] == "exposure"
        and (row["run"].startswith("load-") or row["phi"] == config["anchor_phi"])
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for metric, color, label in [
        ("atomic", "#2563eb", "Atomic recall"),
        ("II", "#d97706", "Native unseen composition"),
        ("OO", "#059669", "OOD composition"),
    ]:
        plot_curve(axis, load, "bits_per_parameter", metric, label, color)
    axis.set(
        xlabel="Independent knowledge bits / model parameter",
        ylabel="Answer + generated EOS (%)",
        ylim=(-2, 102),
        title="Knowledge load — fixed model and matched exposures",
    )
    axis.grid(alpha=0.2)
    if load:
        axis.legend()
    for extension in ("png", "pdf"):
        figure.savefig(out / ("knowledge-load-curves." + extension), dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    by_run = defaultdict(list)
    for row in learning:
        by_run[row["run"]].append(row)
    for name, rows in by_run.items():
        if rows[0]["phase"] != "development":
            continue
        axis.plot([row["step"] for row in rows], [100 * row["II"] for row in rows], label=name)
    axis.set(
        xlabel="Optimizer updates",
        ylabel="Native unseen composition (%)",
        ylim=(-2, 102),
        title="Development calibration trajectories",
    )
    axis.grid(alpha=0.2)
    if any(rows[0]["phase"] == "development" for rows in by_run.values()):
        axis.legend(fontsize=8)
    figure.savefig(out / "development-learning.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"runs_observed": len(by_run), "report": str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report(args.root)


if __name__ == "__main__":
    main()
