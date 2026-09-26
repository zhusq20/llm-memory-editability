"""Render completed failure/representation diagnostics without pooling precision repeats."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_rows(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-dev-v1")
    parser.add_argument("--output", default="results/bios-dev-v1/failure-analysis-20260926")
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    analysis = read(out / "failure-analysis.json")
    methods = [
        "FT-MLP",
        "FT-ALL",
        "FT-DOWN",
        "FT-MLP+shared",
        "FT-MLP+independent",
        "FT-MLP+random",
    ]
    signatures = ["pass", "D", "U", "D+U", "E", "E+D", "E+U", "E+D+U"]
    colors = [
        "#2a9d8f",
        "#457b9d",
        "#e9c46a",
        "#8e6c8a",
        "#e76f51",
        "#264653",
        "#f4a261",
        "#707070",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    for row, window in enumerate((3, 0)):
        for col, kind in enumerate(("coherent", "exception")):
            ax = axes[row, col]
            groups = [
                next(
                    g
                    for g in analysis["groups"]
                    if (g["window"], g["method"], g["order"], g["kind"])
                    == (window, method, order, kind)
                )
                for method in methods
                for order in ("SA", "AS")
            ]
            bottom = np.zeros(len(groups))
            for signature, color in zip(signatures, colors, strict=True):
                values = np.array(
                    [100 * g["signatures"].get(signature, 0) / g["n"] for g in groups]
                )
                ax.bar(range(len(groups)), values, bottom=bottom, label=signature, color=color)
                bottom += values
            ax.set_xticks(
                range(len(groups)),
                [
                    g["method"].replace("FT-", "").replace("MLP+", "+")
                    + "\n"
                    + g["order"]
                    + f" (n={g['n']})"
                    for g in groups
                ],
                rotation=55,
                ha="right",
                fontsize=8,
            )
            ax.set(
                title=f"Layers {window + 1}–{window + 3}: {kind}", ylabel="Cases (%)", ylim=(0, 100)
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=8)
    fig.suptitle(
        "Final failure combinations at step 512 · E: direct update; D: propagation; U: retention",
        y=1.04,
    )
    save(fig, out, "failure-combinations")

    timeline = read(out / "edit-trajectories.json")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for row, method in enumerate(methods[:3]):
        for order, color in (("SA", "#2874a6"), ("AS", "#d35400")):
            for kind, style in (("coherent", "-"), ("exception", "--")):
                group = [
                    p
                    for p in timeline
                    if (p["window"], p["method"], p["order"], p["kind"]) == (3, method, order, kind)
                ]
                by_step = defaultdict(list)
                for p in group:
                    by_step[p["step"]].append(
                        [
                            p["E"],
                            p["D"],
                            max(v["rate"] for v in p["U_heldout_destruction"].values()),
                        ]
                    )
                x = sorted(by_step)
                for col in range(3):
                    axes[row, col].plot(
                        x,
                        [100 * np.mean(by_step[s], axis=0)[col] for s in x],
                        style,
                        color=color,
                        label=f"{order} {kind}",
                    )
                    axes[row, col].set_xscale("symlog", linthresh=1)
                    axes[row, col].set(
                        title=f"{method}: " + ["E", "D", "Worst heldout stratum damage"][col],
                        xlabel="Edit steps",
                        ylabel="Percent",
                    )
                    axes[row, col].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Middle window · all fixed cases · means across cases, no success filtering")
    save(fig, out, "edit-trajectories")

    measurements, rows, geometry, gradients, branch_checks = [], [], [], [], []
    for run in sorted(root.glob("world-*-seed-*-*")):
        config = read(run / "config.json")
        meta = {
            "run": run.name,
            "world": config["world_seed"],
            "seed": config["seed"],
            "order": config["order"],
        }
        for directory in [f"organization-{s}" for s in (2640, 5280, 13280)] + [
            "organization-cpu-13280"
        ]:
            report = read(run / directory / "measurements.json")
            assert len(report["probes"]) == 320
            measurements.append(
                {
                    **meta,
                    "directory": directory,
                    "step": report["step"],
                    "precision": report["precision"],
                    "probes": len(report["probes"]),
                    "seconds": report.get("measurement_seconds"),
                }
            )
            groups = defaultdict(list)
            for p in report["probes"]:
                groups[(p["layer"], p["intervention"])].append(p)
            for (layer, intervention), ps in sorted(groups.items()):
                entry = {
                    **meta,
                    "step": report["step"],
                    "precision": report["precision"],
                    "layer": layer,
                    "intervention": intervention,
                }
                for field in (
                    "same_company_nonexception",
                    "old_exception",
                    "independent_attribute",
                    "unrelated_company",
                    "selective_shared_score",
                ):
                    entry[field] = float(np.mean([p[field] for p in ps]))
                for group in (
                    "same_company_nonexception",
                    "old_exception",
                    "independent_attribute",
                    "unrelated_company",
                ):
                    generated = [p["generation"][group] for p in ps if "generation" in p]
                    for field in (
                        "donor_generated_delta",
                        "baseline_old_accuracy",
                        "changed_old_accuracy",
                    ):
                        entry[f"generation_{group}_{field}"] = (
                            float(np.mean([g[field] for g in generated])) if generated else None
                        )
                rows.append(entry)
            geometry.extend(
                {**meta, "step": report["step"], "precision": report["precision"], **g}
                for g in report["geometry"]
            )
        gradients.extend(
            {**meta, **g} for g in read(run / "edit-geometry-13280/measurements.json")["records"]
        )
        for directory, window in (("edits", 3), ("edits-window0", 0)):
            checks = read(run / directory / "branch-operation-checks.json")
            branch_checks.append({**meta, "window": window, **checks})
    assert len(measurements) == 32 and len(gradients) == 64
    lookup = {(r["run"], r["precision"], r["step"], r["layer"], r["intervention"]): r for r in rows}
    comparisons = []
    for r in rows:
        if (
            r["precision"] != "BF16 autocast"
            or r["step"] != 13280
            or r["intervention"] == "random_equal_norm"
        ):
            continue
        cpu = lookup[(r["run"], "FP32 CPU diagnostic", 13280, r["layer"], r["intervention"])]
        comparisons.append(
            {
                "run": r["run"],
                "layer": r["layer"],
                "intervention": r["intervention"],
                **{
                    f"{f}_GPU_minus_CPU_pp": 100 * (r[f] - cpu[f])
                    for f in (
                        "same_company_nonexception",
                        "independent_attribute",
                        "unrelated_company",
                        "selective_shared_score",
                    )
                },
            }
        )
    csv_rows(out / "organization-probes.csv", rows)
    csv_rows(out / "organization-geometry.csv", geometry)
    csv_rows(out / "precision-comparison.csv", comparisons)
    dump(
        out / "representation-analysis.json",
        {
            "measurements": measurements,
            "probes": rows,
            "geometry": geometry,
            "edit_geometry": gradients,
            "precision_comparison": comparisons,
            "branch_operation_checks": branch_checks,
            "random_control_note": (
                "CPU and CUDA use different RNG streams; "
                "compare deterministic shifts for paired precision checks."
            ),
        },
    )
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for i, step in enumerate((2640, 5280, 13280)):
        for col, field in enumerate(
            ("same_company_nonexception", "independent_attribute", "unrelated_company")
        ):
            for order, color in (("SA", "#2874a6"), ("AS", "#d35400")):
                values = (
                    np.array(
                        [
                            [
                                lookup[
                                    (
                                        f"world-{world}-seed-{seed}-{order}",
                                        "BF16 autocast",
                                        step,
                                        layer,
                                        "donor",
                                    )
                                ][field]
                                for layer in range(8)
                            ]
                            for world in (0, 1)
                            for seed in (0, 1)
                        ]
                    )
                    * 100
                )
                axes[i, col].plot(range(1, 9), values.mean(0), "o-", color=color, label=order)
                axes[i, col].fill_between(
                    range(1, 9), values.min(0), values.max(0), color=color, alpha=0.12
                )
            axes[i, col].set(
                title=f"Step {step}: "
                + ["same-company members", "independent birth city", "unrelated company"][col],
                xlabel="Layer (one-based)",
                ylabel="Donor probability change (pp)",
            )
            axes[i, col].axhline(0, color="gray", linewidth=0.7)
            axes[i, col].grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(
        "Fixed stage and endpoint probes · GPU BF16 · bands = observed range over 4 models"
    )
    save(fig, out, "organization-stages")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for col, field in enumerate(
        ("same_company_nonexception", "independent_attribute", "unrelated_company")
    ):
        for order, color in (("SA", "#2874a6"), ("AS", "#d35400")):
            for intervention, style in (
                ("donor", "-"),
                ("same_answer_donor", "--"),
                ("wrong_donor", ":"),
                ("random_equal_norm", "-."),
            ):
                values = (
                    np.array(
                        [
                            [
                                lookup[
                                    (
                                        f"world-{world}-seed-{seed}-{order}",
                                        "BF16 autocast",
                                        13280,
                                        layer,
                                        intervention,
                                    )
                                ][field]
                                for layer in range(8)
                            ]
                            for world in (0, 1)
                            for seed in (0, 1)
                        ]
                    )
                    * 100
                )
                axes[col].plot(
                    range(1, 9), values.mean(0), style, color=color, label=f"{order} {intervention}"
                )
        axes[col].set(
            title=field.replace("_", " "),
            xlabel="Layer (one-based)",
            ylabel="Donor probability change (pp)",
        )
        axes[col].grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Endpoint deterministic and random controls · wrong donor scored against original donor"
    )
    save(fig, out, "organization-controls")
    print(
        json.dumps(
            {
                "measurements": len(measurements),
                "probe_conditions": sum(m["probes"] for m in measurements),
                "edit_geometry_records": len(gradients),
            }
        )
    )


if __name__ == "__main__":
    main()
