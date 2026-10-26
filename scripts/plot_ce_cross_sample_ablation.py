"""Plot verified ablation summaries and describe actual parameter updates, without models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--original-diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    original = json.loads(args.original_diagnostics.read_text())
    if report.get("passed") is not True or original.get("passed") is not True:
        raise ValueError("Both supplied evidence bundles must pass their recorded audits")
    cloud_path = args.report.parent / "cloud-verification.json"
    cloud = json.loads(cloud_path.read_text())
    if cloud.get("passed") is not True or cloud["result_sha256"] != sha(args.report):
        raise ValueError("Comparison must have a bound independent cloud receipt")
    args.out.mkdir(parents=True, exist_ok=True)
    colors = {
        "control": "#555555",
        "lower-lr": "#0072B2",
        "carry-adam": "#009E73",
        "no-bound": "#D55E00",
    }
    labels = {
        "control": "Original recipe",
        "lower-lr": "Lower learning rate",
        "carry-adam": "Carry Adam state",
        "no-bound": "Remove elementwise bound",
    }
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), constrained_layout=True)
    geometry, ages = {}, {}
    selected = {2, 3, 4, 8, 16, 32, 64}
    for column, architecture in enumerate(("loop", "discoloop")):
        for variant in colors:
            arm = architecture + "-" + variant
            if arm not in report["summaries"]:
                continue
            summary = report["summaries"][arm]
            data = summary["age_retention"]["atoms"]
            axes[0, column].plot(
                [x["age"] for x in data],
                [
                    100 * x["initially_correct_retained"]["accuracy"]
                    if x["initially_correct_retained"]["accuracy"] is not None
                    else np.nan
                    for x in data
                ],
                color=colors[variant],
                label=labels[variant],
                lw=2,
            )
            trajectories = [summary["streams"][s]["trajectory"] for s in ("forward", "reverse")]
            for row, getter in [
                (1, lambda t: t["final_correct"]["both_atoms_and_composition"]["accuracy"]),
                (
                    2,
                    lambda t: t["old"]["categories"]["compositions"]["originally_correct_retained"][
                        "accuracy"
                    ],
                ),
            ]:
                xs, ys = [], []
                for index in range(64):
                    values = [getter(t[index]) for t in trajectories]
                    if all(v is not None for v in values):
                        xs.append(index + 1)
                        ys.append(100 * np.mean(values))
                axes[row, column].plot(xs, ys, color=colors[variant], lw=2)
            ages[arm] = {
                str(x["age"]): x for x in data if x["age"] in {0, 1, 2, 4, 8, 16, 32, 48, 63}
            }
            if variant == "control":
                rows = [
                    r
                    for r in original["architectures"][architecture]["parameter_inspection"]
                    if r["step"] in selected
                ]
                mapping = {
                    "per_fact_update_rms": "fact_update_rms",
                    "theta0_offset_rms": "offset_rms",
                    "original_region_saturation_fraction": "saturated_fraction",
                    "opposite_boundary_flip_fraction": "opposite_boundary_fraction_all",
                    "previous_offset_projection": "previous_offset_projection_retained",
                }
                geometry[arm] = {
                    "matched_subsequent_writes": len(rows),
                    "medians": {
                        key: float(np.median([r[source] for r in rows]))
                        for key, source in mapping.items()
                    },
                }
            else:
                rows = [
                    t["write_info"]["geometry"]
                    for stream in trajectories
                    for t in stream
                    if t["step"] in selected
                ]
                geometry[arm] = {
                    "matched_subsequent_writes": len(rows),
                    "medians": {key: float(np.median([r[key] for r in rows])) for key in rows[0]},
                }
        axes[0, column].set(
            title="Loop" if architecture == "loop" else "DiscoLoop",
            xlabel="Subsequent facts since writing",
            ylabel="Initially correct new atoms retained (%)",
            ylim=(-2, 102),
        )
        axes[0, column].set_xscale("symlog", linthresh=1)
        axes[0, column].set_xticks(
            [0, 1, 2, 4, 8, 16, 32, 63], labels=["0", "1", "2", "4", "8", "16", "32", "63"]
        )
        axes[1, column].set(
            xlabel="Facts written in stream",
            ylabel="Both atoms + double-new answer correct (%)",
            xlim=(32, 64),
            ylim=(-2, 102),
        )
        axes[2, column].set(
            xlabel="Facts written in stream",
            ylabel="Originally correct old compositions retained (%)",
            xlim=(0, 64),
            ylim=(0, 102),
        )
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, names, fontsize=9, loc="lower left")
    fig.suptitle(
        "Continuous new-fact learning: fixed architecture, one-factor interventions\n"
        "32 fact pairs; both stream orders; one observed graph and initialization",
        fontsize=13,
    )
    fig.savefig(args.out / "ablation-curves.png", dpi=170)
    fig.savefig(args.out / "ablation-curves.pdf")
    plt.close(fig)
    result = {
        "analysis": "read-only descriptive visualization of audited and cloud-verified results; "
        "no new selection",
        "optimizer_updates": 0,
        "model_constructions": 0,
        "matched_steps_per_stream": sorted(selected),
        "geometry": geometry,
        "age_retention": ages,
        "source_sha256": sha(__file__),
        "input_hashes": {
            str(p): sha(p) for p in (args.report, cloud_path, args.original_diagnostics)
        },
    }
    (args.out / "parameter-and-curve-analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
