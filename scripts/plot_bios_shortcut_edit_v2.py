"""Plot every prespecified E39 checkpoint, with endpoint integer denominators."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source):
    source = Path(source).resolve()
    summary = source / "summary"
    audit = json.loads((summary / "audit.json").read_text())
    if not audit["complete"] or audit["cases"] != 96 or audit["missing"]:
        raise ValueError("A complete independently rescored E39 matrix is required")
    rows = []
    for row in csv.DictReader((summary / "editing.csv").open()):
        for key, value in row.items():
            try:
                row[key] = float(value) if value else None
            except ValueError:
                pass
        rows.append(row)
    metrics = {
        "roots": ("E_roots_accuracy", "E_roots_correct", "E_roots_n"),
        "actual": ("E_actual_accuracy", "E_actual_correct", "E_actual_n"),
        "paired_D9": (
            "paired_reference_D_heldout_accuracy",
            "paired_reference_D_heldout_correct",
            "paired_reference_D_heldout_n",
        ),
        "U_global_damage": ("U_full_damage", "U_full_broken", "U_full_known"),
        "U_local_damage": (
            "U_full_strata_0_damage",
            "U_full_strata_0_broken",
            "U_full_strata_0_known",
        ),
        "U_global_coverage": ("U_full_coverage", "U_full_known", "U_full_n"),
        "U_local_coverage": (
            "U_full_strata_0_coverage",
            "U_full_strata_0_known",
            "U_full_strata_0_n",
        ),
    }
    checkpoints = (0, 32, 128, 512)
    table = []
    for phase in ("low", "high"):
        for kind in ("coherent", "exception"):
            for step in checkpoints:
                group = [
                    r for r in rows if (r["phase"], r["kind"], r["step"]) == (phase, kind, step)
                ]
                if len(group) != 24:
                    raise ValueError("Trajectory plotting cannot omit any case")
                for metric, (rate, numerator, denominator) in metrics.items():
                    values = [r[rate] for r in group if r[rate] is not None]
                    table.append(
                        dict(
                            phase=phase,
                            kind=kind,
                            step=step,
                            metric=metric,
                            cases=len(group),
                            valid_cases=len(values),
                            case_mean=mean(values) if values else None,
                            numerator=sum(int(r[numerator]) for r in group),
                            denominator=sum(int(r[denominator]) for r in group),
                        )
                    )
    with (summary / "trajectory-figure-data.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lookup = {(r["phase"], r["kind"], r["step"], r["metric"]): r for r in table}
    colors = {"low": "#3671a8", "high": "#cd5b32"}
    phase_labels = {"low": "Low (2/32)", "high": "High (16/32)"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.2))
    panels = (
        (("roots", "roots", "-"), ("actual", "actual", "--")),
        (("paired_D9", "same 9 people", "-"),),
        (("U_global_damage", "global", "-"), ("U_local_damage", "local old exceptions", "--")),
        (("U_global_coverage", "global", "-"), ("U_local_coverage", "local old exceptions", "--")),
    )
    for i, kind in enumerate(("coherent", "exception")):
        for j, panel in enumerate(panels):
            ax = axes[i, j]
            for phase in ("low", "high"):
                for metric, label, style in panel:
                    values = [
                        lookup[phase, kind, step, metric]["case_mean"] for step in checkpoints
                    ]
                    ax.plot(
                        range(4),
                        [100 * v if v is not None else float("nan") for v in values],
                        style,
                        marker="o",
                        markersize=4,
                        color=colors[phase],
                        label=f"{phase_labels[phase]}: {label}",
                    )
            ax.set_xticks(range(4), [str(step) for step in checkpoints])
            ax.set_xlabel("Fixed edit checkpoint")
            ax.set_ylabel("Case-mean rate (%)")
            if j != 2:
                ax.set_ylim(-3, 103)
            else:
                ax.set_ylim(bottom=-0.25)
            ax.grid(alpha=0.2)
            title = (
                "Edited facts E",
                "Heldout D: paired reference"
                if kind == "coherent"
                else "Heldout D: exception conflict",
                "Retained knowledge damaged",
                "Knowledge known before editing",
            )[j]
            ax.set_title(f"{kind.capitalize()} | {title}", fontsize=10)
            ax.legend(fontsize=7, loc="best", framealpha=0.85)
            count_lines = []
            for phase in ("low", "high"):
                pieces = []
                for metric, label, _style in panel:
                    row = lookup[phase, kind, 512, metric]
                    pieces.append(f"{label}: {row['numerator']:,}/{row['denominator']:,}")
                count_lines.append(f"{phase}: " + "; ".join(pieces))
            ax.text(
                0,
                -0.25,
                "Step512 totals\n" + "\n".join(count_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
    fig.suptitle("Common-support E39: fixed trajectories of every low/high parent", fontsize=15)
    fig.text(
        0.5,
        0.935,
        "24 parent models, 96 edits; 24 cases per phase/update type; "
        "2 development worlds × 2 initializations",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.025,
        0.015,
        "E counts = correct/total; D counts = correct/total (9 per case); "
        "damage counts = broken/old-correct; "
        "coverage counts = old-correct/pool.\n"
        "Rates weight cases equally; integer totals provide denominators "
        "and need not equal the mean rate. "
        "Local U = unchanged actual-city facts of original exceptions "
        "in the 3 edited groups (6 per case).",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.885, bottom=0.16, hspace=0.7, wspace=0.29)
    for suffix in ("png", "pdf"):
        fig.savefig(summary / f"E39-fixed-trajectories.{suffix}", dpi=180)
    plt.close(fig)
    print(
        json.dumps({"source": str(source), "rows": len(table), "figure": "E39-fixed-trajectories"})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/bios-mechanism-dev-v1/p3-shortcut-edit-v2")
    plot(parser.parse_args().source)
