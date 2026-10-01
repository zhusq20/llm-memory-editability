"""Publication-format descriptive P0 panels from the archived, audited tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/development-artifacts/mechanism-v1/p0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(source=DEFAULT, output=DEFAULT):
    source, output = Path(source), Path(output)
    for name in (
        "audit.json",
        "summary/two-step-audit.json",
        "h2-analysis.json",
        "h2-two-step-analysis.json",
    ):
        if not json.loads((source / name).read_text())["complete"]:
            raise ValueError(f"Cannot plot unaudited data: {name}")
    path = source / "h2-two-step.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    widths = (64, 128, 256, 768)
    panels = (
        ("learning", "", "QA/heldout/old_exception", "(a) Learning: original exceptions"),
        ("editing", "mlp", "D/heldout/manipulated_conflict_cohort", "(b) Conflict editing: MLP"),
        (
            "editing",
            "all",
            "D/heldout/manipulated_conflict_cohort",
            "(c) Conflict editing: all weights",
        ),
    )
    table = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.25), sharey=True)
    colors = ("#285A8E", "#C66A0A")
    for ax, (phase, scope, subset, title) in zip(axes, panels, strict=True):
        selected = []
        for width in widths:
            matching = [
                r
                for r in rows
                if int(r["width"]) == width
                and r["alignment"] == "all"
                and r["phase"] == phase
                and r["scope"] == scope
                and r["subset"] == subset
                and int(r["step"]) == (15360 if phase == "learning" else 512)
                and r["kind"] == ("" if phase == "learning" else "exception")
            ]
            if len(matching) != 1 or int(matching[0]["cases"]) != 24:
                raise ValueError(f"Incomplete plotted cell: {phase}/{scope}/{width}")
            selected.append(matching[0])
        x = np.arange(len(widths))
        for key, label, color, marker, line in (
            ("direct_correct_per_query", "Direct composed query", colors[0], "o", "-"),
            ("two_step_correct_per_query", "Autonomous two-step", colors[1], "s", "--"),
        ):
            values = np.array([float(row[key]) * 100 for row in selected])
            ax.plot(
                x,
                values,
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.9,
                markersize=5.5,
                label=label,
            )
            for width, value in zip(widths, values, strict=True):
                table.append(
                    dict(
                        phase=phase,
                        scope=scope,
                        width=width,
                        method=label,
                        accuracy_percent=float(value),
                        cases=24,
                    )
                )
        ax.set_title(title, loc="left", pad=10)
        ax.set_xticks(x, ("0.72M", "2.22M", "7.58M", "60.49M"))
        ax.set_xlabel("Model parameters")
        ax.set_ylim(0, 105)
        ax.set_xlim(-0.18, 3.18)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
    axes[0].set_ylabel("Held-out composed-query accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=2, frameon=False
    )
    fig.text(
        0.5,
        0.025,
        "Case-weighted means: 24 model-relation / edit cases per point. "
        "No confidence intervals shown.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.27, top=0.88, wspace=0.2)
    output.mkdir(parents=True, exist_ok=True)
    stem = "p0-direct-vs-autonomous-two-step"
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)
    metadata = {
        "source": str(path.resolve()),
        "source_sha256": sha(path),
        "script_sha256": sha(Path(__file__).resolve()),
        "data": table,
        "population": "Learning: original-exception heldout. Editing: original 45-person "
        "manipulated-conflict cohort, heldout subset, exception edits only.",
        "aggregation": "Equal model-chain/edit-case weighting; all organizations, worlds and "
        "initializations retained. Repeated queries are not independent replicates.",
        "intervals": "None; no query-level confidence intervals.",
        "generation": "First hop is model generated; no true bridge fed to the second call. "
        "Both calls require value plus EOS. Extra calls change inference cost.",
    }
    (output / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
