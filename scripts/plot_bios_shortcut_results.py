"""Descriptive P3 panels: fixed people, answer composition, and available knowledge."""

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
DEFAULT = ROOT / "docs/development-artifacts/mechanism-v1/p3"
COHORTS = ("original_exception", "newly_exception", "remaining_ordinary", "all")
LABELS = ("Original\nexceptions", "Newly\nexceptions", "Remaining\nordinary", "All\nheld-out")
PHASES = ("low", "high")
COLORS = ("#285A8E", "#C66A0A")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def plot(source=DEFAULT, output=DEFAULT):
    source, output = Path(source), Path(output)
    audit_names = ("audit.json", "interpretation-audit.json", "component-audit.json")
    names = (*audit_names, "endpoint-by-world.csv", "components-by-world.csv")
    manifest = json.loads((source / "archive-manifest.json").read_text())
    for name in names:
        if sha(source / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Archived source changed: {name}")
    for name in audit_names:
        if not json.loads((source / name).read_text())["complete"]:
            raise ValueError(f"Incomplete source audit: {name}")
    endpoints = read_csv(source / "endpoint-by-world.csv")
    components = read_csv(source / "components-by-world.csv")

    def endpoint(world, cohort):
        rows = [r for r in endpoints if r["world"] == world and r["cohort"] == cohort]
        if len(rows) != 1 or int(rows[0]["cases"]) != (24 if world == "all" else 12):
            raise ValueError(f"Incomplete endpoint: {world}/{cohort}")
        return rows[0]

    def component(phase):
        rows = [
            r
            for r in components
            if r["world"] == "all" and r["cohort"] == "original_exception" and r["phase"] == phase
        ]
        if len(rows) != 1 or int(rows[0]["cases"]) != 24:
            raise ValueError(f"Incomplete component: {phase}")
        return rows[0]

    original = endpoint("all", "original_exception")
    data = []
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
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.set_axisbelow(True)

    ax = axes[0, 0]
    x = np.arange(4)
    width = 0.36
    for idx, (phase, color) in enumerate(zip(PHASES, COLORS, strict=True)):
        values = [100 * float(endpoint("all", c)[f"{phase}_direct"]) for c in COHORTS]
        bars = ax.bar(x + (idx - 0.5) * width, values, width, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=9)
        data.extend(
            {"panel": "a", "phase": phase, "cohort": cohort, "accuracy_percent": value}
            for cohort, value in zip(COHORTS, values, strict=True)
        )
    ax.set_title("(a) Direct queries: fixed held-out people", loc="left", pad=10)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 111)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)

    ax = axes[0, 1]
    composition_colors = ("#34816C", "#C66A0A", "#9CA5B0")
    composition_labels = ("Correct default", "Wrong actual", "Other error (incl. EOS)")
    for idx, phase in enumerate(PHASES):
        correct = 100 * float(original[f"{phase}_direct"])
        actual = 100 * float(original[f"{phase}_actual_on_conflict"])
        values = (correct, actual, 100 - correct - actual)
        bottom = 0
        for value, color, label in zip(values, composition_colors, composition_labels, strict=True):
            ax.bar(idx, value, 0.5, bottom=bottom, color=color, label=label if idx == 0 else "")
            ax.text(
                idx,
                bottom + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if label != composition_labels[2] else "#222222",
                fontsize=10,
            )
            data.append({"panel": "b", "phase": phase, "answer": label, "percent": value})
            bottom += value
    ax.set_title("(b) Original exceptions: answer composition", loc="left", pad=10)
    ax.set_xticks((0, 1), ("Low exception rate", "High exception rate"))
    ax.set_ylabel("All original-exception queries (%)")
    ax.set_ylim(0, 106)
    ax.set_xlim(-0.6, 2.9)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="center right", bbox_to_anchor=(1.03, 0.5), fontsize=8.2, frameon=False)

    ax = axes[1, 0]
    knowledge_labels = ("Actual\nfact", "Membership\nfact", "Root\nfact", "Autonomous\ntwo-step")
    for idx, (phase, color) in enumerate(zip(PHASES, COLORS, strict=True)):
        row = component(phase)
        values = [100 * float(row[key]) for key in ("actual", "membership", "default")]
        values.append(100 * float(original[f"{phase}_two_step"]))
        bars = ax.bar(x + (idx - 0.5) * width, values, width, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8.5)
        data.extend(
            {"panel": "c", "phase": phase, "measure": label, "accuracy_percent": value}
            for label, value in zip(knowledge_labels, values, strict=True)
        )
    ax.set_title("(c) Original exceptions: accessible knowledge", loc="left", pad=10)
    ax.set_xticks(x, knowledge_labels)
    ax.set_ylabel("Accuracy on the same people (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)

    ax = axes[1, 1]
    world_colors = ("#8063A0", "#267E83")
    for idx, (world, color) in enumerate(zip(("0", "1"), world_colors, strict=True)):
        values = [100 * float(endpoint(world, c)["direct_change"]) for c in COHORTS]
        ys = x + (idx - 0.5) * 0.24
        ax.scatter(values, ys, marker=("o", "s")[idx], color=color, s=38, label=f"World {world}")
        for y, value, cohort in zip(ys, values, COHORTS, strict=True):
            ax.annotate(
                f"{value:+.1f}",
                (value, y),
                xytext=(5 if value >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8.5,
                color=color,
            )
            data.append({"panel": "d", "world": world, "cohort": cohort, "change_pp": value})
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_title("(d) Direct-query changes in each world", loc="left", pad=10)
    ax.set_yticks(
        x, ("Original exceptions", "Newly exceptions", "Remaining ordinary", "All held-out")
    )
    ax.set_ylim(3.55, -0.55)
    ax.set_xlim(-32, 51)
    ax.set_xlabel("High minus low exception rate (percentage points)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(
        handles,
        ("Low: 2/32 exceptions per group", "High: 16/32 exceptions per group"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.047,
        "7.58M parameters; 12 paired models, 2 query relations; "
        "all organization/seed/world cells retained.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.5,
        0.025,
        "Equal case means; fixed people and derived targets; complete value + EOS scoring. "
        "No confidence intervals shown.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.185, top=0.945, wspace=0.34, hspace=0.4)
    output.mkdir(parents=True, exist_ok=True)
    stem = "p3-fixed-person-shortcut-control"
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)
    metadata = {
        "sources_hashes": {name: sha(source / name) for name in names},
        "source_root": str(source.resolve()),
        "script_sha256": sha(Path(__file__).resolve()),
        "data": data,
        "aggregation": "Equal model-relation cases; repeated queries "
        "are not independent replicates.",
        "intervals": "None; no query-level confidence intervals.",
        "bridge": "Autonomous predicted organization only; no true bridge in two-step inference.",
        "root_fact_measure": "True organization indexes the separate root-fact score only.",
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
