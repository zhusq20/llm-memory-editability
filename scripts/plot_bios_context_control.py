"""Descriptive context-control figures from a complete, archived paired audit."""

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
DEFAULT = ROOT / "docs/development-artifacts/mechanism-v1/p3-context"
PHASES = ("unrestricted", "isolated")
COLORS = ("#285A8E", "#C66A0A")
STEPS = (5120, 10240, 15360)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(source=DEFAULT, output=DEFAULT):
    source, output = Path(source), Path(output)
    names = (
        "audit.json",
        "path-summary.csv",
        "paired-summary.csv",
        "component-summary.csv",
        "base-summary.csv",
        "organization-summary.csv",
    )
    manifest = json.loads((source / "archive-manifest.json").read_text())
    for name in names:
        if sha(source / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Archived summary changed: {name}")
    audit = json.loads((source / "audit.json").read_text())
    if not audit["complete"] or audit["complete_pairs"] != 12:
        raise ValueError("Plot requires the full audited matrix")
    tables = {}
    for name in names[1:]:
        with (source / name).open() as stream:
            tables[name] = list(csv.DictReader(stream))

    def cell(name, **keys):
        found = [r for r in tables[name] if all(r[k] == str(v) for k, v in keys.items())]
        if len(found) != 1:
            raise ValueError(f"Ambiguous or missing plotted cell: {name}/{keys}")
        return found[0]

    def path(phase, cohort, step=15360, method="direct", world="all"):
        return cell(
            "path-summary.csv",
            world=world,
            step=step,
            phase=phase,
            method=method,
            split="heldout",
            cohort=cohort,
        )

    data = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.8))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)

    ax = axes[0, 0]
    for phase, color in zip(PHASES, COLORS, strict=True):
        for cohort, style, marker, label in (
            ("original_exception", "-", "o", "Original exceptions"),
            ("ordinary", "--", "s", "Ordinary people"),
        ):
            values = [100 * float(path(phase, cohort, step)["accuracy"]) for step in STEPS]
            ax.plot(
                np.arange(3),
                values,
                color=color,
                linestyle=style,
                marker=marker,
                linewidth=1.8,
                label=f"{phase.title()}: {label}",
            )
            data.extend(
                dict(panel="a", phase=phase, cohort=cohort, step=step, accuracy_percent=value)
                for step, value in zip(STEPS, values, strict=True)
            )
    ax.set_title("(a) Direct queries: same held-out people", loc="left", pad=10)
    ax.set_xticks(np.arange(3), ("5,120", "10,240", "15,360"))
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.66), frameon=False)

    ax = axes[0, 1]
    labels = ("Correct default", "Wrong actual", "Other error (incl. EOS)")
    colors = ("#34816C", "#C66A0A", "#9CA5B0")
    for idx, phase in enumerate(PHASES):
        row = path(phase, "original_exception")
        correct, actual = 100 * float(row["accuracy"]), 100 * float(row["wrong_actual_per_query"])
        values = (correct, actual, 100 - correct - actual)
        bottom = 0
        for label, color, value in zip(labels, colors, values, strict=True):
            ax.bar(idx, value, 0.5, bottom=bottom, color=color, label=label if idx == 0 else "")
            if value >= 4:
                ax.text(
                    idx,
                    bottom + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if label != labels[2] else "#222222",
                    fontsize=9,
                )
            data.append(dict(panel="b", phase=phase, answer=label, percent=value))
            bottom += value
    ax.set_title("(b) Original exceptions: endpoint answers", loc="left", pad=10)
    ax.set_xticks((0, 1), ("Unrestricted", "Fact-isolated"))
    ax.set_xlim(-0.6, 2.9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("All original-exception queries (%)")
    ax.legend(loc="center right", bbox_to_anchor=(1.03, 0.5), fontsize=8, frameon=False)

    ax = axes[1, 0]
    x, width = np.arange(5), 0.36
    for idx, (phase, color) in enumerate(zip(PHASES, COLORS, strict=True)):
        row = cell(
            "component-summary.csv",
            world="all",
            step=15360,
            phase=phase,
            split="heldout",
            cohort="original_exception",
        )
        values = [
            100 * float(row[key])
            for key in ("actual_accuracy", "membership_accuracy", "root_accuracy")
        ]
        values.extend(
            100 * float(path(phase, "original_exception", method=method)["accuracy"])
            for method in ("direct", "two_step")
        )
        bars = ax.bar(x + (idx - 0.5) * width, values, width, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=7.5)
        data.extend(
            dict(panel="c", phase=phase, measure=key, accuracy_percent=value)
            for key, value in zip(
                ("actual", "membership", "root", "direct", "two_step"), values, strict=True
            )
        )
    ax.set_title("(c) Original exceptions: facts and composition", loc="left", pad=10)
    ax.set_xticks(
        x, ("Actual", "Member-\nship", "Person's\nroot", "Direct", "Autonomous\ntwo-step")
    )
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy on the same people (%)")

    ax = axes[1, 1]
    for phase, color in zip(PHASES, COLORS, strict=True):
        for world in ("0", "1", "all"):
            values = [
                float(cell("base-summary.csv", world=world, step=step, phase=phase)["base_nll"])
                for step in STEPS
            ]
            ax.plot(
                np.arange(3),
                values,
                color=color,
                marker="o" if world == "all" else None,
                linewidth=2 if world == "all" else 0.8,
                alpha=1 if world == "all" else 0.4,
                linestyle="-" if world == "all" else "--",
            )
            data.extend(
                dict(panel="d", phase=phase, world=world, step=step, base_nll=value)
                for step, value in zip(STEPS, values, strict=True)
            )
    ax.set_title("(d) Independent base facts: NLL", loc="left", pad=10)
    ax.set_xticks(np.arange(3), ("5,120", "10,240", "15,360"))
    ax.set_xlabel("Training steps; faint dashed lines: individual worlds")
    ax.set_ylabel("Mean value-token NLL (lower is better)")
    ax.set_ylim(bottom=0)

    ax = axes[2, 0]
    world_colors = ("#8063A0", "#267E83")
    effect_keys = ("matching_effect", "company_unmatched_benefit", "project_unmatched_benefit")
    for idx, (world, color) in enumerate(zip(("0", "1"), world_colors, strict=True)):
        for phase, marker, offset in (("unrestricted", "o", -0.14), ("isolated", "s", 0.14)):
            row = cell(
                "organization-summary.csv",
                world=world,
                step=15360,
                phase=phase,
                method="direct",
                split="heldout",
                cohort="all",
            )
            values = [100 * float(row[key]) for key in effect_keys]
            xx = np.arange(3) + (idx - 0.5) * 0.1 + offset
            ax.scatter(
                xx, values, color=color, marker=marker, s=32, label=f"World {world}, {phase}"
            )
            data.extend(
                dict(panel="e", world=world, phase=phase, effect=key, value_pp=value)
                for key, value in zip(effect_keys, values, strict=True)
            )
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title("(e) Organization effects: all held-out queries", loc="left", pad=10)
    ax.set_xticks(
        np.arange(3),
        (
            "Matching\ninteraction",
            "Company org.\nunmatched benefit",
            "Project org.\nunmatched benefit",
        ),
    )
    ax.set_ylabel("Accuracy difference (percentage points)")
    ax.legend(fontsize=7.5, loc="best", frameon=False)

    ax = axes[2, 1]
    x = np.arange(3)
    for idx, (phase, color) in enumerate(zip(PHASES, COLORS, strict=True)):
        values = [
            100 * float(path(phase, "all", world=world)["accuracy"]) for world in ("0", "1", "all")
        ]
        bars = ax.bar(x + (idx - 0.5) * width, values, width, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=8.5)
        data.extend(
            dict(panel="f", world=world, phase=phase, accuracy_percent=value)
            for world, value in zip(("0", "1", "all"), values, strict=True)
        )
    ax.set_title("(f) Endpoint direct accuracy: all held-out queries", loc="left", pad=10)
    ax.set_xticks(x, ("World 0", "World 1", "Both worlds"))
    ax.set_ylim(0, 111)
    ax.set_ylabel("Accuracy (%)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(
        handles,
        ("Original unrestricted training", "Fact-isolated document training"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.054),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.033,
        "7.58M parameters; 12 paired models; both query relations and worlds retained. "
        "Case means; no confidence intervals.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.text(
        0.5,
        0.015,
        "Same low-exception world, targets and training budget; inference remains "
        "unrestricted. Autonomous two-step measured only at 15,360 steps.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.96, wspace=0.28, hspace=0.4)
    output.mkdir(parents=True, exist_ok=True)
    stem = "p3-context-fixed-person-control"
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)
    (output / f"{stem}.json").write_text(
        json.dumps(
            {
                "sources_hashes": {name: sha(source / name) for name in names},
                "script_sha256": sha(Path(__file__)),
                "data": data,
                "aggregation": "Equal case means; two reused development worlds; "
                "no query-level intervals",
                "limitations": "Attention removal changes training computation and may "
                "affect optimization; a smaller direct/two-step gap is not selective "
                "repair if basic knowledge declines.",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    plot(args.source, args.output)


if __name__ == "__main__":
    main()
