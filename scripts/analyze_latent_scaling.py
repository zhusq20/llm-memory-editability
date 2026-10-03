"""Describe retained learning trajectories without forcing grokking labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import file_hash

ARTIFACTS = Path("docs/development-artifacts/latent-scaling-v1")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    rows = []
    for endpoint in summary["runs"]:
        nodes = summary["histories"][endpoint["name"]]
        fit = next(
            (
                n
                for n in nodes
                if n["metrics"]["common_atomic"]["accuracy"] >= 0.99
                and n["metrics"]["train_composite"]["accuracy"] >= 0.99
            ),
            None,
        )

        def score(n):
            return n["metrics"]["familiar_test"]["accuracy"]

        row = {
            "name": endpoint["name"],
            "width": endpoint["width"],
            "layers": endpoint["layers"],
            "repeats": endpoint["repeats"],
            "composition_examples": endpoint["composition_examples"],
            "joint_fit99_first_saved_step": fit["step"] if fit else None,
            "test_at_joint_fit": score(fit) if fit else None,
            "test_at_32000": score(next(n for n in nodes if n["step"] == 32000)),
            "final_test": score(nodes[-1]),
            "maximum_saved_test": max(map(score, nodes)),
            "post_fit_change_pp": 100 * (score(nodes[-1]) - score(fit)) if fit else None,
        }
        rows.append(row)
    with (ARTIFACTS / "trajectory-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        ARTIFACTS / "trajectory-analysis.json",
        {
            "created_utc": utc(),
            "scope": "descriptive retained-node analysis; primary endpoints unchanged",
            "source_sha256": file_hash(__file__),
            "summary_sha256": file_hash(ARTIFACTS / "summary.json"),
            "runs": len(rows),
            "joint_fit99_observed_runs": sum(
                r["joint_fit99_first_saved_step"] is not None for r in rows
            ),
            "saved_node_test90_observed_runs": sum(r["maximum_saved_test"] >= 0.9 for r in rows),
            "final_test90_runs": sum(r["final_test"] >= 0.9 for r in rows),
            "note": "Crossing times refer to sparse saved nodes, not exact transition steps. "
            "Late improvement does not by itself establish an abrupt grokking transition.",
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for column, width in enumerate((128, 256)):
        selected = [
            r
            for r in summary["runs"]
            if r["width"] == width
            and r["layers"] == 1
            and r["composition_count"] == "all"
            and r["repeats"] in (1, 2, 3)
        ]
        for row in selected:
            nodes = summary["histories"][row["name"]]
            early = [n for n in nodes if n["step"] <= 8000]
            for task, style in (("common_atomic", "-"), ("train_composite", "--")):
                axes[0, column].plot(
                    [n["step"] for n in early],
                    [100 * n["metrics"][task]["accuracy"] for n in early],
                    style,
                    color=colors[row["repeats"]],
                    label=f"R{row['repeats']} {'facts' if style == '-' else 'train compositions'}",
                )
            axes[1, column].plot(
                [n["step"] for n in nodes],
                [100 * n["metrics"]["familiar_test"]["accuracy"] for n in nodes],
                marker="o",
                color=colors[row["repeats"]],
                label=f"train R{row['repeats']}",
            )
        axes[0, column].set_title(f"Width {width}: learning facts and fitting training")
        axes[1, column].set_title(f"Width {width}: generalizing after training fit")
    for ax in axes.flat:
        ax.set_xscale("symlog", linthresh=256)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Full generated accuracy (%)")
        ax.set_xlabel("Training updates; fixed 763 compositions")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"knowledge-and-use.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({"analyzed_runs": len(rows)}))


if __name__ == "__main__":
    main()
