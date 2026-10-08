"""Paired-world differences, common baseline mastery and exportable figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "report/summary.json").read_text())
    by_world = defaultdict(dict)
    for row in summary["editing_worlds"]:
        if row["arm"] == "edit":
            by_world[(row["phase"], row["world"], row["memory"], row["role"], row["stratum"])][
                row["training"]
            ] = row
    differences = []
    for (phase, world, memory, role, stratum), arms in by_world.items():
        if len(arms) != 3:
            continue
        for control in ["fixed", "changed_atomic"]:
            differences.append(
                dict(
                    phase=phase,
                    world=world,
                    memory=memory,
                    role=role,
                    stratum=stratum,
                    treatment="changed_use",
                    control=control,
                    D_difference=arms["changed_use"]["D"] - arms[control]["D"],
                    E_difference=arms["changed_use"]["E"] - arms[control]["E"],
                    U_difference=arms["changed_use"]["U"] - arms[control]["U"],
                )
            )
    # Exploratory cohort: common original-answer mastery across all six parents.
    # This conditions on training outcomes; full graph-selected D remains primary.
    cohorts = defaultdict(dict)
    for row in summary["editing_cases"]:
        if row["arm"] == "edit":
            cohorts[(row["phase"], row["world"], row["case"], row["role"], row["stratum"])][
                (row["memory"], row["training"])
            ] = row
    common_rows = []
    for (phase, world, case, role, stratum), arms in cohorts.items():
        if len(arms) != 6:
            continue
        loaded = {}
        task = f"D_{role}_{stratum}"
        for key, row in arms.items():
            directory = root / "runs" / ("edit-" + row["parent"]) / case
            with np.load(directory / "lr0.0001-edit/predictions-000512.npz") as saved:
                loaded[key] = {
                    name: saved[name].copy()
                    for name in [
                        task + "__parent_correct",
                        task + "__correct",
                        task + "__old_rows",
                        task + "__new_rows",
                    ]
                }
        reference = next(iter(loaded.values()))
        for values in loaded.values():
            for field in ["old_rows", "new_rows"]:
                np.testing.assert_array_equal(
                    values[task + "__" + field], reference[task + "__" + field]
                )
        common = np.logical_and.reduce([v[task + "__parent_correct"] for v in loaded.values()])
        for (memory, training), values in loaded.items():
            common_rows.append(
                dict(
                    phase=phase,
                    world=world,
                    case=case,
                    role=role,
                    stratum=stratum,
                    memory=memory,
                    training=training,
                    n=len(common),
                    common_n=int(common.sum()),
                    coverage=float(common.mean()),
                    D=float(values[task + "__correct"][common].mean()) if common.any() else None,
                )
            )
    analysis = dict(
        paired_world_differences=differences,
        exploratory_common_mastery=common_rows,
        common_subset_warning=(
            "Post-training cohort across all six parents; exploratory, "
            "not primary and not independent replicates"
        ),
    )
    write_json(root / "report/analysis.json", analysis)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for phase in sorted({r["phase"] for r in summary["endpoints"]}):
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
            labels = ["Fixed", "Changed: atomics", "Changed: use"]
            arms = ["fixed", "changed_atomic", "changed_use"]
            for memory, offset, color in [
                ("local", -0.17, "#5975a4"),
                ("shared_full", 0.17, "#cc8963"),
            ]:
                base = {
                    r["training"]: r
                    for r in summary["endpoint_means"]
                    if r["phase"] == phase and r["memory"] == memory
                }
                if len(base) != 3:
                    continue
                axes[0].bar(
                    np.arange(3) + offset,
                    [base[a]["familiar"] * 100 for a in arms],
                    width=0.32,
                    label=memory,
                    color=color,
                )
                for axis, role in zip(axes[1:], ["first", "second"], strict=True):
                    rows = {
                        r["training"]: r
                        for r in summary["editing_means"]
                        if r["phase"] == phase
                        and r["memory"] == memory
                        and r["role"] == role
                        and r["stratum"] == "familiar"
                        and r["arm"] == "edit"
                    }
                    if len(rows) == 3:
                        axis.bar(
                            np.arange(3) + offset,
                            [rows[a]["D"] * 100 for a in arms],
                            width=0.32,
                            color=color,
                        )
            for axis, title in zip(
                axes,
                [
                    "Original familiar composition",
                    "After unseen first-hop edits",
                    "After unseen second-hop edits",
                ],
                strict=True,
            ):
                axis.set_title(title, fontsize=10)
                axis.set_xticks(np.arange(3), labels, rotation=15, fontsize=8)
                axis.set_ylim(0, 105)
                axis.set_ylabel("Full-answer accuracy (%)")
                axis.spines[["top", "right"]].set_visible(False)
            axes[0].legend(fontsize=8)
            fig.suptitle(
                phase + ": equal-world means; factual addresses reserved before training",
                fontsize=10,
            )
            fig.tight_layout()
            for extension in ["png", "pdf"]:
                fig.savefig(root / "report" / f"{phase}.{extension}", dpi=160)
            plt.close(fig)
    except ImportError:
        analysis["plot_unavailable"] = True
        write_json(root / "report/analysis.json", analysis)
    artifact = root.parents[1] / "docs/development-artifacts" / root.name / "postprocessing-source"
    artifact.mkdir(parents=True, exist_ok=True)
    path = Path(__file__).resolve()
    target = artifact / path.name
    if target.exists() and target.read_bytes() != path.read_bytes():
        raise FileExistsError("Do not replace a different postprocessing snapshot")
    if not target.exists():
        shutil.copy2(path, target)
    write_json(
        root / "postprocessing-provenance.json",
        dict(
            utc=utc(),
            path=str(target),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            exploratory_common_subset=True,
        ),
    )
    print(json.dumps(dict(paired_worlds=len(differences), common_case_rows=len(common_rows))))


if __name__ == "__main__":
    main()
