"""Summarize fixed comparisons after the registered matrix and audits finish."""

import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json

ROOT = Path("/ossfs/workspace/llm-memory-editability/results/shared-cache-branch-v1")
ARMS = ("local", "shared_full", "shared_window", "shared_detached")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((ROOT / "report/summary.json").read_text())
    assert summary["passed"] and not summary["missing"]
    rows = [r for r in summary["endpoints"] if r["phase"] == "confirmation"]
    aggregate, edit_rows, intervention_rows = [], [], []
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        aggregate.append(
            dict(
                arm=arm,
                worlds=3,
                **{
                    key + "_mean": float(np.mean([r[key] for r in selected]))
                    for key in ["atomic", "train", "familiar", "strict", "two_calls"]
                },
            )
        )
        for stratum in ["familiar", "strict"]:
            for role in ["first", "second"]:
                selected = [
                    r
                    for r in summary["edit_branches"]
                    if r["phase"] == "confirmation"
                    and r["memory_arm"] == arm
                    and r["stratum"] == stratum
                    and r["role"] == role
                ]
                edited = [r for r in selected if r["arm"] == "edit"]
                sham = [r for r in selected if r["arm"] == "sham"]
                assert len(edited) == len(sham) == 3
                key = f"D_{role}_{stratum}"
                edit_rows.append(
                    dict(
                        memory_arm=arm,
                        role=role,
                        stratum=stratum,
                        cases=3,
                        E_new=float(np.mean([r["metrics"]["E_new"]["accuracy"] for r in edited])),
                        D_new=float(np.mean([r["metrics"][key]["accuracy"] for r in edited])),
                        D_sham=float(np.mean([r["metrics"][key]["accuracy"] for r in sham])),
                        D_old=float(
                            np.mean([r["metrics"][key]["old_answer_accuracy"] for r in edited])
                        ),
                        prerequisite_coverage=float(
                            np.mean(
                                [r["metrics"][key]["operation_correct_coverage"] for r in edited]
                            )
                        ),
                        U=float(np.mean([r["metrics"]["U_atomic"]["accuracy"] for r in edited])),
                        case_metrics=[
                            dict(world=r["world"], metrics=r["metrics"][key]) for r in edited
                        ],
                    )
                )
    names = {r["name"]: r for r in rows}
    for record in summary["inference_interventions"]:
        if record["name"] not in names:
            continue
        spec = names[record["name"]]
        normal = record["metrics"]["normal"]
        removed = record["metrics"]["shared_off"]
        folder = ROOT / "runs" / record["name"]
        with (
            np.load(folder / "intervention-normal.npz") as native,
            np.load(folder / "intervention-shared_off.npz") as ablated,
        ):
            common = native["familiar_test_coverage"] & ablated["familiar_test_coverage"]
            conditional = dict(
                exploratory=True,
                n=int(common.sum()),
                coverage=float(common.mean()),
                normal=float(native["familiar_test_correct"][common].mean())
                if common.any()
                else None,
                shared_off=float(ablated["familiar_test_correct"][common].mean())
                if common.any()
                else None,
                limitation="Post-treatment common subset; does not replace full-pool scores.",
            )
        intervention_rows.append(
            dict(
                world=spec["world"],
                arm=spec["arm"],
                familiar_normal=normal["familiar_test"]["accuracy"],
                familiar_shared_off=removed["familiar_test"]["accuracy"],
                atomic_shared_off=removed["common_atomic"]["accuracy"],
                common_prerequisite_subset=conditional,
            )
        )
    result = dict(
        passed=True,
        confirmation_means=aggregate,
        edit_propagation=edit_rows,
        inference_interventions=intervention_rows,
        weighting="Equal worlds; graph-selected cases are not independent worlds.",
        editing_gradient_limit=(
            "shared_detached retains detached cache reads during editing too. "
            "Its edit comparison combines learned-parent and editing-gradient differences; "
            "it cannot isolate pretraining feedback alone."
        ),
    )
    write_json(ROOT / "report/analysis.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key in zip(axes, ["E_new", "D_new", "U"], strict=True):
        for i, arm in enumerate(ARMS):
            values = [
                100 * r[key]
                for r in edit_rows
                if r["memory_arm"] == arm and r["stratum"] == "familiar"
            ]
            ax.bar(i, np.mean(values), alpha=0.65)
            ax.scatter([i] * len(values), values, color="black", s=20)
        ax.set(
            xticks=range(4),
            xticklabels=["Local", "Shared", "Window", "Detached"],
            ylabel="Full generated answer (%)",
            title=key,
            ylim=(-2, 103),
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("512-step local MLP edits | familiar facts | dots are first/second-role means")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(ROOT / "report" / f"edit-propagation.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
