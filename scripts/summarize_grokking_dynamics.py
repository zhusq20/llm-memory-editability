"""Describe fixed-budget contrasts and mechanism prerequisites; never select best runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json

ROOT = Path("docs/development-artifacts/grokking-dynamics-v1")


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main():
    endpoint_summary = json.loads((ROOT / "development/report/summary.json").read_text())
    learning = read_csv(ROOT / "development/report/learning.csv")
    aggregates = []
    for architecture in ["standard2", "loop2"]:
        for decay in [0, 0.01, 0.1]:
            endpoint = [
                r
                for r in endpoint_summary["endpoints"]
                if r["architecture"] == architecture and r["weight_decay"] == decay
            ]
            row = {
                "architecture": architecture,
                "weight_decay": decay,
                "initializations": len(endpoint),
                "independent_worlds": 1,
                "atomic": mean([r["atomic"] for r in endpoint]),
                "train": mean([r["train"] for r in endpoint]),
                "familiar_at_first_fit": mean([r["familiar_at_fit"] for r in endpoint]),
                "familiar512k": mean([r["familiar"] for r in endpoint]),
                "strict512k": mean([r["strict"] for r in endpoint]),
            }
            for step in [8000, 128000, 256000]:
                for task in ["familiar_test", "strict_test"]:
                    values = [
                        float(r["accuracy"])
                        for r in learning
                        if r["architecture"] == architecture
                        and float(r["weight_decay"]) == decay
                        and r["task"] == task
                        and int(r["step"]) == step
                    ]
                    row[f"{task}{step}"] = mean(values)
            aggregates.append(row)
    csv_write(ROOT / "training-aggregate.csv", aggregates)
    branches = []
    for path in Path("results/grokking-dynamics-v1/mechanism/development").glob("*/case*.json"):
        record = json.loads(path.read_text())
        state = path.parent.name
        architecture = "standard2" if "standard2" in state else "loop2"
        node = int(state.rsplit("-t", 1)[1])
        branches.append({"architecture": architecture, "checkpoint": node, **record})
    groups = defaultdict(list)
    for branch in branches:
        case = branch["case"]
        groups[
            branch["architecture"], branch["checkpoint"], case["group"], case["role"], branch["arm"]
        ].append(branch)
    edits = []
    for (architecture, node, group, role, arm), selected in sorted(groups.items()):
        metrics = [b["history"][-1]["metrics"] for b in selected]
        d = [m["D_" + group] for m in metrics]
        op_ok = [
            m["E_new" if arm == "edit" else "E_old"]["accuracy"] == 1
            and m["Kdev_atomic"]["accuracy"] >= 0.95
            for m in metrics
        ]
        edits.append(
            {
                "architecture": architecture,
                "checkpoint": node,
                "group": group,
                "role": role,
                "arm": arm,
                "branches": len(selected),
                "unique_graph_facts": len({b["case"]["atomic_index"] for b in selected}),
                "target_success_fraction": mean(
                    [m["E_new" if arm == "edit" else "E_old"]["accuracy"] for m in metrics]
                ),
                "R_atomic": mean([m["R_atomic"]["accuracy"] for m in metrics]),
                "Kdev_atomic": mean([m["Kdev_atomic"]["accuracy"] for m in metrics]),
                "U_atomic": mean([m["U_atomic"]["accuracy"] for m in metrics]),
                "U_familiar": mean([m["U_familiar"]["accuracy"] for m in metrics]),
                "full_D_fact_equal": mean([m["accuracy"] for m in d]),
                "parent_correct_changed_query_model_pairs": sum(m["eligible_n"] for m in d),
                "parent_correct_changed_D_fact_equal": mean([m["eligible_accuracy"] for m in d]),
                "old_answer_fact_equal": mean([m["eligible_old_answer_accuracy"] for m in d]),
                "E_Kdev_operation_pass_fraction": mean(op_ok),
                "qualified_eligible_query_model_pairs": sum(
                    m["eligible_n"] for m, ok in zip(d, op_ok, strict=True) if ok
                ),
                "qualified_D_fact_equal": mean(
                    [m["eligible_accuracy"] for m, ok in zip(d, op_ok, strict=True) if ok]
                ),
            }
        )
    csv_write(ROOT / "edit-aggregate.csv", edits)
    panels = matched_edit_panels()
    csv_write(ROOT / "edit-matched-panels.csv", panels)
    audit_donor_roles()
    trace_rows = read_csv(ROOT / "mechanism/development/trace.csv")
    traces = []
    grouped = defaultdict(list)
    for r in trace_rows:
        grouped[
            r["architecture"],
            int(r["checkpoint"]),
            r["split"],
            r["style"],
            int(r["recipient_position"]),
            r["component"],
        ].append(r)
    for (architecture, node, split, style, position, component), rows in sorted(grouped.items()):
        traces.append(
            {
                "architecture": architecture,
                "checkpoint": node,
                "split": split,
                "style": style,
                "position": position,
                "component": component,
                "states": len(rows),
                "mean_coverage": mean([float(r["coverage"]) for r in rows]),
                "target_accuracy_delta_pp": mean(
                    [float(r["target_accuracy_delta_pp"]) for r in rows]
                ),
                "target_probability_delta": mean(
                    [float(r["target_probability_delta"]) for r in rows]
                ),
            }
        )
    csv_write(ROOT / "trace-aggregate.csv", traces)
    write_json(
        ROOT / "assessment.json",
        {
            "created_utc": utc(),
            "training": aggregates,
            "edits": edits,
            "matched_edit_panels": panels,
            "traces": traces,
            "unit": "one independent development world; "
            "repeated seeds, checkpoints, facts and queries "
            "are not additional independent worlds",
            "analysis_status": "Fixed nodes and complete pools registered; "
            "this aggregate extraction "
            "and the E/Kdev-qualified display are descriptive after training. U/D never choose LR.",
        },
    )
    plot(learning, edits, traces)
    print(
        json.dumps(
            {
                "training": aggregates,
                "edit_aggregate_rows": len(edits),
                "trace_aggregate_rows": len(traces),
            }
        )
    )


def matched_edit_panels():
    """Hold the parent-correct changed queries fixed across all three checkpoints."""
    collected = defaultdict(dict)
    root = Path("results/grokking-dynamics-v1/mechanism/development")
    for path in root.glob("*/case*-edit.json"):
        branch = json.loads(path.read_text())
        name, node = path.parent.name.rsplit("-t", 1)
        case = branch["case"]
        collected[name, case["atomic_index"]][int(node)] = (branch, path.with_suffix(".npz"))
    panels = []
    for (name, index), stages in sorted(collected.items()):
        assert set(stages) == {8000, 128000, 512000}
        first = stages[8000][0]
        case = first["case"]
        task = "D_" + case["group"]
        original_n = len(case["original_d"][task])
        common = np.ones(original_n, dtype=bool)
        for branch, path in stages.values():
            raw = np.load(path)
            common &= raw[task + "_changed"] & raw[task + "_parent_correct"]
            common &= branch["parent_old_target_atomic_correct"]
            common &= branch["parent_new_successors_correct"]
        for node, (branch, path) in sorted(stages.items()):
            raw = np.load(path)
            rows = np.asarray(case["tasks"][task], dtype=np.int64)
            pred = raw[f"step200_{task}_generated"]
            correct = (pred[:, 0] == rows[:, -1]) & (pred[:, 1] == 5) & (pred[:, 2] == 1)
            m = branch["history"][-1]["metrics"]
            panels.append(
                {
                    "name": name,
                    "architecture": "standard2" if "standard2" in name else "loop2",
                    "checkpoint": node,
                    "atomic_index": index,
                    "group": case["group"],
                    "role": case["role"],
                    "fixed_common_query_n": int(common.sum()),
                    "original_pool_n": original_n,
                    "fixed_common_query_indices": json.dumps(np.flatnonzero(common).tolist()),
                    "propagation_accuracy": float(correct[common].mean()) if common.any() else None,
                    "E_new": m["E_new"]["accuracy"],
                    "U_atomic": m["U_atomic"]["accuracy"],
                    "Kdev_atomic": m["Kdev_atomic"]["accuracy"],
                }
            )
    return panels


def audit_donor_roles():
    """Document when a counterfactual donor changes the successor's training role."""
    rows = []
    for out in sorted(Path("results/grokking-dynamics-v1/mechanism/development").glob("*")):
        raw_file = out / "trace/trace.npz"
        if not raw_file.exists():
            continue
        raw = np.load(raw_file)
        parent = Path("results/grokking-dynamics-v1/development") / out.name.rsplit("-t", 1)[0]
        world = np.load(parent / "world.npz")
        training = world["train_composite"]
        first = {tuple(map(int, r[:2])) for r in training}
        second = {tuple(map(int, r[[2, 3]])) for r in training}
        for split in ["familiar", "strict"]:
            for style in ["same_bridge", "different_bridge"]:
                donors = raw[f"{split}_{style}_donor_rows"]
                queries = raw[f"{split}_rows"][raw[f"{split}_{style}_ids"]]
                rows.append(
                    {
                        "state": out.name,
                        "split": split,
                        "style": style,
                        "n": len(donors),
                        "donor_first_has_composition_role": mean(
                            [tuple(map(int, donor[:2])) in first for donor in donors]
                        ),
                        "counterfactual_successor_has_composition_role": mean(
                            [
                                (int(donor[2]), int(query[3])) in second
                                for donor, query in zip(donors, queries, strict=True)
                            ]
                        ),
                    }
                )
    csv_write(ROOT / "donor-role-audit.csv", rows)
    summary = []
    for split in ["familiar", "strict"]:
        for style in ["same_bridge", "different_bridge"]:
            selected = [r for r in rows if r["split"] == split and r["style"] == style]
            summary.append(
                {
                    "split": split,
                    "style": style,
                    "donor_first_role_fraction": mean(
                        [r["donor_first_has_composition_role"] for r in selected]
                    ),
                    "counterfactual_successor_role_fraction": mean(
                        [r["counterfactual_successor_has_composition_role"] for r in selected]
                    ),
                }
            )
    write_json(
        ROOT / "donor-role-audit.json",
        {
            "status": "posthoc descriptive graph audit",
            "summary": summary,
            "warning": "Different-bridge donors move strict recipients "
            "to familiar successor facts. "
            "Counterfactual redirection is not repair of the original strict chain.",
        },
    )


def plot(learning, edits, traces):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, arch in enumerate(["standard2", "loop2"]):
        for j, task in enumerate(["familiar_test", "strict_test"]):
            ax = axes[i, j]
            for decay, color in [(0, "#4477aa"), (0.01, "#ee7733"), (0.1, "#228833")]:
                rows = [
                    r
                    for r in learning
                    if r["architecture"] == arch
                    and r["task"] == task
                    and float(r["weight_decay"]) == decay
                    and int(r["step"]) > 0
                ]
                steps = sorted({int(r["step"]) for r in rows})
                values = [
                    mean([float(r["answer_nll"]) for r in rows if int(r["step"]) == s])
                    for s in steps
                ]
                ax.plot(steps, values, color=color, label=f"wd={decay:g}")
            ax.set_xscale("log")
            ax.set_ylabel("Answer NLL")
            ax.set_xlabel("Updates")
            ax.set_title(f"{arch}: {task}")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "answer-nll.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arch in zip(axes, ["standard2", "loop2"], strict=True):
        for role, color in [("first", "#4477aa"), ("second", "#ee7733")]:
            rows = [
                r
                for r in edits
                if r["architecture"] == arch
                and r["role"] == role
                and r["group"] == "familiar"
                and r["arm"] == "edit"
            ]
            ax.plot(
                [r["checkpoint"] for r in rows],
                [100 * r["U_atomic"] for r in rows],
                "--",
                color=color,
                label=f"{role}-hop edit: U atoms",
            )
            ax.plot(
                [r["checkpoint"] for r in rows],
                [
                    100 * r["parent_correct_changed_D_fact_equal"]
                    if r["parent_correct_changed_D_fact_equal"] is not None
                    else np.nan
                    for r in rows
                ],
                "o-",
                color=color,
                label=f"{role}-hop edit: eligible D",
            )
        ax.set_xscale("log")
        ax.set_ylim(-2, 102)
        ax.set_title(arch)
        ax.set_xlabel("Parent updates")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "edit-propagation.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
