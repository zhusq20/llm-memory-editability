"""Re-score atomic editing runs and select a development learning rate.

Selection uses only fixed-endpoint edit E and parent-correct U/necessary
atomics. Target facts have equal weight. D scores never enter the decision.
Queries, target edits, initializations and branches are nested within a world;
only independently generated worlds count as independent replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

THRESHOLDS = {"E_new": 0.9, "U_atomic": 0.98, "necessary_atomic": 0.98}
FINAL_DECISIONS = {"selected", "no_eligible_learning_rate"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def full_correct(generated, rows):
    generated, rows = np.asarray(generated), np.asarray(rows)
    if generated.shape != (len(rows), 3):
        raise ValueError("Every query must have answer, punctuation and EOS predictions")
    return (generated[:, 0] == rows[:, -1]) & np.all(generated[:, 1:] == [5, 1], axis=1)


def score_task(name, rows, current, parent, e_new_success):
    """Recompute denominators from saved tokens, never rounded mean scores."""
    generated = current[name + "__generated"]
    correct = full_correct(generated, rows)
    np.testing.assert_array_equal(correct, current[name + "__correct"])
    n = len(rows)
    record = {"n": n, "correct": int(correct.sum()), "accuracy": ratio(int(correct.sum()), n)}
    if name.startswith(("D_", "T_", "S_same_answer_")):
        old_rows = current[name + "__old_rows"]
        np.testing.assert_array_equal(rows, current[name + "__new_rows"])
        parent_correct = full_correct(parent[name + "__generated"], old_rows)
        np.testing.assert_array_equal(parent_correct, current[name + "__parent_correct"])
        changed = rows[:, -1] != old_rows[:, -1]
        if name.startswith("D_") and not changed.all():
            raise ValueError("Main D must contain only answer-changing untrained queries")
        old_correct = full_correct(generated, old_rows)
        parent_and_e = parent_correct & e_new_success
        e_only = np.full(n, e_new_success, dtype=bool)
        record.update(
            {
                "changed_answer_n": int(changed.sum()),
                "old_answer_correct": int(old_correct.sum()),
                "old_answer_accuracy": ratio(int(old_correct.sum()), n),
                "parent_correct_n": int(parent_correct.sum()),
                "parent_correct_coverage": ratio(int(parent_correct.sum()), n),
                "correct_on_parent_correct": int((correct & parent_correct).sum()),
                "accuracy_on_parent_correct": ratio(
                    int((correct & parent_correct).sum()), int(parent_correct.sum())
                ),
                "E_success_n": int(e_only.sum()),
                "E_success_coverage": ratio(int(e_only.sum()), n),
                "correct_on_E_success": int((correct & e_only).sum()),
                "accuracy_on_E_success": ratio(int((correct & e_only).sum()), int(e_only.sum())),
                "parent_correct_E_success_n": int(parent_and_e.sum()),
                "parent_correct_E_success_coverage": ratio(int(parent_and_e.sum()), n),
                "correct_on_parent_correct_E_success": int((correct & parent_and_e).sum()),
                "accuracy_on_parent_correct_E_success": ratio(
                    int((correct & parent_and_e).sum()), int(parent_and_e.sum())
                ),
                "old_retained_on_parent_correct": int((old_correct & parent_correct).sum()),
            }
        )
    else:
        parent_correct = full_correct(parent[name + "__generated"], rows)
        np.testing.assert_array_equal(parent_correct, parent[name + "__correct"])
        retained = parent_correct & correct
        record.update(
            {
                "parent_correct_n": int(parent_correct.sum()),
                "parent_correct_coverage": ratio(int(parent_correct.sum()), n),
                "retained_parent_correct": int(retained.sum()),
                "retention_on_parent_correct": ratio(
                    int(retained.sum()), int(parent_correct.sum())
                ),
            }
        )
    return record


def score_case_node(tasks, current, parent):
    e_new = bool(full_correct(current["E_new__generated"], tasks["E_new"]).all())
    e_old = bool(full_correct(current["E_old__generated"], tasks["E_old"]).all())
    if len(tasks["E_new"]) != 1 or len(tasks["E_old"]) != 1:
        raise ValueError("Each branch must edit exactly one factual address")
    return {
        "E_new_success": e_new,
        "E_old_success": e_old,
        "tasks": {
            name: score_task(name, rows, current, parent, e_new) for name, rows in tasks.items()
        },
    }


def load_run(directory):
    """Read all currently available nodes; completed/audited status stays explicit."""
    directory = Path(directory)
    manifest = read(directory / "run.json")
    spec, parent_spec = manifest["spec"], manifest["parent_spec"]
    parent_id = Path(spec["parent_dir"]).name
    parent_arm = parent_spec["arm"]
    info = {
        "run": directory.name,
        "parent": parent_id,
        "parent_arm": parent_arm,
        "model_condition": "aligned" if parent_arm.startswith("aligned") else parent_arm,
        "world": parent_spec["world"],
        "initialization": parent_spec["initialization"],
        "phase": spec.get("phase"),
        "case_sha256": manifest["case_sha256"],
        "expected_branches": manifest["branches"],
        "endpoint": spec.get("nodes", [0, 32, 128, 512])[-1],
        "complete": (directory / "complete.json").exists(),
        "audited": (directory / "audit.json").exists()
        and read(directory / "audit.json").get("passed") is True,
    }
    records, issues = [], []
    rates = spec.get("learning_rates", [spec.get("lr", 1e-4)])
    nodes = spec.get("nodes", [0, 32, 128, 512])
    info["learning_rates"] = rates
    for case in manifest["cases"]:
        case_dir = directory / case["case_id"]
        if (
            not (case_dir / "tasks.npz").exists()
            or not (case_dir / "parent-predictions.npz").exists()
        ):
            issues.append(f"{case['case_id']}: parent/tasks pending")
            continue
        with np.load(case_dir / "tasks.npz") as loaded:
            tasks = dict(loaded)
        with np.load(case_dir / "parent-predictions.npz") as loaded:
            parent = dict(loaded)
        for lr in rates:
            for branch in ("edit", "sham"):
                branch_dir = case_dir / f"lr{lr:.8g}-{branch}"
                if not (branch_dir / "learning.json").exists():
                    issues.append(f"{case['case_id']}/{lr}/{branch}: branch pending")
                    continue
                history = read(branch_dir / "learning.json")
                found = [row["step"] for row in history]
                if found != nodes:
                    issues.append(f"{case['case_id']}/{lr}/{branch}: incomplete nodes {found}")
                for row in history:
                    path = branch_dir / f"predictions-{row['step']:06d}.npz"
                    if not path.exists():
                        issues.append(f"{case['case_id']}/{lr}/{branch}: prediction pending")
                        continue
                    with np.load(path) as current:
                        scores = score_case_node(tasks, current, parent)
                    records.append(
                        {
                            **{
                                key: info[key]
                                for key in (
                                    "run",
                                    "parent",
                                    "parent_arm",
                                    "model_condition",
                                    "world",
                                    "initialization",
                                )
                            },
                            "case_id": case["case_id"],
                            "role": case["role"],
                            "case_pool": case["stratum"],
                            "lr": lr,
                            "branch": branch,
                            "node": row["step"],
                            "endpoint": info["endpoint"],
                            "branch_target_success": scores[
                                "E_new_success" if branch == "edit" else "E_old_success"
                            ],
                            **scores,
                        }
                    )
    info["issues"] = issues
    info["valid_for_decision"] = info["complete"] and info["audited"] and not issues
    return info, records


def add_paired_edit_conditions(records):
    """Use the same successful edit targets when comparing edit with sham."""
    keys = ("parent", "lr", "case_id", "node")
    edited = {tuple(row[k] for k in keys): row for row in records if row["branch"] == "edit"}
    for row in records:
        pair = edited.get(tuple(row[k] for k in keys))
        success = pair["E_new_success"] if pair is not None else None
        row["paired_edit_E_new_success"] = success
        for name, values in row["tasks"].items():
            if not name.startswith(("D_", "T_", "S_same_answer_")):
                continue
            if success is None:
                continue
            n = values["n"]
            denominator = values["parent_correct_n"] * int(success)
            numerator = values["correct_on_parent_correct"] * int(success)
            values.update(
                {
                    "parent_correct_paired_edit_E_success_n": denominator,
                    "parent_correct_paired_edit_E_success_coverage": ratio(denominator, n),
                    "correct_on_parent_correct_paired_edit_E_success": numerator,
                    "accuracy_on_parent_correct_paired_edit_E_success": ratio(
                        numerator, denominator
                    ),
                }
            )


RATES = (
    "accuracy",
    "parent_correct_coverage",
    "retention_on_parent_correct",
    "old_answer_accuracy",
    "accuracy_on_parent_correct",
    "E_success_coverage",
    "accuracy_on_E_success",
    "parent_correct_E_success_coverage",
    "accuracy_on_parent_correct_E_success",
    "parent_correct_paired_edit_E_success_coverage",
    "accuracy_on_parent_correct_paired_edit_E_success",
)
COUNTS = (
    "n",
    "correct",
    "parent_correct_n",
    "retained_parent_correct",
    "changed_answer_n",
    "old_answer_correct",
    "correct_on_parent_correct",
    "E_success_n",
    "correct_on_E_success",
    "parent_correct_E_success_n",
    "correct_on_parent_correct_E_success",
    "old_retained_on_parent_correct",
    "parent_correct_paired_edit_E_success_n",
    "correct_on_parent_correct_paired_edit_E_success",
)


def aggregate(records):
    """Equal targets within a parent, then equal initializations/worlds."""
    groups = defaultdict(list)
    parent_keys = (
        "parent",
        "parent_arm",
        "model_condition",
        "world",
        "initialization",
        "lr",
        "role",
        "case_pool",
        "branch",
        "node",
    )
    for record in records:
        for task, scores in record["tasks"].items():
            groups[tuple(record[key] for key in parent_keys) + (task,)].append(scores)
    parent_cells = []
    for key, values in sorted(groups.items()):
        cell = dict(zip((*parent_keys, "task"), key, strict=True))
        cell["target_facts"] = len(values)
        cell["rates_target_equal"] = {name: mean(v.get(name) for v in values) for name in RATES}
        cell["defined_target_facts"] = {
            name: sum(v.get(name) is not None for v in values) for name in RATES
        }
        cell["occurrence_counts"] = {name: sum(v.get(name, 0) for v in values) for name in COUNTS}
        parent_cells.append(cell)
    world_keys = ("model_condition", "world", "lr", "role", "case_pool", "branch", "node", "task")
    grouped_worlds = defaultdict(list)
    for cell in parent_cells:
        grouped_worlds[tuple(cell[k] for k in world_keys)].append(cell)
    world_cells = []
    for key, values in sorted(grouped_worlds.items()):
        world_cells.append(
            {
                **dict(zip(world_keys, key, strict=True)),
                "parents": [v["parent"] for v in values],
                "initializations": len({v["initialization"] for v in values}),
                "rates_initialization_equal": {
                    name: mean(v["rates_target_equal"][name] for v in values) for name in RATES
                },
            }
        )
    all_keys = tuple(k for k in world_keys if k != "world")
    grouped_all = defaultdict(list)
    for cell in world_cells:
        grouped_all[tuple(cell[k] for k in all_keys)].append(cell)
    all_cells = []
    for key, values in sorted(grouped_all.items()):
        all_cells.append(
            {
                **dict(zip(all_keys, key, strict=True)),
                "worlds": [v["world"] for v in values],
                "independent_worlds": len(values),
                "rates_world_equal": {
                    name: mean(v["rates_initialization_equal"][name] for v in values)
                    for name in RATES
                },
            }
        )
    return parent_cells, world_cells, all_cells


def development_decision(records, run_info, expected_names, phase):
    """No D, sham, mid-training node, or query count enters lr selection."""
    decision = {
        "phase": phase,
        "selected_lr": None,
        "thresholds": THRESHOLDS,
        "selection_uses": [
            "edit E_new target success",
            "parent-correct U_atomic retention",
            "parent-correct necessary_atomic retention",
        ],
        "weighting": (
            "Each preselected target of each parent has equal weight. "
            "Retention is conditioned within target before averaging targets."
        ),
        "undefined_policy": (
            "Every target must have nonempty parent-correct U and necessary denominators; "
            "otherwise lr is ineligible."
        ),
        "sham_policy": (
            "Sham target is E_old and is reported separately; "
            "sham success never dilutes edit failure."
        ),
        "D_used_for_selection": False,
    }
    if phase != "development":
        return {
            **decision,
            "status": "evaluation_only",
            "reason": "Confirmation never selects hyperparameters.",
        }
    actual_names = {info["run"] for info in run_info}
    if expected_names is None:
        return {
            **decision,
            "status": "pending",
            "reason": "Missing frozen run matrix; cannot infer which runs remain.",
        }
    if actual_names != set(expected_names) or any(
        not info["valid_for_decision"] for info in run_info
    ):
        return {
            **decision,
            "status": "pending",
            "reason": "The complete frozen run matrix and every independent audit are required.",
            "missing_runs": sorted(set(expected_names) - actual_names),
            "unexpected_runs": sorted(actual_names - set(expected_names)),
        }
    endpoint = [r for r in records if r["branch"] == "edit" and r["node"] == r["endpoint"]]
    grouped = defaultdict(list)
    selection_inputs = []
    for row in endpoint:
        keep = row["tasks"]["U_atomic"]
        necessary = row["tasks"]["necessary_atomic"]
        value = {
            "parent": row["parent"],
            "case_id": row["case_id"],
            "lr": row["lr"],
            "E_new": float(row["E_new_success"]),
            "U_atomic": keep["retention_on_parent_correct"],
            "necessary_atomic": necessary["retention_on_parent_correct"],
            "U_parent_correct_n": keep["parent_correct_n"],
            "U_n": keep["n"],
            "necessary_parent_correct_n": necessary["parent_correct_n"],
            "necessary_n": necessary["n"],
        }
        grouped[row["lr"]].append(value)
        selection_inputs.append(value)
    if not grouped:
        return {**decision, "status": "pending", "reason": "No completed endpoint targets."}
    populations = [
        {(row["parent"], row["case_id"]) for row in values} for values in grouped.values()
    ]
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError("Learning-rate conditions must contain the same preselected targets")
    candidates = []
    for lr, values in sorted(grouped.items()):
        means = {name: mean(row[name] for row in values) for name in THRESHOLDS}
        complete_denominators = all(
            row["U_parent_correct_n"] > 0 and row["necessary_parent_correct_n"] > 0
            for row in values
        )
        passed = complete_denominators and all(
            means[name] >= threshold for name, threshold in THRESHOLDS.items()
        )
        candidates.append(
            {
                "lr": lr,
                "targets": len(values),
                "parents": sorted({row["parent"] for row in values}),
                "means_target_equal": means,
                "all_retention_denominators_nonempty": complete_denominators,
                "eligible": passed,
            }
        )
    selected = next((row["lr"] for row in candidates if row["eligible"]), None)
    return {
        **decision,
        "status": "selected" if selected is not None else "no_eligible_learning_rate",
        "selected_lr": selected,
        "candidates": candidates,
        "selection_input_sha256": content_hash(
            sorted(selection_inputs, key=lambda r: (r["lr"], r["parent"], r["case_id"]))
        ),
        "selection_inputs": selection_inputs,
    }


def frozen_config(root):
    root = Path(root)
    candidates = [
        root / "frozen-config.json",
        Path(__file__).resolve().parents[1]
        / "docs/development-artifacts"
        / root.name
        / "frozen-config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return read(candidate), str(candidate)
    return None, None


def report(root, phase, output=None):
    root = Path(root)
    output = Path(output) if output else root / "report"
    config, config_path = frozen_config(root)
    expected = [s["name"] for s in config["specs"]] if config else None
    expected_specs = {s["name"]: s for s in config["specs"]} if config else {}
    run_info, records = [], []
    for directory in sorted((root / "runs").glob("*")):
        if not (directory / "run.json").exists():
            continue
        info, rows = load_run(directory)
        if info["phase"] != phase:
            continue
        if directory.name in expected_specs:
            actual_spec = read(directory / "run.json")["spec"]
            if content_hash(actual_spec) != content_hash(expected_specs[directory.name]):
                info["issues"].append("Run specification disagrees with the frozen configuration")
                info["valid_for_decision"] = False
        run_info.append(info)
        records.extend(rows)
    add_paired_edit_conditions(records)
    parent_cells, world_cells, world_equal_cells = aggregate(records)
    summary = {
        "phase": phase,
        "root": str(root.resolve()),
        "frozen_config": config_path,
        "status": "complete"
        if expected is not None
        and {i["run"] for i in run_info} == set(expected)
        and all(i["valid_for_decision"] for i in run_info)
        else "partial",
        "independent_worlds": len({row["world"] for row in records}),
        "runs": run_info,
        "case_node_records": len(records),
        "definitions": {
            "E": "Exact answer, punctuation and EOS; edit target E_new, sham target E_old.",
            "D": (
                "Changed-answer untrained combinations only. Full pool, parent-correct, "
                "E_new-success and their intersection are all retained with denominators."
            ),
            "paired_D": (
                "parent_correct_paired_edit_E_success uses the paired edit branch's E_new "
                "success at the same node for both edit and sham. Own-branch E_new "
                "conditions remain separately labelled."
            ),
            "retention": (
                "Within each target, intersect current correctness with correctness "
                "at the unchanged parent; divide by parent-correct count."
            ),
            "pool_overlap": (
                "registered_familiar is a subset of familiar; never sum them. "
                "S_same_answer and T_train are separate from D."
            ),
            "independence": (
                "Worlds are independent units. Targets, initializations, nodes and "
                "queries are nested repeated measurements."
            ),
            "counts": (
                "occurrence_counts may repeatedly evaluate the same atom across edits; "
                "these counts are not independent facts."
            ),
            "aggregation": (
                "Target-equal parent rates, then initialization-equal within world, "
                "then world-equal. Undefined subsets retain coverage and defined-target counts."
            ),
        },
        "parent_cells": parent_cells,
        "world_cells": world_cells,
        "world_equal_cells": world_equal_cells,
    }
    decision = development_decision(records, run_info, expected, phase)
    old_path = output / "decision.json"
    if old_path.exists():
        old = read(old_path)
        if old.get("status") in FINAL_DECISIONS and old != decision:
            raise ValueError(
                "A finalized development decision is frozen; use a separate amended output."
            )
    write(output / "case-scores.json", records)
    write(output / "summary.json", summary)
    write(old_path, decision)
    return summary, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary, decision = report(args.root, args.phase, args.out)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "runs": len(summary["runs"]),
                "decision": decision["status"],
                "selected_lr": decision["selected_lr"],
            }
        )
    )


if __name__ == "__main__":
    main()
