"""Report audited write-target comparisons without selecting cases or training recipes.

Only completed, independently audited runs contribute scores. Partial reports
retain the full expected matrix and explicitly mark all missing runs. Scores
come from saved predictions; worlds, initializations and cases remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from report_interface_editing import score_case_node

OBJECTIVES = ("final_ce", "early_ce", "early_ce_align")
TASKS = (
    "E_new",
    "E_old",
    "necessary_atomic",
    "R_atomic",
    "U_atomic",
    "D_first_strict",
    "S_same_answer_first_strict",
    "U_strict",
    "U_familiar",
    "U_train",
)
PAIR_METRICS = {
    "E_new": ("tasks", "E_new", "accuracy"),
    "D_new": ("tasks", "D_first_strict", "accuracy"),
    "D_old": ("tasks", "D_first_strict", "old_answer_accuracy"),
    "U_atomic": ("tasks", "U_atomic", "accuracy"),
    "U_strict": ("tasks", "U_strict", "accuracy"),
    "early_new": ("early", "new_entity_accuracy"),
    "early_old": ("early", "old_entity_accuracy"),
    "new_direction_cosine": ("early", "new_direction_cosine"),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def ratio(correct, total):
    return correct / total if total else None


def nested(row, path):
    for key in path:
        row = row[key]
    return row


def hierarchical_mean(rows, getter):
    """Equal cases within initialization, equal initializations within world."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["world"], row["initialization"])].append(getter(row))
    worlds = defaultdict(list)
    for (world, _initialization), values in groups.items():
        worlds[world].append(mean(values))
    return mean([mean(values) for values in worlds.values()])


def load_root(root, partial=False):
    root = Path(root).resolve()
    config_path = root / "frozen-config.json"
    config = read(config_path)
    specs = config["specs"]
    states, records = [], []
    provenance = {str(config_path): digest(config_path)}
    for spec in specs:
        directory = root / "runs" / spec["name"]
        complete_path, audit_path = directory / "complete.json", directory / "audit.json"
        state = {
            "run": spec["name"],
            "root": str(root),
            "phase": spec["phase"],
            "expected_cases": len(spec.get("case_ids", range(spec.get("per_cell", 4)))),
            "objectives": spec.get("objectives", list(OBJECTIVES)),
        }
        if (directory / "failure.json").exists():
            state["status"] = "failed"
        elif not complete_path.exists():
            state["status"] = "running" if (directory / "run.json").exists() else "pending"
        elif not audit_path.exists() or read(audit_path).get("passed") is not True:
            state["status"] = "complete_but_not_audited"
        else:
            state["status"] = "complete_and_audited"
        states.append(state)
        if state["status"] != "complete_and_audited":
            continue
        manifest_path = directory / "run.json"
        manifest, complete = read(manifest_path), read(complete_path)
        if manifest["spec"] != spec:
            raise AssertionError(f"Frozen spec differs from executed spec: {directory}")
        provenance.update(
            {str(path): digest(path) for path in [manifest_path, complete_path, audit_path]}
        )
        cases = {case["case_id"]: case for case in manifest["cases"]}
        expected = {
            (case, kind, arm)
            for case in cases
            for kind in state["objectives"]
            for arm in ("edit", "sham")
        }
        actual = [(b["case_id"], b["objective"], b["arm"]) for b in complete["branches"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise AssertionError(f"Incomplete branch matrix: {directory}")
        for branch in complete["branches"]:
            case = cases[branch["case_id"]]
            case_dir = directory / case["case_id"]
            tasks_path = case_dir / "tasks.npz"
            parent_path = case_dir / "parent-predictions.npz"
            prediction_path = directory / branch["path"] / f"predictions-{branch['steps']:06d}.npz"
            for path in (tasks_path, parent_path, prediction_path):
                provenance[str(path)] = digest(path)
            with np.load(tasks_path) as saved:
                tasks = dict(saved)
            with np.load(parent_path) as saved:
                parent = dict(saved)
            with np.load(prediction_path) as saved:
                current = dict(saved)
            scored = score_case_node(tasks, current, parent)
            for name in tasks:
                if scored["tasks"][name]["accuracy"] != branch["metrics"][name]["accuracy"]:
                    raise AssertionError(
                        f"Saved prediction and final score differ: {directory}/{name}"
                    )
            early = branch["metrics"]["early"]
            prediction = int(current["early__entity_prediction"][0])
            if early["new_entity_accuracy"] != float(prediction == case["new_fact"][2]):
                raise AssertionError("Early entity prediction differs from score")
            historical = None
            diagnostic_path = directory / "existing-edit-diagnostics.json"
            if branch["objective"] == "final_ce" and diagnostic_path.exists():
                history_path = directory / branch["path"] / "learning.json"
                historical_records = read(diagnostic_path)["records"]
                history = read(history_path)
                old = {
                    row["step"]: row["model_sha256"]
                    for row in historical_records
                    if row["case_id"] == case["case_id"] and row["arm"] == branch["arm"]
                }
                historical = {
                    "nodes": len(history),
                    "matching_exact_model_digests": sum(
                        old.get(row["step"]) == row["model_sha256"] for row in history
                    ),
                }
                provenance.update(
                    {str(path): digest(path) for path in (diagnostic_path, history_path)}
                )
            records.append(
                {
                    "root": str(root),
                    "run": spec["name"],
                    "phase": spec["phase"],
                    "world": manifest["parent_spec"]["world"],
                    "initialization": manifest["parent_spec"]["initialization"],
                    "parent_model_sha256": manifest["parent_model_sha256"],
                    "case_id": case["case_id"],
                    "old_fact": case["old_fact"],
                    "new_fact": case["new_fact"],
                    "replay_indices": case["replay_indices"],
                    "objective": branch["objective"],
                    "arm": branch["arm"],
                    "steps": branch["steps"],
                    "tasks": scored["tasks"],
                    "early": early,
                    "delta_l2": branch["delta_l2"],
                    "historical_baseline_reproduction": historical,
                }
            )
    if not partial and any(state["status"] != "complete_and_audited" for state in states):
        pending = [state["run"] for state in states if state["status"] != "complete_and_audited"]
        raise ValueError(
            f"Incomplete batch; use --partial for a clearly labelled report: {pending}"
        )
    return records, states, provenance


def aggregate_task(rows, name):
    values = [row["tasks"][name] for row in rows]
    total = sum(value["n"] for value in values)
    correct = sum(value["correct"] for value in values)
    parent_correct = sum(value["parent_correct_n"] for value in values)
    result = {
        "query_evaluations_n": total,
        "correct_n": correct,
        "pooled_accuracy": ratio(correct, total),
        "equal_case_accuracy": mean([value["accuracy"] for value in values]),
        "equal_world_accuracy": hierarchical_mean(rows, lambda row: row["tasks"][name]["accuracy"]),
        "parent_correct_n": parent_correct,
        "parent_correct_coverage": ratio(parent_correct, total),
        "equal_world_parent_correct_coverage": hierarchical_mean(
            rows,
            lambda row: ratio(row["tasks"][name]["parent_correct_n"], row["tasks"][name]["n"]),
        ),
    }
    if name.startswith(("D_", "S_same_answer_")):
        old = sum(value["old_answer_correct"] for value in values)
        retained = sum(value["old_retained_on_parent_correct"] for value in values)
        new_on_parent = sum(value["correct_on_parent_correct"] for value in values)
        result.update(
            {
                "old_answer_correct_n": old,
                "old_answer_accuracy": ratio(old, total),
                "equal_world_old_answer_accuracy": hierarchical_mean(
                    rows, lambda row: row["tasks"][name]["old_answer_accuracy"]
                ),
                "correct_on_parent_correct_n": new_on_parent,
                "accuracy_on_parent_correct": ratio(new_on_parent, parent_correct),
                "old_retained_on_parent_correct_n": retained,
                "old_retained_on_parent_correct": ratio(retained, parent_correct),
            }
        )
    else:
        retained = sum(value["retained_parent_correct"] for value in values)
        result.update(
            {
                "retained_parent_correct_n": retained,
                "retention_on_parent_correct": ratio(retained, parent_correct),
                "equal_world_retention_on_parent_correct": hierarchical_mean(
                    rows, lambda row: row["tasks"][name]["retention_on_parent_correct"]
                ),
            }
        )
    return result


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "worlds_n": len({row["world"] for row in group}),
                "parent_initializations_n": len(
                    {(row["world"], row["initialization"]) for row in group}
                ),
                "case_observations_n": len(group),
                "tasks": {name: aggregate_task(group, name) for name in TASKS},
                "early": {
                    metric: hierarchical_mean(
                        group, lambda row, metric=metric: row["early"][metric]
                    )
                    for metric in group[0]["early"]
                    if metric != "n"
                },
                "mean_delta_l2": hierarchical_mean(group, lambda row: row["delta_l2"]),
            }
        )
    return result


def paired_differences(records, contrast):
    """Paired deltas are formed before averaging over initialization and world."""
    index = {}
    for row in records:
        key = (
            row["phase"],
            row["world"],
            row["initialization"],
            row["case_id"],
            row["objective"],
            row["arm"],
        )
        if key in index:
            raise ValueError(f"Duplicate repeated measure across supplied roots: {key}")
        index[key] = row
    pairs, missing = [], []
    for key, row in index.items():
        if contrast == "edit_minus_sham":
            if row["arm"] != "edit":
                continue
            other_key = (*key[:-1], "sham")
        else:
            if row["arm"] != "edit" or row["objective"] == "final_ce":
                continue
            other_key = (*key[:-2], "final_ce", "edit")
        if other_key not in index:
            missing.append(list(key))
            continue
        other = index[other_key]
        for field in ("old_fact", "new_fact", "replay_indices", "parent_model_sha256", "steps"):
            if row[field] != other[field]:
                raise AssertionError(f"Unmatched pair for {field}: {key}")
        delta = {}
        for name, path in PAIR_METRICS.items():
            first, second = nested(row, path), nested(other, path)
            delta[name] = first - second if first is not None and second is not None else None
        pairs.append(
            {
                "phase": row["phase"],
                "world": row["world"],
                "initialization": row["initialization"],
                "case_id": row["case_id"],
                "objective": row["objective"],
                "delta": delta,
            }
        )
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["phase"], row["objective"])].append(row)
    summaries = []
    for (phase, objective), group in sorted(groups.items()):
        summaries.append(
            {
                "phase": phase,
                "objective": objective,
                "paired_case_observations_n": len(group),
                "worlds_n": len({row["world"] for row in group}),
                "equal_world_delta": {
                    name: hierarchical_mean(group, lambda row, name=name: row["delta"][name])
                    for name in PAIR_METRICS
                },
            }
        )
    by_world = []
    world_groups = defaultdict(list)
    for row in pairs:
        world_groups[(row["phase"], row["world"], row["objective"])].append(row)
    for (phase, world, objective), group in sorted(world_groups.items()):
        by_world.append(
            {
                "phase": phase,
                "world": world,
                "objective": objective,
                "paired_case_observations_n": len(group),
                "initializations_n": len({row["initialization"] for row in group}),
                "equal_initialization_delta": {
                    name: hierarchical_mean(group, lambda row, name=name: row["delta"][name])
                    for name in PAIR_METRICS
                },
            }
        )
    return {
        "contrast": contrast,
        "summaries": summaries,
        "by_world": by_world,
        "pairs": pairs,
        "missing_counterpart_n": len(missing),
        "missing_counterparts": missing,
    }


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def counts(task, old=False):
    key = "old_answer_correct_n" if old else "correct_n"
    return f"{task[key]}/{task['query_evaluations_n']}"


def markdown(summary):
    complete = summary["complete"]
    status = (
        "全部预定运行已完成并通过独立重载"
        if complete
        else "部分结果：矩阵尚未完成，不能作为完整比较"
    )

    def table_row(*values):
        return "| " + " | ".join(map(str, values)) + " |"

    lines = [
        "# 写入目标比较",
        "",
        f"状态：{status}。",
        f"已审核运行 {summary['completed_runs_n']}/{summary['expected_runs_n']}；"
        f"报告时间 {summary['created_utc']}。",
        "",
        "这里只报告保存的证据，不调整训练设置或选择案例。开发与正式配对扩展分开列出；正式部分使用既有三个世界，不能称为新世界确认。",
        "原子 E 与早期实体读出使用相同目标实体。"
        "D 是所有答案发生变化的合法未训练首跳严格组合；答案不变的组合另存 summary.json。",
        "",
        "## 固定终点的完整池结果",
        "",
        "计数分母是查询评测次数；初始化、案例及重复出现的保持查询都不增加独立世界数。世界均值先在初始化内平均案例，再在世界内平均初始化。",
        "",
        "| 阶段 | 目标 | 分支 | 世界/初始化/案例 | E 新 | D 新正确/总数 | D 新世界均值 |"
        " D 旧正确/总数 | U 原子 | U 严格组合 | 早期新实体 | 新方向余弦 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_objective_arm"]:
        tasks, early = row["tasks"], row["early"]
        lines.append(
            table_row(
                row["phase"],
                row["objective"],
                row["arm"],
                f"{row['worlds_n']}/{row['parent_initializations_n']}/{row['case_observations_n']}",
                counts(tasks["E_new"]),
                counts(tasks["D_first_strict"]),
                percentage(tasks["D_first_strict"]["equal_world_accuracy"]),
                counts(tasks["D_first_strict"], True),
                percentage(tasks["U_atomic"]["equal_world_accuracy"]),
                percentage(tasks["U_strict"]["equal_world_accuracy"]),
                percentage(early["new_entity_accuracy"]),
                f"{early['new_direction_cosine']:.4f}",
            )
        )
    lines += [
        "",
        "## 父模型原本会答的 D 子集",
        "",
        "该子集只作辅助解释，完整池结果在上表；覆盖率不等同于编辑成功率。",
        "",
        "| 阶段 | 目标 | 分支 | 父模型正确/完整池 | 覆盖 | 子集新答案 | 子集旧答案残留 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_objective_arm"]:
        d = row["tasks"]["D_first_strict"]
        lines.append(
            table_row(
                row["phase"],
                row["objective"],
                row["arm"],
                f"{d['parent_correct_n']}/{d['query_evaluations_n']}",
                percentage(d["parent_correct_coverage"]),
                f"{d['correct_on_parent_correct_n']}/{d['parent_correct_n']}",
                f"{d['old_retained_on_parent_correct_n']}/{d['parent_correct_n']}",
            )
        )
    lines += [
        "",
        "## 未影响知识与组合的保持",
        "",
        "父模型正确子集直接测量原本会答的查询是否丢失；完整池准确率允许同时出现得失。"
        "父模型准确率和更新后准确率列为世界均值，计数与子集保持率列为合并计数。",
        "",
        "| 阶段 | 目标 | 分支 | 保持集合 | 父模型准确率 | 更新后准确率 |"
        " 父模型正确仍答对/原本答对 | 子集保持率 | 世界均值子集保持率 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_objective_arm"]:
        for name in ("U_atomic", "U_strict", "U_familiar"):
            task = row["tasks"][name]
            lines.append(
                table_row(
                    row["phase"],
                    row["objective"],
                    row["arm"],
                    name,
                    percentage(task["equal_world_parent_correct_coverage"]),
                    percentage(task["equal_world_accuracy"]),
                    f"{task['retained_parent_correct_n']}/{task['parent_correct_n']}",
                    percentage(task["retention_on_parent_correct"]),
                    percentage(task["equal_world_retention_on_parent_correct"]),
                )
            )
    lines += [
        "",
        "## 先配对再平均的差值",
        "",
        "差值为百分点；方向余弦为原始差。缺少对应运行的配对不计入，并记录在 JSON。",
        "",
        "| 比较 | 阶段 | 目标 | 配对案例/世界 | D 新 | D 旧 | U 原子 |"
        " U 严格组合 | 早期新实体 | 方向余弦 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in summary["paired_comparisons"]:
        for row in comparison["summaries"]:
            d = row["equal_world_delta"]
            lines.append(
                table_row(
                    comparison["contrast"],
                    row["phase"],
                    row["objective"],
                    f"{row['paired_case_observations_n']}/{row['worlds_n']}",
                    percentage(d["D_new"]),
                    percentage(d["D_old"]),
                    percentage(d["U_atomic"]),
                    percentage(d["U_strict"]),
                    percentage(d["early_new"]),
                    f"{d['new_direction_cosine']:.4f}",
                )
            )
    lines += [
        "",
        "## 分世界结果",
        "",
        "| 阶段 | 世界 | 目标 | 分支 | D 新正确/总数 | D 旧正确/总数 | U 严格组合 |"
        " U 父正确保持/父正确数 | U 父正确保持率 | 早期新实体 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_world_objective_arm"]:
        d = row["tasks"]["D_first_strict"]
        u = row["tasks"]["U_strict"]
        lines.append(
            table_row(
                row["phase"],
                row["world"],
                row["objective"],
                row["arm"],
                counts(d),
                counts(d, True),
                percentage(u["equal_world_accuracy"]),
                f"{u['retained_parent_correct_n']}/{u['parent_correct_n']}",
                percentage(u["equal_world_retention_on_parent_correct"]),
                percentage(row["early"]["new_entity_accuracy"]),
            )
        )
    if not complete:
        lines += ["", "## 未完成运行", ""]
        lines += [
            f"- {row['run']}: {row['status']}"
            for row in summary["run_states"]
            if row["status"] != "complete_and_audited"
        ]
    lines += [
        "",
        "所有端点、必要原子、sham、保持子集、配对原始差值和输入文件哈希见 summary.json。"
        "该报告不把早期状态读出或方向变化单独视为机制证明。",
        "",
    ]
    return "\n".join(lines)


def report(roots, out=None, partial=False):
    roots = [Path(root).resolve() for root in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Supply one or more distinct batch roots")
    out = Path(out) if out is not None else roots[0] / "report"
    records, states, provenance = [], [], {}
    for root in roots:
        rows, run_states, files = load_root(root, partial)
        records.extend(rows)
        states.extend(run_states)
        provenance.update(files)
    source_hash = digest(__file__)
    snapshot = out / "reporter-source" / source_hash
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in (Path(__file__), Path(__file__).with_name("report_interface_editing.py")):
        target = snapshot / source.name
        if target.exists() and digest(target) != digest(source):
            raise AssertionError("Existing reporter source snapshot changed")
        if not target.exists():
            target.write_bytes(source.read_bytes())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reporter_file": str(Path(__file__).resolve()),
        "reporter_sha256": source_hash,
        "reporter_source_snapshot": str(snapshot.resolve()),
        "scoring_dependency_sha256": digest(
            Path(__file__).with_name("report_interface_editing.py")
        ),
        "roots": list(map(str, roots)),
        "partial_requested": partial,
        "complete": all(row["status"] == "complete_and_audited" for row in states),
        "expected_runs_n": len(states),
        "completed_runs_n": sum(row["status"] == "complete_and_audited" for row in states),
        "independent_unit": "world; initializations and cases are nested paired repeated measures",
        "denominator_unit": "query evaluations, not unique queries or independent observations",
        "run_states": states,
        "by_objective_arm": aggregate(records, ("phase", "objective", "arm")),
        "by_world_objective_arm": aggregate(records, ("phase", "world", "objective", "arm")),
        "paired_comparisons": [
            paired_differences(records, contrast)
            for contrast in ("edit_minus_sham", "objective_minus_final_ce")
        ],
        "case_records": records,
        "input_files_sha256": provenance,
    }
    write(out / "summary.json", summary)
    (out / "report.md").write_text(markdown(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = report(args.root, args.out, args.partial)
    print(
        json.dumps(
            {key: result[key] for key in ("complete", "completed_runs_n", "expected_runs_n")}
        )
    )
