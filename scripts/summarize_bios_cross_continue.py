"""Independently audit P2 continuations and pair them with their exact parent runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from llm_memory_editability.bios_cross import (
    CHAINS,
    CONDITIONS,
    documents,
    edit_pair,
    make_cross_world,
    qa_schedule,
)
from llm_memory_editability.bios_cross_continue import (
    CHECKPOINTS,
    EDIT_CHECKPOINTS,
    FINAL_STEP,
    PARENT_STEP,
    continuation_sources,
    edit_metrics,
    expected_exposure,
    file_hash,
    validate_optimizer,
)
from llm_memory_editability.bios_cross_train import learning_metrics
from llm_memory_editability.bios_data import array_hash, rng_for, write_json
from llm_memory_editability.bios_organization_train import state_hash


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def close(actual, expected):
    if isinstance(expected, dict):
        return all(key in actual and close(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (float, np.floating)):
        return actual is not None and bool(np.isclose(actual, expected, atol=1e-9, rtol=1e-7))
    return actual == expected


def load_arrays(path, truth):
    with np.load(path) as saved:
        arrays = dict(saved)
    correct = (arrays["prediction"] == truth) & arrays["ended"]
    np.testing.assert_array_equal(correct, arrays["correct"])
    if len(correct) != len(truth) or not np.isfinite(arrays["value_nll"]).all():
        raise ValueError(f"Malformed prediction arrays: {path}")
    return arrays


def extra_learning(world, arrays):
    result = learning_metrics(world, arrays)
    correct = arrays["correct"]
    independent = np.isin(world.relation, (3, 4, 5, 6))
    result["independent_attributes_accuracy"] = float(correct[independent].mean())
    result["independent_attributes_n"] = int(independent.sum())
    for chain, name in enumerate(CHAINS):
        for label, mask in (
            ("old_exception", world.exceptions[chain]),
            ("nonexception", ~world.exceptions[chain]),
        ):
            ids = np.intersect1d(world.heldout_ids[chain], world.derived_ids[chain, mask])
            result[f"{name}_heldout_{label}"] = float(correct[ids].mean())
            result[f"{name}_heldout_{label}_n"] = len(ids)
        result[f"{name}_heldout_balanced"] = mean(
            result[f"{name}_heldout_{label}"] for label in ("old_exception", "nonexception")
        )
    return result


def flatten_edit(metrics):
    row = {key: value for key, value in metrics.items() if not isinstance(value, dict)}
    for pool in ("full", "heldout"):
        strata = metrics[f"U_{pool}"]
        row[f"U_{pool}_known"] = sum(value["known"] for value in strata.values())
        row[f"U_{pool}_broken"] = sum(value["broken"] for value in strata.values())
        row[f"U_{pool}_damage"] = (
            row[f"U_{pool}_broken"] / row[f"U_{pool}_known"] if row[f"U_{pool}_known"] else None
        )
        rates = [value["rate"] for value in strata.values()]
        row[f"U_{pool}_macro_damage"] = mean(rates) if None not in rates else None
        for group, values in strata.items():
            row.update({f"U_{pool}_{group}_{key}": value for key, value in values.items()})
    return row


def paired_learning(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["branch"], row["width"], row["world"], row["seed"], row["step"])][
            row["condition"]
        ] = row
    result = []
    for (branch, width, world, seed, step), group in sorted(groups.items()):
        if set(group) != set(CONDITIONS):
            continue
        company, project, neutral = (group[condition] for condition in CONDITIONS)
        matched = (
            (company["company_heldout"] - neutral["company_heldout"])
            + (project["project_heldout"] - neutral["project_heldout"])
        ) / 2
        mismatched = (
            (company["project_heldout"] - neutral["project_heldout"])
            + (project["company_heldout"] - neutral["company_heldout"])
        ) / 2
        result.append(
            {
                "branch": branch,
                "width": width,
                "world": world,
                "seed": seed,
                "step": step,
                "matching_effect": matched - mismatched,
                "matched_vs_neither": matched,
                "mismatched_vs_neither": mismatched,
                "company_mean_vs_neither": company["mean_heldout"] - neutral["mean_heldout"],
                "project_mean_vs_neither": project["mean_heldout"] - neutral["mean_heldout"],
                "company_base_vs_neither": company["base_accuracy"] - neutral["base_accuracy"],
                "project_base_vs_neither": project["base_accuracy"] - neutral["base_accuracy"],
            }
        )
    return result


def paired_editing(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["step"] == 512:
            key = (
                row["branch"],
                row["width"],
                row["world"],
                row["seed"],
                row["phase"],
                row["kind"],
            )
            groups[key][(row["condition"], row["chain"])] = row
    result = []
    for (branch, width, world, seed, phase, kind), group in sorted(groups.items()):
        if len(group) != 6:
            continue
        for metric in (
            "E",
            "D",
            "D_heldout",
            "D_conflict",
            "D_conflict_heldout",
            "U_full_damage",
            "U_heldout_damage",
        ):
            values = [
                group[key][metric]
                for key in (
                    ("company", "company"),
                    ("company", "project"),
                    ("project", "company"),
                    ("project", "project"),
                    ("neither", "company"),
                    ("neither", "project"),
                )
            ]
            if None in values:
                continue
            cc, cp, pc, pp, nc, np_ = values
            result.append(
                {
                    "branch": branch,
                    "width": width,
                    "world": world,
                    "seed": seed,
                    "phase": phase,
                    "kind": kind,
                    "metric": metric,
                    "matching_effect": ((cc - pc) + (pp - cp)) / 2,
                    "matched_vs_neither": ((cc - nc) + (pp - np_)) / 2,
                    "mismatched_vs_neither": ((cp - np_) + (pc - nc)) / 2,
                }
            )
    return result


def summarize(root, require_complete=False):
    contract = json.loads((root / "launch-contract.json").read_text())
    if contract["sources"] != continuation_sources():
        raise ValueError("P2 frozen training source hash mismatch")
    jobs = contract["jobs"]
    if len(jobs) != 36 or len({job["output"] for job in jobs}) != 36:
        raise ValueError("P2 manifest does not contain 36 unique jobs")
    worlds = {seed: make_cross_world(seed) for seed in (0, 1)}
    schedules = {seed: qa_schedule(world, FINAL_STEP) for seed, world in worlds.items()}
    learning, editing, errors = [], [], []
    counts = {
        "learning_runs": 0,
        "learning_checkpoints": 0,
        "edit_cases": 0,
        "edit_checkpoints": 0,
        "parent_reference_case_uses": 0,
    }
    exposure_hashes, configs = defaultdict(set), defaultdict(list)
    verified_parents = {}
    for job in jobs:
        run, parent = Path(job["output"]), Path(job["parent"])
        if not (run / "config.json").exists():
            continue
        config = json.loads((run / "config.json").read_text())
        common = {key: job[key] for key in ("branch", "width", "world", "seed", "condition")}
        w = worlds[job["world"]]
        try:
            for key in ("world", "seed", "condition"):
                if config[key] != job[key]:
                    raise ValueError(f"Identity mismatch: {key}")
            if config["study"]["lr"] != job["lr"] or config["study"]["width"] != job["width"]:
                raise ValueError("Continuation branch or size mismatch")
            if config["sources"] != contract["sources"]:
                raise ValueError("Frozen worker source mismatch")
            if config["parent"]["directory"] != str(parent.resolve()):
                raise ValueError("Parent directory mismatch")
            if str(parent) not in verified_parents:
                for filename, wanted in config["parent"]["files_sha256"].items():
                    if file_hash(parent / filename) != wanted:
                        raise ValueError(f"Parent file changed: {filename}")
                verified_parents[str(parent)] = config["parent"]
            elif verified_parents[str(parent)] != config["parent"]:
                raise ValueError("Branches disagree on the same parent provenance")
            docs, schedule = documents(w, job["condition"]), schedules[w.seed]
            for key, value in (
                ("truth_sha256", w.answers),
                ("prompts_sha256", w.prompts),
                ("documents_sha256", docs),
                ("qa_sha256", schedule),
            ):
                if config[key] != array_hash(value):
                    raise ValueError(f"Frozen data mismatch: {key}")
            if not (run / "learning.json").exists():
                continue
            with np.load(run / "schedule.npz") as actual:
                np.testing.assert_array_equal(actual["documents"], docs)
                np.testing.assert_array_equal(actual["qa"], schedule)
            with np.load(parent / "schedule.npz") as actual:
                np.testing.assert_array_equal(actual["qa"], schedule[:PARENT_STEP])
            configs[(job["branch"], job["width"], w.seed, job["seed"])].append(config)
            timeline = json.loads((run / "learning.json").read_text())
            complete_learning = (run / "learning-complete.json").exists()
            steps = [point["step"] for point in timeline]
            if steps != list(CHECKPOINTS[: len(steps)]) or (
                complete_learning and steps != list(CHECKPOINTS)
            ):
                raise ValueError("Continuation checkpoint grid mismatch")
            old_weighted = expected_exposure(w, docs, schedule, PARENT_STEP, 0.0001)[2]
            new_prefix = expected_exposure(w, docs, schedule, PARENT_STEP, job["lr"])[2]
            for point in timeline:
                step = point["step"]
                arrays = load_arrays(run / f"predictions-{step}.npz", w.answers)
                metrics = learning_metrics(w, arrays)
                if not close(point, metrics):
                    raise ValueError(f"Learning metric mismatch at {step}")
                wanted_counts, wanted_slots, wanted_weighted = expected_exposure(
                    w, docs, schedule, step, job["lr"]
                )
                wanted_weighted += old_weighted - new_prefix
                np.testing.assert_array_equal(arrays["exposure"], wanted_counts)
                np.testing.assert_array_equal(arrays["slots"], wanted_slots)
                np.testing.assert_allclose(arrays["weighted"], wanted_weighted, rtol=0, atol=1e-10)
                for label, field in (
                    ("exposure_sha256", "exposure"),
                    ("slots_sha256", "slots"),
                    ("weighted_sha256", "weighted"),
                ):
                    if point[label] != array_hash(arrays[field]):
                        raise ValueError(f"Exposure hash mismatch at {step}")
                if step == PARENT_STEP:
                    with np.load(parent / f"predictions-{PARENT_STEP}.npz") as original:
                        for key in original.files:
                            np.testing.assert_array_equal(arrays[key], original[key])
                exposure_hashes[(job["branch"], job["width"], w.seed, job["seed"], step)].add(
                    tuple(array_hash(arrays[key]) for key in ("exposure", "slots", "weighted"))
                )
                learning.append({**common, "step": step, **extra_learning(w, arrays)})
                counts["learning_checkpoints"] += 1
            if not complete_learning:
                continue
            completed = json.loads((run / "learning-complete.json").read_text())
            if not close(completed["final"], timeline[-1]):
                raise ValueError("Final learning metadata mismatch")
            resumed = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
            validate_optimizer(resumed["optimizer"], FINAL_STEP, job["lr"])
            if resumed["step"] != FINAL_STEP or resumed["config_sha256"] != file_hash(
                run / "config.json"
            ):
                raise ValueError("Final learning resume mismatch")
            if state_hash(resumed["model"]) != completed["model_sha256"]:
                raise ValueError("Final learning model hash mismatch")
            for key, source in (
                ("counts", "exposure"),
                ("slots", "slots"),
                ("weighted", "weighted"),
            ):
                np.testing.assert_array_equal(resumed[key], arrays[source])
            del resumed
            counts["learning_runs"] += 1
            unexpected = [
                path.name
                for path in (run / "edits").glob("*")
                if path.is_dir() and not path.name.endswith("-mlp")
            ]
            if unexpected:
                raise ValueError(f"Unexpected edit scopes: {unexpected}")
            for phase, location, learning_step in (
                ("parent", parent, PARENT_STEP),
                ("continued", run, FINAL_STEP),
            ):
                old_arrays = load_arrays(location / f"predictions-{learning_step}.npz", w.answers)
                for chain, name in enumerate(CHAINS):
                    pair = edit_pair(w, chain)
                    rng = rng_for(w.seed, 908, chain)
                    e_choices = rng.integers(len(pair["E"]), size=(512, 128))
                    r_choices = rng.integers(len(pair["replay"]), size=(512, 128))
                    for kind in ("coherent", "exception"):
                        dest = location / "edits" / f"{name}-{kind}-mlp"
                        if not (dest / "complete.json").exists():
                            continue
                        with np.load(dest / "sets.npz") as actual:
                            for key, wanted in pair.items():
                                np.testing.assert_array_equal(actual[key], wanted)
                            np.testing.assert_array_equal(
                                actual["old_correct"], old_arrays["correct"]
                            )
                            np.testing.assert_array_equal(actual["edit_sampling"], e_choices)
                            np.testing.assert_array_equal(actual["replay_sampling"], r_choices)
                        edit_timeline = json.loads((dest / "trajectory.json").read_text())
                        if [point["step"] for point in edit_timeline] != list(EDIT_CHECKPOINTS):
                            raise ValueError(f"Incomplete edit grid: {phase}/{dest.name}")
                        for point in edit_timeline:
                            arrays = load_arrays(
                                dest / f"predictions-{point['step']}.npz", pair[kind]
                            )
                            metrics = edit_metrics(w, pair, arrays, old_arrays["correct"])
                            # Added E strata and conflict-heldout metrics were absent in v2.7.
                            required = (
                                metrics
                                if phase == "continued"
                                else {key: value for key, value in metrics.items() if key in point}
                            )
                            if not close(point, required):
                                raise ValueError(
                                    f"Edit metric mismatch: {phase}/{dest.name}/{point['step']}"
                                )
                            if point["step"] == 0:
                                np.testing.assert_array_equal(
                                    arrays["prediction"], old_arrays["prediction"]
                                )
                                np.testing.assert_array_equal(arrays["ended"], old_arrays["ended"])
                            editing.append(
                                {
                                    **common,
                                    "phase": phase,
                                    "learning_step": learning_step,
                                    "chain": name,
                                    "kind": kind,
                                    "scope": "mlp",
                                    "step": point["step"],
                                    **flatten_edit(metrics),
                                }
                            )
                            if phase == "continued":
                                counts["edit_checkpoints"] += 1
                        counts[
                            "edit_cases" if phase == "continued" else "parent_reference_case_uses"
                        ] += 1
        except (ValueError, AssertionError, KeyError, FileNotFoundError) as error:
            errors.append(f"{job['id']}: {error}")
    for key, values in exposure_hashes.items():
        if len(values) != 1:
            errors.append(f"{key}: paired exposure differs across organizations")
    for key, values in configs.items():
        for field in ("initial_sha256", "truth_sha256", "prompts_sha256", "qa_sha256"):
            if len({config[field] for config in values}) != 1:
                errors.append(f"{key}: paired identity differs for {field}")
    learning_pairs, edit_pairs = paired_learning(learning), paired_editing(editing)
    csv_write(root / "learning.csv", learning)
    csv_write(root / "editing.csv", editing)
    csv_write(root / "paired-learning.csv", learning_pairs)
    csv_write(root / "paired-editing.csv", edit_pairs)
    lookup = {
        (row["branch"], row["width"], row["world"], row["seed"], row["condition"], row["step"]): row
        for row in learning
    }
    deltas, rate_deltas = [], []
    for row in learning:
        if row["step"] != FINAL_STEP:
            continue
        fields = ("branch", "width", "world", "seed", "condition")
        base_key = tuple(row[key] for key in fields)
        previous = lookup[(*base_key, PARENT_STEP)]
        deltas.append(
            {
                **{key: row[key] for key in fields},
                **{
                    f"delta_{key}": row[key] - previous[key]
                    for key in (
                        "base_accuracy",
                        "mean_heldout",
                        "company_heldout",
                        "project_heldout",
                        "company_heldout_old_exception",
                        "project_heldout_old_exception",
                    )
                },
            }
        )
        if row["branch"] == "sensitivity":
            reference = lookup.get(("common", *base_key[1:], FINAL_STEP))
            if reference is not None:
                rate_deltas.append(
                    {
                        **{key: row[key] for key in fields if key != "branch"},
                        **{
                            f"sensitivity_minus_common_{key}": row[key] - reference[key]
                            for key in (
                                "base_accuracy",
                                "mean_heldout",
                                "company_heldout",
                                "project_heldout",
                            )
                        },
                    }
                )
    csv_write(root / "learning-change.csv", deltas)
    csv_write(root / "learning-rate-comparison.csv", rate_deltas)
    old_edits = {
        tuple(
            row[key] for key in ("branch", "width", "world", "seed", "condition", "chain", "kind")
        ): row
        for row in editing
        if row["phase"] == "parent" and row["step"] == 512
    }
    edit_deltas = []
    for row in editing:
        if row["phase"] != "continued" or row["step"] != 512:
            continue
        fields = ("branch", "width", "world", "seed", "condition", "chain", "kind")
        previous = old_edits.get(tuple(row[key] for key in fields))
        if previous is None:
            continue
        values = {}
        for key in ("E", "D_heldout", "D_conflict_heldout", "U_full_damage", "U_heldout_damage"):
            values[f"delta_{key}"] = (
                row[key] - previous[key]
                if row[key] is not None and previous[key] is not None
                else None
            )
        edit_deltas.append({**{key: row[key] for key in fields}, **values})
    csv_write(root / "editing-change.csv", edit_deltas)
    complete = counts["learning_runs"] == 36 and counts["edit_cases"] == 144
    audit = {
        **counts,
        "expected_learning_runs": 36,
        "expected_learning_checkpoints": 144,
        "expected_edit_cases": 144,
        "expected_edit_checkpoints": 576,
        "errors": errors,
        "passed": not errors,
        "complete": complete,
        "parent_files_verified": len(verified_parents),
    }
    write_json(root / "audit.json", audit)
    report(root, learning, editing, learning_pairs, audit)
    if errors or (require_complete and not complete):
        raise ValueError(f"P2 audit failed or required completion missing: {audit}")
    return audit


def report(root, learning, editing, pairs, audit):
    lines = [
        "# P2：共同续训与学习率敏感性",
        "",
        f"学习 {audit['learning_runs']}/36；新编辑 {audit['edit_cases']}/144；"
        f"核验 {audit['learning_checkpoints']} 个学习、"
        f"{audit['edit_checkpoints']} 个新编辑检查点。审计错误 {len(audit['errors'])}。",
        "",
        "common：原1e-4续训，两个初始化；sensitivity：同一父节点改3e-4，仅初始化0。学习率之间仅作初始化0的逐块配对，不能直接比较两分支全体均值。所有结果仍只有两个开发世界。",
        "",
        "## 固定30720步学习终点",
        "",
        "| 分支 | 宽度 | 组织 | 块数 | 基础事实 | 公司留出 | 项目留出 | 公司旧例外 | 项目旧例外 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in ("common", "sensitivity"):
        for width in (128, 256):
            for condition in CONDITIONS:
                group = [
                    row
                    for row in learning
                    if (row["branch"], row["width"], row["condition"], row["step"])
                    == (branch, width, condition, FINAL_STEP)
                ]
                if group:
                    values = [
                        f"{100 * mean(row[key] for row in group):.2f}%"
                        for key in (
                            "base_accuracy",
                            "company_heldout",
                            "project_heldout",
                            "company_heldout_old_exception",
                            "project_heldout_old_exception",
                        )
                    ]
                    lines.append(
                        f"| {branch} | {width} | {condition} | {len(group)} | "
                        + " | ".join(values)
                        + " |"
                    )
    lines += [
        "",
        "## 学习组织配对效应",
        "",
        "单位为百分点，四块或两块不是四个或两个以上独立数据世界。",
        "",
        "| 分支 | 宽度 | 步数 | 配对块 | 匹配收益 | 不匹配任务−中性 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in ("common", "sensitivity"):
        for width in (128, 256):
            for step in CHECKPOINTS:
                group = [
                    row
                    for row in pairs
                    if (row["branch"], row["width"], row["step"]) == (branch, width, step)
                ]
                if group:
                    lines.append(
                        f"| {branch} | {width} | {step} | {len(group)} | "
                        f"{100 * mean(row['matching_effect'] for row in group):+.3f} | "
                        f"{100 * mean(row['mismatched_vs_neither'] for row in group):+.3f} |"
                    )
    lines += [
        "",
        "## 512步MLP编辑：原父模型与续训模型",
        "",
        "损伤是每案例旧正确知识的损伤率，再对案例等权平均；parent行是历史编辑的引用，不是新实验。",
        "",
        "| 分支 | 宽度 | 阶段 | 更新 | 案例 | E | 留出D | 留出冲突D | 全池损伤 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for branch in ("common", "sensitivity"):
        for width in (128, 256):
            for phase in ("parent", "continued"):
                for kind in ("coherent", "exception"):
                    group = [
                        row
                        for row in editing
                        if (row["branch"], row["width"], row["phase"], row["kind"], row["step"])
                        == (branch, width, phase, kind, 512)
                    ]
                    if group:
                        values = [
                            f"{100 * mean(row[key] for row in group if row[key] is not None):.3f}%"
                            for key in ("E", "D_heldout", "D_conflict_heldout", "U_full_damage")
                        ]
                        lines.append(
                            f"| {branch} | {width} | {phase} | {kind} | {len(group)} | "
                            + " | ".join(values)
                            + " |"
                        )
    lines += [
        "",
        "coherent行的冲突子集采用与exception配对的相同人物集合，其coherent目标本身不冲突。详细已知覆盖、整数损伤与全部检查点见editing.csv；学习率敏感性只读learning-rate-comparison.csv中的同初始化配对。",
        "",
        "共同续训效应与相近知识水平分析不等同于组织的直接因果效应；不据固定预算的学习不足宣称容量上限。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bios-mechanism-dev-v1/p2")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.output).resolve(), args.require_complete), indent=2))


if __name__ == "__main__":
    main()
