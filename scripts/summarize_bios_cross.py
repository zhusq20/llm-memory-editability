"""Re-score saved crossover predictions and report paired interaction contrasts."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from llm_memory_editability.bios_cross_train import edit_metrics, learning_metrics
from llm_memory_editability.bios_data import array_hash, write_json


def close(actual, expected):
    if isinstance(expected, dict):
        return all(k in actual and close(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (float, np.floating)):
        return actual is not None and bool(np.isclose(actual, expected, atol=1e-9, rtol=1e-7))
    return actual == expected


def csv_write(path, rows):
    if rows:
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(root):
    contract = json.loads((root / "launch-contract.json").read_text())
    study = contract["study"]
    worlds = {seed: make_cross_world(seed) for seed in study["worlds"]}
    learning, editing, errors, configs = [], [], [], []
    counts = {"learning_runs": 0, "learning_checkpoints": 0, "edit_cases": 0, "edit_checkpoints": 0}
    exposure_hashes = defaultdict(set)
    for path in sorted(root.glob("world-*/config.json")):
        run = path.parent
        config = json.loads(path.read_text())
        configs.append(config)
        if config["sources"] != contract["sources"] or config["study"] != study:
            errors.append(f"{run.name}: frozen configuration/source mismatch")
        w = worlds[config["world"]]
        if config["truth_sha256"] != array_hash(w.answers):
            errors.append(f"{run.name}: truth mismatch")
        common = {
            "world": config["world"],
            "seed": config["seed"],
            "condition": config["condition"],
        }
        if not (run / "learning.json").exists():
            continue
        timeline = json.loads((run / "learning.json").read_text())
        if (run / "learning-complete.json").exists() and [p["step"] for p in timeline] != study[
            "checkpoints"
        ]:
            errors.append(f"{run.name}: incomplete learning checkpoint grid")
        for point in timeline:
            step = point["step"]
            with np.load(run / f"predictions-{step}.npz") as data:
                arrays = dict(data)
            correct = (arrays["prediction"] == w.answers) & arrays["ended"]
            if not np.array_equal(correct, arrays["correct"]):
                errors.append(f"{run.name}/{step}: scoring mismatch")
            metrics = learning_metrics(w, arrays)
            if not close(point, metrics):
                errors.append(f"{run.name}/{step}: reported metric mismatch")
            independent = np.isin(w.relation, [3, 4, 5, 6])
            metrics["independent_attributes_accuracy"] = float(correct[independent].mean())
            metrics["independent_attributes_n"] = int(independent.sum())
            for chain, name in enumerate(CHAINS):
                for label, mask in (
                    ("old_exception", w.exceptions[chain]),
                    ("nonexception", ~w.exceptions[chain]),
                ):
                    ids = np.intersect1d(w.heldout_ids[chain], w.derived_ids[chain, mask])
                    metrics[f"{name}_heldout_{label}"] = float(correct[ids].mean())
                    metrics[f"{name}_heldout_{label}_n"] = len(ids)
            if arrays["exposure"][w.heldout_ids.ravel()].sum():
                errors.append(f"{run.name}/{step}: heldout QA exposure")
            expected = np.full(w.n_base, step // 128, dtype=np.int64)
            expected[w.root_ids.ravel()] *= 32
            if not np.array_equal(arrays["exposure"][: w.n_base], expected):
                errors.append(f"{run.name}/{step}: per-fact exposure mismatch")
            if arrays["exposure"][w.train_ids.ravel()].sum() != step * 40:
                errors.append(f"{run.name}/{step}: QA exposure mismatch")
            if not np.array_equal(
                arrays["slots"], np.repeat((expected // 10)[:, None], 10, axis=1)
            ):
                errors.append(f"{run.name}/{step}: position exposure mismatch")
            exposure_hashes[(w.seed, config["seed"], step)].add(
                tuple(array_hash(arrays[k]) for k in ("exposure", "slots", "weighted"))
            )
            learning.append({**common, "step": step, **metrics})
            counts["learning_checkpoints"] += 1
        if timeline[-1]["step"] == study["steps"] and (run / "learning-complete.json").exists():
            counts["learning_runs"] += 1
        else:
            continue
        with np.load(run / f"predictions-{study['steps']}.npz") as data:
            old_correct = data["correct"]
        for dest in sorted((run / "edits").glob("*")):
            if not (dest / "complete.json").exists():
                continue
            chain_name, kind, scope = dest.name.split("-")
            chain = CHAINS.index(chain_name)
            pair = edit_pair(w, chain)
            target = pair[kind]
            with np.load(dest / "sets.npz") as data:
                for field, expected in pair.items():
                    if not np.array_equal(data[field], expected):
                        errors.append(f"{run.name}/{dest.name}: set mismatch {field}")
                if not np.array_equal(data["old_correct"], old_correct):
                    errors.append(f"{run.name}/{dest.name}: baseline correct mismatch")
            edit_timeline = json.loads((dest / "trajectory.json").read_text())
            if [p["step"] for p in edit_timeline] != study["edit_checkpoints"]:
                errors.append(f"{run.name}/{dest.name}: incomplete edit checkpoint grid")
            for point in edit_timeline:
                step = point["step"]
                with np.load(dest / f"predictions-{step}.npz") as data:
                    arrays = dict(data)
                if not np.array_equal(
                    arrays["correct"], (arrays["prediction"] == target) & arrays["ended"]
                ):
                    errors.append(f"{run.name}/{dest.name}/{step}: scoring mismatch")
                metrics = edit_metrics(w, pair, arrays, old_correct)
                if not close(point, metrics):
                    errors.append(f"{run.name}/{dest.name}/{step}: metric mismatch")
                row = {
                    **common,
                    "chain": chain_name,
                    "kind": kind,
                    "scope": scope,
                    "step": step,
                    **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
                }
                for pool in ("full", "heldout"):
                    strata = metrics[f"U_{pool}"]
                    rates = [v["rate"] for v in strata.values()]
                    row[f"U_{pool}_macro_damage"] = mean(rates) if None not in rates else None
                    row[f"U_{pool}_known"] = sum(v["known"] for v in strata.values())
                    row[f"U_{pool}_broken"] = sum(v["broken"] for v in strata.values())
                    for g, v in strata.items():
                        for k, value in v.items():
                            row[f"U_{pool}_{g}_{k}"] = value
                editing.append(row)
                counts["edit_checkpoints"] += 1
            counts["edit_cases"] += 1
    for key, values in exposure_hashes.items():
        if len(values) != 1:
            errors.append(f"{key}: cross-condition exposure mismatch")
    by_block = defaultdict(list)
    for c in configs:
        by_block[(c["world"], c["seed"])].append(c)
    for key, cs in by_block.items():
        for field in ("initial_sha256", "truth_sha256", "prompts_sha256", "qa_sha256"):
            if len({c[field] for c in cs}) != 1:
                errors.append(f"{key}: paired {field} mismatch")
    paired = []
    for step in study["checkpoints"]:
        for w in study["worlds"]:
            for seed in study["seeds"]:
                group = {
                    r["condition"]: r
                    for r in learning
                    if (r["step"], r["world"], r["seed"]) == (step, w, seed)
                }
                if set(group) != set(CONDITIONS):
                    continue
                company, project, neutral = (group[c] for c in CONDITIONS)
                matched = mean(
                    [
                        company["company_heldout"] - neutral["company_heldout"],
                        project["project_heldout"] - neutral["project_heldout"],
                    ]
                )
                transfer = mean(
                    [
                        company["project_heldout"] - neutral["project_heldout"],
                        project["company_heldout"] - neutral["company_heldout"],
                    ]
                )
                paired.append(
                    {
                        "world": w,
                        "seed": seed,
                        "step": step,
                        "matching_effect": matched - transfer,
                        "matched_vs_neither": matched,
                        "mismatched_vs_neither": transfer,
                        "company_mean_vs_neither": company["mean_heldout"]
                        - neutral["mean_heldout"],
                        "project_mean_vs_neither": project["mean_heldout"]
                        - neutral["mean_heldout"],
                    }
                )
    csv_write(root / "learning.csv", learning)
    csv_write(root / "editing.csv", editing)
    csv_write(root / "paired-learning.csv", paired)
    paired_edits = []
    for scope in ("mlp", "all"):
        for kind in ("coherent", "exception"):
            for world in study["worlds"]:
                for seed in study["seeds"]:
                    group = {
                        (r["condition"], r["chain"]): r
                        for r in editing
                        if (r["scope"], r["kind"], r["world"], r["seed"], r["step"])
                        == (scope, kind, world, seed, study["edit_steps"])
                    }
                    if len(group) != 6:
                        continue
                    for metric in ("E", "D", "D_heldout", "D_conflict"):
                        cc, cp, pc, pp, nc, np_ = [
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
                        paired_edits.append(
                            {
                                "world": world,
                                "seed": seed,
                                "kind": kind,
                                "scope": scope,
                                "metric": metric,
                                "matching_effect": ((cc - pc) + (pp - cp)) / 2,
                                "matched_vs_neither": ((cc - nc) + (pp - np_)) / 2,
                                "mismatched_vs_neither": ((cp - np_) + (pc - nc)) / 2,
                            }
                        )
    csv_write(root / "paired-editing.csv", paired_edits)
    complete = (
        counts["learning_runs"] == study["learning_runs"]
        and counts["edit_cases"] == study["edit_cases"]
    )
    write_json(
        root / "audit.json",
        {**counts, "errors": errors, "passed": not errors, "complete": complete},
    )
    lines = [
        "# 训练组织 × 测试关系交叉实验",
        "",
        f"学习 {counts['learning_runs']}/{study['learning_runs']}；"
        f"编辑 {counts['edit_cases']}/{study['edit_cases']}；"
        f"已复核 {counts['learning_checkpoints']} 个学习、"
        f"{counts['edit_checkpoints']} 个编辑检查点；审计错误 {len(errors)}。",
        "",
        "两世界×两初始化构成四个配对块，仅两个数据世界；分数为完整答案生成准确率。",
        "",
        "## 固定15360步学习终点",
        "",
        "| 训练组织 | 完整块数 | 基础事实 | 公司链留出 | 项目链留出 | 两任务平均 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    final = [r for r in learning if r["step"] == study["steps"]]
    for c in CONDITIONS:
        group = [r for r in final if r["condition"] == c]
        if group:
            values = [
                f"{mean(r[k] for r in group) * 100:.2f}%"
                for k in ("base_accuracy", "company_heldout", "project_heldout", "mean_heldout")
            ]
            lines.append(f"| {c} | {len(group)} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 配对学习比较",
        "",
        "单位为百分点；匹配效应=匹配相对中性增益−不匹配相对中性增益。",
        "",
        "| 步数 | 配对块数 | 匹配效应 | 匹配−中性 | 不匹配−中性 | "
        "公司组织两任务平均−中性 | 项目组织两任务平均−中性 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for step in study["checkpoints"]:
        group = [r for r in paired if r["step"] == step]
        if group:
            values = [
                f"{100 * mean(r[k] for r in group):+.3f}"
                for k in (
                    "matching_effect",
                    "matched_vs_neither",
                    "mismatched_vs_neither",
                    "company_mean_vs_neither",
                    "project_mean_vs_neither",
                )
            ]
            lines.append(f"| {step} | {len(group)} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 固定512步例外编辑",
        "",
        "| 组织 | 测试链 | 方法 | 块数 | E | D | 留出D | 冲突D | "
        "全部保持损伤条数均值 | 留出保持损伤条数均值 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in CONDITIONS:
        for chain in CHAINS:
            for scope in ("mlp", "all"):
                group = [
                    r
                    for r in editing
                    if (r["condition"], r["chain"], r["kind"], r["scope"], r["step"])
                    == (c, chain, "exception", scope, study["edit_steps"])
                ]
                if group:
                    values = [
                        f"{100 * mean(r[k] for r in group):.2f}%"
                        for k in ("E", "D", "D_heldout", "D_conflict")
                    ]
                    values += [
                        f"{mean(r[k] for r in group):.2f}"
                        for k in ("U_full_broken", "U_heldout_broken")
                    ]
                    lines.append(
                        f"| {c} | {chain} | {scope} | {len(group)} | " + " | ".join(values) + " |"
                    )
    lines += [
        "",
        "所有原始预测、学习与编辑检查点均保留。连续指标不以联合达标替代。",
        "新增对称项目链与中性人物聚合对照，不能直接与历史A/B/C数值作受控比较。",
        "匹配收益和跨任务平均收益可以并存；没有一致排序时保留交互与取舍，不强行选赢家。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({**counts, "errors": errors, "complete": complete}))
    if errors:
        raise ValueError("Crossover audit failed")
    return learning, paired, editing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bios-cross-dev-v1")
    args = parser.parse_args()
    summarize(Path(args.output))
