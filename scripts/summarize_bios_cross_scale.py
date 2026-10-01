"""Re-score every size, validate reuse, and compare the complete paired size matrix."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from summarize_bios_cross import csv_write, summarize

from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from llm_memory_editability.bios_cross_scale import (
    retention_coverage,
    validate_run_grid,
    validate_size_study,
)
from llm_memory_editability.bios_cross_train import source_hashes
from llm_memory_editability.bios_data import write_json

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def collect(out, require_complete=False):
    contract = json.loads((out / "launch-contract.json").read_text())
    manifest = contract["manifest"]
    if manifest != json.loads((ROOT / "configs/bios-cross-scale-development-v1.json").read_text()):
        raise ValueError("Scale manifest changed")
    if contract["sources"] != source_hashes():
        raise ValueError("Frozen training sources changed")
    reference = json.loads((ROOT / manifest["reference_config"]).read_text())
    if [s["width"] for s in manifest["sizes"]] != [64, 128, 256, 768]:
        raise ValueError("Incomplete size grid")
    worlds = {w: make_cross_world(w) for w in reference["worlds"]}
    all_learning, all_paired, all_editing, all_paired_edits, audits = [], [], [], [], []
    common_hashes = defaultdict(set)
    exposure_hashes = defaultdict(set)
    environments = set()
    counts = {"learning_runs": 0, "learning_checkpoints": 0, "edit_cases": 0, "edit_checkpoints": 0}
    for size in manifest["sizes"]:
        width, root = size["width"], ROOT / size["output"]
        study = json.loads((ROOT / size["config"]).read_text())
        validate_size_study(reference, study, width, size["heads"])
        frozen = json.loads((root / "launch-contract.json").read_text())
        if frozen != {"study": study, "sources": contract["sources"]}:
            raise ValueError(f"Width {width}: incompatible frozen run")
        configs = [json.loads(p.read_text()) for p in sorted(root.glob("world-*/config.json"))]
        validate_run_grid(configs, study, require_complete or size["reuse"])
        for c in configs:
            environments.add(
                tuple(c[k] for k in ("torch", "cuda", "python", "numpy", "precision", "gpu"))
            )
            if c["parameters"] != size["parameters_worlds_0_1"][c["world"]]:
                raise ValueError("Parameter count mismatch")
            for field in ("truth_sha256", "prompts_sha256", "qa_sha256", "documents_sha256"):
                common_hashes[(c["world"], c["seed"], c["condition"], field)].add(c[field])
        learning, paired, editing = summarize(root)
        audit = json.loads((root / "audit.json").read_text())
        if (size["reuse"] or require_complete) and not audit["complete"]:
            raise ValueError(f"Width {width}: incomplete required batch")
        for key in counts:
            counts[key] += audit[key]
        audits.append({"width": width, "reused": size["reuse"], **audit})
        config_index = {(c["world"], c["seed"], c["condition"]): c for c in configs}

        def decorated(row, width=width):
            return {"width": width, **row}

        for row in learning:
            cfg = config_index[row["world"], row["seed"], row["condition"]]
            run = root / f"world-{row['world']}-seed-{row['seed']}-{row['condition']}"
            timeline = json.loads((run / "learning.json").read_text())
            point = next(p for p in timeline if p["step"] == row["step"])
            all_learning.append(
                {
                    **decorated(row),
                    "parameters": cfg["parameters"],
                    "train_seconds": point["train_seconds"],
                    "train_matmul_flops_estimate": point["train_matmul_flops_estimate"],
                }
            )
            for field in ("exposure_sha256", "slots_sha256", "weighted_sha256"):
                exposure_hashes[(row["world"], row["seed"], row["step"], field)].add(point[field])
        all_paired.extend(decorated(row) for row in paired)
        baselines = {}
        pairs = {
            (w, chain): edit_pair(worlds[w], CHAINS.index(chain))
            for w in worlds
            for chain in CHAINS
        }
        for row in editing:
            key = row["world"], row["seed"], row["condition"]
            if key not in baselines:
                run = root / f"world-{key[0]}-seed-{key[1]}-{key[2]}"
                with np.load(run / f"predictions-{study['steps']}.npz") as data:
                    baselines[key] = data["correct"]
            added = retention_coverage(pairs[row["world"], row["chain"]], baselines[key], row)
            all_editing.append({**decorated(row), **added})
        all_paired_edits.extend(decorated(row) for row in read_csv(root / "paired-editing.csv"))
    if any(len(v) != 1 for v in common_hashes.values()):
        raise ValueError("Cross-size data or query-stream mismatch")
    if any(len(v) != 1 for v in exposure_hashes.values()):
        raise ValueError("Cross-size exposure mismatch")
    if len(environments) != 1:
        raise ValueError("Cross-size software, precision or hardware-type mismatch")
    complete = (
        counts["learning_runs"] == manifest["learning_runs"]
        and counts["edit_cases"] == manifest["edit_cases"]
        and all(a["complete"] for a in audits)
    )
    if require_complete and not complete:
        raise ValueError("Incomplete scale matrix")
    for name, rows in (
        ("learning", all_learning),
        ("paired-learning", all_paired),
        ("editing", all_editing),
        ("paired-editing", all_paired_edits),
    ):
        csv_write(out / f"{name}.csv", rows)
    write_json(out / "audit.json", {**counts, "complete": complete, "errors": [], "sizes": audits})
    report(out, manifest, reference, counts, complete, all_learning, all_paired, all_editing)
    print(json.dumps({**counts, "complete": complete}))


def report(out, manifest, study, counts, complete, learning, paired, editing):
    lines = [
        "# 全规模训练组织 × 测试关系交叉评测",
        "",
        f"状态：{'完整矩阵已完成' if complete else '进行中，以下为已完成部分'}。"
        f"学习{counts['learning_runs']}/{manifest['learning_runs']}；"
        f"编辑{counts['edit_cases']}/{manifest['edit_cases']}。",
        "",
        "每格为两个开发世界×两个初始化的等权均值；四块不是四个独立世界。"
        "每个规模相同15360步、相同数据曝光、相同学习率；计算量随规模变化。"
        "复用已完成最大档，三档小模型在新增项目关系的共同数据上重新训练。",
        "",
        "## 固定预算学习",
        "",
        "| 参数量 | 组织 | 块数 | 基础事实 | 公司留出QA | 项目留出QA | 两任务平均 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    terminal = [r for r in learning if r["step"] == study["steps"]]
    for size in manifest["sizes"]:
        for condition in CONDITIONS:
            rows = [
                r for r in terminal if (r["width"], r["condition"]) == (size["width"], condition)
            ]
            if rows:
                values = [
                    f"{100 * mean(r[k] for r in rows):.2f}%"
                    for k in ("base_accuracy", "company_heldout", "project_heldout", "mean_heldout")
                ]
                lines.append(
                    f"| {size['parameters_worlds_0_1'][0] / 1e6:.3f}M | {condition} | "
                    f"{len(rows)} | " + " | ".join(values) + " |"
                )
    lines += [
        "",
        "## 配对学习效应（百分点）",
        "",
        "正匹配效应表示组织与测试关系相同时更有利；另列跨两任务平均收益，二者可以并存。"
        "范围是四块的最小/最大值，不是置信区间。",
        "",
        "| 宽度 | 块数 | 匹配效应 | 块间范围 | 匹配任务−中性 | 不匹配任务−中性 | "
        "公司组织均值−中性 | 项目组织均值−中性 |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for size in manifest["sizes"]:
        rows = [r for r in paired if (r["width"], r["step"]) == (size["width"], study["steps"])]
        if rows:
            effects = [100 * r["matching_effect"] for r in rows]
            gains = [
                100 * mean(r[k] for r in rows)
                for k in (
                    "matched_vs_neither",
                    "mismatched_vs_neither",
                    "company_mean_vs_neither",
                    "project_mean_vs_neither",
                )
            ]
            lines.append(
                f"| {size['width']} | {len(rows)} | {mean(effects):+.3f} | "
                f"{min(effects):+.3f} … {max(effects):+.3f} | "
                + " | ".join(f"{gain:+.3f}" for gain in gains)
                + " |"
            )
    lines += [
        "",
        "## 512步例外编辑",
        "",
        "每格对两条链和四个配对块等权平均。保持分母仅含编辑前答对的旧知识；"
        "覆盖率=这些旧正确条数/整个保持池。损伤率先在单案例计算，再等权平均。"
        "各分层的已知/损伤条数及编辑器留出保持结果见editing.csv，不能以全池均值替代局部损伤。",
        "",
        "| 宽度 | 组织 | 范围 | 案例数 | 原传播知识准确率 | E | 留出D | 冲突D | "
        "旧知识覆盖率 | 平均损伤条数/原正确条数 | 保持损伤率 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for size in manifest["sizes"]:
        for condition in CONDITIONS:
            for scope in ("mlp", "all"):
                rows = [
                    r
                    for r in editing
                    if (r["width"], r["condition"], r["scope"], r["kind"], r["step"])
                    == (size["width"], condition, scope, "exception", study["edit_steps"])
                ]
                if not rows:
                    continue
                values = [
                    f"{100 * mean(r[k] for r in rows):.2f}%"
                    for k in ("D_old_accuracy", "E", "D_heldout", "D_conflict", "U_full_coverage")
                ]
                known, broken = [
                    mean(r[k] for r in rows) for k in ("U_full_known", "U_full_broken")
                ]
                rates = [r["U_full_micro_damage"] for r in rows]
                rate = f"{100 * mean(rates):.3f}%" if all(r is not None for r in rates) else "NA"
                lines.append(
                    f"| {size['width']} | {condition} | {scope} | {len(rows)} | "
                    + " | ".join(values)
                    + f" | {broken:.1f}/{known:.1f} | {rate} |"
                )
    lines += [
        "",
        "固定预算低分可能来自未收敛或超参数适配，不能识别容量上限。"
        "原知识学得不同也会影响编辑与保持，条件子集分数附在learning.csv并注明分母。"
        "已训练QA、留出QA、旧例外与普通成员、完整轨迹和所有一致/例外编辑均保留。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bios-cross-scale-dev-v1")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    collect(ROOT / args.output, args.require_complete)
