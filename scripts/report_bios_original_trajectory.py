#!/usr/bin/env python3
"""Independently audit native attribute trajectories and match their people to existing QA."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

OUTPUT = Path("docs/development-artifacts/bios-original-trajectory-v1")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    lock = json.loads((OUTPUT / "lock.json").read_text())
    cfg = lock["config"]
    cases = json.loads((OUTPUT / "cases.json").read_text())
    lookup = {(r["world"], r["person_id"], r["attribute"]): r for r in cases}
    expected = {
        (w, i, c, n)
        for w in cfg["world_seeds"]
        for i in cfg["initialization_seeds"]
        for c in cfg["pretrain"]["conditions"]
        for n in lock["nodes"]
    }
    found, node_rows, inputs = set(), [], []
    for path in sorted((OUTPUT / "nodes").glob("*.json")):
        node = json.loads(path.read_text())
        key = (node["world"], node["initialization"], node["condition"], node["passes"])
        if key in found or key not in expected:
            raise ValueError("Unexpected/duplicate model node")
        found.add(key)
        if node["lock_sha256"] != sha(OUTPUT / "lock.json"):
            raise ValueError("Source lock mismatch")
        seen = set()
        for row in node["predictions"]:
            case_key = (node["world"], row["person_id"], row["attribute"])
            if case_key in seen:
                raise ValueError("Duplicate native case")
            seen.add(case_key)
            if any(row[k] != v for k, v in lookup[case_key].items()):
                raise ValueError("Frozen case was altered")
            n = len(row["target_ids"])
            correct = row["generated_ids"][:n] == row["target_ids"]
            boundary = correct and row["generated_ids"][n] == row["boundary_token"]
            if correct != row["attribute_exact"] or boundary != row["attribute_and_boundary_exact"]:
                raise ValueError("Greedy scoring mismatch")
            if len(row["generated_ids"]) != lock["max_new_tokens"]:
                raise ValueError("Generation length changed")
            if not math.isclose(
                math.exp(-row["full_attribute_nll"]),
                row["full_attribute_probability"],
                rel_tol=1e-12,
            ):
                raise ValueError("Full-attribute probability mismatch")
            node_rows.append(dict(row, condition=node["condition"], passes=node["passes"]))
        if seen != {k for k in lookup if k[0] == node["world"]}:
            raise ValueError("Missing native cases")
        inputs.append(dict(path=str(path), sha256=sha(path)))
    if found != expected:
        raise ValueError(f"Incomplete matrix: {len(found)}/{len(expected)}")
    qa = {}
    for world in cfg["world_seeds"]:
        for condition in cfg["pretrain"]["conditions"]:
            path = Path(cfg["result_root"]) / (
                f"world-{world}/init-{cfg['initialization_seeds'][0]}/{condition}"
                "/adapt/evaluation/predictions.jsonl"
            )
            for line in path.open():
                row = json.loads(line)
                case_key = (world, row["person_id"], row["task"])
                if row["view"] != 0 or row["cot"] or case_key not in lookup:
                    continue
                if row["split"] != "dev":
                    raise ValueError("Selected person is not in the frozen development split")
                key = (*case_key, condition)
                if key in qa:
                    raise ValueError("Duplicate matched QA example")
                qa[key] = row
            inputs.append(dict(path=str(path), sha256=sha(path)))
    matches = []
    for row in node_rows:
        if row["passes"] != 540:
            continue
        q = qa[(row["world"], row["person_id"], row["attribute"], row["condition"])]
        if q["answer"] != row["target"]:
            raise ValueError("QA and native continuation targets differ")
        matches.append(
            dict(
                world=row["world"],
                condition=row["condition"],
                person_id=row["person_id"],
                attribute=row["attribute"],
                target=row["target"],
                native_exact=row["attribute_exact"],
                native_boundary_exact=row["attribute_and_boundary_exact"],
                native_probability=row["full_attribute_probability"],
                qa_exact=q["correct"],
                qa_generated=q["generated"],
                native_generated=row["generated"],
            )
        )
    with (OUTPUT / "matched-people-qa.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matches[0]))
        writer.writeheader()
        writer.writerows(matches)
    groups = defaultdict(list)
    for row in matches:
        groups[(row["world"], row["condition"], row["attribute"])].append(row)
    grouped = [
        dict(
            world=w,
            condition=c,
            attribute=a,
            n=len(rows),
            native_accuracy=sum(r["native_exact"] for r in rows) / len(rows),
            native_boundary_accuracy=sum(r["native_boundary_exact"] for r in rows) / len(rows),
            qa_accuracy=sum(r["qa_exact"] for r in rows) / len(rows),
        )
        for (w, c, a), rows in groups.items()
    ]
    input_hashes = [
        dict(path=r["path"], match=sha(r["path"]) == r["sha256"]) for r in lock["inputs"]
    ]
    if not all(row["match"] for row in input_hashes):
        raise ValueError("Frozen data/config/tokenizer changed")
    save(
        OUTPUT / "audit.json",
        dict(
            scope="CPU re-scoring, matrix and provenance audit; no extra model inference",
            model_nodes=len(found),
            native_predictions=len(node_rows),
            matched_QA_rows=len(matches),
            source=dict(path=__file__, sha256=sha(__file__)),
            inputs=inputs,
            failures=[],
            frozen_input_hashes=input_hashes,
        ),
    )
    save(OUTPUT / "matched-people-summary.json", grouped)
    nonexact = [r for r in node_rows if r["passes"] == 540 and not r["attribute_exact"]]
    for row in nonexact:
        row["literal_target_in_fixed_generation"] = row["target"].casefold() in (
            row["generated"].casefold()
        )
    save(OUTPUT / "endpoint-nonexact-cases.json", nonexact)
    inserted = sum(r["literal_target_in_fixed_generation"] for r in nonexact)
    lines = [
        "# 原生 BIO 属性记忆形成曲线",
        "",
        "36 个既有预训练权重、2304 条固定案例续写已完成。完整序列是目标属性的全部 token，"
        "不包含人为添加的 EOS；生成固定 9 token。CPU 独立复算逐例完整属性与原后缀首 token"
        "的得分，并核对完整矩阵、冻结样本及来源。没有新增训练。",
        "",
        "下面比较相同的每世界 32 位开发人物。BIO 列使用预训练 540 曝光终点的已见传记前缀；"
        "QA 列使用之后共同 QA 适配端点、规范问法。因此是不同训练阶段与输入条件的描述性比较，"
        "不能把差异全部解释为单一内部接口。出生地 BIO 前缀包含真实完整日期。",
        "",
        "| 条件 | 日期 BIO | 日期 BIO 加边界 | 日期 QA |"
        " 出生地 BIO | 出生地 BIO 加边界 | 出生地 QA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in ("S", "M", "MP"):
        values = []
        for attr in ("date", "birthcity"):
            data = [r for r in grouped if r["condition"] == condition and r["attribute"] == attr]
            values.extend(
                100 * sum(r[k] for r in data) / len(data)
                for k in ("native_accuracy", "native_boundary_accuracy", "qa_accuracy")
            )
        lines.append(f"| {condition} | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
    lines += [
        "",
        "各节点的完整属性概率与生成正确率见 `curves.csv`、`native-attribute-curves.png/pdf`；"
        "逐例预测包含完整前缀、目标与生成 token。完整属性概率是各例概率的均值，"
        "不等于对平均 NLL 取指数。不同长度属性不混为一个得分。",
        "",
        f"540 终点共有 {len(nonexact)} 条未命中立即属性序列；其中 {inserted} 条固定长度生成中"
        "仍出现完整目标文字，详见 `endpoint-nonexact-cases.json`。例如模型可能在正确日期前"
        "续写 `the memorable date of`，因而未通过原先冻结的立即属性计分。"
        "这些是表达差异的审阅线索，不自动改写为正确；尤其不能把全部严格失分都当作事实遗忘。",
        "",
        "候选模板覆盖、输入前提、固定样本与执行前分析约定见 `preregistration.md`、"
        "`coverage.json` 与 `lock.json`。S/M 的原 variant-0 前缀是已见顺序/措辞；"
        "MP 同句子的连续顺序实际曝光未单独重构。本实验测已见文本条件下的属性续写，"
        "不是未见表达泛化、裸模型 QA 或 reasoning 的形成曲线。",
        "",
        "复现原生评价：`PYTHONPATH=src .venv/bin/python "
        "scripts/run_bios_original_trajectory.py run`；独立复核与匹配 QA："
        "`PYTHONPATH=src .venv/bin/python scripts/report_bios_original_trajectory.py`。",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
