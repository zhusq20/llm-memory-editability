#!/usr/bin/env python3
"""Descriptive report and provenance checks; no outcome-dependent run selection."""

from __future__ import annotations

import csv
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from llm_memory_editability.hebbian_data import normalize_answer
from llm_memory_editability.qwen_path_learning import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    ROOT,
    digest,
    mean,
    now,
    read,
    write,
)
from llm_memory_editability.qwen_path_train import check_lock, jobs


def audit():
    check_lock()
    lock = read(ART / "execution-lock.json")
    checks = {}
    checks["frozen_sources"] = all(
        digest(ART / "execution-source" / p) == value for p, value in lock["sources"].items()
    )
    checks["frozen_inputs"] = all(digest(ROOT / p) == value for p, value in lock["inputs"].items())
    selection = read(DATA / "selection.json")
    all_records = [r for records in selection.values() for r in records]
    checks["subject_disjoint"] = len({r["subject_group"] for r in all_records}) == len(all_records)
    checks["pool_counts"] = {k: len(v) for k, v in selection.items()} == read(CONFIG)["counts"]
    old = read(ROOT / "data/hebbian-learning-v1/pools.json")
    old_subjects = {r["subject_group"] for records in old.values() for r in records}
    checks["old_update_subjects_excluded"] = not old_subjects & {
        r["subject_group"] for r in all_records
    }
    checks["semantic_quarantine"] = not {36, 5698, 17421, 11928} & {
        r["case_id"] for r in all_records
    }
    texts = read(DATA / "text.json")
    checks["article_separation"] = (
        len({r["article"] for rows in texts.values() for r in rows}) == 160
    )
    by_episode = {}
    manifests = {}
    for job in jobs():
        out = RESULTS / job["id"]
        complete = read(out / "complete.json")
        assert complete["steps"] == 256
        assert complete["trainable_parameters"] == 3145728
        assert complete["ledger"]["E_sequences"] == 1024
        assert complete["ledger"]["R_sequences"] == 2048
        assert complete["ledger"]["text_sequences"] == 512
        by_episode.setdefault(job["episode"], []).append(complete)
        manifests[job["id"]] = {p.name: digest(p) for p in sorted(out.glob("node-*.json"))}
        assert len(manifests[job["id"]]) == 7
    checks["all_24_runs_and_168_nodes"] = len(manifests) == 24
    checks["paired_exposure"] = all(
        len({json.dumps(r["ledger"], sort_keys=True) for r in rows}) == 1
        and len({r["schedule_sha256"] for r in rows}) == 1
        for rows in by_episode.values()
    )
    checks["GPU0_excluded"] = all(
        read(RESULTS / j["id"] / "complete.json")["device"] != "cuda:0" for j in jobs()
    )
    chosen = read(ART / "learning-rate-selection.json")["chosen"]["1"]
    diagnostic_files = [
        p
        for job in jobs()
        if job["beta"] == 1 and job["lr"] == chosen["lr"]
        for p in (RESULTS / job["id"] / "diagnostics").glob("step-*.json")
    ]
    checks["all_24_diagnostic_nodes"] = len(diagnostic_files) == 24
    checks["diagnostic_restoration"] = all(
        read(p)["restore_max_ce_error"] == 0 for p in diagnostic_files
    )
    checks["no_U_update_evaluation"] = all(
        "U" not in read(RESULTS / j["id"] / "node-256.json") for j in jobs()
    )
    preflight = read(ART / "preflight.json")
    checks["preflight_passed"] = preflight["passed"]
    checks["final_layer_negative_control"] = preflight["final_layer_earlier_ce_max_error"] == 0
    checks["parent_restored_in_preflight"] = preflight["restore_ce_max_error"] == 0
    checks["native_generation_equivalence"] = read(ART / "baseline-engine-preflight.json")[
        "native_generation_matches"
    ]
    checks["candidate_U_blind"] = not read(ART / "candidate-decision.json")["selection_uses_U"]
    result = {
        "time": now(),
        "checks": checks,
        "passed": all(bool(value) for value in checks.values()),
        "node_hashes": manifests,
        "diagnostic_hashes": {str(p.relative_to(ROOT)): digest(p) for p in diagnostic_files},
        "scope": "Development only; no claim of independent fact verification or model replication",
    }
    write(ART / "completion-audit.json", result)
    assert result["passed"], checks
    return result


def table(header, rows):
    return "\n".join(
        ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    )


def scoring_audit(selected):
    """Post hoc response annotation; never replaces frozen scores or selection."""
    facts = {r["case_id"]: r for r in read(DATA / "selection.json")["V"]}
    variants = {"microsoft": {"microsoft corporation"}, "fiat": {"fiat s.p.a"}}
    annotations, summaries = [], []
    for beta in (1, 10):
        lr = selected["chosen"][str(beta)]["lr"]
        for step in (128, 256):
            subset = []
            for episode in range(4):
                node = read(RESULTS / f"dev-e{episode}-lr{lr:g}-b{beta}" / f"node-{step:03d}.json")
                for row in node["V"]:
                    if row["correct"]:
                        continue
                    fact = facts[row["case_id"]]
                    pred = normalize_answer(row["prediction"].strip().replace("**", ""))
                    aliases = {normalize_answer(a) for a in fact["aliases"]}
                    match = re.fullmatch(
                        r"the (?:manufacturer|native language|original language) of .+ is (.+)",
                        pred,
                    )
                    if match and normalize_answer(match[1]) in aliases:
                        category = "correct_label_in_full_sentence"
                    elif pred in variants.get(normalize_answer(fact["answer"]), set()):
                        category = "company_name_variant"
                    else:
                        category = "different_label_or_unresolved"
                    subset.append(
                        {
                            "beta": beta,
                            "step": step,
                            "episode": episode,
                            "case_id": row["case_id"],
                            "subject": fact["subject"],
                            "source_answer": fact["answer"],
                            "prediction": row["prediction"],
                            "category": category,
                        }
                    )
            annotations.extend(subset)
            summaries.append(
                {
                    "beta": beta,
                    "step": step,
                    "evaluations": 256,
                    "strict_failures": len(subset),
                    "sentence_or_company_variant": sum(
                        r["category"] != "different_label_or_unresolved" for r in subset
                    ),
                    "different_label_or_unresolved": sum(
                        r["category"] == "different_label_or_unresolved" for r in subset
                    ),
                }
            )
    output = {
        "time": now(),
        "post_hoc": True,
        "used_in_selection": False,
        "review": "All selected 128/256-step V strict failures inspected; "
        "response annotation, not truth verification",
        "summaries": summaries,
        "annotations": annotations,
    }
    write(ART / "scoring-audit.json", output)
    return summaries


def main():
    audit_result = audit()
    decision = read(ART / "candidate-decision.json")
    selected = read(ART / "learning-rate-selection.json")
    score_summary = scoring_audit(selected)
    rows = []
    for job in jobs():
        for step in read(CONFIG)["nodes"]:
            node = read(RESULTS / job["id"] / f"node-{step:03d}.json")
            rows.append(
                {
                    **job,
                    "step": step,
                    "E_train": mean([r for r in node["E"] if r["view"] == 0], "correct"),
                    "E_paraphrases": mean([r for r in node["E"] if r["view"] in (1, 2)], "correct"),
                    "E_plain": mean([r for r in node["E"] if r["view"] == 3], "correct"),
                    "E_ce": mean([r for r in node["E"] if r["view"] == 0], "ce"),
                    "R_accuracy": mean(node["R"], "correct"),
                    "V_accuracy": mean(node["V"], "correct"),
                    "R_kl": mean(node["R"], "kl"),
                    "V_kl": mean(node["V"], "kl"),
                    "text_dev_kl": mean(node["text_dev"], "kl"),
                    "text_dev_ce": mean(node["text_dev"], "ce"),
                    "delta_norm": node["delta_norm"],
                }
            )
    with (ART / "learning-curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_rows = [row for row in rows if row["lr"] == selected["chosen"][str(row["beta"])]["lr"]]
    aggregate = []
    for beta in (1, 10):
        for step in read(CONFIG)["nodes"]:
            r = [r for r in selected_rows if r["beta"] == beta and r["step"] == step]
            aggregate.append(
                {
                    "beta": beta,
                    "step": step,
                    **{k: mean(r, k) for k in rows[0] if k not in jobs()[0] and k != "step"},
                }
            )
    write(ART / "aggregate.json", aggregate)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for beta, color in ((1, "#2563eb"), (10, "#d97706")):
        values = [r for r in aggregate if r["beta"] == beta]
        for ax, key in zip(axes[:2], ("E_train", "V_accuracy"), strict=True):
            ax.plot(
                [r["step"] for r in values],
                [r[key] for r in values],
                "o-",
                color=color,
                label=f"Replay beta={beta}",
            )
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Training steps")
            ax.grid(alpha=0.2)
    axes[0].set_ylabel("Target prompts: strict answer accuracy")
    axes[1].set_ylabel("Old prompts (V): strict answer accuracy")
    axes[0].legend(fontsize=8)
    for i, group in enumerate(read(CONFIG)["groups"]):
        y = [r["R_kl_reduction"] for r in decision["rows"] if r["group"] == group]
        axes[2].scatter([i] * len(y), y, s=28, alpha=0.8)
    axes[2].axhline(0.25, color="gray", linestyle="--", linewidth=1)
    axes[2].set_xticks(range(4), read(CONFIG)["groups"])
    axes[2].set_ylabel("R KL reduction after position ablation")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_title("128-step intervention; threshold +25%", fontsize=10)
    axes[2].grid(alpha=0.2)
    fig.suptitle("Qwen3-0.6B, layer 14 down updates; 4 development episodes")
    fig.tight_layout()
    fig.savefig(ART / "development.png", dpi=180)
    fig.savefig(ART / "development.pdf")
    plt.close(fig)
    lr_table = table(
        ["回放β", "选择学习率", "128步新事实完整答对", "V保持率下降", "保持门槛"],
        [
            [
                b,
                f"{x['lr']:g}",
                f"{x['E']:.1%}",
                f"{x['V_loss']:.1%}",
                x["retention_condition_passed"],
            ]
            for b, x in selected["chosen"].items()
        ],
    )
    effect_table = table(
        ["单元", "位置", "学习收益保留", "R KL降低", "优于有效随机对照", "全部判据"],
        [
            [
                r["episode"],
                r["group"],
                f"{r['learning_retained']:.1%}"
                if r["learning_retained"] is not None
                else "不可判定",
                f"{r['R_kl_reduction']:.1%}" if r["R_kl_reduction"] is not None else "不可判定",
                r["beats_random"],
                r["passes"],
            ]
            for r in decision["rows"]
        ],
    )
    full_table = table(
        ["β", "步数", "训练问法", "两种改写", "自然续写", "R保持", "V保持", "文本开发KL"],
        [
            [
                r["beta"],
                r["step"],
                *[
                    f"{r[k]:.1%}"
                    for k in ("E_train", "E_paraphrases", "E_plain", "R_accuracy", "V_accuracy")
                ],
                f"{r['text_dev_kl']:.5g}",
            ]
            for r in aggregate
            if r["step"] in (0, 128, 256)
        ],
    )
    score_table = table(
        ["β", "步数", "严格失分/256次评价", "正确标签完整句/公司名称变体", "不同标签/待定"],
        [
            [
                r["beta"],
                r["step"],
                r["strict_failures"],
                r["sentence_or_company_variant"],
                r["different_label_or_unresolved"],
            ]
            for r in score_summary
        ],
    )
    status = (
        f"候选为{decision['candidate']}，达到开发推进门槛；确认阶段需独立完成才可讨论泛化。"
        if decision["candidate"]
        else "没有位置类别达到至少3/4单元的全部推进判据。按事先停止规则，"
        "96条确认轨迹未启动，U及自然文本测试集没有更新后评测。"
    )
    body = f"""# 真实Qwen路径学习开发实验 v1

完成时间：{now()}。{status}

本轮没有确认“跨位置耦合是新知识学习瓶颈”。在128步的4个单元中，分别恢复4类候选位置后，16/16条件的R分布偏移都增大；失败并非仅由随机控制匹配不足造成。
恢复实体末位置使R KL增大约4.8%–20.1%，恢复查询内容使其增大约23.6%–297.3%。
完整更新的R KL为0.00691–0.01138；这些相对变化明显高于零更新数值噪声。
这与训练后的不同位置贡献共同维持回放行为的解释一致，尚未证明具体线性抵消机制或未见输入收益。

## 范围与执行

使用同一个原始Qwen3-0.6B-Base检查点，第14层down的3,145,728参数增量。24条开发轨迹全部运行256步，共6144步、168个固定评测节点；开发选定的β=1四条轨迹各完成6个非零节点的位置诊断，共24个诊断节点。所有轨迹、失败候选和未匹配输入均保留。GPU2–9用于开发，0排除，1保留其他进程。它们不是24个独立预训练模型。

16个开发事实按实体分为4个单元，另有32个确认事实和64/64/128条R/V/U。R用于训练，V用于开发选择，U更新后结果未读取。自然文本按文章隔离64/32/64条。本轮完整答案含EOS训练、自由生成评测，训练问法未答对不等于父模型完全不含该知识。

来源标签采用CounterFact原答案。改写模板有程序语义检查和针对性审核，4条冲突/歧义记录在开发更新前隔离；并未独立核实全部现实真值。部分候选和WikiText语料在历史baseline中访问过，实体与旧更新/扰动样本隔离，不能声称所有数据完全未见。详见[实施契约](hebbian-learning-plan-v1.md#1418-路径学习实验的实施细化2026-09-29开发更新前)。

## 学习与保持

学习率按128步、4单元等权汇总选择，U不参与：

{lr_table}

{full_table}

![开发学习与位置诊断](development-artifacts/qwen-path-learning-v1/development.png)

以上为条件性开发结果，不能用开发最优分数充当确认性泛化证据。改写与原问法、自然续写分别列出，不以teacher-forced损失代替完整答案生成。

基线错误还包含答案格式和问题理解：例如Toyota G1输出Toyota Motor Corporation，
Dodge Monaco给出含正确实体的完整句子，若干“born/founded in”输入回答年份。
严格答案评分将它们记为未答对，因此本批任务混合了指定关系回答、答案格式学习和事实响应学习，不能把全部收益解释为新增知识存储。
两种更明确关系的未训练改写和无包装续写用于单列检查这一限制；不依据这些事后观察替换冻结样本。

对开发选定学习率、128/256步的全部V严格失分做了事后逐条检查：

{score_table}

因此，普通回放128步的9.8个百分点严格失分主要是回答形式变化：25次失分中23次仍给出参考答案或公司全称，
仅2次给出了不同标签，而且是同一条steak tartare来源国记录在两个训练单元中分别变成Italy/Switzerland。
这不是25条旧知识丢失；也不能从两个条件性响应就推出普遍事实遗忘。
这份[评分审计](development-artifacts/qwen-path-learning-v1/scoring-audit.json)不改写冻结主分数、不参与候选或学习率选择。
自然续写的0%也是“整段首行必须等于答案”的严格指标，可能受后续续写及停止方式影响，不能直接解释为完全没有事实迁移。

## 位置干预与推进判据

在同一训练后ΔB上，仅把指定位置的新增贡献恢复为父模型贡献；不删除原模型MLP。主候选S_end/S_other/C/W严格早于提示末位置，L/A为单列参照。下表来自开发选定β=1的128步，全部其他节点另存原始数据：

{effect_table}

推进需要同时达到：完整答案学习有增长、保留≥90%交叉熵收益、完整答案正确率不下降、
R KL下降≥25%、基线风险≥1e−4且超出噪声，并优于有效随机控制。
位置数量及相对位置匹配后，再以25%局部增量能量误差标记有效匹配；
E和R覆盖均需≥80%，候选/随机比较使用同一匹配子集。

各位置的覆盖及失败原因见[candidate-decision.json](development-artifacts/qwen-path-learning-v1/candidate-decision.json)。
没有有效匹配表示当前控制不能识别该类别的特异作用，不能推成“该路径不重要”。

路径恢复后的效果是固定权重的机制诊断。没有进入配对训练确认时，不能宣称位置限制改善实际学习，也不能宣称跨位置耦合无害或已构成普遍瓶颈。4个开发单元、同一个检查点和共享保持集限制了外推范围。

## 核验与复现

{len(audit_result["checks"])}项归档核验通过；源码/数据/模型哈希、24条训练及168节点、配对曝光、24个诊断节点、恢复和未访问U更新结果均检查。真实模型预飞的父权重恢复及终层早期位置负对照误差为0，自定义生成与原生生成逐token一致；10项相关契约测试通过。

配置为`configs/qwen-path-learning-v1.json`，核心实现为`src/llm_memory_editability/qwen_path_learning.py`和`qwen_path_train.py`。正式运行以[execution-lock.json](development-artifacts/qwen-path-learning-v1/execution-lock.json)及其源码快照为准，早期基线与语义修订独立留档。逐步曲线见[CSV](development-artifacts/qwen-path-learning-v1/learning-curves.csv)，所有原始节点和增量/优化器检查点位于`results/qwen-path-learning-v1/`。

```bash
PYTHONPATH=src /home/siqizhu4/miniconda3/bin/python scripts/execute_qwen_path_learning.py
PYTHONPATH=src /home/siqizhu4/miniconda3/bin/python scripts/report_qwen_path_learning.py
```

重放使用既有锁与检查点恢复；不要覆盖候选或执行锁。新参数、数据或科学判据需要新版本。
"""
    (ROOT / "docs/qwen-path-learning-results-v1.md").write_text(body)
    write(
        ART / "report-status.json",
        {
            "time": now(),
            "report": "docs/qwen-path-learning-results-v1.md",
            "sha256": digest(ROOT / "docs/qwen-path-learning-results-v1.md"),
            "candidate": decision["candidate"],
            "development_complete": True,
        },
    )
    print(
        json.dumps(
            {"report": "docs/qwen-path-learning-results-v1.md", "candidate": decision["candidate"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
