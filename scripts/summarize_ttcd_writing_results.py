#!/usr/bin/env python3
"""Summarize audited one-step/merge results without recomputing scientific scores."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PERSISTENT = Path("/ossfs/workspace/llm-memory-editability")
REPO = Path(__file__).resolve().parents[1]


def percent(value):
    return "NA" if value is None else f"{100 * value:.2f}%"


def fraction(value):
    return f"{value['correct']}/{value['n']} ({percent(value['accuracy'])})"


def main():
    path = PERSISTENT / "ttcd-one-step-training-v1/control-report/result-aggregate.json"
    result = json.loads(path.read_text())
    if result["four_step_training_control"]["status"] != "complete":
        raise ValueError("All audited four-step controls must be present")
    sources = [path]
    for batch in ("ttcd-one-step-second-hop-v1", "ttcd-merge-hops-v1"):
        state = json.loads((PERSISTENT / batch / "controller-state.json").read_text())
        if state["state"] != "complete":
            raise ValueError("Finish original evaluation and merge pipelines before summarizing")
        for arm in ("transformer", "loop", "discoloop"):
            for stage in ("audit", "cloud-audit"):
                receipt = PERSISTENT / batch / arm / "evaluation" / f"result-{stage}.json"
                if json.loads(receipt.read_text()).get("passed") is not True:
                    raise ValueError("Every architecture requires independent and cloud audits")
                sources.append(receipt)
    rows = []
    for writer_steps in (1, 4):
        for arm in ("transformer", "loop", "discoloop"):
            summary = (
                result["all_results"][arm]
                if writer_steps == 1
                else result["four_step_training_control"]["architectures"][arm][
                    "all_control_results"
                ]
            )
            for method in ("single", "continuous", "independent_merge"):
                t = "1" if method == "single" else "32"
                second = summary["roles"]["second"][method][t]
                first = summary["roles"]["first"][method][t]
                rows.append(
                    {
                        "training_writer_steps": writer_steps,
                        "architecture": arm,
                        "method": method,
                        "written_facts": int(t),
                        "E": summary["E"][method][t],
                        "first_hop": first["all"],
                        "second_hop": second["all"],
                        "second_baseline_correct": second["baseline_correct"],
                        "second_premises": second["baseline_correct_and_both_current_atoms"],
                        "old_atoms_retained": summary["U"][method][t]["U_atoms"]["retained"],
                        "old_composition_retained": summary["U"][method][t]["U_chains"]["retained"],
                    }
                )
    output = REPO / "results/ttcd-evaluation-summary-20261016"
    output.mkdir(parents=True, exist_ok=True)
    evidence = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    data = {
        "rows": rows,
        "sources": evidence,
        "new_training_updates_during_recovery": 0,
        "independent_worlds": 1,
        "initializations": 1,
        "rotations_are_repeated_measures": True,
    }
    (output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        fields = [
            "training_writer_steps",
            "architecture",
            "method",
            "written_facts",
            "E",
            "first_hop",
            "second_hop",
            "second_baseline_correct",
            "second_premises",
            "old_atoms_retained",
            "old_composition_retained",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (fraction(row[k]) if isinstance(row[k], dict) else row[k]) for k in fields}
            )
    lines = [
        "# 一步写入与merge完整结果（2026-10-16恢复）",
        "",
        "恢复新增基础/外层训练0；原训练、检查点和原一步训练主报告均保留。四步训练终点的一步部署、两跳位merge、独立重放与完整云端JSON核验全部完成。",
        "",
        "本批32事实为原图已知键的replacement纠正，复用冻结面板。下表单事实为每条从A0独立编辑；连续与merge为写完32条后的结果。训练内写次数为1或4，部署全部每事实一步；merge只加从A0独立计算的增量。原子、两跳答案均按原答案＋EOS评分。",
        "",
        "|训练内写次数|架构|部署写法|已写事实可答|第一跳编辑后的组合|第二跳编辑后的组合|原正确第二跳组合更新成功|两原子当前均正确的第二跳子集|旧原子保持|旧组合保持|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["training_writer_steps"]),
                    row["architecture"],
                    row["method"],
                    fraction(row["E"]),
                    fraction(row["first_hop"]),
                    fraction(row["second_hop"]),
                    fraction(row["second_baseline_correct"]),
                    fraction(row["second_premises"]),
                    fraction(row["old_atoms_retained"]),
                    fraction(row["old_composition_retained"]),
                ]
            )
            + "|"
        )
    lines += [
        "",
        "第一跳54查询覆盖27/32事实，5事实保留零覆盖；第二跳253查询覆盖32事实。所有旋转/前缀是同32事实上的重复测量，不把1024或8096个状态/题对当作独立事实。条件子集的覆盖、第一跳完整成绩、同题同顺序配对差及三架构共同A0正确集合见原完整JSON。",
        "",
        "两原子当前均正确子集还要求组合在A0答对，分母随方法改变；0/0记为NA，不能解释为组合成功率0%。旧保持检查的是同64道旧原子与64道旧组合在32个状态下的重复读出，只在A0正确题上计算保持率。",
        "",
        "独立审计要求答案与EOS、写入状态精确一致，浮点读出沿用冻结代码的1e-5容差；云端完整JSON以哈希核对。",
        "",
        "模型基础能力、参数及快矩阵数量不同；单图单初始化，不以旧保持高但写入失败证明有效学习，也不以原子成功替代组合传播。GPU7仅本项目评测共享，耗时不作独占效率比较。",
        "",
        "## 证据",
        "",
    ]
    lines.extend(f"- `{p}`：`{sha}`" for p, sha in evidence.items())
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"report": str(output / "report.md"), "rows": len(rows), "new_training_updates": 0}
        )
    )


if __name__ == "__main__":
    main()
