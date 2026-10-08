"""World-weighted reports of training in fixed and changed factual states."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["repository"]) / "results" / config["batch"]
    out = root / "report"
    out.mkdir(exist_ok=True)
    endpoints, edits, missing = [], [], []
    for spec in config["specs"] + config["editing_specs"]:
        directory = root / "runs" / spec["name"]
        path = directory / "complete.json"
        if not path.exists():
            missing.append(spec["name"])
            continue
        result = json.loads(path.read_text())
        audit_path = directory / "audit.json"
        assert audit_path.exists() and json.loads(audit_path.read_text())["passed"]
        if spec["job_type"] == "reader":
            endpoints.append(
                dict(
                    name=spec["name"],
                    world=spec["world"],
                    phase=spec["phase"],
                    memory=spec["memory_arm"],
                    training=spec["training_arm"],
                    metrics=result["endpoint"]["metrics"],
                    inner_success=float(np.mean([e["inner_success"] for e in result["episodes"]])),
                    training_seconds=result["training_seconds"],
                )
            )
        elif spec["job_type"] == "editing":
            for branch in result["branches"]:
                task = f"D_{branch['role']}_{branch['stratum']}"
                d = branch["metrics"][task]
                edits.append(
                    dict(
                        parent=spec["parent_name"],
                        world=spec["world"],
                        phase=spec["phase"],
                        memory=spec["memory_arm"],
                        training=spec["training_arm"],
                        role=branch["role"],
                        stratum=branch["stratum"],
                        arm=branch["arm"],
                        case=branch["case_id"],
                        E=branch["metrics"]["E_new"]["accuracy"],
                        U=branch["metrics"]["U_atomic"]["accuracy"],
                        D=d["accuracy"],
                        n=d["n"],
                        baseline_coverage=d["parent_correct_coverage"],
                        parent_correct_D=d["new_following_parent_correct"],
                        operation_coverage=d["operation_correct_coverage"],
                        operation_D=d["new_following_operation_correct"],
                        old_answer=d["old_answer_accuracy"],
                        U_composition=branch["metrics"]["U_familiar"]["accuracy"],
                    )
                )
    grouping = defaultdict(list)
    for row in endpoints:
        grouping[(row["phase"], row["memory"], row["training"])].append(row)
    means = []
    for (phase, memory, training), rows in grouping.items():
        means.append(
            dict(
                phase=phase,
                memory=memory,
                training=training,
                worlds=len(rows),
                familiar=float(np.mean([r["metrics"]["familiar_test"]["accuracy"] for r in rows])),
                strict=float(np.mean([r["metrics"]["strict_test"]["accuracy"] for r in rows])),
                atomic=float(np.mean([r["metrics"]["common_atomic"]["accuracy"] for r in rows])),
                inner_success=float(np.mean([r["inner_success"] for r in rows])),
            )
        )
    # Equal cases within each world, then equal worlds. Cases are not independent worlds.
    cells = defaultdict(list)
    for row in edits:
        cells[
            (
                row["phase"],
                row["memory"],
                row["training"],
                row["role"],
                row["stratum"],
                row["arm"],
                row["world"],
            )
        ].append(row)
    world_cells = []
    for key, rows in cells.items():
        phase, memory, training, role, stratum, arm, world = key
        values = {}
        for field in [
            "E",
            "U",
            "D",
            "baseline_coverage",
            "parent_correct_D",
            "operation_coverage",
            "operation_D",
            "old_answer",
            "U_composition",
        ]:
            valid = [r[field] for r in rows if r[field] is not None]
            values[field] = float(np.mean(valid)) if valid else None
        world_cells.append(
            dict(
                phase=phase,
                memory=memory,
                training=training,
                role=role,
                stratum=stratum,
                arm=arm,
                world=world,
                cases=len(rows),
                **values,
            )
        )
    groups = defaultdict(list)
    for r in world_cells:
        groups[
            tuple(r[k] for k in ["phase", "memory", "training", "role", "stratum", "arm"])
        ].append(r)
    edit_means = []
    for key, rows in groups.items():
        values = {}
        for field in [
            "E",
            "U",
            "D",
            "baseline_coverage",
            "parent_correct_D",
            "operation_coverage",
            "operation_D",
            "old_answer",
            "U_composition",
        ]:
            valid = [r[field] for r in rows if r[field] is not None]
            values[field] = float(np.mean(valid)) if valid else None
        edit_means.append(
            dict(
                zip(["phase", "memory", "training", "role", "stratum", "arm"], key, strict=True),
                worlds=len(rows),
                **values,
            )
        )
    report = dict(
        endpoints=endpoints,
        editing_cases=edits,
        endpoint_means=means,
        editing_worlds=world_cells,
        editing_means=edit_means,
        missing=missing,
        limitations=[
            "Synthetic template task; first-block MLP restored after each training episode; "
            "remaining parameters trained",
            "Not second-order meta-learning or unconstrained changing-world pretraining",
            "Atomic-change control uses unaffected composition queries; "
            "query exposures differ from changed-use treatment",
            "Static and changed-use share query prefixes, budgets and parent; "
            "changed labels add supervised counterfactual information",
            "Three fresh confirmation worlds, one initialization per world",
            "Finite-KL local MLP editing does not guarantee unchanged behavior",
            "Conditional D and coverage always reported alongside full-pool D",
        ],
    )
    write_json(out / "summary.json", report)
    lines = [
        "# 固定知识与知识变化训练：结果",
        "",
        "本批检验：在临时事实参数变化下学习组合使用，是否改善未见事实的原子编辑后传播。普通Loop与共享KV各比较fixed、changed_atomic、changed_use。",
        "",
        "开发世界与新世界分开；按世界等权，不把编辑案例当成独立世界。",
        "",
        "| 阶段 | 架构 | 训练 | 世界 | 原子 | 熟悉组合 | 严格组合 | 训练写入成功 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in means:
        lines.append(
            f"|{r['phase']}|{r['memory']}|{r['training']}|{r['worlds']}|{r['atomic']:.2%}|{r['familiar']:.2%}|{r['strict']:.2%}|{r['inner_success']:.2%}|"
        )
    lines += [
        "",
        "| 阶段 | 架构 | 训练 | 编辑角色 | 事实池 | 世界 | "
        "E更新 | U原子 | D传播 | 父模型正确覆盖 | 父正确D |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in edit_means:
        if r["arm"] != "edit":
            continue
        conditional = f"{r['parent_correct_D']:.2%}" if r["parent_correct_D"] is not None else "—"
        lines.append(
            f"|{r['phase']}|{r['memory']}|{r['training']}|{r['role']}|{r['stratum']}|{r['worlds']}|{r['E']:.2%}|{r['U']:.2%}|{r['D']:.2%}|{r['baseline_coverage']:.2%}|{conditional}|"
        )
    lines += [
        "",
        "全部原始案例、sham、必要事实条件和旧答案保留见summary.json。",
        "",
        "缺失项：" + str(missing),
        "",
        "解释边界：",
        "",
    ] + ["- " + x for x in report["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(dict(endpoints=len(endpoints), edit_branches=len(edits), missing=len(missing)))
    )


if __name__ == "__main__":
    main()
