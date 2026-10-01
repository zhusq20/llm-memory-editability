"""Describe every fixed P2 endpoint after the complete formal audit passes."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="results/bios-mechanism-dev-v1/p2")
root = Path(parser.parse_args().source).resolve()
audit = json.loads((root / "audit.json").read_text())
assert audit["complete"] and audit["passed"] and not audit["errors"]


def read(name):
    out = []
    for row in csv.DictReader((root / name).open()):
        parsed = {}
        for key, value in row.items():
            if value == "":
                parsed[key] = None
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def subset(rows, **fields):
    return [r for r in rows if all(r[k] == v for k, v in fields.items())]


def avg(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return mean(values) if values else None


def percent(value):
    return "NA" if value is None else f"{value * 100:.3f}"


def change(before, after):
    return f"{percent(before)} → {percent(after)}"


def table(keys, rows):
    return ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"] + [
        "| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows
    ]


learning = read("learning.csv")
editing = read("editing.csv")
paired = read("paired-learning.csv")
editpaired = read("paired-editing.csv")
heterogeneity = []
for width in (128, 256):
    for world in (0, 1):
        for seed in (0, 1):
            a = subset(paired, branch="common", width=width, world=world, seed=seed, step=15360)[0]
            b = subset(paired, branch="common", width=width, world=world, seed=seed, step=30720)[0]
            heterogeneity.append(
                dict(
                    width=width,
                    world=world,
                    seed=seed,
                    matching_before=a["matching_effect"],
                    matching_after=b["matching_effect"],
                    cross_relation_before=a["mismatched_vs_neither"],
                    cross_relation_after=b["mismatched_vs_neither"],
                    company_mean_before=a["company_mean_vs_neither"],
                    company_mean_after=b["company_mean_vs_neither"],
                    project_mean_before=a["project_mean_vs_neither"],
                    project_mean_after=b["project_mean_vs_neither"],
                )
            )

learning_summary = []
for branch in ("common", "sensitivity"):
    for width in (128, 256):
        for condition in ("company", "project", "neither"):
            for step in (15360, 30720):
                rows = subset(learning, branch=branch, width=width, condition=condition, step=step)
                learning_summary.append(
                    dict(
                        branch=branch,
                        width=width,
                        condition=condition,
                        step=step,
                        n=len(rows),
                        base=avg(rows, "base_accuracy"),
                        heldout=avg(rows, "mean_heldout"),
                        exception=mean(
                            [
                                r[f"{c}_heldout_old_exception"]
                                for r in rows
                                for c in ("company", "project")
                            ]
                        ),
                        ordinary=mean(
                            [
                                r[f"{c}_heldout_nonexception"]
                                for r in rows
                                for c in ("company", "project")
                            ]
                        ),
                        independent=avg(rows, "independent_attributes_accuracy"),
                    )
                )

editing_summary = []
edit_blocks = []
for branch in ("common", "sensitivity"):
    for width in (128, 256):
        for kind in ("coherent", "exception"):
            for phase in ("parent", "continued"):
                rows = subset(editing, branch=branch, width=width, kind=kind, phase=phase, step=512)
                metrics = (
                    "E",
                    "D_heldout",
                    "D_conflict_heldout",
                    "U_full_damage",
                    "U_heldout_damage",
                    "U_full_known",
                    "U_full_broken",
                    "U_heldout_known",
                    "U_heldout_broken",
                )
                item = dict(
                    branch=branch,
                    width=width,
                    kind=kind,
                    phase=phase,
                    n=len(rows),
                    **{k: avg(rows, k) for k in metrics},
                )
                for stratum in range(4):
                    item[f"U_full_stratum{stratum}_damage"] = avg(rows, f"U_full_{stratum}_rate")
                editing_summary.append(item)
                for world in (0, 1):
                    for seed in (0, 1) if branch == "common" else (0,):
                        selected = subset(rows, world=world, seed=seed)
                        edit_blocks.append(
                            dict(
                                branch=branch,
                                width=width,
                                kind=kind,
                                phase=phase,
                                world=world,
                                seed=seed,
                                n=len(selected),
                                **{k: avg(selected, k) for k in metrics},
                            )
                        )

rate = []
for width in (128, 256):
    for world in (0, 1):
        a = subset(paired, branch="common", width=width, world=world, seed=0, step=30720)[0]
        b = subset(paired, branch="sensitivity", width=width, world=world, seed=0, step=30720)[0]
        la = subset(learning, branch="common", width=width, world=world, seed=0, step=30720)
        lb = subset(learning, branch="sensitivity", width=width, world=world, seed=0, step=30720)
        ea = subset(
            editing,
            branch="common",
            width=width,
            world=world,
            seed=0,
            phase="continued",
            kind="exception",
            step=512,
        )
        eb = subset(
            editing,
            branch="sensitivity",
            width=width,
            world=world,
            seed=0,
            phase="continued",
            kind="exception",
            step=512,
        )
        rate.append(
            dict(
                width=width,
                world=world,
                seed=0,
                delta_matching=b["matching_effect"] - a["matching_effect"],
                delta_cross_relation=b["mismatched_vs_neither"] - a["mismatched_vs_neither"],
                delta_base=avg(lb, "base_accuracy") - avg(la, "base_accuracy"),
                delta_heldout=avg(lb, "mean_heldout") - avg(la, "mean_heldout"),
                delta_exception_QA=mean(
                    r[f"{c}_heldout_old_exception"] for r in lb for c in ("company", "project")
                )
                - mean(r[f"{c}_heldout_old_exception"] for r in la for c in ("company", "project")),
                delta_exception_edit_E=avg(eb, "E") - avg(ea, "E"),
                delta_exception_edit_Dconflict=avg(eb, "D_conflict_heldout")
                - avg(ea, "D_conflict_heldout"),
                delta_Ufull=avg(eb, "U_full_damage") - avg(ea, "U_full_damage"),
            )
        )

result = dict(
    audit=audit,
    learning_block_heterogeneity=heterogeneity,
    learning_summary=learning_summary,
    editing_summary=editing_summary,
    editing_block_heterogeneity=edit_blocks,
    learning_rate_paired_blocks=rate,
    limitations=[
        "Two reused worlds; initializations do not create independent worlds",
        "Every planned fixed30720 endpoint is retained; no good-model subset",
        "Sensitivity comparisons use only matching seed0 parents",
        "Coherent D_conflict is only a paired-person reference; "
        "conflict is an exception-update label",
        "Damage averages are casewise rates with explicit old-correct denominators; "
        "training data and time differ across stages",
    ],
)
(root / "paired-interpretation.json").write_text(json.dumps(result, indent=2) + "\n")
lines = [
    "# P2 全部固定终点的配对描述",
    "",
    "先报告世界/初始化差异，再作条件平均。主分析纳入全部36个30720步终点，不按知识水平筛选。数值为百分比或百分点；这些是两个开发世界的描述性结果。",
    "",
    "## 共同续训：逐世界/初始化的组织效应",
    "",
]
rows = []
for r in heterogeneity:
    rows.append(
        dict(
            width=r["width"],
            world=r["world"],
            seed=r["seed"],
            matching=change(r["matching_before"], r["matching_after"]),
            cross_relation=change(r["cross_relation_before"], r["cross_relation_after"]),
        )
    )
lines += table(["width", "world", "seed", "matching", "cross_relation"], rows)
lines += [
    "",
    "matching是组织×查询关系的交叉效应；cross_relation是不匹配组织相对于neither的任务增益。箭头为15360→30720步。",
    "",
    "## 共同续训：基础知识与普通/旧例外 QA",
    "",
]
rows = []
for width in (128, 256):
    for condition in ("company", "project", "neither"):
        a = subset(learning_summary, branch="common", width=width, condition=condition, step=15360)[
            0
        ]
        b = subset(learning_summary, branch="common", width=width, condition=condition, step=30720)[
            0
        ]
        rows.append(
            dict(
                width=width,
                condition=condition,
                base=change(a["base"], b["base"]),
                heldout=change(a["heldout"], b["heldout"]),
                old_exception=change(a["exception"], b["exception"]),
                ordinary=change(a["ordinary"], b["ordinary"]),
            )
        )
lines += table(["width", "condition", "base", "heldout", "old_exception", "ordinary"], rows)
lines += ["", "## 固定512步编辑：全部 common 案例", ""]
rows = []
for r in editing_summary:
    if r["branch"] != "common":
        continue
    rows.append(
        dict(
            width=r["width"],
            kind=r["kind"],
            phase=r["phase"],
            cases=r["n"],
            E=percent(r["E"]),
            Dheldout=percent(r["D_heldout"]),
            paired_D=percent(r["D_conflict_heldout"]),
            U_damage=percent(r["U_full_damage"]),
            U_known=f"{r['U_full_known']:.1f}",
            U_broken=f"{r['U_full_broken']:.2f}",
        )
    )
lines += table(
    [
        "width",
        "kind",
        "phase",
        "cases",
        "E",
        "Dheldout",
        "paired_D",
        "U_damage",
        "U_known",
        "U_broken",
    ],
    rows,
)
lines += [
    "",
    "paired_D在exception中为冲突人群；在coherent中仅指相同人员的配对参照。U_known/U_broken为每案例平均整数计数的均值，不能用其比值替换案例等权损伤率。逐世界/初始化编辑结果与各局部U分层见JSON。",
    "",
    "## 学习率：仅相同父节点、初始化0的3e-4减1e-4",
    "",
]
rows = [{k: (percent(v) if k.startswith("delta_") else v) for k, v in r.items()} for r in rate]
lines += table(
    [
        "width",
        "world",
        "delta_base",
        "delta_heldout",
        "delta_exception_QA",
        "delta_matching",
        "delta_cross_relation",
        "delta_exception_edit_Dconflict",
    ],
    rows,
)
lines += [
    "",
    "学习率分支只有每世界一个初始化，不与common全体均值直接比较。充分训练后的表现变化不能单独证明内部路径；仍需与独立的行为及因果干预证据联合解释。",
    "",
]
(root / "paired-interpretation.md").write_text("\n".join(lines))
print(json.dumps(result, indent=2))
