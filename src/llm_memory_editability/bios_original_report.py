"""Read-only, CPU aggregation of the completed original bioS development run."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from llm_memory_editability.bios_original_data import (
    ATTRS,
    MONTHS,
    TASKS,
    answer,
    question,
)

DATE_PATTERN = re.compile(
    "(?:" + "|".join(m.lower() for m in MONTHS) + r") (?:[1-9]|1\d|2[0-8]), \d{4}"
)
DIMENSIONS = ("world", "initialization", "condition", "stage", "split", "task", "view", "cot")
COUNTS = (
    "n",
    "correct",
    "eos",
    "format_recognized",
    "correct_eos",
    "correct_unrecognized",
    "wrong_recognized",
    "wrong_unrecognized",
    "terminal_answer_match",
    "target_text_present",
    "city_swap",
    "comparison_outside_pair",
    "sum_full_logp",
    "sum_mean_logp",
    "sum_answer_probability",
)


def normalize(text):
    """Independently reproduce the historical frozen evaluator's normalization."""
    return " ".join(text.strip().rstrip(".").casefold().split())


def extract(row):
    text = row["generated"]
    return text.rsplit("Answer:", 1)[-1].strip() if row["cot"] else text


def recognized(text, task, vocabulary):
    value = normalize(text)
    if task == "parity":
        return value in ("yes", "no")
    if task == "year":
        return bool(re.fullmatch(r"\d{4}", value))
    if task == "date":
        return bool(DATE_PATTERN.fullmatch(value))
    if task in ("company_city", "city_company"):
        fields = ("company", "workcity") if task == "company_city" else ("workcity", "company")
        parts = value.split(";")
        return len(parts) == 2 and all(
            part.strip() in vocabulary[field] for part, field in zip(parts, fields, strict=True)
        )
    return value in vocabulary["name" if task == "comparison" else task]


def review_flags(row, person, vocabulary, partner=None):
    """Mechanical error labels only; these do not replace the frozen score."""
    text, target = extract(row), normalize(row["answer"])
    correct = normalize(text) == target
    form = recognized(text, row["task"], vocabulary)
    suffix = row["generated"].rsplit("Answer:", 1)[-1]
    terminal = not correct and "Answer:" in row["generated"] and normalize(suffix) == target
    # A literal occurrence is an inspection cue, not a semantic-correctness claim.
    present = not correct and bool(
        re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", normalize(row["generated"]))
    )
    other = {"birthcity": "workcity", "workcity": "birthcity"}.get(row["task"])
    swap = bool(not correct and other and normalize(text) == normalize(person[other]))
    outside = bool(
        row["task"] == "comparison"
        and partner is not None
        and normalize(text) not in (normalize(person["name"]), normalize(partner["name"]))
    )
    return dict(
        correct=correct,
        format_recognized=form,
        terminal_answer_match=terminal,
        target_text_present=present,
        city_swap=swap,
        comparison_outside_pair=outside,
        wrong_recognized=not correct and form,
        wrong_unrecognized=not correct and not form,
        correct_unrecognized=correct and not form,
    )


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalized(row):
    result = dict(row)
    for key in COUNTS:
        result.setdefault(key, 0)
    n = result["n"]
    for key in COUNTS[1:]:
        result[key.removeprefix("sum_") + "_rate"] = result[key] / n if n else None
    return result


def rollup(cells, views=False):
    """Pool repeated prompts descriptively, without treating them as independent worlds."""
    totals = {}
    for row in cells:
        key = tuple(row[k] for k in DIMENSIONS if k != "view")
        group = "canonical" if row["view"] == 0 else "heldout"
        if views:
            key = (*key, group)
        if key not in totals:
            totals[key] = {k: row[k] for k in DIMENSIONS if k != "view"}
            totals[key]["view_group"] = group if views else "all"
            totals[key].update(dict.fromkeys(COUNTS, 0))
        for count in COUNTS:
            totals[key][count] += row[count]
    return [finalized(v) for v in totals.values()]


def overall_rows(groups):
    totals = {}
    for row in groups:
        # Six-attribute extraction and eleven-task averages remain separately labelled.
        for family in ["all_tasks"] + (["six_attributes"] if row["task"] in ATTRS else []):
            dims = ("world", "initialization", "condition", "stage", "split", "cot", "view_group")
            key = (*[row[k] for k in dims], family)
            if key not in totals:
                totals[key] = {k: row[k] for k in dims}
                totals[key]["family"] = family
                totals[key].update(dict.fromkeys(COUNTS, 0))
            for count in COUNTS:
                totals[key][count] += row[count]
    return [finalized(v) for v in totals.values()]


def record_index(person_id, task, view, cot):
    return ((person_id * len(TASKS) + TASKS.index(task)) * 6 + view) * 2 + int(cot)


def check_identity(seen, person_id, task, view, cot):
    index = record_index(person_id, task, view, cot)
    if seen[index] != 255:
        raise ValueError(f"Duplicate prediction: {(person_id, task, view, cot)}")
    return index


def conditional_rows(scores, people, meta, modes):
    rows = []
    for split in ("dev", "test"):
        population = [p for p in people if p["split"] == split]
        for view in range(6):
            for cot in modes:
                for task in ("year", "parity", "company_city", "city_company", "comparison"):
                    n = qualified = correct = full_correct = 0
                    for p in population:
                        required = [(p["id"], "date")]
                        if task in ("company_city", "city_company"):
                            required = [(p["id"], "company"), (p["id"], "workcity")]
                        if task == "comparison":
                            required.append((p["partner"], "date"))
                        score = scores[record_index(p["id"], task, view, cot)]
                        if score == 255:
                            continue
                        prerequisites = [scores[record_index(i, t, view, cot)] for i, t in required]
                        if 255 in prerequisites:
                            raise ValueError("Missing foundation extraction prediction")
                        n += 1
                        full_correct += score
                        if all(prerequisites):
                            qualified += 1
                            correct += score
                    rows.append(
                        dict(
                            **meta,
                            split=split,
                            task=task,
                            view=view,
                            cot=cot,
                            n=n,
                            full_correct=full_correct,
                            qualified=qualified,
                            correct=correct,
                            coverage=qualified / n if n else None,
                            conditional_accuracy=correct / qualified if qualified else None,
                            full_accuracy=full_correct / n if n else None,
                        )
                    )
    return rows


def read_endpoint(root, meta, cfg, people):
    path = root / "evaluation/predictions.jsonl"
    completion_path = root / "evaluation/evaluation-complete.json"
    completion = json.loads(completion_path.read_text())
    vocabulary = {field: {normalize(p[field]) for p in people} for field in ("name", *ATTRS[1:])}
    tasks = ATTRS if meta["stage"] == "adapt" else TASKS
    modes = (False, True) if meta["condition"] == "MP-CoT" else (False,)
    seen = bytearray([255]) * (len(people) * len(TASKS) * 6 * 2)
    cells, examples, sample_counts = {}, [], Counter()
    digest = hashlib.sha256()
    checks = Counter()
    for line_number, raw in enumerate(path.open("rb"), 1):
        digest.update(raw)
        row = json.loads(raw)
        p = people[row["person_id"]]
        other = people[p["partner"]]
        if (
            row["split"] != p["split"]
            or row["split"] not in cfg["evaluation"]["splits"]
            or row["task"] not in tasks
            or row["view"] not in cfg["evaluation"]["views"]
            or row["cot"] not in modes
        ):
            raise ValueError(f"Unexpected cell: {path}:{line_number}")
        if row["answer"] != answer(p, row["task"], other):
            raise ValueError(f"Ground truth mismatch: {path}:{line_number}")
        if row["prompt"] != question(p, row["task"], row["view"], other, row["cot"]):
            raise ValueError(f"Prompt mismatch: {path}:{line_number}")
        if row["partner"] != (other["id"] if row["task"] == "comparison" else None):
            raise ValueError(f"Partner mismatch: {path}:{line_number}")
        flags = review_flags(row, p, vocabulary, other)
        if any(flags[k] != row[k] for k in ("correct", "format_recognized")):
            raise ValueError(f"Frozen score mismatch: {path}:{line_number}")
        index = check_identity(seen, p["id"], row["task"], row["view"], row["cot"])
        seen[index] = int(flags["correct"])
        checks["rows_truth_prompt_identity_score_verified"] += 1
        key = (row["split"], row["task"], row["view"], row["cot"])
        if key not in cells:
            cells[key] = dict(**meta, **dict(zip(DIMENSIONS[4:], key, strict=True)))
            cells[key].update(dict.fromkeys(COUNTS, 0))
        cell = cells[key]
        cell["n"] += 1
        for field, value in flags.items():
            cell[field] += int(value)
        cell["eos"] += int(row["eos"])
        cell["correct_eos"] += int(flags["correct"] and row["eos"])
        for source, target in (
            ("answer_full_logp", "sum_full_logp"),
            ("answer_mean_logp", "sum_mean_logp"),
        ):
            value = row[source]
            if not math.isfinite(value) or value > 1e-5:
                raise ValueError(f"Invalid log probability: {path}:{line_number}")
            cell[target] += value
        cell["sum_answer_probability"] += math.exp(row["answer_full_logp"])
        for category in (
            "terminal_answer_match",
            "target_text_present",
            "city_swap",
            "comparison_outside_pair",
            "wrong_recognized",
            "wrong_unrecognized",
            "correct_unrecognized",
        ):
            sample_key = (row["task"], row["view"] == 0, row["cot"], category)
            if flags[category] and sample_counts[sample_key] < 2:
                sample_counts[sample_key] += 1
                examples.append(
                    dict(
                        **meta,
                        category=category,
                        source=str(path),
                        line=line_number,
                        **row,
                        review_status="deterministic inspection sample; not human-labelled",
                    )
                )
    expected_cells = len(cfg["evaluation"]["splits"]) * len(tasks) * 6 * len(modes)
    if len(cells) != expected_cells or len(completion["totals"]) != expected_cells:
        raise ValueError(f"Incomplete evaluation cell matrix: {path}")
    counts = Counter(p["split"] for p in people)
    for (split, task, view, cot), cell in cells.items():
        if cell["n"] != counts[split]:
            raise ValueError(f"Incomplete population: {path}: {split}/{task}/{view}/{cot}")
        old = completion["totals"][f"{split}/{task}/view-{view}/cot-{cot}"]
        if any(cell[k] != old[k] for k in ("n", "correct", "format_recognized")):
            raise ValueError(f"Historical totals mismatch: {path}")
        if not math.isclose(cell["sum_full_logp"], old["sum_full_logp"], abs_tol=1e-8):
            raise ValueError(f"Historical logp total mismatch: {path}")
        checks["cells_population_and_historical_totals_verified"] += 1
    provenance = dict(
        **meta,
        path=str(path),
        sha256=digest.hexdigest(),
        bytes=path.stat().st_size,
        checkpoint_sha256=completion["checkpoint_sha256"],
        checks=dict(checks),
        completion_sha256=hashlib.sha256(completion_path.read_bytes()).hexdigest(),
    )
    conditional = conditional_rows(seen, people, meta, modes) if meta["stage"] == "task" else []
    return [finalized(v) for v in cells.values()], examples, provenance, conditional


def pretrain_rows(cfg):
    curves, inventory = [], []
    for world in cfg["world_seeds"]:
        for init in cfg["initialization_seeds"]:
            for condition in cfg["pretrain"]["conditions"]:
                root = Path(cfg["result_root"]) / f"world-{world}/init-{init}/{condition}/pretrain"
                path = root / "metrics.jsonl"
                metrics = [json.loads(line) for line in path.read_text().splitlines()]
                expected = list(range(1, cfg["pretrain"]["passes"] + 1))
                if [r["passes"] for r in metrics] != expected:
                    raise ValueError(f"Incomplete pretraining curve: {path}")
                for row in metrics:
                    curves.append(
                        dict(world=world, initialization=init, condition=condition, **row)
                    )
                inventory.append(
                    dict(
                        world=world,
                        initialization=init,
                        condition=condition,
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        passes=len(metrics),
                        final_nll=metrics[-1]["nll"],
                        nodes=[
                            dict(passes=n, checkpoint_exists=(root / f"pass-{n:03d}.pt").exists())
                            for n in cfg["pretrain"]["nodes"]
                        ],
                    )
                )
    return curves, inventory


def paired_differences(groups):
    pairs = [("S", "M"), ("M", "MP"), ("MP", "MP-R"), ("MP-R", "MP-Random"), ("MP", "MP-CoT")]
    lookup = {}
    for row in groups:
        key = tuple(
            row[k]
            for k in ("world", "initialization", "stage", "split", "task", "view_group", "cot")
        )
        lookup.setdefault(key, {})[row["condition"]] = row
    rows = []
    for key, conditions in lookup.items():
        for left, right in pairs:
            if left not in conditions or right not in conditions:
                continue
            a, b = conditions[left], conditions[right]
            if a["n"] != b["n"]:
                raise ValueError("Unequal paired populations")
            rows.append(
                dict(
                    **dict(
                        zip(
                            (
                                "world",
                                "initialization",
                                "stage",
                                "split",
                                "task",
                                "view_group",
                                "cot",
                            ),
                            key,
                            strict=True,
                        )
                    ),
                    left=left,
                    right=right,
                    n_each=a["n"],
                    left_accuracy=a["correct_rate"],
                    right_accuracy=b["correct_rate"],
                    difference_pp=100 * (b["correct_rate"] - a["correct_rate"]),
                )
            )
    return rows


def make_plots(output, curves, groups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for world in sorted({r["world"] for r in curves}):
        for c, color in zip(("S", "M", "MP"), ("#9b5f1b", "#4477aa", "#228833"), strict=True):
            subset = [r for r in curves if r["world"] == world and r["condition"] == c]
            axes[0].plot(
                [r["passes"] for r in subset],
                [r["nll"] for r in subset],
                color=color,
                alpha=0.8,
                label=f"{c}, world {world}",
            )
    axes[0].set(xlabel="Biography exposures per person", ylabel="Training NLL", yscale="log")
    axes[0].legend(fontsize=7)
    for offset, view, color in [(-0.17, "canonical", "#4477aa"), (0.17, "heldout", "#cc6677")]:
        means = []
        for c in ("S", "M", "MP"):
            data = [
                r["correct_rate"]
                for r in groups
                if r["stage"] == "adapt"
                and r["split"] == "test"
                and r["condition"] == c
                and r["view_group"] == view
                and r["family"] == "six_attributes"
            ]
            means.append(100 * sum(data) / len(data))
        axes[1].bar([i + offset for i in range(3)], means, width=0.32, label=view, color=color)
    axes[1].set(
        xticks=range(3),
        xticklabels=("S", "M", "MP"),
        ylabel="QA extraction accuracy (%)",
        ylim=(0, 105),
    )
    axes[1].legend()
    fig.suptitle("Original bioS development: training fit and later QA extraction")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"bios-pretrain-and-extraction.{suffix}", dpi=160)
    plt.close(fig)


def report_text(overall, groups, endpoints):
    def mean(rows, field="correct_rate"):
        # Exactly one initialization per world in this development configuration.
        per_world = {}
        for row in rows:
            per_world.setdefault(row["world"], []).append(row[field])
        values = [sum(v) / len(v) for v in per_world.values()]
        return 100 * sum(values) / len(values) if values else float("nan")

    def select(rows, **conditions):
        return [r for r in rows if all(r[k] == v for k, v in conditions.items())]

    lines = [
        "# 原版 bioS 开发预测复盘",
        "",
        "本报告独立复算两开发世界的 570 万条既有预测、18 个评价端点与全部 3240 个预训练"
        "日志节点。没有新增训练、GPU 前向或更改历史预测。原始评分与完成文件逐单元一致。",
        "",
        "表中百分比先在每个世界内汇总，再对两个世界等权；这是开发结果，无新增独立确认。"
        "同一人物的五个留出问法，以及同一比较人物对的两个问序，属于重复测量。",
        "",
        "## 六属性 QA 提取",
        "",
        "| BIO 条件 | 规范问法 | 五个留出问法均值 |",
        "| --- | ---: | ---: |",
    ]
    for condition in ("S", "M", "MP"):
        values = [
            mean(
                select(
                    overall,
                    stage="adapt",
                    split="test",
                    condition=condition,
                    family="six_attributes",
                    view_group=v,
                    cot=False,
                )
            )
            for v in ("canonical", "heldout")
        ]
        lines.append(f"| {condition} | {values[0]:.2f}% | {values[1]:.2f}% |")
    lines += [
        "",
        "S/M/MP 使用相同人物与语义事实曝光；M 增加表述，MP 再重排句序。"
        "QA 适配使用训练人物，测试人物只参加 BIO 预训练。这里测的是经共同 QA 适配后的提取，"
        "不能称为预训练终点未经适配的问答表现。工作城市已出现于传记，属于提取任务。",
        "",
        "## 任务适配后的规范问法",
        "",
        "| 模型 / 推理方式 | 日期 | 年份 | 月份奇偶 | 出生早晚比较 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition, cot in [
        ("MP", False),
        ("MP-R", False),
        ("MP-Random", False),
        ("MP-CoT", False),
        ("MP-CoT", True),
    ]:
        values = [
            mean(
                select(
                    groups,
                    stage="task",
                    split="test",
                    condition=condition,
                    view_group="canonical",
                    cot=cot,
                    task=task,
                )
            )
            for task in ("date", "year", "parity", "comparison")
        ]
        mode = "CoT" if cot else "直接"
        lines.append(f"| {condition} / {mode} | " + " | ".join(f"{v:.2f}%" for v in values) + " |")
    lines += [
        "",
        "年份和奇偶需要从已知属性执行操作，不能与六属性提取的平均值混为一项。"
        "MP-CoT 的两种推理方式使用同一适配端点，但解释提示、监督格式及输出计算不同。"
        "MP-R 的收益须与匹配的 MP-Random 同时检查。",
        "",
        "## 错误与计分审阅",
        "",
    ]
    for condition, cot in (("MP", False), ("MP-CoT", True)):
        value = mean(
            select(
                groups,
                stage="task",
                split="test",
                condition=condition,
                view_group="canonical",
                cot=cot,
                task="comparison",
            ),
            "comparison_outside_pair_rate",
        )
        lines.append(
            f"- {condition} / {'CoT' if cot else '直接'} 的比较输出中，"
            f"{value:.2f}% 未精确返回问题中任一人的名字；这不能只解释为日期运算错误。"
        )
    suffix = mean(
        select(
            groups,
            stage="task",
            split="test",
            condition="MP-CoT",
            view_group="heldout",
            cot=False,
            task="date",
        ),
        "terminal_answer_match_rate",
    )
    lines += [
        f"- MP-CoT 直接作答的留出日期问法中，{suffix:.2f}% 的全量输出虽未通过冻结评分，"
        "最后一个 `Answer:` 后恰为目标日期。这是可复核的格式失分，保留为事后辅助统计；"
        "没有改写主分数。",
        "- `wrong_recognized` 只表示输出满足原格式规则但不等于标签；"
        "`target_text_present` 只表示原输出出现目标文字，不认定语义正确。"
        "城市互换与问题外人名由已知人物表确定。",
        "- `error-review-samples.jsonl` 按任务、问法组、模式、错误标签保存最先两条，"
        "是确定性审阅样本，不是错误率的随机估计，也不是全量人工语义标注。",
        "- CoT 的历史 `answer_full_logp` 是解释提示后直接接短答案的概率，"
        "不是所生成推理链或最终答案的概率，不用于判断 CoT 是否掌握知识。",
        "- `conditional-task-scores.csv` 在同一人物、相同问法和输出模式下检查源属性是否答对；"
        "比较题要求两个人的日期都答对。它是条件子集分析，覆盖率与全量分数同时保留，"
        "不等同于把模型自产属性重新输入后的两次调用实验。",
        "",
        "## 预训练曲线与剩余工作",
        "",
        "`pretrain-curves.csv` 与图保留六条各 540 次曝光的训练 NLL。不同数据条件的文本"
        "可预测性不同，NLL 不能当作统一知识量刻度。已保存权重节点存在不等于已测出各节点的"
        "知识/推理能力；当前没有这样的语义学习曲线。",
        "",
        "本 CPU 报告没有完成独立权重重载、oracle 属性输入或自主提取后运算；"
        "100k 正式矩阵与日期编辑也未执行。它们的后续结果应写入独立产物。",
        "",
        "复现：`PYTHONPATH=src .venv/bin/python scripts/report_bios_original.py`。"
        "原预测、配置、数据及报告源码哈希见 `audit.json`。",
        "",
        f"已核验端点：{len(endpoints)}；完整单元表 `cells.csv`，按任务与问法组汇总"
        " `task-view-groups.csv`，世界内配对差 `paired-world-differences.csv`。",
    ]
    return "\n".join(lines) + "\n"


def run_report(config_path, output):
    cfg = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    cells, examples, endpoints, conditional = [], [], [], []
    for world in cfg["world_seeds"]:
        people_path = Path(cfg["data_root"]) / f"world-{world}/people.json"
        people = json.loads(people_path.read_text())
        if [p["id"] for p in people] != list(range(cfg["people"])):
            raise ValueError("People index contract changed")
        for init in cfg["initialization_seeds"]:
            for stage, conditions in (
                ("adapt", cfg["pretrain"]["conditions"]),
                ("task", cfg["task"]["conditions"]),
            ):
                for condition in conditions:
                    meta = dict(world=world, initialization=init, condition=condition, stage=stage)
                    root = (
                        Path(cfg["result_root"]) / f"world-{world}/init-{init}/{condition}/{stage}"
                    )
                    a, b, c, d = read_endpoint(root, meta, cfg, people)
                    cells.extend(a)
                    examples.extend(b)
                    endpoints.append(c)
                    conditional.extend(d)
                    print(
                        json.dumps(dict(**meta, rows=sum(r["n"] for r in a), status="verified")),
                        flush=True,
                    )
    groups = rollup(cells, views=True)
    overall = overall_rows(groups)
    curves, inventory = pretrain_rows(cfg)
    write_csv(output / "cells.csv", cells)
    write_csv(output / "task-view-groups.csv", groups)
    write_csv(output / "overall.csv", overall)
    write_csv(output / "paired-world-differences.csv", paired_differences(groups))
    write_csv(output / "conditional-task-scores.csv", conditional)
    write_csv(output / "pretrain-curves.csv", curves)
    with (output / "error-review-samples.jsonl").open("w") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    save_json(
        output / "audit.json",
        dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            scope="CPU predictions/logs audit; no checkpoint reload or GPU inference",
            config=str(config_path),
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            raw_prediction_rows=sum(r["n"] for r in cells),
            endpoint_count=len(endpoints),
            cells=len(cells),
            pretraining_runs=len(inventory),
            pretrain_nodes=len(curves),
            endpoints=endpoints,
            pretrain=inventory,
            failures=[],
            sources=[
                dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in (
                    Path(__file__),
                    Path("scripts/report_bios_original.py"),
                    Path("src/llm_memory_editability/bios_original_data.py"),
                    Path("src/llm_memory_editability/bios_original_train.py"),
                    *[
                        Path(cfg["data_root"]) / f"world-{w}/people.json"
                        for w in cfg["world_seeds"]
                    ],
                )
            ],
            missing_scientific_deliverables=[
                "Independent GPU checkpoint reload remains unverified by this CPU report",
                "Oracle-input and extract-then-operate inference are absent from these inputs",
                "Pretraining semantic trajectories absent; existing curve is training NLL only",
                "Sampled output inspection is separate from exhaustive human semantic adjudication",
            ],
        ),
    )
    save_json(
        output / "summary.json",
        dict(
            development_worlds=cfg["world_seeds"],
            independent_worlds=len(cfg["world_seeds"]),
            test_overall=[r for r in overall if r["split"] == "test"],
            probability_caveat=(
                "CoT logp scores the short gold answer immediately after the explanation prompt; "
                "it is not the probability of the generated CoT or its final answer."
            ),
            scoring=(
                "Historical answer EM; EOS is reported independently. Views and paired "
                "comparison orientations are repeated measurements, not independent worlds."
            ),
        ),
    )
    make_plots(output, curves, overall)
    (output / "report.md").write_text(report_text(overall, groups, endpoints))
    return overall
