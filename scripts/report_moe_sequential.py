"""Recount sparse/dense sequential comparisons and export standalone curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from report_loop_learning import COSTS, POOLS, difference, metric, read, transitions, write

ARMS = ("D4", "M4", "W4")
LABELS = {"D4": "Ordinary", "M4": "MoE top-2", "W4": "Wide dense"}


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(config):
    root = Path(config["results_root"])
    entries = [
        {**spec, "run_dir": str(root / "runs" / spec["name"]), "reused": False}
        for spec in config["specs"]
    ]
    entries += [{**entry, "reused": True} for entry in config["reused_runs"]]
    rows, curves, sources = [], [], {}
    for entry in entries:
        out = Path(entry["run_dir"])
        metadata, history = read(out / "run.json", {}), read(out / "learning.json", [])
        spec = metadata.get("spec", entry)
        arm = entry["architecture"]
        boundary = spec["stage_a_steps"]
        final = boundary + spec["stage_b_steps"]
        a = next((row for row in history if row["step"] == boundary), None)
        b = next((row for row in history if row["step"] == final), None)
        raw_a = read(out / f"predictions-{boundary:07d}.json")
        raw_b = read(out / f"predictions-{final:07d}.json")
        ledger = read(out / "architecture.json", {})
        audit, complete = read(out / "audit.json", {}), read(out / "complete.json", {})
        sampling = read(out / "sampling-plan.json", {})
        row = {
            "name": entry["name"],
            "architecture": arm,
            "initialization": entry["initialization"],
            "reused": entry["reused"],
            "run_dir": str(out),
            "complete": complete.get("state") == "complete"
            and audit.get("passed") is True
            and b is not None,
            "plan_sha256": sampling.get("plan_sha256"),
            "multiset_sha256": sampling.get("multiset_sha256"),
            "stage_a": {pool: metric(a, pool, raw_a) for pool in POOLS},
            "endpoint": {pool: metric(b, pool, raw_b) for pool in POOLS},
            "transitions": {
                pool: transitions(raw_a, raw_b, pool) for pool in ("AA", "BB", "BA", "AB")
            },
            "parameters": ledger.get("total_parameters", ledger.get("unique_parameters")),
            "nominal_parameters_selected_per_token": ledger.get(
                "nominal_parameters_selected_per_token", ledger.get("active_unique_parameters")
            ),
            "costs": {key: b.get(key) if b else None for key in COSTS},
            "stage_b_costs": {
                key: b[key] - a[key] if a and b and key in a and key in b else None
                for key in COSTS
                if key != "peak_gpu_memory_bytes"
            },
        }
        for pool in ("AA", "BA", "AB", "BB"):
            if b and pool + "_autonomous" in b["metrics"]:
                row["endpoint"][pool + "_autonomous"] = metric(b, pool + "_autonomous", raw_b)
        rows.append(row)
        for node in history:
            raw = read(out / f"predictions-{node['step']:07d}.json")
            curves.append(
                {
                    "name": entry["name"],
                    "architecture": arm,
                    "initialization": entry["initialization"],
                    "step": node["step"],
                    "stage": node["stage"],
                    "costs": {key: node.get(key) for key in COSTS},
                    "metrics": {pool: metric(node, pool, raw) for pool in POOLS},
                    "exposures": node.get("exposures", {}),
                }
            )
        for file in (
            "run.json",
            "learning.json",
            "audit.json",
            "sampling-plan.json",
            "architecture.json",
            f"predictions-{boundary:07d}.json",
            f"predictions-{final:07d}.json",
        ):
            if (out / file).exists():
                sources[str(out / file)] = hashlib.sha256((out / file).read_bytes()).hexdigest()
    grouped = defaultdict(dict)
    for row in rows:
        if row["architecture"] in grouped[row["initialization"]]:
            raise ValueError("Duplicate architecture within an initialization")
        grouped[row["initialization"]][row["architecture"]] = row
    pairs = []
    for seed, group in sorted(grouped.items()):
        if set(group) != set(ARMS):
            raise ValueError("The three-arm matrix has a missing condition")
        hashes = {r["plan_sha256"] for r in group.values() if r["plan_sha256"]}
        if len(hashes) > 1:
            raise ValueError("Paired architectures have different sample streams")
        for control in ("D4", "W4"):
            first, second = group["M4"], group[control]
            if not first["complete"] or not second["complete"]:
                continue
            differences = {}
            for pool in POOLS:
                differences[pool] = (
                    first["endpoint"][pool]["answer_accuracy"]
                    - second["endpoint"][pool]["answer_accuracy"]
                )
            differences["BB_gain"] = (
                first["transitions"]["BB"]["net_accuracy_change"]
                - second["transitions"]["BB"]["net_accuracy_change"]
            )
            differences["AA_retention"] = difference(
                first["transitions"]["AA"]["retention_of_stage_a_correct"],
                second["transitions"]["AA"]["retention_of_stage_a_correct"],
            )
            pairs.append(
                {"initialization": seed, "contrast": "M4-" + control, "differences": differences}
            )
    means = {}
    for arm in ARMS:
        group = [row for row in rows if row["architecture"] == arm and row["complete"]]
        means[arm] = {
            "completed": len(group),
            "parameters": mean([r["parameters"] for r in group]),
            "nominal_parameters_selected_per_token": mean(
                [r["nominal_parameters_selected_per_token"] for r in group]
            ),
            "stage_a": {
                pool: mean([r["stage_a"][pool]["answer_accuracy"] for r in group]) for pool in POOLS
            },
            "endpoint": {
                pool: mean([r["endpoint"][pool]["answer_accuracy"] for r in group])
                for pool in POOLS
            },
            "BB_gain": mean([r["transitions"]["BB"]["net_accuracy_change"] for r in group]),
            "AA_retention": mean(
                [r["transitions"]["AA"]["retention_of_stage_a_correct"] for r in group]
            ),
            "costs": {key: mean([r["costs"][key] for r in group]) for key in COSTS},
        }
    return {
        "complete": bool(rows) and all(r["complete"] for r in rows),
        "new_runs": len(config["specs"]),
        "reused_runs": len(config["reused_runs"]),
        "units": config["analysis"]["unit"],
        "runs": rows,
        "means": means,
        "paired_differences": pairs,
        "curves": curves,
        "source_sha256": sources,
        "limitations": [
            "One previously observed development split, two initializations.",
            "Ordinary time measurements were inherited, not concurrent throughput.",
            "FLOPs are leading-matmul estimates; dispatch and optimizer excluded.",
            "No no-B continuation; BB gain is a pipeline before/after change.",
            "Common recipe, no per-architecture optimality claim.",
        ],
    }


def percent(value):
    return "pending" if value is None else f"{100 * value:.2f}%"


def report(summary, out):
    out = Path(out)
    write(out / "summary.json", summary)
    lines = [
        "# 普通模型、MoE与宽模型的顺序知识学习",
        "",
        f"完成：{summary['complete']}；新增4条，复用2条普通训练。",
        "",
        "同一已观察2Wiki开发划分、两个初始化；三臂配对样本流相同。",
        "",
        "| 模型 | 总参数M | 名义选中参数M | AA：A末→B末 | BB：A末→B末 | "
        "BB净增pp | 原正确AA保持 | 训练FLOPs |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = summary["means"][arm]
        if not row["completed"]:
            continue
        lines.append(
            f"| {LABELS[arm]} | {row['parameters'] / 1e6:.3f} | "
            f"{row['nominal_parameters_selected_per_token'] / 1e6:.3f} | "
            f"{percent(row['stage_a']['AA'])}→{percent(row['endpoint']['AA'])} | "
            f"{percent(row['stage_a']['BB'])}→{percent(row['endpoint']['BB'])} | "
            f"{100 * row['BB_gain']:+.2f} | {percent(row['AA_retention'])} | "
            f"{row['costs']['estimated_matmul_training_flops']:.3e} |"
        )
    lines += [
        "",
        "各初始化分别报告，均值不增加独立世界数。"
        "完整原子、训练题、自主调用、逐题保持与分母见summary.json。",
        "",
        "## 逐初始化配对差",
        "",
    ]
    for pair in summary["paired_differences"]:
        delta = pair["differences"]
        retention = (
            "无父正确覆盖"
            if delta["AA_retention"] is None
            else f"{100 * delta['AA_retention']:+.2f}pp"
        )
        lines.append(
            f"- 初始化{pair['initialization']}，{pair['contrast']}："
            f"BB终点{100 * delta['BB']:+.2f}pp；BB净增{100 * delta['BB_gain']:+.2f}pp；"
            f"AA保持{retention}。"
        )
    lines += ["", "## 解释边界", ""] + ["- " + text for text in summary["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    with (out / "learning-curves.csv").open("w", newline="") as stream:
        fields = ["architecture", "initialization", "step", "stage", *COSTS, *POOLS]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for node in summary["curves"]:
            writer.writerow(
                {key: node.get(key) for key in fields[:4]}
                | node["costs"]
                | {pool: node["metrics"][pool]["answer_accuracy"] for pool in POOLS}
            )
    if not summary["complete"]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"D4": "#2868A7", "M4": "#D17A26", "W4": "#398364"}
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharey="row")
    pools = ("atomic_A", "atomic_B", "AA", "BB")
    coordinates = ("step", "supervised_tokens", "estimated_matmul_training_flops")
    for i, pool in enumerate(pools):
        for j, coordinate in enumerate(coordinates):
            ax = axes[i, j]
            for arm in ARMS:
                grouped = defaultdict(list)
                for node in summary["curves"]:
                    if node["architecture"] == arm:
                        grouped[node["step"]].append(node)
                x, y = [], []
                for step, nodes in sorted(grouped.items()):
                    x.append(
                        step
                        if coordinate == "step"
                        else mean([node["costs"][coordinate] for node in nodes])
                    )
                    y.append(
                        100 * mean([node["metrics"][pool]["answer_accuracy"] for node in nodes])
                    )
                ax.plot(x, y, "o-", color=colors[arm], markersize=3, label=LABELS[arm])
            if coordinate == "step":
                ax.axvline(8000, color="grey", linestyle=":", linewidth=1)
            ax.set_ylim(-2, 102)
            ax.grid(alpha=0.2)
            ax.set_xlabel(
                {
                    "step": "Optimizer updates",
                    "supervised_tokens": "Supervised tokens",
                    "estimated_matmul_training_flops": "Estimated training matmul FLOPs",
                }[coordinate]
            )
            if j == 0:
                ax.set_ylabel(pool + " accuracy (%)")
            if i == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Sequential knowledge learning: A 8k + B 4k; two initializations, one split")
    fig.tight_layout()
    fig.savefig(out / "learning-curves.png", dpi=180)
    fig.savefig(out / "learning-curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    summary = summarize(config)
    report(summary, Path(config["results_root"]) / "report")
    print(
        json.dumps(
            {
                "complete": summary["complete"],
                "runs": len(summary["runs"]),
                "means": summary["means"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
