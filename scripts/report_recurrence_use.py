"""World-balanced behavior and update-response summaries, without predictors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.recurrence_use import (
    TASKS,
    mean_metrics,
    prediction_metrics,
    query_means,
)
from llm_memory_editability.storage_composition import file_hash

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/development-artifacts/recurrence-use-v1"


def balanced(values):
    """Initialization means within world, followed by equal world weights."""
    grouped = defaultdict(list)
    for world, value in values:
        if value is not None:
            grouped[world].append(value)
    worlds = {str(w): float(np.mean(v)) for w, v in sorted(grouped.items())}
    return {
        "mean": float(np.mean(list(worlds.values()))) if worlds else None,
        "worlds": worlds,
        "n_worlds": len(worlds),
    }


def report(phase):
    config_path = ROOT / f"configs/recurrence-use-{phase}-v1.json"
    config = json.loads(config_path.read_text())
    folder = ROOT / "results/recurrence-use-v1" / phase
    aggregate, summaries, checks = defaultdict(list), [], []
    for entry in config["runs"]:
        run = folder / entry["name"]
        done, audited = (
            json.loads((run / "complete.json").read_text()),
            json.loads((run / "audit.json").read_text()),
        )
        for metadata in (done, audited):
            if not metadata["passed"] or metadata["config_sha256"] != file_hash(config_path):
                raise ValueError("Incomplete or mismatched reload audit")
        spec = entry["spec"]
        world, phi = spec["world_seed"], spec["phi"]
        for r in config["test_repeats"]:
            summary = json.loads((run / f"r{r}.json").read_text())
            with np.load(run / f"r{r}.npz") as raw:
                for task in TASKS:
                    for name in ["baseline"] + [
                        f"skip_{s}_{i}" for s in ("all", "r1") for i in range(r)
                    ]:
                        key = f"{task}/{name}"
                        pred = {
                            field: raw[f"{task}_{name}_{field}"]
                            for field in ("generated", "logits")
                        }
                        metrics = prediction_metrics(pred, raw[task + "_rows"][:, -1])
                        if mean_metrics(metrics) != summary["conditions"][key]["full_pool"]:
                            raise AssertionError("Saved predictions do not reproduce summary")
                        cell = summary["conditions"][key]
                        for pool in ("full_pool", "common_atomic_pool"):
                            if pool not in cell:
                                continue
                            for metric, value in cell[pool].items():
                                group = f"phi{phi:g}/R{r}/{key}/{pool}/{metric}"
                                aggregate[group].append((world, value))
                        if task != "atomic":
                            atomic = raw[f"atomic_{name}_complete"]
                            necessary = atomic[raw[task + "_atomic_indices"]].all(1).mean()
                            if necessary != cell["necessary_atomic_coverage"]:
                                raise AssertionError("Necessary-fact linking differs")
                            aggregate[f"phi{phi:g}/R{r}/{key}/necessary_atomic_coverage"].append(
                                (world, float(necessary))
                            )
                            if name != "baseline":
                                for metric in ("answer", "complete", "nll", "margin"):
                                    delta = (
                                        cell["full_pool"][metric]
                                        - summary["conditions"][f"{task}/baseline"]["full_pool"][
                                            metric
                                        ]
                                    )
                                    aggregate[f"phi{phi:g}/R{r}/{key}/delta/{metric}"].append(
                                        (world, delta)
                                    )
                                for response in cell.get("responses", []):
                                    group = (
                                        f"phi{phi:g}/R{r}/{key}/response/"
                                        f"{response['component']}/{response['site']}/j{response['target']}"
                                    )
                                    for metric in ("N", "B", "C"):
                                        aggregate[group + "/" + metric].append(
                                            (world, response[metric])
                                        )
                for label in ("experienced", "strict"):
                    recipients = raw[f"selection_{label}_recipients"]
                    for component in ("full", "mlp"):
                        key = f"same_{label}_{component}"
                        if key not in summary["conditions"]:
                            continue
                        pred = {field: raw[key + "_" + field] for field in ("generated", "logits")}
                        metrics = prediction_metrics(pred, raw["strict_2_rows"][recipients, -1])
                        means = {
                            k: query_means(v, recipients, len(raw["strict_2_rows"]))
                            for k, v in metrics.items()
                        }
                        for pool, mask in (
                            ("common_graph_pool", raw["selection_common"]),
                            ("common_known_pool", raw["same_common_known_mask"]),
                        ):
                            if mean_metrics(means, mask) != summary["conditions"][key][pool]:
                                raise AssertionError("Query-balanced donor scoring differs")
            for key, cell in summary["conditions"].items():
                if key.startswith(("same_", "changed_")):
                    pools = cell if key.startswith("same_") else {"full_pool": cell}
                    for pool, metrics in pools.items():
                        for metric, value in metrics.items():
                            aggregate[f"phi{phi:g}/R{r}/{key}/{pool}/{metric}"].append(
                                (world, value)
                            )
            for component in ("full", "mlp"):
                for pool in ("common_graph_pool", "common_known_pool"):
                    a = summary["conditions"].get("same_experienced_" + component, {}).get(pool, {})
                    b = summary["conditions"].get("same_strict_" + component, {}).get(pool, {})
                    if a.get("complete") is not None and b.get("complete") is not None:
                        aggregate[f"phi{phi:g}/R{r}/same_delta_{component}/{pool}/complete"].append(
                            (world, a["complete"] - b["complete"])
                        )
            summaries.append(
                {
                    "run": entry["name"],
                    "R": r,
                    "seconds": summary["seconds"],
                    "condition_count": len(summary["conditions"]),
                    "array_count": next(c["arrays"] for c in done["checks"] if c["R"] == r),
                }
            )
        checks.append(
            {
                "run": entry["name"],
                "complete_sha256": file_hash(run / "complete.json"),
                "audit_sha256": file_hash(run / "audit.json"),
                "run_seconds": done["seconds"],
                "audit_seconds": audited["seconds"],
            }
        )
    values = {key: balanced(items) for key, items in aggregate.items()}
    output = ARTIFACT / phase
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "finished_utc": utc(),
        "phase": phase,
        "training_updates": 0,
        "config_sha256": file_hash(config_path),
        "model_runs": len(checks),
        "model_budgets": len(summaries),
        "scientific_units": "Worlds; initializations nested",
        "analysis_status": config["status"],
        "balanced": values,
        "checks": checks,
        "budget_rows": summaries,
        "seconds": {
            "forward": sum(r["run_seconds"] for r in checks),
            "reload_audit": sum(r["audit_seconds"] for r in checks),
        },
    }
    write_json(output / "summary.json", result)

    def get(key):
        return values.get(key, {}).get("mean")

    lines = [
        "# 循环计算与首跳使用：固定权重结果",
        "",
        f"阶段：{phase}；新增训练0；{len(checks)}权重、{len(summaries)}权重×预算，全部独立重载。",
        "",
        "均值先在世界内平均初始化，再等权平均世界；开发与后续分开。后续使用已观察世界，不称全新确认。",
        "",
        "| 支持 | 测试R | 单跳 | 两跳熟悉 | 两跳严格 | 首次整块删除后熟悉 | 首次r1删除后熟悉 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phi in (1, 4):
        for r in (2, 4, 6):
            prefix = f"phi{phi}/R{r}/"
            keys = [
                "atomic/baseline/full_pool/complete",
                "familiar_2/baseline/full_pool/complete",
                "strict_2/baseline/full_pool/complete",
                "familiar_2/skip_all_0/full_pool/complete",
                "familiar_2/skip_r1_0/full_pool/complete",
            ]
            scores = [get(prefix + k) for k in keys]
            lines.append(
                f"| {phi} | {r} | "
                + " | ".join("—" if v is None else f"{100 * v:.2f}%" for v in scores)
                + " |"
            )
    lines += [
        "",
        "| 支持 | 测试R | 同桥共同覆盖 | 未干预 | 经历供体状态 | 严格供体状态 | 配对差 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phi in (1, 4):
        for r in (2, 4, 6):
            p = f"phi{phi}/R{r}/"
            keys = [
                "same_baseline/common_graph_pool/coverage",
                "same_baseline/common_graph_pool/complete",
                "same_experienced_full/common_graph_pool/complete",
                "same_strict_full/common_graph_pool/complete",
                "same_delta_full/common_graph_pool/complete",
            ]
            scores = [get(p + k) for k in keys]
            lines.append(
                f"| {phi} | {r} | "
                + " | ".join("—" if v is None else f"{100 * v:.2f}%" for v in scores)
                + " |"
            )
    lines += [
        "",
        "完整三/四跳、所有删除位置、原子共同子集覆盖、响应分子/分母、MLP和attention及逐世界差异见summary.json；原始生成和逐题响应见results/recurrence-use-v1。",
        "",
        "响应强不自动证明正确组合；共同原子正确子集为处理后筛选，不能替代全池成绩。同桥供体来自不同事实，完整状态还有查询身份。不同桥的新路径命中不等于原严格路径修复。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    plot(values, output)
    print(
        json.dumps(
            {
                "report": str(output),
                "runs": len(checks),
                "budgets": len(summaries),
                "all_verified": True,
            }
        ),
        flush=True,
    )


def plot(values, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for phi, color in ((1, "#377eb8"), (4, "#e41a1c")):

        def curve(tail, phi=phi):
            return [100 * values[f"phi{phi}/R{r}/{tail}"]["mean"] for r in (2, 4, 6)]

        axes[0].plot(
            [2, 4, 6],
            curve("familiar_2/baseline/full_pool/complete"),
            "o-",
            color=color,
            label=f"Support {phi}",
        )
        axes[0].plot(
            [2, 4, 6], curve("familiar_2/skip_r1_0/full_pool/complete"), "o--", color=color
        )
        axes[1].plot(
            [2, 4, 6],
            curve("atomic/baseline/full_pool/complete"),
            "o-",
            color=color,
            label=f"Support {phi}",
        )
        axes[1].plot([2, 4, 6], curve("atomic/skip_all_0/full_pool/complete"), "o--", color=color)
        for donor, style in (("experienced", "o-"), ("strict", "s--")):
            key = f"same_{donor}_full/common_graph_pool/complete"
            if all(
                values.get(f"phi{phi}/R{r}/{key}", {}).get("mean") is not None for r in (2, 4, 6)
            ):
                axes[2].plot(
                    [2, 4, 6], curve(key), style, color=color, label=f"S{phi} {donor} donor"
                )
    for ax, title in zip(
        axes,
        (
            "Familiar two-hop: solid normal / dashed r1 delete",
            "Atomic: solid normal / dashed block delete",
            "Strict recipients, same-bridge donors",
        ),
        strict=True,
    ):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Inference loops (trained at 4)")
        ax.set_ylabel("Complete generation (%)")
        ax.set_xticks([2, 4, 6])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "followup"), required=True)
    report(parser.parse_args().phase)
