"""Recount the fixed endpoints, verify pairing/exposure and report all worlds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import EpochStream, write_json
from llm_memory_editability.latent_scaling import build_world
from llm_memory_editability.representation_alignment import pack_training

TASKS = ("common_atomic", "train_composite", "familiar_test", "strict_test")


def recount(path, spec):
    with (
        np.load(path / "world.npz") as world,
        np.load(path / f"predictions-{spec['steps']:06d}.npz") as prediction,
    ):
        scores = {}
        for task in TASKS:
            generated = prediction[task + "_generated"]
            correct = (
                (generated[:, 0] == world[task][:, -1])
                & (generated[:, 1] == 5)
                & (generated[:, 2] == 1)
            )
            np.testing.assert_array_equal(correct, prediction[task + "_correct"])
            scores[task] = float(correct.mean())
        for task in ("familiar_test", "strict_test"):
            coverage = prediction[task + "_coverage"]
            scores[task + "_atomic_coverage"] = float(coverage.mean())
            scores[task + "_autonomous_two_calls"] = float(prediction[task + "_two_calls"].mean())
    return scores


def exposure_audit(path, spec):
    _, sizes = pack_training(build_world(spec))
    exposures = np.load(path / "exposures.npz")
    for j, size in enumerate(sizes):
        stream = EpochStream(size, spec["stream_seed"] + j)
        drawn = stream.take(spec["steps"] * spec["batch_size"] // 3)
        expected = np.bincount(drawn, minlength=size)
        np.testing.assert_array_equal(expected, exposures[f"stratum{j}"])


def aggregate(rows, keys):
    result, by_world = {}, []
    for world in sorted({r["world"] for r in rows}):
        for arm in ("bridge_ce", "aligned"):
            subset = [r for r in rows if r["world"] == world and r["arm"] == arm]
            if not subset:
                continue
            by_world.append(
                dict(
                    world=world,
                    arm=arm,
                    **{k: float(np.mean([r[k] for r in subset])) for k in keys},
                )
            )
    for arm in ("bridge_ce", "aligned"):
        subset = [r for r in by_world if r["arm"] == arm]
        if subset:
            result[arm] = {k: float(np.mean([r[k] for r in subset])) for k in keys}
    return result, by_world


def report(config_path):
    config = json.loads(Path(config_path).read_text())
    root = Path(config["repository"]) / "results" / config["batch"]
    target = root / "report"
    target.mkdir(parents=True, exist_ok=True)
    rows, histories = [], {}
    for spec in config["specs"]:
        path = root / "runs" / spec["name"]
        if not (path / "audit.json").exists():
            continue
        audit = json.loads((path / "audit.json").read_text())
        assert audit["passed"]
        complete = json.loads((path / "complete.json").read_text())
        scores = recount(path, spec)
        for task in TASKS:
            assert scores[task] == complete["metrics"][task]["accuracy"]
        exposure_audit(path, spec)
        diagnostics = json.loads((path / "state-diagnostics.json").read_text())
        scores.update(
            {
                task + "_first_hop_readout": diagnostics[task][
                    "first_hop_original_readout_accuracy"
                ]
                for task in ("common_atomic", "familiar_test", "strict_test")
            }
        )
        scores.update(
            {
                task + "_cosine": diagnostics[task]["mean_cosine_to_input_embedding"]
                for task in ("familiar_test", "strict_test")
            }
        )
        rows.append(
            dict(
                name=spec["name"],
                phase=spec["phase"],
                world=spec["world"],
                initialization=spec["initialization"],
                arm=spec["arm"],
                parameters=spec["parameters"],
                steps=spec["steps"],
                **scores,
                training_seconds=complete["training_seconds"],
                **{
                    k: complete[k]
                    for k in ["world_sha256", "initial_model_sha256", "sample_stream_sha256"]
                },
            )
        )
        histories[spec["name"]] = json.loads((path / "learning.json").read_text())
    if not rows:
        raise RuntimeError("No audited endpoints")
    pairs = []
    keys = [
        k
        for k in rows[0]
        if k in TASKS
        or k.endswith(
            ("_first_hop_readout", "_cosine", "_atomic_coverage", "_autonomous_two_calls")
        )
    ]
    for world, initialization in sorted({(r["world"], r["initialization"]) for r in rows}):
        pair = {
            r["arm"]: r
            for r in rows
            if (r["world"], r["initialization"]) == (world, initialization)
        }
        assert set(pair) == {"bridge_ce", "aligned"}, pair.keys()
        for k in ["world_sha256", "initial_model_sha256", "sample_stream_sha256"]:
            assert pair["bridge_ce"][k] == pair["aligned"][k], k
        a, b = pair["aligned"], pair["bridge_ce"]
        pairs.append(
            dict(
                phase=a["phase"],
                world=world,
                initialization=initialization,
                **{k + "_difference": a[k] - b[k] for k in keys},
            )
        )
        with (
            np.load(root / "runs" / a["name"] / "exposures.npz") as x,
            np.load(root / "runs" / b["name"] / "exposures.npz") as y,
        ):
            for k in x.files:
                np.testing.assert_array_equal(x[k], y[k])
    confirmation = [r for r in rows if r["phase"] == "confirmation"]
    means, world_means = aggregate(confirmation, keys)
    development, _ = aggregate([r for r in rows if r["phase"] == "development"], keys)
    prereqs = [
        r
        for r in confirmation
        if min(
            r["common_atomic"],
            r["train_composite"],
            r["common_atomic_first_hop_readout"],
            r["strict_test_first_hop_readout"],
        )
        < 0.99
    ]
    summary = dict(
        batch=config["batch"],
        audited_runs=len(rows),
        confirmation_runs=len(confirmation),
        confirmation_worlds=len({r["world"] for r in confirmation}),
        means=means,
        world_means=world_means,
        pairs=pairs,
        development=development,
        pairing_and_exposure_audit_passed=True,
        prereq_unmatched_runs=[r["name"] for r in prereqs],
        total_training_gpu_hours=sum(r["training_seconds"] for r in rows) / 3600,
        historical_reference=dict(
            bridge_ce_strict=0.1260, aligned_strict=0.9232, unpaired_historical_context=True
        ),
    )
    write_json(target / "summary.json", summary)
    with (target / "endpoints.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (target / "pairs.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    text = [
        "# 普通独立层的表示训练干预复核",
        "",
        f"固定16k端点：{len(rows)}条重载通过，正式{len(confirmation)}条，三个新世界各两个初始化。",
        "开发不进入正式均值；先平均每世界初始化，再世界等权。",
        "",
        "| 两层普通Transformer | 单跳 | 训练组合 | 中间实体读出（严格池） "
        "| 熟悉留出组合 | 严格只见单跳事实的组合 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in [("bridge_ce", "同首跳实体CE"), ("aligned", "同首跳CE＋表示对齐")]:
        if arm in means:
            m = means[arm]
            text.append(
                "| "
                + label
                + " | "
                + " | ".join(
                    f"{100 * m[k]:.2f}%"
                    for k in [
                        "common_atomic",
                        "train_composite",
                        "strict_test_first_hop_readout",
                        "familiar_test",
                        "strict_test",
                    ]
                )
                + " |"
            )
    if means:
        delta = 100 * (means["aligned"]["strict_test"] - means["bridge_ce"]["strict_test"])
        positives = sum(
            p["strict_test_difference"] > 0 for p in pairs if p["phase"] == "confirmation"
        )
        text += [
            "",
            f"严格组合差值{delta:+.2f}个百分点，六配对中{positives}个为正。",
            f"正式前提不匹配的轨迹：{len(prereqs)}条，全部保留；见summary.json。",
            "",
            "| 世界 | 首跳CE严格 | 对齐严格 | 差值（百分点） |",
            "| --- | ---: | ---: | ---: |",
        ]
        for world in sorted({r["world"] for r in confirmation}):
            w = {r["arm"]: r for r in world_means if r["world"] == world}
            x, y = w["bridge_ce"]["strict_test"], w["aligned"]["strict_test"]
            text.append(f"| {world} | {100 * x:.2f}% | {100 * y:.2f}% | {100 * (y - x):+.2f} |")
    text += [
        "",
        "结果解释限定于宽128、两独立层、人工事实和模板两跳。组内只改变对齐权重。",
        "旧共享模型12.60%／92.32%是不同世界上的历史参照；独立层有更多参数且标准初始化缩放不同，跨批次差值不能归因为纯共享约束。",
        "方向、首跳身份与组合同时改善不能唯一定位MLP或证明方向对齐是必要条件。新知识迁移尚未开展。",
        "",
        f"科学训练段合计{summary['total_training_gpu_hours']:.4f}GPU小时；容器启动、评价和重载分配耗时另见controller-state.json。",
        "逐题完整生成重计分、配对初值/世界/采样、逐事实曝光与独立重载均通过。W&B云端完成需另见tracking-cloud-audit.json。",
        "",
    ]
    (target / "report.md").write_text("\n".join(text))
    plot(rows, histories, target)
    print(json.dumps(dict(audited=len(rows), means=means, report=str(target)), ensure_ascii=False))


def plot(rows, histories, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, task, title in zip(
        axes,
        ["familiar_test", "strict_test"],
        ["Familiar facts: held-out chains", "Facts with atomic training only"],
        strict=True,
    ):
        for arm, color in [("bridge_ce", "#5973a6"), ("aligned", "#cf733c")]:
            values = []
            for world in sorted({r["world"] for r in rows if r["phase"] == "confirmation"}):
                subset = [
                    r
                    for r in rows
                    if r["phase"] == "confirmation" and r["world"] == world and r["arm"] == arm
                ]
                if subset:
                    values.append(
                        np.mean(
                            [
                                [p["metrics"][task]["accuracy"] for p in histories[r["name"]]]
                                for r in subset
                            ],
                            axis=0,
                        )
                    )
            if values:
                nodes = [p["step"] for p in histories[subset[0]["name"]]]
                scores = 100 * np.asarray(values)
                ax.plot(nodes, scores.mean(0), label=arm, color=color)
                ax.fill_between(nodes, scores.min(0), scores.max(0), color=color, alpha=0.15)
        ax.set(
            title=title,
            xlabel="Optimizer updates",
            ylabel="Full generation accuracy (%)",
            ylim=(0, 100),
        )
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(target / "learning.png", dpi=180)
    fig.savefig(target / "learning.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    report(parser.parse_args().config)
