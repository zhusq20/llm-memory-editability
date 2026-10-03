#!/usr/bin/env python
"""Independent saved-output scoring and architecture audit for the Qwen batch."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def independently_grade(text, gold):
    output = text.strip().split("\n")[0]

    def clean(s):
        return re.sub(r"\s+", " ", s.strip().rstrip(".").strip()).casefold()

    return clean(output) == clean(gold)


def shapes(path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {k: handle.get_slice(k).get_shape() for k in handle.keys()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    art, output = Path(cfg["artifact_root"]), Path(cfg["output_root"])
    summary = read(art / "summary.json")
    original_shapes = shapes(Path(cfg["model_path"]) / "model.safetensors")
    cells, failures, predictions, reloads = [], [], 0, 0
    for world in cfg["world_seeds"]:
        for arm in cfg["arms"]:
            name = f"w{world}-{arm}"
            run = summary["runs"][name]
            root = output / name
            assert shapes(root / "model/model.safetensors") == original_shapes
            assert read(root / "model/config.json")["num_hidden_layers"] == 28
            assert run["parameter_count"] == 596049920
            reload = read(root / "reload-audit.json")
            assert reload["exact_generated_tokens"] and reload["exact_autonomous_prompts"]
            reloads += reload["predictions"]
            endpoint = None
            for point in read(root / "curve.json"):
                rows = read(root / f"predictions-{point['step']:05d}.json")
                predictions += len(rows)
                for row in rows:
                    if independently_grade(row["text"], row["answer"]) != row["correct"]:
                        failures.append(dict(name=name, step=point["step"], row=row["id"]))
                if point["step"] == cfg["steps"]:
                    endpoint = rows
            assert endpoint is not None
            atoms = {(r["id"], r["view"]): r for r in endpoint if r["mode"] == "atomic"}
            for row in endpoint:
                if row["mode"] == "autonomous":
                    first = atoms[row["atom_ids"][0], row["view"]]
                    generated = first["text"].strip().split("\n")[0].strip().rstrip(".").strip()
                    assert row["generated_bridge"] == generated
                    assert row["first_tokens"] == first["tokens"]
                    assert generated in row["prompt"]
            for metric, cell in run["metrics"].items():
                cells.append(
                    dict(
                        world=world,
                        arm=arm,
                        metric=metric,
                        correct=cell["correct"],
                        n=cell["n"],
                        accuracy=cell["correct"] / cell["n"] if cell["n"] else None,
                    )
                )
    assert not failures
    write(
        art / "independent-audit.json",
        dict(
            status="passed",
            independently_rescored_predictions=predictions,
            exact_reload_predictions=reloads,
            checkpoint_tensor_shapes=len(original_shapes),
            architectures=len(cfg["world_seeds"]) * len(cfg["arms"]),
            parameter_count=596049920,
            autonomous_bridge_source="actual saved first-hop generation",
            failures=failures,
        ),
    )
    with (art / "endpoint-cells.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("v0_heldout_direct", "v0_strict_direct"),
        ("New chains of familiar facts", "Facts with atomic training only"),
        strict=True,
    ):
        for index, arm in enumerate(cfg["arms"]):
            rows = [c for c in cells if c["metric"] == metric and c["arm"] == arm]
            ax.bar(
                [i + 0.35 * index for i in range(len(rows))],
                [r["accuracy"] * 100 for r in rows],
                0.35,
                label=f"{arm}: {8 if arm == 'low' else 32} support chains",
                color=("#8999ad" if arm == "low" else "#227d9b"),
            )
        ax.set_xticks(
            [i + 0.175 for i in range(len(cfg["world_seeds"]))],
            [str(w) for w in cfg["world_seeds"]],
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("World")
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Complete-answer accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(art / "per-world.png", dpi=160)
    fig.savefig(art / "per-world.pdf")
    plt.close(fig)
    lines = [
        "# 完整Qwen组合支持比较",
        "",
        f"{len(cfg['world_seeds'])}个世界、一个配对训练种子、low8/high32支持，每臂固定{cfg['steps']}步。主评分为view0闭卷完整首行答案EM。",
        "",
        "| 世界 | 支持 | 熟悉新链 | 严格原子训练链 | 单跳 | 训练组合 | 自主两次调用（熟悉/严格） |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    aggregate = {}
    for arm in cfg["arms"]:
        aggregate[arm] = {}
        for metric in ("v0_heldout_direct", "v0_strict_direct"):
            selected = [c for c in cells if c["arm"] == arm and c["metric"] == metric]
            aggregate[arm][metric] = dict(
                correct=sum(c["correct"] for c in selected),
                n=sum(c["n"] for c in selected),
                world_mean=sum(c["accuracy"] for c in selected) / len(selected),
            )
    low = aggregate["low"]["v0_heldout_direct"]
    high = aggregate["high"]["v0_heldout_direct"]
    lines.insert(
        4,
        f"熟悉新链low为{low['correct']}/{low['n']}（{100 * low['world_mean']:.2f}%），"
        f"high为{high['correct']}/{high['n']}（{100 * high['world_mean']:.2f}%）。"
        "本配方未观察到较多组合支持带来稳定的未训练组合收益；"
        "原子提取、训练组合掌握和自主两次调用的完整结果同时保留。",
    )
    lines.insert(5, "")
    for world in cfg["world_seeds"]:
        for arm in cfg["arms"]:
            m = summary["runs"][f"w{world}-{arm}"]["metrics"]

            def format_cell(key, m=m):
                return f"{m[key]['correct']}/{m[key]['n']}"

            atomic_keys = [
                f"v0_{pool}_{role}"
                for pool in ("familiar", "strict")
                for role in ("first", "second")
            ]
            atomic = (
                f"{sum(m[k]['correct'] for k in atomic_keys)}/{sum(m[k]['n'] for k in atomic_keys)}"
            )
            lines.append(
                f"| {world} | {arm} | {format_cell('v0_heldout_direct')} "
                f"| {format_cell('v0_strict_direct')} | {atomic} "
                f"| {format_cell('v0_training_direct')} "
                f"| {format_cell('v0_heldout_autonomous')} "
                f"/ {format_cell('v0_strict_autonomous')} |"
            )
    lines.extend(
        [
            "",
            "共同原子QA流与loss尺度完全匹配；支持数同时改变组合角色覆盖、每例重复和组合监督内容。高支持必要熟悉原子均参与组合，四关系搭配在两臂都训练过；严格池人物及城市隔离。固定终点与完整节点保留，不选择最好中途点。",
            "",
            "本比较提供完整预训练LM中的受控行为证据；不定位唯一MLP接口，不据一个模型规模宣称scale规律。自主两次调用提供已知分解及额外计算。查询和问法不视为独立世界，另一个留出问法的全量分数见endpoint-cells.csv。",
            "",
            f"独立重载{reloads}条生成词元完全一致；独立重计分{predictions}条节点预测、各完整checkpoint的{len(original_shapes)}张量形状与28层原架构核验通过。",
            "",
            f"纯训练合计{summary['total_train_seconds']:.2f}秒；运行内合计{summary['total_process_seconds']:.2f}秒（包含模型加载/训练/生成/保存/重载；不包含冻结前模型下载/解释器启动）。",
        ]
    )
    (art / "report.md").write_text("\n".join(lines) + "\n")
    write(art / "aggregate.json", aggregate)
    write(
        art / "completion-manifest.json",
        dict(
            status="complete",
            phase=cfg["phase"],
            config=args.config,
            independent_worlds=len(cfg["world_seeds"]),
            training_initializations=1,
            runs=len(cfg["world_seeds"]) * len(cfg["arms"]),
            updates=summary["total_updates"],
            input_tokens=sum(r["budget"]["input_tokens"] for r in summary["runs"].values()),
            supervised_tokens=sum(
                r["budget"]["supervised_tokens"] for r in summary["runs"].values()
            ),
            estimated_training_matrix_flops=sum(
                r["estimated_training_matrix_flops"] for r in summary["runs"].values()
            ),
            exact_reload_predictions=reloads,
            independent_node_prediction_checks=predictions,
            source_lock=str(art / "lock.json"),
            checkpoint_hashes={name: r["checkpoint_files"] for name, r in summary["runs"].items()},
            train_seconds=summary["total_train_seconds"],
            process_seconds=summary["total_process_seconds"],
        ),
    )


if __name__ == "__main__":
    main()
