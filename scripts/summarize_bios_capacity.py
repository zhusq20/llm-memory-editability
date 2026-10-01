"""Recompute metrics and exposure audits from every saved capacity prediction."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_capacity import qa_split, split_metrics
from llm_memory_editability.bios_data import N_BASE, load_world, write_json


def summarize(root):
    rows, finals, errors = [], [], []
    configs = {}
    for config_path in sorted(root.glob("*/config.json")):
        run = config_path.parent
        config = json.loads(config_path.read_text())
        configs[run.name] = config
        world = load_world(config["world"])
        train_ids, heldout_ids = qa_split(world, config["qa_split"])
        if (
            train_ids.tolist() != config["qa_train_ids"]
            or heldout_ids.tolist() != config["qa_heldout_ids"]
        ):
            errors.append(f"{run.name}: split mismatch")
        with np.load(run / "schedule.npz") as schedule:
            if not np.isin(schedule["derived"], train_ids).all():
                errors.append(f"{run.name}: heldout leakage in schedule")
        path = run / "learning.json"
        if not path.exists():
            continue
        learning = json.loads(path.read_text())
        for record in learning:
            step = record["step"]
            with np.load(run / f"predictions-{step}.npz") as arrays:
                recomputed = (arrays["prediction"] == world.answers) & arrays["ended"]
                if not np.array_equal(recomputed, arrays["correct"]):
                    errors.append(f"{run.name}/{step}: incorrect prediction scoring")
                if arrays["exposure"][heldout_ids].sum() != 0:
                    errors.append(f"{run.name}/{step}: heldout exposure")
                if (
                    arrays["exposure"][:N_BASE].sum() != step * 112
                    or arrays["exposure"][N_BASE:].sum() != step * 28
                ):
                    errors.append(f"{run.name}/{step}: exposure total mismatch")
                metrics = split_metrics(world, arrays, train_ids, heldout_ids)
                base = float(recomputed[:N_BASE].mean())
                nll = float(arrays["value_nll"][:N_BASE].mean())
                for key, value in {"base_accuracy": base, "value_nll": nll, **metrics}.items():
                    if value is None:
                        good = record.get(key) is None
                    else:
                        good = record.get(key) is not None and np.isclose(
                            record[key], value, atol=1e-8
                        )
                    if not good:
                        errors.append(f"{run.name}/{step}: metric mismatch {key}")
                row = {
                    "run": run.name,
                    "world": world.seed,
                    "seed": config["seed"],
                    "condition": config["condition"],
                    "width": config["model"]["width"],
                    "parameters": config["parameters"],
                    "lr": config["lr"],
                    "qa_split": config["qa_split"],
                    "step": step,
                    "base_accuracy": base,
                    "value_nll": nll,
                    "train_seconds": record["train_seconds"],
                    **metrics,
                    **{f"stratum_{k}": v for k, v in record["strata"].items()},
                }
                rows.append(row)
        if learning[-1]["step"] == config["steps"] and (run / "complete.json").exists():
            end = rows[-1].copy()
            middle = next(r for r in learning if r["step"] == 14336)
            end["base_change_from_14336"] = end["base_accuracy"] - middle["base_accuracy"]
            end["nll_relative_change_from_14336"] = (end["value_nll"] - middle["value_nll"]) / max(
                middle["value_nll"], 1e-12
            )
            end["candidate_plateau"] = (
                abs(end["base_change_from_14336"]) <= 0.005
                and abs(end["nll_relative_change_from_14336"]) <= 0.05
            )
            finals.append(end)
    # A/B/C must share actual initial weights and exposure, not merely nominal seeds.
    groups = {}
    for name, c in configs.items():
        key = (c["world_seed"], c["seed"], c["model"]["width"], c["lr"], c["qa_split"])
        groups.setdefault(key, []).append((name, c))
    for group in groups.values():
        for field in (
            "model_initial_sha256",
            "derived_schedule_sha256",
            "truth_sha256",
            "prompts_sha256",
        ):
            if len({c[field] for _, c in group}) != 1:
                errors.append(f"{group[0][0]}: cross-condition mismatch {field}")
        timelines = [
            json.loads((root / name / "learning.json").read_text())
            for name, _ in group
            if (root / name / "learning.json").exists()
        ]
        common = (
            set.intersection(*[{r["step"] for r in timeline} for timeline in timelines])
            if timelines
            else set()
        )
        for step in common:
            records = [next(r for r in timeline if r["step"] == step) for timeline in timelines]
            for field in ("exposure_sha256", "slot_exposure_sha256", "lr_weighted_exposure_sha256"):
                if len({r[field] for r in records}) != 1:
                    errors.append(f"{group[0][0]}/{step}: cross-condition {field}")
    for name, values in (("learning.csv", rows), ("endpoints.csv", finals)):
        if values:
            with (root / name).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    write_json(
        root / "audit.json",
        {
            "runs_seen": len(configs),
            "completed": len(finals),
            "checkpoints": len(rows),
            "errors": errors,
            "passed": not errors,
        },
    )
    lines = [
        "# 宽度扫描与组合 QA 留出",
        "",
        f"已完成 {len(finals)} 条轨迹；复核 {len(rows)} 个检查点；审计错误 {len(errors)}。",
        "",
        "两个已有开发世界、每世界一个初始化；下表为两世界均值，仅完整的两世界条件组展示。",
        "",
        "| QA | 宽度 | 学习率 | 组织 | 基础准确率 | 相比原预算变化/百分点 | 留出组合准确率 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    aggregate = {}
    for r in finals:
        key = (r["qa_split"], r["width"], r["lr"], r["condition"])
        aggregate.setdefault(key, []).append(r)
    for (split, width, lr, condition), group in sorted(aggregate.items()):
        if len(group) != 2:
            continue
        base = np.mean([r["base_accuracy"] for r in group]) * 100
        delta = np.mean([r["base_change_from_14336"] for r in group]) * 100
        heldout = (
            f"{np.mean([r['derived_heldout_accuracy'] for r in group]) * 100:.2f}%"
            if split == "half"
            else "—"
        )
        lines.append(
            f"| {split} | {width} | {lr:g} | {condition} | {base:.2f}% | {delta:+.2f} | {heldout} |"
        )
    lines += [
        "",
        "候选平台仅是预定两预算间的准确率/NLL稳定性标记，不证明绝对表达能力上限。未达平台、反向和零效应均保留。留出流固定总预算，因此纳入QA的单条曝光约加倍。完整逐模型和已知组成事实子集指标见CSV。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if errors:
        raise ValueError("Audit failed: " + "; ".join(errors[:5]))
    return {"completed": len(finals), "checkpoints": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/bios-capacity-dev-v1"))
    print(json.dumps(summarize(parser.parse_args().root)))
