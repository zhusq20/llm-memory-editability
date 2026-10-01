"""Fixed-512 E39 contrasts, retaining every world/initialization and matched cohort."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

METRICS = (
    "E_accuracy",
    "E_roots_accuracy",
    "E_actual_accuracy",
    "D_heldout_accuracy",
    "paired_reference_D_heldout_accuracy",
    "D_original_exception_accuracy",
    "D_newly_exception_accuracy",
    "D_remaining_ordinary_unedited_accuracy",
    "U_full_damage",
    "U_full_coverage",
    "U_full_known",
    "U_full_broken",
    "U_heldout_damage",
    "U_heldout_coverage",
    "U_heldout_known",
    "U_heldout_broken",
    "U_full_strata_0_damage",
    "U_full_strata_1_damage",
)


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    rows = []
    for row in csv.DictReader(path.open()):
        for key, value in row.items():
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
        rows.append(row)
    return rows


def summarize_pairs(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        item = dict(zip(keys, key, strict=True))
        item["paired_cases"] = len(group)
        for metric in METRICS:
            for prefix in ("low_", "high_", "high_minus_low_"):
                name = prefix + metric
                values = [r[name] for r in group if r[name] is not None]
                item[name] = mean(values) if values else None
                item[name + "_valid_cases"] = len(values)
        output.append(item)
    return output


def common_known_controls(source):
    """Secondary retention diagnostic; do not replace each-phase primary U rates."""
    rows = []
    for world in (0, 1):
        for seed in (0, 1):
            for condition in ("company", "project", "neither"):
                name = f"width-256/world-{world}-seed-{seed}-{condition}"
                for chain in ("company", "project"):
                    for kind in ("coherent", "exception"):
                        case = f"{chain}-{kind}-mlp"
                        paths = {phase: source / phase / name / case for phase in ("low", "high")}
                        sets, arrays = {}, {}
                        for phase, path in paths.items():
                            with np.load(path / "sets.npz") as saved:
                                sets[phase] = dict(saved)
                            with np.load(path / "predictions-512.npz") as saved:
                                arrays[phase] = dict(saved)
                            truth = sets[phase][f"{phase}_{kind}"]
                            np.testing.assert_array_equal(
                                arrays[phase]["correct"],
                                (arrays[phase]["prediction"] == truth) & arrays[phase]["ended"],
                            )
                        common_known = sets["low"]["old_correct"] & sets["high"]["old_correct"]
                        same_truth = sets["low"][f"low_{kind}"] == sets["high"][f"high_{kind}"]
                        for pool in ("U_full", "U_heldout"):
                            np.testing.assert_array_equal(sets["low"][pool], sets["high"][pool])
                            for stratum in (-1, 0, 1, 2, 3, 4):
                                ids = sets["low"][pool]
                                if stratum >= 0:
                                    ids = ids[sets["low"]["U_strata"][ids] == stratum]
                                for control in ("common_known", "same_truth_common_known"):
                                    mask = common_known.copy()
                                    if control == "same_truth_common_known":
                                        mask &= same_truth
                                    retained = ids[mask[ids]]
                                    row = dict(
                                        world=world,
                                        seed=seed,
                                        condition=condition,
                                        chain=chain,
                                        kind=kind,
                                        step=512,
                                        pool=pool,
                                        stratum=stratum,
                                        control=control,
                                        pool_n=len(ids),
                                        known=len(retained),
                                    )
                                    for phase in ("low", "high"):
                                        broken = int((~arrays[phase]["correct"][retained]).sum())
                                        row[phase + "_broken"] = broken
                                        row[phase + "_damage"] = (
                                            broken / len(retained) if len(retained) else None
                                        )
                                    row["high_minus_low_damage"] = (
                                        row["high_damage"] - row["low_damage"]
                                        if len(retained)
                                        else None
                                    )
                                    rows.append(row)
    return rows


def analyze(source):
    source = Path(source).resolve()
    output = source / "summary"
    audit = json.loads((output / "audit.json").read_text())
    if not audit["complete"] or audit["missing"] or audit["cases"] != 96:
        raise ValueError("Only the complete audited 96-case E39 matrix may be analyzed")
    paired = [r for r in read_csv(output / "paired-low-high.csv") if r["step"] == 512]
    if len(paired) != 48:
        raise ValueError("Missing fixed-endpoint matched pairs")
    blocks = summarize_pairs(paired, ("world", "seed", "kind"))
    if len(blocks) != 8 or any(row["paired_cases"] != 6 for row in blocks):
        raise ValueError("World/initialization block is incomplete")
    worlds = summarize_pairs(paired, ("world", "kind"))
    overall = summarize_pairs(paired, ("kind",))
    organizations = summarize_pairs(paired, ("condition", "kind"))
    controls = common_known_controls(source)
    for filename, rows in (
        ("endpoint-paired-cases.csv", paired),
        ("endpoint-world-seed.csv", blocks),
        ("endpoint-world.csv", worlds),
        ("endpoint-overall.csv", overall),
        ("endpoint-organization.csv", organizations),
        ("common-known-retention.csv", controls),
    ):
        write_csv(output / filename, rows)
    result = {
        "fixed_step": 512,
        "paired_cases": len(paired),
        "world_seed_blocks": blocks,
        "worlds": worlds,
        "overall": overall,
        "organizations": organizations,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary_audit_sha256": hashlib.sha256((output / "audit.json").read_bytes()).hexdigest(),
        "limitations": [
            "Two reused development worlds; initialization repeats are not independent worlds.",
            "All 24 parents and 96 edits are retained at fixed step512 without outcome selection.",
            "The 18/9-person coherent subset is paired reference, not conflict.",
            "Primary retention uses each parent own truth and old-correct coverage. "
            "Common-known controls are secondary and retain their integer denominators.",
            "E39 is a separate common-support manipulation, never pooled with historical E93.",
        ],
    }
    (output / "endpoint-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# E39 固定终点配对分析",
        "",
        "全部24父模型、96编辑固定512步；先报告世界/初始化，再给两个世界的描述性均值。"
        "下列D均为相同留出9人：exception中为冲突传播，coherent中仅为配对参照。",
        "",
        "| 世界 | 初始化 | 更新 | 低比例D | 高比例D | 差异pp | 低比例E actual | 高比例E actual |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in blocks:
        values = [
            row[k] * 100
            for k in (
                "low_paired_reference_D_heldout_accuracy",
                "high_paired_reference_D_heldout_accuracy",
                "high_minus_low_paired_reference_D_heldout_accuracy",
                "low_E_actual_accuracy",
                "high_E_actual_accuracy",
            )
        ]
        lines.append(
            f"| {int(row['world'])} | {int(row['seed'])} | {row['kind']} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + " |"
        )
    lines += [
        "",
        "全池/未见/局部U的阶段自身覆盖与旧正确整数损伤见editing.csv；"
        "共同已知ID以及进一步限制低高真值一致的次要控制见common-known-retention.csv。"
        "不得用次要子集替换全体主分析，也不将两次初始化当成新增数据世界。",
        "",
    ]
    (output / "endpoint-report.md").write_text("\n".join(lines))
    print(json.dumps({"paired_cases": len(paired), "blocks": len(blocks), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/bios-mechanism-dev-v1/p3-shortcut-edit-v2")
    analyze(parser.parse_args().source)
