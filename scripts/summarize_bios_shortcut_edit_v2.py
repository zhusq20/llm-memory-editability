"""Lightweight E39 audit: independently rescore saved query arrays, without weights."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, make_cross_world
from llm_memory_editability.bios_cross_continue import file_hash
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_shortcut_control import high_exception_world
from llm_memory_editability.bios_shortcut_edit_v2 import (
    CHECKPOINTS,
    STUDY,
    edit_metrics,
    editing_sources,
    sampling_stream,
    score_arrays,
)
from llm_memory_editability.bios_shortcut_matched_edit import KINDS, PHASES, make_matched_edit_pair

ROOT = Path(__file__).resolve().parents[1]


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair_rows(rows, varying, levels):
    keys = ["phase", "world", "seed", "condition", "chain", "kind", "step"]
    keys.remove(varying)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[tuple(row[key] for key in keys)][row[varying]] = row
    result = []
    for key, group in sorted(grouped.items()):
        if set(group) != set(levels):
            continue
        a, b = (group[level] for level in levels)
        metrics = [name for name in a if name not in keys + [varying]]
        row = dict(zip(keys, key, strict=True))
        for name in metrics:
            if name not in b:
                continue
            row[f"{levels[0]}_{name}"] = a[name]
            row[f"{levels[1]}_{name}"] = b[name]
            row[f"{levels[1]}_minus_{levels[0]}_{name}"] = (
                b[name] - a[name] if a[name] is not None and b[name] is not None else None
            )
        result.append(row)
    return result


def summarize(source, output, require_complete=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "launch-contract.json").read_text())
    if manifest["sources"] != editing_sources():
        raise ValueError("Frozen E39 source hashes changed")
    if manifest["runner_sha256"] != file_hash(ROOT / "scripts/run_bios_shortcut_edit_v2.py"):
        raise ValueError("Frozen E39 runner changed")
    jobs = manifest["jobs"]
    expected = {(p, w, s, c) for p in PHASES for w in (0, 1) for s in (0, 1) for c in CONDITIONS}
    actual = {(j["phase"], j["world"], j["seed"], j["condition"]) for j in jobs}
    if len(jobs) != 24 or actual != expected or len({job["id"] for job in jobs}) != 24:
        raise ValueError("E39 manifest is not the full balanced 24-parent matrix")
    worlds, pairs = {}, {}
    for w in (0, 1):
        low = make_cross_world(w)
        high, _, _, _ = high_exception_world(low)
        worlds[w] = {"low": low, "high": high}
        for chain in range(2):
            pairs[w, chain] = make_matched_edit_pair(low, high, chain)
    rows, missing, ledger = [], [], {}
    models, cases, checkpoints = 0, 0, 0
    for job in jobs:
        run = source / job["id"]
        if not (run / "config.json").exists():
            missing.append(str(run))
            continue
        config = json.loads((run / "config.json").read_text())
        for key in ("phase", "world", "seed", "condition"):
            if config[key] != job[key]:
                raise ValueError(f"E39 run identity mismatch: {run}/{key}")
        if config["study"] != STUDY or config["sources"] != manifest["sources"]:
            raise ValueError("E39 worker study/source mismatch")
        parent = Path(job["parent"])
        if config["parent"]["directory"] != str(parent.resolve()):
            raise ValueError("E39 parent identity mismatch")
        for filename in ("config.json", "predictions-15360.npz", "learning-complete.json"):
            observed = file_hash(parent / filename)
            if observed != config["parent"]["files_sha256"][filename]:
                raise ValueError(f"E39 parent metadata/predictions changed: {filename}")
            ledger[str(parent / filename)] = observed
        phase, w = job["phase"], job["world"]
        world = worlds[w][phase]
        parent_config = json.loads((parent / "config.json").read_text())
        if parent_config["study"]["steps"] != 15360 or parent_config["model"]["width"] != 256:
            raise ValueError("E39 parent is not width256/15360")
        if parent_config["truth_sha256"] != array_hash(world.answers):
            raise ValueError("E39 parent truth mismatch")
        baseline = score_arrays(load_arrays(parent / "predictions-15360.npz"), world.answers)
        local_cases = 0
        for chain, chain_name in enumerate(CHAINS):
            pair = pairs[w, chain]
            if config["data_contracts"][chain] != pair["contract"]:
                raise ValueError("E39 common-support contract changed")
            e_choices, r_choices = sampling_stream(w, chain)
            for kind in KINDS:
                dest = run / f"{chain_name}-{kind}-mlp"
                if not (dest / "complete.json").exists():
                    missing.append(str(dest))
                    continue
                completion = json.loads((dest / "complete.json").read_text())
                if any(
                    completion[key] != value
                    for key, value in {
                        "status": "complete",
                        "scope": "mlp",
                        "phase": phase,
                        "chain": chain_name,
                        "kind": kind,
                    }.items()
                ):
                    raise ValueError("Invalid E39 completion record")
                if json.loads((dest / "data-contract.json").read_text()) != pair["contract"]:
                    raise ValueError("E39 case data contract differs")
                sets = load_arrays(dest / "sets.npz")
                expected_sets = {k: v for k, v in pair.items() if isinstance(v, np.ndarray)}
                expected_sets.update(
                    old_correct=baseline["correct"],
                    edit_sampling=e_choices,
                    replay_sampling=r_choices,
                )
                for key, value in expected_sets.items():
                    np.testing.assert_array_equal(sets[key], value)
                timeline = json.loads((dest / "trajectory.json").read_text())
                if [point["step"] for point in timeline] != list(CHECKPOINTS):
                    raise ValueError("E39 trajectory has missing or extra checkpoints")
                for point in timeline:
                    step = point["step"]
                    path = dest / f"predictions-{step}.npz"
                    arrays = score_arrays(load_arrays(path), pair[f"{phase}_{kind}"])
                    ledger[str(path)] = file_hash(path)
                    if step == 0:
                        for key in ("prediction", "ended"):
                            np.testing.assert_array_equal(arrays[key], baseline[key])
                    metrics = edit_metrics(pair, phase, kind, arrays, baseline["correct"])
                    for key, value in metrics.items():
                        if point[key] != value:
                            raise ValueError(
                                f"Stored E39 metric does not reproduce: {dest}/{step}/{key}"
                            )
                    if step == 512 and completion["final"] != point:
                        raise ValueError("E39 completion and final trajectory disagree")
                    rows.append(
                        {
                            **{k: job[k] for k in ("phase", "world", "seed", "condition")},
                            "chain": chain_name,
                            "kind": kind,
                            "step": step,
                            **flatten(metrics),
                        }
                    )
                    checkpoints += 1
                local_cases += 1
                cases += 1
        models += local_cases == 4 and (run / "complete.json").exists()
    paired_phase = pair_rows(rows, "phase", ("low", "high"))
    paired_type = pair_rows(rows, "kind", ("coherent", "exception"))
    blocks = []
    group_keys = ("phase", "world", "seed", "kind", "step")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    metrics = (
        "E_accuracy",
        "E_roots_accuracy",
        "E_actual_accuracy",
        "D_heldout_accuracy",
        "paired_reference_D_heldout_accuracy",
        "U_full_damage",
        "U_full_coverage",
        "U_full_known",
        "U_full_broken",
        "U_heldout_damage",
    )
    for key, group in sorted(grouped.items()):
        if len(group) != 6:
            continue
        blocks.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "cases": 6,
                **{
                    metric: mean(row[metric] for row in group if row[metric] is not None)
                    if any(row[metric] is not None for row in group)
                    else None
                    for metric in metrics
                },
            }
        )
    complete = models == 24 and cases == 96 and checkpoints == 384 and not missing
    audit = {
        "complete": complete,
        "models": models,
        "cases": cases,
        "checkpoints": checkpoints,
        "expected_models": 24,
        "expected_cases": 96,
        "expected_checkpoints": 384,
        "missing": missing,
        "model_weights_read": False,
        "query_arrays_rescored": True,
        "summary_source_sha256": file_hash(__file__),
        "sources": manifest["sources"],
    }
    write_csv(output / "editing.csv", rows)
    write_csv(output / "paired-low-high.csv", paired_phase)
    write_csv(output / "paired-update-type.csv", paired_type)
    write_csv(output / "world-seed-blocks.csv", blocks)
    write_json(output / "audit.json", audit)
    write_json(output / "sources.json", ledger)
    lines = [
        "# 同支持 E39：低/高比例知识更新",
        "",
        f"完整状态：{complete}；{models}/24模型，{cases}/96编辑，{checkpoints}/384检查点。",
        "",
        "所有终点固定512步；保留0/32/128/512逐查询计分。低/高按相同世界、初始化、组织、"
        "链和更新类型配对。coherent中的18/9人仅称配对参照；只有exception中称冲突传播。",
        "",
        "| 比例 | 更新 | 案例 | E root | E actual | 留出D | 配对留出9人 | U损伤 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        for kind in KINDS:
            group = [
                row
                for row in rows
                if row["phase"] == phase and row["kind"] == kind and row["step"] == 512
            ]
            if group:
                values = [
                    mean(row[key] for row in group if row[key] is not None)
                    for key in (
                        "E_roots_accuracy",
                        "E_actual_accuracy",
                        "D_heldout_accuracy",
                        "paired_reference_D_heldout_accuracy",
                        "U_full_damage",
                    )
                ]
                lines.append(
                    f"| {phase} | {kind} | {len(group)} | "
                    + " | ".join(f"{100 * v:.3f}%" for v in values)
                    + " |"
                )
    lines += [
        "",
        "world-seed-blocks.csv先保留世界/初始化异质性；paired-low-high.csv给相同人员/支持的逐案例配对。"
        "editing.csv另含原例外、新增例外、未编辑普通人、全池及未见U、五种局部分层和旧正确覆盖。",
        "",
        "轻量审计重新计算预测值+EOS与各自阶段真值的得分，不加载模型权重。"
        "原E93结果不纳入此比较；两个开发世界不构成独立确认样本。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    if require_complete and not complete:
        raise ValueError(f"E39 matrix is incomplete: {audit}")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/bios-mechanism-dev-v1/p3-shortcut-edit-v2")
    parser.add_argument("--output")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                args.source, args.output or Path(args.source) / "summary", args.require_complete
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
