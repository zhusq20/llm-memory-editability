"""Pair P3 shortcut-reliability runs with unchanged low-prevalence QA targets.

Reads predictions and small provenance files only; no model weights or GPU calls.
The primary comparisons use exactly the same people on both sides. Native
ordinary/exception averages are labeled separately because membership changes.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_cross import (
    CHAINS,
    CONDITIONS,
    documents,
    make_cross_world,
    qa_schedule,
)
from llm_memory_editability.bios_cross_continue import expected_exposure
from llm_memory_editability.bios_cross_train import learning_metrics, source_hashes
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_shortcut_control import high_exception_world, shortcut_sources

ROOT = Path(__file__).resolve().parents[1]
MECHANISM = ROOT / "results/bios-mechanism-dev-v1"
CHECKPOINTS = (0, 1280, 2560, 5120, 10240, 15360)


def read_json(path, ledger):
    content = path.read_bytes()
    ledger[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def read_arrays(path, ledger):
    content = path.read_bytes()
    ledger[str(path)] = hashlib.sha256(content).hexdigest()
    # Avoid reading a mutable output twice while generating its provenance hash.
    import io

    with np.load(io.BytesIO(content), allow_pickle=False) as saved:
        return dict(saved)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def boolean(array, name):
    if np.asarray(array).dtype != np.bool_:
        raise ValueError(f"Non-boolean {name}")


def score_direct(arrays, truth):
    for name in ("prediction", "ended", "correct", "value_nll"):
        if arrays[name].shape != truth.shape:
            raise ValueError(f"Wrong shape for {name}")
    boolean(arrays["ended"], "ended")
    boolean(arrays["correct"], "correct")
    correct = (arrays["prediction"] == truth) & arrays["ended"]
    np.testing.assert_array_equal(correct, arrays["correct"])
    if not np.isfinite(arrays["value_nll"]).all():
        raise ValueError("Nonfinite prediction NLL")
    return arrays


def score_two_step(arrays, world, chain, direct):
    ids = world.derived_ids[chain]
    np.testing.assert_array_equal(arrays["query_id"], ids)
    for key, value in arrays.items():
        if value.shape != ids.shape:
            raise ValueError(f"Wrong two-step shape for {key}")
    for key in (
        "bridge_ended",
        "bridge_valid",
        "bridge_correct",
        "ended",
        "correct",
        "direct_ended",
        "direct_correct",
    ):
        boolean(arrays[key], key)
    valid = arrays["bridge_ended"] & np.isin(
        arrays["bridge_prediction"], world.prompts[world.root_ids[chain], 1]
    )
    bridge = arrays["bridge_ended"] & (
        arrays["bridge_prediction"] == world.answers[world.membership_ids[chain]]
    )
    correct = valid & arrays["ended"] & (arrays["prediction"] == world.answers[ids])
    for key, expected in (
        ("bridge_valid", valid),
        ("bridge_correct", bridge),
        ("correct", correct),
    ):
        np.testing.assert_array_equal(arrays[key], expected)
    if np.any(arrays["ended"][~valid]) or np.any(arrays["prediction"][~valid] != -1):
        raise ValueError("Invalid bridge must skip second generation")
    for key in ("prediction", "ended", "correct"):
        np.testing.assert_array_equal(arrays[f"direct_{key}"], direct[key][ids])
    return arrays


def cohorts(old, high, chain, phase):
    """Fixed IDs are primary; phase-native labels are descriptive only."""
    n = old.derived_ids.shape[1]
    original = old.exceptions[chain]
    newly = high.exceptions[chain] & ~original
    remaining = ~high.exceptions[chain]
    if (int(original.sum()), int(newly.sum()), int(remaining.sum())) != (128, 896, 1024):
        raise ValueError("Unexpected fixed-cohort sizes")
    native = (old if phase == "low" else high).exceptions[chain]
    return (
        ("fixed", "all", np.ones(n, dtype=bool)),
        ("fixed", "original_exception", original),
        ("fixed", "newly_exception", newly),
        ("fixed", "remaining_ordinary", remaining),
        ("phase_native", "ordinary", ~native),
        ("phase_native", "exception", native),
    )


def chain_rows(identity, old, high, chain, phase, arrays, method):
    world = old if phase == "low" else high
    ids = world.derived_ids[chain]
    if method == "direct":
        a = {key: arrays[key][ids] for key in ("prediction", "ended", "correct", "value_nll")}
    else:
        a = arrays
    answer_actual = world.answers[world.actual_ids[chain]]
    conflict = answer_actual != world.answers[ids]
    rows = []
    for split, pool in (
        ("all", ids),
        ("trained", world.train_ids[chain]),
        ("heldout", world.heldout_ids[chain]),
    ):
        split_mask = np.isin(ids, pool)
        for kind, cohort, mask in cohorts(old, high, chain, phase):
            take = split_mask & mask
            n = int(take.sum())
            correct = int(a["correct"][take].sum())
            conflicting = take & conflict
            actual_match = a["ended"] & (a["prediction"] == answer_actual)
            row = {
                **identity,
                "phase": phase,
                "method": method,
                "chain": CHAINS[chain],
                "split": split,
                "cohort_kind": kind,
                "cohort": cohort,
                "n": n,
                "correct": correct,
                "accuracy": correct / n,
                "termination_error": int((~a["ended"][take]).sum()),
                "nll": float(a["value_nll"][take].mean()) if method == "direct" else None,
                "actual_conflict_n": int(conflicting.sum()),
                "actual_on_conflict": int(actual_match[conflicting].sum()),
                "actual_on_conflict_rate": float(actual_match[conflicting].mean())
                if conflicting.any()
                else None,
            }
            if method == "two_step":
                direct = a["direct_correct"][take]
                after = a["correct"][take]
                row.update(
                    {
                        "bridge_valid": int(a["bridge_valid"][take].sum()),
                        "bridge_correct": int(a["bridge_correct"][take].sum()),
                        "direct_only": int((direct & ~after).sum()),
                        "two_step_only": int((~direct & after).sum()),
                        "both_wrong": int((~direct & ~after).sum()),
                        "both_correct": int((direct & after).sum()),
                        "second_termination_error": int(
                            (a["bridge_valid"][take] & ~a["ended"][take]).sum()
                        ),
                    }
                )
            rows.append(row)
    return rows


def paired_rows(identity, old, high, chain, low_arrays, high_arrays, method):
    ids = old.derived_ids[chain]
    a = low_arrays["correct"][ids] if method == "direct" else low_arrays["correct"]
    b = high_arrays["correct"][ids] if method == "direct" else high_arrays["correct"]
    rows = []
    for split, pool in (
        ("all", ids),
        ("trained", old.train_ids[chain]),
        ("heldout", old.heldout_ids[chain]),
    ):
        for kind, cohort, mask in cohorts(old, high, chain, "low"):
            if kind != "fixed":
                continue
            take = mask & np.isin(ids, pool)
            low, hi = a[take], b[take]
            rows.append(
                {
                    **identity,
                    "method": method,
                    "chain": CHAINS[chain],
                    "split": split,
                    "cohort": cohort,
                    "n": int(take.sum()),
                    "low_correct": int(low.sum()),
                    "high_correct": int(hi.sum()),
                    "low_accuracy": float(low.mean()),
                    "high_accuracy": float(hi.mean()),
                    "high_minus_low": float(hi.mean() - low.mean()),
                    "both_correct": int((low & hi).sum()),
                    "low_only": int((low & ~hi).sum()),
                    "high_only": int((~low & hi).sum()),
                    "both_wrong": int((~low & ~hi).sum()),
                }
            )
    return rows


def macro_rows(rows):
    groups = defaultdict(dict)
    keys = ("world", "seed", "condition", "step", "phase", "method", "chain", "split")
    for row in rows:
        if row["cohort_kind"] == "phase_native":
            groups[tuple(row[key] for key in keys)][row["cohort"]] = row
    result = []
    for key, group in groups.items():
        if set(group) != {"ordinary", "exception"}:
            raise ValueError("Incomplete phase-native macro average")
        ordinary, exception = group["ordinary"], group["exception"]
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "cohort_definition": "phase_native_not_same_people",
                "ordinary_n": ordinary["n"],
                "exception_n": exception["n"],
                "ordinary_accuracy": ordinary["accuracy"],
                "exception_accuracy": exception["accuracy"],
                "balanced_accuracy": (ordinary["accuracy"] + exception["accuracy"]) / 2,
            }
        )
    return result


def organization_rows(rows):
    """Within-block crossover effects, then differences of those effects."""
    groups = defaultdict(dict)
    keys = ("world", "seed", "step", "phase", "method", "split", "cohort")
    for row in rows:
        if row["cohort_kind"] == "fixed":
            groups[tuple(row[key] for key in keys)][row["condition"], row["chain"]] = row[
                "accuracy"
            ]
    result, missing = [], []
    for key, values in groups.items():
        if len(values) != 6:
            missing.append(dict(zip(keys, key, strict=True)))
            continue
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "matching_effect": 0.5
                * (
                    values["company", "company"]
                    - values["project", "company"]
                    + values["project", "project"]
                    - values["company", "project"]
                ),
                **{
                    f"{org}_mean_benefit": 0.5
                    * sum(values[org, chain] - values["neither", chain] for chain in CHAINS)
                    for org in ("company", "project")
                },
            }
        )
    paired = defaultdict(dict)
    paired_keys = tuple(key for key in keys if key != "phase")
    for row in result:
        paired[tuple(row[key] for key in paired_keys)][row["phase"]] = row
    changes = []
    for key, phases in paired.items():
        if set(phases) != {"low", "high"}:
            continue
        changes.append(
            {
                **dict(zip(paired_keys, key, strict=True)),
                **{
                    f"high_minus_low_{metric}": phases["high"][metric] - phases["low"][metric]
                    for metric in (
                        "matching_effect",
                        "company_mean_benefit",
                        "project_mean_benefit",
                    )
                },
            }
        )
    return result, changes, missing


def audit_run(low_path, high_path, old, high, manipulation, selections, donors, ledger):
    low = read_json(low_path / "config.json", ledger)
    conf = read_json(high_path / "config.json", ledger)
    launch = read_json(high_path / "launch-contract.json", ledger)
    for key in (
        "world",
        "seed",
        "condition",
        "model",
        "initial_sha256",
        "documents_sha256",
        "qa_sha256",
        "prompts_sha256",
        "torch",
        "numpy",
        "precision",
    ):
        if low[key] != conf[key]:
            raise ValueError(f"Low/high control changed: {key}")
    if conf["world"] != old.seed or conf["model"]["width"] != 256:
        raise ValueError("Wrong world or width")
    if low["sources"] != source_hashes() or conf["sources"] != source_hashes():
        raise ValueError("Archived trainer changed")
    if (
        launch["sources"] != shortcut_sources()
        or conf["study"]["shortcut_contract"] != launch["sources"]
    ):
        raise ValueError("P3 adapter provenance changed")
    if launch["manipulation"] != manipulation:
        raise ValueError("Incorrect manipulation provenance")
    for key in ("world", "seed", "condition"):
        if launch[key] != conf[key]:
            raise ValueError(f"Launch identity mismatch: {key}")
    for key in (
        "width",
        "layers",
        "heads",
        "steps",
        "checkpoints",
        "lr",
        "documents_per_step",
        "facts_per_document",
        "QA_per_chain_per_step",
        "document_QA_weights",
    ):
        if low["study"][key] != conf["study"][key] or launch["study"][key] != conf["study"][key]:
            raise ValueError(f"Training hyperparameter changed: {key}")
    if conf["study"]["checkpoints"] != list(CHECKPOINTS) or conf["study"]["steps"] != 15360:
        raise ValueError("Wrong P3 learning checkpoints")
    if low["truth_sha256"] != array_hash(old.answers) or conf["truth_sha256"] != array_hash(
        high.answers
    ):
        raise ValueError("Wrong truth hash")
    saved = read_arrays(high_path / "manipulation.npz", ledger)
    expected = {
        "selections": selections,
        "donors": donors,
        "old_answers": old.answers,
        "answers": high.answers,
        "exceptions": high.exceptions,
        "train_ids": high.train_ids,
        "heldout_ids": high.heldout_ids,
    }
    for key, value in expected.items():
        np.testing.assert_array_equal(saved[key], value)
    low_schedule = read_arrays(low_path / "schedule.npz", ledger)
    high_schedule = read_arrays(high_path / "schedule.npz", ledger)
    expected_schedule = {
        "documents": documents(old, conf["condition"]),
        "qa": qa_schedule(old, 15360),
    }
    for key, value in expected_schedule.items():
        np.testing.assert_array_equal(low_schedule[key], value)
        np.testing.assert_array_equal(high_schedule[key], value)
        if conf[f"{key}_sha256"] != array_hash(value):
            raise ValueError(f"Schedule provenance mismatch: {key}")
    if conf["prompts_sha256"] != array_hash(old.prompts):
        raise ValueError("Prompt provenance mismatch")
    if list((high_path / "edits").glob("*")):
        raise ValueError("P3 learning-only output unexpectedly contains edits")
    return conf, expected_schedule


def write_report(output, audit, paired):
    lines = [
        "# P3：个人实际属性捷径可靠性干预",
        "",
        f"完成 {audit['complete_pairs']}/12 对训练、{audit['learning_pairs']}/72 对学习检查点、"
        f"{audit['two_step_pairs']}/24 对链级两步诊断。完整审计：{audit['complete']}。",
        "",
        "低/高比例分别为每组 2/32 和 16/32 例外；两条链同时改变。"
        "主要比较使用完全相同的 derived QA 问题、真值和人员 ID。每链固定三层："
        "原有例外 128 人、新增例外 896 人、仍为普通 1024 人；各层一半留出。",
        "",
        "| 组织 | 查询链 | 同人层 | 完整配对块 | 低比例直接答题 | 高比例直接答题 | 差值 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in paired:
        if row["step"] == 15360 and row["split"] == "heldout" and row["method"] == "direct":
            groups[row["condition"], row["chain"], row["cohort"]].append(row)
    for key, group in sorted(groups.items()):
        means = [
            np.mean([row[field] for row in group])
            for field in ("low_accuracy", "high_accuracy", "high_minus_low")
        ]
        lines.append(
            "| "
            + " | ".join(map(str, (*key, len(group))))
            + " | "
            + " | ".join(f"{100 * value:.2f}%" for value in means)
            + " |"
        )
    lines += [
        "",
        "paired-learning.csv 和 paired-two-step.csv 保留每个世界/初始化的配对计数；"
        "queries/ 保存逐查询低/高预测、EOS、真值、人员层和桥接结果。"
        "organization-effects.csv 与 organization-interactions.csv "
        "分别给组织交叉效应及其高减低变化。",
        "",
        "phase-native-macro.csv 的普通/例外宏平均使用各比例自己的分组，人员构成不同；"
        "它是单独的描述指标，不能替代固定同人层对比。actual_on_conflict 仅统计实际属性与"
        "derived QA 真值不同的查询；普通人答案重合不作为路径证据。",
        "",
        "自主两步使用模型预测的桥接实体。它衡量可访问的知识及外部组合效果，"
        "不能单独证明一次前向内部执行同一算法。两世界均为开发数据，四个配对块不等于四个独立世界。",
        "",
        "本阶段没有编辑案例；高比例世界不套用原 E93 更新。审计不读取模型权重，"
        "权重来源由冻结训练身份与两步记录关联，已明确记录这一审计边界。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def summarize(high_root, low_root, two_step_root, output, require_complete=False):
    high_root, low_root, two_step_root, output = map(
        Path, (high_root, low_root, two_step_root, output)
    )
    high_root = high_root.resolve()
    if (high_root / "width-256").exists():
        high_root /= "width-256"
    low_root, two_step_root, output = low_root.resolve(), two_step_root.resolve(), output.resolve()
    if (two_step_root / "width-256").exists():
        two_step_root /= "width-256"
    for source in (high_root, low_root, two_step_root):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Summary output must be separate from source directories")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "queries"
    raw.mkdir(exist_ok=True)
    ledger, missing, rows, pairs, metric_rows = {}, [], [], [], []
    counts = {"complete_pairs": 0, "learning_pairs": 0, "two_step_pairs": 0}
    worlds = {}
    for w in (0, 1):
        old = make_cross_world(w, ROOT / "data/bios-organization-v1")
        worlds[w] = (old, *high_exception_world(old))
    for w in (0, 1):
        old, high, manipulation, selections, donors = worlds[w]
        for seed in (0, 1):
            for condition in CONDITIONS:
                name = f"world-{w}-seed-{seed}-{condition}"
                low_path, high_path = low_root / name, high_root / name
                if not (high_path / "config.json").exists():
                    missing.append(str(high_path))
                    continue
                conf, schedule = audit_run(
                    low_path, high_path, old, high, manipulation, selections, donors, ledger
                )
                if (conf["world"], conf["seed"], conf["condition"]) != (w, seed, condition):
                    raise ValueError("Run path does not match its identity")
                timeline = {
                    point["step"]: point for point in read_json(high_path / "learning.json", ledger)
                }
                terminal = None
                local_steps, local_two = 0, 0
                for step in CHECKPOINTS:
                    files = [path / f"predictions-{step}.npz" for path in (low_path, high_path)]
                    if not all(path.exists() for path in files):
                        missing.extend(str(path) for path in files if not path.exists())
                        continue
                    low_a = score_direct(read_arrays(files[0], ledger), old.answers)
                    high_a = score_direct(read_arrays(files[1], ledger), high.answers)
                    expected = expected_exposure(
                        old, schedule["documents"], schedule["qa"], step, conf["study"]["lr"]
                    )
                    for key, value in zip(("exposure", "slots", "weighted"), expected, strict=True):
                        np.testing.assert_array_equal(low_a[key], high_a[key])
                        np.testing.assert_allclose(high_a[key], value, rtol=1e-11, atol=1e-12)
                    if step == 0:
                        for key in ("prediction", "ended"):
                            np.testing.assert_array_equal(low_a[key], high_a[key])
                    metrics = learning_metrics(high, high_a)
                    if step not in timeline:
                        raise ValueError("Saved prediction absent from learning timeline")
                    for key, value in metrics.items():
                        observed = timeline[step][key]
                        if (value is None and observed is not None) or (
                            value is not None and not np.isclose(value, observed)
                        ):
                            raise ValueError(f"Learning metric mismatch: {step}/{key}")
                    identity = {"world": w, "seed": seed, "condition": condition, "step": step}
                    for phase, world, arrays in (("low", old, low_a), ("high", high, high_a)):
                        metric_rows.append(
                            {**identity, "phase": phase, **learning_metrics(world, arrays)}
                        )
                        for chain in range(2):
                            rows.extend(
                                chain_rows(identity, old, high, chain, phase, arrays, "direct")
                            )
                    for chain in range(2):
                        pairs.extend(
                            paired_rows(identity, old, high, chain, low_a, high_a, "direct")
                        )
                    people_cohort = np.stack(
                        [
                            old.exceptions.astype(np.int8)[chain]
                            + 2 * (high.exceptions[chain] & ~old.exceptions[chain])
                            for chain in range(2)
                        ]
                    )
                    np.savez_compressed(
                        raw / f"{name}-learning-{step}.npz",
                        query_id=np.arange(len(old.answers)),
                        low_truth=old.answers,
                        high_truth=high.answers,
                        derived_ids=old.derived_ids,
                        person_cohort=people_cohort,
                        train_ids=old.train_ids,
                        heldout_ids=old.heldout_ids,
                        **{
                            f"{phase}_{key}": a[key]
                            for phase, a in (("low", low_a), ("high", high_a))
                            for key in ("prediction", "ended", "correct", "value_nll")
                        },
                    )
                    counts["learning_pairs"] += 1
                    local_steps += 1
                    if step == 15360:
                        terminal = low_a, high_a
                low_two = two_step_root / name / "learning-15360"
                high_complete = high_path / "complete.json"
                if (
                    terminal is not None
                    and high_complete.exists()
                    and (low_two / "complete.json").exists()
                ):
                    completion = read_json(high_complete, ledger)
                    if (
                        completion["status"] != "complete"
                        or completion["edit_cases"] != 0
                        or completion["learning_steps"] != 15360
                    ):
                        raise ValueError("P3 completion contract mismatch")
                    provenance = read_json(low_two / "complete.json", ledger)
                    if not provenance["complete"] or provenance["oracle_bridging"]:
                        raise ValueError("Invalid low-prevalence two-step record")
                    if Path(provenance["identity"]["run"]).resolve() != low_path:
                        raise ValueError("Low two-step source mismatch")
                    if (
                        provenance["identity"]["config_sha256"]
                        != ledger[str(low_path / "config.json")]
                    ):
                        raise ValueError("Low two-step configuration mismatch")
                    if (
                        provenance["identity"]["direct_predictions_sha256"]
                        != ledger[str(low_path / "predictions-15360.npz")]
                    ):
                        raise ValueError("Low two-step direct prediction mismatch")
                    if provenance["identity"]["answer_sha256"] != array_hash(old.answers):
                        raise ValueError("Low two-step scoring truth mismatch")
                    for chain, chain_name in enumerate(CHAINS):
                        a = score_two_step(
                            read_arrays(low_two / f"{chain_name}.npz", ledger),
                            old,
                            chain,
                            terminal[0],
                        )
                        b = score_two_step(
                            read_arrays(high_path / f"two-step-{chain_name}.npz", ledger),
                            high,
                            chain,
                            terminal[1],
                        )
                        identity = {"world": w, "seed": seed, "condition": condition, "step": 15360}
                        for phase, arrays in (("low", a), ("high", b)):
                            rows.extend(
                                chain_rows(identity, old, high, chain, phase, arrays, "two_step")
                            )
                        pairs.extend(paired_rows(identity, old, high, chain, a, b, "two_step"))
                        np.savez_compressed(
                            raw / f"{name}-two-step-{chain_name}.npz",
                            truth=old.answers[old.derived_ids[chain]],
                            **{
                                f"{phase}_{key}": value
                                for phase, arr in (("low", a), ("high", b))
                                for key, value in arr.items()
                            },
                        )
                        counts["two_step_pairs"] += 1
                        local_two += 1
                else:
                    missing.extend(
                        str(path)
                        for path in (high_complete, low_two / "complete.json")
                        if not path.exists()
                    )
                counts["complete_pairs"] += local_steps == 6 and local_two == 2
    native = macro_rows(rows)
    organization, interactions, incomplete_org = organization_rows(rows)
    complete = (
        counts == {"complete_pairs": 12, "learning_pairs": 72, "two_step_pairs": 24} and not missing
    )
    audit = {
        "complete": complete,
        **counts,
        "expected_pairs": 12,
        "missing": missing,
        "incomplete_organization_blocks": incomplete_org,
        "model_weights_read": False,
        "edit_cases": 0,
        "cohort_codes": {
            "0": "remaining_ordinary",
            "1": "original_exception",
            "2": "newly_exception",
        },
        "source_roots": {
            "high": str(high_root),
            "low": str(low_root),
            "low_two_step": str(two_step_root),
        },
    }
    for filename, data in (
        ("learning-strata.csv", [r for r in rows if r["method"] == "direct"]),
        ("two-step-strata.csv", [r for r in rows if r["method"] == "two_step"]),
        ("paired-learning.csv", [r for r in pairs if r["method"] == "direct"]),
        ("paired-two-step.csv", [r for r in pairs if r["method"] == "two_step"]),
        ("learning-metrics.csv", metric_rows),
        ("phase-native-macro.csv", native),
        ("organization-effects.csv", organization),
        ("organization-interactions.csv", interactions),
    ):
        write_csv(output / filename, data)
    write_json(output / "audit.json", audit)
    write_json(output / "sources.json", ledger)
    write_report(output, audit, pairs)
    if require_complete and not complete:
        raise ValueError(f"P3 matrix incomplete: {counts}, missing {len(missing)} artifacts")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-root", type=Path, default=MECHANISM / "p3")
    parser.add_argument(
        "--low-root", type=Path, default=ROOT / "results/bios-cross-scale-dev-v1/width-256"
    )
    parser.add_argument("--two-step-root", type=Path, default=MECHANISM / "p0/two-step")
    parser.add_argument("--output", type=Path, default=MECHANISM / "p3-summary")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                args.high_root,
                args.low_root,
                args.two_step_root,
                args.output,
                args.require_complete,
            )
        )
    )


if __name__ == "__main__":
    main()
