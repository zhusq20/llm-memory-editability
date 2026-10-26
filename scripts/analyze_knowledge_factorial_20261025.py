"""Read-only endpoint analysis of the frozen eight-arm Knowledge batch.

Re-running is safe while a remaining arm trains: incomplete arms never enter an
endpoint comparison. No model, CUDA, network, or W&B client is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from llm_memory_editability.knowledge_method_paper_eval import extract_grade

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = ROOT / "results/knowledge-method-batchsize-20261022/runtime/.."
DEFAULT_OUTPUT = ROOT / "results/knowledge-factorial-analysis-20261025"
ARMS = (
    "pd_lora",
    "sft_lora",
    "pd_full",
    "sft_full",
    "pd_down",
    "sft_down",
    "pd_down_lora",
    "sft_down_lora",
)
GROUPS = ("closed", "oracle_context")


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def divide(a, b):
    return a / b if b else None


def grade_state(value):
    return 0 if value is True else (1 if value is False else 2)


def paired_metrics(before, after, *, replicates=20000, seed=20261025, clusters=None):
    """Paired case bootstrap, with invalid grades kept as a separate state.

    Multinomial resampling of the nine paired outcome categories is exactly the
    empirical case bootstrap for these statistics, without expanding huge arrays.
    """
    if before.keys() != after.keys() or not before:
        raise ValueError("Nonempty and identical before/after case coverage required")
    counts = np.zeros((3, 3), dtype=np.int64)
    for key in sorted(before):
        counts[grade_state(before[key]), grade_state(after[key])] += 1
    n = len(before)
    b = counts.sum(axis=1)
    a = counts.sum(axis=0)
    bv, av = int(b[:2].sum()), int(a[:2].sum())
    retained, lost, new = int(counts[0, 0]), int(counts[0, 1]), int(counts[1, 0])
    result = {
        "n": n,
        "before": {
            "correct_n": int(b[0]),
            "valid_n": bv,
            "invalid_n": int(b[2]),
            "accuracy_valid": divide(int(b[0]), bv),
        },
        "after": {
            "correct_n": int(a[0]),
            "valid_n": av,
            "invalid_n": int(a[2]),
            "accuracy_valid": divide(int(a[0]), av),
        },
        "accuracy_change_pp": (100 * (a[0] / av - b[0] / bv)) if bv and av else None,
        "correct_fraction_all_change_pp": 100 * float(a[0] - b[0]) / n,
        "paired_valid_n": int(counts[:2, :2].sum()),
        "initially_correct_n": int(b[0]),
        "retained_n": retained,
        "retained_fraction": divide(retained, int(b[0])),
        "lost_to_false_n": lost,
        "initially_correct_after_invalid_n": int(counts[0, 2]),
        "initially_false_n": int(b[1]),
        "new_from_false_n": new,
        "new_from_false_fraction": divide(new, int(b[1])),
        "initially_false_after_invalid_n": int(counts[1, 2]),
        "net_gained_valid_transitions_n": new - lost,
        "transition_counts": counts.tolist(),
        "transition_order": ["true", "false", "invalid"],
    }
    if replicates <= 0:
        return result
    rng = np.random.default_rng(seed)
    if clusters is None:
        sampled = rng.multinomial(n, counts.ravel() / n, size=replicates)
        result["bootstrap_units_n"] = n
    else:
        if clusters.keys() != before.keys():
            raise ValueError("Cluster coverage differs from paired cases")
        names = sorted(set(clusters.values()))
        offsets = {name: i for i, name in enumerate(names)}
        grouped = np.zeros((len(names), 9), dtype=np.int64)
        for key in before:
            grouped[
                offsets[clusters[key]], 3 * grade_state(before[key]) + grade_state(after[key])
            ] += 1
        weights = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=replicates)
        sampled = weights @ grouped
        result["bootstrap_units_n"] = len(names)
    sampled = sampled.reshape(-1, 3, 3)
    sb, sa = sampled.sum(axis=2), sampled.sum(axis=1)

    def ratio(num, den):
        return np.divide(num, den, out=np.full(len(num), np.nan), where=den != 0)

    measures = {
        "accuracy_change_pp": 100
        * (ratio(sa[:, 0], sa[:, :2].sum(1)) - ratio(sb[:, 0], sb[:, :2].sum(1))),
        "correct_fraction_all_change_pp": 100 * ratio(sa[:, 0] - sb[:, 0], sa.sum(1)),
        "retained_fraction": ratio(sampled[:, 0, 0], sb[:, 0]),
        "new_from_false_fraction": ratio(sampled[:, 1, 0], sb[:, 1]),
    }
    result["ci95"] = {
        name: np.quantile(values[np.isfinite(values)], [0.025, 0.975]).tolist()
        if np.isfinite(values).any()
        else None
        for name, values in measures.items()
    }
    return result


def curve_seconds(records):
    """Sum logged process-clock segments, including discarded pre-resume work."""
    segments, previous = [], None
    for record in records:
        current = float(record["seconds_this_process"])
        if previous is not None and current < previous:
            segments.append(previous)
        previous = current
    if previous is not None:
        segments.append(previous)
    return {"seconds": sum(segments), "segments": segments}


def execution_seconds(directory):
    start = read_json(directory / "command.json", {})
    end = read_json(directory / "exit.json", {})
    if "utc" not in start or "utc" not in end:
        return None
    return (
        datetime.fromisoformat(end["utc"]) - datetime.fromisoformat(start["utc"])
    ).total_seconds()


def arm_cost(batch, arm, endpoint):
    original = batch / "runs" / arm
    continuation = batch / "batchsize-20261022/runs" / arm
    selection = read_json(continuation / "selection.json", {})
    manifest = read_json(original / "manifest.json", {})
    engineering = read_json(batch / f"engineering/results/exercise-{arm}.json", {})
    old_curve = list(rows(original / "curve.jsonl"))
    new_curve = list(rows(continuation / "curve.jsonl"))
    old_time, new_time = curve_seconds(old_curve), curve_seconds(new_curve)
    source_step = selection.get("source_step", 0)
    logged_old_step = old_curve[-1]["step"] if old_curve else 0
    audit = read_json(batch / f"batchsize-20261022/handoff/{arm}-post-stop-audit.json", {})
    recomputed = audit.get("observed_recomputed_updates", max(0, logged_old_step - source_step))
    source_record = next((r for r in reversed(old_curve) if r["step"] == source_step), {})
    last_old = old_curve[-1] if old_curve else {}
    probes = read_json(continuation / "benchmark.json", {}).get("measurements", [])
    clock_seconds = old_time["seconds"] + new_time["seconds"]
    return {
        "exposure": endpoint.get("exposure"),
        "target_exposure": selection.get("target_exposure", 1489200),
        "optimizer_updates": endpoint.get("optimizer_updates", endpoint.get("step")),
        "answer_tokens": endpoint.get("answer_tokens"),
        "forward_tokens_student_plus_teacher_if_pd": endpoint.get("forward_tokens"),
        "teacher_forward_tokens_separate": None if arm.startswith("pd_") else 0,
        "student_forward_tokens_separate": endpoint.get("forward_tokens")
        if arm.startswith("sft_")
        else None,
        "teacher_forward_sample_passes": endpoint.get("exposure") if arm.startswith("pd_") else 0,
        "teacher_mode": manifest.get(
            "teacher_mode",
            "adapter_disabled"
            if arm == "pd_down_lora"
            else "none"
            if arm.startswith("sft_")
            else None,
        ),
        "independent_teacher_parameters": manifest.get("independent_teacher_parameters", 0),
        "trainable_parameters": manifest.get(
            "trainable_parameters", engineering.get("parameters", {}).get("trainable_parameters")
        ),
        "parameter_count_source": "training_manifest"
        if manifest
        else "full_size_engineering_audit"
        if engineering
        else None,
        "source_optimizer_step": source_step,
        "source_exposure": selection.get("source_exposure"),
        "original_batch_size": selection.get("original_batch_size"),
        "continuation_batch_size": selection.get("batch_size"),
        "training_clock_hours_logged": clock_seconds / 3600,
        "training_clock_original": old_time,
        "training_clock_continuation": new_time,
        "clock_scope": "logged training-loop process clocks; shared GPU and I/O waits included; "
        "probe/model-load/final-save time excluded; unlogged tails unknown",
        "recomputed_updates_logged": recomputed,
        "recomputed_updates_unlogged_upper_bound": 99 if source_step else 0,
        "recomputed_tokens_logged": {
            key: last_old[key] - source_record[key]
            if key in last_old and key in source_record
            else 0
            for key in ("exposure", "answer_tokens", "forward_tokens")
        },
        "probe_successful_updates": sum(p.get("engineering_updates", 0) for p in probes),
        "probe_successful_timed_seconds": sum(p.get("seconds", 0) for p in probes),
        "probe_failed_candidates": [p["batch_size"] for p in probes if not p["passed"]],
        "probe_failed_candidate_compute": "unknown; may include updates before allocation failure",
        "probe_peak_reserved_mib": max((p.get("peak_reserved_mib", 0) for p in probes), default=0),
        "formal_training_peak_memory": None,
        "evaluation_execution_hours": divide(
            execution_seconds(batch / f"batchsize-20261022/execution/{arm}/evaluate"), 3600
        )
        if execution_seconds(batch / f"batchsize-20261022/execution/{arm}/evaluate") is not None
        else None,
        "flops": None,
        "exclusive_gpu_hours": None,
        "unknowns": [
            "FLOPs",
            "exclusive GPU hours",
            "formal-only peak memory",
            "unlogged process tails",
            "failed probe compute",
        ]
        + (["separate student/teacher forward token counts"] if arm.startswith("pd_") else []),
    }


def retention_clusters(records):
    """Keep all city/country/currency templates for one country in one block."""
    country_by_city = {
        row["id"].rsplit("/", 1)[-1]: row["gold"]
        for row in records
        if row["id"].startswith(("ret/capital_rev/", "ret/city_country/"))
    }
    result = {}
    for row in records:
        key = row["id"]
        entity = key.rsplit("/", 1)[-1]
        result[key] = entity if key.startswith("ret/currency/") else country_by_city[entity]
    return result


def collect_arm(batch, arm, replicates, seed):
    run = batch / "runs" / arm
    complete = read_json(run / "complete.json", {})
    audit = read_json(run / "audit/audit.json", {})
    evaluate = read_json(run / "paper-eval/evaluate.json", {})
    progress = complete or read_json(run / "progress.json", {})
    ready = (
        complete.get("exposure") == 1489200
        and not complete.get("smoke", False)
        and audit.get("passed") is True
        and evaluate.get("complete") is True
    )
    result = {
        "status": "complete" if ready else "incomplete",
        "training_complete": bool(complete),
        "reload_passed": audit.get("passed", False),
        "evaluation_complete": evaluate.get("complete", False),
        "cloud_terminal_verification": "separate audit; not asserted here",
        "cost": arm_cost(batch, arm, progress),
    }
    if not ready:
        return result, {}
    if complete["weights_sha256"] != evaluate["source"]["weights_sha256"]:
        raise ValueError(f"Evaluation/endpoint weight hash mismatch: {arm}")
    grading_path = run / "paper-eval/judge/grading.jsonl"
    grading_hash = digest(grading_path)
    if grading_hash != evaluate["judge"]["grading_sha256"]:
        raise ValueError(f"Grading hash mismatch: {arm}")
    observations = {group: {phase: {} for phase in ("before", "after")} for group in GROUPS}
    invalid = []
    baseline_answers = {}
    for row in rows(grading_path):
        parsed = extract_grade(row["judge_raw_output"])
        if parsed.get("grade") != row.get("grade") or parsed["valid_grade"] != row["valid_grade"]:
            raise ValueError(f"Raw judge/parser mismatch: {row['id']}")
        phase, group, case = row["phase"], row["group"], row["case_id"]
        bucket = observations[group][phase]
        if case in bucket:
            raise ValueError(f"Duplicate case/phase/group: {row['id']}")
        bucket[case] = parsed.get("grade")
        if phase == "before":
            baseline_answers[f"{group}/{case}"] = row["answer"]
        if not parsed["valid_grade"]:
            invalid.append({"id": row["id"], "phase": phase, "group": group, "case_id": case})
    result["groups"] = {}
    for group in GROUPS:
        pair = observations[group]
        if len(pair["before"]) != 1000 or len(pair["after"]) != 1000:
            raise ValueError(f"Incomplete 1000-case endpoint: {arm}/{group}")
        measured = paired_metrics(pair["before"], pair["after"], replicates=replicates, seed=seed)
        for phase in ("before", "after"):
            frozen = evaluate["summary"][group][phase]
            for key in ("correct_n", "valid_n", "invalid_n"):
                if measured[phase][key] != frozen[key]:
                    raise ValueError(f"Frozen summary mismatch: {arm}/{group}/{phase}/{key}")
        result["groups"][group] = measured
    auxiliary_path = run / "paper-eval/auxiliary.json"
    auxiliary = read_json(auxiliary_path)
    retention = {
        phase: {r["id"]: r["strict"] for r in auxiliary["retention"][phase]}
        for phase in ("before", "after")
    }
    clusters = retention_clusters(auxiliary["retention"]["before"])
    if len(retention["before"]) != 317 or len(retention["after"]) != 317:
        raise ValueError(f"Retention denominator differs: {arm}")
    result["old_retention"] = paired_metrics(
        **retention, replicates=replicates, seed=seed, clusters=clusters
    )
    result["old_retention"]["bootstrap_unit"] = (
        "country cluster; all related question templates together"
    )
    frozen_retention = evaluate["summary"]["retention"]
    if result["old_retention"]["retained_n"] != frozen_retention["retained_n"]:
        raise ValueError(f"Frozen retention mismatch: {arm}")
    observations["old_retention"] = {**retention, "clusters": clusters}
    result["invalid_judgments"] = invalid
    result["source"] = evaluate["source"]
    result["input_sha256"] = {
        "grading.jsonl": grading_hash,
        "auxiliary.json": digest(auxiliary_path),
        "complete.json": digest(run / "complete.json"),
        "evaluate.json": digest(run / "paper-eval/evaluate.json"),
    }
    result["baseline_answers_sha256"] = hashlib.sha256(
        json.dumps(baseline_answers, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return result, observations


def preparation_cost(batch):
    """Count shared generation once, including failed/retried raw requests."""
    result = {
        "scope": "new 968 cases; original 32-case generation is reused and not recharged",
        "shards": {},
        "generation": {},
    }
    for shard in sorted((batch / "preparation").glob("shard-*")):
        result["shards"][shard.name] = {
            "execution_hours": execution_seconds(batch / f"execution/{shard.name}/prepare-shard")
        }
        seconds = result["shards"][shard.name]["execution_hours"]
        result["shards"][shard.name]["execution_hours"] = seconds / 3600 if seconds else None
        for name in ("questions-raw.jsonl", "answers-raw.jsonl"):
            for record in rows(shard / "shared" / name):
                kind = record.get("kind", "questions")
                count = result["generation"].setdefault(
                    kind,
                    {
                        "requests": 0,
                        "failed_requests": 0,
                        "input_tokens": 0,
                        "generated_tokens": 0,
                        "missing_token_records": 0,
                    },
                )
                generation = record.get("generation", {})
                count["requests"] += 1
                count["failed_requests"] += bool(record.get("failed"))
                count["input_tokens"] += generation.get("input_tokens", 0)
                count["generated_tokens"] += generation.get("generated_tokens", 0)
                count["missing_token_records"] += not all(
                    key in generation for key in ("input_tokens", "generated_tokens")
                )
    result["shared_preparation_execution_hours_sum"] = sum(
        part["execution_hours"] or 0 for part in result["shards"].values()
    )
    result["scope_notes"] = [
        "Wall times include process startup and saving; not exclusive GPU hours",
        "Three shards overlap in calendar time; their sum is process-hours",
        "Original 32-case generation cost not newly measured",
        "Generation tokens do not include training forward tokens",
    ]
    return result


def audit_cloud(batch, arms, snapshot_path):
    """Check an independently fetched snapshot, without making network requests."""
    snapshot = read_json(snapshot_path, {}) if snapshot_path is not None else {}
    config = read_json(batch / "source/configs/knowledge-method-factorial-1000-v1.json", {})
    result = {
        "snapshot_utc": snapshot.get("utc"),
        "arms": {},
        "complete": False,
        "snapshot_sha256": digest(snapshot_path) if snapshot else None,
    }
    for arm, local in arms.items():
        stages = {}
        for stage in ("train", "paper-evaluate"):
            key = f"{arm}-{stage}"
            identity = read_json(batch / f"tracking/{key}.json", {})
            remote = snapshot.get("runs", {}).get(key, {})
            mismatches = []
            if remote:
                if not identity or remote.get("id") != identity.get("id"):
                    mismatches.append("run_id")
                if remote.get("url") != identity.get("url"):
                    mismatches.append("run_url")
                values = {
                    name: value.get("value")
                    if isinstance(value, dict) and "value" in value
                    else value
                    for name, value in remote.get("config", {}).items()
                }
                if not config:
                    mismatches.append("missing_frozen_config")
                mismatches.extend(
                    f"config/{name}" for name, value in config.items() if values.get(name) != value
                )
                counters = remote.get("summary", {})
                if local["status"] == "complete":
                    if stage == "train":
                        for name in ("exposure", "optimizer_updates", "answer_tokens"):
                            if counters.get(name) != local["cost"][name]:
                                mismatches.append(f"summary/{name}")
                        if (
                            counters.get("forward_tokens")
                            != local["cost"]["forward_tokens_student_plus_teacher_if_pd"]
                        ):
                            mismatches.append("summary/forward_tokens")
                    else:
                        for group in GROUPS:
                            for phase in ("before", "after"):
                                for name in ("correct_n", "valid_n", "invalid_n"):
                                    field = f"{group}.{phase}.{name}"
                                    if counters.get(field) != local["groups"][group][phase][name]:
                                        mismatches.append(f"summary/{field}")
                        if (
                            counters.get("retention.retained_n")
                            != local["old_retention"]["retained_n"]
                        ):
                            mismatches.append("summary/retention.retained_n")
            stages[stage] = {
                "state": remote.get("state", "missing"),
                "run_id": remote.get("id"),
                "url": remote.get("url"),
                "identity_and_config_match": bool(remote) and not mismatches,
                "mismatches": mismatches,
                "exposure": remote.get("summary", {}).get("exposure"),
            }
        verified = local["status"] == "complete" and all(
            value["state"] == "finished" and value["identity_and_config_match"]
            for value in stages.values()
        )
        result["arms"][arm] = {"terminal_verified": verified, "stages": stages}
    result["complete"] = len(result["arms"]) == len(ARMS) and all(
        arm["terminal_verified"] for arm in result["arms"].values()
    )
    return result


def analyze(
    batch, *, replicates=20000, seed=20261025, include_preparation=True, cloud_snapshot=None
):
    arms, observations = {}, {}
    for arm in ARMS:
        arms[arm], observations[arm] = collect_arm(batch, arm, replicates, seed)
    complete = [a for a in ARMS if arms[a]["status"] == "complete"]
    hashes = {arms[a]["baseline_answers_sha256"] for a in complete}
    if len(hashes) > 1:
        raise ValueError("Baseline generations differ across arms")
    for field in ("model_revision", "cases_sha256", "train_data_sha256", "retention_source_sha256"):
        if len({arms[a]["source"][field] for a in complete}) > 1:
            raise ValueError(f"Factorial source differs across arms: {field}")
    contrasts = {}
    for scope in ("lora", "full", "down", "down_lora"):
        pd_arm, sft_arm = f"pd_{scope}", f"sft_{scope}"
        key = f"pd_minus_sft/{scope}"
        if pd_arm not in complete or sft_arm not in complete:
            contrasts[key] = {"status": "incomplete"}
            continue
        contrasts[key] = {"status": "complete", "groups": {}}
        for group in (*GROUPS, "old_retention"):
            contrasts[key]["groups"][group] = paired_metrics(
                observations[sft_arm][group]["after"],
                observations[pd_arm][group]["after"],
                replicates=replicates,
                seed=seed,
                clusters=observations[sft_arm][group].get("clusters"),
            )
    cloud = audit_cloud(batch, arms, cloud_snapshot)
    complete_with_cloud = len(complete) == len(ARMS) and cloud["complete"]
    for arm in arms:
        arms[arm]["cloud_terminal_verification"] = cloud["arms"][arm]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "batch": "knowledge-method-factorial-1000-v1",
        "batch_path": str(batch.resolve()),
        "status": "complete" if complete_with_cloud else "incomplete",
        "complete": complete_with_cloud,
        "local_status": "complete" if len(complete) == len(ARMS) else "incomplete",
        "complete_arms": complete,
        "pending_arms": [a for a in ARMS if a not in complete],
        "pending_cloud_arms": [a for a in ARMS if not cloud["arms"][a]["terminal_verified"]],
        "cloud": cloud,
        "analysis": {
            "kind": "post-hoc descriptive endpoint analysis",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "bootstrap_unit": "paired HotpotQA case (1000); country cluster for 317 old questions",
            "interval": "percentile 95%; unadjusted for multiple comparisons",
            "inference": "conditional on these cases and one seed; excludes seed/task uncertainty",
            "invalid_handling": "true/(true+false); invalid separate; all-case score also reported",
        },
        "arms": arms,
        "contrasts": contrasts,
        "shared_preparation": preparation_cost(batch) if include_preparation else None,
    }, observations


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def change(metric):
    point = metric["accuracy_change_pp"]
    interval = metric.get("ci95", {}).get("accuracy_change_pp")
    if point is None:
        return "—"
    return f"{point:+.2f} [{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else f"{point:+.2f}"


def report(summary):
    lines = [
        "# Qwen 八组：终点收益、保持与成本",
        "",
        f"读取时间（UTC）：{summary['created_utc']}。状态：**{summary['status']}**；"
        f"训练、独立重载和终点评价齐全 {len(summary['complete_arms'])}/8。",
        "未完成组仅报告进度与已用成本，不用中途结果替代终点。",
        f"独立云端快照时点（UTC）：{summary['cloud']['snapshot_utc']}；"
        f"核对 run ID、冻结配置、终点计数及评价后，通过 "
        f"{sum(value['terminal_verified'] for value in summary['cloud']['arms'].values())}/8。"
        "云端与本地状态分别记录；旧快照不能替代最后一组完成后的新核验。",
        "",
        "## 收益与保持",
        "",
        "全量 HotpotQA 1000 案例；闭卷基座 25.60%，原文档在场基座 79.10%。"
        "原文档在场是 oracle-context 参照。表中变化单位为百分点，"
        "括号为配对案例 bootstrap 95% 区间。",
        "这批自然事实并未保证基础模型此前未知，前 32 案例已用于开发；因此不能视为全新知识确认集。",
        "",
        "| 组 | 闭卷终点 | 相对基座变化 [95% CI] | 原文档在场 |"
        " 相对基座变化 [95% CI] | 旧正确题保持 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, item in summary["arms"].items():
        if item["status"] != "complete":
            exposure = item["cost"].get("exposure") or 0
            lines.append(f"| {arm} | 未完成（曝光 {exposure / 1489200:.2%}） | — | — | — | — |")
            continue
        closed, oracle, old = (
            item["groups"]["closed"],
            item["groups"]["oracle_context"],
            item["old_retention"],
        )
        lines.append(
            f"| {arm} | {pct(closed['after']['accuracy_valid'])} | {change(closed)} | "
            f"{pct(oracle['after']['accuracy_valid'])} | {change(oracle)} | "
            f"{old['retained_n']}/{old['initially_correct_n']} ({pct(old['retained_fraction'])}) |"
        )
    lines += [
        "",
        "旧保持是另一个 317 题集合的原别名评分，基座 247 题正确；"
        "不能把这一保持率与 HotpotQA 原正确 256 题的保持率混用。",
        "",
        "### 闭卷收益从哪里来",
        "",
        "| 组 | 原正确保留 / 256 | 原正确→错 | 原正确→无效 | 原错→对 / 744 | 原错→无效 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["complete_arms"]:
        m = summary["arms"][arm]["groups"]["closed"]
        lines.append(
            f"| {arm} | {m['retained_n']} | {m['lost_to_false_n']} | "
            f"{m['initially_correct_after_invalid_n']} | {m['new_from_false_n']} | "
            f"{m['initially_false_after_invalid_n']} |"
        )
    lines += [
        "",
        "### 独立旧保持集合：保持与新增分开",
        "",
        "| 组 | 全量正确 / 317 | 原正确保留 / 247 | 保持率 95% CI | 原正确→错 | 原错→对 / 70 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["complete_arms"]:
        m = summary["arms"][arm]["old_retention"]
        interval = m["ci95"]["retained_fraction"]
        lines.append(
            f"| {arm} | {m['after']['correct_n']} | {m['retained_n']} | "
            f"[{pct(interval[0])}, {pct(interval[1])}] | "
            f"{m['lost_to_false_n']} | {m['new_from_false_n']} |"
        )
    lines += [
        "",
        "旧保持问句涉及相同国家、首都与货币的多个模板，不能当成 317 个独立事实。"
        "这里按 91 个国家簇重采样，相关单跳与多跳问法放在同一簇，"
        "同时保留原 317 题／原正确 247 题的评分分母。",
    ]
    lines += [
        "",
        "### 同一参数范围的 PD−SFT",
        "",
        "| 参数范围 | 闭卷差值 [95% CI] | 原文档在场差值 [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for key, contrast in summary["contrasts"].items():
        scope = key.split("/")[1]
        if contrast["status"] != "complete":
            lines.append(f"| {scope} | 待两组齐全 | — |")
        else:
            lines.append(
                f"| {scope} | {change(contrast['groups']['closed'])} | "
                f"{change(contrast['groups']['oracle_context'])} |"
            )
    lines += [
        "",
        "主评分保留有效判分分母；无效判断不自动记成错误。"
        "原始 true/false/invalid 转移、全案例分母敏感性结果与旧保持全量变化见 summary.json；"
        "各案例配对结果见 cases.csv。区间是事后描述性分析、未作多重比较校正，"
        "只表达给定这批案例和同一个训练种子的案例变动，不包括种子、数据生成或裁判不确定性。",
        "",
        "## 成本",
        "",
        "| 组 | 可训练参数(M) | batch 切换（源曝光） | 实际更新 | 答案 token(M) |"
        " 前向 token(M) | 训练计时(h) | 已记录重算更新 + 未记录上界 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, item in summary["arms"].items():
        c = item["cost"]

        def million(key, c=c):
            return f"{c[key] / 1e6:.3f}" if c.get(key) is not None else "—"

        lines.append(
            f"| {arm}{'（进行中）' if item['status'] != 'complete' else ''} | "
            f"{million('trainable_parameters')} | {c['original_batch_size']}→"
            f"{c['continuation_batch_size']}（{c['source_exposure']}） | "
            f"{c['optimizer_updates']} | {million('answer_tokens')} | "
            f"{million('forward_tokens_student_plus_teacher_if_pd')} | "
            f"{c['training_clock_hours_logged']:.2f} | {c['recomputed_updates_logged']} + "
            f"≤{c['recomputed_updates_unlogged_upper_bound']} |"
        )
    lines += [
        "",
        "终点均为 1,489,200 次样本曝光。答案 token 是实际 CE/KL 答案掩码计数；"
        "PD 前向 token 已含固定上下文教师，SFT 只有学生，两者不能当同计算量。"
        "教师 token 未单独记录，PD 额外教师样本前向次数等于曝光数；full/down 使用独立冻结教师，"
        "两种 LoRA 禁用 adapter 得到固定教师。r=1024 的原范围 LoRA 仍有约 12.73 亿可训练参数。",
        "",
        "前向 token 计数来自 attention mask（不含 padding）；collate 对整个 batch 删除最后一列，"
        "因此随 batch 改变存在少量计数差异。它不包含反向、重计算、padding 的实际算力，"
        "不能换称 FLOPs。",
        "",
        "训练计时相加原训练与接续曲线的进程时钟，包含已记录的废弃重算、共享 GPU 与存储等待，"
        "不含探测、模型载入和最终保存，不是独占 GPU 小时或公平速度测量。"
        "各组 batch、切换点和开始时刻不同，不能仅凭墙钟给出方法效率排名。"
        "重算按停机后曲线和 post-stop audit 复核；原训练尾部每组最多另有 99 次未记录更新，"
        "从基座直接启动的两组 down-LoRA 没有旧训练重算。",
        "",
        "工程试跑成功更新、失败候选、已计时部分和峰值在 summary.json 单列。"
        "失败候选已发生多少计算、纯正式训练显存峰值、精确教师耗时、FLOPs 与独占 GPU 小时均未知；"
        "不把包含工程试跑的累计 CUDA 峰值写成正式训练峰值。",
    ]
    prep = summary.get("shared_preparation")
    if prep:
        lines += [
            "",
            f"新增 968 案例的数据准备合计 {prep['shared_preparation_execution_hours_sum']:.2f} "
            "进程小时（三片重叠运行），作为八组共享成本计一次；旧 32 案例数据复用。",
            "",
            "| 生成部分 | 原始请求（含重试） | 失败请求 | 输入 token | 生成 token |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for kind, count in prep["generation"].items():
            lines.append(
                f"| {kind} | {count['requests']} | {count['failed_requests']} | "
                f"{count['input_tokens']} | {count['generated_tokens']} |"
            )
    complete_items = [summary["arms"][arm] for arm in summary["complete_arms"]]
    improved = sum(item["groups"]["closed"]["accuracy_change_pp"] > 0 for item in complete_items)
    oracle_losses = sum(
        item["groups"]["oracle_context"]["accuracy_change_pp"] < 0 for item in complete_items
    )
    old_losses = sum(item["old_retention"]["retained_fraction"] < 1 for item in complete_items)
    lines += [
        "",
        "## 目前能作出的判断",
        "",
        f"已完成 {len(complete_items)} 组中，{improved} 组闭卷多跳分数高于共同基座；"
        f"{oracle_losses} 组原文档在场准确率下降，{old_losses} 组存在原正确旧题丢失。"
        "闭卷净增并非原有正确题全部保住后增加：新答对与旧答错同时发生，必须一起报告。",
        "",
        "这里报告各组点估计及配对变化；最终八组矩阵尚未齐全时，结论保持临时状态。"
        "即便八组齐全，"
        "单 seed、不同 batch 接续、PD/SFT 的回答温度和答案不同仍限制普遍方法排名，"
        "PD−SFT 不能归因成纯 KL−CE 效应。用途覆盖比较应据原独立契约推进，"
        "不能根据这张表事后重选或删掉八组。",
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/analyze_knowledge_factorial_20261025.py",
        "```",
        "",
        "脚本仅读持久批次并写本目录。最后一组训练、重载、终点评价完成后运行同一命令，"
        "并刷新独立 cloud-snapshot.json；本地八组与云端八组均通过后，"
        "summary.complete 才为 true、status 才为 complete。不会启动训练或更改冻结产物。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--cloud-snapshot", type=Path)
    args = parser.parse_args()
    summary, observations = analyze(
        args.batch,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
        cloud_snapshot=args.cloud_snapshot or args.output / "cloud-snapshot.json",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "report.md").write_text(report(summary))
    with (args.output / "cases.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "group", "case_id", "before", "after"])
        for arm in summary["complete_arms"]:
            for group, pair in observations[arm].items():
                for case in sorted(pair["before"]):
                    writer.writerow([arm, group, case, pair["before"][case], pair["after"][case]])
    print(
        json.dumps(
            {
                "status": summary["status"],
                "complete_arms": summary["complete_arms"],
                "pending_arms": summary["pending_arms"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
