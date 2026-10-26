#!/usr/bin/env python3
"""CPU-only paired analysis of controlled coverage and natural-document application."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.knowledge_use_coverage_data import natural_retention_clusters

ARM_NAMES = ("A", "B", "C")
ATOMIC_SELECTION_GROUP = "atomic_seen/direct"
TASK_CATEGORIES = {
    "atomic_seen",
    "atomic_unseen",
    "seen_use_new_wording",
    "double_new_composition",
    "heldout_relation_composition",
    "heldout_use_category",
}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def select_development_node(batch, config):
    """Select only after exact configured PD development nodes are fully evaluated.

    No formal or natural artifact is opened. The original-question atomic metric
    is intentionally separate from held-out wording. This function performs no I/O
    writes and is safe for a controller to import without a GPU dependency.
    """
    batch = Path(batch)
    dev_nodes, formal_nodes = config["development_token_nodes"], config["formal_token_nodes"]
    result = {
        "status": "incomplete",
        "target_met": None,
        "selection_group": ATOMIC_SELECTION_GROUP,
        "selection_data": "development_only",
        "development_selected_tokens": None,
        "formal_selected_tokens": None,
        "rewrite_selected_tokens": None,
        "nodes": [],
        "problems": [],
    }
    if len(dev_nodes) != len(formal_nodes) or dev_nodes != sorted(set(dev_nodes)):
        result.update(
            status="invalid", problems=["development/formal node correspondence is invalid"]
        )
        return result
    target = config["calibration"]["atomic_recall_target"]
    result["target"] = target
    expected_n = 2 * config["development_pairs"]
    for node in dev_nodes:
        values = {}
        for arm in ARM_NAMES:
            path = (
                batch
                / "runs"
                / f"development-pd-{arm}-addition"
                / f"node-{node}"
                / "evaluation/summary.json"
            )
            summary = read(path)
            group = (summary or {}).get("groups", {}).get(ATOMIC_SELECTION_GROUP)
            if not summary or summary.get("complete") is not True or not group:
                values[arm] = None
                result["problems"].append(f"incomplete exact node: {path}")
                continue
            n, correct = group.get("n"), group.get("correct")
            accuracy = group.get("accuracy")
            if n != expected_n or not isinstance(correct, int) or not 0 <= correct <= n:
                result["status"] = "invalid"
                result["problems"].append(f"invalid atomic count: {path}")
                values[arm] = None
                continue
            if not isinstance(accuracy, (int, float)) or abs(accuracy - correct / n) > 1e-10:
                result["status"] = "invalid"
                result["problems"].append(f"inconsistent atomic accuracy: {path}")
                values[arm] = None
                continue
            values[arm] = {"n": n, "correct": correct, "accuracy": accuracy}
        result["nodes"].append({"tokens": node, "arms": values})
    if result["problems"]:
        return result
    qualified = [
        i
        for i, row in enumerate(result["nodes"])
        if all(row["arms"][a]["accuracy"] >= target for a in ARM_NAMES)
    ]
    index = qualified[0] if qualified else len(dev_nodes) - 1
    rewrite = formal_nodes[index] * config["rewrite_token_fraction"]
    if rewrite != int(rewrite):
        result.update(status="invalid", problems=["rewrite budget is not an integer token count"])
        return result
    result.update(
        status="complete",
        target_met=bool(qualified),
        development_selected_tokens=dev_nodes[index],
        formal_selected_tokens=formal_nodes[index],
        rewrite_selected_tokens=int(rewrite),
        selected_index=index,
    )
    return result


def row_key(row):
    """Mode is required because direct and self-generated-chain IDs intentionally coincide."""
    identity = re.sub(r"/(?:addition|rewrite)/(evaluation|retention)/", r"/state/\1/", row["id"])
    return identity, row.get("mode", row.get("variant", "direct"))


def cluster_id(row):
    if row.get("category") == "natural_old_retention":
        return row.get("cluster_id")
    return row.get("cluster_id") or row.get("case_id") or row.get("pair_id") or row["id"]


def _group(row):
    return f"{row['category']}/{row.get('mode', row.get('variant', 'direct'))}"


def cluster_ratio_ci(values, repeats=2000, seed=20261025):
    """Pairs (cluster, numerator, denominator), with entire clusters resampled.

    Unequal cluster sizes keep their observed row weights. Empty denominators are
    unknown, including bootstrap replicates lacking an initially-correct example.
    """
    aggregate = defaultdict(lambda: [0.0, 0.0])
    numerator_total, denominator_total, unknown = 0.0, 0.0, 0
    for cluster, numerator, denominator in values:
        numerator_total += numerator
        denominator_total += denominator
        if cluster is None:
            unknown += 1
            continue
        aggregate[str(cluster)][0] += numerator
        aggregate[str(cluster)][1] += denominator
    data = np.array(list(aggregate.values()), dtype=float)
    if unknown:
        return {
            "value": numerator_total / denominator_total if denominator_total else None,
            "ci95": None,
            "clusters": len(data),
            "bootstrap_valid": 0,
            "unknown_cluster_rows": unknown,
            "interval_status": "unknown_country_cluster; no IID fallback",
        }
    if not len(data) or data[:, 1].sum() == 0:
        return {"value": None, "ci95": None, "clusters": len(data), "bootstrap_valid": 0}
    value = float(data[:, 0].sum() / data[:, 1].sum())
    if repeats < 1:
        return {"value": value, "ci95": None, "clusters": len(data), "bootstrap_valid": 0}
    rng = np.random.default_rng(seed)
    sampled = data[rng.integers(0, len(data), (repeats, len(data)))].sum(axis=1)
    valid = sampled[:, 1] > 0
    ratios = sampled[valid, 0] / sampled[valid, 1]
    return {
        "value": value,
        "ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "clusters": len(data),
        "bootstrap_valid": int(valid.sum()),
    }


def summarize_rows(rows, repeats=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[_group(row)].append(row)
    result = {}
    for key, items in groups.items():
        valid = [r for r in items if isinstance(r.get("correct"), bool)]
        measure = cluster_ratio_ci([(cluster_id(r), int(r["correct"]), 1) for r in valid], repeats)
        result[key] = {
            "n": len(items),
            "valid_n": len(valid),
            "unknown_n": len(items) - len(valid),
            "correct_n": sum(r["correct"] for r in valid),
            "coverage": len(valid) / len(items),
            "accuracy": measure["value"],
            "accuracy_ci95": measure["ci95"],
            "clusters": measure["clusters"],
            "unknown_cluster_rows": measure.get("unknown_cluster_rows", 0),
            "unparsed_n": sum(r.get("parse_status") == "unparsed" for r in items),
            "truncated_n": sum(bool(r.get("truncated", False)) for r in items),
        }
    return result


def paired_statistics(before, after, repeats=2000):
    old = {row_key(r): r for r in before}
    groups = defaultdict(list)
    for row in after:
        groups[_group(row)].append(row)
    result = {}
    for key, items in groups.items():
        pairs = [
            (old[row_key(r)], r)
            for r in items
            if row_key(r) in old
            and isinstance(r.get("correct"), bool)
            and isinstance(old[row_key(r)].get("correct"), bool)
        ]
        gained = sum(not b["correct"] and a["correct"] for b, a in pairs)
        lost = sum(b["correct"] and not a["correct"] for b, a in pairs)
        initial = sum(b["correct"] for b, _ in pairs)
        initially_wrong = len(pairs) - initial
        delta = cluster_ratio_ci(
            [(cluster_id(a), int(a["correct"]) - int(b["correct"]), 1) for b, a in pairs], repeats
        )
        retention = cluster_ratio_ci(
            [
                (cluster_id(a), int(a["correct"] and b["correct"]), int(b["correct"]))
                for b, a in pairs
            ],
            repeats,
        )
        unchanged = all(
            b.get("gold") == a.get("gold") and b.get("question") == a.get("question")
            for b, a in pairs
        )
        result[key] = {
            "after_n": len(items),
            "paired_n": len(pairs),
            "unpaired_or_unknown_n": len(items) - len(pairs),
            "paired_coverage": len(pairs) / len(items),
            "gained_n": gained,
            "lost_n": lost,
            "gain_rate_among_initially_wrong": gained / initially_wrong
            if initially_wrong
            else None,
            "loss_rate_among_initially_correct": lost / initial if initial else None,
            "initially_correct_n": initial,
            "retained_n": initial - lost,
            "accuracy_difference": delta,
            "initially_correct_success": retention,
            "same_answer_preservation": unchanged,
            "interpretation": "paired unchanged-task gain/loss"
            if unchanged
            else "success at each knowledge state; changed answers are not preservation targets",
        }
    return result


def _load_controlled(directory, expected=None, require_chains=True):
    directory = Path(directory)
    summary = read(directory / "summary.json")
    predictions, chains = (
        read_rows(directory / "predictions.jsonl"),
        read_rows(directory / "chains.jsonl"),
    )
    old_rows = [r for r in predictions if r.get("category") == "natural_old_retention"]
    countries = natural_retention_clusters(old_rows)
    for row in old_rows:
        row["cluster_id"] = countries[row["id"]]
        row["cluster_status"] = "country_identified" if countries[row["id"]] else "unknown"
    problems = []
    if not summary or summary.get("complete") is not True:
        problems.append("evaluation summary missing or incomplete")
    if not (directory / "predictions.jsonl").exists():
        problems.append("predictions missing")
    if expected is None:
        problems.append("materialized evaluation/retention input contract missing")
    if any(not isinstance(r.get("correct"), bool) for r in predictions + chains):
        problems.append("controlled prediction correctness missing or invalid")
    keys = [row_key(r) for r in predictions + chains]
    if len(set(keys)) != len(keys):
        problems.append("duplicate prediction identity/mode")
    if expected is not None:
        missing = {row_key(r) for r in expected} - {row_key(r) for r in predictions}
        extra = {row_key(r) for r in predictions} - {row_key(r) for r in expected}
        if missing or extra:
            problems.append(
                f"prediction coverage differs: missing={len(missing)}, unexpected={len(extra)}"
            )
        expected_chains = {
            row_key(r)[0] for r in expected if r.get("chain") and r.get("mode") == "direct"
        }
        if require_chains and expected_chains != {row_key(r)[0] for r in chains}:
            problems.append("self-generated-chain coverage incomplete")
    return {
        "status": "incomplete" if problems else "complete",
        "problems": problems,
        "rows": predictions + chains,
        "summary": summary,
        "path": str(directory),
    }


def _load_natural(directory, expected_cases=None):
    directory = Path(directory)
    summary = read(directory / "summary.json")
    graded = read_rows(directory / "judge/grading.jsonl")
    retained = read_rows(directory / "retention.jsonl")
    retention_clusters = natural_retention_clusters(retained)
    records = []
    for r in graded:
        records.append(
            {
                **r,
                "id": f"{r['case_id']}/{r['group']}",
                "cluster_id": r["case_id"],
                "category": f"natural_{r['group']}",
                "mode": "direct",
                "correct": r.get("grade") if r.get("valid_grade") is True else None,
            }
        )
    for r in retained:
        records.append(
            {
                **r,
                "original_source_id": r.get("original_source_id", r["id"]),
                "cluster_id": retention_clusters[r["id"]],
                "cluster_status": "country_identified"
                if retention_clusters[r["id"]]
                else "unknown",
                "category": "natural_old_retention",
                "mode": "direct",
                "correct": r.get("strict"),
            }
        )
    problems = []
    if not summary or summary.get("complete") is not True:
        problems.append("natural evaluation summary missing or incomplete")
    if not graded or len(retained) != 317:
        problems.append("natural generation/judgment or 317-question retention incomplete")
    if any(not isinstance(r.get("strict"), bool) for r in retained):
        problems.append("natural retention correctness missing or invalid")
    if any(r.get("valid_grade") is True and not isinstance(r.get("grade"), bool) for r in graded):
        problems.append("natural valid judgment has no boolean grade")
    if expected_cases is not None:
        for group in ("closed", "oracle_context"):
            cases = {r["case_id"] for r in graded if r["group"] == group}
            if len(cases) != expected_cases:
                problems.append(f"natural {group} case coverage {len(cases)}/{expected_cases}")
    keys = [row_key(r) for r in records]
    if len(set(keys)) != len(keys):
        problems.append("duplicate natural case/group or retention identity")
    return {
        "status": "incomplete" if problems else "complete",
        "problems": problems,
        "rows": records,
        "summary": summary,
        "path": str(directory),
    }


def prerequisite_subsets(rows, baseline, source_present, repeats=2000):
    """Freeze background and source-present premises; retain full and conditional results."""
    background = {}
    for row in baseline:
        if row.get("mode") == "direct" and row["category"] == "retention_related_background":
            for fact in row["fact_ids"]:
                background[fact] = row.get("correct")
    atomic = {}
    for row in rows:
        if row.get("mode") == "direct" and row["category"] == "atomic_seen":
            for fact in row["fact_ids"]:
                atomic[fact] = row.get("correct")
    oracle = {
        (r["question"], r["gold"], r.get("mode", "direct")): r.get("correct")
        for r in source_present
    }
    eligible = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(lambda: {"total": 0, "observed": 0, "eligible": 0}))
    for row in rows:
        if row["category"] not in TASK_CATEGORIES:
            continue
        mode = "direct" if row.get("mode") == "self_generated_chain" else row.get("mode", "direct")
        b = [background.get(f) for f in row.get("background_fact_ids", [])]
        a = [atomic.get(f) for f in row.get("fact_ids", []) if not f.startswith("background/")]
        values = {
            "background": None if any(v is None for v in b) else all(b),
            "source_present": oracle.get((row["question"], row["gold"], mode)),
            "atomic": None if any(v is None for v in a) else all(a),
        }
        values["all_prerequisites"] = (
            None if any(v is None for v in values.values()) else all(values.values())
        )
        for name, value in values.items():
            group = counts[name][_group(row)]
            group["total"] += 1
            group["observed"] += int(value is not None)
            group["eligible"] += int(value is True)
            if value is True:
                eligible[name].append(row)
    result = {}
    for name, groups in counts.items():
        summaries = summarize_rows(eligible[name], repeats)
        result[name] = {
            key: {
                **n,
                "observed_coverage": n["observed"] / n["total"],
                "subset_coverage": n["eligible"] / n["total"],
                "conditional": summaries.get(key),
                "no_background_needed_is_eligible": name == "background",
            }
            for key, n in groups.items()
        }
    return result


def natural_prerequisite(rows, baseline, repeats=2000):
    oracle = {
        r["case_id"]: r.get("correct")
        for r in baseline
        if r["category"] == "natural_oracle_context"
    }
    closed = [r for r in rows if r["category"] == "natural_closed"]
    selected = [r for r in closed if oracle.get(r["case_id"]) is True]
    return {
        "total": len(closed),
        "observed": sum(oracle.get(r["case_id"]) is not None for r in closed),
        "eligible": len(selected),
        "subset_coverage": len(selected) / len(closed) if closed else None,
        "conditional": summarize_rows(selected, repeats),
        "premise": "released-base oracle-context semantic correctness; application only",
    }


def node_cost(run, node):
    """Use the exact node record, not a later cumulative endpoint or an estimate."""
    run = Path(run)
    learning = read(run / "learning.json", [])
    complete = read(run / "complete.json")
    matches = [r for r in learning if r.get("supervised_tokens") == node]
    if complete and complete.get("supervised_tokens") == node:
        matches.append(complete)
    exact = matches[-1] if matches else {}
    keys = (
        "step",
        "supervised_tokens",
        "replay_tokens",
        "examples",
        "training_seconds",
        "elapsed_total_seconds",
        "student_tokens",
        "teacher_tokens",
        "student_padded_tokens",
        "teacher_padded_tokens",
        "peak_cuda_bytes",
    )
    result = {key: exact.get(key) for key in keys}
    result.update(
        exact_node_observed=bool(matches),
        stage_complete=bool(complete) and complete.get("smoke") is False,
        training_failure=read(run / "failure.json"),
        total_flops=None,
        dedicated_gpu_hours=None,
        note="shared-GPU wall time is observed time, not exclusive throughput or FLOPs",
        cost_limitations=exact.get("cost_limitations"),
    )
    exposure = read(run / "exposure.json")
    if exposure is not None and complete and complete.get("supervised_tokens") == node:
        result["complete_counters"] = {
            k: v for k, v in exposure.items() if not k.startswith("main/")
        }
        result["fact_row_supervised_tokens"] = {
            k.removeprefix("main/"): v for k, v in exposure.items() if k.startswith("main/")
        }
    else:
        result["complete_counters"] = None
        result["fact_row_supervised_tokens"] = None
    result["unknown_fields"] = [key for key in keys if result[key] is None]
    return result


def resolved_run_config(config, cohort):
    """Use the runner's actual CPU-only resolver and fixed container mount paths."""
    if cohort != "natural":
        return config
    source = Path(__file__).with_name("run_knowledge_use_coverage_natural.py")
    spec = importlib.util.spec_from_file_location("coverage_natural_config", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.natural_config(config, Path("/factorial"))


def validate_run_contract(run, config, cohort, objective, arm, stage, nodes, parent_node=None):
    """Compare the submitted trajectory with the requested frozen scientific DAG."""
    run = Path(run)
    metadata = read(run / "run.json", {})
    parent = (
        f"/batch/runs/{cohort}-{objective}-{arm}-addition/node-{parent_node}/resume.pt"
        if stage == "rewrite"
        else None
    )
    expected = {
        "config": resolved_run_config(config, cohort),
        "cohort": cohort,
        "objective": objective,
        "arm": arm,
        "stage": stage,
        "nodes": nodes,
        "parent": parent,
    }
    observed = metadata.get("spec", {})
    problems = [
        f"run.json spec.{key} differs from requested resolved contract"
        for key, value in expected.items()
        if key not in observed or observed[key] != value
    ]
    for key, value in (
        ("model", config.get("model")),
        ("initial_model_sha256", config.get("model_revision")),
        ("tracking_group", config.get("batch")),
    ):
        if value is None or metadata.get(key) != value:
            problems.append(f"run.json {key} differs or is missing")
    parameters = metadata.get("parameters", {})
    names = parameters.get("trainable_names", [])
    trainable, total = (
        parameters.get("trainable_parameters", 0),
        parameters.get("total_parameters", 0),
    )
    if not names or not 0 < trainable <= total:
        problems.append("run.json trainable parameter contract missing or invalid")
    elif config.get("scope") == "down" and any(".mlp.down_proj." not in name for name in names):
        problems.append("run.json trainable tensor names differ from down scope")
    elif config.get("scope") == "full" and trainable != total:
        problems.append("run.json full scope does not train all parameters")
    manifest = run.parents[1] / "data" / cohort / "manifest.json"
    if (
        manifest.exists()
        and metadata.get("world_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest()
    ):
        problems.append("run.json materialized manifest hash differs")
    complete = read(run / "complete.json", {})
    if complete.get("nodes") != nodes or complete.get("supervised_tokens") != max(nodes):
        problems.append("complete.json nodes or final token count differs")
    if complete.get("smoke") is not False:
        problems.append("complete.json is missing or is an engineering smoke run")
    if (
        complete.get("parent_optimizer_reset") is not bool(parent)
        or complete.get("teacher_frozen") is not True
    ):
        problems.append("complete.json teacher or rewrite optimizer contract differs")
    return {
        "status": "incomplete" if problems else "complete",
        "problems": problems,
        "observed": metadata,
    }


def engineering_cost(batch):
    """Keep raw stage records; only exercise files record measured stage seconds."""
    root = Path(batch) / "engineering"
    records = [
        {"path": str(p), "stage": p.stem, "seconds": read(p).get("seconds"), "record": read(p)}
        for p in sorted(root.glob("**/*.json"))
        if p.name in {"exercise.json", "reload.json", "preflight.json", "complete.json"}
    ]
    measured = [
        r["seconds"]
        for r in records
        if r["stage"] == "exercise" and isinstance(r["seconds"], (int, float))
    ]
    return {
        "records": records,
        "observed_exercise_seconds": sum(measured) if measured else None,
        "total_engineering_seconds": None,
        "note": "reload/preflight have no independent timing; "
        "preflight update totals overlap exercises",
    }


def _expected(batch, cohort, stage):
    folder = Path(batch) / "data" / cohort
    paths = [folder / f"evaluation-{stage}.jsonl", folder / f"retention-{stage}.jsonl"]
    if not all(p.exists() for p in paths):
        return None
    return [row for p in paths for row in read_rows(p)]


def analyze(batch, config, repeats=2000):
    batch = Path(batch)
    selection = select_development_node(batch, config)
    report = {
        "analysis_version": "knowledge-use-coverage-v1",
        "status": "incomplete",
        "selection": selection,
        "development": {},
        "formal": {},
        "natural_application": {},
        "bootstrap": {
            "unit": "fact pair; rewrite uses two-pair swap block; natural uses case; "
            "old retention uses country",
            "repeats": repeats,
            "seed": 20261025,
            "interval": "paired cluster percentile 95%",
        },
        "failures": [],
        "missing": [],
        "abc_comparisons": {},
        "cost": {},
        "interpretation": {
            "natural": "previously used Hotpot cases; application validation, not new confirmation",
            "methods": "PD/SFT retain their sampling recipes; not a pure KL-versus-CE ablation",
            "scope": config.get("scope"),
            "scope_status": config.get("scope_status", "unknown"),
        },
    }
    raw, baselines, states = {}, {}, {}
    for cohort in ("development", "formal"):
        baseline = _load_controlled(
            batch / "baseline" / cohort / "closed", _expected(batch, cohort, "addition")
        )
        context = _load_controlled(
            batch / "baseline" / cohort / "context", _expected(batch, cohort, "addition"), False
        )
        rewritten = _load_controlled(
            batch / "baseline" / cohort / "rewrite-context",
            _expected(batch, cohort, "rewrite"),
            False,
        )
        baselines[cohort] = {"closed": baseline, "context": context, "rewrite-context": rewritten}
        report[cohort]["baselines"] = {}
        for label, data in baselines[cohort].items():
            report[cohort]["baselines"][label] = {k: v for k, v in data.items() if k != "rows"}
            report[cohort]["baselines"][label]["metrics"] = summarize_rows(data["rows"], repeats)
            if data["status"] != "complete":
                report["missing"].append(
                    {"item": f"baseline/{cohort}/{label}", "problems": data["problems"]}
                )
    natural_cases = config.get("natural_validation_cases", [32, 96])
    natural_n = natural_cases[1] - natural_cases[0]
    natural_base = _load_natural(batch / "baseline/natural", natural_n)
    report["natural_application"]["baseline"] = {
        k: v for k, v in natural_base.items() if k != "rows"
    }
    report["natural_application"]["baseline"]["metrics"] = summarize_rows(
        natural_base["rows"], repeats
    )
    if natural_base["status"] != "complete":
        report["missing"].append({"item": "baseline/natural", "problems": natural_base["problems"]})
    specs = [
        ("development", "pd", arm, "addition", node)
        for node in config["development_token_nodes"]
        for arm in ARM_NAMES
    ]
    if selection["status"] == "complete":
        for objective in ("pd", "sft"):
            for arm in ARM_NAMES:
                specs.extend(
                    [
                        ("formal", objective, arm, "addition", selection["formal_selected_tokens"]),
                        ("formal", objective, arm, "rewrite", selection["rewrite_selected_tokens"]),
                    ]
                )
        natural_node = selection["formal_selected_tokens"]
        if (
            config.get("natural_token_nodes", config["formal_token_nodes"])
            != config["formal_token_nodes"]
        ):
            report["missing"].append(
                {
                    "item": "natural budget contract",
                    "problems": [
                        "natural budgets must match formal; selection uses development only"
                    ],
                }
            )
        specs.extend(
            ("natural", objective, arm, "addition", natural_node)
            for objective in ("pd", "sft")
            for arm in ARM_NAMES
        )
    else:
        report["missing"].append(
            {"item": "formal/natural budget selection", "problems": selection["problems"]}
        )
    for cohort, objective, arm, stage, node in specs:
        name = f"{cohort}-{objective}-{arm}-{stage}"
        identity = f"{name}/node-{node}"
        run = batch / "runs" / name
        path = run / f"node-{node}"
        data = (
            _load_natural(path / "evaluation", natural_n)
            if cohort == "natural"
            else _load_controlled(path / "evaluation", _expected(batch, cohort, stage))
        )
        raw[identity] = data["rows"]
        reload_audit = read(path / "audit.json")
        cloud = read(path / "cloud-verification.json")
        tracking = read(batch / "tracking-wandb" / name / "state.json", {})
        expected_nodes = config["development_token_nodes"] if cohort == "development" else [node]
        contract = validate_run_contract(
            run,
            config,
            cohort,
            objective,
            arm,
            stage,
            expected_nodes,
            selection.get("formal_selected_tokens"),
        )
        entry = {k: v for k, v in data.items() if k != "rows"}
        entry.update(
            tokens=node,
            objective=objective,
            arm=arm,
            stage=stage,
            metrics=summarize_rows(data["rows"], repeats),
            reload_audit=reload_audit,
            cloud_verification=cloud,
            run_contract=contract,
            cost=node_cost(run, node),
        )
        before = (
            natural_base["rows"] if cohort == "natural" else baselines[cohort]["closed"]["rows"]
        )
        if stage == "rewrite":
            parent = (
                f"{cohort}-{objective}-{arm}-addition/node-{selection['formal_selected_tokens']}"
            )
            before = raw.get(parent, [])
            entry["paired_reference"] = parent
        else:
            entry["paired_reference"] = (
                f"baseline/{cohort}/closed" if cohort != "natural" else "baseline/natural"
            )
        entry["paired_change"] = paired_statistics(before, data["rows"], repeats)
        released = (
            natural_base["rows"] if cohort == "natural" else baselines[cohort]["closed"]["rows"]
        )
        entry["base_retention"] = paired_statistics(
            released,
            [
                r
                for r in data["rows"]
                if r["category"]
                in {"natural_old_retention", "retention_related_background", "retention_arithmetic"}
            ],
            repeats,
        )
        if cohort == "natural":
            entry["prerequisite_subsets"] = natural_prerequisite(data["rows"], before, repeats)
        else:
            oracle = baselines[cohort]["rewrite-context" if stage == "rewrite" else "context"][
                "rows"
            ]
            entry["prerequisite_subsets"] = prerequisite_subsets(
                data["rows"], baselines[cohort]["closed"]["rows"], oracle, repeats
            )
        if not reload_audit or reload_audit.get("passed") is not True:
            entry["status"] = "incomplete"
            entry["problems"].append("independent reload missing or failed")
        if (
            not cloud
            or not all(
                cloud.get(key) is True for key in ("passed", "config_equal", "node_summary_equal")
            )
            or cloud.get("state") != "finished"
        ):
            entry["status"] = "incomplete"
            entry["problems"].append("cloud node verification missing or incomplete")
        elif not tracking.get("run_id") or cloud.get("run_id") != tracking["run_id"]:
            entry["status"] = "incomplete"
            entry["problems"].append("cloud run identity differs or is missing")
        if not entry["cost"]["stage_complete"] or not entry["cost"]["exact_node_observed"]:
            entry["status"] = "incomplete"
            entry["problems"].append("training completion or exact node counters missing")
        if contract["status"] != "complete":
            entry["status"] = "incomplete"
            entry["problems"].extend(contract["problems"])
        states[identity] = entry["status"]
        section = "natural_application" if cohort == "natural" else cohort
        report[section].setdefault("runs", {})[identity] = entry
        if entry["status"] != "complete":
            report["missing"].append({"item": identity, "problems": entry["problems"]})
    matrices = sorted({(c, o, s, n) for c, o, _, s, n in specs})
    for cohort, objective, stage, node in matrices:
        for first, second in (("A", "B"), ("B", "C"), ("A", "C")):
            left = f"{cohort}-{objective}-{first}-{stage}/node-{node}"
            right = f"{cohort}-{objective}-{second}-{stage}/node-{node}"
            key = f"{cohort}/{objective}/{stage}/node-{node}/{second}-minus-{first}"
            report["abc_comparisons"][key] = {
                "status": "complete"
                if states.get(left) == states.get(right) == "complete"
                else "incomplete",
                "groups": paired_statistics(raw.get(left, []), raw.get(right, []), repeats),
            }
    for cohort in ("development", "formal", "natural"):
        report["cost"][cohort] = {
            "preparation": read(batch / "data" / cohort / "prepare-complete.json"),
            "allocation": "shared preparation; not silently charged once per arm",
        }
    report["cost"]["engineering"] = engineering_cost(batch)
    report["failures"] = [
        {"path": str(p), "failure": read(p)}
        for root in (batch / "runs", batch / "engineering", batch / "data", batch / "baseline")
        for p in sorted(root.glob("**/failure*.json"))
    ]
    report["status"] = (
        "complete" if not report["missing"] and not report["failures"] else "incomplete"
    )
    report["complete"] = report["status"] == "complete"
    report["completion_scope"] = (
        "local results, exact node counts, independent reload and cloud identity"
    )
    return report


def _percent(value):
    return "待测" if value is None else f"{100 * value:.2f}%"


def _difference(measure):
    if not measure or measure.get("value") is None:
        return "待测"
    ci = measure.get("ci95")
    interval = f" [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]" if ci else ""
    return f"{100 * measure['value']:+.2f}" + interval


def render_report(summary):
    lines = [
        "# 用途覆盖：收益、保持与成本",
        "",
        f"状态：`{summary['status']}`。缺失结果保留为未知，不记作零分。",
        "",
        "开发用于选择预算；正式受控比较与真实文档应用分别解释。PD/SFT保留各自采样配方。",
        "",
        "## 开发选择",
        "",
        "校准只用 `atomic_seen/direct`，不使用未练问法或正式结果。",
        "",
        f"选择状态：`{summary['selection']['status']}`；目标是否达到：`{summary['selection']['target_met']}`；"
        f"开发／正式／改写 token：{summary['selection']['development_selected_tokens']}／"
        f"{summary['selection']['formal_selected_tokens']}／{summary['selection']['rewrite_selected_tokens']}。",
        "",
    ]
    for section, title in (
        ("development", "开发全部节点"),
        ("formal", "正式受控比较"),
        ("natural_application", "真实文档应用"),
    ):
        if section == "natural_application":
            lines.extend(
                [
                    f"## {title}",
                    "",
                    "| 运行 | 状态 | 闭卷 | 原文在场 | 闭卷判分覆盖 |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            for name, run in summary[section].get("runs", {}).items():
                groups = run["metrics"]
                closed = groups.get("natural_closed/direct", {})
                oracle = groups.get("natural_oracle_context/direct", {})
                lines.append(
                    f"| {name} | {run['status']} | {_percent(closed.get('accuracy'))} | "
                    f"{_percent(oracle.get('accuracy'))} | {_percent(closed.get('coverage'))} |"
                )
            lines.extend(
                [
                    "",
                    "该64案例已在既有批次使用，作为应用验证；自然应用没有安排分步或自生成桥评价。",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                "| 运行 | 状态 | 原问法 | 未练问法 | 双新组合直接 | 分步 | 自生成桥 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, run in summary[section].get("runs", {}).items():
            groups = run["metrics"]
            metrics = [
                groups.get(k, {}).get("accuracy")
                for k in (
                    "atomic_seen/direct",
                    "atomic_unseen/direct",
                    "double_new_composition/direct",
                    "double_new_composition/step_by_step",
                    "double_new_composition/self_generated_chain",
                )
            ]
            lines.append(
                f"| {name} | {run['status']} | " + " | ".join(map(_percent, metrics)) + " |"
            )
        if not summary[section].get("runs"):
            lines.append(
                "| 尚未选择或未规划执行节点 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |"
            )
        lines.append("")
    lines.extend(
        [
            "## 保持与配对",
            "",
            "JSON逐组保存A→B、B→C、A→C的配对差值和聚类95%区间，以及基座原正确题的保持率。"
            "改写以前一阶段为参考；变化的依赖答案不当作应保持旧答案。location保持以前次写入终点为基线。",
            "",
            "背景、原文在场、原子掌握及联合前提子集均保留分母与覆盖率，完整结果始终同时保存。"
            "直接、显式分步、自生成桥链不混合。自然语义评分无效记录为未知，另外报有效覆盖率。",
            "旧知识317题按国家共同聚类；源ID无法识别时保留点估计，区间记未知，不按独立题替代。",
            "",
            f"缺失或不完整项：{len(summary['missing'])}；保留训练失败：{len(summary['failures'])}。",
            "",
            "完整分母、前提子集、配对区间、计数与缺口见 `summary.json`；"
            "完整性见 `validation.json`。",
            "",
        ]
    )
    lines.extend(
        [
            "| 配对比较 | 主用途差值及95%区间（百分点） | 配对问题数 | 独立块数 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, comparison in summary["abc_comparisons"].items():
        key = (
            "natural_closed/direct"
            if name.startswith("natural/")
            else "double_new_composition/direct"
        )
        metric = comparison["groups"].get(key, {})
        difference = metric.get("accuracy_difference", {})
        lines.append(
            f"| {name} | {_difference(difference)} | {metric.get('paired_n', '待测')} | "
            f"{difference.get('clusters', '待测')} |"
        )
    lines.append("")
    lines.extend(
        [
            "## 成本与完整性",
            "",
            "训练成本只读取精确token节点的learning/complete计数。输入、padding、教师、主监督、回放分列；"
            "缺少的FLOPs、独占GPU时间和中间节点exposure不估算补齐。共享生成准备不重复计入每臂。",
            "",
            "| 运行 | 基座原正确旧题保持 | 丢失旧题 | 主／回放监督token | "
            "学生／教师处理token | 训练秒 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for section in ("development", "formal", "natural_application"):
        for name, run in summary[section].get("runs", {}).items():
            old = run["base_retention"].get("natural_old_retention/direct", {})
            value = old.get("initially_correct_success", {}).get("value")
            c = run["cost"]
            lines.append(
                f"| {name} | {_percent(value)} | {old.get('lost_n', '待测')} | "
                f"{c['supervised_tokens']}/{c['replay_tokens']} | "
                f"{c['student_tokens']}/{c['teacher_tokens']} | {c['training_seconds']} |"
            )
    lines.extend(
        ["", "表内None表示该精确节点成本未知；重载、评测和共享生成时间另存，不混入训练秒。", ""]
    )
    return "\n".join(lines)


def write_outputs(batch, config_path, output=None, repeats=2000):
    batch, config_path = Path(batch), Path(config_path)
    output = Path(output) if output else batch
    summary = analyze(batch, read(config_path), repeats)
    output.mkdir(parents=True, exist_ok=True)
    validation = {
        "status": summary["status"],
        "complete": summary["complete"],
        "missing": summary["missing"],
        "failures": summary["failures"],
        "selection": summary["selection"],
        "gpu_used": False,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for name, value in (("summary.json", summary), ("validation.json", validation)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "report.md").write_text(render_report(summary))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = write_outputs(args.batch, args.config, args.output, args.bootstrap)
    print(
        json.dumps(
            {
                "status": result["status"],
                "missing": len(result["missing"]),
                "failures": len(result["failures"]),
                "selection": result["selection"]["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
