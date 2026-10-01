#!/usr/bin/env python3
"""Independently audit saved same-bridge endpoints without model forwards.

No production selector, evaluator or reporting function is imported. Original
checkpoint tensors are read on CPU solely to verify their serialized state
identity. Saved logits, graph rows and recipient denominators are recomputed.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/development-artifacts/grok-loop-same-bridge-v1"
METRICS = (
    "answer_accuracy",
    "complete_accuracy",
    "eos_accuracy",
    "target_probability",
    "target_log_probability",
    "target_margin",
)
COMPONENTS = ("attention", "mlp", "both", "full")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def metrics(logits, answer, stop, targets):
    values = np.asarray(logits, dtype=np.float64)
    # logaddexp reduction is independent of the execution scorer's max-shift sum.
    normalizer = np.logaddexp.reduce(values, axis=1)
    correct_logit = values[np.arange(len(targets)), targets]
    wrong = np.where(np.arange(values.shape[1])[None, :] == targets[:, None], -np.inf, values)
    logp = correct_logit - normalizer
    return {
        "answer_accuracy": (answer == targets).astype(float),
        "complete_accuracy": ((answer == targets) & (stop == 1)).astype(float),
        "eos_accuracy": (stop == 1).astype(float),
        "target_probability": np.exp(logp),
        "target_log_probability": logp,
        "target_margin": correct_logit - wrong.max(axis=1),
    }


def balanced(values, indices, mask):
    by_query = defaultdict(list)
    for index, value in zip(indices, values, strict=True):
        if mask[int(index)]:
            by_query[int(index)].append(float(value))
    return float(np.mean([np.mean(items) for items in by_query.values()])) if by_query else None


def main():
    started = time.perf_counter()
    import torch

    cfg_path = ROOT / "configs/grok-loop-same-bridge-v1.json"
    config = read(cfg_path)
    counters, errors, maximum = Counter(), [], defaultdict(float)
    inputs, run_results = {}, []
    recorded_gpu_evaluation_seconds = 0.0

    def check(value, label, run=None):
        counters[label] += 1
        if not bool(np.asarray(value).all()):
            errors.append({"run": run, "check": label})

    def equal(left, right, label, run=None, tolerance=0):
        left, right = np.asarray(left), np.asarray(right)
        if left.shape != right.shape:
            check(False, label + ":shape", run)
            return
        if left.dtype.kind in "fci" and right.dtype.kind in "fci":
            delta = (
                float(np.max(np.abs(left.astype(float) - right.astype(float))))
                if left.size
                else 0.0
            )
            maximum[label] = max(maximum[label], delta)
            check(delta <= tolerance, label, run)
        else:
            check(np.array_equal(left, right), label, run)

    for phase in ("development", "confirmation"):
        lock_path = ARTIFACT / config["locks"][phase]
        lock = read(lock_path)
        lock_hash = digest(lock_path)
        inputs[str(lock_path.relative_to(ROOT))] = lock_hash
        check(lock["config_sha256"] == digest(cfg_path), "config_lock_hash")
        check(lock["phase"] == phase, "lock_phase")
        check(len(lock["runs"]) == config["expected_runs"][phase], "locked_matrix_count")
        historical_cfg = read(ROOT / config["historical_configs"][phase])
        expected = {
            name
            for name, value in historical_cfg["runs"].items()
            if {**historical_cfg["base"], **value}["hops"] == 2
        }
        check({entry["run"] for entry in lock["runs"]} == expected, "registered_matrix_identity")
        actual_paths = ROOT / config["output_root"] / phase
        check(
            {path.name for path in actual_paths.iterdir() if path.is_dir()} == expected,
            "output_matrix_identity",
        )
        for name, expected_hash in lock["analysis_source_hashes"].items():
            check(digest(ROOT / name) == expected_hash, "frozen_execution_source_hash")
            check(
                digest(ARTIFACT / (lock_path.stem + "-source") / name) == expected_hash,
                "frozen_source_snapshot_hash",
            )
        for entry in lock["runs"]:
            run = entry["run"]
            source = Path(entry["directory"])
            output = actual_paths / run
            saved = arrays(output / "predictions.npz")
            report = read(output / "report.json")
            recorded_gpu_evaluation_seconds += report["seconds"]
            check(len(report["conditions"]) == 23, "all_23_conditions_present", run)
            complete = read(output / "complete.json")
            metadata = read(output / "metadata.json")
            donors = arrays(ROOT / entry["donors_file"])
            world = arrays(source / "world.npz")
            original = saved["original_rows"]
            n = len(original)
            common = saved["common_valid"]
            check(complete["passed"] and complete["run"] == run, "completion_identity", run)
            for item in (complete, metadata):
                check(item["lock_sha256"] == lock_hash, "output_lock_identity", run)
            for item in (metadata, report):
                check(item["spec"] == entry["spec"], "run_specification_identity", run)
            for name, expected_hash in entry["inputs"].items():
                actual_hash = digest(source / name)
                inputs[str((source / name).relative_to(ROOT))] = actual_hash
                check(actual_hash == expected_hash, "historical_input_hash", run)
            for name, expected_hash in complete["output_hashes"].items():
                actual_hash = digest(output / name)
                inputs[str((output / name).relative_to(ROOT))] = actual_hash
                check(actual_hash == expected_hash, "output_seal_hash", run)
            for field, hash_field in (
                ("donors_file", "donors_sha256"),
                ("donor_audit_file", "donor_audit_sha256"),
            ):
                actual_hash = digest(ROOT / entry[field])
                inputs[entry[field]] = actual_hash
                check(actual_hash == entry[hash_field], "frozen_donor_hash", run)
            for name, value in donors.items():
                equal(value, saved[name], "frozen_donor_array_identity", run)
            equal(original, world["ood_composite"], "unfiltered_recipient_identity", run)
            check(
                report["n_original_queries"] == n and report["n_common_queries"] == common.sum(),
                "reported_denominator_identity",
                run,
            )
            graph = {(int(h), int(r)): (int(t), j) for j, (h, r, t) in enumerate(world["atomic"])}
            ids = {tuple(map(int, row)) for row in world["id_atomic"]}
            first_position = saved["atomic_composite_actual_position_exposure"][:, 0]
            for i, (head, r1, r2, target) in enumerate(original):
                bridge, first = graph[int(head), int(r1)]
                tail, second = graph[bridge, int(r2)]
                check(tail == target, "recipient_graph_truth", run)
                equal(
                    saved["original_atomic_indices"][i],
                    [first, second],
                    "recipient_fact_indices",
                    run,
                )
                check(
                    tuple(world["atomic"][first]) not in ids
                    and tuple(world["atomic"][second]) not in ids,
                    "pure_ood_recipient",
                    run,
                )
                for family in ("id", "ood"):
                    candidates = []
                    for fact in sorted(map(tuple, world["atomic"])):
                        dh, dr, db = map(int, fact)
                        if dr != r1 or db != bridge or dh in (head, target):
                            continue
                        fact_index = graph[dh, dr][1]
                        eligible = (
                            fact in ids and first_position[fact_index] > 0
                            if family == "id"
                            else fact not in ids
                        )
                        if eligible:
                            candidates.append(fact)
                    offsets = saved[family + "_candidate_offsets"]
                    equal(
                        saved[family + "_donor_rows"][offsets[i] : offsets[i + 1]],
                        np.asarray(candidates, dtype=np.int64).reshape(-1, 3),
                        "all_eligible_donors_enumerated",
                        run,
                    )
            for family in ("id", "ood"):
                indices = saved[family + "_recipient_indices"]
                equal(
                    indices,
                    np.repeat(np.arange(n), saved[family + "_candidate_count"]),
                    "all_flat_recipient_indices",
                    run,
                )
                check(len(indices) == len(saved[family + "_donor_rows"]), "all_flat_length", run)
            equal(
                common,
                (saved["id_candidate_count"] > 0) & (saved["ood_candidate_count"] > 0),
                "graph_common_coverage",
                run,
            )
            pi, pid, pod = (
                saved[name]
                for name in ("pair_recipient_indices", "pair_id_indices", "pair_ood_indices")
            )
            equal(saved["id_recipient_indices"][pid], pi, "paired_id_index_identity", run)
            equal(saved["ood_recipient_indices"][pod], pi, "paired_ood_index_identity", run)
            equal(saved["pair_rows"], original[pi], "paired_recipient_row_identity", run)
            equal(
                np.bincount(pi, minlength=n),
                saved["id_candidate_count"] * saved["ood_candidate_count"],
                "complete_cartesian_pair_count",
                run,
            )
            equal(
                np.bincount(pi, weights=saved["pair_weights"], minlength=n),
                common.astype(float),
                "recipient_pair_weight_normalization",
                run,
                1e-14,
            )
            atom = world["atomic"]
            equal(
                saved["atomic_answer_logits"].argmax(axis=1),
                saved["atomic_answer"],
                "atomic_argmax_identity",
                run,
            )
            atomic_correct = (saved["atomic_answer"] == atom[:, -1]) & (saved["atomic_stop"] == 1)
            equal(atomic_correct, saved["atomic_complete_correct"], "atomic_complete_truth", run)
            equal(
                atomic_correct[saved["original_atomic_indices"]],
                saved["original_atomic_complete_correct"],
                "recipient_atomic_prerequisite_truth",
                run,
            )
            measured = {}
            for condition, description in report["conditions"].items():
                indices = saved[condition + "_recipient_indices"]
                expected_indices = (
                    saved[condition.split("_")[0] + "_recipient_indices"]
                    if condition.startswith(("id_", "ood_"))
                    else np.arange(n)
                )
                equal(indices, expected_indices, "condition_recipient_index_identity", run)
                length = len(indices)
                for field in ("answer", "stop", "answer_logits", *METRICS):
                    check(
                        len(saved[condition + "_" + field]) == length,
                        "physical_padding_absent_from_logical_arrays",
                        run,
                    )
                check(description["n_evaluations"] == length, "condition_evaluation_count", run)
                audit = report["engineering"]["native_and_prefix_checks"][condition]
                physical = min(n, config["batch_size"])
                discarded = (-length) % physical if length else 0
                check(
                    audit["physical_batch_size"] == physical
                    and audit["scored_evaluations"] == length
                    and audit["discarded_duplicate_padding"] == discarded,
                    "physical_padding_count",
                    run,
                )
                logits = saved[condition + "_answer_logits"]
                check(
                    logits.shape
                    == (length, 2 + entry["spec"]["entities"] + entry["spec"]["relations"]),
                    "full_vocabulary_logit_shape",
                    run,
                )
                check(np.isfinite(logits).all(), "finite_logits", run)
                equal(
                    logits.argmax(axis=1),
                    saved[condition + "_answer"],
                    "answer_argmax_identity",
                    run,
                )
                measured[condition] = metrics(
                    logits,
                    saved[condition + "_answer"],
                    saved[condition + "_stop"],
                    original[indices, -1],
                )
                for name, values in measured[condition].items():
                    equal(
                        values, saved[condition + "_" + name], "metric_vector_" + name, run, 2e-12
                    )
                for group, summary in report["scores"].items():
                    mask = saved["subset_" + group]
                    expected_n = len(set(map(int, indices[mask[indices]])))
                    check(
                        summary[condition]["n_recipients"] == expected_n,
                        "score_unique_recipient_denominator",
                        run,
                    )
                    check(
                        summary[condition]["n_donor_evaluations"] == mask[indices].sum(),
                        "score_donor_evaluation_denominator",
                        run,
                    )
                    for name, vector in measured[condition].items():
                        value = balanced(vector, indices, mask)
                        reported = summary[condition][name]
                        if value is None:
                            check(reported is None, "empty_score_is_none", run)
                        else:
                            equal(value, reported, "query_balanced_summary_" + name, run, 2e-12)
            for family in ("id", "ood"):
                indices = saved[family + "_recipient_indices"]
                for field in ("answer", "stop", "answer_logits"):
                    equal(
                        saved[family + "_baseline_" + field],
                        saved["baseline_" + field][indices],
                        "family_matched_baseline_bitwise",
                        run,
                    )
                    equal(
                        saved[family + "_self_" + field],
                        saved[family + "_baseline_" + field],
                        "family_self_bitwise",
                        run,
                    )
                    equal(
                        saved[family + "_both_" + field],
                        saved[family + "_full_" + field],
                        "first_layer_both_full_bitwise",
                        run,
                    )
                    if entry["spec"]["layers"] * entry["spec"]["repeats"] == 1:
                        for component in COMPONENTS:
                            equal(
                                saved[family + "_" + component + "_" + field],
                                saved[family + "_baseline_" + field],
                                "c1_patch_inert_bitwise",
                                run,
                            )
            for component in COMPONENTS:
                for condition in ("self_" + component, "original_prefix_" + component):
                    for field in ("answer", "stop", "answer_logits"):
                        equal(
                            saved[condition + "_" + field],
                            saved["baseline_" + field],
                            "self_original_prefix_baseline_bitwise",
                            run,
                        )
                for group in report["scores"]:
                    mask = saved["subset_" + group]
                    for name in METRICS:
                        delta = (
                            measured["id_" + component][name][pid]
                            - measured["ood_" + component][name][pod]
                        )
                        base_delta = (
                            measured["id_baseline"][name][pid] - measured["ood_baseline"][name][pod]
                        )
                        for contrast, vector in (
                            ("id_minus_ood", delta),
                            ("baseline_adjusted_id_minus_ood", delta - base_delta),
                        ):
                            value = balanced(vector, pi, mask)
                            reported = report["contrasts"][group + ":" + component][contrast][name]
                            if value is None:
                                check(reported is None, "empty_contrast_is_none", run)
                            else:
                                equal(value, reported, "paired_contrast_" + name, run, 2e-12)
            historical = arrays(source / f"predictions-{entry['spec']['steps']:07d}.npz")
            for field in ("answer", "stop"):
                equal(
                    saved["baseline_" + field],
                    historical["ood_composite_" + field],
                    "historical_prediction_identity",
                    run,
                )
            equal(
                -measured["baseline"]["target_log_probability"],
                historical["ood_composite_nll"][:, 0],
                "historical_answer_nll",
                run,
                3e-5,
            )
            weight_path = source / f"weights-{entry['spec']['steps']:07d}.pt"
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
            state_hash = hashlib.sha256()
            for name, tensor in sorted(checkpoint["model"].items()):
                value = tensor.numpy()
                state_hash.update(name.encode())
                state_hash.update(str(value.dtype).encode())
                state_hash.update(str(value.shape).encode())
                state_hash.update(np.ascontiguousarray(value).tobytes())
            for field in ("model_state_sha256_before", "model_state_sha256_after"):
                check(
                    report["engineering"][field] == state_hash.hexdigest(),
                    "original_model_state_identity",
                    run,
                )
            check(
                report["provenance"]["checkpoint_sha256"] == digest(weight_path),
                "reported_checkpoint_hash_identity",
                run,
            )
            prerequisites = atomic_correct[saved["original_atomic_indices"]].all(axis=1)
            known_pair = prerequisites[pi] & atomic_correct[saved["id_donor_atomic_indices"]][pid]
            known_pair &= atomic_correct[saved["ood_donor_atomic_indices"]][pod]
            equal(
                known_pair,
                saved["pair_all_atomics_correct"],
                "paired_atomic_prerequisite_truth",
                run,
            )
            cf_pair = measured["id_counterfactual"]["complete_accuracy"][pid].astype(bool)
            cf_pair &= measured["ood_counterfactual"]["complete_accuracy"][pod].astype(bool)
            equal(
                cf_pair,
                saved["pair_both_counterfactual_correct"],
                "paired_counterfactual_prerequisite_truth",
                run,
            )
            expected_subsets = {
                "all": np.ones(n, dtype=bool),
                "paired": common,
                "id_only": (saved["id_candidate_count"] > 0) & (saved["ood_candidate_count"] == 0),
                "ood_only": (saved["id_candidate_count"] == 0) & (saved["ood_candidate_count"] > 0),
                "neither": (saved["id_candidate_count"] == 0) & (saved["ood_candidate_count"] == 0),
                "paired_original_atomics_correct": common & prerequisites,
                "paired_baseline_failed": common & (measured["baseline"]["complete_accuracy"] == 0),
                "paired_baseline_correct": common
                & (measured["baseline"]["complete_accuracy"] == 1),
            }
            for name, flag in (
                ("all_atomics_correct", known_pair),
                ("both_counterfactual_correct", cf_pair),
            ):
                mask = np.zeros(n, dtype=bool)
                for recipient in np.flatnonzero(common):
                    mask[recipient] = flag[pi == recipient].all()
                expected_subsets["paired_" + name] = mask
            check(
                set(report["scores"]) == set(expected_subsets),
                "complete_predefined_subset_family",
                run,
            )
            for name, mask in expected_subsets.items():
                equal(saved["subset_" + name], mask, "independent_subset_mask_truth", run)
            summary = {
                "run": run,
                "phase": phase,
                "world": entry["spec"]["world_seed"],
                "initialization": entry["spec"]["initialization"],
                "architecture": entry["spec"]["architecture"],
                "all_n": n,
                "common_n": int(common.sum()),
                "pair_n": len(pi),
                "atomic_accuracy": float(atomic_correct.mean()),
                "common_original_atomics_accuracy": float(prerequisites[common].mean()),
                "all_donor_pair_atomics_accuracy": balanced(known_pair, pi, common),
                "both_counterfactual_pair_accuracy": balanced(cf_pair, pi, common),
                "conditions": {},
                "contrasts": {},
            }
            for component in ("full", "mlp"):
                summary["contrasts"][component] = {
                    name: balanced(
                        measured["id_" + component][name][pid]
                        - measured["ood_" + component][name][pod],
                        pi,
                        common,
                    )
                    for name in METRICS
                }
            for condition in (
                "baseline",
                "id_full",
                "ood_full",
                "id_mlp",
                "ood_mlp",
                "id_counterfactual",
                "ood_counterfactual",
            ):
                indices = saved[condition + "_recipient_indices"]
                summary["conditions"][condition] = {
                    name: balanced(vector, indices, common)
                    for name, vector in measured[condition].items()
                }
            run_results.append(summary)
    check(not torch.cuda.is_initialized(), "gpu_context_never_initialized")
    world_results = []
    for architecture in ("c1", "c2", "cd", "l1", "l2"):
        for world in config["followup_worlds"]:
            rows = [
                row
                for row in run_results
                if row["phase"] == "confirmation"
                and row["architecture"] == architecture
                and row["world"] == world
            ]
            check(len(rows) == 2, "world_initialization_matrix")
            world_results.append(
                {
                    "architecture": architecture,
                    "world": world,
                    "full_id_minus_ood_complete_pp": float(
                        np.mean([row["contrasts"]["full"]["complete_accuracy"] for row in rows])
                        * 100
                    ),
                    "mlp_id_minus_ood_complete_pp": float(
                        np.mean([row["contrasts"]["mlp"]["complete_accuracy"] for row in rows])
                        * 100
                    ),
                    "full_id_minus_ood_probability": float(
                        np.mean([row["contrasts"]["full"]["target_probability"] for row in rows])
                    ),
                    "mlp_id_minus_ood_probability": float(
                        np.mean([row["contrasts"]["mlp"]["target_probability"] for row in rows])
                    ),
                }
            )
    result = {
        "passed": not errors,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Independent graph/logaddexp/argmax/recipient reductions; no model forwards or training"
        ),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": digest(__file__),
        "config_sha256": digest(cfg_path),
        "expected_runs": 35,
        "audited_runs": len(run_results),
        "conditions_per_endpoint": 23,
        "audited_condition_n": sum(23 for _ in run_results),
        "audit_seconds": time.perf_counter() - started,
        "recorded_gpu_endpoint_evaluation_seconds_sum": recorded_gpu_evaluation_seconds,
        "new_model_forward_passes_in_this_audit": 0,
        "training_updates_in_this_audit": 0,
        "check_n": sum(counters.values()),
        "checks": dict(counters),
        "errors": errors,
        "maximum_absolute_errors": dict(maximum),
        "sources_sha256": inputs,
        "run_summaries": run_results,
        "world_paired_contrasts": world_results,
        "limitations": (
            "Saved-output audit verifies identities and calculations; "
            "it does not repeat GPU forward evaluation"
        ),
    }
    target = ARTIFACT / "independent-verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "runs": len(run_results),
                "checks": result["check_n"],
                "errors": errors[:20],
                "world_contrasts": world_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
