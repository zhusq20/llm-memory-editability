"""Same-bridge prefix interventions, with query-balanced donor summaries.

The donor selector never sees model scores. This evaluator reuses the audited
first-block component mixer; it does not retrain models or recompute the current
MLP after mixing. All candidate donors are evaluated, then averaged within the
recipient query before any world-level aggregation.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch

from .grok_loop_mechanism import COMPONENTS, evaluate_condition, executed_blocks

METRICS = (
    "answer_accuracy",
    "complete_accuracy",
    "eos_accuracy",
    "target_probability",
    "target_log_probability",
    "target_margin",
)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def prediction_metrics(prediction, targets):
    targets = np.asarray(targets, dtype=np.int64)
    logits = np.asarray(prediction["answer_logits"], dtype=np.float64)
    answer, stop = prediction["answer"], prediction["stop"]
    if logits.ndim != 2 or logits.shape[0] != len(targets):
        raise ValueError("Logits and targets must align")
    if answer.shape != targets.shape or stop.shape != targets.shape:
        raise ValueError("Generated answers and EOS must align with targets")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite answer logits")
    if len(targets) and (targets.min() < 0 or targets.max() >= logits.shape[1]):
        raise ValueError("Target outside vocabulary")
    maximum = logits.max(axis=1) if len(targets) else np.empty(0)
    partition = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
    target_logits = logits[np.arange(len(targets)), targets]
    competitors = logits.copy()
    competitors[np.arange(len(targets)), targets] = -np.inf
    log_probability = target_logits - partition
    return {
        "answer_accuracy": (answer == targets).astype(np.float64),
        "complete_accuracy": ((answer == targets) & (stop == 1)).astype(np.float64),
        "eos_accuracy": (stop == 1).astype(np.float64),
        "target_probability": np.exp(log_probability),
        "target_log_probability": log_probability,
        "target_margin": target_logits - competitors.max(axis=1),
    }


def query_mean(values, recipient_indices, n_recipients):
    """Missing recipients remain NaN; repeated donors never add query weight."""
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(recipient_indices, dtype=np.int64)
    if values.shape != indices.shape or not np.isfinite(values).all():
        raise ValueError("Finite, aligned one-dimensional donor values are required")
    if len(indices) and (indices.min() < 0 or indices.max() >= n_recipients):
        raise ValueError("Recipient index outside query pool")
    count = np.bincount(indices, minlength=n_recipients)
    sums = np.bincount(indices, weights=values, minlength=n_recipients)
    result = np.full(n_recipients, np.nan)
    np.divide(sums, count, out=result, where=count > 0)
    return result


def summarize(values, recipient_indices, selected, n_recipients):
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (n_recipients,):
        raise ValueError("Selection must align with the complete recipient pool")
    indices = np.asarray(recipient_indices, dtype=np.int64)
    n = int(selected.sum())
    metrics = {}
    available = np.zeros(n_recipients, dtype=bool)
    available[indices] = True
    mask = selected & available
    for name in METRICS:
        balanced = query_mean(values[name], indices, n_recipients)
        metrics[name] = float(balanced[mask].mean()) if mask.any() else None
    return {
        "n_recipients": int(mask.sum()),
        "selected_recipients": n,
        "total_recipients": n_recipients,
        "coverage": float(mask.mean()) if n_recipients else None,
        "n_donor_evaluations": int(selected[indices].sum()),
        **metrics,
    }


def _equal_predictions(left, right, label):
    for name in ("answer", "stop", "answer_logits"):
        if not np.array_equal(left[name], right[name]):
            raise AssertionError(f"{label}: {name} differs")


@torch.no_grad()
def evaluate_same_bridge(model, world, donors, *, device="cuda:0", batch_size=1024):
    started = time.perf_counter()
    model.eval()
    before = state_digest(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("All model parameters must be frozen")
    rows = donors["original_rows"]
    if not np.array_equal(rows, world["ood_composite"]) or rows.shape[1] != 4:
        raise ValueError("Expected the unchanged complete pure-OOD two-hop pool")
    n = len(rows)
    arrays = {k: v.copy() for k, v in donors.items() if isinstance(v, np.ndarray)}
    results, values, descriptions, engineering = {}, {}, {}, {}

    def evaluate(name, inputs, indices, **kwargs):
        prediction, audit = evaluate_condition(
            model, inputs, device, 4, batch_size=batch_size, **kwargs
        )
        indices = np.asarray(indices, dtype=np.int64)
        measured = prediction_metrics(prediction, inputs[:, -1])
        for field, vector in {**prediction, **measured}.items():
            arrays[name + "_" + field] = vector
        arrays[name + "_recipient_indices"] = indices
        results[name], values[name] = prediction, measured
        descriptions[name] = {
            "n_evaluations": len(inputs),
            "patch_layer": kwargs.get("patch_layer"),
            "component": kwargs.get("component"),
            "identity": kwargs.get("identity", False),
            "prefix_only": kwargs.get("donor_prefixes") is not None,
        }
        engineering[name] = audit
        return prediction

    all_indices = np.arange(n, dtype=np.int64)
    baseline = evaluate("baseline", rows, all_indices, verify_trace=True)
    for component in COMPONENTS:
        self_result = evaluate(
            "self_" + component,
            rows,
            all_indices,
            patch_layer=0,
            component=component,
            identity=True,
        )
        prefix_result = evaluate(
            "original_prefix_" + component,
            rows,
            all_indices,
            patch_layer=0,
            component=component,
            donor_prefixes=rows[:, :2],
        )
        _equal_predictions(baseline, self_result, "self " + component)
        _equal_predictions(baseline, prefix_result, "original prefix " + component)

    atomic_prediction, atomic_audit = evaluate_condition(
        model, world["atomic"], device, 4, batch_size=batch_size, verify_trace=True
    )
    arrays["atomic_rows"] = world["atomic"].copy()
    for field, vector in atomic_prediction.items():
        arrays["atomic_" + field] = vector
    atomic_complete = (atomic_prediction["answer"] == world["atomic"][:, -1]) & (
        atomic_prediction["stop"] == 1
    )
    arrays["atomic_complete_correct"] = atomic_complete
    arrays["original_atomic_complete_correct"] = atomic_complete[donors["original_atomic_indices"]]
    engineering["atomic_native"] = atomic_audit
    atom_lookup = {tuple(map(int, r)): i for i, r in enumerate(world["atomic"])}
    for family in ("id", "ood"):
        indices = donors[family + "_recipient_indices"]
        inputs = rows[indices]
        donor_rows = donors[family + "_donor_rows"]
        prefixes = donor_rows[:, :2]
        counterfactual = inputs.copy()
        counterfactual[:, 0] = donor_rows[:, 0]
        atomic_indices = np.asarray(
            [atom_lookup[tuple(map(int, r))] for r in donor_rows], dtype=np.int64
        )
        arrays[family + "_donor_atomic_indices"] = atomic_indices
        arrays[family + "_donor_atomic_complete_correct"] = atomic_complete[atomic_indices]
        matched = evaluate(family + "_baseline", inputs, indices)
        _equal_predictions(
            {k: v[indices] for k, v in baseline.items()}, matched, family + " batch match"
        )
        evaluate(family + "_counterfactual", counterfactual, indices)
        self_result = evaluate(
            family + "_self",
            inputs,
            indices,
            patch_layer=0,
            component="full",
            identity=True,
        )
        _equal_predictions(matched, self_result, family + " self")
        for component in COMPONENTS:
            evaluate(
                family + "_" + component,
                inputs,
                indices,
                patch_layer=0,
                component=component,
                donor_prefixes=prefixes,
            )
            if len(executed_blocks(model)) == 1:
                _equal_predictions(matched, results[family + "_" + component], "C1 inert patch")
        _equal_predictions(
            results[family + "_both"], results[family + "_full"], family + " both/full"
        )

    common = donors["common_valid"]
    groups = {
        "all": np.ones(n, dtype=bool),
        "paired": common,
        "id_only": (donors["id_candidate_count"] > 0) & (donors["ood_candidate_count"] == 0),
        "ood_only": (donors["id_candidate_count"] == 0) & (donors["ood_candidate_count"] > 0),
        "neither": (donors["id_candidate_count"] == 0) & (donors["ood_candidate_count"] == 0),
        "paired_original_atomics_correct": common
        & arrays["original_atomic_complete_correct"].all(axis=1),
        "paired_baseline_failed": common & (values["baseline"]["complete_accuracy"] == 0),
        "paired_baseline_correct": common & (values["baseline"]["complete_accuracy"] == 1),
    }
    # Conditional diagnostics are reported alongside the unfiltered paired pool.
    pid, pod = donors["pair_id_indices"], donors["pair_ood_indices"]
    pair_indices = donors["pair_recipient_indices"]
    known_pair = arrays["original_atomic_complete_correct"][pair_indices].all(axis=1)
    known_pair &= arrays["id_donor_atomic_complete_correct"][pid]
    known_pair &= arrays["ood_donor_atomic_complete_correct"][pod]
    cf_pair = values["id_counterfactual"]["complete_accuracy"][pid].astype(bool)
    cf_pair &= values["ood_counterfactual"]["complete_accuracy"][pod].astype(bool)
    arrays["pair_all_atomics_correct"] = known_pair
    arrays["pair_both_counterfactual_correct"] = cf_pair
    # A query is included in the following conditional subset only if every
    # registered donor pair satisfies the prerequisite, avoiding donor picking.
    for name, flag in (
        ("all_atomics_correct", known_pair),
        ("both_counterfactual_correct", cf_pair),
    ):
        average = query_mean(flag.astype(float), pair_indices, n)
        groups["paired_" + name] = common & (average == 1)
    for name, mask in groups.items():
        arrays["subset_" + name] = mask

    scores = {}
    for group, selected in groups.items():
        scores[group] = {
            condition: summarize(measured, arrays[condition + "_recipient_indices"], selected, n)
            for condition, measured in values.items()
        }
    contrasts = {}
    for component in COMPONENTS:
        id_values, ood_values = values["id_" + component], values["ood_" + component]
        difference = {
            metric: id_values[metric][pid] - ood_values[metric][pod] for metric in METRICS
        }
        adjusted = {
            metric: (id_values[metric][pid] - values["id_baseline"][metric][pid])
            - (ood_values[metric][pod] - values["ood_baseline"][metric][pod])
            for metric in METRICS
        }
        for group, mask in groups.items():
            contrasts[group + ":" + component] = {
                "id_minus_ood": summarize(difference, pair_indices, mask, n),
                "baseline_adjusted_id_minus_ood": summarize(adjusted, pair_indices, mask, n),
            }
    after = state_digest(model)
    if before != after:
        raise AssertionError("Interventions changed model parameters or buffers")
    return arrays, {
        "seconds": time.perf_counter() - started,
        "n_original_queries": n,
        "n_common_queries": int(common.sum()),
        "n_pairs": len(pair_indices),
        "conditions": descriptions,
        "scores": scores,
        "contrasts": contrasts,
        "engineering": {
            "native_and_prefix_checks": engineering,
            "all_self_and_original_prefix_conditions_equal_baseline": True,
            "both_equals_full_first_execution_layer": True,
            "c1_last_block_inert": len(executed_blocks(model)) == 1,
            "parameters_unchanged": True,
            "model_state_sha256_before": before,
            "model_state_sha256_after": after,
            "mlp_recomputed_after_mixing": False,
            "donor_nonpadding_tokens": 2,
            "patch_layer": 0,
            "patch_position": 1,
            "eos_uses_generated_answer": True,
            "training_updates": 0,
        },
        "statistical_unit": (
            "mean over all donors within recipient; initialization within world; equal worlds"
        ),
        "selection": "graph and registered training exposures only; scores do not select donors",
    }
