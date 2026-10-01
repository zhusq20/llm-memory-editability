"""Prefix-only component interventions at every executed Transformer block.

The recipient block runs normally before mixing cached attention and MLP
increments at r1. The current block's MLP is never recomputed after mixing.
Only later blocks consume the intervention. Donors are graph-selected before
evaluating the model and receive only [head, r1] followed by fixed PAD tokens.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from numbers import Integral

import numpy as np
import torch
from torch.nn import functional as F

from .grok_multihop import pack_rows

COMPONENTS = ("attention", "mlp", "both", "full")
SPLITS = ("test_composite", "test_full_composite", "ood_composite")
TRACE_FIELDS = ("input", "attention", "mlp", "residual")


def _choose(candidates, world_hash, seed, row, family):
    if not candidates:
        return None
    material = json.dumps([world_hash, int(seed), list(map(int, row)), family])
    index = int.from_bytes(hashlib.sha256(material.encode()).digest(), "big") % len(candidates)
    return sorted(candidates)[index]


def _path(head, relations, graph):
    edges = []
    for relation in relations:
        fact = graph.get((int(head), int(relation)))
        if fact is None:
            return None
        head, index = fact
        edges.append(index)
    return int(head), tuple(edges)


def select_donors(world, seed, split="test_composite"):
    """Select same-r1 donors using only truth and split membership.

    ID recipients and every donor-path fact are restricted to ID facts. OOD
    recipients and donor paths analogously use only OOD facts. Different-bridge
    paths must change the target and their complete queries must be untrained.
    Same-bridge paths may be trained; their split is saved for separate scoring.
    Rows without donors remain present with explicit masks and reasons.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rows = world[split]
    n, hops = len(rows), rows.shape[1] - 2
    if not 2 <= hops <= 4:
        raise ValueError("Interventions require two-, three- or four-hop rows")
    atom_index = {tuple(map(int, row)): i for i, row in enumerate(world["atomic"])}
    atomic_split = "ood_atomic" if split == "ood_composite" else "id_atomic"
    graph = {
        (int(h), int(r)): (int(t), atom_index[int(h), int(r), int(t)])
        for h, r, t in world[atomic_split]
    }
    by_relation = defaultdict(list)
    for (head, relation), (tail, _) in graph.items():
        by_relation[relation].append((head, relation, tail))
    training = {tuple(map(int, row[:-1])) for row in world["train_composite"]}
    reserved = {tuple(map(int, row[:-1])) for row in world["test_full_composite"]}
    result = {
        "split": np.asarray(split),
        "atomic_split": np.asarray(atomic_split),
        "original_rows": rows.copy(),
        "original_bridge": np.full(n, -1, dtype=np.int64),
        "original_atomic_indices": np.full((n, hops), -1, dtype=np.int64),
    }
    for family in ("different", "same"):
        result.update(
            {
                family + "_donor": np.full((n, 3), -1, dtype=np.int64),
                family + "_counterfactual_rows": np.full_like(rows, -1),
                family + "_atomic_indices": np.full((n, hops), -1, dtype=np.int64),
                family + "_valid": np.zeros(n, dtype=bool),
                family + "_candidate_count": np.zeros(n, dtype=np.int64),
                family + "_reason": np.full(n, "no_eligible_same_relation_path", dtype="U64"),
                family + "_counterfactual_split": np.full(n, "missing", dtype="U16"),
            }
        )
    reasons = (
        "same_head",
        "same_bridge",
        "missing_successor",
        "same_target",
        "head_is_answer",
        "trained_counterfactual",
    )
    for reason in reasons:
        result["different_rejected_" + reason] = np.zeros(n, dtype=np.int64)
    for i, row in enumerate(rows):
        head, r1, target = int(row[0]), int(row[1]), int(row[-1])
        original = _path(head, row[1:-1], graph)
        if original is None or original[0] != target:
            raise ValueError(f"Recipient is not a true all-{atomic_split} path")
        bridge = graph[head, r1][0]
        result["original_bridge"][i] = bridge
        result["original_atomic_indices"][i] = original[1]
        candidates = {"different": [], "same": []}
        for dh, dr, db in by_relation[r1]:
            continuation = _path(dh, row[1:-1], graph)
            if continuation is None:
                reason = "missing_successor"
            else:
                dt, edges = continuation
                query = (dh, *map(int, row[1:-1]))
                if dh != head and db == bridge and dh != target:
                    candidates["same"].append((dh, dr, db, dt, edges))
                reason = (
                    "same_head"
                    if dh == head
                    else "same_bridge"
                    if db == bridge
                    else "same_target"
                    if dt == target
                    else "head_is_answer"
                    if dh in (target, dt)
                    else "trained_counterfactual"
                    if query in training
                    else None
                )
                if reason is None:
                    candidates["different"].append((dh, dr, db, dt, edges))
            if reason:
                result["different_rejected_" + reason][i] += 1
        for family in ("different", "same"):
            choices = candidates[family]
            result[family + "_candidate_count"][i] = len(choices)
            donor = _choose(choices, world["metadata"]["dataset_sha256"], seed, row, family)
            if donor is None:
                continue
            dh, dr, db, dt, edges = donor
            cf = (dh, *map(int, row[1:-1]), dt)
            result[family + "_donor"][i] = dh, dr, db
            result[family + "_counterfactual_rows"][i] = cf
            result[family + "_atomic_indices"][i] = edges
            result[family + "_valid"][i] = True
            result[family + "_reason"][i] = "eligible"
            result[family + "_counterfactual_split"][i] = (
                "ood"
                if split == "ood_composite"
                else "train"
                if cf[:-1] in training
                else "reserved_test"
                if cf[:-1] in reserved
                else "unused"
            )
    return result


def executed_blocks(model):
    return list(model.iter_blocks()) if hasattr(model, "iter_blocks") else list(model.blocks)


@torch.no_grad()
def traced_forward(
    model,
    tokens,
    positions=None,
    *,
    patch_layer=None,
    component="both",
    donor_trace=None,
    identity=False,
    return_trace=False,
):
    """Run eval forward, optionally mixing one executed layer's r1 state.

    Trace fields contain only position r1, with shape [batch, width]. For a
    component patch the output is (recipient input + mixed A) + mixed M. Full
    replacement also changes the incoming residual, so it equals replacing A
    and M only at execution layer zero for same-r1 donors, not at later layers.
    """
    if model.training:
        raise ValueError("Interventions require model.eval()")
    blocks = executed_blocks(model)
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}")
    if patch_layer is not None and (
        isinstance(patch_layer, bool)
        or not isinstance(patch_layer, Integral)
        or not 0 <= patch_layer < len(blocks)
    ):
        raise ValueError("patch_layer must index an executed block")
    if identity and donor_trace is not None:
        raise ValueError("Identity and donor patches are mutually exclusive")
    if (donor_trace is not None or identity) != (patch_layer is not None):
        raise ValueError("A patch layer requires exactly one donor or identity patch")
    if donor_trace is not None and len(donor_trace) != len(blocks):
        raise ValueError("Donor trace must cover every executed layer")
    if tokens.shape[1] < 2:
        raise ValueError("The r1 position requires at least two tokens")
    x = model.token(tokens) + model.position(torch.arange(tokens.shape[1], device=tokens.device))
    trace = []
    for layer, block in enumerate(blocks):
        z = block.ln1(x)
        batch, length, width = z.shape
        attention = block.attention
        q, k, v = (
            attention.qkv(z)
            .view(batch, length, 3, attention.heads, width // attention.heads)
            .unbind(2)
        )
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
        )
        a = attention.proj(a.transpose(1, 2).reshape(batch, length, width))
        m = block.mlp(block.ln2(x + a))
        residual = (x + a) + m
        current = {
            "input": x[:, 1].clone(),
            "attention": a[:, 1].clone(),
            "mlp": m[:, 1].clone(),
            "residual": residual[:, 1].clone(),
        }
        if return_trace:
            trace.append(current)
        if layer == patch_layer:
            donor = current if identity else donor_trace[layer]
            if any(donor[key].shape != current[key].shape for key in TRACE_FIELDS):
                raise ValueError("Donor trace has incompatible batch or width")
            if component == "full":
                replacement = donor["residual"]
            else:
                mixed_a = donor["attention"] if component in ("attention", "both") else a[:, 1]
                mixed_m = donor["mlp"] if component in ("mlp", "both") else m[:, 1]
                replacement = (x[:, 1] + mixed_a) + mixed_m
            residual = residual.clone()
            residual[:, 1] = replacement
        x = residual
    x = model.ln_final(x)
    if positions is not None:
        x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
    logits = F.linear(x, model.token.weight)
    return (logits, trace) if return_trace else logits


@torch.no_grad()
def prefix_trace(model, prefixes, padded_length):
    """Evaluate only two supplied tokens and a fixed zero suffix; never a target."""
    if prefixes.ndim != 2 or prefixes.shape[1] != 2 or padded_length < 4:
        raise ValueError("Need two-token prefixes and the fixed hops+2 sequence length")
    tokens = torch.zeros((len(prefixes), padded_length), dtype=torch.long, device=prefixes.device)
    tokens[:, :2] = prefixes
    positions = torch.ones((len(prefixes), 1), dtype=torch.long, device=prefixes.device)
    return traced_forward(model, tokens, positions, return_trace=True)[1]


@torch.no_grad()
def evaluate_condition(
    model,
    rows,
    device,
    padded_length,
    *,
    donor_prefixes=None,
    patch_layer=None,
    component="both",
    identity=False,
    batch_size=1024,
    verify_trace=False,
):
    """Generate answer then EOS, reapplying exactly the same donor intervention."""
    if donor_prefixes is not None and donor_prefixes.shape != (len(rows), 2):
        raise ValueError("Donor prefixes must align one-to-one with recipient rows")
    answers, stops, logits_saved = [], [], []
    max_delta = prefix_delta = 0.0
    packed = pack_rows(rows, padded_length)
    for start in range(0, len(rows), batch_size):
        x, pos, _ = (torch.as_tensor(a[start : start + batch_size], device=device) for a in packed)
        x = x.clone()
        donor = None
        if donor_prefixes is not None:
            prefixes = torch.as_tensor(donor_prefixes[start : start + batch_size], device=device)
            donor = prefix_trace(model, prefixes, padded_length)
        arguments = dict(
            patch_layer=patch_layer,
            component=component,
            donor_trace=donor,
            identity=identity,
        )
        logits, trace = traced_forward(model, x, pos, return_trace=True, **arguments)
        if verify_trace:
            if patch_layer is not None:
                raise ValueError("Native-forward verification applies to baseline only")
            reference = model(x, pos)
            max_delta = max(max_delta, float((reference - logits).abs().max()))
            if not torch.equal(reference.argmax(-1), logits.argmax(-1)):
                raise AssertionError("Traced forward changes native argmax")
            prefix = prefix_trace(model, x[:, :2], padded_length)
            for full, truncated in zip(trace, prefix, strict=True):
                for field in TRACE_FIELDS:
                    prefix_delta = max(
                        prefix_delta, float((full[field] - truncated[field]).abs().max())
                    )
        answer = logits[:, 0].argmax(-1)
        logits_saved.append(logits[:, 0].cpu().numpy())
        x[torch.arange(len(x), device=device), pos[:, 1]] = answer
        continuation = traced_forward(model, x, pos, **arguments)
        if verify_trace:
            reference = model(x, pos)
            max_delta = max(max_delta, float((reference - continuation).abs().max()))
            if not torch.equal(reference.argmax(-1), continuation.argmax(-1)):
                raise AssertionError("Generated-answer trace changes native argmax")
        answers.append(answer.cpu().numpy())
        stops.append(continuation[:, 1].argmax(-1).cpu().numpy())
    if max_delta > 1e-5 or prefix_delta > 1e-5:
        raise AssertionError(f"Native/prefix trace disagreement: {max_delta}, {prefix_delta}")
    return {
        "answer": np.concatenate(answers) if answers else np.empty(0, dtype=np.int64),
        "stop": np.concatenate(stops) if stops else np.empty(0, dtype=np.int64),
        "answer_logits": np.concatenate(logits_saved)
        if logits_saved
        else np.empty((0, model.token.num_embeddings), dtype=np.float32),
    }, {
        "native_comparison_performed": verify_trace,
        "native_max_logit_delta": max_delta,
        "prefix_only_vs_full_all_layers_max_delta": prefix_delta,
    }


def _metrics(arrays, condition, selected, original, target, baseline):
    mask = selected & arrays[condition + "_valid"]
    n, total = int(mask.sum()), len(mask)
    answer, stop = arrays[condition + "_answer"], arrays[condition + "_stop"]

    def mean(value):
        return float(value[mask].mean()) if n else None

    return {
        "n": n,
        "total_n": total,
        "coverage": n / total if total else None,
        "target_answer_accuracy": mean(answer == target),
        "target_complete_accuracy": mean((answer == target) & (stop == 1)),
        "original_answer_accuracy": mean(answer == original),
        "original_complete_accuracy": mean((answer == original) & (stop == 1)),
        "other_answer_rate": mean((answer != target) & (answer != original)),
        "eos_accuracy": mean(stop == 1),
        "answer_change_rate": mean(answer != baseline),
    }


def evaluate_run(model, world, donors, device="cuda:0", batch_size=1024):
    """Evaluate every executed layer, preserving all rows and prerequisite masks."""
    started = time.perf_counter()
    model.eval()
    split = str(donors["split"].item())
    rows = world[split]
    if not np.array_equal(rows, donors["original_rows"]):
        raise ValueError("Donor rows differ from the registered evaluation split")
    n, padded_length = len(rows), int(world["metadata"]["hops"]) + 2
    arrays = {key: value.copy() for key, value in donors.items()}
    all_rows = np.ones(n, dtype=bool)
    conditions, audits = {}, {}

    def evaluate(name, mask, inputs, *, family=None, **kwargs):
        result, audit = evaluate_condition(
            model, inputs[mask], device, padded_length, batch_size=batch_size, **kwargs
        )
        for key in ("answer", "stop"):
            values = np.full(n, -1, dtype=np.int64)
            values[mask] = result[key]
            arrays[name + "_" + key] = values
        arrays[name + "_valid"] = mask.copy()
        targets = rows[:, -1] if family is None else donors[family + "_counterfactual_rows"][:, -1]
        for suffix, labels in (("original_logit", rows[:, -1]), ("target_logit", targets)):
            values = np.full(n, np.nan, dtype=np.float32)
            values[mask] = result["answer_logits"][np.arange(mask.sum()), labels[mask]]
            arrays[name + "_" + suffix] = values
        conditions[name] = {
            "family": family,
            **{k: v for k, v in kwargs.items() if k in ("patch_layer", "component", "identity")},
        }
        audits[name] = audit

    evaluate("baseline", all_rows, rows, verify_trace=True)
    for family in ("different", "same"):
        valid = donors[family + "_valid"]
        # Match batch shape as well as sequence shape for floating-point controls.
        evaluate(family + "_matched_baseline", valid, rows, family=family)
        evaluate(
            family + "_counterfactual_input",
            valid,
            donors[family + "_counterfactual_rows"],
            family=family,
        )
    blocks = executed_blocks(model)
    for layer in range(len(blocks)):
        for component in COMPONENTS:
            identity_name = f"e{layer:03d}_identity_{component}"
            evaluate(
                identity_name, all_rows, rows, patch_layer=layer, component=component, identity=True
            )
            for key in ("answer", "stop", "original_logit", "target_logit"):
                if not np.array_equal(arrays[identity_name + "_" + key], arrays["baseline_" + key]):
                    raise AssertionError(f"Identity patch changes {identity_name}.{key}")
            for family in ("different", "same"):
                valid = donors[family + "_valid"]
                name = f"e{layer:03d}_{family}_{component}"
                evaluate(
                    name,
                    valid,
                    rows,
                    family=family,
                    patch_layer=layer,
                    component=component,
                    donor_prefixes=donors[family + "_donor"][valid, :2],
                )
                if layer == len(blocks) - 1:
                    for key in ("answer", "stop", "original_logit"):
                        if not np.array_equal(
                            arrays[name + "_" + key][valid],
                            arrays[family + "_matched_baseline_" + key][valid],
                        ):
                            raise AssertionError("Last-block r1 patch changes later answer/EOS")
        if layer == 0:
            for family in ("different", "same"):
                valid = donors[family + "_valid"]
                for key in ("answer", "stop", "original_logit", "target_logit"):
                    left = arrays[f"e000_{family}_both_{key}"][valid]
                    right = arrays[f"e000_{family}_full_{key}"][valid]
                    if not np.array_equal(left, right):
                        raise AssertionError("First-layer same-r1 both/full differ")
    atomics, _ = evaluate_condition(
        model, world["atomic"], device, padded_length, batch_size=batch_size
    )
    atom_correct = (atomics["answer"] == world["atomic"][:, -1]) & (atomics["stop"] == 1)
    arrays["atomic_rows"] = world["atomic"].copy()
    arrays["atomic_answer"], arrays["atomic_stop"] = atomics["answer"], atomics["stop"]
    groups = {"all_rows": all_rows}
    prerequisites = {}
    for family in ("original", "different", "same"):
        valid = all_rows if family == "original" else donors[family + "_valid"]
        indices = donors[family + "_atomic_indices"]
        correct = np.zeros_like(indices, dtype=bool)
        correct[valid] = atom_correct[indices[valid]]
        arrays[family + "_atomic_complete_correct"] = correct
        complete = valid & correct.all(axis=1)
        groups[family + "_all_atomic_correct"] = complete
        prerequisites[family] = {
            "n": int(valid.sum()),
            "total_n": n,
            "coverage": float(valid.mean()) if n else None,
            "all_hops_complete_accuracy": float(complete[valid].mean()) if valid.any() else None,
            "per_hop_complete_accuracy": correct[valid].mean(axis=0).tolist()
            if valid.any()
            else None,
        }
    for family in ("different", "same"):
        valid = donors[family + "_valid"]
        baseline_correct = (arrays[family + "_matched_baseline_answer"] == rows[:, -1]) & (
            arrays[family + "_matched_baseline_stop"] == 1
        )
        groups[family + "_donor_available"] = valid
        groups[family + "_donor_missing"] = ~valid
        cf = family + "_counterfactual_input"
        cf_correct = valid & (
            arrays[cf + "_answer"] == donors[family + "_counterfactual_rows"][:, -1]
        )
        cf_correct &= arrays[cf + "_stop"] == 1
        groups[family + "_native_original_and_counterfactual_correct"] = (
            baseline_correct & cf_correct
        )
        for category in ("train", "reserved_test", "unused", "ood"):
            groups[family + "_counterfactual_" + category] = valid & (
                donors[family + "_counterfactual_split"] == category
            )
    scores = {}
    for group, mask in groups.items():
        arrays["subset_" + group] = mask.copy()
        scored = {}
        for condition, description in conditions.items():
            family = description["family"]
            target = (
                rows[:, -1] if family is None else donors[family + "_counterfactual_rows"][:, -1]
            )
            scored[condition] = _metrics(
                arrays,
                condition,
                mask,
                rows[:, -1],
                target,
                arrays[(family + "_matched_baseline" if family else "baseline") + "_answer"],
            )
        scores[group] = {
            "n": int(mask.sum()),
            "total_n": n,
            "coverage": float(mask.mean()) if n else None,
            "conditions": scored,
        }
    return {
        "split": split,
        "atomic_split": str(donors["atomic_split"].item()),
        "sample_rule": "training-independent fixed probe"
        if split == "test_composite"
        else "full split",
        "hops": padded_length - 2,
        "executed_depth": len(blocks),
        "unique_layers": len(model.blocks),
        "repeats": getattr(model, "repeats", 1),
        "patch_position": 1,
        "conditions": conditions,
        "scores": scores,
        "atomic_preconditions": prerequisites,
        "donor_coverage": {
            family: {
                "n": int(donors[family + "_valid"].sum()),
                "total_n": n,
                "reasons": dict(Counter(donors[family + "_reason"].tolist())),
                "counterfactual_split": dict(
                    Counter(donors[family + "_counterfactual_split"].tolist())
                ),
            }
            for family in ("different", "same")
        },
        "engineering_checks": {
            "native_trace": audits["baseline"],
            "identity_all_layers_components": True,
            "first_layer_same_r1_both_equals_full": True,
            "last_layer_r1_patch_cannot_change_later_answer_or_eos": True,
            "donor_nonpadding_tokens": 2,
            "donor_padded_length": padded_length,
            "mlp_recomputed_after_mixing": False,
            "eos_uses_generated_answer_and_reapplies_same_patch": True,
        },
        "interpretation": (
            "End-to-end prefix-state interventions, not proof that facts reside only in MLP. "
            "Component mixtures retain recipient incoming residual. Both/full are equivalent "
            "only at first execution layer with same r1; later differences are expected. "
            "Facts and queries within one world are not independent experimental worlds."
        ),
        "evaluation_seconds": time.perf_counter() - started,
    }, arrays
