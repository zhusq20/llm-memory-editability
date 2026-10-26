"""Persistent MLP writes with an immutable global anchor and read-only queries."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .ce_scope_editing import arrays, freeze, parameter_names, set_scope


def formal_pairs(panel):
    return [p for p in panel["pairs"] if p["split"] == "evaluation"]


def facts_by_id(panel):
    return {p[side]["fact_id"]: p[side]["fact"] for p in formal_pairs(panel) for side in ("A", "B")}


def validate_panel(panel, world):
    keys = set()
    for pair in panel["pairs"]:
        a, r, b = pair["A"]["fact"]
        head, relation, tail = pair["B"]["fact"]
        if head != b or len({a, b, tail}) != 3:
            raise ValueError("Two new edges must form a nondegenerate chain")
        if pair["query"] != [a, r, relation] or pair["answer"] != tail:
            raise ValueError("Double-new truth differs from its two new edges")
        for side in ("A", "B"):
            h, rel, t = pair[side]["fact"]
            if not (0 <= h < world.entities and 0 <= t < world.entities):
                raise ValueError("New entity is outside the existing vocabulary")
            if not 0 <= rel < world.metadata["relations"]:
                raise ValueError("New relation is outside the existing vocabulary")
            if world.lookup[h, rel] != -1 or (h, rel) in keys:
                raise ValueError("All new keys must be unique and absent")
            keys.add((h, rel))
    facts = facts_by_id(panel)
    for stream in panel["streams"].values():
        if len(stream) != len(facts) or set(stream) != set(facts):
            raise ValueError("Every stream must include all formal facts exactly once")
    if panel["streams"]["reverse"] != panel["streams"]["forward"][::-1]:
        raise ValueError("The second stream must reverse the entire first stream")
    for pair in formal_pairs(panel):
        for stream in panel["streams"].values():
            gap = abs(stream.index(pair["A"]["fact_id"]) - stream.index(pair["B"]["fact_id"]))
            if gap != len(formal_pairs(panel)):
                raise ValueError("Within-pair separation changed")
    for row in panel["mixed_queries"]:
        h, relation, tail = facts[row["fact_id"]]
        old = world.atoms[row["old_atom_index"]].tolist()
        if old != row["old_fact"]:
            raise ValueError("Old mixed-query premise changed")
        if row["direction"] == "new_old":
            expected = ([h, relation, old[1]], old[2])
            valid = old[0] == tail
        else:
            expected = ([old[0], old[1], relation], tail)
            valid = old[2] == h and row["direction"] == "old_new"
        if not valid or (row["query"], row["answer"]) != expected:
            raise ValueError("Mixed query must use its one new and one unchanged edge")
    return {"new_keys": len(keys), "formal_pairs": len(formal_pairs(panel)), "passed": True}


def query_groups(panel):
    pairs = formal_pairs(panel)
    return {
        "atoms": [p[side]["fact"] for p in pairs for side in ("A", "B")],
        "double_new": [p["query"] + [p["answer"]] for p in pairs],
        "mixed": [q["query"] + [q["answer"]] for q in panel["mixed_queries"]],
        "old_companion": [q["old_fact"] for q in panel["mixed_queries"]],
        **{"old/" + name: rows for name, rows in panel["old_eval_pools"].items()},
    }


def settings_for(plan, architecture, steps=None):
    source = plan["adaptation"]
    return {
        "steps": source["steps_per_fact"] if steps is None else steps,
        "learning_rate": plan["architectures"][architecture]["learning_rate"],
        "kl_factor": source["kl_factor"],
        "norm_constraint": source["parameter_constraint"]["maximum_absolute_element_change"],
        "gradient_clip": source["gradient_clip"],
        "betas": tuple(source["betas"]),
        "eps": source["eps"],
        "weight_decay": source["weight_decay"],
    }


def atomic_write(reader, fact, metadata, settings, teachers, anchor, progress=None):
    """Only this fact enters the new loss; Adam is fresh, the anchor is never rolled."""
    if len(fact) != 3 or len(teachers) != settings["steps"]:
        raise ValueError("One fact and exactly the fixed teacher batches are required")
    for key in ("learning_rate", "kl_factor", "norm_constraint", "gradient_clip"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("Invalid positive write setting")
    names = parameter_names(reader.backbone, "mlp")
    if set(anchor) != set(names):
        raise ValueError("Global anchor must contain exactly all unique physical MLP parameters")
    set_scope(reader, "mlp")
    parameters = [reader.backbone.get_parameter(name) for name in names]
    device = next(reader.parameters()).device
    tokens, positions, labels = arrays([fact], metadata, device)
    tokens, positions, labels = tokens[:, :2], positions[:, :1], labels[:, :1]
    optimizer = torch.optim.Adam(
        parameters,
        lr=settings["learning_rate"],
        betas=settings["betas"],
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
    )
    history = []
    for step, batches in enumerate(teachers, 1):
        optimizer.zero_grad(set_to_none=True)
        ce = F.cross_entropy(reader(tokens, positions)[:, 0], labels[:, 0])
        if not torch.isfinite(ce):
            raise FloatingPointError("Nonfinite new-fact CE")
        ce.backward()
        kls = []
        for batch in batches:
            x, p, old = (batch[k].to(device) for k in ("tokens", "positions", "log_probability"))
            logits = reader(x, p)
            kl = F.kl_div(logits.log_softmax(-1), old, log_target=True, reduction="sum")
            kl = kl / (logits.shape[0] * logits.shape[1])
            if not torch.isfinite(kl):
                raise FloatingPointError("Nonfinite protected old-output KL")
            (settings["kl_factor"] * kl / len(batches)).backward()
            kls.append(float(kl.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite write gradient")
        optimizer.step()
        eps = settings["norm_constraint"]
        with torch.no_grad():
            for name, parameter in zip(names, parameters, strict=True):
                base = anchor[name]
                parameter.copy_(torch.clamp(parameter, min=base - eps, max=base + eps))
        entry = {
            "step": step,
            "new_ce_before": float(ce.detach()),
            "old_KL_before": kls,
            "gradient_norm_before_clip": float(norm),
        }
        history.append(entry)
        if progress:
            progress(step, entry)
    freeze(reader)
    with torch.no_grad():
        maximum, saturated, elements = 0.0, 0, 0
        for name, parameter in zip(names, parameters, strict=True):
            base = anchor[name]
            maximum = max(maximum, float((parameter - base).abs().max()))
            saturated += int(((parameter == base - eps) | (parameter == base + eps)).sum())
            elements += parameter.numel()
    return {
        "steps": settings["steps"],
        "settings": settings,
        "optimizer_reset_before_fact": True,
        "global_theta0_anchor": True,
        "maximum_absolute_parameter_change": maximum,
        "constraint_saturated_elements": saturated,
        "trainable_parameters": elements,
        "new_supervised_tokens": settings["steps"],
        "old_examples": sum(int(b["tokens"].shape[0]) for bs in teachers for b in bs),
        "new_composition_supervision": 0,
        "history": history,
    }


@torch.no_grad()
def greedy_rows(reader, rows, metadata, batch_size=64):
    """Two causal native forwards: answer then EOS after the generated answer."""
    rows = np.asarray(rows, dtype=np.int64)
    output = {
        k: []
        for k in (
            "rows",
            "answer",
            "stop",
            "expected",
            "answer_correct",
            "final_correct",
            "answer_nll",
            "answer_target_logit",
            "answer_target_rank",
            "stop_nll",
            "stop_target_logit",
            "stop_target_rank",
        )
    }
    device = next(reader.parameters()).device
    for offset in range(0, len(rows), batch_size):
        part = rows[offset : offset + batch_size]
        tokens, positions, labels = arrays(part, metadata, device)
        scores = reader(tokens, positions)[:, 0].float()
        expected = labels[:, 0]
        target = scores.gather(1, expected[:, None]).squeeze(1)
        answer = scores.argmax(-1)
        output["answer_nll"].extend(
            F.cross_entropy(scores, expected, reduction="none").cpu().tolist()
        )
        output["answer_target_logit"].extend(target.cpu().tolist())
        output["answer_target_rank"].extend((1 + (scores > target[:, None]).sum(-1)).cpu().tolist())
        tokens[torch.arange(len(part), device=device), positions[:, 1]] = answer
        stop_scores = reader(tokens, positions)[:, 1].float()
        stop_target = stop_scores.gather(1, labels[:, 1, None]).squeeze(1)
        stop = stop_scores.argmax(-1)
        output["stop_nll"].extend(
            F.cross_entropy(stop_scores, labels[:, 1], reduction="none").cpu().tolist()
        )
        output["stop_target_logit"].extend(stop_target.cpu().tolist())
        output["stop_target_rank"].extend(
            (1 + (stop_scores > stop_target[:, None]).sum(-1)).cpu().tolist()
        )
        output["rows"].extend(part.tolist())
        output["answer"].extend(answer.cpu().tolist())
        output["stop"].extend(stop.cpu().tolist())
        output["expected"].extend(expected.cpu().tolist())
        output["answer_correct"].extend((answer == expected).cpu().tolist())
        output["final_correct"].extend(
            ((answer == expected) & (stop == labels[:, 1])).cpu().tolist()
        )
    return output


@torch.no_grad()
def read_state(reader, panel, batch_size=64):
    versions = {n: p._version for n, p in reader.named_parameters()}
    result = {"old": {}}
    for name, rows in query_groups(panel).items():
        readout = greedy_rows(reader, rows, panel["metadata"], batch_size)
        if name.startswith("old/"):
            result["old"][name[4:]] = readout
        elif name == "old_companion":
            result["mixed"]["old_companion"] = readout
        else:
            result[name] = readout
    if versions != {n: p._version for n, p in reader.named_parameters()}:
        raise AssertionError("Read-only evaluation changed parameters")
    return result


def compare_readouts(actual, expected, tolerance=1e-5, path=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"Readout keys differ at {path}")
        for key in expected:
            compare_readouts(actual[key], expected[key], tolerance, path + "/" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"Readout lengths differ at {path}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare_readouts(a, e, tolerance, f"{path}/{i}")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or abs(actual - expected) > tolerance:
            raise AssertionError(f"Floating readout differs at {path}: {actual}, {expected}")
    elif actual != expected:
        raise AssertionError(f"Discrete readout differs at {path}")
