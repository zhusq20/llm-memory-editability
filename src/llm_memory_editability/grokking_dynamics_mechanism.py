"""Pure first-hop state interventions and held-out, local fact-update branches.

The text template's factual answer is predicted at `is`, rather than at r1.
All targets and donors are selected from the graph before model inference.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .depth_step_mechanism import traced_forward
from .grok_depth import write_json
from .latent_scaling import model_digest
from .storage_composition import generate_rows, pack_sentences
from .text_pretrain import atomic_sentence, composite_sentence

PARAMETER = "blocks.0.mlp.down.weight"


def taught_atoms(world):
    return np.concatenate([world["common_atomic"], world["anchor_atomic"]])


def update_case(world, atomic_index, group, replay_n=32, keep_n=32):
    """Change one graph edge; recompute the entire two-hop affected closure."""
    atoms = taught_atoms(world)
    old = atoms[atomic_index].copy()
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    first_role = 13 <= old[1] <= 16
    pool = world[group + "_test"]
    if first_role:
        affected = (pool[:, 0] == old[0]) & (pool[:, 1] == old[1])
        relation_rows = pool[affected]
        options = sorted({int(row[2]) for row in atoms if 13 <= row[1] <= 16})
        # Prefer the same familiar/strict bridge bank; all successor facts must be taught.
        own_bank = sorted({int(row[2]) for row in pool})
        candidates = [v for v in own_bank if v != old[2]] + [
            v for v in options if v != old[2] and v not in own_bank
        ]
        new_value = next(
            v
            for v in candidates
            if all((v, int(row[3])) in lookup for row in relation_rows)
            and any(lookup[v, int(row[3])] != row[4] for row in relation_rows)
        )
    else:
        affected = (pool[:, 2] == old[0]) & (pool[:, 3] == old[1])
        tails = sorted({int(row[2]) for row in atoms if 17 <= row[1] <= 20})
        new_value = tails[(tails.index(int(old[2])) + 1) % len(tails)]
    new = old.copy()
    new[2] = new_value
    lookup[int(new[0]), int(new[1])] = int(new[2])
    tasks = {"E_new": np.asarray([new]), "E_old": np.asarray([old])}
    original_d = {}
    successor_keys = set()
    for split in ["familiar", "strict"]:
        rows = world[split + "_test"]
        mask = (
            ((rows[:, 0] == old[0]) & (rows[:, 1] == old[1]))
            if first_role
            else ((rows[:, 2] == old[0]) & (rows[:, 3] == old[1]))
        )
        original = rows[mask].copy()
        changed = original.copy()
        for row in changed:
            row[2] = lookup[int(row[0]), int(row[1])]
            row[4] = lookup[int(row[2]), int(row[3])]
            successor_keys.add((int(row[2]), int(row[3])))
        name = "D_" + split
        tasks[name] = changed
        original_d[name] = original
        tasks["U_" + split] = rows[~mask].copy()
    index_lookup = {tuple(map(int, row[:2])): i for i, row in enumerate(atoms)}
    successor_indices = sorted(index_lookup[key] for key in successor_keys)
    successor = atoms[successor_indices].copy()
    if not first_role and len(successor):
        successor[(successor[:, 0] == old[0]) & (successor[:, 1] == old[1])] = new
    tasks["successor_atomic"] = successor
    order = sorted(range(len(atoms)), key=lambda i: tuple(atoms[i]))
    excluded = {atomic_index, *successor_indices}
    replay = [i for i in order if i not in excluded][:replay_n]
    keep = [i for i in order if i not in excluded and i not in replay][:keep_n]
    unused = [i for i in range(len(atoms)) if i not in {atomic_index, *replay, *keep}]
    tasks["R_atomic"] = atoms[replay].copy()
    tasks["Kdev_atomic"] = atoms[keep].copy()
    tasks["U_atomic"] = atoms[unused].copy()
    if len(replay) != replay_n or len(keep) != keep_n:
        raise ValueError("Insufficient independent replay and calibration keep facts")
    return {
        "atomic_index": atomic_index,
        "group": group,
        "role": "first" if first_role else "second",
        "old_fact": old.tolist(),
        "new_fact": new.tolist(),
        "replay_indices": replay,
        "keep_indices": keep,
        "unused_indices": unused,
        "tasks": tasks,
        "original_d": original_d,
    }


def select_cases(world, n_per_group_role=2, calibration=False):
    """Graph-defined, largest affected-pool targets; no checkpoint-dependent selection."""
    atoms = taught_atoms(world)
    chosen = []
    groups = ["familiar"] if calibration else ["familiar", "strict"]
    for group in groups:
        rows = world[group + "_test"]
        for role in ["first", "second"]:
            counts = {}
            for row in rows:
                key = tuple(map(int, row[:2] if role == "first" else row[[2, 3]]))
                counts[key] = counts.get(key, 0) + 1
            candidates = [i for i, atom in enumerate(atoms) if tuple(map(int, atom[:2])) in counts]
            candidates.sort(key=lambda i: (-counts[tuple(map(int, atoms[i, :2]))], tuple(atoms[i])))
            for i in candidates[:n_per_group_role]:
                chosen.append(update_case(world, i, group))
    if len(chosen) != len(groups) * 2 * n_per_group_role:
        raise ValueError("Not enough distinct graph-defined editor targets")
    return chosen


def serialize_case(case):
    return {
        key: {name: rows.tolist() for name, rows in value.items()}
        if key in ["tasks", "original_d"]
        else value
        for key, value in case.items()
    }


def answer_batch(rows, device):
    """Factual answer, punctuation and EOS only; random prefix targets are excluded."""
    tokens, labels = pack_sentences(rows)
    first = 4 if rows.shape[1] == 3 else 6
    positions = np.tile([first, first + 1, first + 2], (len(rows), 1))
    selected = labels[np.arange(len(rows))[:, None], positions]
    np.testing.assert_array_equal(
        selected, np.c_[rows[:, -1], np.full(len(rows), 5), np.ones(len(rows), dtype=np.int64)]
    )
    return tuple(torch.as_tensor(x, device=device) for x in [tokens, positions, selected])


def safe_generate(model, rows, device):
    if len(rows):
        return generate_rows(model, rows, device)
    return {"n": 0, "accuracy": None, "answer_accuracy": None, "answer_nll": None}, {
        "generated": np.empty((0, 3), dtype=np.int64),
        "correct": np.empty(0, dtype=bool),
        "answer_nll": np.empty(0, dtype=np.float32),
    }


def editor(model, case, device, arm, lr, nodes=(0, 200)):
    """Independent local update, with parent KL reference and exact final tensor saved."""
    if arm not in ["edit", "review"]:
        raise ValueError("Unknown editor arm")
    started = time.perf_counter()
    parent_hash = model_digest(model)
    edited = copy.deepcopy(model).eval()
    for name, parameter in edited.named_parameters():
        parameter.requires_grad_(name == PARAMETER)
        parameter.grad = None
    before = {key: value.detach().clone() for key, value in edited.state_dict().items()}
    parameter = dict(edited.named_parameters())[PARAMETER]
    optimizer = torch.optim.Adam([parameter], lr=lr)
    target = answer_batch(
        np.asarray([case["new_fact"] if arm == "edit" else case["old_fact"]]), device
    )
    replay = answer_batch(case["tasks"]["R_atomic"], device)
    with torch.no_grad():
        parent_log = model(replay[0], positions=replay[1]).log_softmax(-1).detach()
    parent_d = {
        name: safe_generate(model, rows, device)[1] for name, rows in case["original_d"].items()
    }
    parent_old_fact_ok = safe_generate(model, case["tasks"]["E_old"], device)[1]["correct"].all()
    parent_successors_ok = (
        safe_generate(model, case["tasks"]["successor_atomic"], device)[1]["correct"].all()
        if case["role"] == "first"
        else True
    )
    history, raw, step = [], {}, 0
    for node in nodes:
        while step < node:
            optimizer.zero_grad(set_to_none=True)
            logits = edited(target[0], positions=target[1])
            ce = F.cross_entropy(logits.flatten(0, 1), target[2].flatten())
            current = edited(replay[0], positions=replay[1]).log_softmax(-1)
            kl = F.kl_div(
                current.flatten(0, 1),
                parent_log.flatten(0, 1),
                reduction="batchmean",
                log_target=True,
            )
            loss = ce + kl
            loss.backward()
            optimizer.step()
            step += 1
        metrics = {}
        for name, rows in case["tasks"].items():
            metric, prediction = safe_generate(edited, rows, device)
            if name in case["original_d"]:
                original = case["original_d"][name]
                changed = rows[:, -1] != original[:, -1]
                parent_correct = parent_d[name]["correct"]
                eligible = changed & parent_correct & parent_old_fact_ok & parent_successors_ok
                pred = prediction["generated"]
                old = (pred[:, 0] == original[:, -1]) & (pred[:, 1] == 5) & (pred[:, 2] == 1)
                metric.update(
                    {
                        "changed_n": int(changed.sum()),
                        "parent_old_correct_n": int(parent_correct.sum()),
                        "eligible_n": int(eligible.sum()),
                        "changed_accuracy": float(prediction["correct"][changed].mean())
                        if changed.any()
                        else None,
                        "eligible_accuracy": float(prediction["correct"][eligible].mean())
                        if eligible.any()
                        else None,
                        "eligible_old_answer_accuracy": float(old[eligible].mean())
                        if eligible.any()
                        else None,
                    }
                )
                raw[name + "_changed"] = changed
                raw[name + "_parent_correct"] = parent_correct
                raw[name + "_original_rows"] = original
            metrics[name] = metric
            raw.update({f"step{node}_{name}_{key}": value for key, value in prediction.items()})
        history.append({"step": step, "metrics": metrics})
    changed_tensors = [
        key for key, value in edited.state_dict().items() if not torch.equal(value, before[key])
    ]
    assert not set(changed_tensors) - {PARAMETER}
    assert model_digest(model) == parent_hash
    final = parameter.detach().cpu().clone()
    record = {
        "arm": arm,
        "lr": lr,
        "case": serialize_case(case),
        "history": history,
        "parent_model_sha256": parent_hash,
        "final_model_sha256": model_digest(edited),
        "changed_tensors": changed_tensors,
        "delta_l2": float((parameter - before[PARAMETER]).norm()),
        "seconds": time.perf_counter() - started,
        "parent_old_target_atomic_correct": bool(parent_old_fact_ok),
        "parent_new_successors_correct": bool(parent_successors_ok),
    }
    # Exact-tensor replay verifies every generated endpoint; no rounded delta reconstruction.
    reloaded = copy.deepcopy(model).eval()
    with torch.no_grad():
        dict(reloaded.named_parameters())[PARAMETER].copy_(final.to(device))
    assert model_digest(reloaded) == record["final_model_sha256"]
    for name, rows in case["tasks"].items():
        _, prediction = safe_generate(reloaded, rows, device)
        for key, values in prediction.items():
            reference = raw[f"step{nodes[-1]}_{name}_{key}"]
            if key == "answer_nll":
                np.testing.assert_allclose(values, reference, rtol=1e-5, atol=1e-5)
            else:
                np.testing.assert_array_equal(values, reference)
    record["exact_weight_reload_passed"] = True
    return record, raw, final


def select_donors(world, rows):
    atoms = world["common_atomic"]
    lookup = {tuple(map(int, row[:2])): int(row[2]) for row in atoms}
    first = [(i, atom) for i, atom in enumerate(atoms) if 13 <= atom[1] <= 16]
    first.sort(key=lambda pair: tuple(pair[1]))
    records = {}
    for style in ["same_bridge", "different_bridge"]:
        indices, tails = [], []
        for h, r1, bridge, r2, tail in rows:
            eligible = [
                (i, atom)
                for i, atom in first
                if atom[1] == r1
                and atom[0] != h
                and (int(atom[2]), int(r2)) in lookup
                and (
                    (atom[2] == bridge)
                    if style == "same_bridge"
                    else (atom[2] != bridge and lookup[int(atom[2]), int(r2)] != tail)
                )
            ]
            indices.append(eligible[0][0] if eligible else -1)
            tails.append(lookup[int(eligible[0][1][2]), int(r2)] if eligible else -1)
        records[style] = {"indices": np.asarray(indices), "tails": np.asarray(tails)}
    return records


@torch.no_grad()
def patched_generation(model, tokens, patch):
    generated = []
    first_prob = None
    for step in range(3):
        logits, _ = traced_forward(model, tokens, patch)
        if step == 0:
            first_prob = logits[:, -1].softmax(-1)
        answer = logits[:, -1].argmax(-1)
        generated.append(answer.cpu().numpy())
        tokens = torch.cat([tokens, answer[:, None]], dim=1)
    return np.stack(generated, axis=1), first_prob


@torch.no_grad()
def trace(model, world, device, out):
    """Two fixed, causally reachable sites; donors contain only a first-hop prefix."""
    model.eval()
    out = Path(out)
    records, raw = [], {}
    atoms = world["common_atomic"]
    atom_metric, atom_prediction = generate_rows(model, atoms, device)
    index_lookup = {tuple(map(int, row[:2])): i for i, row in enumerate(atoms)}
    for split in ["familiar", "strict"]:
        rows = world[split + "_test"]
        tokens = torch.as_tensor([composite_sentence(row)[:7] for row in rows], device=device)
        reference = model(tokens)
        logits, cache = traced_forward(model, tokens)
        torch.testing.assert_close(logits, reference, rtol=1e-5, atol=2e-6)
        assert torch.equal(logits.argmax(-1), reference.argmax(-1))
        baseline_pred, baseline_prob = patched_generation(model, tokens, {})
        raw[split + "_rows"] = rows
        raw[split + "_baseline_generated"] = baseline_pred
        for position in [3, 6]:
            state = cache["postresidual"][0, :, position]
            readout = F.linear(model.ln_final(state), model.token.weight)
            bridge = torch.as_tensor(rows[:, 2], device=device)
            rank = 1 + (readout > readout.gather(-1, bridge[:, None])).sum(-1)
            raw[f"{split}_bridge_rank_p{position}"] = rank.cpu().numpy()
        raw[split + "_attention"] = cache["attention_map"].cpu().numpy()
        for recipient, _donor_position in [(3, 3), (6, 4)]:
            for component in ["postresidual", "mlp_delta"]:
                self_patch = {(0, component, recipient): cache[component][0, :, recipient]}
                self_pred, self_prob = patched_generation(model, tokens, self_patch)
                np.testing.assert_array_equal(self_pred, baseline_pred)
                torch.testing.assert_close(self_prob, baseline_prob, rtol=1e-5, atol=2e-6)
        for style, donor in select_donors(world, rows).items():
            valid = donor["indices"] >= 0
            ids = np.flatnonzero(valid)
            if not len(ids):
                continue
            donor_rows = atoms[donor["indices"][valid]]
            donor_tokens = torch.as_tensor(
                [atomic_sentence(row)[:5] for row in donor_rows], device=device
            )
            _, donor_cache = traced_forward(model, donor_tokens)
            expected = donor["tails"][valid]
            expected_tensor = torch.as_tensor(expected, device=device)
            premise = np.asarray(
                [
                    atom_prediction["correct"][index_lookup[int(row[0]), int(row[1])]]
                    and atom_prediction["correct"][index_lookup[int(row[2]), int(row[3])]]
                    and atom_prediction["correct"][int(donor_index)]
                    and atom_prediction["correct"][index_lookup[int(donor_row[2]), int(row[3])]]
                    for row, donor_row, donor_index in zip(
                        rows[valid], donor_rows, donor["indices"][valid], strict=True
                    )
                ]
            )
            raw[f"{split}_{style}_ids"] = ids
            raw[f"{split}_{style}_donor_rows"] = donor_rows
            raw[f"{split}_{style}_expected"] = expected
            for recipient, donor_position in [(3, 3), (6, 4)]:
                for component in ["postresidual", "mlp_delta"]:
                    patch = {
                        (0, component, recipient): donor_cache[component][0, :, donor_position]
                    }
                    pred, prob = patched_generation(model, tokens[valid], patch)
                    correct = (pred[:, 0] == expected) & (pred[:, 1] == 5) & (pred[:, 2] == 1)
                    base = baseline_pred[valid]
                    base_correct = (base[:, 0] == expected) & (base[:, 1] == 5) & (base[:, 2] == 1)
                    selected = torch.arange(len(ids), device=device)
                    new_probability = prob[selected, expected_tensor].cpu().numpy()
                    original_probability = (
                        baseline_prob[valid][selected, expected_tensor].cpu().numpy()
                    )
                    name = f"{split}_{style}_p{recipient}_{component}"
                    raw[name + "_generated"] = pred
                    raw[name + "_gold_probability"] = new_probability
                    raw[name + "_baseline_gold_probability"] = original_probability
                    record = {
                        "split": split,
                        "style": style,
                        "recipient_position": recipient,
                        "donor_position": donor_position,
                        "component": component,
                        "n": len(ids),
                        "pool_n": len(rows),
                        "coverage": float(valid.mean()),
                        "necessary_atoms_correct_n": int(premise.sum()),
                        "baseline_target_accuracy": float(base_correct.mean()),
                        "patched_target_accuracy": float(correct.mean()),
                        "target_accuracy_delta_pp": 100
                        * float((correct.astype(float) - base_correct).mean()),
                        "target_probability_delta": float(
                            (new_probability - original_probability).mean()
                        ),
                        "conditional_target_accuracy": float(correct[premise].mean())
                        if premise.any()
                        else None,
                    }
                    records.append(record)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "trace.npz", **raw)
    result = {
        "records": records,
        "traced_forward_and_self_patch_passed": True,
        "common_atomic": atom_metric,
    }
    write_json(out / "trace.json", result)
    return result
