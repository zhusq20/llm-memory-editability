"""Descriptive entity readouts, pure-first-hop interventions and MLP edits.

No fitted probe or performance predictor is used. Residual readouts and raw MLP
cosines are descriptions; only interventions test a causal contribution. The
truth graph chooses queries, donors and edits before any model is evaluated.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .depth_step import ENTITY_OFFSET, EOS, construct, pack_rows, prompt_rows, truth_path_details
from .grok_depth import utc, write_json
from .storage_composition import file_hash


def traced_forward(model, tokens, patches=None):
    """Use the real SDPA computation and record every executed block/position.

    Patches map (zero-based execution, 'mlp_delta' or 'postresidual', position)
    to a batch of replacement vectors. Repeated executions of a shared block
    remain separately indexed. Attention maps are descriptive manual softmax;
    the forward uses the same SDPA primitive as LoopGPT.
    """
    if model.training:
        raise ValueError("Tracing requires evaluation mode")
    patches = patches or {}
    length = tokens.shape[1]
    x = model.token(tokens) + model.position(torch.arange(length, device=tokens.device))
    cache = {key: [] for key in ("attention_map", "attention_delta", "mlp_delta", "postresidual")}
    consumed = set()
    for index, block in enumerate(model.iter_blocks()):
        z = block.ln1(x)
        batch, _, width = z.shape
        a = block.attention
        q, k, v = a.qkv(z).view(batch, length, 3, a.heads, width // a.heads).unbind(2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // a.heads)
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        attention_map = scores.masked_fill(~causal, -torch.inf).softmax(-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        attention_delta = a.proj(y.transpose(1, 2).reshape(batch, length, width))
        x = x + attention_delta
        mlp_delta = block.mlp(block.ln2(x))
        for key, replacement in patches.items():
            layer, component, position = key
            if layer == index and component == "mlp_delta":
                if replacement.shape != mlp_delta[:, position].shape:
                    raise ValueError("Patch shape differs from selected MLP position")
                mlp_delta = mlp_delta.clone()
                mlp_delta[:, position] = replacement
                consumed.add(key)
        x = x + mlp_delta
        for key, replacement in patches.items():
            layer, component, position = key
            if layer == index and component == "postresidual":
                if replacement.shape != x[:, position].shape:
                    raise ValueError("Patch shape differs from selected residual position")
                x = x.clone()
                x[:, position] = replacement
                consumed.add(key)
        for key, value in (
            ("attention_map", attention_map),
            ("attention_delta", attention_delta),
            ("mlp_delta", mlp_delta),
            ("postresidual", x),
        ):
            cache[key].append(value.detach())
    if consumed != set(patches):
        raise ValueError("Unknown execution/component in patch")
    logits = F.linear(model.ln_final(x), model.token.weight)
    return logits, {key: torch.stack(values) for key, values in cache.items()}


def _separator(world):
    return int(world["metadata"]["separator_token"])


def _mean(values):
    return float(np.mean(values)) if len(values) else None


@torch.no_grad()
def generate(model, rows, world, device, batch_size=256):
    """Free answer then free EOS, retaining full-vocabulary probabilities."""
    rows = np.asarray(rows, dtype=np.int64)
    predictions, probabilities, gold_probabilities = [], [], []
    for begin in range(0, len(rows), batch_size):
        selected = rows[begin : begin + batch_size]
        tokens = torch.as_tensor(prompt_rows(selected, _separator(world)), device=device)
        probability = model(tokens)[:, -1].softmax(-1)
        answer = probability.argmax(-1)
        eos = model(torch.cat((tokens, answer[:, None]), dim=1))[:, -1].argmax(-1)
        predictions.append(torch.stack((answer, eos), 1).cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        gold_probabilities.append(
            probability[
                torch.arange(len(selected), device=device),
                torch.as_tensor(selected[:, -1], device=device),
            ]
            .cpu()
            .numpy()
        )
    vocab = model.config.vocab_size
    predicted = np.concatenate(predictions) if predictions else np.empty((0, 2), dtype=np.int64)
    prob = (
        np.concatenate(probabilities) if probabilities else np.empty((0, vocab), dtype=np.float32)
    )
    gold_prob = np.concatenate(gold_probabilities) if gold_probabilities else np.empty(0)
    correct = predicted[:, 0] == rows[:, -1]
    return {
        "n": len(rows),
        "answer_accuracy": _mean(correct),
        "accuracy": _mean(correct & (predicted[:, 1] == EOS)),
        "gold_probability": _mean(gold_prob),
    }, {"rows": rows, "predictions": predicted, "answer_probabilities": prob}


def select_donors(world, rows):
    """Lexicographic donors know exactly one fact and no target second relation."""
    atoms = world["atomic"]
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    order = sorted(range(len(atoms)), key=lambda i: tuple(atoms[i]))
    nodes, _ = truth_path_details(world, rows)
    same, changed, changed_tail = [], [], []
    for row, path in zip(rows, nodes, strict=True):
        h, r1, r2, old_tail = map(int, row)
        bridge = int(path[1])
        identical = next(
            (i for i in order if int(atoms[i, 2]) == bridge and tuple(atoms[i, :2]) != (h, r1)), -1
        )
        different = next(
            (
                i
                for i in order
                if int(atoms[i, 2]) != bridge
                and (int(atoms[i, 2]), r2) in lookup
                and lookup[(int(atoms[i, 2]), r2)] != old_tail
            ),
            -1,
        )
        same.append(identical)
        changed.append(different)
        changed_tail.append(lookup[(int(atoms[different, 2]), r2)] if different >= 0 else -1)
    return {
        "same_bridge_atomic_indices": np.asarray(same, dtype=np.int64),
        "different_bridge_atomic_indices": np.asarray(changed, dtype=np.int64),
        "different_bridge_tail": np.asarray(changed_tail, dtype=np.int64),
    }


@torch.no_grad()
def entity_readouts(model, cache, targets, entities):
    """Apply the unmodified final norm/tied readout; raw MLP cosine is separate."""
    residual = cache["postresidual"]
    logits = F.linear(model.ln_final(residual), model.token.weight)
    entity_logits = logits[..., ENTITY_OFFSET : ENTITY_OFFSET + entities]
    probabilities = logits.softmax(-1)
    entity_probabilities = entity_logits.softmax(-1)
    embedding = F.normalize(model.token.weight[ENTITY_OFFSET : ENTITY_OFFSET + entities], dim=-1)
    cosine = F.normalize(cache["mlp_delta"], dim=-1) @ embedding.T
    arrays = {}
    for name, token_ids in targets.items():
        ids = torch.as_tensor(token_ids, device=residual.device)
        full_ids = ids[None, :, None, None].expand(*residual.shape[:-1], 1)
        entity_ids = full_ids - ENTITY_OFFSET
        gold_logit = entity_logits.gather(-1, entity_ids).squeeze(-1)
        gold_cosine = cosine.gather(-1, entity_ids).squeeze(-1)
        arrays[name + "_residual_entity_rank"] = 1 + (entity_logits > gold_logit[..., None]).sum(-1)
        arrays[name + "_residual_vocab_probability"] = probabilities.gather(-1, full_ids).squeeze(
            -1
        )
        arrays[name + "_residual_entity_probability"] = entity_probabilities.gather(
            -1, entity_ids
        ).squeeze(-1)
        arrays[name + "_mlp_embedding_cosine"] = gold_cosine
        arrays[name + "_mlp_embedding_cosine_rank"] = 1 + (cosine > gold_cosine[..., None]).sum(-1)
    return {name: value.cpu().numpy() for name, value in arrays.items()}


@torch.no_grad()
def _patched_generation(model, tokens, patch):
    logits, _ = traced_forward(model, tokens, patch)
    probability = logits[:, -1].softmax(-1)
    answer = probability.argmax(-1)
    # Reapply the same intervention while freely generating the EOS token.
    extended = torch.cat((tokens, answer[:, None]), dim=1)
    eos_logits, _ = traced_forward(model, extended, patch)
    eos = eos_logits[:, -1].argmax(-1)
    return torch.stack((answer, eos), 1).cpu().numpy(), probability.cpu().numpy()


def trace_analysis(model, world, out, device, max_queries=128):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model.eval()
    started = time.perf_counter()
    raw, baseline = {}, {}
    for split in ("atomic", "familiar_2", "strict_2"):
        metrics, prediction = generate(model, world[split], world, device)
        baseline[split] = metrics
        raw.update({f"baseline_{split}_{key}": value for key, value in prediction.items()})
    indices = np.arange(min(max_queries, len(world["familiar_2"])))
    rows = world["familiar_2"][indices]
    nodes, edges = truth_path_details(world, rows)
    tokens = torch.as_tensor(prompt_rows(rows, _separator(world)), device=device)
    with torch.no_grad():
        reference = model(tokens)
        actual, cache = traced_forward(model, tokens)
    difference = float((actual - reference).abs().max())
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=2e-6)
    if not torch.equal(actual.argmax(-1), reference.argmax(-1)):
        raise ValueError("Traced forward changes full-vocabulary argmax")
    raw.update(
        {
            "query_indices": indices,
            "query_rows": rows,
            "truth_nodes": nodes,
            "truth_atomic_indices": edges,
        }
    )
    raw.update({"cache_" + key: value.cpu().numpy() for key, value in cache.items()})
    raw.update(
        entity_readouts(
            model,
            cache,
            {"bridge": nodes[:, 1], "tail": nodes[:, -1]},
            world["metadata"]["entities"],
        )
    )
    donors = select_donors(world, rows)
    raw.update(donors)
    atom_predictions = raw["baseline_atomic_predictions"]
    atoms = world["atomic"]
    atom_correct = (atom_predictions[:, 0] == atoms[:, -1]) & (atom_predictions[:, 1] == EOS)
    atomic_key_indices = {tuple(map(int, row[:2])): i for i, row in enumerate(atoms)}
    raw["original_constituent_atoms_correct"] = atom_correct[edges].all(1)
    records = []
    for style in ("self", "same_bridge", "different_bridge"):
        donor_ids = edges[:, 0] if style == "self" else donors[style + "_atomic_indices"]
        valid = donor_ids >= 0
        selected = np.flatnonzero(valid)
        if not len(selected):
            continue
        donor_rows = world["atomic"][donor_ids[valid]]
        donor_tokens = torch.as_tensor(prompt_rows(donor_rows, _separator(world)), device=device)
        with torch.no_grad():
            _, donor_cache = traced_forward(model, donor_tokens)
        expected = (
            donors["different_bridge_tail"][valid]
            if style == "different_bridge"
            else rows[valid, -1]
        )
        new_second_indices = np.asarray(
            [
                atomic_key_indices[(int(donor[2]), int(query[2]))]
                for donor, query in zip(donor_rows, rows[valid], strict=True)
            ]
        )
        premises = (
            atom_correct[edges[valid]].all(1)
            & atom_correct[donor_ids[valid]]
            & atom_correct[new_second_indices]
        )
        raw[style + "_premises_correct"] = premises
        for layer in range(model.effective_depth):
            for component in ("mlp_delta", "postresidual"):
                for position in (2, 1):
                    key = f"{style}_layer{layer}_{component}_pos{position}"
                    patch = {
                        (layer, component, position): donor_cache[component][layer, :, position]
                    }
                    prediction, probability = _patched_generation(model, tokens[valid], patch)
                    raw[key + "_selected_query_indices"] = selected
                    raw[key + "_predictions"] = prediction
                    raw[key + "_answer_probabilities"] = probability
                    following = prediction[:, 0] == expected
                    correct = prediction[:, 0] == rows[valid, -1]
                    base_pred = raw["baseline_familiar_2_predictions"][indices[valid]]
                    base_correct = (base_pred[:, 0] == rows[valid, -1]) & (base_pred[:, 1] == EOS)
                    if style == "self" and not np.array_equal(prediction, base_pred):
                        raise ValueError("Pure-first-hop self patch changes generation")
                    record = {
                        "donor": style,
                        "execution": layer,
                        "component": component,
                        "position": position,
                        "n": len(selected),
                        "selected_query_coverage": len(selected) / len(rows),
                        "full_pool_coverage": len(selected) / len(world["familiar_2"]),
                        "new_route_answer_accuracy": _mean(following),
                        "new_route_accuracy": _mean(following & (prediction[:, 1] == EOS)),
                        "original_accuracy": _mean(correct & (prediction[:, 1] == EOS)),
                        "baseline_correct_coverage": _mean(base_correct),
                        "all_required_atomic_correct_coverage": _mean(premises),
                        "new_route_conditional_on_required_atomics": _mean(
                            (following & (prediction[:, 1] == EOS))[premises]
                        ),
                        "new_route_conditional_on_baseline_correct": _mean(
                            (following & (prediction[:, 1] == EOS))[base_correct]
                        ),
                    }
                    records.append(record)
    with torch.no_grad():
        entity_embedding = model.token.weight[
            ENTITY_OFFSET : ENTITY_OFFSET + world["metadata"]["entities"]
        ]
        normalized = F.normalize(entity_embedding, dim=-1)
        raw["entity_embedding_cosine_matrix"] = (normalized @ normalized.T).cpu().numpy()
        raw["entity_embedding_singular_values"] = (
            torch.linalg.svdvals(entity_embedding).cpu().numpy()
        )
    np.savez_compressed(out / "trace-raw.npz", **raw)
    report = {
        "phase": "development",
        "created_utc": utc(),
        "baseline": baseline,
        "query_selection": (
            "First 128 familiar_2 rows in frozen data order, independent of predictions"
        ),
        "selected_queries": len(rows),
        "full_pool_queries": len(world["familiar_2"]),
        "selection_coverage": len(rows) / len(world["familiar_2"]),
        "forward_max_absolute_difference": difference,
        "forward_all_argmax_identical": True,
        "donor_information": (
            "Only BOS, head, first relation, SEP; no second relation or final answer"
        ),
        "patch_position_roles": {"1": "head control", "2": "first relation"},
        "interventions": records,
        "wall_seconds": time.perf_counter() - started,
        "limits": [
            "Unfitted final-norm tied readout and raw cosine are descriptive",
            "Attention maps are descriptive; patch tests entire state or MLP contribution",
            "Layers, positions and queries are dependent measurements in one world",
        ],
    }
    write_json(out / "trace-summary.json", report)
    return report


def edit_cases(world, n_facts=8, replay_n=32):
    """Choose graph-based first-hop edits before observing any model behavior."""
    atoms = world["atomic"]
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    candidate_first = {(int(row[0]), int(row[1])) for row in world["familiar_2"]}
    order = sorted(range(len(atoms)), key=lambda i: tuple(atoms[i]))
    cases = []
    for index in order:
        h, relation, old_bridge = map(int, atoms[index])
        if (h, relation) not in candidate_first:
            continue
        first_rows = world["familiar_2"][
            (world["familiar_2"][:, 0] == h) & (world["familiar_2"][:, 1] == relation)
        ]
        bridges = sorted({int(t) for _, _, t in atoms})
        new_bridge = next(
            (
                b
                for b in bridges
                if b != old_bridge
                and all(
                    (b, int(row[2])) in lookup and lookup[(b, int(row[2]))] != int(row[-1])
                    for row in first_rows
                )
            ),
            None,
        )
        if new_bridge is None:
            continue
        replay_indices = np.asarray([i for i in order if i != index][:replay_n])
        if len(replay_indices) != replay_n:
            raise ValueError("Insufficient nonedited replay atoms")
        new_fact = np.asarray([[h, relation, new_bridge]], dtype=np.int64)
        old_fact = atoms[index : index + 1].copy()
        edited_lookup = {**lookup, (h, relation): new_bridge}
        tasks = {"E_new": new_fact, "E_old": old_fact, "R_atomic": atoms[replay_indices].copy()}
        original_d_rows = {}
        unreplayed = [i for i in range(len(atoms)) if i != index and i not in replay_indices]
        tasks["U_atomic"] = atoms[unreplayed].copy()
        successor_keys = set()
        for split in ("familiar_2", "strict_2"):
            rows = world[split]
            nodes, path_indices = truth_path_details(world, rows)
            first = path_indices[:, 0] == index
            second = (path_indices[:, 1] == index) & ~first
            unaffected = ~(first | second)
            for role, mask in (("first", first), ("second", second)):
                changed = rows[mask].copy()
                for i, row in enumerate(changed):
                    bridge = edited_lookup[(int(row[0]), int(row[1]))]
                    changed[i, -1] = edited_lookup[(bridge, int(row[2]))]
                    if role == "first" and (bridge, int(row[2])) != (h, relation):
                        successor_keys.add((bridge, int(row[2])))
                tasks[f"D_{role}_{split}"] = changed
                original_d_rows[f"D_{role}_{split}"] = rows[mask].copy()
            tasks["U_" + split] = rows[unaffected].copy()
            # All facts occurring as either hop in U retain their graph labels.
            if index in path_indices[unaffected]:
                raise ValueError("Edited fact leaked into the unaffected holdout")
            del nodes
        successor_indices = [i for i, row in enumerate(atoms) if tuple(row[:2]) in successor_keys]
        tasks["necessary_successor_atomic"] = atoms[successor_indices].copy()
        cases.append(
            {
                "atomic_index": index,
                "old_fact": old_fact[0].tolist(),
                "new_fact": new_fact[0].tolist(),
                "replay_indices": replay_indices.tolist(),
                "tasks": tasks,
                "original_d_rows": original_d_rows,
                "propagation_scope": {
                    name: {
                        "n": len(old_rows),
                        "changed_answer_n": int(
                            np.count_nonzero(tasks[name][:, -1] != old_rows[:, -1])
                        ),
                        "source_pool_n": len(world[name.split("_", 2)[2]]),
                    }
                    for name, old_rows in original_d_rows.items()
                },
            }
        )
        if len(cases) == n_facts:
            break
    if len(cases) != n_facts:
        raise ValueError("Fewer graph-valid fixed edit cases than requested")
    return cases


def _atomic_loss(model, rows, world, device):
    tokens, labels = pack_rows(rows, separator=_separator(world))
    tokens = torch.as_tensor(tokens, device=device)
    labels = torch.as_tensor(labels, device=device)
    logits = model(tokens)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)


def edit_one(model, case, world, device, arm, nodes=(0, 20, 100, 200), lr=0.01):
    """Independently edit one checkpoint; only unique block-0 MLP down changes."""
    if arm not in {"edit", "review"}:
        raise ValueError("Arm must be edit or review")
    if tuple(nodes) != tuple(sorted(set(nodes))) or nodes[0] != 0:
        raise ValueError("Nodes must be increasing with initial measurement")
    edited = copy.deepcopy(model).eval()
    target_name = "blocks.0.mlp.down.weight"
    for name, parameter in edited.named_parameters():
        parameter.requires_grad_(name == target_name)
        parameter.grad = None
    before = {name: value.detach().clone() for name, value in edited.state_dict().items()}
    parameter = dict(edited.named_parameters())[target_name]
    optimizer = torch.optim.Adam([parameter], lr=lr)
    objective_rows = np.asarray([case["new_fact"] if arm == "edit" else case["old_fact"]])
    history, raw = [], {}
    for name, old_rows in case["original_d_rows"].items():
        raw[name + "_original_rows"] = old_rows
        raw[name + "_changed_answer_mask"] = case["tasks"][name][:, -1] != old_rows[:, -1]
    step = 0
    final_loss = None
    for node in nodes:
        while step < node:
            optimizer.zero_grad(set_to_none=True)
            loss = 0.5 * _atomic_loss(edited, objective_rows, world, device)
            loss = loss + 0.5 * _atomic_loss(edited, case["tasks"]["R_atomic"], world, device)
            loss.backward()
            optimizer.step()
            step += 1
            final_loss = float(loss.detach())
        measures = {}
        for name, rows in case["tasks"].items():
            metrics, prediction = generate(edited, rows, world, device)
            if name in case["original_d_rows"]:
                old_rows = case["original_d_rows"][name]
                changed_mask = rows[:, -1] != old_rows[:, -1]
                predicted = prediction["predictions"]
                correct = (predicted[:, 0] == rows[:, -1]) & (predicted[:, 1] == EOS)
                old_correct = (predicted[:, 0] == old_rows[:, -1]) & (predicted[:, 1] == EOS)
                metrics.update(
                    {
                        "changed_answer_n": int(changed_mask.sum()),
                        "changed_answer_coverage": _mean(changed_mask),
                        "changed_answer_accuracy": _mean(correct[changed_mask]),
                        "old_answer_accuracy": _mean(old_correct),
                    }
                )
            measures[name] = metrics
            raw.update({f"step{step}_{name}_{key}": value for key, value in prediction.items()})
        history.append({"step": step, "loss": final_loss, "metrics": measures})
    changed = [
        name for name, value in edited.state_dict().items() if not torch.equal(value, before[name])
    ]
    if set(changed) - {target_name}:
        raise ValueError("Parameters outside the frozen MLP projection changed")
    delta = parameter.detach() - before[target_name]
    raw["mlp_down_weight_delta"] = delta.cpu().numpy()
    return {
        "arm": arm,
        "atomic_index": case["atomic_index"],
        "old_fact": case["old_fact"],
        "new_fact": case["new_fact"],
        "replay_indices": case["replay_indices"],
        "propagation_scope": case["propagation_scope"],
        "updated_parameter": target_name,
        "changed_state_tensors": changed,
        "weight_delta_l2": float(delta.norm()),
        "history": history,
    }, raw


def edit_analysis(model, world, out, device, n_facts=8, nodes=(0, 20, 100, 200), lr=0.01):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cases = edit_cases(world, n_facts=n_facts)
    serializable_cases = [
        {key: value for key, value in case.items() if key not in {"tasks", "original_d_rows"}}
        | {"task_sizes": {key: len(rows) for key, rows in case["tasks"].items()}}
        for case in cases
    ]
    # This is written before evaluating or updating any case.
    write_json(out / "data-selected-edit-cases.json", serializable_cases)
    records = []
    for number, case in enumerate(cases):
        for arm in ("edit", "review"):
            record, raw = edit_one(model, case, world, device, arm, nodes=nodes, lr=lr)
            np.savez_compressed(out / f"case{number:02d}-{arm}-raw.npz", **raw)
            write_json(out / f"case{number:02d}-{arm}.json", record)
            records.append(record)
    report = {
        "phase": "development",
        "created_utc": utc(),
        "n_facts": n_facts,
        "nodes": list(nodes),
        "lr": lr,
        "optimizer": "Adam, no weight decay",
        "objective": "0.5 target atomic full-token CE + 0.5 fixed 32-atomic replay full-token CE",
        "trained_compositions": 0,
        "unique_block_edited": 0,
        "executions_affected_by_shared_edit": model.repeats,
        "independent_layers": len(model.blocks),
        "execution_depth": model.effective_depth,
        "scope_difference": (
            "Loop block-0 update applies at every repeat; ordinary block-0 applies once"
        ),
        "records": records,
        "wall_seconds": time.perf_counter() - started,
        "limits": [
            "Edits use the same fixed budget; failed target changes remain in results",
            "MLP-only parameter fine-tuning is not a constrained-preservation theorem",
            "Each case independently starts from the same checkpoint",
            "A single development world is not independent confirmation",
        ],
    }
    write_json(out / "edit-summary.json", report)
    return report


def load_run(run_dir, checkpoint, device):
    run_dir, checkpoint = Path(run_dir), Path(checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = construct(saved["spec"], device).eval()
    model.load_state_dict(saved["model"], strict=True)
    arrays = np.load(run_dir / "world.npz", allow_pickle=False)
    world = {key: arrays[key] for key in arrays.files}
    world["metadata"] = json.loads((run_dir / "world-metadata.json").read_text())
    from .depth_step import audit_world

    audit = audit_world(world)
    if audit["dataset_sha256"] != saved["spec"].get("frozen_data_sha256", audit["dataset_sha256"]):
        raise ValueError("Checkpoint and archived data digest disagree")
    return (
        model,
        world,
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint),
            "step": saved["step"],
            "spec": saved["spec"],
            "dataset_sha256": audit["dataset_sha256"],
        },
    )
