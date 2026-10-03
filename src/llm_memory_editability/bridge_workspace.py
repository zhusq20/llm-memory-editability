"""Causal entity-coordinate interventions using an averaged prefix Jacobian.

This is a restricted adaptation of Anthropic's Jacobian lens: calibration and
donors contain only the first fact's query prefix. No relation suffix or answer
gradient is used to construct a direction. Execution layers count shared block
occurrences separately. All model parameters remain frozen.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import utc, write_json
from .latent_scaling import model_digest
from .storage_composition import file_hash

SWAPS = ("swap", "swap_x2", "random_swap", "full_prefix", "swap_back")
ERASURES = (
    "erase_state",
    "erase_mlp",
    "erase_attention",
    "random_state",
    "random_mlp",
    "random_attention",
    "restore_mlp",
    "fixed_entity_mlp",
    "fixed_entity_random_mlp",
)


class ModelView:
    """Expose native attention/MLP increments without replacing trained modules."""

    def __init__(self, model):
        if model.training:
            raise ValueError("Interventions require eval mode")
        self.model = model
        self.large = hasattr(model, "transformer")
        if self.large:
            t = model.transformer
            self.embedding, self.position, self.norm = t.wte, t.wpe, t.ln_f
            self.blocks = list(t.h) * model.repeats
            self.heads = model.config.n_head
        else:
            self.embedding, self.position, self.norm = model.token, model.position, model.ln_final
            self.blocks = list(model.iter_blocks())
            self.heads = model.config.heads
        self.width = self.embedding.weight.shape[1]
        self.device = self.embedding.weight.device

    def embed(self, tokens):
        return self.embedding(tokens) + self.position(
            torch.arange(tokens.shape[1], device=self.device)
        )

    def parts(self, x, block, *, math_attention=False):
        if self.large:
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(self.width, dim=-1)
            project, ln2 = block.attn.c_proj, block.ln_2
        else:
            z = block.ln1(x)
            q, k, v = block.attention.qkv(z).chunk(3, dim=-1)
            project, ln2 = block.attention.proj, block.ln2
        shape = (len(x), x.shape[1], self.heads, self.width // self.heads)
        q, k, v = (a.reshape(shape).transpose(1, 2) for a in (q, k, v))
        if math_attention:
            # The explicit FP32 expression permits batched VJPs on torch 2.6.
            scores = q @ k.transpose(-1, -2) / (self.width // self.heads) ** 0.5
            mask = torch.ones(x.shape[1], x.shape[1], device=self.device, dtype=torch.bool).tril()
            y = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = project(y.transpose(1, 2).reshape(len(x), x.shape[1], self.width))
        after_attention = x + attention
        mlp = block.mlp(ln2(after_attention))
        return after_attention + mlp, attention, mlp

    def hidden(
        self, tokens, *, layer=None, sender=None, delta=None, capture=False, math_attention=False
    ):
        x, states, attentions, mlps = self.embed(tokens), [], [], []
        for index, block in enumerate(self.blocks):
            x, attention, mlp = self.parts(x, block, math_attention=math_attention)
            if capture:
                states.append(x[:, sender].detach())
                attentions.append(attention[:, sender].detach())
                mlps.append(mlp[:, sender].detach())
            if index == layer:
                if sender is None or delta is None or delta.shape != x[:, sender].shape:
                    raise ValueError("A patch requires one delta per example at a fixed sender")
                mask = F.one_hot(torch.tensor(sender, device=self.device), x.shape[1]).to(x.dtype)
                x = x + mask[None, :, None] * delta[:, None, :]
        if capture:
            return x, tuple(torch.stack(a) for a in (states, attentions, mlps))
        return x

    def logits(self, tokens, **patch):
        return F.linear(self.norm(self.hidden(tokens, **patch)[:, -1]), self.embedding.weight)


def average_prefix_jacobian(view, prefixes, sender, *, chunk=32):
    """E[d h_final,sender / d h_layer,sender], with labels absent from the graph.

    A shared shift applied to all independent calibration prompts differentiates
    their mean output, exactly averaging their Jacobians. Parameters need no
    gradients. The endpoint is the residual before the final LayerNorm.
    """
    if any(p.requires_grad for p in view.model.parameters()):
        raise ValueError("Freeze model parameters before constructing the lens")
    with torch.no_grad():
        x, boundaries = view.embed(prefixes), []
        for block in view.blocks:
            x, _, _ = view.parts(x, block, math_attention=True)
            boundaries.append(x.detach())
    jacobians = []
    identity = torch.eye(view.width, device=view.device)
    for layer, boundary in enumerate(boundaries):
        shift = torch.zeros(view.width, device=view.device, requires_grad=True)
        mask = F.one_hot(torch.tensor(sender, device=view.device), prefixes.shape[1]).float()
        x = boundary + mask[None, :, None] * shift[None, None, :]
        for block in view.blocks[layer + 1 :]:
            x, _, _ = view.parts(x, block, math_attention=True)
        endpoint = x[:, sender].mean(0)
        rows = []
        for start in range(0, view.width, chunk):
            rows.append(
                torch.autograd.grad(
                    endpoint,
                    shift,
                    grad_outputs=identity[start : start + chunk],
                    is_grads_batched=True,
                    retain_graph=start + chunk < view.width,
                )[0].detach()
            )
        jacobians.append(torch.cat(rows))
    return torch.stack(jacobians)


def swap_coordinates(state, basis):
    """Swap two coefficients in a nonorthogonal frame, preserving its complement."""
    if basis.shape != (*state.shape, 2):
        raise ValueError("Expected [example, width, two semantic axes]")
    coefficients = (torch.linalg.pinv(basis.double()) @ state.double()[..., None]).squeeze(-1)
    difference = coefficients.flip(-1) - coefficients
    return state + (basis.double() @ difference[..., None]).squeeze(-1).to(state.dtype)


def projection(state, direction):
    direction = F.normalize(direction, dim=-1)
    return (state * direction).sum(-1, keepdim=True) * direction


def match_norm(delta, reference):
    norm = delta.norm(dim=-1, keepdim=True)
    if bool(((norm < 1e-12) & (reference.norm(dim=-1, keepdim=True) > 1e-6)).any()):
        raise ValueError("Degenerate random control")
    return delta * reference.norm(dim=-1, keepdim=True) / norm.clamp_min(1e-12)


def control_observation(changed, baseline):
    """Record numerical control outcomes without silently dropping sensitive cases."""
    mask = np.any(changed != baseline, axis=1)
    return {
        "n_queries": len(baseline),
        "generated_rows_changed": int(mask.sum()),
        "answer_rows_changed": int((changed[:, 0] != baseline[:, 0]).sum()),
        "changed_query_indices": np.flatnonzero(mask).tolist(),
        "changed_fraction": float(mask.mean()),
        "exact": bool(not mask.any()),
    }


def _rank(seed, *values):
    return hashlib.sha256(json.dumps([seed, *map(int, values)]).encode()).digest()


def select_pairs(atoms, first_families, seed, per_family, *, ood=None):
    """Choose prefixes/donors and all shared continuations using graph truth only.

    Donors use the same r1 and first-fact family. Every pair has at least two
    distinct shared r2 with different original/counterfactual tails. A pair is
    selected before scoring; its exact same prefix intervention serves every r2.
    """
    atoms = np.asarray(atoms, dtype=np.int64)
    outgoing, relation = defaultdict(dict), defaultdict(list)
    for index, (head, r, tail) in enumerate(atoms):
        if int(r) in outgoing[int(head)]:
            raise ValueError("Ambiguous atomic truth")
        outgoing[int(head)][int(r)] = (int(tail), index)
        relation[int(r)].append(index)
    groups, queries, coverage = [], [], {}
    for name, indices in first_families.items():
        allowed = set(map(int, indices))
        eligible = []
        for first in sorted(allowed):
            head, r1, bridge = map(int, atoms[first])
            candidates = []
            for donor in relation[r1]:
                dh, _, db = map(int, atoms[donor])
                if donor not in allowed or dh == head or db == bridge:
                    continue
                common = sorted(set(outgoing[bridge]) & set(outgoing[db]))
                common = [
                    r
                    for r in common
                    if outgoing[bridge][r][0] != outgoing[db][r][0]
                    and not {outgoing[bridge][r][0], outgoing[db][r][0]} & {head, dh, bridge, db}
                ]
                if len(common) >= 2:
                    candidates.append((donor, common))
            if candidates:
                donor, common = min(candidates, key=lambda v: _rank(seed, first, v[0]))
                eligible.append((first, donor, common))
        eligible.sort(key=lambda v: _rank(seed, v[0]))
        coverage[name] = {"candidate_first_facts": len(allowed), "eligible": len(eligible)}
        for first, donor, common in eligible[:per_family]:
            head, r1, bridge = map(int, atoms[first])
            dh, _, db = map(int, atoms[donor])
            group = len(groups)
            groups.append([head, r1, bridge, dh, db, first, donor])
            for r2 in common:
                tail, second = outgoing[bridge][r2]
                ct, cf_second = outgoing[db][r2]
                kind = (
                    name
                    if ood is None
                    else ("ood" if ood[first] else "id") + ("_ood" if ood[second] else "_id")
                )
                queries.append((group, r2, tail, ct, second, cf_second, kind))
        coverage[name]["selected"] = min(len(eligible), per_family)
    if not groups:
        raise ValueError("No graph-eligible cross-relation pairs; do not reroll")
    return {
        "groups": np.asarray(groups, dtype=np.int64),
        "queries": np.asarray([r[:6] for r in queries], dtype=np.int64),
        "strata": np.asarray([r[6] for r in queries]),
        "counterfactual_strata": np.asarray(
            [
                r[6]
                if ood is None
                else ("ood" if ood[groups[r[0]][6]] else "id") + ("_ood" if ood[r[5]] else "_id")
                for r in queries
            ]
        ),
        "coverage": coverage,
    }


def prefixes(facts, large, metadata):
    facts = np.asarray(facts)
    if large:
        return np.column_stack(
            (facts[:, 0] + metadata["entity_offset"], facts[:, 1] + metadata["relation_offset"])
        )
    return np.column_stack(
        (np.full(len(facts), 2), facts[:, 0], np.full(len(facts), 3), facts[:, 1])
    )


def prompts(rows, large, metadata):
    if large:
        from .grokking_reproduction import encode_rows

        return encode_rows(rows, metadata)[0][:, : rows.shape[1] - 1]
    from .text_pretrain import atomic_sentence, composite_sentence

    render = atomic_sentence if rows.shape[1] == 3 else composite_sentence
    length = 5 if rows.shape[1] == 3 else 7
    return np.asarray([render(row)[:length] for row in rows])


def query_training_masks(world, payload, groups, queries, large):
    """Count composition exposure without using scores or answer correctness."""
    if large:
        support = round(payload["spec"]["phi"] * (~world["ood"]).sum())
        trained = world["chains"][world["train_order"][:support]][:, :3]
    else:
        trained = world["train_composite"][:, [0, 1, 3]]
    trained = set(map(tuple, trained))
    recipient = np.asarray([(groups[g, 0], groups[g, 1], r2) in trained for g, r2, *_ in queries])
    donor = np.asarray([(groups[g, 3], groups[g, 1], r2) in trained for g, r2, *_ in queries])
    return {
        "recipient_query_trained": recipient,
        "counterfactual_query_trained": donor,
        "both_queries_untrained": ~recipient & ~donor,
    }


def load_task(spec, device):
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    if "checkpoint_step" in spec:
        saved_step = payload.get("step", payload["spec"].get("steps"))
        assert saved_step == spec["checkpoint_step"], "Wrong checkpoint training step"
    if spec["family"] == "reproduction":
        from .grokking_reproduction import construct

        data = Path(spec["data"])
        metadata = json.loads((data / "complete.json").read_text())
        model = construct(payload["spec"], metadata, device)
        world = dict(np.load(data / "world.npz"))
        panels = dict(np.load(data / "panels.npz"))
        atoms = world["atoms"].astype(np.int64)
        families = {"id": np.flatnonzero(~world["ood"]), "ood": np.flatnonzero(world["ood"])}
        ood = world["ood"]
        entities = np.arange(
            metadata["entity_offset"], metadata["entity_offset"] + metadata["entities"]
        )
    else:
        from .representation_alignment import new_model

        model = new_model(payload["spec"], device)
        world = dict(np.load(Path(spec["checkpoint"]).parent / "world.npz"))
        metadata = {"entity_offset": 0}
        atoms = np.concatenate((world["common_atomic"], world["anchor_atomic"]))
        families = {}
        for name in ("familiar_test", "strict_test"):
            firsts = set(map(tuple, world[name][:, :3]))
            families[name] = np.asarray([i for i, a in enumerate(atoms) if tuple(a) in firsts])
        ood = None
        entities = np.arange(21, model.config.vocab_size)
        panels = {k: world[k] for k in ("familiar_test", "strict_test")}
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)
    return ModelView(model), payload, metadata, world, atoms, families, ood, entities, panels


@torch.no_grad()
def trace_in_batches(view, tokens, sender, batch_size, amp):
    collected = []
    for start in range(0, len(tokens), batch_size):
        batch = torch.as_tensor(tokens[start : start + batch_size], device=view.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            _, trace = view.hidden(batch, sender=sender, capture=True)
        collected.append(tuple(a.float() for a in trace))
    return tuple(torch.cat([a[j] for a in collected], dim=1) for j in range(3))


@torch.no_grad()
def generate(
    view,
    tokens,
    targets,
    counterfactual,
    metadata,
    batch_size,
    *,
    layer=None,
    sender=None,
    delta=None,
):
    """Greedy generation with feedback; scoring retains both target probabilities."""
    offset = metadata["entity_offset"]
    targets, counterfactual = targets + offset, counterfactual + offset
    generated, original_logp, cf_logp, top_ids, top_scores = [], [], [], [], []
    for start in range(0, len(tokens), batch_size):
        x = torch.as_tensor(tokens[start : start + batch_size], device=view.device)
        patch = (
            {}
            if layer is None
            else {"layer": layer, "sender": sender, "delta": delta[start : start + batch_size]}
        )
        draws = []
        for index in range(2 if view.large else 3):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=view.large):
                logits = view.logits(x, **patch).float()
            if index == 0:
                logs = logits.log_softmax(-1)
                arange = torch.arange(len(x), device=view.device)
                original_logp.append(
                    logs[
                        arange,
                        torch.as_tensor(targets[start : start + batch_size], device=view.device),
                    ]
                    .cpu()
                    .numpy()
                )
                cf_logp.append(
                    logs[
                        arange,
                        torch.as_tensor(
                            counterfactual[start : start + batch_size], device=view.device
                        ),
                    ]
                    .cpu()
                    .numpy()
                )
                scores, ids = logs.topk(min(10, logs.shape[-1]), dim=-1)
                top_ids.append(ids.cpu().numpy())
                top_scores.append(scores.cpu().numpy())
            answer = logits.argmax(-1)
            draws.append(answer.cpu().numpy())
            x = torch.cat((x, answer[:, None]), dim=1)
        generated.append(np.stack(draws, axis=1))
    return {
        "generated": np.concatenate(generated),
        "original_logp": np.concatenate(original_logp),
        "cf_logp": np.concatenate(cf_logp),
        "top_ids": np.concatenate(top_ids),
        "top_logp": np.concatenate(top_scores),
    }


def summarize(result, targets, counterfactual, metadata, strata, groups=None):
    generated = result["generated"]
    original = generated[:, 0] == targets + metadata["entity_offset"]
    cf = generated[:, 0] == counterfactual + metadata["entity_offset"]
    format_ok = (
        generated[:, 1] == metadata["end_marker"]
        if "end_marker" in metadata
        else (generated[:, 1] == 5) & (generated[:, 2] == 1)
    )
    output = {}
    for name in ("all", *sorted(set(strata))):
        mask = np.ones(len(strata), dtype=bool) if name == "all" else strata == name
        row = {
            "n_queries": int(mask.sum()),
            "answer_accuracy": float(original[mask].mean()),
            "accuracy": float((original & format_ok)[mask].mean()),
            "cf_answer_accuracy": float(cf[mask].mean()),
            "cf_accuracy": float((cf & format_ok)[mask].mean()),
            "format_accuracy": float(format_ok[mask].mean()),
            "answer_nll": float(-result["original_logp"][mask].mean()),
            "cf_nll": float(-result["cf_logp"][mask].mean()),
        }
        if groups is not None:
            unique = np.unique(groups[mask])
            row["n_first_facts"] = len(unique)
            row["all_relations_cf_accuracy"] = float(
                np.mean([cf[mask & (groups == g)].all() for g in unique])
            )
            row["mean_first_fact_cf_accuracy"] = float(
                np.mean([cf[mask & (groups == g)].mean() for g in unique])
            )
        output[name] = row
    return output


def run(spec, settings, out, gpu):
    """Run one checkpoint with all execution layers, including failed interventions."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(f"Do not overwrite an attempt: {out}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{gpu}")
    began = time.perf_counter()
    view, payload, metadata, world, atoms, families, ood, entities, panels = load_task(spec, device)
    digest_before = model_digest(view.model)
    write_json(
        out / "run.json",
        {
            "spec": {
                **spec,
                "settings": settings,
                "checkpoint_step": payload.get("step", payload["spec"].get("steps")),
            },
            "checkpoint_sha256": file_hash(spec["checkpoint"]),
            "initial_model_sha256": digest_before,
            "world_sha256": file_hash(Path(spec["data"]) / "world.npz")
            if view.large
            else file_hash(Path(spec["checkpoint"]).parent / "world.npz"),
            "job_type": "causal-intervention",
            "pid": os.getpid(),
            "gpu": gpu,
            "gpu_name": torch.cuda.get_device_name(gpu),
            "dtype": "BF16 native / FP32 Jacobian" if view.large else "FP32",
            "parameters": sum(p.numel() for p in view.model.parameters()),
            "vocab_size": len(view.embedding.weight),
            "tags": ["bridge-workspace", spec["phase"], "no-training"],
            "created_utc": utc(),
        },
    )
    history, cursor = [], 0

    def log(stage, metrics):
        nonlocal cursor
        history.append(
            {
                "step": cursor,
                "metrics": metrics,
                "wall_seconds": time.perf_counter() - began,
                "checkpoint_training_step": payload.get("step", payload["spec"].get("steps", 0)),
            }
        )
        write_json(out / "learning.json", history)
        write_json(out / "status.json", {"state": "running", "stage": stage, "step": cursor})
        print(
            json.dumps(
                {
                    "run": spec["name"],
                    "stage": stage,
                    "index": cursor,
                    "wall_seconds": history[-1]["wall_seconds"],
                }
            ),
            flush=True,
        )
        cursor += 1

    log("loaded", {})
    pairs = select_pairs(
        atoms, families, settings["selection_seed"], settings["prefixes_per_family"], ood=ood
    )
    groups, q = pairs["groups"], pairs["queries"]
    used_first = set(groups[:, 5]) | set(groups[:, 6])
    calibration = np.asarray([i for i in range(len(atoms)) if i not in used_first])
    calibration = sorted(calibration, key=lambda i: _rank(settings["calibration_seed"], i))[
        : settings["calibration_prompts"]
    ]
    sender = 1 if view.large else 3
    cp = prefixes(atoms[calibration], view.large, metadata)
    tensors = torch.as_tensor(cp, device=device)
    jacobians = average_prefix_jacobian(view, tensors, sender, chunk=settings["jacobian_chunk"])
    torch.save(
        {"jacobians": jacobians.cpu(), "calibration_atomic_indices": calibration, "prefixes": cp},
        out / "lens.pt",
    )
    log("jacobian_lens", {"calibration_prompts": len(cp), "executed_layers": len(view.blocks)})
    trace = trace_in_batches(
        view,
        prefixes(groups[:, :3], view.large, metadata),
        sender,
        settings["batch_size"],
        view.large,
    )
    donor_facts = np.column_stack((groups[:, 3], groups[:, 1], groups[:, 4]))
    donor_states = trace_in_batches(
        view,
        prefixes(donor_facts, view.large, metadata),
        sender,
        settings["batch_size"],
        view.large,
    )[0]
    rows = (
        np.column_stack((groups[q[:, 0], 0], groups[q[:, 0], 1], q[:, 1], q[:, 2]))
        if view.large
        else np.column_stack(
            (groups[q[:, 0], 0], groups[q[:, 0], 1], groups[q[:, 0], 2], q[:, 1], q[:, 2])
        )
    )
    tokens = prompts(rows, view.large, metadata)
    training_masks = query_training_masks(world, payload, groups, q, view.large)
    # Fix a native panel per stratum without consulting any predictions.
    native_rows, native_strata = [], []
    for name in (
        ("test_ii", "test_io", "test_oi", "test_oo")
        if view.large
        else ("familiar_test", "strict_test")
    ):
        selected = panels[name][: settings["native_per_stratum"]]
        native_rows.append(selected)
        native_strata.extend([name] * len(selected))
    native_rows, native_strata = np.concatenate(native_rows), np.asarray(native_strata)
    lookup = {(int(h), int(r)): (int(t), i) for i, (h, r, t) in enumerate(atoms)}
    native_facts = np.asarray([(h, r, lookup[int(h), int(r)][0]) for h, r in native_rows[:, :2]])
    unique_bridges, bridge_counts = np.unique(native_facts[:, 2], return_counts=True)
    fixed_entity = int(unique_bridges[bridge_counts.argmax()])
    fixed_target = native_facts[:, 2] == fixed_entity
    ntokens = prompts(native_rows, view.large, metadata)
    ntrace = trace_in_batches(
        view,
        prefixes(native_facts, view.large, metadata),
        sender,
        settings["batch_size"],
        view.large,
    )
    all_needed = set(groups[:, 5]) | set(groups[:, 6]) | set(q[:, 4]) | set(q[:, 5])
    for row, fact in zip(native_rows, native_facts, strict=True):
        r2 = row[2] if view.large else row[3]
        all_needed.add(lookup[int(fact[0]), int(fact[1])][1])
        all_needed.add(lookup[int(fact[2]), int(r2)][1])
    needed_ids = np.asarray(sorted(all_needed))
    needed = atoms[needed_ids]
    atrace = trace_in_batches(
        view, prefixes(needed, view.large, metadata), sender, settings["batch_size"], view.large
    )
    torch.save(
        {
            "recipient": tuple(a.cpu() for a in trace),
            "donor_states": donor_states.cpu(),
            "native": tuple(a.cpu() for a in ntrace),
            "atomic": tuple(a.cpu() for a in atrace),
        },
        out / "prefix-traces.pt",
    )
    atomic = generate(
        view,
        prompts(needed, view.large, metadata),
        needed[:, -1],
        needed[:, -1],
        metadata,
        settings["batch_size"],
    )
    atomic_ok = atomic["generated"][:, 0] == needed[:, -1] + metadata["entity_offset"]
    atomic_correct = dict(zip(needed_ids, atomic_ok, strict=True))
    prereq = np.asarray(
        [
            all(atomic_correct[int(i)] for i in (groups[g, 5], groups[g, 6], second, cf_second))
            for g, _, _, _, second, cf_second in q
        ]
    )
    np.savez_compressed(
        out / "selection.npz",
        **{k: v for k, v in pairs.items() if k != "coverage"},
        rows=rows,
        native_rows=native_rows,
        native_strata=native_strata,
        fixed_entity=np.asarray(fixed_entity),
        fixed_entity_target=fixed_target,
        **training_masks,
        prerequisite_correct=prereq,
        needed_atomic_indices=needed_ids,
        **{"atomic_" + k: v for k, v in atomic.items()},
    )
    write_json(out / "coverage.json", pairs["coverage"])
    # Independent native implementation equivalence and prefix-causality audit.
    audit_tokens = torch.as_tensor(tokens[: min(8, len(tokens))], device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=view.large):
        native = view.model(
            audit_tokens,
            positions=torch.full((len(audit_tokens), 1), audit_tokens.shape[1] - 1, device=device),
        )[:, 0].float()
        traced = view.logits(audit_tokens).float()
        _, full_trace = view.hidden(audit_tokens, sender=sender, capture=True)
    causal_error = float(
        (full_trace[0].float() - trace[0][:, q[: len(audit_tokens), 0]]).abs().max()
    )
    torch.testing.assert_close(
        full_trace[0].float(),
        trace[0][:, q[: len(audit_tokens), 0]],
        atol=0.05 if view.large else 2e-5,
        rtol=0.005 if view.large else 2e-5,
    )
    torch.testing.assert_close(
        native, traced, atol=0.05 if view.large else 1e-5, rtol=0.005 if view.large else 1e-5
    )
    if not torch.equal(native.argmax(-1), traced.argmax(-1)):
        raise AssertionError("Native/traced greedy answers differ")
    summary = {}

    def measure(
        family,
        condition,
        layer,
        input_tokens,
        target,
        cf_target,
        strata,
        delta=None,
        query_groups=None,
    ):
        result = generate(
            view,
            input_tokens,
            target,
            cf_target,
            metadata,
            settings["batch_size"],
            layer=layer if delta is not None else None,
            sender=sender,
            delta=delta,
        )
        np.savez_compressed(out / f"{family}-{condition}-l{layer:02d}.npz", **result)
        metrics = summarize(result, target, cf_target, metadata, strata, query_groups)
        if family == "swap" and prereq.any():
            metrics["prerequisites_correct"] = summarize(
                {k: a[prereq] for k, a in result.items()},
                target[prereq],
                cf_target[prereq],
                metadata,
                strata[prereq],
                query_groups[prereq],
            )["all"]
        if family == "swap" and training_masks["both_queries_untrained"].any():
            mask = training_masks["both_queries_untrained"]
            metrics["both_queries_untrained"] = summarize(
                {k: a[mask] for k, a in result.items()},
                target[mask],
                cf_target[mask],
                metadata,
                strata[mask],
                query_groups[mask],
            )["all"]
        if family == "native" and (condition.startswith("fixed_entity") or condition == "baseline"):
            for label, mask in (
                ("fixed_entity_target", fixed_target),
                ("fixed_entity_unrelated", ~fixed_target),
            ):
                if mask.any():
                    metrics[label] = summarize(
                        {k: a[mask] for k, a in result.items()},
                        target[mask],
                        cf_target[mask],
                        metadata,
                        strata[mask],
                    )["all"]
        summary[f"{family}/{condition}/{layer}"] = metrics
        write_json(out / "summary.json", summary)
        log(f"{family}-{condition}-l{layer}", {family: {condition: {f"layer_{layer}": metrics}}})

    measure("swap", "baseline", -1, tokens, q[:, 2], q[:, 3], pairs["strata"], query_groups=q[:, 0])
    measure(
        "native", "baseline", -1, ntokens, native_rows[:, -1], native_rows[:, -1], native_strata
    )
    measure(
        "atomic",
        "baseline",
        -1,
        prompts(needed, view.large, metadata),
        needed[:, -1],
        needed[:, -1],
        np.full(len(needed), "needed_atoms"),
    )
    states, attentions, mlps = trace
    rng = torch.Generator(device=device).manual_seed(settings["rotation_seed"])
    rotation, _ = torch.linalg.qr(torch.randn(view.width, view.width, device=device, generator=rng))
    swap_errors, restore_errors = [], []
    diagnostics = []
    axis_families = settings.get("axis_families", ["prefix_jacobian"])
    if not set(axis_families) <= {"prefix_jacobian", "input_embedding"}:
        raise ValueError("Unknown direction family")
    for layer, jacobian, axis_family in (
        (layer, j, family) for layer, j in enumerate(jacobians) for family in axis_families
    ):
        suffix = "_embedding" if axis_family == "input_embedding" else ""
        directions = F.normalize(
            view.embedding.weight[torch.as_tensor(entities, device=device)].float()
            @ (torch.eye(view.width, device=device) if suffix else jacobian),
            dim=-1,
        )
        index = {int(e): i for i, e in enumerate(entities)}
        source = directions[[index[int(b + metadata["entity_offset"])] for b in groups[:, 2]]]
        dest = directions[[index[int(b + metadata["entity_offset"])] for b in groups[:, 4]]]
        basis = torch.stack((source, dest), dim=-1)
        swapped = swap_coordinates(states[layer], basis)
        restored = swap_coordinates(swapped, basis)
        swap_errors.append(float((restored - states[layer]).abs().max()))
        random_basis = rotation[None] @ basis
        swap_delta = swapped - states[layer]
        torch.save(
            {
                "basis": basis.cpu(),
                "rotation": rotation.cpu(),
                "swap_delta": swap_delta.cpu(),
                "axis_family": axis_family,
            },
            out / f"patch-{axis_family}-l{layer:02d}.pt",
        )
        deltas = {
            "swap": swap_delta,
            "swap_x2": 2 * swap_delta,
            "random_swap": match_norm(
                swap_coordinates(states[layer], random_basis) - states[layer], swap_delta
            ),
            "full_prefix": donor_states[layer] - states[layer],
            "swap_back": restored - states[layer],
        }
        for condition in SWAPS:
            measure(
                "swap",
                condition + suffix,
                layer,
                tokens,
                q[:, 2],
                q[:, 3],
                pairs["strata"],
                deltas[condition][q[:, 0]],
                q[:, 0],
            )
        native_direction = directions[
            [index[int(b + metadata["entity_offset"])] for b in native_facts[:, 2]]
        ]
        native_state, native_attention, native_mlp = (a[layer] for a in ntrace)
        fixed_direction = directions[index[fixed_entity + metadata["entity_offset"]]][
            None
        ].expand_as(native_direction)
        fixed_mlp = projection(native_mlp, fixed_direction)
        removed_mlp = projection(native_mlp, native_direction)
        # All sublayers still execute. Only the selected sender's semantic component
        # is subtracted from the actual local update at the boundary.
        local_restore = native_state - removed_mlp + removed_mlp
        restore_errors.append(float((local_restore - native_state).abs().max()))
        erasures = {
            "erase_state": -projection(native_state, native_direction),
            "erase_mlp": -removed_mlp,
            "erase_attention": -projection(native_attention, native_direction),
            "random_state": match_norm(
                -projection(native_state, native_direction @ rotation.T),
                -projection(native_state, native_direction),
            ),
            "random_mlp": match_norm(
                -projection(native_mlp, native_direction @ rotation.T), -removed_mlp
            ),
            "random_attention": match_norm(
                -projection(native_attention, native_direction @ rotation.T),
                -projection(native_attention, native_direction),
            ),
            "restore_mlp": local_restore - native_state,
            "fixed_entity_mlp": -fixed_mlp,
            "fixed_entity_random_mlp": match_norm(
                -projection(native_mlp, fixed_direction @ rotation.T),
                -fixed_mlp,
            ),
        }
        for condition in ERASURES:
            measure(
                "native",
                condition + suffix,
                layer,
                ntokens,
                native_rows[:, -1],
                native_rows[:, -1],
                native_strata,
                erasures[condition],
            )
        # Relevant single-hop facts are scored under exactly the same local deletion.
        adirection = directions[[index[int(b + metadata["entity_offset"])] for b in needed[:, 2]]]
        atomic_delta = -projection(atrace[2][layer], adirection)
        measure(
            "atomic",
            "erase_mlp" + suffix,
            layer,
            prompts(needed, view.large, metadata),
            needed[:, -1],
            needed[:, -1],
            np.full(len(needed), "needed_atoms"),
            atomic_delta,
        )
        with torch.no_grad():
            raw = F.linear(view.norm(states[layer]), view.embedding.weight).argmax(-1)
            lens = F.linear(view.norm(states[layer] @ jacobian.T), view.embedding.weight).argmax(-1)
        diagnostics.append(
            {
                "layer": layer,
                "axis_family": axis_family,
                "native_bridge_readout_accuracy": float(
                    (
                        raw
                        == torch.as_tensor(groups[:, 2] + metadata["entity_offset"], device=device)
                    )
                    .float()
                    .mean()
                ),
                "prefix_jacobian_bridge_readout_accuracy": float(
                    (
                        lens
                        == torch.as_tensor(groups[:, 2] + metadata["entity_offset"], device=device)
                    )
                    .float()
                    .mean()
                ),
                "swap_norm_mean": float(swap_delta.norm(dim=-1).mean()),
                "native_mlp_removed_norm_mean": float(removed_mlp.norm(dim=-1).mean()),
            }
        )
    digest_after = model_digest(view.model)
    if digest_before != digest_after or any(p.grad is not None for p in view.model.parameters()):
        raise AssertionError("Intervention changed model parameters or accumulated gradients")
    write_json(
        out / "implementation-audit.json",
        {
            "parameters_unchanged": True,
            "native_tracer_max_logit_error": float((native - traced).abs().max()),
            "prefix_causal_state_max_error": causal_error,
            "swap_inverse_max_errors": swap_errors,
            "mlp_restore_max_errors": restore_errors,
            "n_needed_atomic_facts": len(needed),
            "needed_atomic_answer_accuracy": float(atomic_ok.mean()),
            "swap_prerequisite_query_coverage": float(prereq.mean()),
            "scoring": (
                "greedy feedback, original/counterfactual answer plus termination; "
                "query and first-fact denominators separate"
            ),
        },
    )
    write_json(out / "diagnostics.json", diagnostics)
    log("complete", {"complete": {"evaluations": len(summary)}})
    write_json(
        out / "interventions-complete.json",
        {
            "finished_utc": utc(),
            "evaluations": len(summary),
            "model_sha256": digest_after,
            "wall_seconds": time.perf_counter() - began,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "new_training_updates": 0,
        },
    )
    write_json(out / "status.json", {"state": "awaiting_reload_audit", "step": cursor - 1})


def audit_run(out, gpu):
    """Reload in an independent process and reproduce fixed raw prediction subsets."""
    out = Path(out)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    metadata_run = json.loads((out / "run.json").read_text())
    spec = metadata_run["spec"]
    settings = spec["settings"]
    view, _, metadata, _, atoms, _, _, entities, _ = load_task(spec, f"cuda:{gpu}")
    assert model_digest(view.model) == metadata_run["initial_model_sha256"]
    assert os.getpid() != metadata_run["pid"], "Reload must use a separate process"
    selection = dict(np.load(out / "selection.npz"))
    groups, queries = selection["groups"], selection["queries"]
    jacobians = torch.load(out / "lens.pt", map_location=view.device, weights_only=False)[
        "jacobians"
    ]
    sender = 1 if view.large else 3
    trace = trace_in_batches(
        view,
        prefixes(groups[:, :3], view.large, metadata),
        sender,
        settings["batch_size"],
        view.large,
    )
    states = trace[0]
    native_rows = selection["native_rows"]
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    facts = np.asarray([(h, r, lookup[int(h), int(r)]) for h, r in native_rows[:, :2]])
    ntrace = trace_in_batches(
        view, prefixes(facts, view.large, metadata), sender, settings["batch_size"], view.large
    )
    swap_indices = np.unique(np.linspace(0, len(queries) - 1, min(32, len(queries)), dtype=int))
    native_indices = np.unique(
        np.linspace(0, len(native_rows) - 1, min(32, len(native_rows)), dtype=int)
    )
    summary = json.loads((out / "summary.json").read_text())
    comparisons, max_logp_error, identity_controls = [], 0.0, []

    def check(family, condition, layer, indices, rows, target, cf, delta=None):
        nonlocal max_logp_error

        # Baseline is independently executed through the model's own forward.
        class NativeView(ModelView):
            def logits(self, tokens, **_):
                positions = torch.full((len(tokens), 1), tokens.shape[1] - 1, device=self.device)
                return self.model(tokens, positions=positions)[:, 0]

        actual = generate(
            NativeView(view.model) if condition == "baseline" else view,
            prompts(rows, view.large, metadata),
            target,
            cf,
            metadata,
            settings["batch_size"],
            layer=layer if delta is not None else None,
            sender=sender,
            delta=delta,
        )
        # Preserve original batch shapes: compacting selected rows changes BF16
        # GEMM rounding. Select the audit subset only after the full generation.
        actual = {key: value[indices] for key, value in actual.items()}
        expected = dict(np.load(out / f"{family}-{condition}-l{layer:02d}.npz"))
        if not np.array_equal(actual["generated"], expected["generated"][indices]):
            raise AssertionError(f"Reload generation mismatch: {family}/{condition}/{layer}")
        for key in ("original_logp", "cf_logp"):
            error = float(np.max(np.abs(actual[key] - expected[key][indices])))
            max_logp_error = max(max_logp_error, error)
            np.testing.assert_allclose(
                actual[key],
                expected[key][indices],
                atol=0.04 if view.large else 3e-5,
                rtol=0.002 if view.large else 2e-5,
            )
        comparisons.append(
            {"family": family, "condition": condition, "layer": layer, "queries": len(indices)}
        )

    check("swap", "baseline", -1, swap_indices, selection["rows"], queries[:, 2], queries[:, 3])
    check(
        "native",
        "baseline",
        -1,
        native_indices,
        native_rows,
        native_rows[:, -1],
        native_rows[:, -1],
    )
    entity_index = {int(e): i for i, e in enumerate(entities)}
    for layer, jacobian in enumerate(jacobians):
        for axis_family in settings.get("axis_families", ["prefix_jacobian"]):
            suffix = "_embedding" if axis_family == "input_embedding" else ""
            directions = F.normalize(
                view.embedding.weight[torch.as_tensor(entities, device=view.device)].float()
                @ (torch.eye(view.width, device=view.device) if suffix else jacobian),
                dim=-1,
            )
            source = directions[
                [entity_index[int(b + metadata["entity_offset"])] for b in groups[:, 2]]
            ]
            dest = directions[
                [entity_index[int(b + metadata["entity_offset"])] for b in groups[:, 4]]
            ]
            delta = (
                swap_coordinates(states[layer], torch.stack((source, dest), dim=-1)) - states[layer]
            )
            check(
                "swap",
                "swap" + suffix,
                layer,
                swap_indices,
                selection["rows"],
                queries[:, 2],
                queries[:, 3],
                delta[queries[:, 0]],
            )
            native_direction = directions[
                [entity_index[int(b + metadata["entity_offset"])] for b in facts[:, 2]]
            ]
            native_delta = -projection(ntrace[2][layer], native_direction)
            check(
                "native",
                "erase_mlp" + suffix,
                layer,
                native_indices,
                native_rows,
                native_rows[:, -1],
                native_rows[:, -1],
                native_delta,
            )
            # Structural and numerical controls must retain native generation.
            for family, condition in (
                ("swap", "swap_back" + suffix),
                ("native", "restore_mlp" + suffix),
            ):
                raw = np.load(out / f"{family}-{condition}-l{layer:02d}.npz")["generated"]
                baseline = np.load(out / f"{family}-baseline-l-1.npz")["generated"]
                identity_controls.append(
                    {
                        "family": family,
                        "condition": condition,
                        "layer": layer,
                        **control_observation(raw, baseline),
                    }
                )
            if layer == len(view.blocks) - 1:
                for key in summary:
                    family, condition, occurrence = key.split("/")
                    if int(occurrence) != layer or family == "atomic":
                        continue
                    raw = np.load(out / f"{family}-{condition}-l{layer:02d}.npz")["generated"]
                    baseline = np.load(out / f"{family}-baseline-l-1.npz")["generated"]
                    assert np.array_equal(raw, baseline), (
                        key,
                        "final sender must not affect later positions",
                    )
    assert model_digest(view.model) == metadata_run["initial_model_sha256"]
    previous = out / "implementation-audit.json"
    if not previous.exists() and (out / "audit.json").exists():
        previous.write_bytes((out / "audit.json").read_bytes())
    write_json(
        out / "audit.json",
        {
            "independent_process": True,
            "pid": os.getpid(),
            "gpu": gpu,
            "finished_utc": utc(),
            "parameters_unchanged": True,
            "generations_exact": True,
            "identity_controls_exact": all(c["exact"] for c in identity_controls),
            "identity_control_observations": identity_controls,
            "scope": (
                "Fixed subset reload predictions exact; numerical inverse/restore "
                "outcomes recorded separately"
            ),
            "max_log_probability_error": max_logp_error,
            "evaluations": comparisons,
            "implementation_audit_sha256": file_hash(previous),
            "audit_source_sha256": file_hash(__file__),
        },
    )
    if (out / "interventions-complete.json").exists():
        complete = json.loads((out / "interventions-complete.json").read_text())
        complete["independently_reloaded"] = True
        complete["audit_finished_utc"] = utc()
        write_json(out / "complete.json", complete)
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "independently_reloaded": True,
            "step": json.loads((out / "learning.json").read_text())[-1]["step"],
        },
    )
    print(
        json.dumps(
            {
                "run": out.name,
                "independent_reload": "passed",
                "evaluations": len(comparisons),
                "max_log_probability_error": max_logp_error,
            }
        ),
        flush=True,
    )
