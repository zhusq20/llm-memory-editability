"""Separate relation reinjection from receiver-state scale under a fixed bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from diagnose_disco_five_hops import load_model  # noqa: E402
from evaluate_loop_beyond_four_hops import build, encode  # noqa: E402
from evaluate_loop_position_controlled_long_hops import sha, state_hash, write_json  # noqa: E402
from trace_invariant_five_hops import numpy, paths_for  # noqa: E402

from llm_memory_editability.self_decode_alignment import _loss_dtype, _readout  # noqa: E402

CONDITIONS = (
    "bridge_old",
    "bridge_scale",
    "bridge_add_relation",
    "bridge_add_wrong_relation",
    "bridge_scale_add_relation",
    "bridge_scale_add_wrong_relation",
    "bridge_zero",
    "bridge_constant",
    "bridge_fresh_atomic",
    "bridge_fresh_atomic_wrong_relation",
    "wrong_bridge_fresh_atomic",
    "native_add_relation",
    "native_add_wrong_relation",
)
INVARIANT_EXTRA = ("bridge_fresh_site", "bridge_fresh_site_wrong_relation")
CF_CONDITIONS = ("bridge_old", "bridge_scale", "bridge_zero", "bridge_constant")


def relation_counterfactuals(rows, metadata, world):
    """Fix 500 prefixes; vary only r5 through every relation, with balanced groups."""
    assert rows.shape == (5000, 7)
    for template in rows.reshape(10, 500, 7):
        assert len(np.unique(template[:, 1:6], axis=0)) == 1
        assert len(np.unique(template[:, 0])) == 500
    chosen = rows.reshape(10, 500, 7)[:, ::10].reshape(500, 7)
    count = world["permutations"].shape[1]
    assert count == 10
    repeated = np.repeat(chosen, count, axis=0)
    repeated[:, -2] = np.tile(np.arange(count), len(chosen))
    n = metadata["entities_per_graph"]
    graph = repeated[:, 0] // n
    node = repeated[:, 0] % n
    for relation in repeated[:, 1:-1].T:
        node = world["permutations"][graph, relation, node]
    repeated[:, -1] = node + graph * n
    return repeated


def match_norm(value, norm):
    return value * (norm / value.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(
        value.dtype
    )


@torch.no_grad()
def make_donors(model, tokens, bridge, wrong_bridge, relation_offset, relation_count):
    """Only input tokens and oracle e4 enter donors; no e5 or atomic block output."""
    scale = 2 if model.handoff == "invariant" else 1
    all_relations = torch.full(
        (relation_count, tokens.shape[1]),
        model.config.pad_token_id,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    all_relations[:, 1] = torch.arange(
        relation_offset, relation_offset + relation_count, device=tokens.device
    )
    all_atomic = scale * model._embedded(all_relations)[:, 1]
    target_norm = all_atomic.float().norm(dim=-1).median()
    all_tokens = model.transformer.wte(all_relations[:, 1])
    token_norm = all_tokens.float().norm(dim=-1).median()
    constant = match_norm(all_atomic.mean(0, keepdim=True), target_norm).expand(len(tokens), -1)
    atom = torch.full_like(tokens, model.config.pad_token_id)
    atom[:, 0], atom[:, 1] = bridge, tokens[:, -1]
    wrong_relation = (tokens[:, -1] - relation_offset + 1) % relation_count + relation_offset
    alternative = atom.clone()
    alternative[:, 1] = wrong_relation
    site = tokens.clone()
    site[:, 5] = wrong_relation
    atomic_input = scale * model._embedded(atom)
    wrong_atomic_input = scale * model._embedded(alternative)
    block = model.transformer.h[0]
    values = block.attn.c_attn(block.ln_1(atomic_input))[..., 2 * model.config.n_embd :]
    wrong_atom = atom.clone()
    wrong_atom[:, 0] = wrong_bridge
    wrong_values = block.attn.c_attn(block.ln_1(scale * model._embedded(wrong_atom)))[
        ..., 2 * model.config.n_embd :
    ]
    return dict(
        value=values[:, 0],
        wrong_value=wrong_values[:, 0],
        relation=match_norm(model.transformer.wte(tokens[:, -1]), token_norm),
        wrong_relation=match_norm(model.transformer.wte(wrong_relation), token_norm),
        fresh_atomic=atomic_input[:, 1],
        fresh_atomic_wrong_relation=wrong_atomic_input[:, 1],
        fresh_site=(scale * model._embedded(tokens))[:, 5],
        fresh_site_wrong_relation=(scale * model._embedded(site))[:, 5],
        constant=constant,
        target_norm=target_norm,
        token_norm=token_norm,
    )


def receiver(old, donor, condition):
    if condition in ("native", "bridge_old"):
        return old
    if condition == "bridge_scale":
        return match_norm(old, donor["target_norm"])
    if condition == "bridge_zero":
        return torch.zeros_like(old)
    if condition == "bridge_constant":
        return donor["constant"]
    if "fresh_atomic" in condition:
        return donor[
            "fresh_atomic_wrong_relation"
            if condition.endswith("wrong_relation")
            else "fresh_atomic"
        ]
    if "fresh_site" in condition:
        return donor[
            "fresh_site_wrong_relation" if condition.endswith("wrong_relation") else "fresh_site"
        ]
    if "add_" in condition:
        base = match_norm(old, donor["target_norm"]) if "scale" in condition else old
        return (
            base + donor["wrong_relation" if condition.endswith("wrong_relation") else "relation"]
        )
    raise ValueError(condition)


@torch.no_grad()
def trace(model, tokens, positions, donor, condition="native", at_round=0, rounds=16):
    """Patch only fifth block input; invariant's historical N/D remain intact."""
    assert tokens.shape[1] == 6 and len(model.transformer.h) == 1 and not model.training
    assert model.handoff in ("invariant", "additive")
    if model.handoff == "invariant":
        assert model.first_round_input == "state_plus_embedding"
    old_round = model._round
    block = model.transformer.h[0]
    current = 0
    captured, records = {}, {}

    def route(module, inputs):
        if current != at_round or condition.startswith("native"):
            return None
        v = donor["wrong_value" if condition.startswith("wrong_bridge") else "value"]
        changed = inputs[0].clone()
        changed[:, 5] = v.to(changed.dtype)
        return (changed,)

    def postattention(module, inputs):
        captured["postattention"] = inputs[0]

    def wrapped(value):
        captured["original_input"] = value[:, 5].clone()
        if current == at_round:
            value = value.clone()
            value[:, 5] = receiver(value[:, 5], donor, condition).to(value.dtype)
        captured["patched_input"] = value[:, 5].clone()
        raw = old_round(value)
        captured["raw"] = raw
        return raw

    handles = [
        block.attn.c_proj.register_forward_pre_hook(route),
        block.ln_2.register_forward_pre_hook(postattention),
    ]
    model._round = wrapped
    old_score = None
    if model.handoff == "invariant":
        old_score = model.score

        def score(candidate, *args):
            captured["candidate"] = candidate
            return old_score(candidate, *args)

        model.score = score

    def save(key, value):
        records.setdefault(key, []).append(numpy(value))

    try:
        injection = _loss_dtype(model._embedded(tokens))
        state = model.initial_state(injection) if model.handoff == "invariant" else None
        x = model._embedded(tokens)
        for current in range(1, rounds + 1):  # noqa: B007 -- consumed by hooks
            if model.handoff == "invariant":
                state, gamma = model.aggregate_round(state, injection)
                output = state[0]
                carry = output
                candidate = captured["candidate"]
                save("gamma", gamma[:, -1, 0])
            else:
                output = model._round(x)
                carry = model._discoloop_boundary(output)
                candidate = carry
                x = carry
            for key, value in [
                ("answer", output),
                ("raw", captured["raw"]),
                ("candidate", candidate),
                ("carry", carry),
                ("postattention", captured["postattention"]),
            ]:
                save(key, _readout(model, value, positions)[:, 0].argmax(-1))
            for key, value in [("raw_prefix", captured["raw"]), ("carry_prefix", carry)]:
                save(key, _readout(model, value).argmax(-1))
            for key in ["original_input", "patched_input"]:
                value = captured[key]
                save(key + "_norm", value.float().norm(dim=-1))
                save(key + "_token", _readout(model, value[:, None])[:, 0].argmax(-1))
            post = captured["postattention"][:, 5].float()
            save("attention_delta_norm", (post - captured["patched_input"].float()).norm(dim=-1))
            save("mlp_delta_norm", (captured["raw"][:, 5].float() - post).norm(dim=-1))
    finally:
        model._round = old_round
        if old_score is not None:
            model.score = old_score
        for handle in handles:
            handle.remove()
    return {k: np.stack(v) for k, v in records.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.16)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    p = args.persistent
    data = p / "loop-hop-development-v1/inputs/data"
    meta = json.loads((data / "complete.json").read_text())
    assert sha(data / "world.npz") == meta["world_sha256"]
    world = dict(np.load(data / "world.npz"))
    long_path = p / "loop-beyond-four-hops-20261020/beyond-panels.npz"
    panels = dict(np.load(long_path))
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(
        state="running",
        completed_pools=0,
        planned_pools=152,
        training_updates=0,
        world_sha256=sha(data / "world.npz"),
        panels_sha256=sha(long_path),
    )
    write_json(args.output / "status.json", status)
    validation = {}
    for arm in ["rope-ema", "invariant"]:
        if arm == "rope-ema":
            model, reference, checkpoint = load_model(p, meta, arm, device)
        else:
            config = json.loads(
                (ROOT / "configs/loop-native-handoff-development-v1.json").read_text()
            )
            ref = json.loads(
                (p / "loop-beyond-four-hops-20261020/invariant-1500k.json").read_text()
            )
            checkpoint = (
                p / "loop-native-handoff-development-v1/runs/invariant/checkpoint-1500000.pt"
            )
            assert sha(checkpoint) == ref["checkpoint_sha256"]
            model, step = build(
                config, meta, {**ref["model"], "checkpoint": str(checkpoint)}, device
            )
            assert step == 1500000
            model.requires_grad_(False)
            reference = p / "loop-beyond-four-hops-20261020/predictions/invariant-1500k"
        before = state_hash(model)
        folder = args.output / arm
        folder.mkdir()
        for panel in ["id_5", "ood_5", "id_5_cf", "ood_5_cf"]:
            counterfactual = panel.endswith("_cf")
            rows = panels[panel.removesuffix("_cf")]
            if counterfactual:
                rows = relation_counterfactuals(rows, meta, world)
            truth = paths_for(rows, meta, world)
            n, offset = meta["entities_per_graph"], meta["entity_offset"]
            graph = rows[:, 0] // n
            wrong_bridge = (truth[:, -2] - offset + 1) % n + graph * n + offset
            relations = rows[:, -2]
            relation_count = world["permutations"].shape[1]
            wrong_relations = (relations + 1) % relation_count
            wrong_relation_answer = (
                world["permutations"][graph, wrong_relations, (truth[:, -2] - offset) % n]
                + graph * n
                + offset
            )
            wrong_bridge_answer = (
                world["permutations"][graph, relations, (wrong_bridge - offset) % n]
                + graph * n
                + offset
            )
            assert (wrong_bridge_answer != truth[:, -1]).all()
            x, pos, _ = encode(rows, meta, 0)
            donors = {}
            for i in range(0, len(rows), 500):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    d = make_donors(
                        model,
                        torch.as_tensor(x[i : i + 500], device=device),
                        torch.as_tensor(truth[i : i + 500, -2], device=device),
                        torch.as_tensor(wrong_bridge[i : i + 500], device=device),
                        meta["relation_offset"],
                        relation_count,
                    )
                for key, value in d.items():
                    donors.setdefault(key, []).append(value)
            donors = {
                key: (torch.cat(vals, dim=0) if vals[0].ndim else vals[0])
                for key, vals in donors.items()
            }
            if arm == "rope-ema":
                assert torch.equal(donors["fresh_atomic"], donors["fresh_site"])
                assert torch.equal(
                    donors["fresh_atomic_wrong_relation"], donors["fresh_site_wrong_relation"]
                )
            np.savez_compressed(
                folder / f"{panel}-context.npz",
                rows=rows,
                truths=truth,
                wrong_relation_id=wrong_relations,
                wrong_relation_answer=wrong_relation_answer,
                wrong_bridge_answer=wrong_bridge_answer,
                **{key: numpy(v) for key, v in donors.items()},
            )
            conditions = (
                CF_CONDITIONS
                if counterfactual
                else CONDITIONS + (INVARIANT_EXTRA if arm == "invariant" else ())
            )
            jobs = [("native", 0)] + [(condition, at) for condition in conditions for at in [4, 5]]
            native = None
            for condition, at in jobs:
                parts = {}
                rounds = 32 if condition == "native" else 16
                for i in range(0, len(rows), 500):
                    d = {key: (v[i : i + 500] if v.ndim else v) for key, v in donors.items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        values = trace(
                            model,
                            torch.as_tensor(x[i : i + 500], device=device),
                            torch.as_tensor(pos[i : i + 500], device=device),
                            d,
                            condition,
                            at,
                            rounds,
                        )
                    for key, v in values.items():
                        parts.setdefault(key, []).append(v)
                values = {key: np.concatenate(v, axis=1) for key, v in parts.items()}
                if condition == "native":
                    native = values
                    if counterfactual:
                        for stage in ["raw_prefix", "carry_prefix"]:
                            grouped = values[stage][:, :, :5].reshape(32, 500, 10, 5)
                            np.testing.assert_array_equal(
                                grouped, np.repeat(grouped[:, :, :1], 10, axis=2)
                            )
                    else:
                        with np.load(reference / f"{panel}.npz") as old:
                            np.testing.assert_array_equal(values["answer"], old["answer"])
                else:
                    np.testing.assert_array_equal(
                        values["answer"][: at - 1], native["answer"][: at - 1]
                    )
                    for stage in ["raw_prefix", "carry_prefix"]:
                        np.testing.assert_array_equal(
                            values[stage][:, :, :-1], native[stage][:16, :, :-1]
                        )
                    if counterfactual and condition in ("bridge_zero", "bridge_constant"):
                        grouped = values["raw"][at - 1].reshape(500, 10)
                        np.testing.assert_array_equal(
                            grouped, np.repeat(grouped[:, :1], 10, axis=1)
                        )
                    if condition == "bridge_old" and not counterfactual:
                        old_path = (
                            p
                            / "disco-five-hop-diagnosis-20261025/evaluation/rope-ema"
                            / f"{panel}-canonical_bridge_force-r{at}.npz"
                            if arm == "rope-ema"
                            else p
                            / "invariant-five-hop-interventions-20261025/evaluation"
                            / f"{panel}-canonical_bridge_force-r{at}.npz"
                        )
                        with np.load(old_path) as old:
                            np.testing.assert_array_equal(values["answer"], old["answer"])
                np.savez_compressed(folder / f"{panel}-{condition}-r{at}.npz", **values)
                metric = dict(
                    arm=arm,
                    panel=panel,
                    condition=condition,
                    round=at,
                    accuracy=(values["answer"] == truth[None, :, -1]).mean(1).tolist(),
                )
                write_json(folder / f"{panel}-{condition}-r{at}.json", metric)
                status.update(
                    completed_pools=status["completed_pools"] + 1,
                    last=f"{arm}/{panel}/{condition}/r{at}",
                    elapsed_seconds=time.monotonic() - start,
                )
                write_json(args.output / "status.json", status)
                print(json.dumps(metric), flush=True)
        assert before == state_hash(model)
        validation[arm] = dict(
            checkpoint_sha256=sha(checkpoint),
            state_hash_before_after=before,
            native_history_exact=True,
            canonical_bridge_history_exact=True,
            other_position_readouts_unchanged=True,
        )
        write_json(args.output / "model-validation.json", validation)
        del model, donors
        torch.cuda.empty_cache()
    assert status["completed_pools"] == 152
    status.update(
        state="complete",
        elapsed_seconds=time.monotonic() - start,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )
    assert (
        status["peak_reserved_mib"] + 6144
        < torch.cuda.get_device_properties(0).total_memory * 0.16 / 2**20
    )
    write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
