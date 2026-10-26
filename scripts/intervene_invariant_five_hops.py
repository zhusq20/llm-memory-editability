"""Fixed, single-round fifth-receiver interventions on the old invariant endpoint."""

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
from evaluate_loop_beyond_four_hops import build, encode  # noqa: E402
from evaluate_loop_position_controlled_long_hops import sha, state_hash, write_json  # noqa: E402
from trace_invariant_five_hops import numpy, paths_for  # noqa: E402

from llm_memory_editability.self_decode_alignment import _loss_dtype, _readout  # noqa: E402

CONDITIONS = (
    "force_bridge",
    "force_original_head",
    "canonical_bridge_force",
    "wrong_bridge_force",
    "atomic_postattention",
    "wrong_atomic_postattention",
)
TIMES = (4, 5, 6, 8)


@torch.no_grad()
def atomic_donor(model, tokens, bridge_tokens):
    """Oracle e4 and observed r5 only; the correct e5 answer is never an input."""
    atomic = torch.full_like(tokens, model.config.pad_token_id)
    atomic[:, 0], atomic[:, 1] = bridge_tokens, tokens[:, -1]
    block = model.transformer.h[0]
    captured = {}

    def postattention(module, inputs):
        captured["postattention"] = inputs[0][:, 1].clone()

    def qkv(module, inputs, output):
        captured["value"] = output[:, 0, 2 * model.config.n_embd :].clone()

    handles = [
        block.ln_2.register_forward_pre_hook(postattention),
        block.attn.c_attn.register_forward_hook(qkv),
    ]
    try:
        raw = model._round(2 * _loss_dtype(model._embedded(atomic)))
        captured["raw_state"] = raw[:, 1].clone()
        positions = torch.ones((len(tokens), 1), dtype=torch.long, device=tokens.device)
        captured["raw_token"] = _readout(model, raw, positions)[:, 0].argmax(-1)
    finally:
        for handle in handles:
            handle.remove()
    logits, _ = model.run(atomic, positions, 16, carry=False)
    captured["atomic_r16_token"] = logits[:, 0].argmax(-1)
    return captured


@torch.no_grad()
def intervene(model, tokens, positions, bridge_tokens, condition, at_round, rounds=16):
    """Modify only attention output at the fifth receiver, at one fixed round."""
    if condition not in (*CONDITIONS, "native"):
        raise ValueError(condition)
    assert tokens.shape[1] == 6 and len(model.transformer.h) == 1
    assert model.handoff == "invariant" and model.first_round_input == "state_plus_embedding"
    assert not model.training
    donor = None
    if condition in CONDITIONS[2:]:
        donor = atomic_donor(model, tokens, bridge_tokens)
    old_round, old_score = model._round, model.score
    block = model.transformer.h[0]
    captured, records = {}, {}
    current = 0

    def qkv(module, inputs, output):
        captured["values"] = output[..., 2 * model.config.n_embd :]

    def route(module, inputs):
        if current != at_round or condition not in CONDITIONS[:4]:
            return None
        changed = inputs[0].clone()
        if condition == "force_bridge":
            replacement = captured["values"][:, 4]
        elif condition == "force_original_head":
            replacement = captured["values"][:, 0]
        else:
            replacement = donor["value"]
        changed[:, 5] = replacement.to(changed.dtype)
        return (changed,)

    # GPT2Block adds the pre-MLP residual separately after ln_2. Replacing only
    # ln_2 input would miss that residual. Replace the returned raw receiver with
    # donor z + MLP(ln_2(z)), an exact postattention-state intervention.
    def wrapped_round(value):
        raw = old_round(value)
        if current == at_round and condition in CONDITIONS[4:]:
            raw = raw.clone()
            raw[:, 5] = donor["raw_state"].to(raw.dtype)
        captured["raw"] = raw
        return raw

    def score(candidate, *args):
        captured["candidate"] = candidate
        return old_score(candidate, *args)

    # No ln_2 hook here: the full residual replacement is performed above.
    handles = [
        block.attn.c_attn.register_forward_hook(qkv),
        block.attn.c_proj.register_forward_pre_hook(route),
    ]
    model._round, model.score = wrapped_round, score

    def save(name, value):
        records.setdefault(name, []).append(numpy(value))

    try:
        injection = _loss_dtype(model._embedded(tokens))
        state = model.initial_state(injection)
        for current in range(1, rounds + 1):  # noqa: B007 -- read by intervention hooks
            state, gamma = model.aggregate_round(state, injection)
            for name, value in (
                ("raw", captured["raw"]),
                ("candidate", captured["candidate"]),
                ("answer", state[0]),
            ):
                save(name, _readout(model, value, positions)[:, 0].argmax(-1))
            save("prefix_token", _readout(model, state[0]).argmax(-1))
            save("gamma", gamma[:, -1, 0])
    finally:
        model._round, model.score = old_round, old_score
        for handle in handles:
            handle.remove()
    result = {k: np.stack(v) for k, v in records.items()}
    if donor is not None:
        result["donor_raw_token"] = numpy(donor["raw_token"])
        result["donor_atomic_r16_token"] = numpy(donor["atomic_r16_token"])
    return result


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
    metadata = json.loads((data / "complete.json").read_text())
    assert sha(data / "world.npz") == metadata["world_sha256"]
    world = dict(np.load(data / "world.npz"))
    long = dict(np.load(p / "loop-beyond-four-hops-20261020/beyond-panels.npz"))
    config = json.loads((ROOT / "configs/loop-native-handoff-development-v1.json").read_text())
    reference = json.loads((p / "loop-beyond-four-hops-20261020/invariant-1500k.json").read_text())
    checkpoint = p / "loop-native-handoff-development-v1/runs/invariant/checkpoint-1500000.pt"
    assert sha(checkpoint) == reference["checkpoint_sha256"]
    model, step = build(
        config, metadata, {**reference["model"], "checkpoint": str(checkpoint)}, device
    )
    assert step == 1500000
    model.requires_grad_(False)
    before = state_hash(model)
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "training_updates": 0, "completed_pools": 0, "planned_pools": 50}
    write_json(args.output / "status.json", status)
    for panel in ("id_5", "ood_5"):
        rows = long[panel]
        truth = paths_for(rows, metadata, world)
        n, offset = metadata["entities_per_graph"], metadata["entity_offset"]
        graph = rows[:, 0] // n
        wrong_bridge = (truth[:, -2] - offset + 1) % n + graph * n
        wrong_answer = (
            world["permutations"][graph, rows[:, -2], wrong_bridge % n] + graph * n + offset
        )
        assert (wrong_answer != truth[:, -1]).all()
        jobs = [("native", 0)] + [(c, t) for c in CONDITIONS for t in TIMES]
        native = None
        for condition, at_round in jobs:
            parts = {}
            for i in range(0, len(rows), 500):
                x, positions, _ = encode(rows[i : i + 500], metadata, 0)
                bridge = wrong_bridge + offset if condition.startswith("wrong") else truth[:, -2]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = intervene(
                        model,
                        torch.as_tensor(x, device=device),
                        torch.as_tensor(positions, device=device),
                        torch.as_tensor(bridge[i : i + 500], device=device),
                        condition,
                        at_round,
                    )
                for k, value in values.items():
                    parts.setdefault(k, []).append(value)
            values = {
                k: np.concatenate(v, axis=0 if k.startswith("donor_") else 1)
                for k, v in parts.items()
            }
            if condition == "native":
                native = values
                with np.load(
                    p
                    / "invariant-five-hop-diagnosis-20261025/evaluation"
                    / f"invariant-{panel}.npz"
                ) as previous:
                    np.testing.assert_array_equal(values["answer"], previous["answer"][:16])
                    np.testing.assert_array_equal(
                        values["prefix_token"], previous["aggregate_token"][:16]
                    )
            else:
                np.testing.assert_array_equal(
                    values["prefix_token"][:, :, :-1], native["prefix_token"][:, :, :-1]
                )
                np.testing.assert_array_equal(
                    values["answer"][: at_round - 1], native["answer"][: at_round - 1]
                )
            np.savez_compressed(
                args.output / f"{panel}-{condition}-r{at_round}.npz",
                rows=rows,
                truths=truth,
                wrong_answer=wrong_answer,
                **values,
            )
            metric = {
                "panel": panel,
                "condition": condition,
                "intervention_round": at_round,
                "accuracy": (values["answer"] == truth[None, :, -1]).mean(1).tolist(),
            }
            write_json(args.output / f"{panel}-{condition}-r{at_round}.json", metric)
            status.update(
                completed_pools=status["completed_pools"] + 1,
                last=metric,
                elapsed_seconds=time.monotonic() - start,
            )
            write_json(args.output / "status.json", status)
            print(json.dumps(metric), flush=True)
    assert state_hash(model) == before
    write_json(
        args.output / "validation.json",
        {
            "checkpoint_sha256": sha(checkpoint),
            "world_sha256": sha(data / "world.npz"),
            "panels_sha256": sha(p / "loop-beyond-four-hops-20261020/beyond-panels.npz"),
            "model_hash_before_after": before,
            "baseline_predictions_exact": True,
            "all_other_positions_unchanged": True,
            "all_rounds_before_intervention_unchanged": True,
        },
    )
    status.update(
        state="complete",
        elapsed_seconds=time.monotonic() - start,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )
    write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
