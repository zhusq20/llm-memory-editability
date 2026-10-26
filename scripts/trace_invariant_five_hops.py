"""Observe fixed native five-hop trajectories and benchmark rotary DiscoLoop."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_loop_beyond_four_hops import CONTROLS, build, encode  # noqa: E402
from evaluate_loop_position_controlled_long_hops import (  # noqa: E402
    evaluate,
    sha,
    state_hash,
    write_json,
)

from llm_memory_editability.loop_position_invariance import LoopPositionGPT  # noqa: E402
from llm_memory_editability.self_decode_alignment import _loss_dtype, _readout  # noqa: E402


def paths_for(rows, metadata, world):
    n = metadata["entities_per_graph"]
    graph = rows[:, 0] // n
    nodes = [rows[:, 0]]
    for relation in rows[:, 1:-1].T:
        nodes.append(world["permutations"][graph, relation, nodes[-1] % n] + graph * n)
    return np.stack(nodes, axis=1) + metadata["entity_offset"]


def numpy(tensor):
    value = tensor.detach().cpu()
    return value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()


@torch.no_grad()
def trace_stages(model, tokens, positions, truths, rounds=32):
    """Observe original computation; true entities are used only by readout scoring."""
    old_round, old_score = model._round, model.score
    captured = {}
    block = model.transformer.h[0]
    assert len(model.transformer.h) == 1 and not model.training
    assert model.first_round_input == "state_plus_embedding" and model.handoff == "invariant"

    def observe_round(x):
        captured["input"] = x
        output = old_round(x)
        captured["raw"] = output
        return output

    def observe_score(candidate, previous, probability, previous_probability):
        captured["candidate"] = candidate
        return old_score(candidate, previous, probability, previous_probability)

    def postattention_hook(module, inputs):
        captured["postattention"] = inputs[0]

    def qkv_hook(module, inputs, output):
        captured["qkv"] = output

    handles = [
        block.ln_2.register_forward_pre_hook(postattention_hook),
        block.attn.c_attn.register_forward_hook(qkv_hook),
    ]
    model._round, model.score = observe_round, observe_score
    records = {}

    def save(key, value):
        records.setdefault(key, []).append(numpy(value))

    try:
        injection = _loss_dtype(model._embedded(tokens))
        state = model.initial_state(injection)
        for _ in range(rounds):
            state, gamma = model.aggregate_round(state, injection)
            captured["aggregate"] = state[0]
            save("answer", _readout(model, state[0], positions)[:, 0].argmax(-1))
            for stage in ("input", "postattention", "raw", "candidate", "aggregate"):
                logits = _readout(model, captured[stage]).float()
                selected = logits.gather(-1, truths[..., None]).squeeze(-1)
                save(stage + "_token", logits.argmax(-1))
                save(
                    stage + "_probability",
                    logits.softmax(-1).gather(-1, truths[..., None]).squeeze(-1),
                )
                save(stage + "_rank", (logits > selected[..., None]).sum(-1) + 1)
            save("gamma", gamma.squeeze(-1))
            q, k, _ = captured["qkv"].split(model.config.n_embd, dim=-1)
            shape = (*tokens.shape, model.config.n_head, model.config.n_embd // model.config.n_head)
            q, k = (v.view(shape).transpose(1, 2).float() for v in (q, k))
            weights = (q[:, :, -1:] @ k.transpose(-2, -1)) / shape[-1] ** 0.5
            save("last_attention_weights", weights.softmax(-1).squeeze(2))
    finally:
        model._round, model.score = old_round, old_score
        for handle in handles:
            handle.remove()
    return {k: np.stack(v) for k, v in records.items()}


def summary(values, truths):
    correct = values["answer"] == truths[None, :, -1]
    before = correct[:16].any(0)
    out = {
        "n": len(truths),
        "accuracy": correct.mean(1).tolist(),
        "ever_correct_by_r16": int(before.sum()),
        "correct_r16": int(correct[15].sum()),
        "correct_then_lost_at_r16": int((before & ~correct[15]).sum()),
        "never_correct_by_r16": int((~before).sum()),
        "first_correct_after_r16": int((~before & correct[16:].any(0)).sum()),
    }
    for stage in ("input", "postattention", "raw", "candidate", "aggregate"):
        c = values[stage + "_token"] == truths[None]
        out[stage + "_prefix_accuracy"] = c[:, :, 1:].mean(1).tolist()
        out[stage + "_ever_prefix_correct_by_r16"] = c[:16, :, 1:].any(0).mean(0).tolist()
        out[stage + "_true_probability"] = values[stage + "_probability"][:, :, 1:].mean(1).tolist()
    candidate = values["candidate_token"][:, :, -1] == truths[None, :, -1]
    raw = values["raw_token"][:, :, -1] == truths[None, :, -1]
    out["raw_ever_correct_by_r16"] = int(raw[:16].any(0).sum())
    out["candidate_ever_correct_by_r16"] = int(candidate[:16].any(0).sum())
    out["candidate_ever_correct_but_aggregate_never_by_r16"] = int(
        (candidate[:16].any(0) & ~before).sum()
    )
    bridge = values["aggregate_token"][:-1, :, -2] == truths[None, :, -2]
    out["previous_round_bridge_correct_counts_r2_to_r16"] = bridge[:15].sum(1).tolist()
    out["raw_fifth_correct_given_previous_bridge_r2_to_r16"] = [
        float(raw[r + 1, bridge[r]].mean()) if bridge[r].any() else None for r in range(15)
    ]
    out["last_attention_weights_mean"] = values["last_attention_weights"].mean(1).tolist()
    return out


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
    short = dict(np.load(data / "panels.npz"))
    long = dict(np.load(p / "loop-beyond-four-hops-20261020/beyond-panels.npz"))
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"state": "running", "training_updates": 0, "completed_pools": 0}
    write_json(args.output / "status.json", status)
    config = json.loads((ROOT / "configs/loop-native-handoff-development-v1.json").read_text())
    ref = json.loads((p / "loop-beyond-four-hops-20261020/invariant-1500k.json").read_text())
    checkpoint = p / "loop-native-handoff-development-v1/runs/invariant/checkpoint-1500000.pt"
    assert sha(checkpoint) == ref["checkpoint_sha256"]
    model, step = build(config, metadata, {**ref["model"], "checkpoint": str(checkpoint)}, device)
    assert step == 1500000
    model.requires_grad_(False)
    before = state_hash(model)
    results = {}
    for panel in ("id_5", "ood_5"):
        rows = long[panel]
        truth = paths_for(rows, metadata, world)
        np.testing.assert_array_equal(truth[:, -1], rows[:, -1] + metadata["entity_offset"])
        parts = {}
        for i in range(0, len(rows), 500):
            x, pos, _ = encode(rows[i : i + 500], metadata, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = trace_stages(
                    model,
                    torch.as_tensor(x, device=device),
                    torch.as_tensor(pos, device=device),
                    torch.as_tensor(truth[i : i + 500], device=device),
                )
            for k, v in values.items():
                parts.setdefault(k, []).append(v)
        values = {k: np.concatenate(v, axis=1) for k, v in parts.items()}
        with np.load(
            p / "loop-beyond-four-hops-20261020/predictions/invariant-1500k" / f"{panel}.npz"
        ) as old:
            np.testing.assert_array_equal(values["answer"], old["answer"])
        np.savez_compressed(
            args.output / f"invariant-{panel}.npz", rows=rows, truths=truth, **values
        )
        results[panel] = summary(values, truth)
        write_json(args.output / "invariant-summary.json", results)
        status.update(
            completed_pools=status["completed_pools"] + 1,
            panel=panel,
            elapsed_seconds=time.monotonic() - start,
        )
        write_json(args.output / "status.json", status)
        print(
            json.dumps(
                {
                    "model": "invariant",
                    "panel": panel,
                    **{k: v for k, v in results[panel].items() if not isinstance(v, list)},
                }
            ),
            flush=True,
        )
    assert state_hash(model) == before
    write_json(
        args.output / "invariant-validation.json",
        {
            "checkpoint_sha256": sha(checkpoint),
            "model_hash_before_after": before,
            "baseline_predictions_exact": True,
        },
    )
    del model
    torch.cuda.empty_cache()
    rope = json.loads((ROOT / "configs/loop-position-invariance-development-v1.json").read_text())
    checkpoint = (
        p / "loop-position-invariance-development-v1/runs/disco-rope-old/checkpoint-1500000.pt"
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 1500000
    hf = GPT2Config(
        **rope["model"],
        n_layer=1,
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    for weights in ("ema", "raw"):
        model = LoopPositionGPT(
            hf,
            4,
            handoff="additive",
            position="rope",
            rope_base=rope["rope_base"],
            **rope["handoff"],
        )
        model.load_state_dict(saved["model"], strict=True)
        if weights == "ema":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(saved["ema"][name])
        model.to(device).eval().requires_grad_(False)
        before = state_hash(model)
        record = {
            "weights": weights,
            "step": saved["step"],
            "checkpoint_sha256": sha(checkpoint),
            "panels": {},
        }
        folder = args.output / f"disco-rope-{weights}"
        folder.mkdir(exist_ok=False)
        for panel, rows in {**long, **{k: short[k] for k in CONTROLS}}.items():
            arrays, metrics = evaluate(model, rows, metadata, device)
            np.savez_compressed(folder / f"{panel}.npz", **arrays)
            record["panels"][panel] = metrics
            status.update(
                completed_pools=status["completed_pools"] + 1,
                panel=f"disco-rope-{weights}-{panel}",
                elapsed_seconds=time.monotonic() - start,
            )
            write_json(args.output / "status.json", status)
            print(
                json.dumps(
                    {
                        "model": f"disco-rope-{weights}",
                        "panel": panel,
                        "R16": metrics["accuracy"][15],
                    }
                ),
                flush=True,
            )
        assert state_hash(model) == before
        record["state_hash_before_after"] = before
        write_json(args.output / f"disco-rope-{weights}.json", record)
        del model
        torch.cuda.empty_cache()
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
    os.umask(0o022)
    main()
