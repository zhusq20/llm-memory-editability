"""Trace native DiscoLoop five-hop states and intervene at one fifth-receiver round."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_loop_beyond_four_hops import build, encode  # noqa: E402
from evaluate_loop_position_controlled_long_hops import sha, state_hash, write_json  # noqa: E402
from trace_invariant_five_hops import numpy, paths_for  # noqa: E402

from llm_memory_editability.loop_position_invariance import LoopPositionGPT, rotate  # noqa: E402
from llm_memory_editability.self_decode_alignment import _readout  # noqa: E402

ARMS = ("rope-ema", "rope-raw", "absolute-fixed-raw", "absolute-native-ema")
PATCH_ARMS = ("rope-ema", "absolute-fixed-raw")
CONDITIONS = (
    "force_bridge",
    "force_original_head",
    "canonical_bridge_force",
    "wrong_bridge_force",
    "atomic_postattention",
    "wrong_atomic_postattention",
)
TIMES = (4, 5, 6, 8)


def load_model(persistent, metadata, arm, device):
    config = json.loads((ROOT / "configs/loop-native-handoff-development-v1.json").read_text())
    if arm.startswith("absolute"):
        name = "A0-1500k" if arm == "absolute-fixed-raw" else "recipe-additive-1500k"
        ref = json.loads(
            (persistent / "loop-beyond-four-hops-20261020" / f"{name}.json").read_text()
        )
        relative = (
            "ungated-loop-continuation-v1/runs/disco-l1-r4"
            if arm == "absolute-fixed-raw"
            else "loop-native-handoff-development-v1/runs/recipe-additive"
        )
        checkpoint = persistent / relative / "checkpoint-1500000.pt"
        assert sha(checkpoint) == ref["checkpoint_sha256"]
        model, step = build(
            config, metadata, {**ref["model"], "checkpoint": str(checkpoint)}, device
        )
        assert step == 1500000
        reference = persistent / "loop-beyond-four-hops-20261020/predictions" / name
    else:
        config = json.loads(
            (ROOT / "configs/loop-position-invariance-development-v1.json").read_text()
        )
        checkpoint = (
            persistent
            / "loop-position-invariance-development-v1/runs/disco-rope-old/checkpoint-1500000.pt"
        )
        weight = arm.split("-")[-1]
        ref = json.loads(
            (
                persistent
                / "invariant-five-hop-diagnosis-20261025/evaluation"
                / f"disco-rope-{weight}.json"
            ).read_text()
        )
        assert sha(checkpoint) == ref["checkpoint_sha256"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["step"] == 1500000
        hf = GPT2Config(
            **config["model"],
            n_layer=1,
            vocab_size=metadata["vocab_size"],
            bos_token_id=metadata["pad_token"],
            eos_token_id=metadata["end_marker"],
            pad_token_id=metadata["pad_token"],
            use_cache=False,
        )
        model = LoopPositionGPT(
            hf,
            4,
            handoff="additive",
            position="rope",
            rope_base=config["rope_base"],
            **config["handoff"],
        )
        model.load_state_dict(saved["model"], strict=True)
        if weight == "ema":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(saved["ema"][name])
        model.to(device).eval()
        reference = (
            persistent / "invariant-five-hop-diagnosis-20261025/evaluation" / f"disco-rope-{weight}"
        )
    model.requires_grad_(False)
    return model, reference, checkpoint


@torch.no_grad()
def atomic_donor(model, tokens, bridge_tokens):
    """Canonical atomic source uses x0=e, with e4 and observed r5 but no target e5."""
    atomic = torch.full_like(tokens, model.config.pad_token_id)
    atomic[:, 0], atomic[:, 1] = bridge_tokens, tokens[:, -1]
    block = model.transformer.h[0]
    captured = {}

    def qkv(module, inputs, output):
        captured["value"] = output[:, 0, 2 * model.config.n_embd :].clone()

    def postattention(module, inputs):
        captured["postattention"] = inputs[0][:, 1].clone()

    handles = [
        block.attn.c_attn.register_forward_hook(qkv),
        block.ln_2.register_forward_pre_hook(postattention),
    ]
    positions = torch.ones((len(tokens), 1), dtype=torch.long, device=tokens.device)
    try:
        raw = model._round(model._embedded(atomic))
        captured["raw_state"] = raw[:, 1].clone()
        captured["raw_token"] = _readout(model, raw, positions)[:, 0].argmax(-1)
    finally:
        for handle in handles:
            handle.remove()
    logits, _, _ = model.trace(atomic, positions, 16)
    captured["atomic_r16_token"] = logits[-1, :, 0].argmax(-1)
    return captured


@torch.no_grad()
def trace(
    model, tokens, positions, truths=None, *, rounds=32, condition="native", at_round=0, donor=None
):
    """Original additive recurrence, with an optional single-receiver intervention."""
    assert model.handoff == "additive" and not model.training
    assert len(model.transformer.h) == 1 and tokens.shape[1] == 6
    if condition not in ("native", *CONDITIONS):
        raise ValueError(condition)
    if condition in CONDITIONS[2:]:
        assert donor is not None
    block = model.transformer.h[0]
    captured, records = {}, {}
    current = 0

    def qkv(module, inputs, output):
        captured["qkv"] = output

    def route(module, inputs):
        if current != at_round or condition not in CONDITIONS[:4]:
            return None
        values = captured["qkv"][..., 2 * model.config.n_embd :]
        if condition == "force_bridge":
            replacement = values[:, 4]
        elif condition == "force_original_head":
            replacement = values[:, 0]
        else:
            replacement = donor["value"]
        changed = inputs[0].clone()
        changed[:, 5] = replacement.to(changed.dtype)
        return (changed,)

    def postattention(module, inputs):
        captured["postattention"] = inputs[0]

    handles = [
        block.attn.c_attn.register_forward_hook(qkv),
        block.attn.c_proj.register_forward_pre_hook(route),
        block.ln_2.register_forward_pre_hook(postattention),
    ]

    def save(name, value):
        records.setdefault(name, []).append(numpy(value))

    try:
        x = model._embedded(tokens)
        for current in range(1, rounds + 1):
            raw = model._round(x)
            if current == at_round and condition in CONDITIONS[4:]:
                # Exact full postattention replacement through the same tokenwise
                # MLP, using the donor's original call shape and complete residual.
                raw = raw.clone()
                raw[:, 5] = donor["raw_state"].to(raw.dtype)
            carry = model._discoloop_boundary(raw)
            save("answer", _readout(model, raw, positions)[:, 0].argmax(-1))
            save("carry_answer", _readout(model, carry, positions)[:, 0].argmax(-1))
            for stage, value in [
                ("input", x),
                ("postattention", captured["postattention"]),
                ("raw", raw),
                ("carry", carry),
            ]:
                # Rich stages are native observations only. Patched postattention
                # is not stored because atomic replacement is applied at raw H.
                if truths is None and stage not in ("raw", "carry"):
                    continue
                logits = _readout(model, value).float()
                save(stage + "_token", logits.argmax(-1))
                if truths is not None:
                    selected = logits.gather(-1, truths[..., None]).squeeze(-1)
                    save(
                        stage + "_probability",
                        logits.softmax(-1).gather(-1, truths[..., None]).squeeze(-1),
                    )
                    save(stage + "_rank", (logits > selected[..., None]).sum(-1) + 1)
                    save(stage + "_norm", value.float().norm(dim=-1))
            if truths is not None:
                q, k, _ = captured["qkv"].split(model.config.n_embd, dim=-1)
                shape = (
                    *tokens.shape,
                    model.config.n_head,
                    model.config.n_embd // model.config.n_head,
                )
                q, k = (v.view(shape).transpose(1, 2) for v in (q, k))
                if getattr(model, "position", "absolute") == "rope":
                    offsets = (
                        torch.arange(tokens.shape[1], device=tokens.device) + model.position_offset
                    )
                    q, k = rotate(q, offsets, model.rope_base), rotate(k, offsets, model.rope_base)
                weights = (q[:, :, -1:].float() @ k.float().transpose(-2, -1)) / shape[-1] ** 0.5
                save("last_attention_weights", weights.softmax(-1).squeeze(2))
            x = carry
    finally:
        for handle in handles:
            handle.remove()
    return {k: np.stack(v) for k, v in records.items()}


def summarize(values, truths):
    correct = values["answer"] == truths[None, :, -1]
    carry_correct = values["carry_answer"] == truths[None, :, -1]
    ever = correct[:16].any(0)
    out = dict(
        n=len(truths),
        accuracy=correct.mean(1).tolist(),
        ever_correct_by_r16=int(ever.sum()),
        never_correct_by_r16=int((~ever).sum()),
        correct_then_lost_r16=int((ever & ~correct[15]).sum()),
        first_correct_after_r16=int((~ever & correct[16:].any(0)).sum()),
        carry_ever_correct_by_r16=int(carry_correct[:16].any(0).sum()),
        raw_correct_carry_wrong_by_r16=int((correct[:16] & ~carry_correct[:16]).any(0).sum()),
    )
    for stage in ["input", "postattention", "raw", "carry"]:
        out[stage + "_prefix_accuracy"] = (
            (values[stage + "_token"][:, :, 1:] == truths[None, :, 1:]).mean(1).tolist()
        )
    bridge = values["carry_token"][:-1, :, -2] == truths[None, :, -2]
    out["previous_carry_bridge_readable_r2_to_r16"] = bridge[:15].sum(1).tolist()
    out["answer_given_previous_bridge_r2_to_r16"] = [
        float(correct[r + 1, bridge[r]].mean()) if bridge[r].any() else None for r in range(15)
    ]
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
    meta = json.loads((data / "complete.json").read_text())
    assert sha(data / "world.npz") == meta["world_sha256"]
    world = dict(np.load(data / "world.npz"))
    long_path = p / "loop-beyond-four-hops-20261020/beyond-panels.npz"
    long = dict(np.load(long_path))
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(
        state="running",
        planned_pools=104,
        completed_pools=0,
        training_updates=0,
        world_sha256=sha(data / "world.npz"),
        panels_sha256=sha(long_path),
    )
    write_json(args.output / "status.json", status)
    results, validation = {}, {}
    for arm in ARMS:
        model, reference, checkpoint = load_model(p, meta, arm, device)
        before = state_hash(model)
        folder = args.output / arm
        folder.mkdir()
        results[arm] = {}
        for panel in ["id_5", "ood_5"]:
            rows = long[panel]
            truth = paths_for(rows, meta, world)
            np.testing.assert_array_equal(truth[:, -1], rows[:, -1] + meta["entity_offset"])
            n, offset = meta["entities_per_graph"], meta["entity_offset"]
            graph = rows[:, 0] // n
            wrong_bridge = (truth[:, -2] - offset + 1) % n + graph * n + offset
            wrong_answer = (
                world["permutations"][graph, rows[:, -2], (wrong_bridge - offset) % n]
                + graph * n
                + offset
            )
            assert (wrong_answer != truth[:, -1]).all()
            native = {}
            donors = {"true": {}, "wrong": {}}
            for i in range(0, len(rows), 500):
                tokens, positions, _ = encode(rows[i : i + 500], meta, 0)
                x, pos = (
                    torch.as_tensor(tokens, device=device),
                    torch.as_tensor(positions, device=device),
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    part = trace(model, x, pos, torch.as_tensor(truth[i : i + 500], device=device))
                    for key, bridge in [("true", truth[:, -2]), ("wrong", wrong_bridge)]:
                        d = atomic_donor(
                            model, x, torch.as_tensor(bridge[i : i + 500], device=device)
                        )
                        for k, v in d.items():
                            donors[key].setdefault(k, []).append(v)
                for k, v in part.items():
                    native.setdefault(k, []).append(v)
            native = {k: np.concatenate(v, axis=1) for k, v in native.items()}
            donors = {
                key: {k: torch.cat(v, dim=0) for k, v in donor.items()}
                for key, donor in donors.items()
            }
            with np.load(reference / f"{panel}.npz") as old:
                np.testing.assert_array_equal(native["answer"], old["answer"])
            np.savez_compressed(folder / f"{panel}-native.npz", rows=rows, truths=truth, **native)
            np.savez_compressed(
                folder / f"{panel}-donors.npz",
                true_bridge=truth[:, -2],
                wrong_bridge=wrong_bridge,
                truth=truth[:, -1],
                wrong_answer=wrong_answer,
                **{
                    key + "_" + k: numpy(v)
                    for key, donor in donors.items()
                    for k, v in donor.items()
                },
            )
            results[arm][panel] = summarize(native, truth)
            write_json(args.output / "native-summary.json", results)
            status.update(
                completed_pools=status["completed_pools"] + 1,
                last=f"{arm}/{panel}/native",
                elapsed_seconds=time.monotonic() - start,
            )
            write_json(args.output / "status.json", status)
            print(
                json.dumps(dict(arm=arm, panel=panel, native=results[arm][panel]["accuracy"][15])),
                flush=True,
            )
            if arm not in PATCH_ARMS:
                continue
            for condition in CONDITIONS:
                for at_round in TIMES:
                    parts = {}
                    for i in range(0, len(rows), 500):
                        tokens, positions, _ = encode(rows[i : i + 500], meta, 0)
                        d = {
                            k: v[i : i + 500]
                            for k, v in donors[
                                "wrong" if condition.startswith("wrong") else "true"
                            ].items()
                        }
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            values = trace(
                                model,
                                torch.as_tensor(tokens, device=device),
                                torch.as_tensor(positions, device=device),
                                rounds=16,
                                condition=condition,
                                at_round=at_round,
                                donor=d,
                            )
                        for k, v in values.items():
                            parts.setdefault(k, []).append(v)
                    values = {k: np.concatenate(v, axis=1) for k, v in parts.items()}
                    np.testing.assert_array_equal(
                        values["answer"][: at_round - 1], native["answer"][: at_round - 1]
                    )
                    for stage in ["raw", "carry"]:
                        np.testing.assert_array_equal(
                            values[stage + "_token"][:, :, :-1],
                            native[stage + "_token"][:16, :, :-1],
                        )
                    if condition.endswith("postattention"):
                        np.testing.assert_array_equal(
                            values["answer"][at_round - 1],
                            numpy(
                                donors["wrong" if condition.startswith("wrong") else "true"][
                                    "raw_token"
                                ]
                            ),
                        )
                    np.savez_compressed(folder / f"{panel}-{condition}-r{at_round}.npz", **values)
                    metric = dict(
                        arm=arm,
                        panel=panel,
                        condition=condition,
                        round=at_round,
                        accuracy=(values["answer"] == truth[None, :, -1]).mean(1).tolist(),
                    )
                    write_json(folder / f"{panel}-{condition}-r{at_round}.json", metric)
                    status.update(
                        completed_pools=status["completed_pools"] + 1,
                        last=f"{arm}/{panel}/{condition}/r{at_round}",
                        elapsed_seconds=time.monotonic() - start,
                    )
                    write_json(args.output / "status.json", status)
                    print(json.dumps(metric), flush=True)
                    assert time.monotonic() - start < 1800, "30-minute bounded inference timeout"
        assert state_hash(model) == before
        validation[arm] = dict(
            checkpoint_sha256=sha(checkpoint),
            model_hash_before_after=before,
            baseline_predictions_exact=True,
            other_position_readouts_unchanged=True,
            pre_intervention_answers_unchanged=True,
        )
        write_json(args.output / "validation.json", validation)
        del model, donors
        torch.cuda.empty_cache()
    status.update(
        state="complete",
        elapsed_seconds=time.monotonic() - start,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )
    assert status["completed_pools"] == status["planned_pools"]
    assert (
        status["peak_reserved_mib"] + 6144
        < torch.cuda.get_device_properties(0).total_memory * 0.16 / 2**20
    )
    write_json(args.output / "status.json", status)


if __name__ == "__main__":
    main()
