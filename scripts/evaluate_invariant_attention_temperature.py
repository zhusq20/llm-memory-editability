#!/usr/bin/env python3
"""Uniform attention temperature, with no training or position-specific rule."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

FROZEN = Path(os.environ["LOOP_DIAGNOSTIC_SOURCE"])
sys.path[:0] = [str(FROZEN / "scripts"), str(FROZEN / "src")]
import run_loop_native_handoff as native  # noqa: E402
from diagnose_invariant_checkpoint_dip import digest, write  # noqa: E402

from llm_memory_editability.loop_hop_data import load, pack_rows, path_details  # noqa: E402
from llm_memory_editability.self_decode_alignment import _loss_dtype, _readout  # noqa: E402


def cpu(x):
    x = x.detach().cpu()
    return (x.float() if x.dtype == torch.bfloat16 else x).numpy()


class TemperatureRound:
    """Replica of the frozen uncached shared block; only SDPA scale can change."""

    def __init__(self, model):
        self.model = model
        self.original = model._round
        self.temperature = 1.0
        self.calls = 0
        self.attention_calls = 0
        self.observed = {}

    def __call__(self, x):
        model, cfg = self.model, self.model.config
        assert not model.training and self.temperature > 0
        self.calls += 1
        batch, length, _ = x.shape
        head_width = cfg.n_embd // cfg.n_head
        for block in model.transformer.h:
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
            shape = (batch, length, cfg.n_head, head_width)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            # The same scalar is applied to every query, key, head, layer and round.
            kwargs = (
                {}
                if self.temperature == 1.0
                else {"scale": 1.0 / (math.sqrt(head_width) * self.temperature)}
            )
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0, **kwargs)
            self.attention_calls += 1
            if self.calls == 2:
                logits = q[:, :, 2:3].float() @ k[:, :, :3].float().transpose(-2, -1)
                weights = (logits / (math.sqrt(head_width) * self.temperature)).softmax(-1)
                self.observed["second_hop_attention"] = weights.squeeze(2)
            y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, cfg.n_embd))
            x = x + F.dropout(y, cfg.resid_pdrop, model.training)
            x = x + block.mlp(block.ln_2(x))
        return x


@torch.no_grad()
def evaluate(model, wrapper, tokens, positions, rounds):
    assert not torch.is_grad_enabled()
    wrapper.calls, wrapper.attention_calls, wrapper.observed = 0, 0, {}
    original_round, original_score = model._round, model.score
    model._round = wrapper
    capture = {}

    def observe(candidate, previous, probability, previous_probability):
        if wrapper.calls == 2:
            capture.update(candidate=candidate, probability=probability)
        return original_score(candidate, previous, probability, previous_probability)

    model.score = observe
    answers, ends, gamma_answer = [], [], []
    injection = _loss_dtype(model._embedded(tokens))
    state = model.initial_state(injection)
    result = {}
    try:
        for r in range(1, rounds + 1):
            state, gamma = model.aggregate_round(state, injection)
            prediction = _readout(model, state[0], positions).argmax(-1)
            answers.append(cpu(prediction[:, 0]).astype(np.int16))
            ends.append(cpu(prediction[:, 1]).astype(np.int16))
            gamma_answer.append(cpu(gamma[:, int(positions[0, 0]), 0]))
            if r == 2:
                result["raw_second"] = cpu(capture["probability"][:, 2].argmax(-1)).astype(np.int16)
                result["candidate_second"] = cpu(
                    _readout(model, capture["candidate"])[:, 2].argmax(-1)
                ).astype(np.int16)
                result["aggregate_second"] = cpu(_readout(model, state[0])[:, 2].argmax(-1)).astype(
                    np.int16
                )
                result["second_attention"] = cpu(wrapper.observed["second_hop_attention"])
    finally:
        model._round, model.score = original_round, original_score
    assert wrapper.calls == rounds
    assert wrapper.attention_calls == rounds * len(model.transformer.h)
    result["answer"] = np.stack(answers, axis=1)
    result["end_marker"] = np.stack(ends, axis=1)
    result["gamma_answer"] = np.stack(gamma_answer, axis=1)
    return result


@torch.no_grad()
def validate(model, wrapper, tokens, positions, device, metadata):
    with native.autocast(device):
        injection = 2 * _loss_dtype(model._embedded(tokens))
        if wrapper.temperature == 1.0:
            assert torch.equal(wrapper(injection), wrapper.original(injection))
        a = evaluate(model, wrapper, tokens, positions, 16)
        if wrapper.temperature == 1.0:
            ordinary, _, _ = model.trace(tokens, positions, 16)
            np.testing.assert_array_equal(a["answer"], cpu(ordinary[:, :, 0].argmax(-1)).T)
            np.testing.assert_array_equal(a["end_marker"], cpu(ordinary[:, :, 1].argmax(-1)).T)
        changed = tokens.clone()
        indices = torch.arange(len(tokens), device=device)
        changed[indices, positions[:, 1]] = 0
        b = evaluate(model, wrapper, changed, positions, 16)
        np.testing.assert_array_equal(a["answer"], b["answer"])
        # Check joint answer+EOS equals greedy two-token exact match at R8.
        generated = tokens.clone()
        generated[indices, positions[:, 1]] = torch.as_tensor(
            a["answer"][:, 7], device=device, dtype=tokens.dtype
        )
        g = evaluate(model, wrapper, generated, positions, 8)
        truth = cpu(tokens[indices, positions[:, 1]])
        joint = (a["answer"][:, 7] == truth) & (a["end_marker"][:, 7] == metadata["end_marker"])
        greedy_joint = (g["answer"][:, 7] == truth) & (
            g["end_marker"][:, 7] == metadata["end_marker"]
        )
        np.testing.assert_array_equal(joint, greedy_joint)
    # Independent float32 numerical test of temperature scaling and causal mask.
    generator = torch.Generator(device=device).manual_seed(314159)
    q, k, v = [torch.randn((2, 12, 6, 64), generator=generator, device=device) for _ in range(3)]
    scale = 1.0 / (8.0 * wrapper.temperature)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    logits = (q @ k.transpose(-2, -1)) * scale
    logits = logits.masked_fill(
        ~torch.ones((6, 6), device=device, dtype=torch.bool).tril(), -torch.inf
    )
    expected = logits.softmax(-1) @ v
    actual = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return dict(
        passed=True,
        temperature=wrapper.temperature,
        n=len(tokens),
        native_exact=wrapper.temperature == 1.0,
        causal_answer_exact=True,
        joint_matches_greedy=True,
        uniform_temperature_formula_verified=True,
    )


def summarize(values, rows, metadata, world):
    truth = rows[:, -1] + metadata["entity_offset"]
    correct = values["answer"] == truth[:, None]
    ends = values["end_marker"] == metadata["end_marker"]
    result = dict(
        n=len(rows),
        answer_accuracy=correct.mean(0).tolist(),
        answer_and_eos_accuracy=(correct & ends).mean(0).tolist(),
        eos_accuracy_teacher_forced=ends.mean(0).tolist(),
        mean_gamma_answer=values["gamma_answer"].mean(0).tolist(),
    )
    if rows.shape[1] >= 4:
        nodes, _ = path_details(rows, metadata, world["permutations"])
        e2 = nodes[:, 2] + metadata["entity_offset"]
        result.update(
            {
                key + "_accuracy": float((values[key] == e2).mean())
                for key in ("raw_second", "candidate_second", "aggregate_second")
            }
        )
        result["second_attention_mean"] = values["second_attention"].mean((0, 1)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(p["config"]).read_text())
    world, panels, metadata = load(config["input_root"])
    for relative, expected in p["source_sha256"].items():
        assert digest(FROZEN / relative) == expected
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(p["gpu_memory_fraction"])
    device = torch.device("cuda:0")
    model = native.initialize(config, "invariant", metadata, device).eval()
    assert model.config.vocab_size < 32768
    wrapper = TemperatureRound(model)
    results, validations, replay = [], [], []
    begin = time.monotonic()
    run = None
    if not args.check_only:
        import wandb

        os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
        run = wandb.init(
            entity="zsqzz",
            project="llm-memory-editability",
            job_type="checkpoint-diagnostic",
            name=p["batch"],
            dir=str(out),
            config=p,
        )
        write(out / "tracking.json", dict(id=run.id, url=run.url))
    try:
        for step in p["steps"]:
            source = Path(p["checkpoint_root"]) / f"checkpoint-{step:07d}.pt"
            assert digest(source) == p["checkpoint_sha256"][str(step)]
            saved = torch.load(source, map_location="cpu", weights_only=False)
            assert saved["step"] == step and saved["arm"] == "invariant"
            model.load_state_dict(saved["model"])
            with torch.no_grad():
                for key, parameter in model.named_parameters():
                    parameter.copy_(saved["ema"][key])
                    parameter.requires_grad_(False)
            state_hash = native.model_digest(model)
            for temperature in p["temperatures"]:
                wrapper.temperature = temperature
                # Engineering calibration uses both longest and shortest input formats.
                for panel in ("ood_4", "atomic_ood"):
                    x, pos, _ = pack_rows(panels[panel][:32], metadata)
                    check = validate(
                        model,
                        wrapper,
                        torch.as_tensor(x, device=device),
                        torch.as_tensor(pos, device=device),
                        device,
                        metadata,
                    )
                    validations.append(dict(step=step, panel=panel, **check))
                    write(out / "engineering-validation.json", validations)
                if args.check_only:
                    continue
                for panel in p["panels"]:
                    rows = panels[panel]
                    x, pos, _ = pack_rows(rows, metadata)
                    chunks = []
                    for start in range(0, len(rows), p["batch_size"]):
                        tick = time.monotonic()
                        end = start + p["batch_size"]
                        with torch.no_grad(), native.autocast(device):
                            values = evaluate(
                                model,
                                wrapper,
                                torch.as_tensor(x[start:end], device=device),
                                torch.as_tensor(pos[start:end], device=device),
                                p["rounds"],
                            )
                        chunks.append(values)
                        torch.cuda.synchronize()
                        time.sleep((time.monotonic() - tick) * p["yield_factor"])
                    values = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
                    folder = out / str(step) / f"tau-{temperature:.2f}"
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / f"{panel}.npz", rows=rows, **values)
                    if temperature == 1.0:
                        previous = (
                            Path(p["previous_results"])
                            / f"b{step}-s{step}-beta{step}"
                            / f"{panel}.npz"
                        )
                        with np.load(previous) as old:
                            np.testing.assert_array_equal(rows, old["rows"])
                            np.testing.assert_array_equal(
                                values["answer"], old["answer"][: p["rounds"]].T
                            )
                            if panel != "atomic_ood":
                                for new, previous_key in [
                                    ("raw_second", "raw_token"),
                                    ("candidate_second", "candidate_token"),
                                    ("aggregate_second", "aggregate_token"),
                                ]:
                                    np.testing.assert_array_equal(
                                        values[new], old[previous_key][1, :, 2]
                                    )
                        replay.append(
                            dict(
                                step=step, panel=panel, n=len(rows), rounds=p["rounds"], exact=True
                            )
                        )
                        write(out / "replay.json", replay)
                    metrics = dict(
                        step=step,
                        temperature=temperature,
                        panel=panel,
                        **summarize(values, rows, metadata, world),
                    )
                    results.append(metrics)
                    write(out / "summary.json", results)
                    run.log(
                        dict(
                            evaluation_index=len(results),
                            checkpoint_step=step,
                            temperature=temperature,
                            panel=panel,
                            answer_r4=metrics["answer_accuracy"][3],
                            answer_r8=metrics["answer_accuracy"][7],
                            answer_r16=metrics["answer_accuracy"][15],
                            joint_r8=metrics["answer_and_eos_accuracy"][7],
                        )
                    )
                print(
                    json.dumps(
                        dict(
                            step=step,
                            temperature=temperature,
                            groups=len(results),
                            seconds=time.monotonic() - begin,
                        )
                    ),
                    flush=True,
                )
            assert native.model_digest(model) == state_hash
            assert digest(source) == p["checkpoint_sha256"][str(step)]
        write(
            out / "complete.json",
            dict(
                passed=True,
                new_training_steps=0,
                parameters_and_checkpoints_unchanged=True,
                groups=len(results),
                query_instances=sum(x["n"] for x in results),
                validation_groups=len(validations),
                replay_groups=len(replay),
                seconds=time.monotonic() - begin,
            ),
        )
        if run:
            run.finish()
    except BaseException:
        if run:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
