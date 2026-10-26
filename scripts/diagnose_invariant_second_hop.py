#!/usr/bin/env python3
"""Inference-only second-hop interventions on the frozen native Loop model."""

from __future__ import annotations

import argparse
import json
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

CONDITIONS = (
    "native",
    "drop_bridge_value",
    "force_bridge",
    "force_original_head",
    "canonical_bridge_v",
    "canonical_bridge_kv",
    "canonical_bridge_force",
    "wrong_bridge_force",
    "atomic_attention_update",
    "atomic_mlp_input",
    "atomic_postattention_state",
    "wrong_atomic_postattention_state",
    "zero_mlp",
    "reset_target_relation",
    "canonical_kv_reset_target",
)


def array(x):
    x = x.detach().cpu()
    return (x.float() if x.dtype == torch.bfloat16 else x).numpy()


def inspect_stage(model, x, position, e1, e2, wrong):
    logits = _readout(model, x)[:, position].float()
    labels = torch.stack((e1, e2, wrong), -1)
    chosen = logits.gather(1, labels)
    probabilities = logits.softmax(-1).gather(1, labels)
    ranks = (logits[:, :, None] > chosen[:, None, :]).sum(1) + 1
    return dict(
        token=array(logits.argmax(-1)),
        probability=array(probabilities),
        rank=array(ranks),
        norm=array(x[:, position].float().norm(dim=-1)),
    )


class RoundProbe:
    def __init__(self, model):
        self.model = model
        self.original = model._round
        self.condition = "native"
        self.donors = {}
        self.stages = {}
        self.round_index = 0

    def block(self, x, condition="native", donors=None):
        original_input = x
        model, cfg = self.model, self.model.config
        assert len(model.transformer.h) == 1 and not model.training
        block = model.transformer.h[0]
        donors = donors or {}
        batch, length, width = x.shape
        if condition in ("reset_target_relation", "canonical_kv_reset_target"):
            x = x.clone()
            x[:, 2] = donors["relation"]
        residual = x
        z = block.ln_1(x)
        q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
        shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
        q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        if condition in (
            "drop_bridge_value",
            "drop_original_head_value",
            "canonical_bridge_v",
            "canonical_bridge_kv",
            "canonical_kv_reset_target",
        ):
            kv, vv = k.clone(), v.clone()
            if condition == "drop_bridge_value":
                vv[:, :, 1] = 0
            elif condition == "drop_original_head_value":
                vv[:, :, 0] = 0
            else:
                vv[:, :, 1] = donors["bridge_v"]
                if condition != "canonical_bridge_v":
                    kv[:, :, 1] = donors["bridge_k"]
            other = F.scaled_dot_product_attention(q, kv, vv, is_causal=True, dropout_p=0.0)
            y = y.clone()
            # Localize every intervention to receiver position 2, round 2.
            y[:, :, 2] = other[:, :, 2]
        if condition in ("mask_original_head", "mask_self"):
            mask = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()
            mask[2, 0 if condition == "mask_original_head" else 2] = False
            other = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            y = y.clone()
            y[:, :, 2] = other[:, :, 2]
        if condition in (
            "force_bridge",
            "force_original_head",
            "canonical_bridge_force",
            "wrong_bridge_force",
        ):
            y = y.clone()
            replacement = {
                "force_bridge": v[:, :, 1],
                "force_original_head": v[:, :, 0],
                "canonical_bridge_force": donors.get("bridge_v"),
                "wrong_bridge_force": donors.get("wrong_v"),
            }[condition]
            y[:, :, 2] = replacement
        attention = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, width))
        if condition == "atomic_attention_update":
            attention = attention.clone()
            attention[:, 2] = donors["atomic"]["attention"][:, 1]
        postattention = residual + F.dropout(attention, cfg.resid_pdrop, model.training)
        if condition in (
            "atomic_postattention_state",
            "wrong_atomic_postattention_state",
        ):
            postattention = postattention.clone()
            key = "wrong_atomic" if condition.startswith("wrong") else "atomic"
            postattention[:, 2] = donors[key]["postattention"][:, 1]
        mlp_input = block.ln_2(postattention)
        if condition == "atomic_mlp_input":
            mlp_input = mlp_input.clone()
            mlp_input[:, 2] = donors["atomic"]["mlp_input"][:, 1]
        mlp = block.mlp(mlp_input)
        if condition == "zero_mlp":
            mlp = mlp.clone()
            mlp[:, 2] = 0
        output = postattention + mlp
        if condition in ("reset_target_relation", "canonical_kv_reset_target"):
            # Resetting the query's residual also changes its K/V. Preserve all
            # other receivers, including later positions that could read those K/V.
            localized = self.original(original_input).clone()
            localized[:, 2] = output[:, 2]
            output = localized
        # Diagnostic probabilities only; native execution above retains original SDPA.
        weights = q[:, :, 2:3].float() @ k[:, :, :3].float().transpose(-2, -1)
        weights = (weights / (width // cfg.n_head) ** 0.5).softmax(-1).squeeze(2)
        self.stages = dict(
            input=residual,
            attention=attention,
            postattention=postattention,
            mlp_input=mlp_input,
            mlp=mlp,
            output=output,
            weights=weights,
            key=k,
            value=v,
        )
        return output

    def __call__(self, x):
        self.round_index += 1
        if self.round_index == 2:
            return self.block(x, self.condition, self.donors)
        return self.original(x)

    def donors_for(self, tokens, bridge, wrong_bridge):
        m, b = self.model, self.model.transformer.h[0]
        result = {}
        # These inputs include entity identity + its position, never the second-hop answer.
        for name, entity in (("bridge", bridge), ("wrong", wrong_bridge)):
            value = 2 * (m.transformer.wte(entity) + m.transformer.wpe.weight[1])
            qkv = b.attn.c_attn(b.ln_1(value[:, None]))
            _, k, v = qkv.split(m.config.n_embd, dim=-1)
            shape = (len(tokens), m.config.n_head, m.config.n_embd // m.config.n_head)
            result[name + "_k"] = k.reshape(shape)
            result[name + "_v"] = v.reshape(shape)
        result["relation"] = 2 * (m.transformer.wte(tokens[:, 2]) + m.transformer.wpe.weight[2])
        for key, entity in (("atomic", bridge), ("wrong_atomic", wrong_bridge)):
            donor_tokens = torch.zeros_like(tokens)
            donor_tokens[:, 0] = entity
            donor_tokens[:, 1] = tokens[:, 2]
            assert torch.count_nonzero(donor_tokens[:, 2:]) == 0
            injection = _loss_dtype(m._embedded(donor_tokens))
            raw = self.block(2 * injection)
            result[key] = {k: v for k, v in self.stages.items()}
            result[key]["raw_token"] = _readout(m, raw)[:, 1].argmax(-1)
        return result


@torch.no_grad()
def trace(model, probe, tokens, positions, condition, donors, e1, e2, wrong, rich=False):
    probe.condition, probe.donors, probe.round_index = condition, donors, 0
    old_round, old_score = model._round, model.score
    model._round = probe
    captured = {}

    def observe(candidate, previous, probability, previous_probability):
        if probe.round_index == 2:
            captured.update(candidate=candidate, probability=probability)
        return old_score(candidate, previous, probability, previous_probability)

    model.score = observe
    injection = _loss_dtype(model._embedded(tokens))
    state = model.initial_state(injection)
    result, answers = {}, []
    try:
        for r in range(1, 9):
            state, gamma = model.aggregate_round(state, injection)
            answers.append(array(_readout(model, state[0], positions)[:, 0].argmax(-1)))
            if r == 1 and rich:
                result["bridge_r1_token"] = array(_readout(model, state[0])[:, 1].argmax(-1))
            if r == 2:
                result["raw_token"] = array(captured["probability"][:, 2].argmax(-1))
                result["candidate_token"] = array(
                    _readout(model, captured["candidate"])[:, 2].argmax(-1)
                )
                result["aggregate_token"] = array(_readout(model, state[0])[:, 2].argmax(-1))
                result["gamma"] = array(gamma[:, 2, 0])
                if rich:
                    result["attention_weights"] = array(probe.stages["weights"])
                    result["source_input_token"] = array(
                        _readout(model, probe.stages["input"])[:, 1].argmax(-1)
                    )
                    for key in ("input", "attention", "postattention", "mlp", "output"):
                        stats = inspect_stage(model, probe.stages[key], 2, e1, e2, wrong)
                        result.update({key + "_" + k: v for k, v in stats.items()})
                    for key in ("postattention", "mlp_input", "mlp"):
                        result[key + "_atomic_cosine"] = array(
                            F.cosine_similarity(
                                probe.stages[key][:, 2].float(),
                                donors["atomic"][key][:, 1].float(),
                                dim=-1,
                            )
                        )
                    for key in ("atomic", "wrong_atomic"):
                        result[key + "_raw_token"] = array(donors[key]["raw_token"])
                        for stage in ("attention", "postattention", "mlp", "output"):
                            stats = inspect_stage(model, donors[key][stage], 1, e1, e2, wrong)
                            result.update(
                                {key + "_" + stage + "_" + k: v for k, v in stats.items()}
                            )
                    # Retain the exact unpatched interface states for subsequent inspection.
                    for key in ("input", "postattention", "mlp"):
                        result["hidden_" + key] = array(probe.stages[key][:, 2])
    finally:
        model._round, model.score = old_round, old_score
    result["answer"] = np.stack(answers, axis=1)
    return result


def metrics(values, rows, e1, e2, wrong, metadata):
    truth = rows[:, -1] + metadata["entity_offset"]
    out = dict(
        n=len(rows),
        answer_accuracy=(values["answer"] == truth[:, None]).mean(0).tolist(),
    )
    for key in ("raw_token", "candidate_token", "aggregate_token"):
        out[key + "_accuracy"] = float((values[key] == e2).mean())
        out[key + "_wrong_fact"] = float((values[key] == wrong).mean())
    if "attention_weights" in values:
        out["attention_by_head"] = values["attention_weights"].mean(0).tolist()
        for key in ("bridge_r1_token", "source_input_token"):
            out[key + "_accuracy"] = float((values[key] == e1).mean())
        for key, val in values.items():
            if key.endswith("_token") and key not in (
                "bridge_r1_token",
                "source_input_token",
            ):
                out[key + "_e2_accuracy"] = float((val == e2).mean())
                out[key + "_e1_accuracy"] = float((val == e1).mean())
            if key.endswith("_cosine"):
                out[key + "_mean"] = float(val.mean())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    conditions = tuple(protocol["conditions"])
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(protocol["config"]).read_text())
    world, panels, metadata = load(config["input_root"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(protocol["gpu_memory_fraction"])
    device = torch.device("cuda:0")
    model = native.initialize(config, "invariant", metadata, device).eval()
    probe = RoundProbe(model)
    checks, results, replay = [], [], []
    run = None
    start_time = time.monotonic()
    try:
        for step in protocol["steps"]:
            checkpoint = Path(protocol["checkpoint_root"]) / f"checkpoint-{step:07d}.pt"
            assert digest(checkpoint) == protocol["checkpoint_sha256"][str(step)]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            with torch.no_grad():
                for key, parameter in model.named_parameters():
                    parameter.copy_(saved["ema"][key])
                    parameter.requires_grad_(False)
            before_hash = native.model_digest(model)
            for panel in protocol["panels"]:
                rows = panels[panel][:32] if args.check_only else panels[panel]
                nodes, _ = path_details(rows, metadata, world["permutations"])
                e1, e2 = (
                    nodes[:, 1] + metadata["entity_offset"],
                    nodes[:, 2] + metadata["entity_offset"],
                )
                n = world["permutations"].shape[-1]
                graph = nodes[:, 1] // n
                wrong_bridge = graph * n + (nodes[:, 1] % n + 1) % n
                wrong = graph * n + world["permutations"][graph, rows[:, 2], wrong_bridge % n]
                wrong += metadata["entity_offset"]
                wrong_bridge += metadata["entity_offset"]
                assert np.all(wrong != e2)
                x, pos, _ = pack_rows(rows, metadata)
                chunks = {c: [] for c in conditions}
                for offset in range(0, len(rows), protocol["batch_size"]):
                    stop = offset + protocol["batch_size"]
                    ts = [
                        torch.as_tensor(a[offset:stop], device=device)
                        for a in (x, pos, e1, e2, wrong, wrong_bridge)
                    ]
                    tokens, positions, bridge, target, wrong_target, wrong_entity = ts
                    begin = time.monotonic()
                    with torch.no_grad(), native.autocast(device):
                        donors = probe.donors_for(tokens, bridge, wrong_entity)
                        if offset == 0:
                            inp = 2 * _loss_dtype(model._embedded(tokens))
                            assert torch.equal(probe.block(inp), probe.original(inp))
                        native_values = None
                        native_output = None
                        for condition in conditions:
                            values = trace(
                                model,
                                probe,
                                tokens,
                                positions,
                                condition,
                                donors,
                                bridge,
                                target,
                                wrong_target,
                                rich=condition == "native",
                            )
                            if condition == "native":
                                native_values = values
                                native_output = probe.stages["output"].clone()
                            elif offset == 0:
                                keep = [0, 1, 3, 4, 5]
                                assert torch.equal(
                                    probe.stages["output"][:, keep],
                                    native_output[:, keep],
                                )
                            chunks[condition].append(values)
                        if offset == 0:
                            ordinary, _, _ = model.trace(tokens, positions, 8)
                            assert np.array_equal(
                                native_values["answer"],
                                array(ordinary[:, :, 0].argmax(-1)).T,
                            )
                            changed = tokens.clone()
                            changed[:, 3:] = (
                                0  # Future of second-hop receiver, including all answers.
                            )
                            causal = trace(
                                model,
                                probe,
                                changed,
                                positions,
                                "native",
                                donors,
                                bridge,
                                target,
                                wrong_target,
                            )
                            for key in (
                                "raw_token",
                                "candidate_token",
                                "aggregate_token",
                                "gamma",
                            ):
                                np.testing.assert_array_equal(native_values[key], causal[key])
                            checks.append(
                                dict(
                                    step=step,
                                    panel=panel,
                                    native_trace_exact=True,
                                    block_exact=True,
                                    causal_prefix_exact=True,
                                    donor_suffix_zero=True,
                                    n=len(tokens),
                                )
                            )
                            write(out / "engineering-validation.json", checks)
                    torch.cuda.synchronize()
                    time.sleep((time.monotonic() - begin) * protocol["yield_factor"])
                for condition in conditions:
                    values = {
                        k: np.concatenate([c[k] for c in chunks[condition]])
                        for k in chunks[condition][0]
                    }
                    folder = out / str(step) / condition
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        folder / f"{panel}.npz",
                        rows=rows,
                        bridge=e1,
                        target=e2,
                        wrong_bridge=wrong_bridge,
                        wrong_target=wrong,
                        **values,
                    )
                    if condition == "native" and not args.check_only:
                        previous = (
                            Path(protocol["previous_results"])
                            / f"b{step}-s{step}-beta{step}"
                            / f"{panel}.npz"
                        )
                        with np.load(previous) as old:
                            np.testing.assert_array_equal(rows, old["rows"])
                            np.testing.assert_array_equal(values["answer"], old["answer"][:8].T)
                            for key in (
                                "raw_token",
                                "candidate_token",
                                "aggregate_token",
                            ):
                                np.testing.assert_array_equal(values[key], old[key][1, :, 2])
                        replay.append(
                            dict(
                                step=step,
                                panel=panel,
                                answers_all_8_rounds_exact=True,
                                second_hop_raw_candidate_aggregate_exact=True,
                                n=len(rows),
                            )
                        )
                        write(out / "replay.json", replay)
                    summary = dict(
                        step=step,
                        panel=panel,
                        condition=condition,
                        **metrics(values, rows, e1, e2, wrong, metadata),
                    )
                    results.append(summary)
                    write(out / "summary.json", results)
                    if not args.check_only:
                        if run is None:
                            import wandb

                            os.environ["WANDB_API_KEY"] = (
                                Path("/run/secrets/wandb").read_text().strip()
                            )
                            run = wandb.init(
                                entity="zsqzz",
                                project="llm-memory-editability",
                                job_type="checkpoint-diagnostic",
                                name=protocol["batch"],
                                dir=str(out),
                                config=protocol,
                            )
                            write(out / "tracking.json", dict(id=run.id, url=run.url))
                        run.log(
                            dict(
                                evaluation_index=len(results),
                                checkpoint_step=step,
                                panel=panel,
                                condition=condition,
                                raw_accuracy=summary["raw_token_accuracy"],
                                candidate_accuracy=summary["candidate_token_accuracy"],
                                aggregate_accuracy=summary["aggregate_token_accuracy"],
                                answer_r8=summary["answer_accuracy"][7],
                            )
                        )
                print(
                    json.dumps(
                        dict(
                            step=step,
                            panel=panel,
                            conditions=len(conditions),
                            seconds=time.monotonic() - start_time,
                        )
                    ),
                    flush=True,
                )
            assert native.model_digest(model) == before_hash
            assert digest(checkpoint) == protocol["checkpoint_sha256"][str(step)]
        write(
            out / "complete.json",
            dict(
                passed=True,
                new_training_steps=0,
                parameter_and_checkpoint_unchanged=True,
                groups=len(results),
                queries=sum(r["n"] for r in results),
                native_replay_groups=len(replay),
                seconds=time.monotonic() - start_time,
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
