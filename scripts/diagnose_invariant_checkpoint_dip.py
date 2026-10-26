#!/usr/bin/env python3
"""Read-only, paired checkpoint diagnosis using the frozen native implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

FROZEN = Path(os.environ["LOOP_DIAGNOSTIC_SOURCE"])
sys.path[:0] = [str(FROZEN / "scripts"), str(FROZEN / "src")]
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.loop_hop_data import load, pack_rows, path_details  # noqa: E402
from llm_memory_editability.self_decode_alignment import _loss_dtype, _readout  # noqa: E402

STEPS = (256000, 700000, 1300000)
PANELS = ("ood_4", "ood_3", "ood_2", "id_4", "atomic_ood")
SCORE_KEYS = (
    "handoff_score_in",
    "handoff_score_in_bias",
    "handoff_score_out",
    "handoff_score_out_bias",
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions():
    # The 3 x 3 matrix isolates a complete handoff from the rest of each model.
    result = [dict(backbone=s, score=s, beta=s, zero=False) for s in STEPS]
    result += [
        dict(backbone=b, score=s, beta=s, zero=False) for b in STEPS for s in STEPS if b != s
    ]
    # Split the two candidate repairs of the trough into score-only and beta-only.
    result += [dict(backbone=700000, score=s, beta=700000, zero=False) for s in (256000, 1300000)]
    result += [dict(backbone=700000, score=700000, beta=s, zero=False) for s in (256000, 1300000)]
    result += [dict(backbone=s, score=s, beta=s, zero=True) for s in STEPS]
    return result


def case_name(case):
    return f"b{case['backbone']}-s{case['score']}-beta{case['beta']}" + (
        "-zero" if case["zero"] else ""
    )


def score_from_features(phi, values):
    hidden = F.gelu(F.linear(phi, values[SCORE_KEYS[0]], values[SCORE_KEYS[1]]))
    return (hidden * values[SCORE_KEYS[2]]).sum(-1) + values[SCORE_KEYS[3]]


@torch.no_grad()
def trace(model, tokens, positions, rounds, heads, zero=False, rich=True):
    """Observe the unchanged aggregate_round; only zero-score is an intervention."""
    original = model.score
    captured = {}

    def observe(self, candidate, previous, probability, previous_probability):
        score = original(candidate, previous, probability, previous_probability)
        actual = torch.zeros_like(score) if zero else score
        if rich:
            captured.update(
                candidate=candidate,
                previous=previous,
                probability=probability,
                phi=self.features(candidate, previous, probability, previous_probability),
                score=actual,
            )
        return actual

    model.score = types.MethodType(observe, model)
    injection = _loss_dtype(model._embedded(tokens))
    state = model.initial_state(injection)
    records = {}

    def append(key, value):
        records.setdefault(key, []).append(value.detach().cpu().numpy())

    try:
        for _ in range(rounds):
            previous = state[0]
            state, gamma = model.aggregate_round(state, injection)
            # Preserve the original readout shape for exact saved-prediction replay.
            answer = _readout(model, state[0], positions)[:, 0].argmax(-1)
            append("answer", answer)
            if rich:
                append("aggregate_token", _readout(model, state[0]).argmax(-1))
                append("candidate_token", _readout(model, captured["candidate"]).argmax(-1))
                append("raw_token", captured["probability"].argmax(-1))
                append("score", captured["score"])
                append("features", captured["phi"])
                append("gamma", gamma.squeeze(-1))
                append(
                    "relative_change",
                    (state[0] - previous).norm(dim=-1) / (previous.norm(dim=-1) + model.eps),
                )
                append("denominator", state[2].squeeze(-1))
                append(
                    "initial_mass", model.beta ** (len(records["answer"])) / state[2].squeeze(-1)
                )
                append(
                    "score_by_head",
                    torch.stack([score_from_features(captured["phi"], h) for h in heads], dim=-1),
                )
    finally:
        model.score = original
    return {key: np.stack(value) for key, value in records.items()}


def summarize(values, rows, metadata, world):
    truth = rows[:, -1] + metadata["entity_offset"]
    correct = values["answer"] == truth[None, :]
    answer_position = rows.shape[1] - 2
    result = {"n": len(rows), "answer_accuracy": correct.mean(axis=1).tolist()}
    if "gamma" not in values:
        return result
    nodes, _ = path_details(rows, metadata, world["permutations"])
    prefix_truth = nodes[:, 1:] + metadata["entity_offset"]
    for key in ("aggregate_token", "candidate_token", "raw_token"):
        pred = values[key][:, :, 1 : answer_position + 1]
        result[key + "_prefix_accuracy"] = (pred == prefix_truth[None, :, :]).mean(axis=1).tolist()
    for key in ("score", "gamma", "relative_change", "initial_mass"):
        value = values[key][:, :, answer_position]
        result[key + "_answer_mean"] = value.mean(axis=1).tolist()
        result[key + "_answer_median"] = np.median(value, axis=1).tolist()
        result[key + "_answer_q10_q90"] = np.quantile(value, [0.1, 0.9], axis=1).T.tolist()
    result["feature_answer_mean"] = values["features"][:, :, answer_position].mean(1).tolist()
    result["ever_correct_fraction"] = float(correct.any(axis=0).mean())
    result["correct_then_wrong_r8_fraction"] = float((correct[:8].any(axis=0) & ~correct[7]).mean())
    return result


def engineering_check(model, tokens, positions, heads, device):
    with native.autocast(device):
        traced = trace(model, tokens, positions, 8, heads)
        ordinary, change, gamma = model.trace(tokens, positions, 8)
        assert np.array_equal(traced["answer"], ordinary[:, :, 0].argmax(-1).cpu().numpy())
        np.testing.assert_allclose(traced["gamma"], gamma.cpu().numpy(), rtol=0, atol=0)
        np.testing.assert_allclose(traced["relative_change"], change.cpu().numpy(), rtol=0, atol=0)
        # Future teacher-forced answer tokens must not affect the diagnostic prefix.
        modified = tokens.clone()
        modified[torch.arange(len(tokens), device=device), positions[:, 1]] = 0
        changed = trace(model, modified, positions, 8, heads)
        assert np.array_equal(traced["answer"], changed["answer"])
        stop = int(positions[0, 0]) + 1
        np.testing.assert_allclose(
            traced["gamma"][:, :, :stop], changed["gamma"][:, :, :stop], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            traced["score"][:, :, :stop], changed["score"][:, :, :stop], rtol=0, atol=0
        )
    return {
        "passed": True,
        "native_answers_and_gamma_exact": True,
        "future_answer_token_does_not_affect_prefix": True,
        "n": len(tokens),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(protocol["config"]).read_text())
    root = Path(protocol["checkpoint_root"])
    world, panels, metadata = load(config["input_root"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.08)
    device = torch.device("cuda:0")
    snapshots = {}
    hashes = {}
    for step in STEPS:
        file = root / f"checkpoint-{step:07d}.pt"
        hashes[str(step)] = digest(file)
        assert hashes[str(step)] == protocol["checkpoint_sha256"][str(step)]
        saved = torch.load(file, map_location="cpu", weights_only=False)
        assert saved["step"] == step and saved["arm"] == "invariant"
        snapshots[step] = saved
    model = native.initialize(config, "invariant", metadata, device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    heads = [{k: snapshots[s]["ema"][k].to(device) for k in SCORE_KEYS} for s in STEPS]
    model.load_state_dict(snapshots[STEPS[0]]["model"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(snapshots[STEPS[0]]["ema"][name])
    x, pos, _ = pack_rows(panels["ood_4"][:32], metadata)
    check = engineering_check(
        model, torch.as_tensor(x, device=device), torch.as_tensor(pos, device=device), heads, device
    )
    write(out / "engineering-validation.json", check)
    if args.check_only:
        return
    import wandb

    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    run = wandb.init(
        entity="zsqzz",
        project="llm-memory-editability",
        job_type="checkpoint-diagnostic",
        name=protocol["batch"],
        dir=str(out),
        config=protocol,
    )
    write(out / "tracking.json", {"id": run.id, "url": run.url})
    replay, results, before = [], [], time.time()
    try:
        for case in conditions():
            name = case_name(case)
            model.load_state_dict(snapshots[case["backbone"]]["model"])
            with torch.no_grad():
                for key, parameter in model.named_parameters():
                    source = case["backbone"]
                    if key in SCORE_KEYS:
                        source = case["score"]
                    if key == "handoff_beta_logit":
                        source = case["beta"]
                    parameter.copy_(snapshots[source]["ema"][key])
            original = case["backbone"] == case["score"] == case["beta"] and not case["zero"]
            rounds = 16 if original else 8
            state_hash_before = native.model_digest(model)
            for panel in PANELS:
                rows = panels[panel]
                x, pos, _ = pack_rows(rows, metadata)
                batches = []
                for start in range(0, len(rows), protocol["batch_size"]):
                    begin = time.monotonic()
                    end = start + protocol["batch_size"]
                    tokens = torch.as_tensor(x[start:end], device=device)
                    positions = torch.as_tensor(pos[start:end], device=device)
                    with torch.no_grad(), native.autocast(device):
                        values = trace(
                            model,
                            tokens,
                            positions,
                            rounds,
                            heads,
                            zero=case["zero"],
                            rich=original,
                        )
                    batches.append(values)
                    torch.cuda.synchronize()
                    # Share only the already-authorized GPU; yield between small batches.
                    time.sleep((time.monotonic() - begin) * protocol["yield_factor"])
                values = {
                    key: np.concatenate([b[key] for b in batches], axis=1) for key in batches[0]
                }
                folder = out / name
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / f"{panel}.npz", rows=rows, **values)
                summary = {
                    "case": case,
                    "name": name,
                    "panel": panel,
                    "beta_value": float(model.beta),
                    **summarize(values, rows, metadata, world),
                }
                if original:
                    for r in (4, 8):
                        old_file = (
                            root
                            / "predictions"
                            / f"{case['backbone']:07d}"
                            / f"ema-r{r}"
                            / f"{panel}.npz"
                        )
                        with np.load(old_file) as saved_predictions:
                            assert np.array_equal(saved_predictions["rows"], rows)
                            mismatches = int(
                                np.count_nonzero(
                                    saved_predictions["answer"] != values["answer"][r - 1]
                                )
                            )
                        replay.append(
                            {
                                "step": case["backbone"],
                                "panel": panel,
                                "rounds": r,
                                "n": len(rows),
                                "mismatches": mismatches,
                            }
                        )
                        if mismatches:
                            write(out / "replay-failure.json", replay)
                            raise AssertionError(f"Original output replay differs: {replay[-1]}")
                results.append(summary)
                write(out / "summary.json", results)
                write(
                    out / "progress.json",
                    {
                        "completed_panels": len(results),
                        "total_panels": 80,
                        "case": name,
                        "panel": panel,
                        "seconds": time.time() - before,
                    },
                )
                run.log(
                    {
                        "evaluation_index": len(results),
                        "case": name,
                        "panel": panel,
                        "accuracy_r4": summary["answer_accuracy"][3],
                        "accuracy_r8": summary["answer_accuracy"][7],
                        "beta": float(model.beta),
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "case": name,
                            "panel": panel,
                            "r8": summary["answer_accuracy"][7],
                            "seconds": time.time() - before,
                        }
                    ),
                    flush=True,
                )
            assert state_hash_before == native.model_digest(model), "Evaluation changed parameters"
        write(out / "replay-validation.json", {"passed": True, "panels": replay})
        assert all(digest(root / f"checkpoint-{s:07d}.pt") == hashes[str(s)] for s in STEPS)
        write(
            out / "complete.json",
            {
                "passed": True,
                "conditions": len(conditions()),
                "panels": len(results),
                "new_training_steps": 0,
                "checkpoint_hashes_unchanged": True,
                "seconds": time.time() - before,
            },
        )
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
