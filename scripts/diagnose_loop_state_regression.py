#!/usr/bin/env python3
"""Inference-only localization of state-MLP checkpoint regression."""

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

SCORE_KEYS = (
    "handoff_score_in",
    "handoff_score_in_bias",
    "handoff_score_out",
    "handoff_score_out_bias",
    "handoff_beta_logit",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array(value):
    return value.detach().float().cpu().numpy()


@torch.no_grad()
def trace(model, tokens, positions, labels, rounds, rich=True):
    """Observe the native aggregation without changing its inputs or arithmetic."""
    old_score = model.score
    captured, records = {}, {}

    def observe(self, candidate, previous, probability, previous_probability):
        score = old_score(candidate, previous, probability, previous_probability)
        captured.update(candidate=candidate, probability=probability, score=score)
        return score

    def append(key, value):
        records.setdefault(key, []).append(array(value))

    model.score = types.MethodType(observe, model)
    try:
        injection = _loss_dtype(model._embedded(tokens))
        state = model.initial_state(injection)
        for _ in range(rounds):
            previous = state[0]
            state, gamma = model.aggregate_round(state, injection)
            logits = _readout(model, state[0], positions).float()
            answer_logits = logits[:, 0]
            target = labels[:, 0]
            other = answer_logits.clone()
            other.scatter_(1, target[:, None], -torch.inf)
            append("answer", answer_logits.argmax(-1))
            append(
                "answer_margin",
                answer_logits.gather(1, target[:, None]).squeeze(1) - other.amax(-1),
            )
            append(
                "nll",
                F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").reshape(
                    -1, 2
                ),
            )
            if rich:
                append("aggregate_token", _readout(model, state[0]).argmax(-1))
                append("candidate_token", _readout(model, captured["candidate"]).argmax(-1))
                append("raw_token", captured["probability"].argmax(-1))
                append("score", captured["score"])
                append("gamma", gamma.squeeze(-1))
                append("denominator", state[2].squeeze(-1))
                append("previous_norm", previous.norm(dim=-1))
                append("state_norm", state[0].norm(dim=-1))
                append("candidate_norm", captured["candidate"].norm(dim=-1))
                append(
                    "relative_change",
                    (state[0] - previous).norm(dim=-1) / (previous.norm(dim=-1) + model.eps),
                )
                append(
                    "candidate_previous_cosine",
                    F.cosine_similarity(captured["candidate"], previous, dim=-1),
                )
    finally:
        model.score = old_score
    return {key: np.stack(value) for key, value in records.items()}


def summarize(values, rows, metadata, world):
    truth = rows[:, -1] + metadata["entity_offset"]
    correct = values["answer"] == truth[None]
    result = {
        "n": len(rows),
        "answer_accuracy": correct.mean(1).tolist(),
        "answer_nll": values["nll"][:, :, 0].mean(1).tolist(),
        "teacher_forced_eos_nll": values["nll"][:, :, 1].mean(1).tolist(),
        "answer_margin_mean": values["answer_margin"].mean(1).tolist(),
        "ever_correct_through_r8": float(correct[:8].any(0).mean()),
        "correct_before_r8_but_wrong_at_r8": float((correct[:8].any(0) & ~correct[7]).mean()),
    }
    if "gamma" not in values:
        return result
    nodes, _ = path_details(rows, metadata, world["permutations"])
    stop = rows.shape[1] - 1
    prefix_truth = nodes[:, 1:] + metadata["entity_offset"]
    for key in ("raw_token", "candidate_token", "aggregate_token"):
        correct_prefix = values[key][:, :, 1:stop] == prefix_truth[None]
        result[key + "_prefix_accuracy"] = correct_prefix.mean(1).tolist()
        if key == "aggregate_token":
            result["prefix_ever_correct_through_r8"] = correct_prefix[:8].any(0).mean(0).tolist()
            result["prefix_lost_at_r8"] = (
                (correct_prefix[:8].any(0) & ~correct_prefix[7]).mean(0).tolist()
            )
    for key in (
        "score",
        "gamma",
        "denominator",
        "previous_norm",
        "state_norm",
        "candidate_norm",
        "relative_change",
        "candidate_previous_cosine",
    ):
        result[key + "_prefix_mean"] = values[key][:, :, 1:stop].mean(1).tolist()
    result["score_prefix_clipped_fraction"] = (
        (np.abs(values["score"][:, :, 1:stop]) >= 8).mean(1).tolist()
    )
    return result


def install(model, snapshots, backbone, scorer, weights="ema"):
    if weights == "raw" and backbone != scorer:
        raise ValueError("Raw-weight controls cannot use cross-checkpoint scorers")
    saved = snapshots[backbone]
    model.load_state_dict(saved["model"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            donor = scorer if name in SCORE_KEYS else backbone
            parameter.copy_(snapshots[donor]["ema" if weights == "ema" else "model"][name])
            parameter.requires_grad_(False)
    model.eval()


def engineering(model, tokens, positions, labels, device):
    before = native.model_digest(model)
    with native.autocast(device):
        values = trace(model, tokens, positions, labels, 8)
        ordinary, change, gamma = model.trace(tokens, positions, 8)
        np.testing.assert_array_equal(values["answer"], array(ordinary[:, :, 0].argmax(-1)))
        np.testing.assert_array_equal(values["gamma"], array(gamma))
        np.testing.assert_array_equal(values["relative_change"], array(change))
        modified = tokens.clone()
        modified[torch.arange(len(tokens), device=device), positions[:, 1]] = 0
        causal = trace(model, modified, positions, labels, 8)
        last = int(positions[0, 0]) + 1
        for key in ("answer", "answer_margin"):
            np.testing.assert_array_equal(values[key], causal[key])
        for key in (
            "aggregate_token",
            "candidate_token",
            "raw_token",
            "score",
            "gamma",
            "state_norm",
        ):
            np.testing.assert_array_equal(values[key][:, :, :last], causal[key][:, :, :last])
    assert native.model_digest(model) == before
    return {
        "native_answer_gamma_and_change_exact": True,
        "future_answer_causal_prefix_exact": True,
        "parameters_unchanged": True,
        "n": len(tokens),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(protocol["config"]).read_text())
    world, panels, metadata = load(config["input_root"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(protocol["gpu_memory_fraction"])
    root = Path(protocol["checkpoint_root"])
    snapshots = {}
    for step in protocol["steps"]:
        path = root / f"checkpoint-{step:07d}.pt"
        assert digest(path) == protocol["checkpoint_sha256"][str(step)]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["step"] == step and saved["arm"] == "state-mlp"
        snapshots[step] = saved
    model = native.initialize(config, "state-mlp", metadata, device).eval()
    install(model, snapshots, protocol["steps"][0], protocol["steps"][0])
    x, pos, y = pack_rows(panels["ood_4"][:32], metadata)
    check = engineering(model, *(torch.as_tensor(a, device=device) for a in (x, pos, y)), device)
    write(out / "engineering-validation.json", check)
    if args.check_only:
        write(
            out / "complete.json",
            {"passed": True, "engineering_only": True, "new_training_updates": 0},
        )
        return
    import wandb

    tracker = wandb.init(
        entity="zsqzz",
        project="llm-memory-editability",
        job_type="checkpoint-diagnostic",
        name=protocol["batch"],
        dir=str(out),
        config=protocol,
        mode=protocol["tracking"]["mode"],
    )
    write(out / "tracking.json", {"id": tracker.id, "url": tracker.url})
    results, replay = [], []
    started = time.monotonic()
    try:
        for case in protocol["cases"]:
            backbone, scorer = case["backbone"], case["scorer"]
            weights = case.get("weights", "ema")
            original = backbone == scorer and weights == "ema"
            rounds = protocol["native_rounds"] if original else 8
            install(model, snapshots, backbone, scorer, weights)
            before = native.model_digest(model)
            name = f"backbone-{backbone}-scorer-{scorer}" + ("-raw" if weights == "raw" else "")
            for panel in protocol["panels"]:
                rows = panels[panel]
                x, pos, y = pack_rows(rows, metadata)
                chunks = []
                for begin in range(0, len(rows), protocol["batch_size"]):
                    batch_start = time.monotonic()
                    if batch_start - started > protocol["timeout_seconds"]:
                        raise TimeoutError("Diagnostic exceeded its explicit runtime bound")
                    end = begin + protocol["batch_size"]
                    tensors = [torch.as_tensor(a[begin:end], device=device) for a in (x, pos, y)]
                    with native.autocast(device):
                        values = trace(model, *tensors, rounds, rich=original)
                    chunks.append(values)
                    torch.cuda.synchronize()
                    time.sleep((time.monotonic() - batch_start) * protocol["yield_factor"])
                values = {
                    key: np.concatenate([c[key] for c in chunks], axis=1) for key in chunks[0]
                }
                folder = out / name
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / f"{panel}.npz", rows=rows, **values)
                if original or weights == "raw":
                    for r in (4, 8) if original else (4,):
                        with np.load(
                            root
                            / "predictions"
                            / f"{backbone:07d}"
                            / f"{weights}-r{r}"
                            / f"{panel}.npz"
                        ) as saved:
                            np.testing.assert_array_equal(saved["rows"], rows)
                            np.testing.assert_array_equal(saved["answer"], values["answer"][r - 1])
                        replay.append(
                            {
                                "step": backbone,
                                "weights": weights,
                                "panel": panel,
                                "rounds": r,
                                "n": len(rows),
                                "answers_exact": True,
                            }
                        )
                    write(out / "replay-validation.json", replay)
                result = {
                    "backbone": backbone,
                    "scorer": scorer,
                    "weights": weights,
                    "panel": panel,
                    "beta": float(model.beta),
                    **summarize(values, rows, metadata, world),
                }
                results.append(result)
                write(out / "summary.json", results)
                progress = {
                    "completed_panels": len(results),
                    "total_panels": len(protocol["cases"]) * len(protocol["panels"]),
                    "case": name,
                    "panel": panel,
                    "seconds": time.monotonic() - started,
                }
                write(out / "progress.json", progress)
                tracker.log(
                    {
                        "evaluation_index": len(results),
                        "backbone": backbone,
                        "scorer": scorer,
                        "panel": panel,
                        "accuracy_r4": result["answer_accuracy"][3],
                        "accuracy_r8": result["answer_accuracy"][7],
                        "answer_nll_r8": result["answer_nll"][7],
                    }
                )
                print(
                    json.dumps({**progress, "accuracy_r8": result["answer_accuracy"][7]}),
                    flush=True,
                )
            assert native.model_digest(model) == before
        assert all(
            digest(root / f"checkpoint-{s:07d}.pt") == protocol["checkpoint_sha256"][str(s)]
            for s in protocol["steps"]
        )
        write(
            out / "complete.json",
            {
                "passed": True,
                "groups": len(results),
                "queries": sum(r["n"] for r in results),
                "exact_replay_groups": len(replay),
                "checkpoint_and_parameters_unchanged": True,
                "new_training_updates": 0,
                "seconds": time.monotonic() - started,
            },
        )
        tracker.finish()
    except BaseException as error:
        write(out / "failure.json", {"error": repr(error), "seconds": time.monotonic() - started})
        tracker.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
