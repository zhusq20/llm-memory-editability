#!/usr/bin/env python3
"""Read-only CPU activation accounting for immutable dual-gate checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

sys.path.insert(0, "/frozen/src")
from llm_memory_editability.dual_gate import DualGateGPT  # noqa: E402
from llm_memory_editability.grokking_reproduction import model_digest  # noqa: E402
from llm_memory_editability.loop_hop_data import load, pack_rows, panel_names  # noqa: E402
from llm_memory_editability.self_decode_alignment import _readout  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def moments(x, prefix):
    x = x.double()
    return {
        prefix + "_second_moment": x.square().mean(-1),
        prefix + "_channel_variance": x.var(-1, correction=1),
        prefix + "_mean_squared": x.mean(-1).square(),
    }


def cosine(x, y):
    return F.cosine_similarity(x.double(), y.double(), dim=-1, eps=1e-30)


def boundary_parts(model, h, letter):
    p = _readout(model, h).softmax(-1)
    v = F.linear(p, model.transformer.wte.weight.T)
    q = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + model.discoloop_eps)
    r = torch.ones_like(h)
    g = torch.ones_like(h[..., :1])
    if letter == "P":
        g = torch.sigmoid(
            (v * model.gate_paper_weight).sum(-1, keepdim=True) + model.gate_paper_bias
        )
    elif letter != "F":
        u = F.layer_norm(h, (h.shape[-1],), eps=model.config.layer_norm_epsilon)
        g = 2 * torch.sigmoid(
            (u * model.gate_u + q * model.gate_q).sum(-1, keepdim=True)
            + model.gate_bias
        )
        if letter in ("D", "V"):
            r = 2 * torch.sigmoid(
                model.gate_r_u * u + model.gate_r_q * q + model.gate_r_bias
            )
    carry = r * h if letter == "D" else h
    injection = g * (r * q) if letter == "V" else g * q
    return carry, injection, q, r, g


class Recorder:
    def __init__(self):
        self.values = defaultdict(lambda: defaultdict(list))

    def add(self, key, masks, metrics):
        for scope, mask in masks.items():
            for name, value in metrics.items():
                selected = value[mask].double().cpu().numpy()
                assert np.isfinite(selected).all(), (key, scope, name)
                self.values[key + "/" + scope][name].append(selected)

    def finish(self):
        result = {}
        for key, metrics in self.values.items():
            row = {}
            for name, pieces in metrics.items():
                x = np.concatenate(pieces)
                row[name] = {
                    "n": len(x),
                    "mean": float(x.mean()),
                    "p05": float(np.quantile(x, 0.05)),
                    "median": float(np.median(x)),
                    "p95": float(np.quantile(x, 0.95)),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }
            if "boundary_channel_variance" in row:
                row["boundary_variance_ratio_of_means"] = (
                    row["boundary_channel_variance"]["mean"]
                    / row["core_channel_variance"]["mean"]
                )
            if "input_channel_variance" in row:
                row["core_variance_ratio_of_means"] = (
                    row["core_channel_variance"]["mean"]
                    / row["input_channel_variance"]["mean"]
                )
            result[key] = row
        return result


@torch.no_grad()
def measure(model, letter, selected, metadata):
    recorder = Recorder()
    scores, validation = (
        {},
        {
            "native_trace_exact": True,
            "max_boundary_formula_error": 0.0,
            "max_energy_accounting_error": 0.0,
            "max_variance_accounting_error": 0.0,
        },
    )
    for panel, rows in selected.items():
        x, pos, labels = [torch.as_tensor(a) for a in pack_rows(rows, metadata)]
        correct = 0
        for begin in range(0, len(rows), 32):
            tokens, positions = x[begin : begin + 32], pos[begin : begin + 32]
            grid = torch.arange(tokens.shape[1])[None, :]
            masks = {
                "causal_query_prefix": grid <= positions[:, :1],
                "answer_prediction_position": grid == positions[:, :1],
                "first_relation_position": (grid == 1).expand(len(tokens), -1),
            }
            state = model._embedded(tokens)
            increments = []
            for index in range(model.repeats):
                h = model._round(state)
                delta = h - state
                increments.append(delta)
                metrics = {
                    **moments(state, "input"),
                    **moments(h, "core"),
                    **moments(delta, "core_increment"),
                }
                if index < model.repeats - 1:
                    carry, injection, q, r, g = boundary_parts(model, h, letter)
                    actual = model._discoloop_boundary(h)
                    error = float((actual - (carry + injection)).abs().max())
                    validation["max_boundary_formula_error"] = max(
                        validation["max_boundary_formula_error"], error
                    )
                    assert error < 1e-5, error
                    c, j = carry.double(), injection.double()
                    exact_sum = c + j
                    cm = c - c.mean(-1, keepdim=True)
                    jm = j - j.mean(-1, keepdim=True)
                    cross = 2 * (c * j).mean(-1)
                    covariance = 2 * (cm * jm).sum(-1) / (h.shape[-1] - 1)
                    energy_error = (
                        (
                            exact_sum.square().mean(-1)
                            - c.square().mean(-1)
                            - j.square().mean(-1)
                            - cross
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    var_error = (
                        (
                            exact_sum.var(-1, correction=1)
                            - c.var(-1, correction=1)
                            - j.var(-1, correction=1)
                            - covariance
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    validation["max_energy_accounting_error"] = max(
                        validation["max_energy_accounting_error"], energy_error
                    )
                    validation["max_variance_accounting_error"] = max(
                        validation["max_variance_accounting_error"], var_error
                    )
                    assert energy_error < 1e-8 and var_error < 1e-8
                    metrics.update(
                        {
                            **moments(actual, "boundary"),
                            **moments(carry, "carry"),
                            **moments(injection, "injection"),
                            "energy_cross_term": cross,
                            "variance_cross_term": covariance,
                            "carry_energy_gain": c.square().mean(-1)
                            / h.double().square().mean(-1).clamp_min(1e-30),
                            "g": g.squeeze(-1),
                            "r_mean": r.mean(-1),
                            "r_fraction_above_one": (r > 1).float().mean(-1),
                            "r_fraction_below_one": (r < 1).float().mean(-1),
                            "r_max": r.amax(-1),
                            "r_min": r.amin(-1),
                            "injection_q_cosine": cosine(injection, q),
                        }
                    )
                    state = actual
                else:
                    state = h
                recorder.add(f"{panel}/round{index + 1}", masks, metrics)
            for i in range(len(increments)):
                for j in range(i + 1, len(increments)):
                    recorder.add(
                        f"{panel}/core_increment_alignment",
                        masks,
                        {
                            f"cosine_{i + 1}_{j + 1}": cosine(
                                increments[i], increments[j]
                            )
                        },
                    )
            traced = _readout(model, state, positions)
            correct += int(
                (traced[:, 0].argmax(-1) == labels[begin : begin + 32, 0]).sum()
            )
            if begin == 0:
                native = model(tokens, positions)
                exact = torch.equal(native, traced)
                validation["native_trace_exact"] &= exact
                assert exact, (letter, panel, (native - traced).abs().max().item())
        scores[panel] = {"n": len(rows), "answer_accuracy": correct / len(rows)}
    return {
        "metrics": recorder.finish(),
        "sampled_scores": scores,
        "validation": validation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config = json.loads(Path("/frozen/configs/dual-gate-multihop-v1.json").read_text())
    _, panels, metadata = load("/current/input")
    rng = np.random.default_rng(2026101701)
    indices = {
        name: np.sort(rng.choice(len(panels[name]), 128, replace=False))
        for name in panel_names()
    }
    selected = {name: panels[name][ids] for name, ids in indices.items()}
    result = {
        "protocol": json.loads(Path("/diagnostic-protocol.json").read_text()),
        "script_sha256": sha(__file__),
        "torch_version": torch.__version__,
        "selected_indices": {k: v.tolist() for k, v in indices.items()},
        "runs": {},
        "input_hashes": {},
    }
    paths = [Path("/current/input") / name for name in ("world.npz", "panels.npz")]
    variants = {"P": "paper", "G": "scalar", "D": "dual", "V": "vector-injection"}
    for step in (0, 8000):
        for letter in ("F", "P", "G", "D", "V"):
            root = (
                Path("/baseline")
                if letter == "F"
                else Path("/current/runs") / variants[letter]
            )
            checkpoint = root / f"checkpoint-{step:07d}.pt"
            paths.append(checkpoint)
            before = sha(checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert saved["step"] == step
            assert saved["world_sha256"] == metadata["world_sha256"]
            hf = GPT2Config(
                **config["model"],
                n_layer=saved["spec"]["unique_layers"],
                vocab_size=metadata["vocab_size"],
                use_cache=False,
            )
            if letter == "F":
                model = SupervisedAlignmentGPT(hf, 4, **saved["architecture"])
            else:
                variant = config["arms"][variants[letter]]["gate_variant"]
                model = DualGateGPT(
                    hf, 4, gate_variant=variant, **saved["architecture"]
                )
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            original_digest = model_digest(model)
            assert original_digest == saved["model_sha256"]
            measured = measure(model, letter, selected, metadata)
            assert model_digest(model) == original_digest
            assert sha(checkpoint) == before
            measured.update(
                checkpoint=str(checkpoint),
                checkpoint_sha256=before,
                model_sha256=original_digest,
                step=step,
            )
            result["runs"][f"{letter}-{step}"] = measured
            print(
                json.dumps(
                    {
                        "arm": letter,
                        "step": step,
                        "validated": measured["validation"],
                        "ood3_accuracy": measured["sampled_scores"]["ood_3"],
                    }
                ),
                flush=True,
            )
            del model, saved
    result["input_hashes"] = {str(path): sha(path) for path in paths}
    result["read_only_replay_passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
