#!/usr/bin/env python3
"""Fixed-sample CPU moment accounting after all target-machine runs finish."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from run_norm_handoff import initialize
from torch.nn import functional as F

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grokking_reproduction import digest, model_digest
from llm_memory_editability.loop_hop_data import load, pack_rows, panel_names
from llm_memory_editability.self_decode_alignment import _readout


def moments(value, prefix):
    value = value.double()
    return {
        f"{prefix}_second_moment": value.square().mean(-1),
        f"{prefix}_channel_variance": value.var(-1, correction=1),
        f"{prefix}_mean_squared": value.mean(-1).square(),
    }


@torch.no_grad()
def measure(model, selected, metadata, batch_size):
    records = defaultdict(lambda: defaultdict(list))
    for panel, rows in selected.items():
        x, positions, _ = [torch.as_tensor(a) for a in pack_rows(rows, metadata)]
        for begin in range(0, len(x), batch_size):
            tokens, pos = x[begin : begin + batch_size], positions[begin : begin + batch_size]
            grid = torch.arange(tokens.shape[1])[None, :]
            masks = {
                "causal_query_prefix": grid <= pos[:, :1],
                "answer_prediction_position": grid == pos[:, :1],
                "first_relation_position": (grid == 1).expand(len(tokens), -1),
            }
            state = model._embedded(tokens)
            for index in range(model.repeats):
                h = model._round(state)
                metrics = {
                    **moments(state, "input"),
                    **moments(h, "core"),
                    **moments(h - state, "core_increment"),
                }
                if index < model.repeats - 1:
                    probability = _readout(model, h).softmax(-1)
                    value = F.linear(probability, model.transformer.wte.weight.T)
                    q = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                    alpha = torch.ones_like(h[..., :1])
                    if model.handoff in ("additive", "complementary"):
                        u = F.layer_norm(h, (h.shape[-1],), eps=model.config.layer_norm_epsilon)
                        alpha = torch.sigmoid(
                            (u * model.gate_u + q * model.gate_q).sum(-1, keepdim=True)
                            + model.gate_bias
                        )
                    carry = (
                        0.5 * h
                        if model.handoff == "additive"
                        else (1 - alpha) * h
                        if model.handoff == "complementary"
                        else h
                    )
                    injection = alpha * q
                    mixed = carry + injection
                    expected = mixed
                    if model.handoff == "normalized":
                        expected = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
                    actual = model._discoloop_boundary(h)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                    c, j = carry.double(), injection.double()
                    centered_c = c - c.mean(-1, keepdim=True)
                    centered_j = j - j.mean(-1, keepdim=True)
                    metrics.update(
                        {
                            **moments(q, "q"),
                            **moments(carry, "raw_carry"),
                            **moments(injection, "raw_injection"),
                            **moments(mixed, "raw_mix"),
                            **moments(actual, "boundary"),
                            "raw_energy_cross_term": 2 * (c * j).mean(-1),
                            "raw_variance_cross_term": 2
                            * (centered_c * centered_j).sum(-1)
                            / (h.shape[-1] - 1),
                            "alpha": alpha[..., 0],
                        }
                    )
                    if model.handoff == "normalized":
                        assert bool((actual.double().square().mean(-1) <= 1 + 1e-6).all())
                    state = actual
                else:
                    state = h
                for scope, mask in masks.items():
                    for key, value in metrics.items():
                        values = value[mask].double().numpy()
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite moment: {panel}/{scope}/{key}")
                        records[f"{panel}/round{index + 1}/{scope}"][key].append(values)
            # The manual trace must preserve the native complete forward exactly.
            assert torch.equal(_readout(model, state), model(tokens))
    result = {}
    for key, metrics in records.items():
        result[key] = {}
        for name, pieces in metrics.items():
            values = np.concatenate(pieces)
            result[key][name] = {
                "n": len(values),
                "mean": float(values.mean()),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    destination = args.base / "activation-diagnostic.json"
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    _, panels, metadata = load(config["input_root"])
    spec = config["diagnostic"]
    rng = np.random.default_rng(spec["selection_seed"])
    indices = {
        name: np.sort(rng.choice(len(panels[name]), spec["examples_per_panel"], replace=False))
        for name in panel_names()
    }
    selected = {name: panels[name][ids] for name, ids in indices.items()}
    result = {
        "protocol": spec,
        "config_sha256": digest(args.config),
        "selected_indices": {name: ids.tolist() for name, ids in indices.items()},
        "torch_version": torch.__version__,
        "runs": {},
        "utc": utc(),
    }
    for name in ("F", *config["arms"]):
        folder = args.baseline if name == "F" else args.base / "runs" / name
        for step in spec["checkpoint_steps"]:
            checkpoint = folder / f"checkpoint-{step:07d}.pt"
            before = digest(checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model, _, _, _ = initialize(
                config, "R" if name == "F" else name, metadata, torch.device("cpu")
            )
            if name == "F":
                model.residual_scale = 1.0
            model.load_state_dict(saved["model"], strict=True)
            model.eval()
            initial = model_digest(model)
            assert initial == saved["model_sha256"]
            assert saved["step"] == step and saved["world_sha256"] == metadata["world_sha256"]
            values = measure(model, selected, metadata, spec["batch_size"])
            assert model_digest(model) == initial and digest(checkpoint) == before
            result["runs"][f"{name}-{step}"] = {
                "checkpoint_sha256": before,
                "model_sha256": initial,
                "moments": values,
            }
            del model, saved
    result["passed"] = True
    result["note"] = (
        "FP32 CPU diagnostic; not BF16 endpoint scoring. Channel variance uses correction=1; "
        "M=((d-1)/d)*variance+mean_squared. Raw cross terms precede N's extra normalization."
    )
    write_json(destination, result)


if __name__ == "__main__":
    main()
