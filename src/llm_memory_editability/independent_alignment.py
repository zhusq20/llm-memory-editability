"""Replicate the existing state objective in two ordinary, independent GPT layers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import representation_alignment as original
from .grok_depth import write_json
from .latent_scaling import build_world, model_digest
from .storage_composition import data_digest, evaluate, file_hash


def validate_model(spec, model):
    if spec["layers"] != 2 or spec["repeats"] != 1:
        raise ValueError("This replication requires two independent layers, executed once")
    blocks = list(model.iter_blocks())
    assert len(blocks) == 2 and blocks[0] is not blocks[1]
    assert not {p.data_ptr() for p in blocks[0].parameters()} & {
        p.data_ptr() for p in blocks[1].parameters()
    }
    assert all(p.requires_grad for p in model.parameters())


@torch.no_grad()
def diagnose(model, world, device):
    """Original readout of a pure first-hop prefix; no fitted probe or final answer."""
    model.eval()
    metrics, predictions = {}, {}
    for task in ("common_atomic", "familiar_test", "strict_test"):
        rows = world[task]
        prefixes = np.column_stack(
            (np.full(len(rows), 2), rows[:, 0], np.full(len(rows), 3), rows[:, 1])
        )
        labels = rows[:, 2]
        guesses, cosines = [], []
        for start in range(0, len(rows), 256):
            tokens = torch.as_tensor(prefixes[start : start + 256], device=device)
            targets = torch.as_tensor(labels[start : start + 256], device=device)
            _, state = model(tokens, return_bridge=True)
            logits = F.linear(model.ln_final(state), model.token.weight)
            guesses.append(logits.argmax(-1).cpu().numpy())
            cosines.append(F.cosine_similarity(state, model.token(targets)).cpu().numpy())
        guess, cosine = np.concatenate(guesses), np.concatenate(cosines)
        correct = guess == labels
        metrics[task] = dict(
            n=len(rows),
            first_hop_original_readout_accuracy=float(correct.mean()),
            mean_cosine_to_input_embedding=float(cosine.mean()),
            pure_prefix=True,
        )
        predictions.update(
            {task + "_guess": guess, task + "_correct": correct, task + "_cosine": cosine}
        )
    return metrics, predictions


def train(spec, out, source, device):
    model = original.new_model(spec, "cpu")
    validate_model(spec, model)
    assert model_digest(model) == spec["initial_model_sha256"]
    assert data_digest(build_world(spec)) == spec["data_sha256"]
    del model
    # Keep the historical trainer, objective, sampling, optimizer and arithmetic intact.
    original.train(spec, out, source, device)
    out = Path(out)
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    model = original.new_model(spec, device)
    model.load_state_dict(payload["model"])
    metrics, predictions = diagnose(model, build_world(spec), device)
    write_json(out / "state-diagnostics.json", metrics)
    np.savez_compressed(out / "state-predictions.npz", **predictions)


def audit(out, device):
    """Independent process reload of full generation and original intermediate readout."""
    out = Path(out)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    spec = payload["spec"]
    model = original.new_model(spec, device)
    validate_model(spec, model)
    model.load_state_dict(payload["model"])
    world = build_world(spec)
    stored_world = np.load(out / "world.npz")
    for key, value in world.items():
        np.testing.assert_array_equal(value, stored_world[key])
    assert data_digest(world) == spec["data_sha256"]
    complete = json.loads((out / "complete.json").read_text())
    assert file_hash(out / "model.pt") == complete["model_sha256"]
    metrics, predictions = evaluate(model, world, "low", device)
    saved = np.load(out / f"predictions-{spec['steps']:06d}.npz")
    maximum = 0.0
    for key, value in predictions.items():
        if key.endswith("nll"):
            maximum = max(maximum, float(np.max(np.abs(value - saved[key]))))
            np.testing.assert_allclose(value, saved[key], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(value, saved[key])
    state_metrics, state_predictions = diagnose(model, world, device)
    saved_state = np.load(out / "state-predictions.npz")
    for key, value in state_predictions.items():
        if key.endswith("cosine"):
            np.testing.assert_allclose(value, saved_state[key], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(value, saved_state[key])
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            metrics=metrics,
            state_metrics=state_metrics,
            max_nll_error=maximum,
            allow_tf32=True,
            independent_parameters=True,
        ),
    )
