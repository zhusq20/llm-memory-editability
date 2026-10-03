"""Fixed-weight, self-decoded bridge re-encoding in existing small GPTs.

The intervention uses the original tied readout over the complete vocabulary.
Only the explicitly named oracle condition accesses graph answers at inference.
The wrong_entity control is an alternative to the model's decoded token, not a
guaranteed incorrect answer: its accidental agreement with truth is reported.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import utc, write_json
from .latent_scaling import build_world, model_digest
from .representation_alignment import RepresentationGPT, new_model
from .storage_composition import data_digest, evaluate, file_hash, generate_rows
from .text_pretrain import WORDS, composite_sentence

CONDITIONS = ("baseline", "self_decode", "wrong_entity", "oracle")


def reencode_state(state, embedding, alpha, variant="norm_matched"):
    """Interpolate residual and normalized embedding, without renormalizing the sum.

    norm_matched gives the replacement embedding the original state's norm.
    paper_unit uses a unit embedding directly. Neither preserves the mixture's
    norm in general; alpha=0 returns the original tensor exactly.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if variant not in ("norm_matched", "paper_unit"):
        raise ValueError("Unknown re-encoding variant")
    if alpha == 0:
        return state
    replacement = F.normalize(embedding, dim=-1)
    if variant == "norm_matched":
        replacement = state.norm(dim=-1, keepdim=True) * replacement
    return (1 - alpha) * state + alpha * replacement


def alternative_entity(decoded, bridge_start, bridge_count):
    """A non-self bridge-entity choice, independent of the true bridge identity."""
    if bridge_count < 2:
        raise ValueError("At least two bridge entity tokens are required")
    return bridge_start + torch.remainder(decoded - bridge_start + 1, bridge_count)


class ReencodingGPT(RepresentationGPT):
    """Original computation with a single intervention at block 0, position 3."""

    def configure(self, condition, alpha, oracle_table=None, variant="norm_matched"):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one")
        if variant not in ("norm_matched", "paper_unit"):
            raise ValueError("Unknown re-encoding variant")
        if condition == "oracle" and oracle_table is None:
            raise ValueError("Oracle condition requires an explicit diagnostic truth table")
        self.condition, self.alpha, self.variant = condition, float(alpha), variant
        # Deliberately outside state_dict: neither weights nor model hashes change.
        self.oracle_table = oracle_table if condition == "oracle" else None

    def _choice(self, state, tokens):
        decoded = F.linear(self.ln_final(state), self.token.weight).argmax(-1)
        selected = decoded
        oracle_available = torch.zeros_like(decoded, dtype=torch.bool)
        if self.condition == "wrong_entity":
            selected = alternative_entity(decoded, self.bridge_start, self.bridge_count)
        elif self.condition == "oracle":
            oracle = self.oracle_table[tokens[:, 1], tokens[:, 3]]
            oracle_available = oracle >= 0
            # Invalid autonomously generated heads have no graph truth. Do not
            # fabricate an oracle answer; retain the model's decoded token.
            selected = torch.where(oracle_available, oracle, decoded)
        return decoded, selected, oracle_available

    def forward(self, tokens, positions=None, repeats=None, return_bridge=False):
        if not return_bridge and (self.condition == "baseline" or self.alpha == 0):
            return super().forward(tokens, positions=positions, repeats=repeats)
        if tokens.shape[1] < 4:
            raise ValueError("The intervention requires the causal first-relation prefix")
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
        info = None
        for index, block in enumerate(self.iter_blocks(repeats)):
            z = block.ln1(x)
            batch, length, width = z.shape
            a = block.attention
            q, k, v = a.qkv(z).view(batch, length, 3, a.heads, width // a.heads).unbind(2)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
                dropout_p=p,
            )
            y = a.proj(y.transpose(1, 2).reshape(batch, length, width))
            x = x + F.dropout(y, p=p, training=self.training)
            x = x + F.dropout(block.mlp(block.ln2(x)), p=p, training=self.training)
            if index == 0:
                state = x[:, 3]
                decoded, selected, oracle_available = self._choice(state, tokens)
                replacement = (
                    state
                    if self.condition == "baseline"
                    else reencode_state(state, self.token(selected), self.alpha, self.variant)
                )
                if self.condition != "baseline" and self.alpha:
                    x = x.clone()
                    x[:, 3] = replacement
                info = {
                    "state": state,
                    "replacement": replacement,
                    "decoded": decoded,
                    "selected": selected,
                    "oracle_available": oracle_available,
                }
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        logits = F.linear(x, self.token.weight)
        return (logits, info) if return_bridge else logits


def _device_settings(device):
    device = torch.device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _load(spec, device):
    parent = Path(spec["parent_dir"]).resolve()
    checkpoint = parent / "model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = new_model(payload["spec"], device)
    model.load_state_dict(payload["model"])
    model.__class__ = ReencodingGPT
    model.bridge_start = len(WORDS) + payload["spec"]["heads_n"]
    model.bridge_count = payload["spec"]["bridges_n"]
    model.configure("baseline", 0)
    model.eval().requires_grad_(False)
    world = dict(np.load(parent / "world.npz"))
    rebuilt = build_world(payload["spec"])
    if data_digest(world) != data_digest(rebuilt):
        raise AssertionError("Saved world differs from the parent specification")
    oracle = torch.full(
        (model.config.vocab_size, model.config.vocab_size), -1, device=device, dtype=torch.long
    )
    for h, r, target in np.concatenate((world["common_atomic"], world["extra_atomic"])):
        oracle[h, r] = int(target)
    return model, world, oracle, payload["spec"], checkpoint


@torch.no_grad()
def _diagnostics(model, rows, device, batch_size):
    prompts = np.asarray([composite_sentence(row)[:7] for row in rows])
    collected = {}
    model.eval()
    for start in range(0, len(rows), batch_size):
        tokens = torch.as_tensor(prompts[start : start + batch_size], device=device)
        _, info = model(tokens, return_bridge=True)
        values = {
            "decoded_bridge": info["decoded"],
            "selected_bridge": info["selected"],
            "oracle_available": info["oracle_available"],
            "state_norm": info["state"].norm(dim=-1),
            "replacement_norm": info["replacement"].norm(dim=-1),
        }
        for key, value in values.items():
            collected.setdefault(key, []).append(value.cpu().numpy())
    result = {key: np.concatenate(value) for key, value in collected.items()}
    result["decoded_bridge_correct"] = result["decoded_bridge"] == rows[:, 2]
    result["selected_bridge_correct"] = result["selected_bridge"] == rows[:, 2]
    result["decoded_is_entity"] = result["decoded_bridge"] >= len(WORDS)
    return result


def _subset_metric(correct, mask):
    return {
        "n": int(mask.sum()),
        "coverage": float(mask.mean()),
        "accuracy": float(correct[mask].mean()) if mask.any() else None,
    }


@torch.no_grad()
def _evaluate_condition(model, world, device, batch_size):
    # Reuse the established complete answer/period/EOS scorer unchanged. Its
    # canonical batch is 512; diagnostics use the same shapes for exact decoding.
    if batch_size != 512:
        raise ValueError("Use batch_size=512 to preserve established evaluation arithmetic")
    metrics, predictions = evaluate(model, world, "low", device)
    model.eval()
    for name in ("common_atomic", "train_composite", "familiar_test", "strict_test"):
        generated = predictions[name + "_generated"]
        metrics[name]["period_accuracy"] = float((generated[:, 1] == 5).mean())
        metrics[name]["eos_accuracy"] = float((generated[:, 2] == 1).mean())
        metrics[name]["format_accuracy"] = float(
            ((generated[:, 1] == 5) & (generated[:, 2] == 1)).mean()
        )
    for name in ("familiar_test", "strict_test"):
        rows = world[name]
        info = _diagnostics(model, rows, device, batch_size)
        predictions.update({name + "_" + key: value for key, value in info.items()})
        for key in ("decoded_bridge_correct", "selected_bridge_correct", "decoded_is_entity"):
            metrics[name][key + "_coverage"] = float(info[key].mean())
        if model.condition == "wrong_entity":
            metrics[name]["alternative_accidental_gold_match"] = float(
                info["selected_bridge_correct"].mean()
            )
        correct = predictions[name + "_correct"]
        metrics[name]["self_bridge_correct_subset"] = _subset_metric(
            correct, info["decoded_bridge_correct"]
        )
        first = rows[:, [0, 1, 2]]
        second = rows[:, [2, 3, 4]]
        for role, atoms in (("first", first), ("second", second)):
            _, raw = generate_rows(model, atoms, device, batch_size=batch_size)
            predictions.update({f"{name}_{role}_{key}": value for key, value in raw.items()})
        autonomous_second = second.copy()
        autonomous_second[:, 0] = predictions[name + "_first_generated"][:, 0]
        _, raw = generate_rows(model, autonomous_second, device, batch_size=batch_size)
        predictions.update({f"{name}_autonomous_{key}": value for key, value in raw.items()})
    return metrics, predictions


def cases(spec):
    """One baseline and the frozen alpha/variant/condition matrix."""
    conditions = list(spec.get("conditions", CONDITIONS))
    if len(conditions) != len(set(conditions)) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Conditions must be distinct known names")
    alphas = spec.get("alphas", [spec.get("alpha", 0)])
    variants = spec.get("variants", ["norm_matched"])
    if len(alphas) != len(set(alphas)) or not alphas or any(not 0 <= a <= 1 for a in alphas):
        raise ValueError("alphas must contain distinct values in [0,1]")
    if len(variants) != len(set(variants)) or not variants:
        raise ValueError("variants must be distinct and nonempty")
    if any(v not in ("norm_matched", "paper_unit") for v in variants):
        raise ValueError("Unknown re-encoding variant")
    matrix = {"baseline": {"condition": "baseline", "alpha": 0, "variant": "norm_matched"}}
    for variant in variants:
        for alpha in alphas:
            for condition in conditions:
                if condition == "baseline":
                    continue
                name = f"{variant}-a{float(alpha):g}-{condition}"
                matrix[name] = {"condition": condition, "alpha": float(alpha), "variant": variant}
    return matrix


def _evaluate_all(model, world, oracle, spec, callback=None):
    matrix = cases(spec)
    metrics, arrays = {}, {}
    for name, case in matrix.items():
        model.configure(
            case["condition"],
            case["alpha"],
            oracle if case["condition"] == "oracle" else None,
            case["variant"],
        )
        metrics[name], predictions = _evaluate_condition(
            model, world, next(model.parameters()).device, spec.get("batch_size", 512)
        )
        arrays.update({name + "__" + key: value for key, value in predictions.items()})
        if callback is not None:
            callback(name, case, metrics[name], predictions)
    for name in ("familiar_test", "strict_test"):
        baseline = arrays[f"baseline__{name}_coverage"]
        # This is the fixed pre-intervention prerequisite pool for every case.
        # In particular, the destructive alternative-entity control cannot
        # determine which chains enter the main conditional comparison.
        arrays[f"common__{name}_atomic_correct"] = baseline
        for condition in matrix:
            correct = arrays[f"{condition}__{name}_correct"]
            arrays[f"{condition}__{name}_baseline_common_atoms"] = baseline
            metrics[condition][name]["fixed_baseline_atoms_subset"] = _subset_metric(
                correct, baseline
            )
            self_ok = arrays[f"{condition}__{name}_decoded_bridge_correct"]
            joint = baseline & self_ok
            arrays[f"{condition}__{name}_common_atoms_and_self_bridge"] = joint
            metrics[condition][name]["common_atoms_and_self_bridge_subset"] = _subset_metric(
                correct, joint
            )
    model.configure("baseline", 0)
    return metrics, arrays


def run(spec, out, device="cuda:2"):
    """Evaluate one existing checkpoint; never train or overwrite an attempt."""
    out = Path(out)
    if (out / "run.json").exists():
        raise FileExistsError(f"Do not overwrite an attempt: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = _device_settings(device)
    started = time.perf_counter()
    model, world, oracle, parent_spec, checkpoint = _load(spec, device)
    digest = model_digest(model)
    source = {
        str(Path(__file__).resolve()): file_hash(__file__),
        "parent_checkpoint": file_hash(checkpoint),
    }
    manifest = {
        "spec": spec,
        "parent_spec": parent_spec,
        "source": source,
        "world_sha256": data_digest(world),
        "model_sha256": digest,
        "pid": os.getpid(),
        "device": str(device),
        "started_utc": utc(),
        "new_training_updates": 0,
        "optimizer_updates": 0,
        "cases": cases(spec),
        "formula_norm_matched": "(1-alpha)*h + alpha*||h||*normalize(E_decoded)",
        "formula_paper_unit": "(1-alpha)*h + alpha*normalize(E_decoded)",
        "decoder": "original ln_final and tied output, argmax over the full vocabulary",
        "wrong_entity_scope": "non-self bridge entity, independent of gold; may equal true bridge",
        "bridge_token_start": model.bridge_start,
        "bridge_token_count": model.bridge_count,
        "main_prerequisite_subset": "both necessary atoms correct under the fixed baseline",
        "alpha_zero_exact_bypass": True,
        "allow_tf32": True,
    }
    write_json(out / "run.json", manifest)
    np.savez_compressed(out / "world.npz", **world)
    history = []

    def record(name, case, metrics, predictions):
        history.append(
            {
                "step": len(history) + 1,
                "case": name,
                **case,
                "metrics": metrics,
                "optimizer_updates": 0,
                "wall_seconds": time.perf_counter() - started,
                "created_utc": utc(),
            }
        )
        # Each condition is durable before the next one starts.
        np.savez_compressed(out / f"predictions-{name}.npz", **predictions)
        write_json(out / "learning.json", history)
        write_json(
            out / "status.json",
            {
                "state": "running",
                "step": len(history),
                "budget": len(manifest["cases"]),
                "optimizer_updates": 0,
            },
        )

    metrics, predictions = _evaluate_all(model, world, oracle, spec, record)
    assert model_digest(model) == digest, "Evaluation changed model parameters"
    # Final common-prerequisite subsets are known only after the full matrix.
    write_json(out / "learning.json", history)
    # At zero weight all conditions must recover complete baseline generation.
    for key, value in predictions.items():
        if key.endswith(("_generated", "_answer_nll")):
            case, suffix = key.split("__", 1)
            if manifest["cases"][case]["alpha"] == 0:
                np.testing.assert_array_equal(value, predictions["baseline__" + suffix])
    np.savez_compressed(out / "predictions.npz", **predictions)
    write_json(out / "metrics.json", metrics)
    result = {
        "state": "evaluation-complete-awaiting-independent-audit",
        "finished_utc": utc(),
        "wall_seconds": time.perf_counter() - started,
        "model_sha256": digest,
        "new_training_updates": 0,
        "optimizer_updates": 0,
        "conditions": list(metrics),
        "prediction_arrays": len(predictions),
        "step": len(history),
    }
    write_json(out / "evaluation-complete.json", result)
    write_json(out / "status.json", {**result, "step": len(history)})
    return result


def audit(out, device="cuda:2"):
    """Reload weights and independently recompute every saved prediction array."""
    out = Path(out)
    device = _device_settings(device)
    manifest = json.loads((out / "run.json").read_text())
    model, world, oracle, _, checkpoint = _load(manifest["spec"], device)
    assert file_hash(checkpoint) == manifest["source"]["parent_checkpoint"]
    assert model_digest(model) == manifest["model_sha256"]
    assert data_digest(world) == manifest["world_sha256"]
    metrics, actual = _evaluate_all(model, world, oracle, manifest["spec"])
    saved = dict(np.load(out / "predictions.npz"))
    assert set(saved) == set(actual), "Prediction array set changed"
    max_error = 0.0
    for key, value in actual.items():
        if value.dtype.kind == "f":
            max_error = max(max_error, float(np.max(np.abs(value - saved[key]))))
            np.testing.assert_allclose(value, saved[key], atol=1e-5, rtol=1e-5)
        else:
            np.testing.assert_array_equal(value, saved[key])
    # Recount complete-answer accuracy independently of the established scorer.
    for condition, groups in metrics.items():
        for name in ("common_atomic", "train_composite", "familiar_test", "strict_test"):
            pred = actual[f"{condition}__{name}_generated"]
            correct = (pred[:, 0] == world[name][:, -1]) & (pred[:, 1] == 5) & (pred[:, 2] == 1)
            np.testing.assert_array_equal(correct, actual[f"{condition}__{name}_correct"])
            assert float(correct.mean()) == groups[name]["accuracy"]
    assert model_digest(model) == manifest["model_sha256"]
    result = {
        "passed": True,
        "independently_reloaded": True,
        "raw_predictions_recounted": True,
        "prediction_arrays": len(actual),
        "max_float_error": max_error,
        "audit_pid": os.getpid(),
        "independent_process": os.getpid() != manifest["pid"],
        "finished_utc": utc(),
        "metrics": metrics,
    }
    write_json(out / "audit.json", result)
    complete = json.loads((out / "evaluation-complete.json").read_text())
    complete.update(state="complete", independently_reloaded=True, audit_finished_utc=utc())
    write_json(out / "complete.json", complete)
    write_json(out / "status.json", {**complete, "step": len(metrics), "optimizer_updates": 0})
    return result
