"""Intermediate supervision and input-compatible states in complete causal GPTs.

The intervention observes the first executed block at the first relation token.
Every arm executes the same forward/backward graph. Ordinary full-token CE,
auxiliary entity CE, and state geometry are recorded separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import EpochStream, GraphStep, make_optimizer, utc, write_json
from .grok_loop_model import LoopGPT, flops
from .latent_scaling import build_world, construct, model_digest
from .storage_composition import data_digest, evaluate, file_hash, pack_sentences
from .storage_frontier import learning_rate


class RepresentationGPT(LoopGPT):
    """Unchanged GPT/Loop computation, optionally exposing a causal early state."""

    def forward(self, tokens, positions=None, repeats=None, return_bridge=False):
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
        bridge = None
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
            if index == 0 and return_bridge:
                bridge = x[:, 3]
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        logits = F.linear(x, self.token.weight)
        return (logits, bridge) if return_bridge else logits


def new_model(spec, device):
    # Construct through the established initializer, then replace only the type.
    model = construct(spec, device)
    model.__class__ = RepresentationGPT
    return model


def pack_training(world):
    strata = [world[k] for k in ("common_atomic", "train_composite", "anchor_atomic")]
    tokens, labels = (
        np.concatenate(parts)
        for parts in zip(*(pack_sentences(rows) for rows in strata), strict=True)
    )
    # Column 2 is the atomic answer or the composition's first-hop answer.
    # It never enters the composed input; it is an explicit auxiliary target.
    targets = np.concatenate([rows[:, 2] for rows in strata])
    return (tokens, labels, targets), [len(rows) for rows in strata]


def objective(model, tokens, labels, targets, bridge_weight, alignment_weight):
    logits, bridge = model(tokens, return_bridge=True)
    text_ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    bridge_logits = F.linear(model.ln_final(bridge), model.token.weight)
    bridge_ce = F.cross_entropy(bridge_logits, targets)
    canonical = model.token(targets).detach()
    alignment = (
        (F.normalize(bridge, dim=-1) - F.normalize(canonical, dim=-1)).square().sum(-1).mean()
    )
    parts = torch.stack((text_ce, bridge_ce, alignment))
    return text_ce + bridge_weight * bridge_ce + alignment_weight * alignment, parts


class RepresentationStep(GraphStep):
    def __init__(self, model, optimizer, table, batch_size, spec):
        self.spec = spec
        super().__init__(model, optimizer, table, batch_size)

    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        tokens, labels, targets = (part[self.index] for part in self.table)
        loss, self.components = objective(
            self.model,
            tokens,
            labels,
            targets,
            self.spec["bridge_weight"],
            self.spec["alignment_weight"],
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


def run_name(spec):
    return f"{spec['phase']}-w{spec['world']}-i{spec['initialization']}-{spec['arm']}"


def train(spec, out, source, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(f"Do not overwrite an attempt: {out}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(device)
    started = time.perf_counter()
    world = build_world(spec)
    np.savez_compressed(out / "world.npz", **world)
    arrays, sizes = pack_training(world)
    model = new_model(spec, device)
    initial_hash = model_digest(model)
    write_json(
        out / "run.json",
        {
            "spec": spec,
            "source": source,
            "world_sha256": data_digest(world),
            "initial_model_sha256": initial_hash,
            "pid": os.getpid(),
            "gpu": device.index,
            "gpu_name": torch.cuda.get_device_name(device),
            "dtype": "float32",
            "parameters": sum(p.numel() for p in model.parameters()),
            "atomic_examples": sizes[0] + sizes[2],
            "composition_examples": sizes[1],
            "vocab_size": model.config.vocab_size,
        },
    )
    table = tuple(torch.as_tensor(a, device=device) for a in arrays)
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    model.train()
    graph = RepresentationStep(model, optimizer, table, spec["batch_size"], spec)
    assert model_digest(model) == initial_hash, "CUDA capture changed initialization"
    streams = [EpochStream(n, spec["stream_seed"] + j) for j, n in enumerate(sizes)]
    offsets = np.cumsum([0, *sizes[:-1]])
    counts = [np.zeros(n, dtype=np.int64) for n in sizes]
    input_hash = hashlib.sha256()
    history, step, training_seconds = [], 0, 0.0
    flop_step = flops(model.config, spec["repeats"], spec["batch_size"], 9, output_positions=9)
    # All arms also execute the auxiliary readout and normalization.
    flop_step += 6 * spec["batch_size"] * model.config.width * model.config.vocab_size

    def measure():
        metrics, predictions = evaluate(model, world, "low", device)
        parts = graph.components.detach().cpu().tolist() if step else [None] * 3
        row = {
            "step": step,
            "metrics": metrics,
            "text_ce": parts[0],
            "bridge_ce": parts[1],
            "alignment_loss": parts[2],
            "training_seconds": training_seconds,
            "wall_seconds": time.perf_counter() - started,
            "examples": step * spec["batch_size"],
            "supervised_tokens": step * spec["batch_size"] // 3 * (7 + 9 + 7),
            "auxiliary_target_presentations": step * spec["batch_size"],
            "estimated_matmul_training_flops": step * flop_step,
            "composition_epochs": step * spec["batch_size"] / 3 / sizes[1],
        }
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        write_json(out / "status.json", {"state": "running", "step": step, "budget": spec["steps"]})
        print(
            json.dumps(
                {
                    "run": run_name(spec),
                    "step": step,
                    "atomic": metrics["common_atomic"]["accuracy"],
                    "train": metrics["train_composite"]["accuracy"],
                    "familiar": metrics["familiar_test"]["accuracy"],
                    "strict": metrics["strict_test"]["accuracy"],
                }
            ),
            flush=True,
        )
        model.train()

    measure()
    for end in spec["nodes"][1:]:
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        while step < end:
            n = min(128, end - step)
            indices = []
            for j, stream in enumerate(streams):
                drawn = stream.take(n * spec["batch_size"] // 3).reshape(n, -1)
                counts[j] += np.bincount(drawn.ravel(), minlength=sizes[j])
                indices.append(drawn + offsets[j])
            indices = np.concatenate(indices, axis=1)
            input_hash.update(indices.tobytes())
            indices = torch.as_tensor(indices, device=device)
            for j in range(n):
                lr.fill_(learning_rate(spec, step + j + 1))
                loss = graph(indices[j])
            step += n
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - began
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f"Nonfinite objective at {step}")
        measure()
    model.eval()
    torch.save(
        {"spec": spec, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        out / "model.pt",
    )
    np.savez_compressed(out / "exposures.npz", **{f"stratum{j}": c for j, c in enumerate(counts)})
    write_json(
        out / "complete.json",
        {
            "spec": spec,
            "source": source,
            "finished_utc": utc(),
            "metrics": history[-1]["metrics"],
            "training_seconds": training_seconds,
            "world_sha256": data_digest(world),
            "initial_model_sha256": initial_hash,
            "sample_stream_sha256": input_hash.hexdigest(),
            "model_sha256": file_hash(out / "model.pt"),
        },
    )
    write_json(out / "status.json", {"state": "complete", "step": step})


def audit(out, device):
    out = Path(out)
    # Match trainer arithmetic before comparing generated tokens and NLL.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    spec = payload["spec"]
    model = new_model(spec, device)
    model.load_state_dict(payload["model"])
    world = build_world(spec)
    metrics, predictions = evaluate(model, world, "low", device)
    saved = np.load(out / f"predictions-{spec['steps']:06d}.npz")
    max_nll_error = 0.0
    for key, actual in predictions.items():
        if key.endswith("nll"):
            max_nll_error = max(max_nll_error, float(np.max(np.abs(actual - saved[key]))))
            np.testing.assert_allclose(actual, saved[key], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(actual, saved[key])
    write_json(
        out / "audit.json",
        {"passed": True, "metrics": metrics, "max_nll_error": max_nll_error, "allow_tf32": True},
    )
