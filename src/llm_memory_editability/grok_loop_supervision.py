"""Paired continuation: terminal versus distributed answer supervision.

Historical model/trainer files remain unchanged. All heads branch off the same
trajectory; their LayerNorm outputs never replace the recurrent residual state.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_model import ModelConfig
from .grok_depth import GraphStep, make_optimizer, utc, write_json
from .grok_loop_data import StratifiedStream
from .grok_loop_model import LoopGPT, flops
from .grok_multihop import pack_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensors):
    h = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SupervisionGPT(LoopGPT):
    def states(self, tokens, repeats):
        """Yield full group-boundary states with exactly the native dropout order."""
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
        for _ in range(repeats):
            for block in self.blocks:
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
            yield x

    def readout(self, x, positions=None):
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(x), device=x.device)[:, None], positions]
        return F.linear(x, self.token.weight)

    def at_loops(self, tokens, positions, loops=(2, 3, 4)):
        if not loops or tuple(sorted(set(loops))) != tuple(loops) or loops[0] < 1:
            raise ValueError("Loop endpoints must be positive, unique, sorted")
        return tuple(
            self.readout(x, positions)
            for r, x in enumerate(self.states(tokens, max(loops)), 1)
            if r in loops
        )


def answer_losses(model, x, positions, labels):
    return torch.stack(
        [F.cross_entropy(z.flatten(0, 1), labels.flatten()) for z in model.at_loops(x, positions)]
    )


class SupervisionStep(GraphStep):
    """Both arms execute all three CE/backward branches, including zero weights."""

    def __init__(self, model, optimizer, table, batch_size, arm):
        if arm not in ("single", "multi"):
            raise ValueError(arm)
        weights = [0.0, 0.0, 1.0] if arm == "single" else [1 / 3] * 3
        self.weights = torch.tensor(weights, device=table[0].device)
        super().__init__(model, optimizer, table, batch_size)

    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        x, pos, labels = (t[self.index] for t in self.table)
        self.components = answer_losses(self.model, x, pos, labels)
        loss = (self.components * self.weights).sum()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


def from_state(state, device):
    spec = state["spec"]
    cfg = ModelConfig(
        vocab_size=2 + spec["entities"] + spec["relations"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    model = SupervisionGPT(
        cfg,
        repeats=spec["repeats"],
        dropout=spec["dropout"],
        initialization=spec["init_scheme"],
    ).to(device)
    model.load_state_dict(state["model"])
    return model


def score_predictions(pred):
    correct = pred["answer"] == pred["target"]
    return {
        "n": len(correct),
        "accuracy": float((correct & (pred["stop"] == 1)).mean()),
        "answer_accuracy": float(correct.mean()),
        "eos_accuracy": float((pred["stop"] == 1).mean()),
        "nll": float(pred["nll"].mean()),
        "answer_nll": float(pred["nll"][:, 0].mean()),
        "answer_probability": float(np.exp(-pred["nll"][:, 0].astype(float)).mean()),
        "margin": float(pred["margin"].mean()),
    }


@torch.no_grad()
def evaluate(model, rows, device, repeats, batch_size=1024):
    was_training = model.training
    model.eval()
    packed = pack_rows(rows, 4)
    parts = []
    for start in range(0, len(rows), batch_size):
        x, pos, labels = (
            torch.as_tensor(a[start : start + batch_size], device=device) for a in packed
        )
        x = x.clone()
        z = model(x, pos, repeats=repeats)
        nll = F.cross_entropy(z.flatten(0, 1), labels.flatten(), reduction="none").view(-1, 2)
        answer = z[:, 0].argmax(-1)
        x[torch.arange(len(x), device=device), pos[:, 1]] = answer
        eos = model(x, pos, repeats=repeats)[:, 1]
        target_score = z[:, 0].gather(1, labels[:, :1]).squeeze(1)
        alternatives = z[:, 0].clone()
        alternatives.scatter_(1, labels[:, :1], -torch.inf)
        parts.append(
            {
                "answer": answer.cpu().numpy(),
                "stop": eos.argmax(-1).cpu().numpy(),
                "target": labels[:, 0].cpu().numpy(),
                "nll": nll.cpu().numpy(),
                "margin": (target_score - alternatives.max(-1).values).cpu().numpy(),
                "logits": z.cpu().numpy(),
                "generated_eos_logits": eos.cpu().numpy(),
            }
        )
    model.train(was_training)
    pred = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return score_predictions(pred), pred


def load_world(source):
    with np.load(Path(source) / "world.npz", allow_pickle=False) as z:
        world = {k: z[k].copy() for k in z.files}
    world["metadata"] = json.loads((Path(source) / "world-metadata.json").read_text())
    return world


def restore_rng(state, device):
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)


def run(spec, out, device, resume=False):
    out, device = Path(out), torch.device(device)
    source = Path(spec["source"])
    origin = torch.load(source / "latest.pt", map_location=device, weights_only=False)
    if origin["step"] != 128000 or origin["spec"]["hops"] != 2:
        raise ValueError("Expected completed historical two-hop checkpoint")
    world = load_world(source)
    current = (
        torch.load(out / "latest.pt", map_location=device, weights_only=False) if resume else origin
    )
    if resume and current["continuation"] != spec:
        raise ValueError("Resume contract changed")
    model = from_state(current, device)
    lr = torch.tensor(spec["lr"], device=device)
    opt = make_optimizer(model, lr, origin["spec"]["weight_decay"])
    opt.load_state_dict(current["optimizer"])
    lr = opt.param_groups[0]["lr"]
    for group in opt.param_groups:
        group["lr"] = lr
    lr.fill_(spec["lr"])
    packed = [pack_rows(world[name], 4) for name in ("atomic", "train_composite")]
    table = tuple(
        torch.as_tensor(np.concatenate(a), device=device) for a in zip(*packed, strict=True)
    )
    stream = StratifiedStream(
        len(world["atomic"]),
        len(world["train_composite"]),
        256,
        32,
        origin["spec"]["stream_seed"],
    )
    stream.load_state_dict(current["stream"])
    restore_rng(current, device)
    step = current["step"] if resume else 0
    training_seconds = current.get("continuation_training_seconds", 0.0) if resume else 0.0
    records = json.loads((out / "learning.json").read_text()) if resume else []
    records = [r for r in records if r["step"] <= step]
    before_capture = tensor_digest(model.state_dict())
    optimizer_before = {
        p: {k: v.clone() for k, v in s.items() if torch.is_tensor(v)} for p, s in opt.state.items()
    }
    model.train()
    graph = SupervisionStep(model, opt, table, 256, spec["arm"])
    capture_ok = before_capture == tensor_digest(model.state_dict()) and all(
        torch.equal(v, opt.state[p][k]) for p, s in optimizer_before.items() for k, v in s.items()
    )
    if not capture_ok:
        raise ValueError("CUDA capture altered source model/optimizer")
    del optimizer_before
    flop_step = flops(model.config, 4, 256, 4, output_positions=6)
    sampling_hash = hashlib.sha256()
    # Persist cumulative per-row exposure as well as stream state for paired audit.
    exposure = (
        current["continuation_exposure"].cpu().numpy()
        if resume
        else np.zeros(len(table[0]), dtype=np.int64)
    )

    def measure(at, components=None):
        rng = {"cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)}
        row = {
            "step": at,
            "utc": utc(),
            "training_seconds": training_seconds,
            "estimated_training_matmul_flops": at * flop_step,
            "logical_examples": at * 256,
            "atomic_presentations": at * 32,
            "composite_presentations": at * 224,
            "weighted_target_tokens": at * 256 * 2,
            "executed_loss_target_tokens": at * 256 * 2 * 3,
            "last_batch_losses_2_3_4": components,
            "evaluations": {},
        }
        repeats = spec["endpoint_repeats"] if at in (0, spec["steps"]) else [2, 3, 4, 8]
        preds = {}
        for name in ("atomic", "test_full_composite", "ood_composite", "train_composite"):
            counts = [2, 3, 4] if name == "train_composite" else repeats
            for r in counts:
                metric, pred = evaluate(model, world[name], device, r)
                key = f"{name}_r{r}"
                row["evaluations"][key] = metric
                preds.update({key + "_" + k: v for k, v in pred.items()})
                if name == "atomic":
                    for subset in ("id_atomic", "ood_atomic"):
                        ids = {tuple(x) for x in world[subset]}
                        mask = np.array([tuple(x) in ids for x in world["atomic"]])
                        row["evaluations"][f"{subset}_r{r}"] = score_predictions(
                            {k: v[mask] for k, v in pred.items()}
                        )
        records.append(row)
        write_json(out / "learning.json", records)
        np.savez_compressed(out / f"predictions-{at:07d}.npz", **preds)
        restore_rng(rng, device)
        state = {
            "spec": origin["spec"],
            "continuation": spec,
            "step": at,
            "source_training_step": origin["step"],
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "stream": stream.state_dict(),
            "cpu_rng": rng["cpu_rng"],
            "cuda_rng": rng["cuda_rng"],
            "continuation_exposure": torch.as_tensor(exposure),
            "continuation_training_seconds": training_seconds,
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        torch.save(
            {"spec": origin["spec"], "step": at, "model": model.state_dict()},
            out / f"weights-{at:07d}.pt",
        )
        ev = row["evaluations"]
        status = {
            "state": "complete" if at == spec["steps"] else "running",
            "step": at,
            "arm": spec["arm"],
            "ood_r4": ev["ood_composite_r4"]["accuracy"],
            "ood_r8": ev["ood_composite_r8"]["accuracy"],
            "atomic_r4": ev["atomic_r4"]["accuracy"],
            "id_r4": ev["test_full_composite_r4"]["accuracy"],
            "utc": utc(),
        }
        write_json(out / "status.json", status)
        print(json.dumps(status), flush=True)
        return row

    if not resume:
        measure(0)
    for end in [n for n in spec["nodes"] if n > step]:
        model.train()
        torch.cuda.synchronize()
        start = time.perf_counter()
        while step < end:
            n = min(256, end - step)
            ix = np.stack([stream.take() for _ in range(n)])
            exposure += np.bincount(ix.ravel(), minlength=len(exposure))
            sampling_hash.update(ix.tobytes())
            gpu_ix = torch.as_tensor(ix, device=device)
            for j in range(n):
                graph(gpu_ix[j])
            step += n
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - start
        values = graph.components.detach().cpu().tolist()
        if not all(math.isfinite(v) for v in values):
            raise FloatingPointError(f"Nonfinite component losses at {step}")
        measure(step, values)
    if not records or records[-1]["step"] != spec["steps"]:
        raise ValueError("Budget not reached")
    write_json(
        out / "complete.json",
        {
            "spec": spec,
            "completed_utc": utc(),
            "capture_restore_passed": capture_ok,
            "training_seconds": training_seconds,
            "endpoint": records[-1],
            "initial_model_sha256": tensor_digest(origin["model"]),
            "final_model_sha256": tensor_digest(model.state_dict()),
            "exposure_sha256": hashlib.sha256(exposure.tobytes()).hexdigest(),
            "sampling_segment_sha256": sampling_hash.hexdigest(),
            "segment_start": current["step"] if resume else 0,
        },
    )
