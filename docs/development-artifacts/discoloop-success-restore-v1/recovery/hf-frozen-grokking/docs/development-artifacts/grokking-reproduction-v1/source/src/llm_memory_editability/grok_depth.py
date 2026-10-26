"""Small GPT-2 composition learning, adapted from Wang et al. (NeurIPS 2024).

Data/target conventions follow GrokkedTransformer commit 734ca654ec7a71dd6737d640407fac14491d538c.
The compact vocabulary, widths and data sizes are explicit experimental adaptations.
CUDA Graph capture is an execution optimization, not a change to the training objective.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_model import CausalLM, ModelConfig, matmul_flops


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


class SmallGPT(CausalLM):
    """GPT-2 pre-LN, tied output, learned positions, GELU, standard dropout."""

    def __init__(self, config, dropout=0.1):
        super().__init__(config)
        self.dropout = dropout

    def forward(self, tokens, positions=None):
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
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
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, self.token.weight)


def pack_rows(rows):
    """Right-pad; supervise tail and EOS only. No BOS or answer delimiter."""
    rows = np.asarray(rows, dtype=np.int64)
    x = np.zeros((len(rows), 4), dtype=np.int64)
    x[:, : rows.shape[1]] = rows
    positions = np.tile([rows.shape[1] - 2, rows.shape[1] - 1], (len(rows), 1))
    labels = np.c_[rows[:, -1], np.ones(len(rows), dtype=np.int64)]
    return x, positions, labels


def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
        capturable=True,
    )


class GraphStep:
    """Capture complete AdamW step and undo warmup without rebinding state tensors."""

    def __init__(self, model, optimizer, table, batch_size, clip=1.0):
        self.index = torch.zeros(batch_size, dtype=torch.long, device=table[0].device)
        self.model, self.optimizer, self.table, self.clip = model, optimizer, table, clip
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state = {
            p: {
                k: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in s.items()
            }
            for p, s in optimizer.state.items()
        }
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(table[0].device)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.eager()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.grad_norm = self.eager()
        torch.cuda.synchronize()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(weights[k])
            for param, values in optimizer.state.items():
                before = state.get(param, {})
                for k, v in values.items():
                    if torch.is_tensor(v):
                        if k in before:
                            v.copy_(before[k])
                        else:
                            v.zero_()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, table[0].device)

    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        x, pos, labels = (t[self.index] for t in self.table)
        logits = self.model(x, positions=pos)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm

    def __call__(self, indices):
        self.index.copy_(indices)
        self.graph.replay()
        return self.loss


class EpochStream:
    """Uniform shuffled epochs of the combined atomic/composite training set."""

    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.remaining = np.empty(0, dtype=np.int64)

    def take(self, n):
        parts = []
        while n:
            if not len(self.remaining):
                self.remaining = self.rng.permutation(self.size)
            count = min(n, len(self.remaining))
            parts.append(self.remaining[:count])
            self.remaining = self.remaining[count:]
            n -= count
        return np.concatenate(parts)

    def state_dict(self):
        return {
            "size": self.size,
            "rng": self.rng.bit_generator.state,
            "remaining": self.remaining.copy(),
        }

    def load_state_dict(self, state):
        assert state["size"] == self.size
        self.rng.bit_generator.state = state["rng"]
        self.remaining = state["remaining"].copy()


@torch.no_grad()
def evaluate_rows(model, rows, device, batch_size=1024):
    model.eval()
    packed = pack_rows(rows)
    answers, stops, losses = [], [], []
    for start in range(0, len(rows), batch_size):
        x, pos, labels = (
            torch.as_tensor(t[start : start + batch_size], device=device) for t in packed
        )
        logits = model(x, pos)
        losses.append(
            F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
            .view(-1, 2)
            .cpu()
            .numpy()
        )
        answer = logits[:, 0].argmax(-1)
        # Actual generated answer is fed back when producing EOS.
        x[torch.arange(len(x), device=device), pos[:, 1]] = answer
        stop = model(x, pos)[:, 1].argmax(-1)
        answers.append(answer.cpu().numpy())
        stops.append(stop.cpu().numpy())
    model.train()
    if not len(rows):
        return {"n": 0, "answer_accuracy": None, "accuracy": None, "nll": None}, {}
    answer, stop, nll = np.concatenate(answers), np.concatenate(stops), np.concatenate(losses)
    correct = answer == rows[:, -1]
    return {
        "n": len(rows),
        "answer_accuracy": float(correct.mean()),
        "accuracy": float((correct & (stop == 1)).mean()),
        "nll": float(nll.mean()),
    }, {"answer": answer, "stop": stop, "nll": nll, "target": rows[:, -1]}


@torch.no_grad()
def two_calls(model, rows, device):
    if not len(rows):
        return {"n": 0, "answer_accuracy": None, "accuracy": None, "nll": None}
    first = np.c_[rows[:, :2], np.zeros(len(rows), dtype=np.int64)]
    _, first_pred = evaluate_rows(model, first, device)
    second = np.c_[first_pred["answer"], rows[:, 2], rows[:, 3]]
    result, second_pred = evaluate_rows(model, second, device)
    result["accuracy"] = float(
        (
            (second_pred["answer"] == rows[:, -1])
            & (second_pred["stop"] == 1)
            & (first_pred["stop"] == 1)
        ).mean()
    )
    return result


def source_hash(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def run(spec, world, out, device="cuda:0", resume=False):
    """Run a fixed scientific budget. Intermediate evaluations never change training."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(spec["initialization"])
    torch.cuda.manual_seed_all(spec["initialization"])
    device = torch.device(device)
    torch.cuda.set_device(device)
    cfg = ModelConfig(
        vocab_size=2 + spec["entities"] + spec["relations"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    model = SmallGPT(cfg, spec["dropout"]).to(device)
    lr = torch.tensor(spec["lr"], dtype=torch.float32, device=device)
    opt = make_optimizer(model, lr, spec["weight_decay"])
    atoms, comps = pack_rows(world["atomic"]), pack_rows(world["train_composite"])
    table = tuple(
        torch.as_tensor(np.concatenate([a, c]), device=device)
        for a, c in zip(atoms, comps, strict=True)
    )
    stream = EpochStream(len(table[0]), spec["stream_seed"])
    counts = {"atomic": 0, "composite": 0}
    elapsed_training, elapsed_eval = 0.0, 0.0
    start_step = 0
    checkpoint = out / "latest.pt"
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        assert state["spec"] == spec
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        # load_state_dict replaces the LR tensor; retain the optimizer's live tensor.
        lr = opt.param_groups[0]["lr"]
        for group in opt.param_groups:
            group["lr"] = lr
        stream.load_state_dict(state["stream"])
        counts = state["counts"]
        elapsed_training = state["elapsed_training"]
        elapsed_eval = state["elapsed_eval"]
        start_step = state["step"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    started = time.perf_counter()
    graph = GraphStep(model, opt, table, spec["batch_size"])
    capture_seconds = time.perf_counter() - started
    names = ("atomic", "train_composite", "test_composite", "ood_composite")
    nparams = sum(p.numel() for p in model.parameters())
    flop_step = matmul_flops(cfg, spec["batch_size"], sequence=4, output_positions=2)
    rows_log = []
    if (out / "learning.json").exists():
        rows_log = json.loads((out / "learning.json").read_text())
    if resume:
        # A crash can leave a log newer than the atomically replaced checkpoint.
        rows_log = list({r["step"]: r for r in rows_log if r["step"] <= start_step}.values())
        rows_log.sort(key=lambda r: r["step"])
        if start_step == spec["steps"]:
            endpoint = rows_log[-1]
            write_json(
                out / "complete.json",
                {
                    "finished_utc": utc(),
                    "spec": spec,
                    "parameters": nparams,
                    "training_seconds": elapsed_training,
                    "evaluation_seconds": elapsed_eval,
                    "endpoint": endpoint,
                    "completion_recovered_from_final_checkpoint": True,
                },
            )
            write_json(
                out / "status.json",
                {
                    "state": "complete",
                    "step": start_step,
                    "budget": spec["steps"],
                    "updated_utc": utc(),
                    "atomic": endpoint["atomic"]["accuracy"],
                    "train": endpoint["train_composite"]["accuracy"],
                    "test": endpoint["test_composite"]["accuracy"],
                },
            )
            return rows_log[-1]

    def measure(step, last_loss=None):
        nonlocal elapsed_eval
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        row = {
            "step": step,
            "utc": utc(),
            "parameters": nparams,
            "training_seconds": elapsed_training,
            "capture_seconds": capture_seconds,
            "examples": step * spec["batch_size"],
            "counts": counts.copy(),
            "effective_input_tokens": counts["atomic"] * 3 + counts["composite"] * 4,
            "supervised_tokens": step * spec["batch_size"] * 2,
            "estimated_training_flops": step * flop_step,
            "last_batch_loss": last_loss,
        }
        predictions = {}
        for name in names:
            row[name], pred = evaluate_rows(model, world[name], device)
            predictions.update({name + "_" + k: v for k, v in pred.items()})
        row["two_calls"] = two_calls(model, world["test_composite"], device)
        torch.cuda.synchronize()
        elapsed_eval += time.perf_counter() - t0
        row["evaluation_seconds"] = elapsed_eval
        rows_log.append(row)
        write_json(out / "learning.json", rows_log)
        np.savez_compressed(out / f"predictions-{step:07d}.npz", **predictions)
        state = {
            "step": step,
            "spec": spec,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "elapsed_training": elapsed_training,
            "elapsed_eval": elapsed_eval,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(checkpoint)
        if step in spec["weight_nodes"]:
            torch.save(
                {"spec": spec, "step": step, "model": model.state_dict()},
                out / f"weights-{step:07d}.pt",
            )
        status = {
            "state": "complete" if step == spec["steps"] else "running",
            "step": step,
            "budget": spec["steps"],
            "updated_utc": utc(),
            "atomic": row["atomic"]["accuracy"],
            "train": row["train_composite"]["accuracy"],
            "test": row["test_composite"]["accuracy"],
        }
        write_json(out / "status.json", status)
        print(json.dumps(status), flush=True)
        return row

    if not resume:
        measure(0)
    nodes = [s for s in spec["nodes"] if s > start_step]
    for end in nodes:
        last_loss = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step = start_step
        while step < end:
            n = min(512, end - step)
            ix = stream.take(n * spec["batch_size"]).reshape(n, spec["batch_size"])
            atom_count = int((ix < len(world["atomic"])).sum())
            counts["atomic"] += atom_count
            counts["composite"] += ix.size - atom_count
            gpu_ix = torch.as_tensor(ix, device=device)
            for j in range(n):
                lr.fill_(spec["lr"] * min(1.0, (step + j + 1) / spec["warmup"]))
                last_loss = graph(gpu_ix[j])
            step += n
        torch.cuda.synchronize()
        elapsed_training += time.perf_counter() - t0
        loss_value = float(last_loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Nonfinite loss at step {end}")
        row = measure(end, loss_value)
        start_step = end
    write_json(
        out / "complete.json",
        {
            "finished_utc": utc(),
            "spec": spec,
            "parameters": nparams,
            "training_seconds": elapsed_training,
            "evaluation_seconds": elapsed_eval,
            "endpoint": row,
        },
    )
    return row
