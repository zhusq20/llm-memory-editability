"""Protocol replication of Wang et al.'s full-size two-hop grokking experiments.

The original GPT-2 vocabulary and configuration, answer/end-marker objective,
random graph, edge-level split, merged sampling and unique-depth initialization
are retained. SDPA, selecting supervised logits, CUDA graphs and bf16 AMP are
explicit execution/numerical adaptations, checked against Hugging Face GPT-2.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from .bios_model import ModelConfig, matmul_flops
from .grok_depth import EpochStream, GraphStep, utc, write_json

ROOT = Path("results/grokking-reproduction-v1")
ARTIFACTS = Path("docs/development-artifacts/grokking-reproduction-v1")
SOURCE_FILES = [
    "src/llm_memory_editability/grokking_reproduction.py",
    "scripts/run_grokking_reproduction.py",
    "tests/test_grokking_reproduction.py",
    "configs/grokking-reproduction-development-v1.json",
    "docs/development-artifacts/grokking-reproduction-v1/design.md",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_graph(seed, entities=2000, relations=200, degree=20, holdout=0.005):
    """Port the author's composition.ipynb; all tokens have shared graph roles.

    The fixed permutation makes different phi supports nested while preserving
    each support's uniform marginal distribution. Mixed OOD chains are retained
    for diagnostic evaluation, never included in training.
    """
    rng = np.random.RandomState(seed)
    atoms = np.empty((entities * degree, 3), dtype=np.int32)
    for h in range(entities):
        begin = h * degree
        atoms[begin : begin + degree, 0] = h
        atoms[begin : begin + degree, 1] = rng.choice(relations, degree, replace=False)
        atoms[begin : begin + degree, 2] = rng.randint(entities, size=degree)
    ood = np.zeros(len(atoms), dtype=bool)
    ood[rng.choice(len(atoms), round(len(atoms) * 0.05), replace=False)] = True
    first = np.repeat(np.arange(len(atoms)), degree)
    second = atoms[first, 2] * degree + np.tile(np.arange(degree), len(atoms))
    chains = np.column_stack(
        [atoms[first, 0], atoms[first, 1], atoms[second, 1], atoms[second, 2]]
    ).astype(np.int32)
    kind = ood[first].astype(np.int8) * 2 + ood[second].astype(np.int8)
    ii = np.flatnonzero(kind == 0)
    held = rng.uniform(size=len(ii)) <= holdout
    train_ids = ii[~held]
    permutation = rng.permutation(len(train_ids))
    return {
        "atoms": atoms,
        "ood": ood,
        "chains": chains,
        "first": first.astype(np.int32),
        "second": second.astype(np.int32),
        "kind": kind,
        "train_order": train_ids[permutation].astype(np.int32),
        "test_ii": ii[held].astype(np.int32),
    }


def audit_graph(world, degree):
    a, c = world["atoms"], world["chains"]
    first, second = world["first"], world["second"]
    assert np.array_equal(a[first, 2], a[second, 0])
    assert np.array_equal(c[:, 0], a[first, 0])
    assert np.array_equal(c[:, 1], a[first, 1])
    assert np.array_equal(c[:, 2], a[second, 1])
    assert np.array_equal(c[:, 3], a[second, 2])
    assert len(np.unique(a[:, :2], axis=0)) == len(a)
    assert np.all(np.bincount(a[:, 0]) == degree)
    assert not np.intersect1d(world["train_order"], world["test_ii"]).size
    assert np.all(world["kind"][world["train_order"]] == 0)
    assert np.all(world["kind"][world["test_ii"]] == 0)


def prepare_data(config):
    path = ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "complete.json").exists():
        complete = json.loads((path / "complete.json").read_text())
        assert complete["world"] == config["world"]
        assert digest(path / "world.npz") == complete["world_sha256"]
        return complete
    world = build_graph(config["world"])
    audit_graph(world, 20)
    np.savez_compressed(path / "world.npz", **world)
    rng = np.random.RandomState(config["evaluation_seed"])

    def sample(indices):
        return rng.choice(indices, min(3000, len(indices)), replace=False)

    min_support = round(min(s["phi"] for s in config["runs"]) * (~world["ood"]).sum())
    panels = {
        "atomic_id": world["atoms"][sample(np.flatnonzero(~world["ood"]))],
        "atomic_ood": world["atoms"][sample(np.flatnonzero(world["ood"]))],
        "train_ii": world["chains"][sample(world["train_order"][:min_support])],
        "test_ii": world["chains"][sample(world["test_ii"])],
        "test_io": world["chains"][sample(np.flatnonzero(world["kind"] == 1))],
        "test_oi": world["chains"][sample(np.flatnonzero(world["kind"] == 2))],
        "test_oo": world["chains"][sample(np.flatnonzero(world["kind"] == 3))],
    }
    np.savez_compressed(path / "panels.npz", **panels)
    tokenizer = GPT2TokenizerFast.from_pretrained(ROOT / "reference/gpt2", local_files_only=True)
    base = len(tokenizer)
    vocab = [f"<e_{i}>" for i in range(2000)] + [f"<r_{i}>" for i in range(200)]
    vocab += ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
    assert tokenizer.add_tokens(vocab) == len(vocab)
    assert tokenizer.convert_tokens_to_ids(vocab) == list(range(base, base + len(vocab)))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(path / "tokenizer")
    result = {
        "world": config["world"],
        "world_sha256": digest(path / "world.npz"),
        "panels_sha256": digest(path / "panels.npz"),
        "entities": 2000,
        "relations": 200,
        "degree": 20,
        "atomic_id": int((~world["ood"]).sum()),
        "atomic_ood": int(world["ood"].sum()),
        "candidate_train_ii": len(world["train_order"]),
        "full_test_ii": len(world["test_ii"]),
        "panels": {k: len(v) for k, v in panels.items()},
        "base_vocabulary": base,
        "vocab_size": len(tokenizer),
        "entity_offset": base,
        "relation_offset": base + 2000,
        "end_marker": tokenizer.convert_tokens_to_ids("</a>"),
        "pad_token": tokenizer.pad_token_id,
        "created_utc": utc(),
    }
    write_json(path / "complete.json", result)
    write_json(ARTIFACTS / "data-audit.json", result)
    return result


def encode_rows(rows, metadata):
    """Only tail and </a> are targets; padding cannot influence earlier positions."""
    rows = np.asarray(rows)
    x = np.full((len(rows), 4), metadata["pad_token"], dtype=np.int64)
    x[:, : rows.shape[1]] = rows
    x[:, 0] += metadata["entity_offset"]
    x[:, rows.shape[1] - 1] += metadata["entity_offset"]
    x[:, 1 : rows.shape[1] - 1] += metadata["relation_offset"]
    pos = np.tile([rows.shape[1] - 2, rows.shape[1] - 1], (len(rows), 1))
    labels = np.column_stack(
        [rows[:, -1] + metadata["entity_offset"], np.full(len(rows), metadata["end_marker"])]
    )
    return x, pos, labels


def audit_support_roles(config):
    world = dict(np.load(ROOT / "data/world.npz"))
    panels = dict(np.load(ROOT / "data/panels.npz"))
    atoms = world["atoms"]
    lookup = np.full((2000, 200), -1, dtype=np.int32)
    lookup[atoms[:, 0], atoms[:, 1]] = np.arange(len(atoms))
    results = []
    for phi in sorted({s["phi"] for s in config["runs"]}):
        support = round(phi * (~world["ood"]).sum())
        selected = world["train_order"][:support]
        first = np.bincount(world["first"][selected], minlength=len(atoms))
        second = np.bincount(world["second"][selected], minlength=len(atoms))
        seen_pairs = np.zeros((2000, 2000), dtype=bool)
        chains = world["chains"][selected]
        seen_pairs[chains[:, 0], chains[:, 3]] = True
        tasks = {}
        for name, rows in panels.items():
            if rows.shape[1] != 4:
                continue
            e1 = lookup[rows[:, 0], rows[:, 1]]
            e2 = lookup[atoms[e1, 2], rows[:, 2]]
            assert np.all(e1 >= 0) and np.all(e2 >= 0)
            tasks[name] = {
                "n": len(rows),
                "first_fact_seen_in_first_role": float((first[e1] > 0).mean()),
                "second_fact_seen_in_second_role": float((second[e2] > 0).mean()),
                "both_seen_in_required_roles": float(((first[e1] > 0) & (second[e2] > 0)).mean()),
                "head_tail_pair_seen_in_training": float(seen_pairs[rows[:, 0], rows[:, 3]].mean()),
            }
        results.append(
            {
                "phi": phi,
                "support": support,
                "composition_fraction": support / (len(atoms) + support),
                "id_first_role_coverage": float((first[~world["ood"]] > 0).mean()),
                "id_second_role_coverage": float((second[~world["ood"]] > 0).mean()),
                "ood_first_role_count": int(first[world["ood"]].sum()),
                "ood_second_role_count": int(second[world["ood"]].sum()),
                "panels": tasks,
            }
        )
        assert results[-1]["ood_first_role_count"] == results[-1]["ood_second_role_count"] == 0
    write_json(ARTIFACTS / "support-role-audit.json", {"supports": results})
    return results


class ReproductionGPT(nn.Module):
    """Native HF initialization and modules, with equivalent supervised readout."""

    def __init__(self, config, repeats=1):
        super().__init__()
        self.config, self.repeats = config, repeats
        self.transformer = GPT2LMHeadModel(config).transformer

    def forward(self, tokens, positions=None):
        cfg, t = self.config, self.transformer
        length = tokens.shape[1]
        x = t.wte(tokens) + t.wpe(torch.arange(length, device=tokens.device))
        x = F.dropout(x, cfg.embd_pdrop, self.training)
        for _ in range(self.repeats):
            for block in t.h:
                z = block.ln_1(x)
                q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
                shape = (len(tokens), length, cfg.n_head, cfg.n_embd // cfg.n_head)
                q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
                )
                y = block.attn.c_proj(y.transpose(1, 2).reshape(len(tokens), length, cfg.n_embd))
                x = x + F.dropout(y, cfg.resid_pdrop, self.training)
                x = x + block.mlp(block.ln_2(x))
        x = t.ln_f(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, t.wte.weight)


def construct(spec, metadata, device):
    torch.manual_seed(spec["initialization"])
    cfg = GPT2Config.from_pretrained(ROOT / "reference/gpt2", local_files_only=True)
    cfg.vocab_size = metadata["vocab_size"]
    cfg.n_layer = spec["unique_layers"]
    cfg.use_cache = False
    return ReproductionGPT(cfg, spec["repeats"]).to(device)


def optimizer_for(model, lr, decay):
    params = list(model.named_parameters())
    return torch.optim.AdamW(
        [
            {
                "params": [p for n, p in params if not any(s in n for s in ["bias", "ln"])],
                "weight_decay": decay,
            },
            {
                "params": [p for n, p in params if any(s in n for s in ["bias", "ln"])],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
        capturable=True,
    )


class ReproductionStep(GraphStep):
    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        x, positions, labels = (part[self.index] for part in self.table)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(x, positions)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


class MergedBatchStream(EpochStream):
    """Match DataLoader's final partial batch using loss-masked graph padding."""

    def batch(self, batch_size):
        if not len(self.remaining):
            self.remaining = self.rng.permutation(self.size)
        count = min(batch_size, len(self.remaining))
        result = np.full(batch_size, self.size, dtype=np.int64)
        result[:count] = self.remaining[:count]
        self.remaining = self.remaining[count:]
        return result, count


@torch.no_grad()
def evaluate_rows(model, rows, metadata, device, batch_size=256):
    model.eval()
    packed = encode_rows(rows, metadata)
    answers, stops, losses = [], [], []
    for start in range(0, len(rows), batch_size):
        x, pos, labels = [
            torch.as_tensor(p[start : start + batch_size], device=device) for p in packed
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, pos)
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
            answer = logits[:, 0].argmax(-1)
            x[torch.arange(len(x), device=device), pos[:, 1]] = answer
            stop = model(x, pos)[:, 1].argmax(-1)
        answers.append(answer.cpu().numpy())
        stops.append(stop.cpu().numpy())
        losses.append(nll.view(-1, 2).float().cpu().numpy())
    model.train()
    answer, stop, nll = np.concatenate(answers), np.concatenate(stops), np.concatenate(losses)
    correct = answer == rows[:, -1] + metadata["entity_offset"]
    return {
        "n": len(rows),
        "answer_accuracy": float(correct.mean()),
        "accuracy": float((correct & (stop == metadata["end_marker"])).mean()),
        "answer_nll": float(nll[:, 0].mean()),
        "nll": float(nll.mean()),
    }, {"answer": answer, "stop": stop, "nll": nll, "rows": rows}


def learning_rate(spec, step):
    # HF's warmup schedule starts the first optimizer update at zero LR.
    return spec["lr"] * min((step - 1) / spec["warmup"], 1.0)


def save_checkpoint(path, model, optimizer, stream, counts, step, spec, training_seconds):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "step": step,
            "spec": spec,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "training_seconds": training_seconds,
        },
        tmp,
    )
    tmp.replace(path)


def train_run(config, spec, gpu):
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(f"cuda:{gpu}")
    out = ROOT / "development" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    metadata = json.loads((ROOT / "data/complete.json").read_text())
    assert digest(ROOT / "data/world.npz") == metadata["world_sha256"]
    assert digest(ROOT / "data/panels.npz") == metadata["panels_sha256"]
    world = dict(np.load(ROOT / "data/world.npz"))
    support = round(spec["phi"] * metadata["atomic_id"])
    selected = world["train_order"][:support]
    assert len(selected) == support
    atomic = encode_rows(world["atoms"], metadata)
    composite = encode_rows(world["chains"][selected], metadata)
    padding = (
        np.full((1, 4), metadata["pad_token"], dtype=np.int64),
        np.array([[2, 3]], dtype=np.int64),
        np.array([[-100, -100]], dtype=np.int64),
    )
    table = tuple(
        torch.as_tensor(np.concatenate([a, b, p]), device=device)
        for a, b, p in zip(atomic, composite, padding, strict=True)
    )
    del atomic, composite
    model = construct(spec, metadata, device)
    initial_hash = model_digest(model)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    counts = np.zeros(len(table[0]) - 1, dtype=np.int64)
    step, training_seconds = 0, 0.0
    latest = out / "latest.pt"
    rng = None
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        assert saved["spec"] == spec
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        # load_state_dict can replace group LR objects; refill the object the optimizer uses.
        lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        stream.load_state_dict(saved["stream"])
        counts, step = saved["counts"], saved["step"]
        training_seconds = saved["training_seconds"]
        rng = (saved["cpu_rng"].cpu(), saved["cuda_rng"].cpu())
        del saved
    if rng is not None:
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
    model.train()
    graph = ReproductionStep(model, optimizer, table, spec["batch_size"])
    panels = dict(np.load(ROOT / "data/panels.npz"))
    learning_path = out / "learning.json"
    history = json.loads(learning_path.read_text()) if learning_path.exists() else []
    history = [r for r in history if r["step"] <= step]
    started = time.perf_counter()
    parameters = sum(p.numel() for p in model.parameters())
    flop_step = matmul_flops(
        ModelConfig(
            metadata["vocab_size"],
            model.config.n_embd,
            model.config.n_layer * model.repeats,
            model.config.n_head,
            1024,
        ),
        spec["batch_size"],
        sequence=4,
        output_positions=2,
    )
    write_json(
        out / "run.json",
        {
            "spec": spec,
            "world_sha256": metadata["world_sha256"],
            "initial_model_sha256": initial_hash,
            "parameters": parameters,
            "atomic_examples": len(world["atoms"]),
            "composition_examples": support,
            "vocab_size": metadata["vocab_size"],
            "gpu": gpu,
            "gpu_name": torch.cuda.get_device_name(gpu),
            "started_utc": utc(),
            "dtype": "bf16 AMP with float32 parameters/Adam state",
            "pid": os.getpid(),
            "resumed_step": step,
        },
    )

    def measure():
        metrics = {}
        for name, panel in panels.items():
            metrics[name], pred = evaluate_rows(model, panel, metadata, device)
            np.savez_compressed(out / f"predictions-{step:07d}-{name}.npz", **pred)
        record = {
            "step": step,
            "metrics": metrics,
            "training_seconds": training_seconds,
            "wall_seconds": time.perf_counter() - started,
            "created_utc": utc(),
            "examples": int(counts.sum()),
            "supervised_tokens": int(counts.sum()) * 2,
            "executed_input_tokens": step * spec["batch_size"] * 4,
            "estimated_matmul_training_flops": step * flop_step,
            "composition_epochs": float(counts[len(world["atoms"]) :].sum() / support),
            "atomic_epochs": float(counts[: len(world["atoms"])].sum() / len(world["atoms"])),
        }
        history.append(record)
        write_json(learning_path, history)
        write_json(
            out / "status.json",
            {
                "state": "running",
                "step": step,
                "budget": spec["steps"],
                "training_seconds": training_seconds,
                "train_ii": metrics["train_ii"]["accuracy"],
                "test_ii": metrics["test_ii"]["accuracy"],
                "test_oo": metrics["test_oo"]["accuracy"],
                "updated_utc": utc(),
            },
        )
        print(
            json.dumps(
                {
                    "name": spec["name"],
                    "step": step,
                    "train": metrics["train_ii"]["accuracy"],
                    "id": metrics["test_ii"]["accuracy"],
                    "oo": metrics["test_oo"]["accuracy"],
                }
            ),
            flush=True,
        )

    def checkpoint():
        save_checkpoint(latest, model, optimizer, stream, counts, step, spec, training_seconds)
        retained = out / f"checkpoint-{step:07d}.pt"
        torch.save({"model": model.state_dict(), "spec": spec, "step": step}, retained)
        np.savez_compressed(out / f"exposure-{step:07d}.npz", counts=counts, selected=selected)

    if not history or history[-1]["step"] != step:
        measure()
    if step == 0 and 0 in config["checkpoint_nodes"]:
        checkpoint()
    for end in [n for n in config["evaluation_nodes"] if n > step]:
        torch.cuda.synchronize()
        segment = time.perf_counter()
        while step < end:
            n = min(128, end - step)
            indices = np.stack([stream.batch(spec["batch_size"])[0] for _ in range(n)])
            counts += np.bincount(indices.ravel(), minlength=len(counts) + 1)[: len(counts)]
            indices = torch.as_tensor(indices, device=device)
            for i in range(n):
                lr.fill_(learning_rate(spec, step + i + 1))
                loss = graph(indices[i])
            step += n
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - segment
        loss_value = float(loss)
        if not math.isfinite(loss_value) or not math.isfinite(float(graph.grad_norm)):
            raise FloatingPointError(f"Nonfinite loss/gradient at {step}")
        measure()
        if step in config["checkpoint_nodes"]:
            checkpoint()
    assert step == spec["steps"]
    epoch_steps = math.ceil(len(counts) / spec["batch_size"])
    full_epochs, remainder = divmod(step, epoch_steps)
    expected_examples = full_epochs * len(counts) + remainder * spec["batch_size"]
    assert counts.sum() == expected_examples
    assert counts.max() - counts.min() <= 1
    # Full endpoint evaluation complements the original fixed 3k training panels.
    full = {}
    for name, data in {
        "atomic_all": world["atoms"],
        "train_ii_all": world["chains"][selected],
    }.items():
        full[name], pred = evaluate_rows(model, data, metadata, device)
        np.savez_compressed(out / f"endpoint-{name}.npz", **pred)
    final = {
        "state": "complete",
        "step": step,
        "full_endpoint": full,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "wall_seconds": time.perf_counter() - started,
        "estimated_matmul_training_flops": step * flop_step,
        "completed_utc": utc(),
        "final_checkpoint_sha256": digest(out / f"checkpoint-{step:07d}.pt"),
        "model_sha256": model_digest(model),
    }
    write_json(out / "complete.json", final)
    write_json(out / "status.json", final)
