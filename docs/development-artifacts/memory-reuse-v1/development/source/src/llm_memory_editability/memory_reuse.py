"""Learned two-hop retrieval followed by an independently fitted memory swap.

The pinned author implementation supplies the SwiGLU memory and GPT blocks.
Only attention Q/K are learned by the reader. Replacement facts are never used
by its optimizer, and all evaluation uses continuous intermediate states.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability import hebbian_interface as interface

ARMS = ("ce", "aligned")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def hash_state(model, *, reader_only=False):
    return interface.tensor_hash(
        {k: v for k, v in model.state_dict().items() if not reader_only or ".mlp." not in k}
    )


def build_world(spec, seed):
    """Geometry, content and held-out heads use independent local random streams."""
    n, d, j = spec["num_entities"], spec["d_model"], spec["junk_vocab_size"]
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n + j + 1, d)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
    mapping_a = np.arange(n + j + 1, dtype=np.int64)
    mapping_b = mapping_a.copy()
    mapping_a[:n] = rng.permutation(n)
    for _ in range(10000):
        candidate = rng.permutation(n)
        if np.all(candidate != mapping_a[:n]):
            mapping_b[:n] = candidate
            break
    else:
        raise RuntimeError("Failed to draw a changed independent permutation")
    order = np.random.default_rng(seed + 20000).permutation(n)
    n_train = int(n * spec["train_head_fraction"])
    if not 0 < n_train < n:
        raise ValueError("Both composition-training and held-out heads are required")
    train_mask = np.zeros(n, dtype=bool)
    train_mask[order[:n_train]] = True
    return {
        "embeddings": embeddings,
        "mapping_A": mapping_a,
        "mapping_B": mapping_b,
        "train_mask": train_mask,
        "seed": np.asarray(seed),
    }


def data_audit(spec, world):
    n, j = spec["num_entities"], spec["junk_vocab_size"]
    for name in ("A", "B"):
        mapping = world[f"mapping_{name}"]
        assert np.array_equal(np.sort(mapping[:n]), np.arange(n))
        assert np.array_equal(mapping[n:], np.arange(n, n + j + 1))
    assert np.all(world["mapping_A"][:n] != world["mapping_B"][:n])
    np.testing.assert_allclose(np.linalg.norm(world["embeddings"], axis=-1), 1, atol=2e-7)
    assert world["train_mask"].sum() == int(n * spec["train_head_fraction"])
    a, b = world["mapping_A"], world["mapping_B"]
    return {
        "entities": n,
        "neutral_tokens": j + 1,
        "A_composition_training_heads": int(world["train_mask"].sum()),
        "A_composition_heldout_heads": int((~world["train_mask"]).sum()),
        "changed_atomic_facts": int((a[:n] != b[:n]).sum()),
        "changed_composite_answers": int((a[a[:n]] != b[b[:n]]).sum()),
        "passed": True,
    }


def make_inputs(spec, repeats, seed):
    """Each head occurs once per repeat; contexts contain only neutral tokens."""
    n, j, length = spec["num_entities"], spec["junk_vocab_size"], spec["junk_length"]
    rng = np.random.default_rng(seed)
    keys = np.tile(np.arange(n, dtype=np.int64), repeats)
    inputs = rng.integers(n, n + j, (len(keys), length + 2), dtype=np.int64)
    positions = rng.integers(0, length + 1, len(keys))
    inputs[np.arange(len(keys)), positions] = keys
    inputs[:, -1] = n + j
    return inputs, keys


def new_memory(spec, init, device):
    cfg = {**interface.DEFAULT_SPEC, **spec}
    interface._source(cfg)
    torch.manual_seed(init)
    return interface._new_memory(cfg, device)


class TwoHopReader(torch.nn.Module):
    """Two causal attention blocks with one shared, frozen fact MLP."""

    def __init__(self, spec, embeddings, memory, init, device):
        super().__init__()
        interface._source(spec)
        from hebbian.transformer.model import GPT, GPTConfig

        torch.manual_seed(init + 100000)
        cfg = GPTConfig(
            block_size=spec["junk_length"] + 2,
            vocab_size=len(embeddings),
            n_layer=2,
            n_head=1,
            n_embd=spec["d_model"],
            dropout=0.0,
            bias=False,
            mlp_residual=False,
            attn_residual=False,
            use_rope=False,
            no_positional_encoding=True,
            mlp_norm_type="unit_rmsnorm",
            attn_norm_type="rmsnorm",
            lm_head_norm_type="unit_rmsnorm",
            tie_embeddings=True,
            freeze_value_dense_identity=True,
            use_identity_mlp=True,
        )
        self.gpt = GPT(cfg).to(device)
        with torch.no_grad():
            self.gpt.transformer.wte.weight.copy_(torch.as_tensor(embeddings, device=device))
        self.install(memory)
        self.gpt.requires_grad_(False)
        for block in self.gpt.transformer.h:
            block.attn.c_q.requires_grad_(True)
            block.attn.c_k.requires_grad_(True)
        self.scale = spec["readout_scale"]

    def install(self, memory):
        memory.requires_grad_(False)
        for block in self.gpt.transformer.h:
            block.mlp = memory

    def forward(self, inputs):
        x = self.gpt.transformer.wte(inputs)
        logits = []
        for block in self.gpt.transformer.h:
            x = block(x)
            z = self.gpt.transformer.ln_f(x[:, -1])
            logits.append(self.scale * self.gpt.lm_head(z))
        return torch.stack(logits, dim=1)


def memory_objective(output, embeddings, labels, arm, alignment_weight):
    ce = F.cross_entropy(output @ embeddings.T, labels)
    alignment = (output - embeddings[labels]).square().sum(-1).mean()
    return ce + (alignment_weight if arm == "aligned" else 0.0) * alignment


@torch.no_grad()
def clean_memory(memory, embeddings, mapping, n):
    output = memory(embeddings)
    cosine = F.normalize(output, dim=-1) @ embeddings.T
    pred = cosine.argmax(-1)
    return {
        "atomic_accuracy": float(pred[:n].eq(mapping[:n]).float().mean()),
        "neutral_accuracy": float(pred[n:].eq(mapping[n:]).float().mean()),
        "target_cosine": float(cosine[torch.arange(n, device=mapping.device), mapping[:n]].mean()),
        "output_norm": float(output[:n].norm(dim=-1).mean()),
        "vector_squared_error": float(
            (output[:n] - embeddings[mapping[:n]]).square().sum(-1).mean()
        ),
    }


def fit_memory(spec, world, mapping_name, arm, init, device, output_dir):
    """Only atomic labels appear here, including for the replacement memory."""
    path = output_dir / f"memory-{mapping_name}.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = new_memory(spec, init, device)
        model.load_state_dict(payload["state_dict"])
        return model.requires_grad_(False).eval(), payload["record"]
    memory = new_memory(spec, init, device)
    embeddings = torch.as_tensor(world["embeddings"], device=device)
    labels = torch.as_tensor(world[f"mapping_{mapping_name}"], device=device)
    optimizer = torch.optim.Adam(memory.parameters(), lr=spec["memory_lr"], foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, spec["memory_steps"], eta_min=spec["memory_min_lr"]
    )
    initial_hash = hash_state(memory)
    curve = []
    began = time.perf_counter()
    for step in range(spec["memory_steps"] + 1):
        if step in spec["memory_nodes"]:
            metrics = clean_memory(memory, embeddings, labels, spec["num_entities"])
            metrics.update(step=step, seconds=time.perf_counter() - began)
            curve.append(metrics)
            write_json(output_dir / f"memory-{mapping_name}-curve.json", curve)
            print(json.dumps(dict(stage=f"memory-{mapping_name}", arm=arm, **metrics)), flush=True)
        if step == spec["memory_steps"]:
            break
        memory.train()
        optimizer.zero_grad(set_to_none=True)
        loss = memory_objective(
            memory(embeddings), embeddings, labels, arm, spec["alignment_weight"]
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory.eval().requires_grad_(False)
    record = dict(
        initial_hash=initial_hash,
        final_hash=hash_state(memory),
        steps=spec["memory_steps"],
        seconds=time.perf_counter() - began,
        metrics=clean_memory(memory, embeddings, labels, spec["num_entities"]),
    )
    torch.save(dict(state_dict=cpu_state(memory), record=record), path)
    return memory, record


@torch.no_grad()
def evaluate(reader, inputs, device, batch_size):
    reader.eval()
    return np.concatenate(
        [
            reader(torch.as_tensor(inputs[start : start + batch_size], device=device)).cpu().numpy()
            for start in range(0, len(inputs), batch_size)
        ]
    )


def accuracy(pred, target, mask=None):
    correct = pred == target
    if mask is not None:
        correct = correct[mask]
    return float(correct.mean()) if len(correct) else None


def score_logits(logits, world, keys, mapping_name):
    """Full-pool scores and explicitly labelled prerequisite/changed subsets."""
    pred = logits.argmax(-1)
    mapping = world[f"mapping_{mapping_name}"]
    y1, y2 = mapping[keys], mapping[mapping[keys]]
    train = world["train_mask"][keys]
    a, b = world["mapping_A"], world["mapping_B"]
    changed = a[a[keys]] != b[b[keys]]
    n = len(world["train_mask"])
    # Inputs are arranged as complete n-head repeats; find each actual successor.
    successor_row = np.arange(len(keys)) // n * n + mapping[keys]
    assert np.array_equal(keys[successor_row], mapping[keys])
    known = (pred[:, 0] == y1) & (pred[successor_row, 0] == mapping[mapping[keys]])
    return {
        "n": len(keys),
        "one_hop": accuracy(pred[:, 0], y1),
        "two_hop": accuracy(pred[:, 1], y2),
        "two_hop_A_training_heads": accuracy(pred[:, 1], y2, train),
        "two_hop_A_heldout_heads": accuracy(pred[:, 1], y2, ~train),
        "two_hop_changed": accuracy(pred[:, 1], y2, changed),
        "changed_n": int(changed.sum()),
        "two_hop_both_atoms_correct": accuracy(pred[:, 1], y2, known),
        "both_atoms_correct_n": int(known.sum()),
        "both_atoms_correct_coverage": float(known.mean()),
    }


def reader_loss(logits, targets, train_mask):
    return (
        F.cross_entropy(logits[:, 0], targets[:, 0])
        + F.cross_entropy(logits[train_mask, 1], targets[train_mask, 1])
    ) / 2


def fit_reader(spec, world, memory, init, device, output_dir):
    reader = TwoHopReader(spec, world["embeddings"], memory, init, device)
    path = output_dir / "reader-A.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        reader.load_state_dict(payload["state_dict"])
        return reader.eval(), payload["record"]
    initial_hash = hash_state(reader, reader_only=True)
    frozen_before = interface.tensor_hash(
        {k: p for k, p in reader.named_parameters() if not p.requires_grad}
    )
    n = spec["num_entities"]
    seed = int(world["seed"])
    train_inputs, _ = make_inputs(spec, spec["reader_steps"], seed + 30000)
    stream_hash = hashlib.sha256(train_inputs.tobytes()).hexdigest()
    train_inputs = torch.as_tensor(
        train_inputs.reshape(-1, n, spec["junk_length"] + 2), device=device
    )
    mapping = torch.as_tensor(world["mapping_A"], device=device)
    targets = torch.stack((mapping[:n], mapping[mapping[:n]]), dim=1)
    train_mask = torch.as_tensor(world["train_mask"], device=device)
    eval_inputs, eval_keys = make_inputs(spec, spec["eval_repeats"], seed + 40000)
    parameters = [p for p in reader.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=spec["reader_lr"], weight_decay=spec["reader_weight_decay"], foreach=False
    )
    curve = []
    began = time.perf_counter()
    for step in range(spec["reader_steps"] + 1):
        if step in spec["reader_nodes"]:
            logits = evaluate(reader, eval_inputs, device, n)
            metrics = score_logits(logits, world, eval_keys, "A")
            curve.append(dict(step=step, seconds=time.perf_counter() - began, **metrics))
            np.savez_compressed(output_dir / f"reader-node-{step:05d}.npz", logits=logits)
            torch.save(cpu_state(reader), output_dir / f"reader-node-{step:05d}.pt")
            write_json(output_dir / "reader-curve.json", curve)
            print(json.dumps(dict(stage="reader", step=step, **metrics)), flush=True)
        if step == spec["reader_steps"]:
            break
        reader.train()
        optimizer.zero_grad(set_to_none=True)
        loss = reader_loss(reader(train_inputs[step]), targets, train_mask)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    frozen_after = interface.tensor_hash(
        {k: p for k, p in reader.named_parameters() if not p.requires_grad}
    )
    assert frozen_before == frozen_after, "Frozen facts or embeddings changed"
    record = dict(
        initial_hash=initial_hash,
        final_hash=hash_state(reader, reader_only=True),
        frozen_hash=frozen_after,
        stream_hash=stream_hash,
        steps=spec["reader_steps"],
        seconds=time.perf_counter() - began,
        trainable_names=[k for k, p in reader.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in parameters),
        total_parameters=sum(p.numel() for p in reader.parameters()),
        metrics=curve[-1],
    )
    torch.save(dict(state_dict=cpu_state(reader), record=record), path)
    return reader.eval(), record


def run(spec, world_seed, init, arm, output_dir, device):
    if arm not in ARMS:
        raise ValueError(arm)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.set_num_threads(spec["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    world = build_world(spec, world_seed)
    np.savez_compressed(output_dir / "world.npz", **world)
    write_json(output_dir / "spec.json", spec)
    audit = data_audit(spec, world)
    started = time.perf_counter()
    result = dict(
        world=world_seed, init=init, arm=arm, spec=spec, data_audit=audit, status="running"
    )
    write_json(output_dir / "status.json", result)
    memory_a, a_record = fit_memory(spec, world, "A", arm, init, device, output_dir)
    reader, reader_record = fit_reader(spec, world, memory_a, init, device, output_dir)
    # Replacement fitting happens only after the A reader has finished.
    memory_b, b_record = fit_memory(spec, world, "B", arm, init, device, output_dir)
    inputs, keys = make_inputs(spec, spec["eval_repeats"], world_seed + 40000)
    logits_a = evaluate(reader, inputs, device, spec["num_entities"])
    non_memory_hash = hash_state(reader, reader_only=True)
    reader.requires_grad_(False)
    reader.install(memory_b)
    logits_b = evaluate(reader, inputs, device, spec["num_entities"])
    assert hash_state(reader, reader_only=True) == non_memory_hash
    assert reader.gpt.transformer.h[0].mlp is reader.gpt.transformer.h[1].mlp
    assert hash_state(memory_b) == b_record["final_hash"]
    embeddings = torch.as_tensor(world["embeddings"], device=device)
    with torch.no_grad():
        memory_arrays = {
            f"memory_{name}_output": mem(embeddings).cpu().numpy()
            for name, mem in (("A", memory_a), ("B", memory_b))
        }
    np.savez_compressed(
        output_dir / "endpoints.npz",
        inputs=inputs,
        keys=keys,
        logits_A=logits_a,
        logits_B=logits_b,
        **memory_arrays,
    )
    # Store a full swapped endpoint to permit a genuinely independent reload.
    torch.save(dict(state_dict=cpu_state(reader)), output_dir / "reader-B.pt")
    endpoints = {
        "A": score_logits(logits_a, world, keys, "A"),
        "B": score_logits(logits_b, world, keys, "B"),
        "wrong_A_on_B": score_logits(logits_a, world, keys, "B"),
    }
    length, d, h, vocab = (
        spec["junk_length"] + 2,
        spec["d_model"],
        spec["hidden_dim"],
        len(embeddings),
    )
    # Approximate multiply-add FLOPs. Backprop through frozen MLP still costs work.
    memory_flops = 2 * spec["memory_steps"] * (18 * vocab * d * h + 6 * vocab**2 * d)
    reader_flops = spec["reader_steps"] * (
        2
        * (
            24 * spec["num_entities"] * length * d**2
            + 12 * spec["num_entities"] * length**2 * d
            + 12 * spec["num_entities"] * length * d * h
        )
    )
    result.update(
        status="complete",
        memories=dict(A=a_record, B=b_record),
        reader=reader_record,
        endpoints=endpoints,
        assertions=dict(
            frozen_reader_after_swap=True, shared_memory=True, no_B_reader_training=True
        ),
        seconds=time.perf_counter() - started,
        budget=dict(
            memory_updates=2 * spec["memory_steps"],
            reader_updates=spec["reader_steps"],
            atomic_exposures=2 * spec["memory_steps"] * spec["num_entities"],
            reader_input_tokens=spec["reader_steps"] * spec["num_entities"] * length,
            reader_supervised_answers=spec["reader_steps"]
            * (spec["num_entities"] + int(world["train_mask"].sum())),
            approximate_matmul_flops=memory_flops + reader_flops,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        ),
    )
    write_json(output_dir / "summary.json", result)
    write_json(output_dir / "status.json", dict(status="complete", seconds=result["seconds"]))
    return result
