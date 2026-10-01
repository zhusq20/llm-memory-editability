"""Fact-load experiments in unmodified GPT-2 blocks and depth-shared GPT-2 blocks.

Template-language data are controlled; the architecture is an ordinary causal LM.
No embeddings/projections are frozen, no intermediate answers are supplied, and
all non-padding next tokens contribute to the training loss.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_model import ModelConfig
from .grok_depth import EpochStream, GraphStep, SmallGPT, make_optimizer, utc, write_json
from .grok_loop_model import LoopGPT, flops
from .text_pretrain import WORDS, atomic_sentence, composite_sentence

SOURCE_FILES = [
    "src/llm_memory_editability/storage_composition.py",
    "src/llm_memory_editability/bios_model.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_loop_model.py",
    "src/llm_memory_editability/text_pretrain.py",
    "scripts/run_storage_composition.py",
    "tests/test_storage_composition.py",
]
ARCHITECTURES = {
    "standard1": (1, 1),
    "standard2": (2, 1),
    "standard3": (3, 1),
    "loop1x2": (1, 2),
    "loop1x3": (1, 3),
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_world(spec):
    """Keep the target subgraph, vocabulary and candidates fixed across loads.

    The low-load extra stream repeats background atomics. High load instead
    teaches distinct distractor atomics. Common atomic and composite streams
    have exactly the same sampled rows. Extra background practice at low load
    is disclosed: this is an interference experiment, not a capacity theorem.
    """
    rng = np.random.default_rng(spec["world"])
    hn, bn, tn = (spec[k] for k in ("heads_n", "bridges_n", "tails_n"))
    familiar, strict = spec["familiar_n"], spec["strict_n"]
    core = familiar + strict
    if not (0 < familiar < core <= min(hn, bn)):
        raise ValueError("Invalid disjoint familiar/strict subgraph sizes")
    heads = np.arange(len(WORDS), len(WORDS) + hn)
    bridges = np.arange(heads[-1] + 1, heads[-1] + 1 + bn)
    tails = np.arange(bridges[-1] + 1, bridges[-1] + 1 + tn)
    first = []
    for index, head in enumerate(heads):
        candidates = (
            bridges[:familiar]
            if index < familiar
            else bridges[familiar:core]
            if index < core
            else bridges
        )
        first.extend((head, r, rng.choice(candidates)) for r in range(13, 17))
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(
        [(b, r, rng.choice(tails)) for b in bridges for r in range(17, 21)], dtype=np.int64
    )
    lookup = {(int(h), int(r)): int(t) for h, r, t in second}
    paths = np.asarray(
        [
            (h, r1, b, r2, lookup[int(b), r2])
            for h, r1, b in first[: core * 4]
            for r2 in range(17, 21)
        ],
        dtype=np.int64,
    )
    familiar_paths = paths[paths[:, 0] < heads[familiar]]
    # Hold out entire (head, terminal) pairs, including alternate relation paths.
    reserved = rng.random((familiar, tn)) < spec["holdout_fraction"]
    held = reserved[familiar_paths[:, 0] - heads[0], familiar_paths[:, 4] - tails[0]]
    world = {
        "common_atomic": np.concatenate([first[: core * 4], second[: core * 4]]),
        "background_atomic": np.concatenate([first[: familiar * 4], second[: familiar * 4]]),
        "extra_atomic": np.concatenate([first[core * 4 :], second[core * 4 :]]),
        "train_composite": familiar_paths[~held],
        "familiar_test": familiar_paths[held],
        "strict_test": paths[paths[:, 0] >= heads[familiar]],
    }
    if any(not len(rows) for rows in world.values()):
        raise ValueError("Empty stratum: do not silently reroll a world")
    atoms = np.concatenate([world["common_atomic"], world["extra_atomic"]])
    audit_world(world)
    assert len(atoms) == 4 * (hn + bn)
    return world


def audit_world(world):
    atoms = np.concatenate([world["common_atomic"], world["extra_atomic"]])
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    assert len(lookup) == len(atoms)
    for name in ("train_composite", "familiar_test", "strict_test"):
        for h, r1, b, r2, t in world[name]:
            assert lookup[int(h), int(r1)] == b
            assert lookup[int(b), int(r2)] == t
    train_pairs = {tuple(r[[0, 4]]) for r in world["train_composite"]}
    assert not train_pairs & {tuple(r[[0, 4]]) for r in world["familiar_test"]}
    train_facts = {(int(h), int(r1), int(b)) for h, r1, b, _r2, _t in world["train_composite"]} | {
        (int(b), int(r2), int(t)) for _h, _r1, b, r2, t in world["train_composite"]
    }
    strict_facts = {(int(h), int(r1), int(b)) for h, r1, b, _r2, _t in world["strict_test"]} | {
        (int(b), int(r2), int(t)) for _h, _r1, b, r2, t in world["strict_test"]
    }
    assert not strict_facts & train_facts
    assert not strict_facts & set(map(tuple, world["extra_atomic"]))


def pack_sentences(rows):
    render = atomic_sentence if rows.shape[1] == 3 else composite_sentence
    x = np.zeros((len(rows), 9), dtype=np.int64)
    labels = np.full_like(x, -100)
    for index, row in enumerate(rows):
        sentence = render(row)
        x[index, : len(sentence) - 1] = sentence[:-1]
        labels[index, : len(sentence) - 1] = sentence[1:]
    return x, labels


class FullTokenStep(GraphStep):
    def eager(self):
        self.optimizer.zero_grad(set_to_none=True)
        tokens, labels = (part[self.index] for part in self.table)
        logits = self.model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


def construct(spec, device):
    layers, repeats = ARCHITECTURES[spec["architecture"]]
    config = ModelConfig(
        vocab_size=len(WORDS) + spec["heads_n"] + spec["bridges_n"] + spec["tails_n"],
        width=spec["width"],
        layers=layers,
        heads=spec["heads"],
        context=16,
    )
    torch.manual_seed(spec["initialization"])
    if repeats == 1:
        model = SmallGPT(config, dropout=spec["dropout"])
    else:
        # Same unique-block initialization as standard1, without zeroing residuals.
        model = LoopGPT(config, repeats, spec["dropout"], initialization="legacy_unique")
    return model.to(device)


@torch.no_grad()
def generate_rows(model, rows, device, repeats=None, batch_size=512):
    """Greedy answer, then '.', then EOS; feed back every generated token."""
    model.eval()
    if rows.shape[1] == 3:
        prompts = np.asarray([atomic_sentence(r)[:5] for r in rows])
    else:
        prompts = np.asarray([composite_sentence(r)[:7] for r in rows])
    predictions, nlls = [], []
    kwargs = {"repeats": repeats} if isinstance(model, LoopGPT) else {}
    for start in range(0, len(rows), batch_size):
        tokens = torch.as_tensor(prompts[start : start + batch_size], device=device)
        target = torch.as_tensor(rows[start : start + batch_size, -1], device=device)
        generated = []
        for i in range(3):
            logits = model(tokens, **kwargs)[:, -1]
            if i == 0:
                nlls.append(F.cross_entropy(logits, target, reduction="none").cpu().numpy())
            next_token = logits.argmax(-1)
            generated.append(next_token.cpu().numpy())
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        predictions.append(np.stack(generated, axis=1))
    pred, nll = np.concatenate(predictions), np.concatenate(nlls)
    answer_ok = pred[:, 0] == rows[:, -1]
    full_ok = answer_ok & (pred[:, 1] == 5) & (pred[:, 2] == 1)
    metrics = {
        "n": len(rows),
        "answer_accuracy": float(answer_ok.mean()),
        "accuracy": float(full_ok.mean()),
        "answer_nll": float(nll.mean()),
    }
    return metrics, {"generated": pred, "correct": full_ok, "answer_nll": nll}


def evaluate(model, world, load, device, repeats=None):
    metrics, predictions = {}, {}
    names = ["common_atomic", "train_composite", "familiar_test", "strict_test"]
    if load == "high":
        names.append("extra_atomic")
    for name in names:
        metrics[name], pred = generate_rows(model, world[name], device, repeats)
        predictions.update({name + "_" + k: v for k, v in pred.items()})
    for name in ("familiar_test", "strict_test"):
        rows = world[name]
        first = rows[:, [0, 1, 2]]
        second = rows[:, [2, 3, 4]]
        _m, p1 = generate_rows(model, first, device, repeats)
        _m, p2 = generate_rows(model, second, device, repeats)
        covered = p1["correct"] & p2["correct"]
        generated_second = second.copy()
        generated_second[:, 0] = p1["generated"][:, 0]
        _m, autonomous = generate_rows(model, generated_second, device, repeats)
        calls_ok = (
            (p1["generated"][:, 1] == 5) & (p1["generated"][:, 2] == 1) & autonomous["correct"]
        )
        direct = predictions[name + "_correct"]
        metrics[name].update(
            {
                "atomic_correct_coverage": float(covered.mean()),
                "conditional_accuracy": float(direct[covered].mean()) if covered.any() else None,
                "autonomous_two_calls": float(calls_ok.mean()),
            }
        )
        predictions.update({name + "_coverage": covered, name + "_two_calls": calls_ok})
    model.train()
    return metrics, predictions


def data_digest(world):
    digest = hashlib.sha256()
    for key, value in sorted(world.items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run(spec, out, device="cuda:2"):
    out = Path(out)
    if (out / "complete.json").exists():
        raise FileExistsError(f"Completed run exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device)
    torch.cuda.set_device(device)
    world = build_world(spec)
    np.savez_compressed(out / "world.npz", **world)
    source = {path: file_hash(path) for path in SOURCE_FILES}
    write_json(
        out / "spec.json", {"spec": spec, "source": source, "data_sha256": data_digest(world)}
    )
    model = construct(spec, device)
    lr = torch.tensor(spec["lr"], device=device)
    opt = make_optimizer(model, lr, spec["weight_decay"])
    extra = world["extra_atomic"] if spec["load"] == "high" else world["background_atomic"]
    strata = [world["common_atomic"], world["train_composite"], extra]
    packed = [pack_sentences(rows) for rows in strata]
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device) for parts in zip(*packed, strict=True)
    )
    sizes = [len(rows) for rows in strata]
    offsets = np.cumsum([0, *sizes[:-1]])
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    batch = spec["batch_size"]
    if batch % 3:
        raise ValueError("Batch must divide into three equal exposure strata")
    graph = FullTokenStep(model, opt, table, batch)
    history, training_seconds = [], 0.0
    layers, repeats = ARCHITECTURES[spec["architecture"]]
    parameters = sum(p.numel() for p in model.parameters())
    flop_step = flops(model.config, repeats, batch, 9, output_positions=9)
    start = 0

    def measure(step, loss=None):
        metrics, predictions = evaluate(model, world, spec["load"], device)
        row = {
            "step": step,
            "metrics": metrics,
            "loss": loss,
            "training_seconds": training_seconds,
            "estimated_training_flops": step * flop_step,
        }
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        state = {
            "spec": spec,
            "step": step,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "streams": [s.state_dict() for s in streams],
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "counts": counts,
            "source": source,
            "data_sha256": data_digest(world),
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        write_json(
            out / "status.json",
            {
                "step": step,
                "budget": spec["steps"],
                "atomic": metrics["common_atomic"]["accuracy"],
                "familiar": metrics["familiar_test"]["accuracy"],
                "strict": metrics["strict_test"]["accuracy"],
            },
        )
        print(json.dumps({"run": str(out), "step": step, "metrics": metrics}), flush=True)
        return row

    measure(0)
    for end in spec["nodes"]:
        if end == 0:
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while start < end:
            n = min(128, end - start)
            indices = []
            for i, stream in enumerate(streams):
                drawn = stream.take(n * batch // 3).reshape(n, batch // 3)
                counts[i] += np.bincount(drawn.ravel(), minlength=sizes[i])
                indices.append(drawn + offsets[i])
            indices = torch.as_tensor(np.concatenate(indices, axis=1), device=device)
            for j in range(n):
                lr.fill_(spec["lr"] * min(1.0, (start + j + 1) / spec["warmup"]))
                last_loss = graph(indices[j])
            start += n
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - t0
        loss = float(last_loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at {end}")
        endpoint = measure(end, loss)
    assert start == spec["steps"]
    # Prespecified test-time repeat sweep; these weights never change.
    repeat_metrics = {}
    if isinstance(model, LoopGPT):
        for count in spec["test_repeats"]:
            metrics, predictions = evaluate(model, world, spec["load"], device, count)
            repeat_metrics[str(count)] = metrics
            np.savez_compressed(out / f"repeat-{count:02d}.npz", **predictions)
    np.savez_compressed(out / "exposures.npz", **{f"stratum{i}": c for i, c in enumerate(counts)})
    write_json(
        out / "complete.json",
        {
            "spec": spec,
            "finished_utc": utc(),
            "parameters": parameters,
            "unique_blocks": layers,
            "executed_blocks": layers * repeats,
            "independent_facts": len(world["common_atomic"])
            + (len(extra) if spec["load"] == "high" else 0),
            "training_seconds": training_seconds,
            "endpoint": endpoint,
            "repeat_metrics": repeat_metrics,
            "examples": start * batch,
            "effective_input_tokens": start * batch * 23 // 3,
            "supervised_tokens": start * batch * 23 // 3,
            "padded_input_tokens": start * batch * 9,
            "source": source,
            "data_sha256": data_digest(world),
            "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(device)},
        },
    )
    return endpoint


def audit_run(out, device="cuda:2"):
    """Independently reload the final checkpoint and regenerate all endpoint tokens."""
    out = Path(out)
    complete = json.loads((out / "complete.json").read_text())
    saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    assert saved["step"] == complete["spec"]["steps"]
    world = build_world(saved["spec"])
    assert data_digest(world) == saved["data_sha256"] == complete["data_sha256"]
    model = construct(saved["spec"], device)
    model.load_state_dict(saved["model"])
    metrics, predictions = evaluate(model, world, saved["spec"]["load"], device)
    expected = np.load(out / f"predictions-{saved['step']:06d}.npz")
    for key, value in predictions.items():
        # Tiny batching/platform differences can affect NLL, never generated tokens.
        np.testing.assert_allclose(value, expected[key], rtol=1e-5, atol=1e-5)
    assert metrics == complete["endpoint"]["metrics"]
    exposures = np.load(out / "exposures.npz")
    for i in range(3):
        np.testing.assert_array_equal(exposures[f"stratum{i}"], saved["counts"][i])
        assert exposures[f"stratum{i}"].sum() == saved["step"] * saved["spec"]["batch_size"] // 3
        assert np.ptp(exposures[f"stratum{i}"]) <= 1
    result = {
        "passed": True,
        "step": saved["step"],
        "prediction_arrays": len(predictions),
        "utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result
