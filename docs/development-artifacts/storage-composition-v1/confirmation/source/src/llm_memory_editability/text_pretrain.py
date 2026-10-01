"""Paired full-token text pretraining with controlled cross-document access.

P0/P1 have identical token arrays and labels; only document attention changes.
P2 replaces an isolated neutral record by a derived fact, an explicit change in
supervision. This is controlled template-language pretraining, not a pretrained
natural-language model or a replication of the original bioS scale.
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
from .grok_depth import EpochStream, GraphStep, make_optimizer, write_json
from .grok_loop_model import LoopGPT, flops

WORDS = [
    "<pad>",
    "<eos>",
    "<bos>",
    "'s",
    "is",
    ".",
    "This",
    "record",
    "has",
    "no",
    "additional",
    "fact",
    "here",
    "mentor",
    "advisor",
    "coach",
    "supervisor",
    "city",
    "district",
    "region",
    "campus",
]
LENGTH = 26
PAD, EOS, BOS = 0, 1, 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vocabulary(spec):
    return (
        WORDS
        + [f"Person{i:03d}" for i in range(spec["heads_n"])]
        + [f"Expert{i:03d}" for i in range(spec["bridges_n"])]
        + [f"Place{i:03d}" for i in range(spec["tails_n"])]
    )


def atomic_sentence(row):
    h, r, t = map(int, row)
    return [BOS, h, 3, r, 4, t, 5, EOS]


def composite_sentence(row):
    h, r1, _b, r2, t = map(int, row)
    return [BOS, h, 3, r1, 3, r2, 4, t, 5, EOS]


def pack_documents(documents, groups):
    """Reset positions per record in every arm; BOS is an exogenous boundary.

    All remaining next-token targets, including names, relations and EOS, are
    supervised. The next record's BOS and padding are never prediction targets.
    """
    seq, positions, segments = [], [], []
    for doc, group in zip(documents, groups, strict=True):
        seq.extend(doc)
        positions.extend(range(len(doc)))
        segments.extend([group] * len(doc))
    if len(seq) > LENGTH:
        raise ValueError("Record exceeds fixed context")
    n = len(seq)
    seq += [PAD] * (LENGTH - n)
    positions += [0] * (LENGTH - n)
    segments += list(range(100, 100 + LENGTH - n))
    labels = np.asarray(seq[1:], dtype=np.int64)
    labels[(labels == BOS) | (labels == PAD)] = -100
    seg = np.asarray(segments[:-1])
    mask = (seg[:, None] == seg[None, :]) & np.tri(LENGTH - 1, dtype=bool)
    return (
        np.asarray(seq[:-1], dtype=np.int64),
        np.asarray(positions[:-1], dtype=np.int64),
        mask[None],
        labels,
    )


def build_world(spec):
    """One supplied world; complete head-tail pairs held out before rendering."""
    rng = np.random.default_rng(spec["world"])
    hn, bn, tn = spec["heads_n"], spec["bridges_n"], spec["tails_n"]
    hs = np.arange(len(WORDS), len(WORDS) + hn)
    bs = np.arange(hs[-1] + 1, hs[-1] + 1 + bn)
    ts = np.arange(bs[-1] + 1, bs[-1] + 1 + tn)
    first = np.array([(h, r, rng.choice(bs)) for h in hs for r in range(13, 17)])
    second = np.array([(b, r, rng.choice(ts)) for b in bs for r in range(17, 21)])
    lookup = {(int(h), int(r)): int(t) for h, r, t in second}
    paths = np.array(
        [(h, r1, b, r2, lookup[int(b), r2]) for h, r1, b in first for r2 in range(17, 21)],
        dtype=np.int64,
    )
    reserved = rng.random((hn, tn)) < spec["holdout_fraction"]
    held = reserved[paths[:, 0] - hs[0], paths[:, 4] - ts[0]]
    background = paths[:, 0] < hs[0] + hn // 2
    groups = {
        "atomic_first": first,
        "atomic_second": second,
        "background_train": paths[background & ~held],
        "background_test": paths[background & held],
        "target_train": paths[~background & ~held],
        "target_test": paths[~background & held],
    }
    for key, rows in groups.items():
        if not len(rows):
            raise ValueError(f"Empty world stratum {key}; do not silently reroll")
    train_pairs = {tuple(r[[0, 4]]) for r in paths[~held]}
    test_pairs = {tuple(r[[0, 4]]) for r in paths[held]}
    assert not train_pairs & test_pairs
    # Typed atomics can never expose a head and a tail together.
    assert not set(first[:, 0]) & set(second[:, 0])
    assert not set(first[:, 2]) & set(second[:, 2])
    groups["atomic"] = np.concatenate([first, second])
    groups["all_paths"] = paths
    return groups


def training_table(world, arm):
    if arm not in ("P0", "P1", "P2"):
        raise ValueError(arm)
    packed = [pack_documents([atomic_sentence(r)], [0]) for r in world["atomic"]]
    base_end = len(packed)
    packed.extend(pack_documents([composite_sentence(r)], [0]) for r in world["background_train"])
    background_end = len(packed)
    neutral = [BOS, 6, 7, 8, 9, 10, 11, 12, 5, EOS]
    for row in world["target_train"]:
        h, r1, b, r2, t = row
        third = composite_sentence(row) if arm == "P2" else neutral
        packed.append(
            pack_documents(
                [atomic_sentence([h, r1, b]), atomic_sentence([b, r2, t]), third],
                [0, 1, 2] if arm == "P0" else [0, 0, 2],
            )
        )
    table = tuple(np.stack([row[i] for row in packed]) for i in range(4))
    return table, [0, base_end, background_end, len(packed)]


class TextGPT(LoopGPT):
    def forward(self, tokens, position_ids=None, attention_mask=None, patch=None, cache=None):
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device)[None]
        p = self.dropout if self.training else 0.0
        x = F.dropout(self.token(tokens) + self.position(position_ids), p=p, training=self.training)
        for layer, block in enumerate(self.iter_blocks()):
            z = block.ln1(x)
            batch, length, width = z.shape
            a = block.attention
            q, k, v = a.qkv(z).view(batch, length, 3, a.heads, width // a.heads).unbind(2)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=p,
            )
            attn = F.dropout(
                a.proj(y.transpose(1, 2).reshape(batch, length, width)), p=p, training=self.training
            )
            mlp = F.dropout(block.mlp(block.ln2(x + attn)), p=p, training=self.training)
            if cache is not None:
                cache[layer] = {
                    "attention": attn.detach(),
                    "mlp": mlp.detach(),
                    "full": (x + attn + mlp).detach(),
                }
            if patch is not None and layer == patch["layer"]:
                pos, value, component = patch["position"], patch["value"], patch["component"]
                if component == "full":
                    x = x + attn + mlp
                    x = x.clone()
                    x[:, pos] = value
                    continue
                if component == "attention":
                    attn = attn.clone()
                    attn[:, pos] = value
                elif component == "mlp":
                    mlp = mlp.clone()
                    mlp[:, pos] = value
                else:
                    raise ValueError(component)
            x = x + attn + mlp
        return F.linear(self.ln_final(x), self.token.weight)


class TextGraphStep(GraphStep):
    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        x, positions, mask, labels = (t[self.index] for t in self.table)
        logits = self.model(x, position_ids=positions, attention_mask=mask)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


def construct(spec, device):
    torch.manual_seed(spec["initialization"])
    cfg = ModelConfig(
        vocab_size=len(vocabulary(spec)),
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=LENGTH,
    )
    return TextGPT(cfg, repeats=spec["repeats"], dropout=spec["dropout"]).to(device)


@torch.inference_mode()
def evaluate(model, rows, device, composite=False, patch=None, batch_size=256):
    model.eval()
    encode = composite_sentence if composite else atomic_sentence
    prompts = np.asarray([encode(r)[:-3] for r in rows], dtype=np.int64)
    answers, stops, probs = [], [], []
    for start in range(0, len(rows), batch_size):
        x = torch.as_tensor(prompts[start : start + batch_size], device=device)
        intervention = (
            None if patch is None else dict(patch, value=patch["value"][start : start + batch_size])
        )
        logits = model(x, patch=intervention)[:, -1]
        target = torch.as_tensor(rows[start : start + batch_size, -1], device=device)
        probs.extend(logits.softmax(-1).gather(1, target[:, None]).squeeze(1).tolist())
        prediction = logits.argmax(-1)
        answers.extend(prediction.tolist())
        x = torch.cat([x, prediction[:, None]], dim=1)
        period = model(x, patch=intervention)[:, -1].argmax(-1)
        x = torch.cat([x, period[:, None]], dim=1)
        eos = model(x, patch=intervention)[:, -1].argmax(-1)
        stops.extend(torch.stack([period, eos], dim=1).tolist())
    ans, stop, probability = np.asarray(answers), np.asarray(stops), np.asarray(probs)
    correct = ans == rows[:, -1]
    full = correct & (stop[:, 0] == 5) & (stop[:, 1] == EOS)
    return dict(
        n=len(rows),
        answer_accuracy=float(correct.mean()),
        accuracy=float(full.mean()),
        answer_probability=float(probability.mean()),
    ), dict(answer=ans, stops=stop, probability=probability, target=rows[:, -1])


def run(spec, output, device):
    output = Path(output)
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads(Path(spec["source_lock"]).read_text())
    for file, expected in lock["sources"].items():
        assert sha(file) == expected, file
    assert lock["config_sha256"] == sha(spec["config_path"])
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.08, device)
    world = build_world(spec)
    cpu_table, bounds = training_table(world, spec["arm"])
    np.savez_compressed(output / "world.npz", **world)
    np.savez_compressed(
        output / "training-table.npz",
        **{k: v for k, v in zip(("tokens", "positions", "mask", "labels"), cpu_table, strict=True)},
    )
    write_json(output / "spec.json", spec)
    write_json(output / "vocabulary.json", vocabulary(spec))
    model = construct(spec, device)
    opt = make_optimizer(model, torch.tensor(spec["lr"], device=device), spec["weight_decay"])
    table = tuple(torch.as_tensor(t, device=device) for t in cpu_table)
    streams = [EpochStream(bounds[i + 1] - bounds[i], spec["world"] + 100 + i) for i in range(3)]
    latest = output / "latest.pt"
    start, elapsed, histories = 0, 0.0, []
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        for stream, saved in zip(streams, state["streams"], strict=True):
            stream.load_state_dict(saved)
        start, elapsed, histories = state["step"], state["seconds"], state["histories"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    counts = spec["batch_parts"]
    supervised_per_step = sum(n * tokens for n, tokens in zip(counts, [7, 9, 23], strict=True))
    step_flops = flops(
        model.config, spec["repeats"], sum(counts), LENGTH - 1, output_positions=LENGTH - 1
    )

    def checkpoint(step, seconds):
        metrics, predictions = {}, {}
        for split in (
            "atomic",
            "background_train",
            "background_test",
            "target_train",
            "target_test",
        ):
            metrics[split], preds = evaluate(
                model, world[split], device, composite=split != "atomic"
            )
            predictions.update({split + "_" + k: v for k, v in preds.items()})
        atomic_ok = predictions["atomic_answer"] == world["atomic"][:, -1]
        knowledge = {
            (int(h), int(r)): bool(ok)
            for (h, r, _t), ok in zip(world["atomic"], atomic_ok, strict=True)
        }
        eligible = np.array(
            [
                knowledge[int(h), int(r1)] and knowledge[int(b), int(r2)]
                for h, r1, b, r2, _t in world["target_test"]
            ]
        )
        target_ok = predictions["target_test_answer"] == world["target_test"][:, -1]
        metrics["target_atomic_correct"] = dict(
            n=int(eligible.sum()),
            coverage=float(eligible.mean()),
            answer_accuracy=float(target_ok[eligible].mean()) if eligible.any() else None,
        )
        record = dict(
            step=step,
            seconds=seconds,
            supervised_tokens=step * supervised_per_step,
            estimated_flops=step * step_flops,
            metrics=metrics,
        )
        histories.append(record)
        np.savez_compressed(output / f"predictions-{step:06d}.npz", **predictions)
        write_json(output / "learning.json", histories)
        model.train()
        state = dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            step=step,
            seconds=seconds,
            histories=histories,
            spec=spec,
            streams=[s.state_dict() for s in streams],
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),
        )
        tmp = output / "latest.tmp"
        torch.save(state, tmp)
        tmp.replace(latest)
        if step in spec["weight_nodes"]:
            torch.save(
                dict(model=state["model"], step=step, spec=spec), output / f"weights-{step:06d}.pt"
            )
        print(json.dumps(dict(run=output.name, **record)), flush=True)

    if start == 0:
        checkpoint(0, elapsed)
    model.train()
    graph = TextGraphStep(model, opt, table, sum(counts))
    tick = time.monotonic()
    for step in range(start + 1, spec["steps"] + 1):
        idx = np.concatenate(
            [
                stream.take(n) + offset
                for stream, n, offset in zip(streams, counts, bounds[:-1], strict=True)
            ]
        )
        for group in opt.param_groups:
            group["lr"].fill_(spec["lr"] * min(1.0, step / spec["warmup"]))
        graph(torch.as_tensor(idx, device=device))
        if step in spec["nodes"]:
            torch.cuda.synchronize()
            elapsed += time.monotonic() - tick
            checkpoint(step, elapsed)
            tick = time.monotonic()
    result = dict(
        status="complete",
        final=histories[-1],
        parameters=sum(p.numel() for p in model.parameters()),
        source_lock_sha256=sha(spec["source_lock"]),
        world_sha256=sha(output / "world.npz"),
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),
    )
    write_json(output / "complete.json", result)
    return result
