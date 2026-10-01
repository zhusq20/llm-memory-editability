"""Matched relation-pair support and transfer to atomically trained facts.

Reuse the full-token text model; change composition support, not atomics,
per-fact role counts, unique chain count, or token marginals. Target heads and
bridges are disjoint from background; answer values are deliberately shared.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .grok_loop_model import flops
from .text_pretrain import (
    LENGTH,
    WORDS,
    TextGraphStep,
    atomic_sentence,
    composite_sentence,
    construct,
    evaluate,
    pack_documents,
    sha,
    vocabulary,
)

ARMS = ("restricted", "broad", "positive")
TESTS = ("id_test", "first_only", "second_only", "background_neither", "strict_test")
SPLITS = ("atomic", "train_composite", "positive_train", *TESTS)


def build_world(spec):
    """Balanced graph; common head-tail holdout defined before any model exists.

    Each background bridge receives four trained heads per first relation.
    Two of its four second facts have composition experience. Restricted
    support pairs r1[0:2] only with even r2 and r1[2:4] only with odd r2.
    Broad support assigns half the incoming facts of EACH r1 to each active r2.
    Thus the relation-pair support has two components versus one, while all
    underlying first/second edge counts match exactly.
    """
    rng = np.random.default_rng(spec["world"])
    hn, bn, tn = spec["heads_n"], spec["bridges_n"], spec["tails_n"]
    if hn != 128 or bn != 32 or tn != 32:
        raise ValueError("This version fixes 128 heads, 32 bridges, 32 shared answer values")
    hs = np.arange(len(WORDS), len(WORDS) + hn)
    bs = np.arange(hs[-1] + 1, hs[-1] + 1 + bn)
    ts = np.arange(bs[-1] + 1, bs[-1] + 1 + tn)
    first = []
    for heads, bridges in ((hs[:64], bs[:16]), (hs[64:96], bs[:16]), (hs[96:], bs[16:])):
        mappings = [rng.permutation(np.tile(bridges, len(heads) // len(bridges))) for _ in range(4)]
        first.extend((h, 13 + ri, mappings[ri][i]) for i, h in enumerate(heads) for ri in range(4))
    first = np.asarray(first, dtype=np.int64)
    active, inactive = [], []
    for bi, b in enumerate(bs[:16]):
        chosen = (17, 18) if bi % 2 == 0 else (19, 20)
        active.extend((b, r) for r in chosen)
        inactive.extend((b, r) for r in range(17, 21) if r not in chosen)
    second = []
    for edges in (active, inactive, [(b, r) for b in bs[16:] for r in range(17, 21)]):
        tails = rng.permutation(np.tile(ts, len(edges) // len(ts)))
        second.extend((*edge, t) for edge, t in zip(edges, tails, strict=True))
    second = np.asarray(sorted(second), dtype=np.int64)
    lookup = {(int(h), int(r)): int(t) for h, r, t in second}
    paths = np.asarray(
        [(h, r1, b, r2, lookup[int(b), r2]) for h, r1, b in first for r2 in range(17, 21)],
        dtype=np.int64,
    )
    trained_first = first[:256]
    active_set = set(active)
    broad_assignment = {}
    for b in bs[:16]:
        choices = sorted(r for bb, r in active if bb == b)
        for r1 in range(13, 17):
            incoming = rng.permutation(
                trained_first[(trained_first[:, 2] == b) & (trained_first[:, 1] == r1), 0]
            )
            if len(incoming) != 4:
                raise ValueError("Unbalanced graph")
            for i, h in enumerate(incoming):
                broad_assignment[int(h), r1] = choices[i // 2]
    trains = {}
    for arm in ARMS[:2]:
        rows = []
        for h, r1, b in trained_first:
            choices = sorted(r for bb, r in active if bb == b)
            r2 = (
                choices[int(r1 >= 15)] if arm == "restricted" else broad_assignment[int(h), int(r1)]
            )
            rows.append((h, r1, b, r2, lookup[int(b), r2]))
        trains[arm] = np.asarray(rows, dtype=np.int64)
    # Two supervised paths per target person; keep the remaining head-tail
    # pairs for an independent positive-control generalization evaluation.
    positive = []
    for h in hs[96:]:
        candidates = paths[paths[:, 0] == h]
        positive.extend(candidates[rng.choice(len(candidates), 2, replace=False)])
    positive = np.asarray(positive, dtype=np.int64)
    train_union = np.concatenate([trains["restricted"], trains["broad"], positive])
    pairs = {tuple(row[[0, 4]]) for row in train_union}
    held = np.asarray([tuple(row[[0, 4]]) not in pairs for row in paths])
    first_seen = paths[:, 0] < hs[64]
    second_seen = np.asarray([(int(b), int(r2)) in active_set for b, r2 in paths[:, [2, 3]]])
    strict = paths[:, 0] >= hs[96]
    world = dict(
        atomic=np.concatenate([first, second]),
        atomic_first=first,
        atomic_second=second,
        restricted_train=trains["restricted"],
        broad_train=trains["broad"],
        positive_train=positive,
        id_test=paths[held & first_seen & second_seen],
        first_only=paths[held & first_seen & ~second_seen],
        second_only=paths[held & ~first_seen & second_seen],
        background_neither=paths[held & ~first_seen & ~second_seen & ~strict],
        strict_test=paths[held & strict],
        strict_all=paths[strict],
        all_paths=paths,
        background_heads=hs[:96],
        background_bridges=bs[:16],
        target_heads=hs[96:],
        target_bridges=bs[16:],
        active_second=np.asarray(active, dtype=np.int64),
    )
    if any(not len(world[key]) for key in TESTS):
        raise ValueError("Empty evaluation stratum; do not reroll worlds")
    return world


def role_counts(rows):
    return (
        Counter(map(tuple, rows[:, :3])),
        Counter(map(tuple, rows[:, 2:])),
    )


def support_components(rows):
    adjacency = {}
    for _h, r1, _b, r2, _t in rows:
        a, b = (0, int(r1)), (1, int(r2))
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    remaining, components = set(adjacency), []
    while remaining:
        stack, found = [min(remaining)], set()
        while stack:
            node = stack.pop()
            if node not in found:
                found.add(node)
                stack.extend(adjacency[node] - found)
        remaining -= found
        components.append(sorted(found))
    return components


def data_audit(world):
    a, b = world["restricted_train"], world["broad_train"]
    assert len(a) == len(b) == 256
    assert len(set(map(tuple, a))) == len(set(map(tuple, b))) == 256
    assert role_counts(a) == role_counts(b)
    assert Counter(t for row in a for t in composite_sentence(row)) == Counter(
        t for row in b for t in composite_sentence(row)
    )
    assert len(support_components(a)) == 2 and len(support_components(b)) == 1
    lookup = {(int(h), int(r)): int(t) for h, r, t in world["atomic"]}
    for h, r1, bridge, r2, t in world["all_paths"]:
        assert lookup[int(h), int(r1)] == bridge and lookup[int(bridge), int(r2)] == t
    train_pairs = {tuple(r[[0, 4]]) for rows in (a, b, world["positive_train"]) for r in rows}
    assert all(not train_pairs & {tuple(r[[0, 4]]) for r in world[s]} for s in TESTS)
    assert not set(world["target_heads"]) & set(
        np.concatenate([a[:, 0], a[:, 2], b[:, 0], b[:, 2]])
    )
    assert not set(world["target_bridges"]) & set(world["background_bridges"])
    assert set(world["strict_all"][:, 2]) <= set(world["target_bridges"])
    first_counts, second_counts = role_counts(a)
    return dict(
        status="passed",
        sizes={k: len(v) for k, v in world.items()},
        unique_chains={arm: len(world[arm + "_train"]) for arm in ARMS[:2]},
        relation_pair_counts={
            arm: np.bincount(
                (world[arm + "_train"][:, 1] - 13) * 4 + world[arm + "_train"][:, 3] - 17,
                minlength=16,
            )
            .reshape(4, 4)
            .tolist()
            for arm in ARMS[:2]
        },
        role_counts_match=True,
        token_marginals_match=True,
        first_edge_usage=list(sorted(set(first_counts.values()))),
        second_edge_usage=list(sorted(set(second_counts.values()))),
        common_holdout_pairs=True,
        isolated_target_heads_and_bridges=True,
        shared_answer_values=True,
    )


def training_table(world, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    rows = world["restricted_train" if arm == "restricted" else "broad_train"]
    if arm == "positive":
        rows = np.concatenate([rows, world["positive_train"]])
    packed = [pack_documents([atomic_sentence(r)], [0]) for r in world["atomic"]]
    packed.extend(pack_documents([composite_sentence(r)], [0]) for r in rows)
    return (
        tuple(np.stack([row[i] for row in packed]) for i in range(4)),
        [0, len(world["atomic"]), len(packed)],
        rows,
    )


def score_predictions(predictions):
    correct = predictions["answer"] == predictions["target"]
    stops = predictions["stops"]
    full = correct & (stops[:, 0] == 5) & (stops[:, 1] == 1)
    return dict(
        n=len(correct),
        answer_accuracy=float(correct.mean()),
        accuracy=float(full.mean()),
        answer_probability=float(predictions["probability"].mean()),
    )


@torch.inference_mode()
def two_calls(model, rows, device):
    first = rows[:, :3]
    first_scores, first_pred = evaluate(model, first, device)
    second = np.c_[first_pred["answer"], rows[:, 3:]]
    _scores, pred = evaluate(model, second, device)
    first_stops = first_pred["stops"]
    format_ok = (first_stops[:, 0] == 5) & (first_stops[:, 1] == 1)
    second_full = (
        (pred["answer"] == rows[:, -1]) & (pred["stops"][:, 0] == 5) & (pred["stops"][:, 1] == 1)
    )
    first_ok = first_pred["answer"] == rows[:, 2]
    scores = score_predictions(pred)
    scores.update(
        accuracy=float((format_ok & second_full).mean()),
        path_accuracy=float((format_ok & first_ok & second_full).mean()),
        first_accuracy=first_scores["accuracy"],
    )
    pred.update(
        first_answer=first_pred["answer"],
        first_stops=first_stops,
        first_target=rows[:, 2],
        first_probability=first_pred["probability"],
    )
    return scores, pred


def evaluate_all(model, world, train_rows, device):
    metrics, predictions = {}, {}
    groups = dict(world, train_composite=train_rows)
    for split in SPLITS:
        scores, pred = evaluate(model, groups[split], device, composite=split != "atomic")
        metrics[split] = scores
        predictions.update({split + "_" + k: v for k, v in pred.items()})
    atomic_ok = predictions["atomic_answer"] == world["atomic"][:, -1]
    knowledge = {
        (int(h), int(r)): bool(ok)
        for (h, r, _t), ok in zip(world["atomic"], atomic_ok, strict=True)
    }
    for split in TESTS:
        rows = world[split]
        eligible = np.asarray(
            [knowledge[int(h), int(r1)] and knowledge[int(b), int(r2)] for h, r1, b, r2, _t in rows]
        )
        predictions[split + "_eligible"] = eligible
        conditional = {
            k: predictions[split + "_" + k][eligible]
            for k in ("answer", "target", "stops", "probability")
        }
        metrics[split + "_conditional"] = dict(
            n=int(eligible.sum()), coverage=float(eligible.mean())
        )
        if eligible.any():
            metrics[split + "_conditional"].update(score_predictions(conditional))
        else:
            metrics[split + "_conditional"].update(
                answer_accuracy=None, accuracy=None, answer_probability=None
            )
        scores, pred = two_calls(model, rows, device)
        metrics[split + "_two_calls"] = scores
        predictions.update({split + "_two_calls_" + k: v for k, v in pred.items()})
    return metrics, predictions


def verify_lock(config, source_lock):
    lock = json.loads(Path(source_lock).read_text())
    assert lock["config_sha256"] == sha(config), config
    for file, expected in lock["sources"].items():
        assert sha(file) == expected, file
    return lock


def run(spec, output, device):
    output = Path(output)
    verify_lock(spec["config_path"], spec["source_lock"])
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.06, device)
    world = build_world(spec)
    write_json(output / "data-audit.json", data_audit(world))
    cpu_table, bounds, train_rows = training_table(world, spec["arm"])
    np.savez_compressed(output / "world.npz", **world, train_composite=train_rows)
    np.savez_compressed(
        output / "training-table.npz",
        **{k: v for k, v in zip(("tokens", "positions", "mask", "labels"), cpu_table, strict=True)},
    )
    write_json(output / "spec.json", spec)
    write_json(output / "vocabulary.json", vocabulary(spec))
    model = construct(spec, device)
    initial_hash = {
        key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for key, value in model.state_dict().items()
    }
    write_json(output / "initialization-hashes.json", initial_hash)
    opt = make_optimizer(model, torch.tensor(spec["lr"], device=device), spec["weight_decay"])
    table = tuple(torch.as_tensor(t, device=device) for t in cpu_table)
    streams = [EpochStream(bounds[i + 1] - bounds[i], spec["world"] + 100 + i) for i in range(2)]
    latest = output / "latest.pt"
    start, elapsed, histories = 0, 0.0, []
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        assert state["spec"] == spec
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        for stream, saved in zip(streams, state["streams"], strict=True):
            stream.load_state_dict(saved)
        start, elapsed, histories = state["step"], state["seconds"], state["histories"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    counts = spec["batch_parts"]
    supervised_per_step = sum(n * tokens for n, tokens in zip(counts, [7, 9], strict=True))
    step_flops = flops(
        model.config, spec["repeats"], sum(counts), LENGTH - 1, output_positions=LENGTH - 1
    )
    started = utc()

    def checkpoint(step, seconds):
        metrics, predictions = evaluate_all(model, world, train_rows, device)
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
        print(
            json.dumps(
                dict(
                    run=output.name,
                    step=step,
                    seconds=seconds,
                    metrics={k: metrics[k] for k in ("atomic", "train_composite", *TESTS)},
                )
            ),
            flush=True,
        )

    if start == 0:
        checkpoint(0, elapsed)
    model.train()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    graph = TextGraphStep(model, opt, table, sum(counts))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
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
        started_utc=started,
        finished_utc=utc(),
        final=histories[-1],
        parameters=sum(p.numel() for p in model.parameters()),
        source_lock_sha256=sha(spec["source_lock"]),
        world_sha256=sha(output / "world.npz"),
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),
        gpu=torch.cuda.get_device_name(device),
        torch_version=torch.__version__,
    )
    write_json(output / "complete.json", result)
    return result
