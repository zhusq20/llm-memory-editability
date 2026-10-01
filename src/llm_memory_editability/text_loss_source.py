"""Factorial loss-source interventions on an unchanged text-pretraining stream."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import EpochStream, GraphStep, make_optimizer, write_json
from .grok_loop_model import flops
from .text_pretrain import construct, evaluate, sha

ARMS = {"AB": (1, 1), "A": (1, 0), "B": (0, 1), "N": (0, 0)}
SPLITS = ("atomic", "background_train", "background_test", "target_train", "target_test")


def loss_sources(labels, bounds):
    """0=atomic text, 1=background composition, 2=neutral text, -1=ignored."""
    result = np.full(labels.shape, -1, dtype=np.int64)
    result[: bounds[1]] = 0
    result[bounds[1] : bounds[2]] = 1
    result[bounds[2] :, :15] = 0
    result[bounds[2] :, 16:] = 2
    result[labels == -100] = -1
    assert np.array_equal(result >= 0, labels != -100)
    return result


def weights_for(source, arm):
    a, b = ARMS[arm]
    return ((source == 0) * a + (source == 1) * b + (source == 2)).astype(np.float32)


def weighted_loss(logits, labels, weights):
    per_token = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none", ignore_index=-100
    ).view_as(labels)
    # Removing a source does NOT renormalize the remaining losses.
    return (per_token * weights).sum() / (labels != -100).sum()


class SourceGraphStep(GraphStep):
    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        x, positions, mask, labels, weights = (t[self.index] for t in self.table)
        logits = self.model(x, position_ids=positions, attention_mask=mask)
        loss = weighted_loss(logits, labels, weights)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


@torch.inference_mode()
def measure(model, world, table, source, device):
    scores, predictions = {}, {}
    for split in SPLITS:
        scores[split], pred = evaluate(model, world[split], device, composite=split != "atomic")
        predictions.update({split + "_" + k: v for k, v in pred.items()})
    atomic = predictions["atomic_answer"] == world["atomic"][:, -1]
    known = {(h, r): ok for (h, r, _), ok in zip(world["atomic"], atomic, strict=True)}
    eligible = np.array([known[h, r1] and known[b, r2] for h, r1, b, r2, t in world["target_test"]])
    full = (predictions["target_test_answer"] == world["target_test"][:, -1]) & (
        predictions["target_test_stops"] == [5, 1]
    ).all(-1)
    scores["both_atomics"] = dict(
        n=int(eligible.sum()),
        coverage=float(eligible.mean()),
        accuracy=float(full[eligible].mean()) if eligible.any() else None,
    )
    sums, counts = np.zeros(3), np.zeros(3, dtype=np.int64)
    for start in range(0, len(table[0]), 128):
        x, pos, mask, labels = [
            torch.as_tensor(t[start : start + 128], device=device) for t in table[:4]
        ]
        logits = model(x, position_ids=pos, attention_mask=mask)
        nll = (
            F.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), reduction="none", ignore_index=-100
            )
            .view_as(labels)
            .cpu()
            .numpy()
        )
        for group in range(3):
            selected = source[start : start + 128] == group
            sums[group] += nll[selected].sum(dtype=np.float64)
            counts[group] += selected.sum()
    scores["training_text_nll"] = {
        name: float(sums[i] / counts[i])
        for i, name in enumerate(("atomic", "background", "neutral"))
    }
    return scores, predictions


def run(cfg, name, device):
    overrides = cfg["runs"][name]
    root = Path(cfg["output_root"]) / name
    if (root / "complete.json").exists():
        return
    lock = json.loads(Path(cfg["lock"]).read_text())
    for p, digest in lock["sources"].items():
        assert sha(p) == digest, p
    assert sha(cfg["config_path"]) == lock["config_sha256"]
    assert sha(cfg["plan"]) == lock["plan_sha256"]
    for p, expected in lock["inputs"].items():
        assert sha(p) == expected, p
    parent = Path(overrides["parent"])
    assert sha(parent / "latest.pt") == lock["parents"][str(parent / "latest.pt")]
    root.mkdir(parents=True, exist_ok=True)
    state = torch.load(parent / "latest.pt", map_location=device, weights_only=False)
    spec = state["spec"]
    assert spec["arm"] == "P1" and state["step"] == 32000
    world = dict(np.load(parent / "world.npz"))
    stored = np.load(parent / "training-table.npz")
    table = tuple(stored[k] for k in ("tokens", "positions", "mask", "labels"))
    bounds = np.cumsum(
        [0, len(world["atomic"]), len(world["background_train"]), len(world["target_train"])]
    )
    source = loss_sources(table[3], bounds)
    weights = weights_for(source, overrides["arm"])
    gpu_table = tuple(torch.as_tensor(t, device=device) for t in (*table, weights))
    model = construct(spec, device)
    model.load_state_dict(state["model"])
    opt = make_optimizer(model, torch.tensor(spec["lr"], device=device), spec["weight_decay"])
    opt.load_state_dict(state["optimizer"])
    streams = [EpochStream(bounds[i + 1] - bounds[i], spec["world"] + 100 + i) for i in range(3)]
    for stream, saved in zip(streams, state["streams"], strict=True):
        stream.load_state_dict(saved)
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    settings = dict(
        parent=str(parent),
        world=spec["world"],
        initialization=spec["initialization"],
        arm=overrides["arm"],
        steps=cfg["steps"],
        parent_sha256=sha(parent / "latest.pt"),
        config=cfg["config_path"],
    )
    write_json(root / "settings.json", settings)
    start, elapsed, histories = 0, 0.0, []
    digest = hashlib.sha256()
    # Recovery is deliberately from this branch's saved stream, never its parent's.
    if (root / "latest.pt").exists():
        saved = torch.load(root / "latest.pt", map_location=device, weights_only=False)
        assert saved["settings"] == settings
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        for stream, value in zip(streams, saved["streams"], strict=True):
            stream.load_state_dict(value)
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        start, elapsed, histories = saved["step"], saved["seconds"], saved["histories"]
        # Rebuild the hash of previously consumed paired indices from the parent state.
        replay = [EpochStream(bounds[i + 1] - bounds[i], 0) for i in range(3)]
        for stream, value in zip(replay, state["streams"], strict=True):
            stream.load_state_dict(value)
        for _ in range(start):
            idx = np.concatenate(
                [
                    stream.take(n) + lo
                    for stream, n, lo in zip(replay, spec["batch_parts"], bounds[:-1], strict=True)
                ]
            )
            digest.update(idx.tobytes())
    initial_vector = torch.nn.utils.parameters_to_vector(
        [v for v in state["model"].values()]
    ).clone()

    def checkpoint(step):
        metrics, predictions = measure(model, world, table, source, device)
        record = dict(
            step=step,
            seconds=elapsed,
            metrics=metrics,
            parameter_l2_from_source=float(
                torch.linalg.vector_norm(
                    torch.nn.utils.parameters_to_vector(model.parameters()).detach()
                    - initial_vector
                )
            ),
            last_loss=None if step == 0 else float(graph.loss),
            last_gradient_norm=None if step == 0 else float(graph.grad_norm),
        )
        histories.append(record)
        np.savez_compressed(root / f"predictions-{step:06d}.npz", **predictions)
        write_json(root / "learning.json", histories)
        model.train()
        current = dict(
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            streams=[s.state_dict() for s in streams],
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),
            step=step,
            seconds=elapsed,
            histories=histories,
            settings=settings,
        )
        tmp = root / "latest.tmp"
        torch.save(current, tmp)
        tmp.replace(root / "latest.pt")
        if step == 0 or step == cfg["steps"]:
            torch.save(
                dict(model=current["model"], step=step, settings=settings),
                root / f"weights-{step:06d}.pt",
            )
        print(
            json.dumps(
                dict(
                    run=name,
                    step=step,
                    target=metrics["target_test"]["accuracy"],
                    atomic=metrics["atomic"]["accuracy"],
                    background=metrics["background_test"]["accuracy"],
                )
            ),
            flush=True,
        )

    if start == 0:
        checkpoint(0)
    graph = SourceGraphStep(model, opt, gpu_table, sum(spec["batch_parts"]))
    tick = time.monotonic()
    for step in range(start + 1, cfg["steps"] + 1):
        idx = np.concatenate(
            [
                stream.take(n) + lo
                for stream, n, lo in zip(streams, spec["batch_parts"], bounds[:-1], strict=True)
            ]
        )
        digest.update(idx.tobytes())
        graph(torch.as_tensor(idx, device=device))
        if step in cfg["nodes"]:
            torch.cuda.synchronize()
            elapsed += time.monotonic() - tick
            checkpoint(step)
            tick = time.monotonic()
    a, b = ARMS[overrides["arm"]]
    write_json(
        root / "complete.json",
        dict(
            settings=settings,
            final=histories[-1],
            sample_trace_sha256=digest.hexdigest(),
            visible_supervised_tokens=cfg["steps"] * 1984,
            active_supervised_tokens=cfg["steps"] * (1120 * a + 288 * b + 576),
            estimated_flops=cfg["steps"]
            * flops(model.config, spec["repeats"], 128, 25, output_positions=25),
            checkpoint_sha256=sha(root / "latest.pt"),
        ),
    )
