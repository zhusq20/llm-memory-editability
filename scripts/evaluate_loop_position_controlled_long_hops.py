"""Inference-only position interventions on the completed native four-arm endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from evaluate_loop_beyond_four_hops import CONTROLS, build, encode  # noqa: E402

MODES = ("absolute", "zero", "clamp4", "interpolate4")
ARMS = ("invariant", "ema", "content", "recipe-additive")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def position_intervention(model, mode, offset=0):
    """Replace only the wpe output; tokens, causal mask and learned weights stay fixed."""
    if mode not in MODES or offset < 0 or (offset and mode != "absolute"):
        raise ValueError("Offsets are a separate absolute-position-only diagnostic")

    def hook(module, inputs, original):
        indices = inputs[0]
        if mode == "absolute":
            return original if offset == 0 else F.embedding(indices + offset, module.weight)
        if mode == "zero":
            return torch.zeros_like(original)
        if mode == "clamp4":
            return F.embedding(indices.clamp(max=4), module.weight)
        if indices.numel() <= 5:
            return original
        coordinates = indices.to(module.weight.dtype) * (4.0 / (indices.numel() - 1))
        lower = coordinates.floor().long().clamp(max=4)
        upper = (lower + 1).clamp(max=4)
        weight = (coordinates - lower).unsqueeze(-1)
        return (
            F.embedding(lower, module.weight) * (1 - weight)
            + F.embedding(upper, module.weight) * weight
        )

    handle = model.transformer.wpe.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def evaluate(model, rows, metadata, device, batch_size=500):
    tokens, positions, target = encode(rows, metadata, 0)
    first = metadata["entity_offset"]
    last = first + 2 * metadata["entities_per_graph"]
    full, entity, nll = [], [], []
    for begin in range(0, len(rows), batch_size):
        x = torch.as_tensor(tokens[begin : begin + batch_size], device=device)
        p = torch.as_tensor(positions[begin : begin + batch_size], device=device)
        y = torch.as_tensor(target[begin : begin + batch_size], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _, _ = model.trace(x, p, 32)
        logits = logits[:, :, 0]
        full.append(logits.argmax(-1).cpu().numpy())
        entity.append((logits[..., first:last].argmax(-1) + first).cpu().numpy())
        nll.append(
            -logits.float()
            .log_softmax(-1)
            .gather(-1, y[None, :, None].expand(32, -1, 1))
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    full, entity, nll = (np.concatenate(values, axis=1) for values in (full, entity, nll))
    metrics = {
        "n": len(rows),
        "accuracy": (full == target[None]).mean(1).tolist(),
        "entity_accuracy_diagnostic": (entity == target[None]).mean(1).tolist(),
        "answer_nll": nll.mean(1).tolist(),
        "eos_rate": (full == metadata["end_marker"]).mean(1).tolist(),
    }
    return dict(rows=rows, target=target, answer=full, entity=entity, answer_nll=nll), metrics


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.16)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    data = args.persistent / "loop-hop-development-v1/inputs/data"
    previous = args.persistent / "loop-beyond-four-hops-20261020"
    metadata = json.loads((data / "complete.json").read_text())
    assert sha(data / "world.npz") == metadata["world_sha256"]
    assert sha(data / "panels.npz") == metadata["panels_sha256"]
    world, short, long = (
        dict(np.load(path))
        for path in (data / "world.npz", data / "panels.npz", previous / "beyond-panels.npz")
    )
    for name, rows in long.items():
        graph = int(name.startswith("ood"))
        node = rows[:, 0] - 500 * graph
        for relation in rows[:, 1:-1].T:
            node = world["permutations"][graph, relation, node]
        np.testing.assert_array_equal(node + 500 * graph, rows[:, -1])
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "status.json").exists():
        raise FileExistsError("Evaluation already has a status; do not duplicate or overwrite it")
    status = {
        "state": "running",
        "completed_panels": 0,
        "planned_panels": 224,
        "training_updates": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "config_sha256": sha(args.config),
        "world_sha256": sha(data / "world.npz"),
        "long_panels_sha256": sha(previous / "beyond-panels.npz"),
    }
    write_json(args.output / "status.json", status)
    for arm in ARMS:
        name = f"{arm}-1500k"
        reference = json.loads((previous / f"{name}.json").read_text())
        checkpoint = args.persistent / "loop-native-handoff-development-v1/runs" / arm
        checkpoint /= "checkpoint-1500000.pt"
        assert sha(checkpoint) == reference["checkpoint_sha256"]
        model, step = build(
            config, metadata, {**reference["model"], "checkpoint": str(checkpoint)}, device
        )
        assert step == 1500000 and model.first_round_input == "state_plus_embedding"
        before = state_hash(model)
        model.requires_grad_(False)
        result = {
            "arm": arm,
            "step": step,
            "weights": "ema",
            "checkpoint_sha256": sha(checkpoint),
            "state_sha256_before": before,
            "modes": {},
            "position_only_shifts": {},
            "baseline_reference_checks": {},
        }
        folder = args.output / "predictions" / arm
        folder.mkdir(parents=True, exist_ok=False)

        def measure(mode, panel, rows, offset=0, *, model=model, folder=folder, arm=arm):
            if time.monotonic() - started > 3300:
                raise TimeoutError("Inference-only batch exceeded its 55-minute budget")
            with position_intervention(model, mode, offset):
                arrays, metrics = evaluate(model, rows, metadata, device)
            key = f"{mode}-{panel}" + (f"-offset{offset}" if offset else "")
            np.savez_compressed(folder / f"{key}.npz", **arrays)
            status.update(
                arm=arm,
                panel=key,
                completed_panels=status["completed_panels"] + 1,
                elapsed_seconds=time.monotonic() - started,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
            if (
                status["peak_reserved_mib"] + 6144
                > torch.cuda.get_device_properties(0).total_memory * 0.16 / 2**20
            ):
                raise RuntimeError("Evaluation exceeded allocation with 6 GiB task reserve")
            write_json(args.output / "status.json", status)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "panel": key,
                        "R16": metrics["accuracy"][15],
                        "completed": status["completed_panels"],
                    }
                ),
                flush=True,
            )
            return arrays, metrics

        for mode in MODES:
            result["modes"][mode] = {"long": {}, "short": {}}
            for panel, rows in sorted(long.items(), key=lambda item: -item[1].shape[1]):
                arrays, metrics = measure(mode, panel, rows)
                result["modes"][mode]["long"][panel] = metrics
                if mode == "absolute":
                    with np.load(previous / "predictions" / name / f"{panel}.npz") as original:
                        disagreement = int((arrays["answer"] != original["answer"]).sum())
                    result["baseline_reference_checks"][panel] = {
                        "token_disagreements": disagreement,
                        "max_accuracy_difference_pp": float(
                            100
                            * np.max(
                                np.abs(np.asarray(metrics["accuracy"]) - reference["beyond"][panel])
                            )
                        ),
                    }
            if mode in ("absolute", "zero"):
                for panel in CONTROLS:
                    arrays, metrics = measure(mode, panel, short[panel])
                    result["modes"][mode]["short"][panel] = metrics
                    if mode == "absolute":
                        with np.load(
                            previous / "predictions" / name / f"{panel}-shift0.npz"
                        ) as original:
                            disagreement = int((arrays["answer"] != original["answer"]).sum())
                        result["baseline_reference_checks"][panel] = {
                            "token_disagreements": disagreement,
                            "max_accuracy_difference_pp": float(
                                100
                                * np.max(
                                    np.abs(
                                        np.asarray(metrics["accuracy"])
                                        - reference["controls"][panel]["0"]
                                    )
                                )
                            ),
                        }
            else:
                result["modes"][mode]["short"] = result["modes"]["absolute"]["short"]
                result["modes"][mode]["short_reuse_reason"] = (
                    "Exact same position embeddings for length <=5; no extra forward"
                )
        for panel in ("id_4", "ood_4"):
            result["position_only_shifts"][panel] = {}
            for offset in (1, 2, 3, 4):
                _, metrics = measure("absolute", panel, short[panel], offset)
                result["position_only_shifts"][panel][str(offset)] = metrics
        result["state_sha256_after"] = state_hash(model)
        assert result["state_sha256_after"] == before
        result["elapsed_seconds_cumulative"] = time.monotonic() - started
        write_json(args.output / f"{arm}.json", result)
        del measure, model
        torch.cuda.empty_cache()
    assert status["completed_panels"] == status["planned_panels"]
    status.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "status.json", status)


if __name__ == "__main__":
    os.umask(0o022)
    main()
