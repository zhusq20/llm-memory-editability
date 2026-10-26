#!/usr/bin/env python3
"""Zero-training test of 5-8 hop questions on saved two-graph Loop checkpoints.

Panels are generated from the same permutation world as training: for each graph and
k in 5..8, ten relation templates are evaluated at every head, so answers are uniform
over the graph's entities (suffix-only chance 1/500). No composition longer than four
hops was ever trained.

Learned absolute position embeddings beyond index 5 never received gradient and are
zero after weight decay. Position controls therefore shift already-solved questions
right with s leading pad tokens (s=1..4) so that their answer sits where the k-hop
answer sits (k = 4 + s). Every question is read out after each of R=1..32 rounds from
one fresh call; only the answer token is scored (EOS would need position k+1).

Two readouts are recorded. ``full`` is the frozen full-vocabulary argmax. ``entity`` is a
post hoc diagnostic added after every endpoint model was found to emit EOS at answer
positions >= 5 (position 5 was only ever trained to predict EOS): the argmax restricted
to the 1000 entity tokens of both graphs. It uses the answer domain, which a deployed
model would not have, and is never the primary metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from llm_memory_editability.loop_hop_data import load, traverse  # noqa: E402
from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT  # noqa: E402

HOPS = (5, 6, 7, 8)
SHIFTS = (0, 1, 2, 3, 4)
CONTROLS = ("atomic_id", "atomic_ood", "id_2", "ood_2", "id_3", "ood_3", "id_4", "ood_4")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def beyond_panels(world, metadata, seed, templates=10):
    """Rows (head, r1..rk, tail) for graphs A (id) and B (ood), k in HOPS."""
    n, r = metadata["entities_per_graph"], metadata["relations"]
    rng = np.random.default_rng(seed)
    panels, audit = {}, {}
    trained = {k: {tuple(row[:-1]) for row in world[f"train_{k}"]} for k in (2, 3, 4)}
    for k in HOPS:
        for graph, split in ((0, "id"), (1, "ood")):
            codes = rng.choice(r**k, templates, replace=False)
            template = np.array([np.unravel_index(int(c), (r,) * k) for c in codes])
            relations = np.repeat(template, n, axis=0)
            heads = np.tile(np.arange(n), templates)
            nodes = traverse(heads, relations, world["permutations"][graph])
            rows = np.column_stack((heads + graph * n, relations, nodes[:, -1] + graph * n))
            name = f"{split}_{k}"
            panels[name] = rows
            # A trailing sub-chain from an intermediate bridge that was itself a trained
            # composition could be answered from memory; count such rows (graph A only).
            leaked = np.zeros(len(rows), dtype=bool)
            if graph == 0:
                for length in (2, 3, 4):
                    start = k - length
                    keys = np.column_stack((nodes[:, start], relations[:, start:]))
                    leaked |= np.array([tuple(key) in trained[length] for key in keys])
            repeated = np.array([len(set(path)) != len(path) for path in nodes])
            answers = np.sort(rows[:, -1].reshape(templates, n), axis=1)
            audit[name] = {
                "n": int(len(rows)),
                "templates": template.tolist(),
                "answers_uniform_per_template": bool((answers == np.arange(n) + graph * n).all()),
                "rows_with_trained_trailing_subchain": int(leaked.sum()),
                "repeated_entity_paths": int(repeated.sum()),
            }
    return panels, audit


def encode(rows, metadata, shift):
    """Tokens [pad]*shift + (head, relations); the answer is read at the last token."""
    hops = rows.shape[1] - 2
    tokens = np.full((len(rows), shift + hops + 1), metadata["pad_token"], np.int64)
    tokens[:, shift] = rows[:, 0] + metadata["entity_offset"]
    tokens[:, shift + 1 :] = rows[:, 1:-1] + metadata["relation_offset"]
    answer = np.full((len(rows), 1), shift + hops, np.int64)
    target = rows[:, -1] + metadata["entity_offset"]
    return tokens, answer, target


@torch.no_grad()
def per_round(model, rows, metadata, shift, rounds, device, batch_size=500):
    tokens, answer, target = encode(rows, metadata, shift)
    first = metadata["entity_offset"]
    last = first + 2 * metadata["entities_per_graph"]
    full, entity = [], []
    for begin in range(0, len(rows), batch_size):
        x = torch.as_tensor(tokens[begin : begin + batch_size], device=device)
        p = torch.as_tensor(answer[begin : begin + batch_size], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _, _ = model.trace(x, p, rounds)
        logits = logits[:, :, 0]
        full.append(logits.argmax(-1).cpu().numpy())
        entity.append((logits[..., first:last].argmax(-1) + first).cpu().numpy())
    full, entity = np.concatenate(full, axis=1), np.concatenate(entity, axis=1)
    return full, entity, (full == target[None]).mean(1), (entity == target[None]).mean(1)


def build(config, metadata, spec, device):
    hf = GPT2Config(
        **config["model"],
        n_layer=config["training"]["unique_layers"],
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    model = LoopNativeHandoffGPT(hf, 4, handoff=spec["handoff"], **config["handoff"])
    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    if spec["weights"] == "ema":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(saved["ema"][name].to(parameter.dtype))
    return model.to(device).eval(), int(saved["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="JSON model spec")
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026102101)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    world, panels, metadata = load(config["input_root"])
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    beyond, audit = beyond_panels(world, metadata, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    panel_path = args.output / "beyond-panels.npz"
    if panel_path.exists():
        with np.load(panel_path) as previous:
            if any(not np.array_equal(previous[k], v) for k, v in beyond.items()):
                raise ValueError("Regenerated panels differ from the saved ones")
    else:
        np.savez_compressed(panel_path, **beyond)
        (args.output / "beyond-panels-audit.json").write_text(json.dumps(audit, indent=1))
    for raw in args.model:
        spec = json.loads(raw)
        destination = args.output / f"{spec['name']}.json"
        if destination.exists():
            raise FileExistsError(destination)
        model, step = build(config, metadata, spec, device)
        result = {
            "model": spec,
            "step": step,
            "checkpoint_sha256": sha(spec["checkpoint"]),
            "rounds": args.rounds,
            "beyond": {},
            "controls": {},
            "beyond_entity": {},
            "controls_entity": {},
        }
        folder = args.output / "predictions" / spec["name"]
        folder.mkdir(parents=True, exist_ok=False)
        for name, rows in beyond.items():
            full, entity, accuracy, restricted = per_round(
                model, rows, metadata, 0, args.rounds, device
            )
            np.savez_compressed(folder / f"{name}.npz", rows=rows, answer=full, entity=entity)
            result["beyond"][name] = accuracy.round(6).tolist()
            result["beyond_entity"][name] = restricted.round(6).tolist()
        for name in CONTROLS:
            result["controls"][name], result["controls_entity"][name] = {}, {}
            for shift in SHIFTS:
                full, entity, accuracy, restricted = per_round(
                    model, panels[name], metadata, shift, args.rounds, device
                )
                np.savez_compressed(folder / f"{name}-shift{shift}.npz", answer=full, entity=entity)
                result["controls"][name][str(shift)] = accuracy.round(6).tolist()
                result["controls_entity"][name][str(shift)] = restricted.round(6).tolist()
        result["source_sha256"] = {
            p: sha(ROOT / p)
            for p in (
                "scripts/evaluate_loop_beyond_four_hops.py",
                "src/llm_memory_editability/loop_native_handoff.py",
                "src/llm_memory_editability/loop_hop_data.py",
            )
        }
        result["device"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        destination.write_text(json.dumps(result, indent=1))
        print(json.dumps({"done": spec["name"], "step": step}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.umask(0o022)
    main()
