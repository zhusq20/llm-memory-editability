"""Measure endpoint E/R geometry and actual edit-loss gradients before any update."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_data import load_world, paired_edit, write_json
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_train import code_fingerprint, precision, tensor_data


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-20)


def pair_metrics(z, gradient, world, ids, target):
    members = world.relation[ids] == 2
    same = (world.company[ids, None] == world.company[ids][None, :]) & members[:, None] & members
    same &= world.answers[ids, None] == world.answers[ids][None, :]
    same &= np.triu(np.ones(same.shape, dtype=bool), 1)
    same_target = target[ids, None] == target[ids][None, :]
    z_cosine = normalize(z) @ normalize(z).T
    # Full two-token teacher-forced CE loss; gradient spans every prefix position.
    g = gradient.reshape(len(ids), -1)
    g_cosine = normalize(g) @ normalize(g).T
    result = {}
    for name, mask in (
        ("same_new_answer", same & same_target),
        ("different_new_answer", same & ~same_target),
    ):
        result[name] = {
            "pairs": int(mask.sum()),
            "z_cosine": float(z_cosine[mask].mean()) if mask.any() else None,
            "gradient_cosine": float(g_cosine[mask].mean()) if mask.any() else None,
            "negative_gradient_fraction": float((g_cosine[mask] < 0).mean())
            if mask.any()
            else None,
        }
    return result


def measure(model, world, data, pair, kind, layers):
    ids = pair["E"]
    device = data["tokens"].device
    target_data = tensor_data(world, device, pair[kind])
    captured_z, captured_h, handles = {}, {}, []
    for layer in layers:

        def capture_z(_module, _inputs, output, layer=layer):
            captured_z[layer] = output

        def capture_h(_module, _inputs, output, layer=layer):
            captured_h[layer] = output

        handles.append(model.blocks[layer].mlp.activation.register_forward_hook(capture_z))
        handles.append(model.blocks[layer].mlp.register_forward_hook(capture_h))
    try:
        with precision(device):
            logits = model(target_data["tokens"][ids], target_data["positions"][ids]).float()
            per_query = (
                F.cross_entropy(
                    logits.reshape(-1, world.vocab_size),
                    target_data["labels"][ids].reshape(-1),
                    reduction="none",
                )
                .reshape(len(ids), 2)
                .mean(-1)
            )
        gradients = torch.autograd.grad(per_query.sum(), [captured_h[layer] for layer in layers])
        z = {layer: captured_z[layer][:, 3].detach().float().cpu().numpy() for layer in layers}
        gradient = {
            layer: g.detach().float().cpu().numpy()
            for layer, g in zip(layers, gradients, strict=True)
        }
        losses = per_query.detach().cpu().numpy()
        captured_z.clear()
        captured_h.clear()
        replay_z = {layer: [] for layer in layers}
        with torch.no_grad(), precision(device):
            for begin in range(0, len(pair["replay"]), 256):
                ri = pair["replay"][begin : begin + 256]
                model(data["prompts"][ri], data["lengths"][ri, None] - 1)
                for layer in layers:
                    replay_z[layer].append(captured_z[layer][:, 3].float().cpu().numpy())
        records, arrays = [], {"E": ids, "R": pair["replay"], "per_query_two_token_ce": losses}
        for layer in layers:
            ez, rz = z[layer], np.concatenate(replay_z[layer])
            cosine = normalize(ez) @ normalize(rz).T
            record = {
                "layer": layer,
                "kind": kind,
                "support": pair["support"],
                "E_n": len(ids),
                "R_n": len(pair["replay"]),
                "E_z_norm": float(np.linalg.norm(ez, axis=-1).mean()),
                "R_z_norm": float(np.linalg.norm(rz, axis=-1).mean()),
                "E_R_mean_cosine": float(cosine.mean()),
                "E_R_mean_nearest_cosine": float(cosine.max(-1).mean()),
                "initial_mean_two_token_ce": float(losses.mean()),
                "h_loss_gradient_rms": float(np.sqrt(np.mean(gradient[layer] ** 2))),
                "within_same_company_relation_old_answer": pair_metrics(
                    ez, gradient[layer], world, ids, pair[kind]
                ),
                "R_by_stratum": {},
            }
            for group in range(4):
                mask = pair["strata"][pair["replay"]] == group
                record["R_by_stratum"][str(group)] = {
                    "n": int(mask.sum()),
                    "mean_cosine": float(cosine[:, mask].mean()) if mask.any() else None,
                }
            records.append(record)
            arrays[f"E_z_layer_{layer}"] = ez
            arrays[f"E_h_loss_gradient_layer_{layer}"] = gradient[layer]
        return records, arrays
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-dev-v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.root)
    for run in sorted(root.glob("world-*-seed-*-*")):
        out = run / "edit-geometry-13280"
        if (out / "measurements.json").exists():
            continue
        started = time.perf_counter()
        config = json.loads((run / "config.json").read_text())
        world = load_world(f"data/bios-work-v1/world-{config['world_seed']}")
        checkpoint_path = run / "model-13280.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        device = torch.device(args.device)
        model = CausalLM(ModelConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        data = tensor_data(world, device)
        out.mkdir(exist_ok=True)
        records = []
        for support in (0, 1):
            pair = paired_edit(world, support)
            for kind in ("coherent", "exception"):
                rows, arrays = measure(model, world, data, pair, kind, (1, 4))
                records.extend(rows)
                np.savez_compressed(out / f"support-{support}-{kind}.npz", **arrays)
        write_json(
            out / "measurements.json",
            {
                "run": run.name,
                "step": 13280,
                "records": records,
                "scope": (
                    "E and permitted replay R only; no D or heldout U features; "
                    "descriptive post-hoc development analysis, not a validated predictor"
                ),
                "gradient_definition": (
                    "d mean(two supervised token CE) / d MLP output h for each E query; "
                    "actual targets and answer prefixes; not embedding differences"
                ),
                "precision": "BF16 autocast" if device.type == "cuda" else "FP32",
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "seconds": time.perf_counter() - started,
                **code_fingerprint(),
            },
        )
        print(
            json.dumps(
                {"run": run.name, "records": len(records), "seconds": time.perf_counter() - started}
            ),
            flush=True,
        )
        del model, data


if __name__ == "__main__":
    main()
