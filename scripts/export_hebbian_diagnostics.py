#!/usr/bin/env python3
"""Reconstruct auxiliary inputs and raw prescribed gradients from frozen checkpoints.

This archive is retrospective. Prospective P0/P1/P2 predictions and the original
node-time cosine diagnostics are preserved and never refitted by this export.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch

from llm_memory_editability.hebbian_learning import (
    ARTIFACTS,
    DATA,
    RESULTS,
    now,
    read_json,
    sha256,
    write_json,
)
from llm_memory_editability.hebbian_model import QwenExperiment
from llm_memory_editability.hebbian_train import gradient_cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    pools = read_json(DATA / "pools.json")
    episodes = read_json(DATA / "episodes.json")
    records = {r["case_id"]: r for values in pools.values() for r in values}
    out = RESULTS / "diagnostic-archive"
    out.mkdir(exist_ok=True)
    engine = QwenExperiment(args.device)
    if not (out / "upstream-inputs.npz").exists():
        ids, inputs, features = [], [], []
        rows = [
            (r, v) for r in sorted(records.values(), key=lambda r: r["case_id"]) for v in range(3)
        ]
        for start in range(0, len(rows), 24):
            chunk = rows[start : start + 24]
            phi = engine.features([r["encoded"][v] for r, v in chunk])
            features.append(phi.cpu().numpy())
            inputs.append(engine.last_layer_inputs.cpu().numpy())
            ids.extend((r["case_id"], v) for r, v in chunk)
        np.savez_compressed(
            out / "upstream-inputs.npz",
            case_view=np.array(ids),
            post_attention_layernorm=np.concatenate(inputs),
            original_phi=np.concatenate(features),
        )
        write_json(
            out / "upstream-inputs.json",
            {
                "exported_at": now(),
                "status": "retrospective_reconstruction_from_original_model",
                "reason": "All parameters upstream of layer 14 were frozen in every condition",
                "rows": len(rows),
                "sha256": sha256(out / "upstream-inputs.npz"),
            },
        )
    engine.configure_trainable("adapt")
    blr = read_json(ARTIFACTS / "B-lock.json")["learning_rate"]
    tasks = [
        (f"B/eval-lr{blr:g}-e{e}", [min(case_ids)], [0])
        for e, case_ids in enumerate(episodes["B_eval"])
    ]
    tasks += [
        (f"C/adapt-{c}-s{s}-e{e}", sorted(case_ids)[:4], [0, 16, 128])
        for c, s, (e, case_ids) in itertools.product(
            ["C0", "C1", "C2"], [0, 1, 2], enumerate(episodes["C_eval"])
        )
    ]
    receipts = []
    for name, case_ids, nodes in tasks:
        path = RESULTS / name
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        for node in nodes:
            receipt_path = target / f"raw-gradients-{node:04d}.json"
            if receipt_path.exists():
                receipts.append(read_json(receipt_path))
                continue
            checkpoint = path / f"checkpoint-{node:04d}.pt"
            while args.follow and not checkpoint.exists():
                time.sleep(10)
            if not checkpoint.exists():
                raise RuntimeError(f"Missing frozen checkpoint: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            with torch.no_grad():
                for key, parameter in engine.mlp.named_parameters():
                    parameter.copy_(state["parameters"][key])
            raw, cosines = {}, {}
            for case in case_ids:
                grads = [engine.gradient(e)[0].cpu() for e in records[case]["encoded"]]
                raw[str(case)] = torch.stack(grads)
                cosines[case] = [gradient_cos(grads[0], grad) for grad in grads[1:]]
            max_error = None
            if name.startswith("C/"):
                original = read_json(path / f"gradients-{node:04d}.json")
                max_error = max(
                    abs(cosines[r["case_id"]][i] - r[key])
                    for r in original
                    for i, key in enumerate(["p0_p1_gradient_cos", "p0_p2_gradient_cos"])
                )
                if max_error > 5e-6:
                    raise RuntimeError(
                        f"Reconstructed gradient disagrees: {name}/{node}: {max_error}"
                    )
            file = target / f"raw-gradients-{node:04d}.pt"
            torch.save(
                {
                    "gradients": raw,
                    "case_ids": case_ids,
                    "views": [0, 1, 2],
                    "parameter": "model.layers.14.mlp.down_proj.weight",
                    "loss": "complete answer incl EOS mean-token NLL",
                    "step": node,
                    "parent_hash": state["meta"]["parent_hash"],
                },
                file,
            )
            receipt = {
                "run": name,
                "step": node,
                "exported_at": now(),
                "status": "retrospective_raw_gradient_archive_at_preselected_nodes",
                "checkpoint_sha256": sha256(checkpoint),
                "raw_sha256": sha256(file),
                "gradients": len(case_ids) * 3,
                "max_cosine_error_vs_node_diagnostic": max_error,
            }
            write_json(receipt_path, receipt)
            receipts.append(receipt)
        print(name, "archived", flush=True)
    write_json(
        ARTIFACTS / "diagnostic-archive-audit.json",
        {
            "time": now(),
            "records": receipts,
            "raw_gradient_tensors": sum(r["gradients"] for r in receipts),
            "scope": "B first ID per episode at step0; C first4 IDs per episode at 0/16/128",
            "prospective_predictions_changed": False,
            "pass": True,
        },
    )


if __name__ == "__main__":
    main()
