#!/usr/bin/env python3
"""Inference-only follow-up on candidate interfaces and input-only embedding scale."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

FROZEN = Path(os.environ["LOOP_DIAGNOSTIC_SOURCE"])
sys.path[:0] = [str(FROZEN / "scripts"), str(FROZEN / "src")]
import diagnose_invariant_second_hop as second  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "state_regression",
    os.environ.get("LOOP_REGRESSION_MODULE", "/control/diagnose_loop_state_regression.py"),
)
regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)


def input_norm_hook(embedding, early_weight, metadata):
    """Change the B-entity input lookup only; never alter the tied weight tensor."""
    early = early_weight.norm(dim=-1).to(embedding.weight.device)
    late = embedding.weight.norm(dim=-1)
    ratio = torch.ones_like(late)
    n, offset = metadata["entities_per_graph"], metadata["entity_offset"]
    ratio[offset + n : offset + 2 * n] = (
        early[offset + n : offset + 2 * n] / late[offset + n : offset + 2 * n]
    )

    def input_only(module, inputs, output):
        return output * ratio[inputs[0]][..., None]

    return embedding.register_forward_hook(input_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    base = json.loads(Path(p["base_protocol"]).read_text())
    config = json.loads(Path(base["config"]).read_text())
    world, panels, metadata = regression.load(config["input_root"])
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.06)
    model = regression.native.initialize(config, "state-mlp", metadata, device).eval()
    root = Path(base["checkpoint_root"])
    saved = {}
    for step in p["steps"]:
        file = root / f"checkpoint-{step:07d}.pt"
        assert regression.digest(file) == base["checkpoint_sha256"][str(step)]
        saved[step] = torch.load(file, map_location="cpu", weights_only=False)
    import wandb

    tracker = wandb.init(
        entity="zsqzz",
        project="llm-memory-editability",
        job_type="checkpoint-diagnostic",
        name=p["batch"],
        mode="offline",
        dir=str(out),
        config=p,
    )
    results, checks = [], []
    started = time.monotonic()
    try:
        for step in p["steps"]:
            regression.install(model, saved, step, step)
            before = regression.native.model_digest(model)
            probe = second.RoundProbe(model)
            for panel in p["second_hop_panels"]:
                rows = panels[panel]
                nodes, _ = second.path_details(rows, metadata, world["permutations"])
                e1, e2 = (nodes[:, j] + metadata["entity_offset"] for j in (1, 2))
                n = metadata["entities_per_graph"]
                graph = nodes[:, 1] // n
                wrong_bridge = graph * n + (nodes[:, 1] % n + 1) % n
                wrong = (
                    graph * n
                    + world["permutations"][graph, rows[:, 2], wrong_bridge % n]
                    + metadata["entity_offset"]
                )
                wrong_bridge += metadata["entity_offset"]
                assert np.all(wrong != e2)
                x, pos, _ = regression.pack_rows(rows, metadata)
                chunks = {condition: [] for condition in p["conditions"]}
                for begin in range(0, len(rows), 256):
                    start = time.monotonic()
                    if start - started > p["timeout_seconds"]:
                        raise TimeoutError("Follow-up runtime bound exceeded")
                    tensors = [
                        torch.as_tensor(a[begin : begin + 256], device=device)
                        for a in (x, pos, e1, e2, wrong, wrong_bridge)
                    ]
                    tokens, positions, bridge, target, wrong_target, wrong_entity = tensors
                    with torch.no_grad(), regression.native.autocast(device):
                        donors = probe.donors_for(tokens, bridge, wrong_entity)
                        original_output = None
                        for condition in p["conditions"]:
                            values = second.trace(
                                model,
                                probe,
                                tokens,
                                positions,
                                condition,
                                donors,
                                bridge,
                                target,
                                wrong_target,
                                rich=condition == "native",
                            )
                            if condition == "native":
                                original_output = probe.stages["output"].clone()
                            else:
                                keep = [j for j in range(tokens.shape[1]) if j != 2]
                                assert torch.equal(
                                    probe.stages["output"][:, keep], original_output[:, keep]
                                )
                            chunks[condition].append(values)
                        if begin == 0:
                            inp = 2 * regression._loss_dtype(model._embedded(tokens))
                            assert torch.equal(probe.block(inp), probe.original(inp))
                            changed = tokens.clone()
                            changed[:, 3:] = 0
                            causal = second.trace(
                                model,
                                probe,
                                changed,
                                positions,
                                "native",
                                donors,
                                bridge,
                                target,
                                wrong_target,
                            )
                            for key in ("raw_token", "candidate_token", "aggregate_token", "gamma"):
                                np.testing.assert_array_equal(chunks["native"][0][key], causal[key])
                            checks.append(
                                {
                                    "step": step,
                                    "panel": panel,
                                    "native_block_exact": True,
                                    "causal_prefix_exact": True,
                                    "other_receiver_positions_exact": True,
                                    "donor_contains_no_second_hop_target": True,
                                }
                            )
                    torch.cuda.synchronize()
                    time.sleep(2 * (time.monotonic() - start))
                for condition, parts in chunks.items():
                    values = {
                        key: np.concatenate([part[key] for part in parts]) for key in parts[0]
                    }
                    folder = out / str(step) / condition
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        folder / f"{panel}.npz",
                        rows=rows,
                        bridge=e1,
                        target=e2,
                        wrong_bridge=wrong_bridge,
                        wrong_target=wrong,
                        **values,
                    )
                    if condition == "native":
                        with np.load(
                            root / "predictions" / f"{step:07d}" / "ema-r8" / f"{panel}.npz"
                        ) as original:
                            np.testing.assert_array_equal(original["rows"], rows)
                            np.testing.assert_array_equal(
                                original["answer"], values["answer"][:, 7]
                            )
                    summary = {
                        "step": step,
                        "panel": panel,
                        "condition": condition,
                        **second.metrics(values, rows, e1, e2, wrong, metadata),
                    }
                    results.append(summary)
                    tracker.log(
                        {
                            "evaluation_index": len(results),
                            "checkpoint_step": step,
                            "panel": panel,
                            "condition": condition,
                            "candidate_accuracy": summary["candidate_token_accuracy"],
                            "accuracy_r8": summary["answer_accuracy"][7],
                        }
                    )
                regression.write(out / "summary.json", results)
                regression.write(out / "engineering-validation.json", checks)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "panel": panel,
                            "completed": len(results),
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            assert regression.native.model_digest(model) == before
        # Isolate only the input role of the tied embedding. Readout and soft feedback
        # continue to use the original late weight matrix. This is an oracle diagnostic,
        # using B identity and early row norms, and is not a deployable architecture.
        regression.install(model, saved, 500000, 500000)
        before = regression.native.model_digest(model)
        hook = input_norm_hook(
            model.transformer.wte, saved[256000]["ema"]["transformer.wte.weight"], metadata
        )
        try:
            for panel in base["panels"]:
                rows = panels[panel]
                x, pos, y = regression.pack_rows(rows, metadata)
                parts = []
                for begin in range(0, len(rows), 256):
                    start = time.monotonic()
                    tensors = [
                        torch.as_tensor(a[begin : begin + 256], device=device) for a in (x, pos, y)
                    ]
                    with torch.no_grad(), regression.native.autocast(device):
                        parts.append(regression.trace(model, *tensors, 8, rich=True))
                    torch.cuda.synchronize()
                    time.sleep(2 * (time.monotonic() - start))
                values = {
                    key: np.concatenate([part[key] for part in parts], axis=1) for key in parts[0]
                }
                folder = out / "500000" / "input-B-norm-only"
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / f"{panel}.npz", rows=rows, **values)
                if panel == "id_4":
                    with np.load(root / "predictions/0500000/ema-r8/id_4.npz") as original:
                        np.testing.assert_array_equal(original["answer"], values["answer"][7])
                summary = {
                    "step": 500000,
                    "panel": panel,
                    "condition": "input-B-norm-only",
                    **regression.summarize(values, rows, metadata, world),
                }
                results.append(summary)
                regression.write(out / "summary.json", results)
                tracker.log(
                    {
                        "evaluation_index": len(results),
                        "panel": panel,
                        "condition": "input-B-norm-only",
                        "accuracy_r8": summary["answer_accuracy"][7],
                    }
                )
                print(
                    json.dumps(
                        {
                            "step": 500000,
                            "panel": panel,
                            "condition": "input-B-norm-only",
                            "r8": summary["answer_accuracy"][7],
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        finally:
            hook.remove()
        assert regression.native.model_digest(model) == before
        assert all(
            regression.digest(root / f"checkpoint-{step:07d}.pt")
            == base["checkpoint_sha256"][str(step)]
            for step in p["steps"]
        )
        regression.write(
            out / "complete.json",
            {
                "passed": True,
                "groups": len(results),
                "new_training_updates": 0,
                "parameters_and_checkpoints_unchanged": True,
                "seconds": time.monotonic() - started,
            },
        )
        tracker.finish()
    except BaseException as error:
        regression.write(
            out / "failure.json", {"error": repr(error), "seconds": time.monotonic() - started}
        )
        tracker.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
