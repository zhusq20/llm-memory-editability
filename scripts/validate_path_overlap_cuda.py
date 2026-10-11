"""CUDA execution checks without running the frozen fact-selection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.path_overlap import Executor, World, load_model, margin


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    base = Path(args.base)
    root = base / "inputs/grokking-reproduction-v1"
    destination = base / "validation"
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / ("cuda-audit.json" if args.audit else "cuda-validation.json")
    if result_path.exists():
        raise FileExistsError(f"Preserve existing validation: {result_path}")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}")
    receipt = json.loads((base / "input-receipt.json").read_text())
    expected = {(entry["run"], entry["step"]): entry for entry in receipt["checkpoints"]}
    result = {
        "passed": False,
        "fresh_process": bool(args.audit),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "optimizer_updates": 0,
        "fact_selection_executed": False,
        "two_hop_interventions_executed": False,
        "device": torch.cuda.get_device_name(device),
        "models": [],
    }
    try:
        matrix = torch.randn(128, 128, device=device, requires_grad=True)
        matrix.square().mean().backward()
        torch.cuda.synchronize(device)
        assert torch.isfinite(matrix.grad).all()
        world = World(root / "data")
        tokens = torch.as_tensor(world.atomic_tokens(np.arange(16)), device=device)
        for run in ("standard8-phi7.2-wd0.1", "loop4x2-phi7.2-wd0.1"):
            # Run names are matched against the receipt rather than inferred from a document.
            if (run, 1_500_000) not in expected:
                candidates = [
                    name
                    for name, step in expected
                    if step == 1_500_000
                    and name.startswith(run.split("-phi")[0])
                    and "phi7.2" in name
                    and name.endswith("wd0.1")
                ]
                assert len(candidates) == 1, (run, candidates)
                run = candidates[0]
            checkpoint = root / "development" / run / "checkpoint-1500000.pt"
            checkpoint_sha = digest_file(checkpoint)
            assert checkpoint_sha == expected[run, 1_500_000]["sha256"]
            model, _, _ = load_model(root, run, 1_500_000, world.metadata, device)
            before = digest_parameters(model)
            executor = Executor(model)
            with torch.no_grad():
                original = model(tokens)[:, 1]
                replica = executor.logits(tokens, 1)
            assert torch.allclose(original, replica, rtol=1e-4, atol=2e-4)
            assert torch.equal(original.argmax(-1), replica.argmax(-1))
            sample = tokens[:1]
            target = replica[:1].argmax(-1)
            gates = torch.ones(
                1, executor.unique, executor.neurons, device=device, requires_grad=True
            )
            loss = margin(
                executor.logits(sample, 1, gates=gates, gate_scope=executor.scope("all")), target
            ).sum()
            (derivative,) = torch.autograd.grad(loss, gates)
            unit = int(derivative.flatten().argmax())
            slope = float(derivative.flatten()[unit])
            assert slope > 0 and np.isfinite(slope)
            finite = []
            for epsilon in (0.01, 0.003):
                plus, minus = gates.detach().clone(), gates.detach().clone()
                plus.flatten()[unit] += epsilon
                minus.flatten()[unit] -= epsilon
                with torch.no_grad():
                    high = margin(
                        executor.logits(sample, 1, gates=plus, gate_scope=executor.scope("all")),
                        target,
                    )
                    low = margin(
                        executor.logits(sample, 1, gates=minus, gate_scope=executor.scope("all")),
                        target,
                    )
                estimate = float((high - low) / (2 * epsilon))
                assert np.isclose(slope, estimate, rtol=0.05, atol=0.003), (slope, estimate)
                finite.append({"epsilon": epsilon, "estimate": estimate})
            after = digest_parameters(model)
            assert before == after
            artifact = destination / f"{run}-logits.npy"
            values = replica.detach().cpu().numpy()
            if args.audit:
                np.testing.assert_array_equal(values, np.load(artifact))
                original_result = json.loads((destination / "cuda-validation.json").read_text())
                assert (original_result["hostname"], original_result["pid"]) != (
                    socket.gethostname(),
                    os.getpid(),
                )
                reference = next(item for item in original_result["models"] if item["run"] == run)
                assert reference["parameters_sha256"] == before
                assert reference["derivative"] == slope
                assert reference["finite_difference"] == finite
            else:
                np.save(artifact, values)
            result["models"].append(
                {
                    "run": run,
                    "checkpoint_sha256": checkpoint_sha,
                    "parameters_sha256": before,
                    "parameters_unchanged": before == after,
                    "max_logit_difference": float((original - replica).abs().max()),
                    "discrete_agreement": 1.0,
                    "derivative": slope,
                    "finite_difference": finite,
                    "logits_sha256": digest_file(artifact),
                }
            )
            del model, executor, original, replica, gates, derivative
            torch.cuda.empty_cache()
        result["passed"] = True
        write_json(result_path, result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        result["error"] = repr(error)
        write_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
