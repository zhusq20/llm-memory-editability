#!/usr/bin/env python3
"""Train, independently reload, or measure the registered composition curves."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def preflight(config, out):
    import numpy as np
    import torch

    from llm_memory_editability.composition_curves import construct
    from llm_memory_editability.grok_depth import GraphStep, make_optimizer, utc, write_json

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    results = []
    for layers in config["calibration_layers"]:
        spec = {**config["base_spec"], "layers": layers, "initialization": config["initialization"]}
        model, _ = construct(spec, "cuda:0")
        lr = torch.tensor(spec["lr"], device="cuda:0")
        optimizer = make_optimizer(model, lr, spec["weight_decay"])
        batch = spec["batch_size"]
        x = torch.randint(2, spec["max_entities"] + 2, (batch, 4), device="cuda:0")
        x[:, 1:3] = torch.randint(
            spec["max_entities"] + 2,
            spec["max_entities"] + spec["relations"] + 2,
            (batch, 2),
            device="cuda:0",
        )
        pos = torch.tensor([2, 3], device="cuda:0").expand(batch, 2).clone()
        labels = torch.stack([x[:, 3], torch.ones(batch, device="cuda:0", dtype=torch.long)], 1)
        graph = GraphStep(model, optimizer, (x, pos, labels), batch)
        ix = torch.arange(batch, device="cuda:0")
        for _ in range(20):
            graph(ix)
        torch.cuda.synchronize()
        before = model.token.weight.detach().clone()
        begun = time.perf_counter()
        for _ in range(200):
            loss = graph(ix)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begun
        passed = bool(torch.isfinite(loss)) and bool(torch.isfinite(graph.grad_norm))
        passed = passed and not torch.equal(before, model.token.weight)
        if not passed:
            raise RuntimeError("CUDA forward/backward/optimizer validation failed")
        results.append(
            {
                "layers": layers,
                "parameters": sum(p.numel() for p in model.parameters()),
                "ms_per_update": 1000 * seconds / 200,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "actual_forward_backward_optimizer_sync": True,
            }
        )
        del graph, model, optimizer
        torch.cuda.empty_cache()
    write_json(
        out,
        {
            "passed": True,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "models": results,
            "utc": utc(),
        },
    )
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "audit", "preflight"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.action == "preflight":
        preflight(config, args.out)
        return
    spec = next(value for value in config["runs"] if value["name"] == args.run)
    from llm_memory_editability import composition_curves as curves

    if args.action == "train":
        curves.train(spec, args.out, resume=args.resume)
    else:
        curves.audit(spec, args.out)


if __name__ == "__main__":
    main()
