"""Real-size CUDA forward/backward preflight, without scientific outcome claims."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch

from llm_memory_editability import parametric_architecture as architecture
from llm_memory_editability import parametric_architecture_train as training
from llm_memory_editability.grokking_reproduction import model_digest
from llm_memory_editability.realworld_composition import BatchStream
from llm_memory_editability.realworld_composition_data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = next(s for s in config["runs"] if s["architecture"] == args.architecture)
    spec = {**spec, "name": args.architecture + "-engineering", "steps": 4}
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    x = torch.randn(128, 128, device=device, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = (x @ x).square().mean()
    y.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all()
    model = architecture.construct(config["model"], spec, device)
    optimizer = architecture.optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    expected = {"D8": 96088320, "M8": 209500416, "W8": 209408256, "IHC8": 96530848, "HC8": 97317552}
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == expected[args.architecture], (parameters, expected[args.architecture])
    write_json(
        args.out / "run.json",
        {
            "spec": spec,
            "model": config["model"],
            "world_sha256": sha256(config["data_file"]),
            "initial_model_sha256": model_digest(model),
            "parameters": parameters,
            "pid": os.getpid(),
            "gpu": int(os.environ["PHYSICAL_GPU"]),
            "gpu_name": torch.cuda.get_device_name(),
            "phase": "engineering",
            "tracking_group": "parametric-architecture-engineering-v1",
            "job_type": "preflight",
        },
    )
    write_json(args.out / "learning.json", [])
    data = json.loads(Path(config["data_file"]).read_text())
    records = data["atoms"] + data["train_compositions"]
    stream = BatchStream(len(records), spec["sampling_seed"])
    torch.manual_seed(spec["dropout_seed"])
    history = []
    for step in range(1, 5):
        selected = [records[i] for i in stream.batch(spec["batch_size"])]
        if step == 4:
            # Allocation stress only: 64 valid input positions and the longest
            # development target. This is never added to scientific datasets.
            worst = max(records, key=lambda r: len(r["encoded"]["target"]))
            selected = [copy.deepcopy(worst) for _ in range(spec["batch_size"])]
            for record in selected:
                record["encoded"]["input"] += [50256] * (64 - len(record["encoded"]["input"]))
        torch.cuda.synchronize()
        start = time.perf_counter()
        values = training.update(model, optimizer, selected, spec, device, 50256)
        torch.cuda.synchronize()
        history.append(
            {
                "step": step,
                **values,
                "update_seconds": time.perf_counter() - start,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                "metrics": {},
            }
        )
        write_json(args.out / "learning.json", history)
        print(
            json.dumps({k: v for k, v in history[-1].items() if not isinstance(v, dict)}),
            flush=True,
        )
    report = {
        "passed": True,
        "cuda_bf16_matmul_backward_sync": True,
        "architecture": args.architecture,
        "parameters": parameters,
        "microbatch_size": spec["microbatch_size"],
        "effective_batch_size": 512,
        "update_seconds": [h["update_seconds"] for h in history],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "scope": (
            "Three development engineering updates plus one 64-token allocation stress; "
            "no scientific results"
        ),
    }
    write_json(args.out / "profile.json", report)
    write_json(args.out / "complete.json", report)


if __name__ == "__main__":
    main()
