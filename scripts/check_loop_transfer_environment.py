#!/usr/bin/env python3
"""Target-machine computation and exact old-prediction replay; no training."""

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import run_loop_robustness as runner
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.16, 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    versions = {
        k: importlib.metadata.version(k)
        for k in ("torch", "transformers", "numpy", "wandb", "huggingface-hub")
    }
    import wandb

    api = wandb.Api(timeout=60)
    assert api.viewer is not None
    # Check authentication and access without creating an extra scientific run.
    list(api.runs("zsqzz/llm-memory-editability", per_page=1)[:1])
    (args.output / "requirements.txt").write_text(
        subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    )
    plan = json.loads(Path("configs/loop-invariant-robustness-v1.json").read_text())
    job = plan["jobs"][args.job]
    config = json.loads(Path(job["config"]).read_text())
    for weights in ["raw", "ema"] if job["arm"] == "invariant" else ["ema"]:
        runner.depth(
            config,
            job["arm"],
            Path(job["run"]),
            args.output / weights,
            [{"name": "full-r8", "rows": None, "rounds": 8}],
            weights,
            verify_existing=True,
        )
    (args.output / "complete.json").write_text(
        json.dumps(
            {
                "passed": True,
                "versions": versions,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "matrix_backward_sync": True,
                "old_r4_r8_exact_replay": True,
                "new_training_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
