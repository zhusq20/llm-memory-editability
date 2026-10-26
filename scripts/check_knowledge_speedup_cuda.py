#!/usr/bin/env python3
"""Bound shared admission with the original full-size worst-input checks."""

import json
from pathlib import Path

import torch
from check_knowledge_method_factorial_cuda import exercise
from knowledge_checkpoint_io import CheckpointTransport

from llm_memory_editability import knowledge_method_diagnosis as shared


def main():
    torch.cuda.set_per_process_memory_fraction(0.44)
    config = json.loads(
        Path("/project/configs/knowledge-method-factorial-1000-v1.json").read_text()
    )
    out = Path("/batch/speedup-20261022/engineering")
    import sys

    arm = sys.argv[1]
    result = exercise(config, out, arm, pressure=True)
    # Validate ownership of CUDA tensors and RNG while the background writer runs.
    live = torch.arange(1024, device="cuda", dtype=torch.float32)
    expected = live.cpu()
    rng = torch.cuda.get_rng_state_all()
    transport = CheckpointTransport(shared.atomic_torch_save, 32 * 1024 * 1024)
    path = out / arm / "resume.pt"
    try:
        transport.save({"live": live, "rng": rng}, path)
        live.add_(1)
        transport.drain()
        reloaded = torch.load(path, weights_only=False)
        assert torch.equal(reloaded["live"], expected)
        assert all(
            torch.equal(a, b) for a, b in zip(rng, torch.cuda.get_rng_state_all(), strict=True)
        )
    finally:
        transport.close()
    result["cuda_async_snapshot_exact"] = True
    result["cuda_rng_unchanged"] = True
    shared.write(out / f"{arm}.json", result)
    print(json.dumps({"arm": arm, "passed": True, "updates": result["updates"]}), flush=True)


if __name__ == "__main__":
    main()
