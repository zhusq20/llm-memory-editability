#!/usr/bin/env python3
"""Reuse execution parity/resume checks with atomic plus length-six input rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import source_hash, utc, write_json
from llm_memory_editability.grok_multihop import pack_rows

ROOT = Path(__file__).resolve().parents[1]


def make_table(device, vocab=128):
    rng = np.random.default_rng(97)
    atomic = pack_rows(rng.integers(2, vocab, (127, 3)), 6)
    composite = pack_rows(rng.integers(2, vocab, (129, 6)), 6)
    return tuple(
        torch.as_tensor(np.concatenate([a, b]), device=device)
        for a, b in zip(atomic, composite, strict=True)
    )


def main():
    spec = importlib.util.spec_from_file_location(
        "historical_grok_preflight", ROOT / "scripts/preflight_grok_depth.py"
    )
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.make_table = make_table
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = {
        "started_utc": utc(),
        "length": 6,
        "purpose": "Execution-only random labels; not a learning or research result",
        "gpu": torch.cuda.get_device_name(device),
        "tf32": True,
        "files": source_hash(
            [
                "scripts/preflight_grok_multihop.py",
                "scripts/preflight_grok_depth.py",
                "src/llm_memory_editability/grok_depth.py",
                "src/llm_memory_editability/grok_multihop.py",
            ]
        ),
    }
    for name in ("graph_parity", "resume_equivalence"):
        result[name] = getattr(old, name)(device)
        print(name, result[name], flush=True)
    result["passed"] = True
    result["finished_utc"] = utc()
    write_json(ROOT / "docs/development-artifacts/grok-multihop-v1/preflight.json", result)


if __name__ == "__main__":
    main()
