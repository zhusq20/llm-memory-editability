#!/usr/bin/env python3
"""Run the original frozen factorial CLI with checkpoint I/O moved off-thread."""

import hashlib
import json
import runpy
from pathlib import Path

from knowledge_checkpoint_io import CheckpointTransport


def main():
    extension = Path(__file__).resolve().parent
    manifest = json.loads((extension / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((extension / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Checkpoint extension hash mismatch: {name}")
    from llm_memory_editability import knowledge_method_diagnosis as shared

    original = shared.atomic_torch_save
    transport = CheckpointTransport(original)
    shared.atomic_torch_save = transport.save
    try:
        runpy.run_path("/project/scripts/run_knowledge_method_factorial.py", run_name="__main__")
    finally:
        shared.atomic_torch_save = original
        transport.close()


if __name__ == "__main__":
    main()
