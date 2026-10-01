#!/usr/bin/env python3
"""Fixed eight-case FP32 replication of every layer and intervention."""

import argparse
import time

import torch
from run_twohop_frozen import ART, DATA, RESULTS, ROOT, Engine, digest, mechanism, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["small", "main"])
    args = parser.parse_args()
    cfg = read(ROOT / "configs/twohop-frozen-v1.json")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = Engine(cfg["models"][args.model], cfg)
    engine.model.float()
    engine.name = args.model + "-fp32"
    cases = read(DATA / "cases.json")
    selected = []
    for dataset in ["mquake", "2wiki"]:
        selected += [
            r
            for r in cases
            if r["dataset"] == dataset and r["split"] == "evaluation" and r["mechanism"]
        ][:4]
    out = RESULTS / engine.name
    out.mkdir(parents=True, exist_ok=True)
    lock = ART / f"precision-lock-{args.model}.json"
    if lock.exists():
        raise FileExistsError(lock)
    write(
        lock,
        dict(
            cases=[dict(dataset=r["dataset"], id=r["id"]) for r in selected],
            dtype="float32",
            weight_note="BF16 weights promoted to FP32; computation changes, no training.",
            selection="First four evaluation mechanism cases per dataset; no outcome filter.",
            runner_sha256=digest(ROOT / "scripts/run_twohop_frozen.py"),
            script_sha256=digest(ROOT / "scripts/check_twohop_frozen_precision.py"),
        ),
    )
    start = time.perf_counter()
    mechanism(engine, selected, RESULTS / args.model / "behavior.jsonl", out / "mechanism.jsonl")
    assert all(not p.requires_grad for p in engine.model.parameters())
    write(
        ART / f"precision-completion-{args.model}.json",
        dict(
            state="complete",
            cases=len(selected),
            seconds=time.perf_counter() - start,
            stats=engine.stats,
            max_memory_bytes=torch.cuda.max_memory_allocated(engine.device),
            files={str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
