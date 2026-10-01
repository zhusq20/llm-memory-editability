#!/usr/bin/env python3
"""Run a frozen, resumable small-model mechanism matrix."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import platform
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch
import transformers

from llm_memory_editability.architecture_bridge import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    ROOT,
    Engine,
    digest,
    prepare,
    read,
    write,
)


def snapshot():
    paths = [
        CONFIG,
        ROOT / "src/llm_memory_editability/architecture_bridge.py",
        Path(__file__),
        ROOT / "tests/test_architecture_bridge.py",
        ART / "preregistration.md",
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    lock = ART / "execution-lock.json"
    if lock.exists():
        if read(lock)["files"] != hashes:
            raise ValueError("Execution source changed after freeze; document an amendment")
        return
    for path in paths:
        destination = ART / "execution-source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(
        lock,
        {
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "files": hashes,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "data_lock_sha256": digest(ART / "data-lock.json"),
        },
    )


def run(args):
    cfg = read(CONFIG)
    lock = read(ART / "data-lock.json")
    if (
        digest(CONFIG) != lock["config_sha256"]
        or digest(DATA / "cases.json") != lock["cases_sha256"]
        or digest(DATA / "learning.json") != lock["learning_sha256"]
    ):
        raise ValueError("Frozen config or cases changed")
    out = RESULTS / args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out / "worker.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot()
        engine = Engine(args.model, args.device)
        started = time.monotonic()
        cases = [r for r in read(DATA / "cases.json") if args.split in ("all", r["split"])]
        if args.limit:
            cases = cases[: args.limit]
        if args.phase in ["all", "interventions"]:
            for index, row in enumerate(cases):
                destination = out / "cases" / f"{row['dataset']}-{row['id']}.json"
                if destination.exists():
                    continue
                write(
                    out / "status.json",
                    {
                        "state": "running",
                        "phase": "interventions",
                        "current": row["id"],
                        "index": index,
                    },
                )
                try:
                    result = engine.case(row)
                    write(destination, result)
                except Exception:
                    write(
                        out / "failure.json",
                        {"case": row["id"], "traceback": traceback.format_exc()},
                    )
                    raise
                print(
                    json.dumps(
                        {
                            "model": args.model,
                            "case": row["id"],
                            "seconds": result["seconds"],
                            "index": index,
                            "unassisted_em": result["unassisted"]["em"],
                        }
                    ),
                    flush=True,
                )
        if args.phase in ["all", "learning"] and not args.limit and args.split == "all":
            for episode in range(cfg["local_learning"]["episodes"]):
                folder = out / "learning" / str(episode)
                destination = folder / "complete.json"
                if destination.exists():
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                write(
                    out / "status.json",
                    {"state": "running", "phase": "learning", "episode": episode},
                )
                result = engine.learning(episode, folder)
                write(destination, result)
                print(
                    json.dumps(
                        {"model": args.model, "episode": episode, "seconds": result["seconds"]}
                    ),
                    flush=True,
                )
        write(
            out / "last-run.json",
            {
                "seconds": time.monotonic() - started,
                "stats": dict(engine.stats),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(engine.device),
                "device": str(engine.device),
                "gpu_name": torch.cuda.get_device_name(engine.device),
                "parameters": sum(p.numel() for p in engine.model.parameters()),
                "flops_estimate_method": (
                    "Approximate 2*P*input_tokens forward plus 4*P*backward_input_tokens; "
                    "attention/cache overhead excluded, backward estimate is a full-gradient "
                    "upper proxy despite frozen upstream parameters."
                ),
                "forward_flops_estimate": 2
                * sum(p.numel() for p in engine.model.parameters())
                * engine.stats["input_tokens"],
                "backward_flops_upper_proxy": 4
                * sum(p.numel() for p in engine.model.parameters())
                * engine.stats["backward_input_tokens"],
                "args": vars(args),
            },
        )
        write(
            out / "status.json",
            {
                "state": "complete_requested_phase",
                "phase": args.phase,
                "split": args.split,
                "limit": args.limit,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "qwen35", "prepare"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--phase", choices=["all", "interventions", "learning"], default="all")
    parser.add_argument("--split", choices=["all", "development", "evaluation"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.model == "prepare":
        print(prepare())
    else:
        run(args)


if __name__ == "__main__":
    main()
