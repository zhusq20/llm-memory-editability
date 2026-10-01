#!/usr/bin/env python3
"""Sequentially execute an already frozen longer-path configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts", default="docs/development-artifacts/grok-multihop-v1")
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if cfg.get("analysis_only") or "source_lock" not in cfg:
        raise ValueError("Only frozen execution configurations can be queued")
    artifacts = ROOT / args.artifacts
    records = []
    run_ids = list(cfg["runs"]) if args.runs is None else args.runs
    if len(set(run_ids)) != len(run_ids) or any(key not in cfg["runs"] for key in run_ids):
        raise ValueError("Requested queue contains duplicate or unregistered run IDs")
    for run_id in run_ids:
        item = cfg["runs"][run_id]
        spec = {**cfg["base"], **item}
        out = ROOT / cfg["output_root"] / spec["phase"] / run_id
        completed = out / "complete.json"
        if completed.exists():
            if json.loads(completed.read_text())["spec"] != spec:
                raise ValueError("Completed run has a different registered specification")
            records.append({"run": run_id, "state": "already_complete"})
            continue
        record = {"run": run_id, "state": "running", "started_utc": utc()}
        records.append(record)
        status_path = artifacts / f"{cfg['experiment']}-queue.json"
        write_json(status_path, {"config": args.config, "runs": records})
        with (artifacts / f"{run_id}.log").open("x") as output:
            result = subprocess.run(
                [sys.executable, "scripts/run_grok_multihop.py", run_id, "--config", args.config],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update(
            state="complete" if result.returncode == 0 else "failed",
            finished_utc=utc(),
            exit_code=result.returncode,
        )
        write_json(status_path, {"config": args.config, "runs": records})
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
