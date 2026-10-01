#!/usr/bin/env python3
"""Create a completed-batch receipt only after all registered endpoints are audited."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from llm_memory_editability.grok_depth import source_hash, utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def collect(config_path, audit_path, expected_count):
    config = json.loads((ROOT / config_path).read_text())
    audit = json.loads((ROOT / audit_path).read_text())
    if not audit["passed"] or audit["device"] is None:
        raise ValueError("Finalization requires a passing GPU reload audit")
    if len(config["runs"]) != expected_count:
        raise ValueError("Unexpected batch size")
    audited = {run["run"] for run in audit["runs"] if run["passed"]}
    if audited != set(config["runs"]):
        raise ValueError("Reload audit does not cover the exact registered matrix")
    runs = []
    for run_id, item in config["runs"].items():
        spec = {**config["base"], **item}
        directory = ROOT / config["output_root"] / spec["phase"] / run_id
        meta = json.loads((directory / "metadata.json").read_text())
        result = json.loads((directory / "complete.json").read_text())
        environment = json.loads((directory / "environment.json").read_text())
        learning = json.loads((directory / "learning.json").read_text())
        row = result["endpoint"]
        if result["spec"] != spec or row["step"] != spec["steps"]:
            raise ValueError("Recorded run differs from the final analysis manifest")
        runs.append(
            {
                "run_id": run_id,
                "phase": spec["phase"],
                "steps": row["step"],
                "examples": row["examples"],
                "effective_input_tokens": row["effective_input_tokens"],
                "estimated_training_flops": row["estimated_training_flops"],
                "training_seconds": result["training_seconds"],
                "evaluation_seconds": result["evaluation_seconds"],
                "capture_seconds": row["capture_seconds"],
                "evaluation_nodes": len(learning),
                "visible_devices": meta["visible_devices"],
                "gpu": environment["gpu"],
                "started_utc": meta["started_utc"],
                "finished_utc": result["finished_utc"],
                "execution_files": meta["files"],
                "receipt_hash": source_hash([directory / "complete.json"]),
            }
        )
    names = (
        "steps",
        "examples",
        "effective_input_tokens",
        "estimated_training_flops",
        "training_seconds",
        "evaluation_seconds",
        "capture_seconds",
        "evaluation_nodes",
    )
    return {
        "config": config_path,
        "audit": audit_path,
        "run_count": len(runs),
        "totals": {name: sum(run[name] for run in runs) for name in names},
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-config", default="configs/grok-multihop-analysis-v1.json")
    parser.add_argument(
        "--development-audit",
        default="docs/development-artifacts/grok-multihop-v1/all-development-endpoint-audit.json",
    )
    parser.add_argument("--confirmation-config")
    parser.add_argument("--confirmation-audit")
    args = parser.parse_args()
    groups = {"development": collect(args.development_config, args.development_audit, 7)}
    if bool(args.confirmation_config) != bool(args.confirmation_audit):
        raise ValueError("Confirmation config and audit must be supplied together")
    if args.confirmation_config:
        groups["confirmation"] = collect(args.confirmation_config, args.confirmation_audit, 4)
    runs = [run for group in groups.values() for run in group["runs"]]
    start = min(datetime.fromisoformat(run["started_utc"]) for run in runs)
    end = max(datetime.fromisoformat(run["finished_utc"]) for run in runs)
    files = [
        Path(__file__).resolve(),
        ROOT / "scripts/report_grok_multihop.py",
        ROOT / "scripts/audit_grok_multihop.py",
        ROOT / "scripts/execute_grok_multihop.py",
        ROOT / args.development_config,
        ROOT / args.development_audit,
        ROOT / "docs/development-artifacts/grok-multihop-v1/preflight.json",
        ROOT / "docs/development-artifacts/grok-multihop-v1/development-lock.json",
        ROOT / "docs/development-artifacts/grok-multihop-v1/coverage-development-lock.json",
        ROOT / "docs/development-artifacts/grok-multihop-v1/shallow-development-lock.json",
        ROOT / "tests/test_grok_multihop.py",
    ]
    if args.confirmation_config:
        files.extend([ROOT / args.confirmation_config, ROOT / args.confirmation_audit])
        files.append(
            ROOT / "docs/development-artifacts/grok-multihop-v1/boundary-confirmation-lock.json"
        )
    write_json(
        ROOT / "docs/development-artifacts/grok-multihop-v1/completion-manifest.json",
        {
            "finished_utc": utc(),
            "status": "complete",
            "groups": groups,
            "run_count": len(runs),
            "first_run_metadata_utc": start.isoformat(),
            "last_run_complete_utc": end.isoformat(),
            "run_utc_span_seconds": (end - start).total_seconds(),
            "wall_time_scope": (
                "First launch metadata to last training completion; "
                "excludes prior implementation and later auditing"
            ),
            "cost_scope": "Only this longer-path batch; no historical or concurrent two-hop runs",
            "verification_files": source_hash(files),
            "CPU_contract_tests_passed": 14,
            "GPU_execution_preflight": "preflight.json",
        },
    )


if __name__ == "__main__":
    main()
