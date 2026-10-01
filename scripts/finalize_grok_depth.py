#!/usr/bin/env python3
"""Write a completion manifest only when all required runs and endpoint audits pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from report_grok_depth import exposure, load_runs, read_json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/development-artifacts/grok-depth-v1"
RESULTS = ROOT / "results/grok-depth-v1"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inspect_completion():
    config_path = ROOT / "configs/grok-depth-v1.json"
    config = read_json(config_path)
    lock_path = ARTIFACTS / "confirmation-lock.json"
    lock = read_json(lock_path)
    require(lock is not None, "Confirmation source lock is missing.")
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        require(path.exists() and digest(path) == expected, f"Frozen source changed: {relative}")
    require(lock["primary_runs"] == config["primary_runs"], "Primary registry differs from lock.")
    require(
        lock["conditional_wide_runs"] == config["conditional_wide_runs"],
        "Conditional registry differs from lock.",
    )
    development = sorted(
        name for name, spec in config["runs"].items() if spec["phase"] == "development"
    )
    primary = config["primary_runs"]
    conditional = config["conditional_wide_runs"]
    require(
        (len(development), len(primary), len(conditional)) == (2, 12, 6),
        "Expected registration is two development, twelve primary, six conditional runs.",
    )
    require(len(set(development + primary + conditional)) == 20, "Run registries overlap.")
    decision_path = ARTIFACTS / "wide-control-decision.json"
    decision = read_json(decision_path)
    if decision is not None:
        require(type(decision.get("execute")) is bool, "Wide-control execute must be a boolean.")
    execute_wide = decision is not None and decision["execute"] is True
    expected = development + primary + (conditional if execute_wide else [])
    groups = {name: "development" for name in development}
    groups.update({name: "primary" for name in primary})
    groups.update({name: "conditional_wide" for name in conditional if execute_wide})
    missing = []
    for name in expected:
        directory = RESULTS / config["runs"][name]["phase"] / name
        complete = read_json(directory / "complete.json")
        if not complete or complete.get("endpoint", {}).get("step") != config["base"]["steps"]:
            missing.append(name)
    require(not missing, "Missing completed endpoints: " + ", ".join(missing))
    require(decision is not None, "Wide-control decision is still pending.")

    audits, audited_runs = {}, {}
    for phase, names in (
        ("development", development),
        ("confirmation", primary + (conditional if execute_wide else [])),
    ):
        audit_path = ARTIFACTS / f"endpoint-audit-{phase}.json"
        audit = read_json(audit_path)
        require(
            audit is not None and audit.get("passed") is True,
            f"Formal {phase} endpoint audit has not passed.",
        )
        require(audit.get("phase") == phase, f"Audit phase mismatch: {phase}")
        require(
            set(audit["matrix"]["expected_from_frozen_configs"]) == set(names)
            and set(audit["matrix"]["observed"]) == set(names)
            and not audit["matrix"].get("missing")
            and not audit["matrix"].get("unexpected"),
            f"Audit matrix mismatch: {phase}",
        )
        require(
            {row["run_id"] for row in audit["runs"]} == set(names),
            f"Audited run list mismatch: {phase}",
        )
        require(
            all(row["state"] == "passed" for row in audit["runs"]),
            f"At least one {phase} run did not pass audit.",
        )
        if phase == "confirmation":
            control = audit.get("conditional_wide_control", {})
            require(
                control.get("decision_sha256") == digest(decision_path)
                and control.get("execute") is execute_wide,
                "Confirmation audit predates the current wide-control decision.",
            )
        audits[phase] = {
            "path": str(audit_path),
            "sha256": digest(audit_path),
            "device": audit["device"],
            "finished_utc": audit["finished_utc"],
            "auditor_sha256": audit["auditor_sha256"],
            "passed": True,
        }
        audited_runs.update({row["run_id"]: row for row in audit["runs"]})

    loaded = {run["run_id"]: run for run in load_runs(RESULTS)}
    require(
        set(loaded) == set(expected), "Observed run directories differ from the required matrix."
    )
    rows, detailed = [], []
    for name in expected:
        run = loaded[name]
        directory = Path(run["directory"])
        metadata = read_json(directory / "metadata.json")
        complete = read_json(directory / "complete.json")
        audit = audited_runs[name]
        spec = {**config["base"], **config["runs"][name]}
        require(
            run["status"] == "complete" and metadata["spec"] == complete["spec"] == spec,
            f"Run completion or specification mismatch: {name}",
        )
        learning = read_json(directory / "learning.json")
        require(
            [point["step"] for point in learning] == spec["nodes"],
            f"Registered evaluation nodes are missing, duplicated or reordered: {name}",
        )
        endpoint = learning[-1]
        require(complete["endpoint"] == endpoint, f"Completion endpoint differs from log: {name}")
        require(
            audit["spec"] == spec and audit["checkpoint_step"] == spec["steps"],
            f"Audited specification or checkpoint differs: {name}",
        )
        require(
            digest(directory / "latest.pt") == audit["checkpoint_sha256"],
            f"Checkpoint changed after formal audit: {name}",
        )
        require(
            run["dataset_sha256"] == audit["world_audit"]["dataset_sha256"],
            f"Dataset differs from formal audit: {name}",
        )
        for original, expected_hash in metadata["files"].items():
            relative = (
                Path(original).relative_to(ROOT) if Path(original).is_absolute() else Path(original)
            )
            copied = directory / "source" / relative
            require(
                copied.exists()
                and digest(copied) == expected_hash
                and audit["source_hashes"].get(relative.as_posix()) == expected_hash,
                f"Historical source differs from metadata/audit: {name}, {relative}",
            )
            if spec["phase"] == "confirmation":
                require(
                    lock["files"].get(relative.as_posix()) == expected_hash,
                    f"Confirmation snapshot differs from source lock: {name}, {relative}",
                )
        require(
            complete["parameters"] == endpoint["parameters"] == audit["parameters"],
            f"Parameter count differs: {name}",
        )
        require(
            endpoint["counts"] == audit["exposure"]["counts"],
            f"Endpoint exposure differs from audit: {name}",
        )
        start = datetime.fromisoformat(metadata["started_utc"])
        finish = datetime.fromisoformat(complete["finished_utc"])
        threshold = run["t90"]
        row = {
            "run_id": name,
            "group": groups[name],
            "phase": spec["phase"],
            "world_seed": spec["world_seed"],
            "initialization": spec["initialization"],
            "layers": spec["layers"],
            "width": spec["width"],
            "parameters": endpoint["parameters"],
            "budget_steps": spec["steps"],
            "actual_steps": endpoint["step"],
            "evaluation_nodes": len(learning),
            "examples": endpoint["examples"],
            **exposure(endpoint, run["data_counts"]),
            "effective_input_tokens": endpoint["effective_input_tokens"],
            "supervised_tokens": endpoint["supervised_tokens"],
            "estimated_training_flops": endpoint["estimated_training_flops"],
            "training_seconds": complete["training_seconds"],
            "evaluation_seconds": complete["evaluation_seconds"],
            "last_capture_seconds": endpoint["capture_seconds"],
            "started_utc": metadata["started_utc"],
            "finished_utc": complete["finished_utc"],
            "wall_seconds": (finish - start).total_seconds(),
            "t90_reached": threshold["reached"],
            "t90_step": threshold["step"],
            "t90_confirmed_at_step": threshold["confirmed_at_step"],
            "t90_examples": threshold["examples"],
            "t90_atomic_exposures": threshold["atomic_exposures"],
            "t90_estimated_training_flops": threshold["estimated_training_flops"],
        }
        for split in ("atomic", "train_composite", "test_composite", "ood_composite", "two_calls"):
            audited_scores = (
                audit["two_calls"] if split == "two_calls" else audit["endpoint"][split]["scores"]
            )
            require(
                all(
                    endpoint[split][key] == audited_scores[key]
                    for key in ("n", "accuracy", "answer_accuracy")
                ),
                f"Endpoint scores differ from formal audit: {name}, {split}",
            )
            row[f"{split}_accuracy"] = endpoint[split]["accuracy"]
            row[f"{split}_n"] = endpoint[split]["n"]
        require(
            run["atomic_split_endpoint"]["available"], f"Atomic split scores unavailable: {name}"
        )
        for split in ("id_atomic", "ood_atomic"):
            values = run["atomic_split_endpoint"][split]
            row[f"{split}_accuracy"], row[f"{split}_n"] = values["accuracy"], values["n"]
        segments = []
        prev_train, prev_eval, prev_step = 0.0, 0.0, 0
        for point in learning:
            train_seconds = point["training_seconds"] - prev_train
            eval_seconds = point["evaluation_seconds"] - prev_eval
            require(
                train_seconds >= 0 and eval_seconds >= 0, f"Cumulative timers decreased: {name}"
            )
            segments.append(
                {
                    "from_step": prev_step,
                    "to_step": point["step"],
                    "evaluation_started_utc": point["utc"],
                    "training_seconds": train_seconds,
                    "evaluation_seconds": eval_seconds,
                }
            )
            prev_train, prev_eval, prev_step = (
                point["training_seconds"],
                point["evaluation_seconds"],
                point["step"],
            )
        require(
            prev_train == complete["training_seconds"]
            and prev_eval == complete["evaluation_seconds"],
            f"Timer totals differ: {name}",
        )
        rows.append(row)
        detailed.append(
            {
                **row,
                "t90": threshold,
                "timing_segments": segments,
                "dataset_sha256": run["dataset_sha256"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "spec_sha256": hashlib.sha256(
                    json.dumps(spec, sort_keys=True).encode()
                ).hexdigest(),
                "complete_sha256": digest(directory / "complete.json"),
                "learning_sha256": digest(directory / "learning.json"),
            }
        )
    earliest = min(datetime.fromisoformat(row["started_utc"]) for row in rows)
    latest = max(datetime.fromisoformat(row["finished_utc"]) for row in rows)
    manifest = {
        "state": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "required_run_count": len(expected),
        "completed_run_count": len(rows),
        "expected_runs": expected,
        "conditional_wide_execute": execute_wide,
        "decision_sha256": digest(decision_path),
        "source_lock_sha256": digest(lock_path),
        "source_lock_verified": True,
        "formal_audits": audits,
        "independent_worlds": {
            group: sorted({row["world_seed"] for row in rows if row["group"] == group})
            for group in ("development", "primary", "conditional_wide")
        },
        "independent_world_counts": {
            group: len({row["world_seed"] for row in rows if row["group"] == group})
            for group in ("development", "primary", "conditional_wide")
        },
        "totals": {
            field: sum(row[field] for row in rows)
            for field in (
                "actual_steps",
                "evaluation_nodes",
                "examples",
                "effective_input_tokens",
                "supervised_tokens",
                "estimated_training_flops",
                "training_seconds",
                "evaluation_seconds",
            )
        },
        "started_utc": earliest.isoformat(),
        "finished_utc": latest.isoformat(),
        "elapsed_wall_seconds": (latest - earliest).total_seconds(),
        "timing_note": "Training/evaluation totals sum run timers, including concurrent runs; "
        "they are not wall time. Segment timings are differences of cumulative node timers. "
        "last_capture_seconds records the last capture, not a sum over resumed processes.",
        "t90_note": "First of three consecutive registered ID exact-accuracy nodes >=0.9; "
        "null means not established within the completed budget. Seeds are repeats within worlds.",
        "runs": detailed,
        "finalizer_sha256": digest(Path(__file__)),
    }
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="Validate without writing artifacts."
    )
    args = parser.parse_args()
    try:
        manifest, rows = inspect_completion()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"state": "refused", "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc
    if args.check_only:
        print(json.dumps({"state": "ready", "runs": len(rows), "artifacts_written": False}))
        return
    table_path = ARTIFACTS / "endpoint-table.csv"
    table_tmp = table_path.with_suffix(".csv.tmp")
    with table_tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_tmp.replace(table_path)
    manifest["endpoint_table_sha256"] = digest(table_path)
    manifest_path = ARTIFACTS / "completion-manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    manifest_tmp.replace(manifest_path)
    print(
        json.dumps(
            {
                "state": "complete",
                "runs": len(rows),
                "manifest": str(manifest_path),
                "endpoint_table": str(table_path),
            }
        )
    )


if __name__ == "__main__":
    main()
