#!/usr/bin/env python3
"""Summarize all eight conditions and verify the actual W&B curve after reload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_memory_editability.experiment_tracking import learning_metrics  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(config, base):
    rows, exposures = {}, []
    for name in config["arms"]:
        path = base / "runs" / name
        if not (path / "complete.json").exists():
            rows[name] = {"state": "incomplete"}
            continue
        complete, metadata = read(path / "complete.json"), read(path / "run.json")
        audit = read(path / "audit.json") if (path / "audit.json").exists() else {}
        with np.load(path / "exposure.npz") as saved:
            exposure_hash = hashlib.sha256(saved["counts"].tobytes()).hexdigest()
        exposures.append(exposure_hash)
        rows[name] = {
            "state": complete["state"],
            "step": complete["step"],
            "independent_reload": audit.get("passed", False),
            "prefix_trace": (path / "prefix-trace.json").exists(),
            "initial_model_sha256": complete["initial_model_sha256"],
            "world_sha256": metadata["world_sha256"],
            "model_sha256": complete["model_sha256"],
            "parameters": metadata["parameters"],
            "executed_layers": metadata["executed_layers"],
            "channel_calls_per_forward": metadata["channel_calls_per_forward"],
            "exposure_sha256": exposure_hash,
            "endpoint": complete["endpoint"],
        }
    all_complete = all(
        row.get("independent_reload") and row.get("step") == 300000 for row in rows.values()
    )
    if exposures and len(set(exposures)) != 1:
        raise ValueError("Sample-wise exposure differs across arms")
    differences = {}
    if all_complete:
        for layers, rounds in ((1, 2), (1, 3), (1, 4), (2, 2)):
            plain, disco = [rows[f"{arch}-l{layers}-r{rounds}"] for arch in ("loop", "discoloop")]
            if plain["initial_model_sha256"] != disco["initial_model_sha256"]:
                raise ValueError("Architecture pair did not start from the same weights")
            differences[f"l{layers}-r{rounds}"] = {
                f"ood_{k}_answer_pp": 100
                * (
                    disco["endpoint"]["metrics"][f"ood_{k}"]["answer_accuracy"]
                    - plain["endpoint"]["metrics"][f"ood_{k}"]["answer_accuracy"]
                )
                for k in (2, 3, 4)
            }
    return {
        "all_eight_reloaded": all_complete,
        "single_development_world": True,
        "arms": rows,
        "discoloop_minus_loop": differences,
    }


def cloud_verify(config, base):
    import wandb

    api = wandb.Api(timeout=60)
    findings = {}
    for name in config["arms"]:
        path = base / "runs" / name
        state = read(base / "tracking-wandb" / name / "state.json")
        identity = f"{state['entity']}/{state['project']}/{state['run_id']}"
        run = api.run(identity)
        metadata = read(path / "run.json")
        local = read(path / "learning.json")
        remote = list(run.scan_history(page_size=1000))
        by_step = {
            int(row["training/step"]): row for row in remote if row.get("training/step") is not None
        }
        comparisons, previous = [], None
        for record in local:
            wanted = learning_metrics(record, previous)
            previous = record
            observed = by_step.get(record["step"], {})
            mismatches = []
            for key, value in wanted.items():
                other = observed.get(key)
                if (
                    other is None
                    or not isinstance(other, int | float)
                    or not math.isclose(value, other, rel_tol=1e-7, abs_tol=1e-7)
                ):
                    mismatches.append(key)
            comparisons.append(
                {"step": record["step"], "checked_fields": len(wanted), "mismatches": mismatches}
            )
        checks = {
            "state_finished": run.state == "finished",
            "spec_exact": dict(run.config["spec"]) == metadata["spec"],
            "world_exact": run.config.get("world_sha256") == metadata["world_sha256"],
            "initial_weights_exact": run.config.get("initial_model_sha256")
            == metadata["initial_model_sha256"],
            "independent_reload_reported": run.summary.get("independently_reloaded") is True,
            "final_step_exact": run.summary.get("scientific_final_step") == 300000,
            "all_learning_fields_match": all(not row["mismatches"] for row in comparisons),
        }
        directory = base / "cloud-verification" / name
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / "config.json", dict(run.config))
        write(directory / "summary.json", dict(run.summary))
        write(directory / "history.json", remote)
        findings[name] = {
            "passed": all(checks.values()),
            "url": run.url,
            "checks": checks,
            "nodes": comparisons,
        }
    receipt = {"passed": all(row["passed"] for row in findings.values()), "arms": findings}
    write(base / "cloud-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()
    config, base = read(args.config), args.base
    result = summarize(config, base)
    write(base / "comparison.json", result)
    lines = [
        "# Layers × recurrence: DiscoLoop and vanilla Loop",
        "",
        "One development world/initialization. All models train atoms plus 2/3/4-hop graphA "
        "compositions. GraphB receives only atomic supervision. Fixed300k endpoint, answer "
        "argmax primary; joint generatedEOS and premise-conditioned results remain in "
        "comparison.json.",
        "",
        "| Arm | Atomic A | Atomic B | B2 | B3 | B4 | Parameters |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["arms"].items():
        if row["state"] != "complete":
            lines.append(f"| {name} | incomplete | | | | | |")
            continue
        metrics = row["endpoint"]["metrics"]
        scores = [
            f"{100 * metrics[key]['answer_accuracy']:.2f}%"
            for key in ("atomic_id", "atomic_ood", "ood_2", "ood_3", "ood_4")
        ]
        lines.append(f"| {name} | " + " | ".join(scores) + f" | {row['parameters']:,} |")
    lines += [
        "",
        "L1R4 and L2R2 both execute four complete blocks; unique parameters and the number "
        "of DiscoLoop channels differ. Equal exposure is not equal compute. A negative result "
        "at this budget is not a depth impossibility theorem. Native prefix readouts describe "
        "decodability, not causal use. Existing historical batches were not restarted.",
        "",
    ]
    (base / "report.md").write_text("\n".join(lines))
    if args.cloud:
        # SDK uploads may finish shortly after the tracker process exits.
        deadline = time.monotonic() + 600
        while not (base / "tracking-completion.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Tracking completion receipt missing")
            time.sleep(10)
        receipt = cloud_verify(config, base)
        if not receipt["passed"]:
            raise RuntimeError("Remote W&B science fields do not match local artifacts")
    write(
        base / "completion-manifest.json",
        {
            "passed": result["all_eight_reloaded"] and (not args.cloud or receipt["passed"]),
            "scientific_updates": 2400000,
            "all_eight_reloaded": result["all_eight_reloaded"],
            "cloud_verified": args.cloud,
        },
    )


if __name__ == "__main__":
    main()
