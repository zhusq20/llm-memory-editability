#!/usr/bin/env python3
"""Verify the complete registered loop study on CPU and issue its completion receipt.

--check-only reports every missing artifact without writing a completion marker.
GPU reload receipts are reused; no model evaluation or training is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import postprocess_grok_loop as post  # noqa: E402
from audit_grok_loop import audit_sampling_pairs, independent_flops_per_step  # noqa: E402

ARTIFACT = Path("docs/development-artifacts/grok-loop-v1")
CONFIGS = [
    f"configs/grok-loop-{phase}-v1.json" for phase in ("development", "sensitivity", "confirmation")
]
PHASE_COUNTS = {"development": 15, "development-sensitivity": 6, "confirmation": 90}
TOTAL_FIELDS = (
    "steps",
    "evaluation_nodes",
    "examples",
    "atomic_examples",
    "composite_examples",
    "effective_input_tokens",
    "supervised_tokens",
    "estimated_training_flops",
    "training_seconds",
    "evaluation_seconds",
)
require, read = post.require, post.read


class Verification:
    def __init__(self):
        self.missing, self.errors, self.hashes = [], [], {}
        self._digests = {}

    def digest(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = self._digests.get(path)
        if previous is None or previous[0] != signature:
            self._digests[path] = (signature, post.digest(path))
        return self._digests[path][1]

    def evidence(self, path):
        self.hashes[str(Path(path).resolve())] = self.digest(path)

    def attempt(self, kind, run, path, callback):
        path = Path(path)
        if not path.exists():
            self.missing.append({"kind": kind, "run": run, "path": str(path)})
            return None
        try:
            result = callback()
            self.evidence(path)
            return result
        except Exception as exc:
            self.errors.append({"kind": kind, "run": run, "path": str(path), "error": repr(exc)})
            return None


def verify_lock(path, verification, current=True):
    lock = read(path)
    snapshot = path.with_name(path.stem + "-source")
    require(bool(lock["files"]), f"Empty source lock: {path}")
    for relative, expected in lock["files"].items():
        current_path, frozen = ROOT / relative, snapshot / relative
        if current:
            require(
                verification.digest(current_path) == expected,
                f"Locked source changed: {current_path}",
            )
        require(verification.digest(frozen) == expected, f"Source-lock snapshot changed: {frozen}")
    if "previous_lock" in lock:
        require(
            verification.digest(ROOT / lock["previous_lock"]) == lock["previous_lock_sha256"],
            "Previous analysis lock identity changed",
        )
    return lock


def verify_training(job, lock, verification):
    directory, spec = Path(job["directory"]), job["spec"]
    require(post.training_complete(job), f"Training incomplete: {directory}")
    metadata, complete, history = (
        read(directory / name) for name in ("metadata.json", "complete.json", "learning.json")
    )
    require(metadata["spec"] == spec, "Training metadata specification mismatch")
    require(lock is not None, "Source lock has not passed verification")
    for original, expected in metadata["files"].items():
        original = Path(original)
        relative = original.relative_to(ROOT) if original.is_absolute() else original
        require(
            lock["files"].get(relative.as_posix()) == expected,
            f"Run source differs from lock: {relative}",
        )
        require(
            verification.digest(ROOT / relative) == expected,
            f"Current run source changed: {relative}",
        )
        require(
            verification.digest(directory / "source" / relative) == expected,
            f"Run source snapshot changed: {relative}",
        )
    require(
        [row["step"] for row in history] == spec["nodes"],
        "Missing, repeated or reordered registered nodes",
    )
    require(
        complete["endpoint"] == history[-1], "Completion endpoint differs from learning history"
    )
    previous_train, previous_eval = 0.0, 0.0
    for row in history:
        step, batch, atoms = row["step"], spec["batch_size"], spec["n_atomic_per_batch"]
        require(
            row["counts"] == {"atomic": step * atoms, "composite": step * (batch - atoms)},
            "Node sampling counts differ",
        )
        require(
            row["examples"] == step * batch and row["supervised_tokens"] == 2 * step * batch,
            "Node examples or supervised tokens differ",
        )
        require(
            row["effective_input_tokens"]
            == step * (atoms * 3 + (batch - atoms) * (spec["hops"] + 2)),
            "Node effective tokens differ",
        )
        require(
            row["estimated_training_flops"] == step * independent_flops_per_step(spec),
            "Node FLOPs differ",
        )
        require(
            math.isfinite(row["training_seconds"]) and row["training_seconds"] >= previous_train,
            "Invalid training timer",
        )
        require(
            math.isfinite(row["evaluation_seconds"]) and row["evaluation_seconds"] >= previous_eval,
            "Invalid evaluation timer",
        )
        previous_train, previous_eval = row["training_seconds"], row["evaluation_seconds"]
        require(
            (directory / f"predictions-{step:07d}.npz").stat().st_size > 0,
            "Missing node prediction file",
        )
    for step in spec["weight_nodes"]:
        require(
            (directory / f"weights-{step:07d}.pt").stat().st_size > 0,
            "Missing registered saved-weight node",
        )
    require(
        previous_train == complete["training_seconds"]
        and previous_eval == complete["evaluation_seconds"],
        "Final timing differs from learning history",
    )
    checkpoint = directory / f"weights-{spec['steps']:07d}.pt"
    bound = {
        **job,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": verification.digest(checkpoint),
        "training_complete_sha256": verification.digest(directory / "complete.json"),
    }
    if spec["phase"] == "development":
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(
            state["spec"] == spec and state["step"] == spec["steps"],
            "Development checkpoint identity differs",
        )
        del state
    row = history[-1]
    start, finish = (
        datetime.fromisoformat(metadata["started_utc"]),
        datetime.fromisoformat(complete["finished_utc"]),
    )
    require(
        start.tzinfo is not None and finish.tzinfo is not None and finish >= start,
        "Invalid UTC timing span",
    )
    verification.evidence(directory / "metadata.json")
    verification.evidence(directory / "learning.json")
    environment = read(directory / "environment.json")
    cost = {
        "run": job["run"],
        "phase": spec["phase"],
        "spec": spec,
        **{
            name: row[name]
            for name in (
                "examples",
                "effective_input_tokens",
                "supervised_tokens",
                "estimated_training_flops",
            )
        },
        "steps": spec["steps"],
        "evaluation_nodes": len(history),
        "atomic_examples": row["counts"]["atomic"],
        "composite_examples": row["counts"]["composite"],
        "training_seconds": complete["training_seconds"],
        "evaluation_seconds": complete["evaluation_seconds"],
        "started_utc": metadata["started_utc"],
        "finished_utc": complete["finished_utc"],
        "visible_devices": metadata["visible_devices"],
        "gpu": environment["gpu"],
        "complete_sha256": bound["training_complete_sha256"],
        "checkpoint_sha256": bound["checkpoint_sha256"],
        "dataset_sha256": read(directory / "world-metadata.json")["dataset_sha256"],
    }
    return bound, cost


def verify_audit_run(audit, job, cost):
    require(
        audit["run"] == job["run"] and audit["passed"] is True, "GPU audit run mismatch or failure"
    )
    error = audit["reload_max_nll_absolute_error"]
    require(
        error is not None and math.isfinite(error) and 0 <= error <= 1e-5,
        "GPU reload absent or failed",
    )
    require(
        bool(audit["checks"]) and all(item["passed"] is True for item in audit["checks"]),
        "GPU audit checks not all passing",
    )
    condition = {
        k: v for k, v in job["spec"].items() if k not in {"architecture", "layers", "repeats"}
    }
    require(audit["sampling_condition"] == condition, "GPU audit sampling specification mismatch")
    require(audit["dataset_sha256"] == cost["dataset_sha256"], "GPU audit dataset mismatch")
    require(
        audit["learning_nodes_checked"] == cost["evaluation_nodes"], "GPU audit node count mismatch"
    )
    require(
        audit["endpoint_counts"]
        == {"atomic": cost["atomic_examples"], "composite": cost["composite_examples"]},
        "GPU audit exposure mismatch",
    )
    return {
        **audit,
        "spec": job["spec"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "complete_sha256": job["training_complete_sha256"],
    }


def verify_development_audit(path, expected):
    result = read(path)
    require(
        result["passed"] is True
        and str(result["device"]).startswith("cuda")
        and result.get("gpu")
        and result.get("auditor_source"),
        "Development GPU audit missing provenance or failed",
    )
    runs = result["runs"]
    require(
        len(runs) == len(expected) and {r["run"] for r in runs} == set(expected),
        "Development audit does not cover exact registration",
    )
    return {run["run"]: run for run in runs}


def verify_mechanism(path, job, split, verification):
    post.validate_mechanism(path, job, split)
    summary, metadata = read(path / "summary.json"), read(path / "metadata.json")
    checks = summary["engineering_checks"]
    for name in (
        "identity_all_layers_components",
        "first_layer_same_r1_both_equals_full",
        "last_layer_r1_patch_cannot_change_later_answer_or_eos",
        "eos_uses_generated_answer_and_reapplies_same_patch",
    ):
        require(checks.get(name) is True, f"Mechanism engineering check failed: {name}")
    native = checks["native_trace"]
    require(
        native["native_comparison_performed"] is True
        and native["native_max_logit_delta"] <= 1e-5
        and native["prefix_only_vs_full_all_layers_max_delta"] <= 1e-5,
        "Mechanism native/prefix check failed",
    )
    require(
        summary["executed_depth"] == job["spec"]["layers"] * job["spec"]["repeats"],
        "Mechanism executed depth differs",
    )
    require(
        checks["donor_nonpadding_tokens"] == 2 and checks["mlp_recomputed_after_mixing"] is False,
        "Mechanism intervention changed",
    )
    require(
        metadata["provenance"]["checkpoint_step"] == job["spec"]["steps"],
        "Mechanism checkpoint step differs",
    )
    for original, expected in metadata["analysis_source_hashes"].items():
        original = Path(original)
        relative = original.relative_to(ROOT) if original.is_absolute() else original
        require(
            verification.digest(path / "analysis-source" / relative) == expected,
            f"Mechanism source snapshot changed: {relative}",
        )
    for name in ("metadata.json", "status.json"):
        verification.evidence(path / name)
    return {
        "run": job["run"],
        "split": split,
        "path": str(path),
        "checkpoint_sha256": job["checkpoint_sha256"],
        "passed": True,
    }


def recurrence_catalog(roots, verification):
    found = defaultdict(list)
    for root in sorted(set(roots)):
        for path in sorted(root.glob("recurrence-*/summary.json")):
            try:
                result = read(path)
                for run in result.get("runs", []):
                    found[run["run"]].append((path, result, run))
            except Exception as exc:
                verification.errors.append(
                    {"kind": "recurrence_summary", "path": str(path), "error": repr(exc)}
                )
    return found


def verify_recurrence(path, report, run, job, verification):
    require(
        report.get("finished_utc") and report["repeats"] == post.REPEATS,
        "Recurrence scan incomplete or counts changed",
    )
    require(
        run["run"] == job["run"]
        and run["spec"] == job["spec"]
        and run["checkpoint"] == {job["checkpoint"]: job["checkpoint_sha256"]},
        "Recurrence checkpoint/spec differs",
    )
    require(
        [m["repeats"] for m in run["measurements"]] == post.REPEATS,
        "Missing recurrence measurements",
    )
    legacy = "analysis_source_lock" not in report
    require(
        not legacy or job["spec"]["phase"] == "development",
        "Only historical development scans may lack new provenance",
    )
    if not legacy:
        require(
            report["training_source_lock"]["verified"]
            and report["analysis_source_lock"]["verified"]
            and report["environment"].get("gpu"),
            "Recurrence source/GPU verification absent",
        )
        require(
            run["provenance"]["complete_sha256"] == job["training_complete_sha256"],
            "Recurrence training receipt changed",
        )
        for name in ("training_source_lock", "analysis_source_lock"):
            lock_path = Path(report[name]["path"])
            require(
                verification.digest(lock_path) == report[name]["sha256"],
                "Recurrence source-lock identity differs",
            )
            locked = verify_lock(lock_path, verification, current=name == "training_source_lock")
            copied_name = (
                "training-source-lock.json"
                if name == "training_source_lock"
                else "recurrence-analysis-lock.json"
            )
            require(
                verification.digest(path.parent / copied_name) == report[name]["sha256"],
                "Recurrence copied source lock changed",
            )
            for relative, expected in locked["files"].items():
                require(
                    report["source"].get(str(ROOT / relative)) == expected,
                    "Recurrence source inventory differs from its analysis version",
                )
            verification.evidence(lock_path)
        for original, expected in report["source"].items():
            original = Path(original)
            relative = original.relative_to(ROOT) if original.is_absolute() else original
            require(
                verification.digest(path.parent / "source" / relative) == expected,
                f"Recurrence source snapshot changed: {relative}",
            )
    native_count = job["spec"]["repeats"]
    require(native_count in post.REPEATS, "Registered recurrence is not in the scan")
    prediction = Path(job["directory"]) / f"predictions-{job['spec']['steps']:07d}.npz"
    native_checks = {}
    with (
        np.load(path.parent / f"{job['run']}.npz", allow_pickle=False) as scan,
        np.load(prediction, allow_pickle=False) as training,
    ):
        for split in ("atomic", *post.SPLITS):
            metric = next(item for item in run["measurements"] if item["repeats"] == native_count)[
                split
            ]
            if not metric["n"]:
                native_checks[split] = {"passed": True, "n": 0, "max_nll_absolute_error": None}
                continue
            for count in post.REPEATS:
                measure = next(item for item in run["measurements"] if item["repeats"] == count)[
                    split
                ]
                prefix = f"r{count}_{split}_"
                answer, stop, target = (
                    scan[prefix + name] for name in ("answer", "stop", "target")
                )
                require(
                    len(target) == measure["n"]
                    and np.array_equal(target, training[split + "_target"]),
                    "Recurrence targets/denominator differ",
                )
                require(
                    measure["accuracy"] == float(((answer == target) & (stop == 1)).mean()),
                    "Recurrence raw accuracy differs from summary",
                )
            prefix = f"r{native_count}_{split}_"
            for name in ("answer", "stop", "target"):
                require(
                    np.array_equal(scan[prefix + name], training[split + "_" + name]),
                    f"Native recurrence predictions differ: {split}.{name}",
                )
            delta = float(np.max(np.abs(scan[prefix + "nll"] - training[split + "_nll"])))
            require(math.isfinite(delta) and delta <= 1e-5, "Native recurrence NLL differs")
            if not legacy:
                require(
                    run["native_repeats_audit"][split]["passed"] is True,
                    "Recorded native recurrence audit failed",
                )
            native_checks[split] = {
                "passed": True,
                "n": metric["n"],
                "max_nll_absolute_error": delta,
            }
    return {
        "run": job["run"],
        "path": str(path),
        "passed": True,
        "legacy_development_provenance": legacy,
        "native_cpu_recount": native_checks,
    }


def total_cost(rows):
    if not rows:
        return {"run_count": 0}
    start = min(datetime.fromisoformat(row["started_utc"]) for row in rows)
    end = max(datetime.fromisoformat(row["finished_utc"]) for row in rows)
    return {
        "run_count": len(rows),
        **{name: sum(row[name] for row in rows) for name in TOTAL_FIELDS},
        "first_run_metadata_utc": start.isoformat(),
        "last_run_complete_utc": end.isoformat(),
        "run_utc_span_seconds": (end - start).total_seconds(),
    }


def inspect_completion():
    verification = Verification()
    post.ROOT = ROOT
    jobs = post.load_jobs(CONFIGS)
    counts = Counter(job["spec"]["phase"] for job in jobs)
    require(dict(counts) == PHASE_COUNTS, f"Registered matrix differs: {dict(counts)}")
    locks, configs = {}, {}
    for config_path in CONFIGS:
        config = read(ROOT / config_path)
        configs[config_path] = config
        path = ROOT / config["source_lock"]
        locks[config_path] = verification.attempt(
            "source_lock", None, path, lambda path=path: verify_lock(path, verification)
        )
        verification.evidence(ROOT / config_path)
    recurrence_lock = ROOT / ARTIFACT / "recurrence-analysis-lock-v2.json"
    verification.attempt(
        "analysis_lock", None, recurrence_lock, lambda: verify_lock(recurrence_lock, verification)
    )
    dev_path = ROOT / ARTIFACT / "development-endpoint-audit.json"
    dev_expected = [job["run"] for job in jobs if job["spec"]["phase"] == "development"]
    development = verification.attempt(
        "development_gpu_audit",
        None,
        dev_path,
        lambda: verify_development_audit(dev_path, dev_expected),
    )
    scans = recurrence_catalog([Path(job["output_root"]) for job in jobs], verification)
    costs, audited, mechanisms, recurrences = [], defaultdict(list), [], []
    required_scans = {
        job["run"]
        for job in jobs
        if "sensitivity" not in job["spec"]["phase"] and job["spec"]["architecture"] in {"l1", "l2"}
    }
    for raw_job in jobs:
        run, spec = raw_job["run"], raw_job["spec"]
        directory = Path(raw_job["directory"])
        inspected = verification.attempt(
            "training",
            run,
            directory / "complete.json",
            lambda raw_job=raw_job: verify_training(
                raw_job, locks[raw_job["config"]], verification
            ),
        )
        job, cost = inspected if inspected else (raw_job, None)
        if cost:
            costs.append(cost)
        if spec["phase"] == "development":
            if development is not None and cost is not None:
                audit = verification.attempt(
                    "gpu_audit_identity",
                    run,
                    dev_path,
                    lambda run=run, job=job, cost=cost: verify_audit_run(
                        development[run], job, cost
                    ),
                )
                if audit:
                    audited[job["config"]].append(audit)
        else:
            path = ROOT / post.ARTIFACT / run / "audit.json"

            def verify_saved_audit(path=path, job=job, cost=cost):
                require(cost is not None, "Training identity not yet verified")
                post.validate_audit(path, job)
                receipt = read(path)
                return {
                    **verify_audit_run(receipt["audit"], job, cost),
                    "reload_receipt": str(path),
                    "reload_receipt_sha256": verification.digest(path),
                    "environment": receipt["environment"],
                    "auditor_source": receipt["auditor_source"],
                }

            audit = verification.attempt("gpu_audit", run, path, verify_saved_audit)
            if audit:
                audited[job["config"]].append(audit)
        if "sensitivity" in spec["phase"]:
            continue
        for split in post.SPLITS:
            path = (
                Path(job["output_root"])
                / f"mechanism-{spec['phase']}"
                / run
                / f"step-{spec['steps']:07d}-{split}"
            )

            def verify_saved_mechanism(path=path, job=job, split=split, cost=cost):
                require(cost is not None, "Training identity not yet verified")
                return verify_mechanism(path, job, split, verification)

            item = verification.attempt(
                "mechanism", run, path / "summary.json", verify_saved_mechanism
            )
            if item:
                mechanisms.append(item)
        if run not in required_scans:
            continue
        candidates = scans.get(run, [])
        if not candidates:
            verification.missing.append(
                {
                    "kind": "recurrence",
                    "run": run,
                    "path": str(Path(job["output_root"]) / f"recurrence-{spec['phase']}-{run}"),
                }
            )
        elif len(candidates) != 1:
            verification.errors.append(
                {
                    "kind": "recurrence",
                    "run": run,
                    "error": "Duplicate scan results; cannot select one",
                }
            )
        else:
            path, report, result = candidates[0]

            def verify_saved_recurrence(
                path=path, report=report, result=result, job=job, cost=cost
            ):
                require(cost is not None, "Training identity not yet verified")
                return verify_recurrence(path, report, result, job, verification)

            item = verification.attempt("recurrence", run, path, verify_saved_recurrence)
            if item:
                recurrences.append(item)
    aggregates = {}
    for config_path, config in configs.items():
        runs = audited[config_path]
        pairs = audit_sampling_pairs(runs)
        complete = len(runs) == len(config["runs"])
        if not all(pair["passed"] for pair in pairs):
            verification.errors.append(
                {
                    "kind": "paired_sampling",
                    "config": config_path,
                    "error": "Datasets, terminal streams or exposure differ",
                }
            )
        aggregates[config_path] = {
            "config": config_path,
            "device": "previous independent GPU reloads; CPU aggregation",
            "finished_utc": post.utc(),
            "passed": complete and all(pair["passed"] for pair in pairs),
            "runs": runs,
            "paired_sampling": pairs,
        }
    for pattern in ("scripts/*grok_loop*.py", "tests/test_grok_loop*.py"):
        for path in ROOT.glob(pattern):
            verification.evidence(path)
    for path in (ROOT / ARTIFACT).glob("*report/**/*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".png", ".pdf"}:
            verification.evidence(path)
    for name in ("validation.json", "independent-review.json", "model-preflight.json"):
        path = ROOT / ARTIFACT / name
        if path.exists():
            verification.evidence(path)
    expected_nodes = sum(len(job["spec"]["nodes"]) for job in jobs)
    expected_mechanisms = sum("sensitivity" not in job["spec"]["phase"] for job in jobs) * len(
        post.SPLITS
    )
    ready = not verification.missing and not verification.errors
    result = {
        "state": "ready" if ready else "incomplete",
        "generated_utc": post.utc(),
        "expected": {
            "training_runs": len(jobs),
            "evaluation_nodes": expected_nodes,
            "gpu_audits": len(jobs),
            "mechanism_splits": expected_mechanisms,
            "recurrence_runs": len(required_scans),
        },
        "verified": {
            "training_runs": len(costs),
            "evaluation_nodes": sum(row["evaluation_nodes"] for row in costs),
            "gpu_audits": sum(len(items) for items in audited.values()),
            "mechanism_splits": len(mechanisms),
            "recurrence_runs": len(recurrences),
        },
        "missing": verification.missing,
        "errors": verification.errors,
        "phase_totals": {
            phase: total_cost([row for row in costs if row["phase"] == phase]) for phase in counts
        },
        "totals": total_cost(costs),
        "runs": costs,
        "mechanisms": mechanisms,
        "recurrences": recurrences,
        "optional_scan_runs": sorted(set(scans) - required_scans),
        "development_audit_binding": (
            "Historical GPU audit has no checkpoint digest. CPU checkpoint spec/step and saved "
            "mechanism checkpoint hashes bind its current endpoint; no retrospective audit hash "
            "is invented."
        ),
        "timing_scope": (
            "Training/evaluation seconds sum recorded run timers, including concurrency. UTC span "
            "is first metadata to last training completion; it excludes implementation and later "
            "analysis and is not an architecture speed comparison."
        ),
        "verification_files": verification.hashes,
    }
    require(
        not ready or result["verified"] == result["expected"],
        "Finalized coverage does not equal the complete matrix",
    )
    return result, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result, aggregates = inspect_completion()
    except Exception as exc:
        print(json.dumps({"state": "refused", "error": repr(exc), "artifacts_written": False}))
        raise SystemExit(1) from exc
    if args.check_only or result["state"] != "ready":
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("state", "expected", "verified", "missing", "errors")
                },
                indent=2,
            )
        )
        if result["state"] != "ready":
            raise SystemExit(1)
        return
    for config, aggregate in aggregates.items():
        if "development-v1" in config:
            continue
        phase = "sensitivity" if "sensitivity" in config else "confirmation"
        path = ROOT / ARTIFACT / f"{phase}-endpoint-audit.json"
        post.write(path, aggregate)
        result["verification_files"][str(path)] = post.digest(path)
    result["state"] = "complete"
    path = ROOT / ARTIFACT / "completion-manifest.json"
    post.write(path, result)
    print(json.dumps({"state": "complete", "manifest": str(path), "verified": result["verified"]}))


if __name__ == "__main__":
    main()
