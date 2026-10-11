"""Read-only endpoint verification of the eight Z1 audit-tolerance W&B runs.

Wait for the controller and sidecar to exit, including the sidecar's drain
receipt, then compare all 42 uploaded analysis records with their local records.
No model is loaded, no evaluation is repeated, and no W&B run is created.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import netrc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/ossfs/workspace/research-recovery/zero-training-path-overlap-v3") / (
    "execution-attempts/audit-tolerance-v1"
)
CREDENTIALS = Path("/ossfs/workspace/.lm-memory-credentials/wandb.netrc")
SCRIPT_NAME = "check_path_overlap_cloud.py"
# W&B's serialized metrics may have a different floating representation. This
# tolerance applies only to upload verification, never to scientific replay.
UPLOAD_ABS_TOL, UPLOAD_REL_TOL = 1e-8, 1e-6
TERMINAL = {"complete", "finished_with_failures", "development_prerequisite_not_met"}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_alive(metadata):
    if not metadata.exists():
        return False
    value = read(metadata)
    proc = Path(f"/proc/{value.get('pid')}")
    try:
        state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        if state == "Z":
            return False
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        return any(
            str(part) in command
            for part in value.get("command", [])
            if "orchestration/" in str(part)
        )
    except FileNotFoundError:
        return False


def wait_terminal(base, timeout, poll_seconds):
    deadline = time.monotonic() + timeout
    while True:
        state_path = base / "controller-state.json"
        state = read(state_path) if state_path.exists() else {}
        if (
            state.get("state") in TERMINAL
            and not state.get("active")
            and not process_alive(base / "controller-process.json")
            and not process_alive(base / "tracking-process.json")
        ):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("Controller and tracker did not become terminal before the deadline")
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def require_local_endpoint(base, controller):
    if controller.get("state") != "complete" or controller.get("failed"):
        raise ValueError("Controller failed or did not complete; cloud verification cannot pass")
    manifest = read(base / "completion-manifest.json")
    if (
        manifest.get("passed") is not True
        or manifest.get("independently_replayed_nodes") != 112
        or manifest.get("new_optimizer_updates") != 0
        or manifest.get("new_scientific_interventions") != 0
    ):
        raise ValueError("Independent replay completion is incomplete")
    receipt = read(base / "audit-recovery-receipt.json")
    if (
        manifest.get("audit_recovery_receipt_sha256") != sha(base / "audit-recovery-receipt.json")
        or receipt.get("scientific_freeze_sha256") != sha(base / "frozen/freeze.json")
        or manifest.get("comparison_contract") != receipt.get("comparison_contract")
        or manifest.get("summary_sha256") != sha(base / "summary.json")
    ):
        raise ValueError("Local scientific and audit-comparison lineage changed")
    drain = read(base / "tracking-completion.json")
    expected = set(controller["completed"])
    if len(expected) != 8 or set(manifest["runs"]) != expected:
        raise ValueError("Expected eight independently audited runs")
    if (
        drain.get("passed") is not True
        or drain.get("registered_runs") != 8
        or set(drain.get("runs", {})) != expected
        or any(
            run.get("last_logged_step") != 42 or run.get("final_step") != 42
            for run in drain["runs"].values()
        )
    ):
        raise ValueError("The original tracker has not drained all 42 records for all eight runs")
    return sorted(expected)


def load_tracking(base):
    frozen = read(base / "frozen/freeze.json")
    path = base / "frozen/source/src/llm_memory_editability/experiment_tracking.py"
    if sha(path) != frozen["source_files"]["src/llm_memory_editability/experiment_tracking.py"]:
        raise ValueError("Frozen metric mapping source changed")
    spec = importlib.util.spec_from_file_location("frozen_endpoint_metric_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_uploaded_metrics(history, learning_metrics, nodes=None):
    if [row.get("step") for row in history] != list(range(1, 43)):
        raise ValueError("Local history does not contain exactly 42 ordered analysis records")
    if nodes is not None and [row.get("source_training_step") for row in history] != nodes * 3:
        raise ValueError("Local history does not correspond to the frozen 14 analysis nodes")
    expected, previous = {}, None
    for record in history:
        if record.get("optimizer_updates") != 0:
            raise ValueError("Zero-training analysis records cannot contain optimizer updates")
        step = record["step"]
        stage = "select" if step <= 14 else "science" if step <= 28 else "audit"
        if record.get("stage") != stage:
            raise ValueError("Local analysis stage differs from the predetermined 14+14+14 order")
        metrics = learning_metrics(record, previous)
        for key in ("case", "case_id", "condition", "variant", "arm"):
            if key in record:
                metrics["condition/" + key] = record[key]
        metrics["evaluation/index"] = step
        metrics["condition/step_unit"] = "evaluation_index"
        metrics.pop("perf/updates_per_second", None)
        metrics.pop("perf/training_ms_per_update", None)
        if previous is not None:
            seconds = record.get("wall_seconds", 0) - previous.get("wall_seconds", 0)
            updates = record["optimizer_updates"] - previous.get("optimizer_updates", 0)
            if seconds > 0:
                metrics["perf/updates_per_second"] = updates / seconds
        expected[step] = metrics
        previous = record
    return expected


def require_metric_value(expected, actual, key):
    if isinstance(expected, float):
        if (
            not isinstance(actual, int | float)
            or isinstance(actual, bool)
            or not math.isfinite(actual)
        ):
            raise ValueError(f"Cloud metric is not a finite number: {key}")
        if not math.isclose(expected, actual, abs_tol=UPLOAD_ABS_TOL, rel_tol=UPLOAD_REL_TOL):
            raise ValueError(f"Cloud metric differs from its local analysis record: {key}")
    elif actual != expected or isinstance(expected, bool) != isinstance(actual, bool):
        raise ValueError(f"Cloud metric differs from its local analysis record: {key}")


def compare_cloud_run(remote, expected, local_spec):
    summary = dict(remote.summary)
    if (
        remote.state != "finished"
        or summary.get("scientific_final_step") != 42
        or summary.get("independently_reloaded") is not True
        or summary.get("has_failure_record") is not False
        or summary.get("training_state") != "complete"
    ):
        raise ValueError("Cloud run is unfinished, failed, or lacks the independent-audit endpoint")
    if remote.config.get("spec") != local_spec:
        raise ValueError("Cloud scientific/execution specification differs from local run metadata")
    rows = {}
    for row in remote.scan_history(page_size=1000):
        if row.get("training/step") is None:
            continue
        step = row["training/step"]
        if isinstance(step, bool) or step != int(step) or step not in expected or step in rows:
            raise ValueError("Cloud analysis indexes are duplicate or outside the local 42 records")
        if row.get("_step") != step:
            raise ValueError("Cloud SDK history index differs from the analysis index")
        rows[step] = row
    if set(rows) != set(expected):
        raise ValueError("Cloud history does not cover every one of the 42 local analysis records")
    checked = 0
    canonical = []
    for step, metrics in expected.items():
        row = rows[step]
        for key, value in metrics.items():
            if key not in row:
                raise ValueError(f"Cloud history is missing a local key metric: {key}")
            require_metric_value(value, row[key], key)
            checked += 1
        canonical.append({key: row[key] for key in sorted(metrics)})
    return {
        "passed": True,
        "cloud_state": remote.state,
        "scientific_final_step": summary["scientific_final_step"],
        "independently_reloaded": summary["independently_reloaded"],
        "analysis_records": len(rows),
        "metric_values_checked": checked,
        "verified_uploaded_metrics_sha256": hashlib.sha256(
            json.dumps(canonical, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def verify_endpoint(base, api):
    controller = read(base / "controller-state.json")
    runs = require_local_endpoint(base, controller)
    tracking = load_tracking(base)
    frozen = read(base / "frozen/freeze.json")
    contract_path = base / "frozen/source" / frozen["contract"]["path"]
    if sha(contract_path) != frozen["contract"]["sha256"]:
        raise ValueError("Frozen scientific contract changed")
    cfg = read(contract_path)
    nodes = cfg["source"]["nodes"]
    if sorted(cfg["source"]["runs"]) != runs or len(nodes) != 14:
        raise ValueError("Local completed matrix differs from the frozen scientific contract")
    results, actual_ids = {}, set()
    for run in runs:
        folder = base / "runs" / run
        audit, complete = read(folder / "audit.json"), read(folder / "complete.json")
        if (
            (folder / "failure.json").exists()
            or audit.get("passed") is not True
            or audit.get("fresh_process") is not True
            or audit.get("fresh_container") is not True
            or len(audit.get("nodes", [])) != 14
            or [node.get("step") for node in audit.get("nodes", [])] != nodes
            or any(node.get("passed") is not True for node in audit["nodes"])
            or complete.get("audit_sha256") != sha(folder / "audit.json")
            or complete.get("analysis_index") != 42
        ):
            raise ValueError("Local run lacks a complete, fresh independent 14-node replay")
        metadata = read(folder / "run.json")
        state_path = base / "tracking-wandb" / run / "state.json"
        state = read(state_path)
        run_id = state.get("wandb_run_id")
        if not run_id or state.get("mode") != "online" or state.get("last_logged_step") != 42:
            raise ValueError("Actual tracker state has no complete online run identity")
        path = f"{state['entity']}/{state['project']}/{run_id}"
        if path in actual_ids:
            raise ValueError("Two local trajectories share the same actual cloud run identity")
        actual_ids.add(path)
        history_path = folder / "learning.json"
        expected = expected_uploaded_metrics(read(history_path), tracking.learning_metrics, nodes)
        remote = api.run(path)
        results[run] = {
            **compare_cloud_run(remote, expected, tracking.public_config(metadata["spec"])),
            "actual_wandb_run_id": run_id,
            "url": f"https://wandb.ai/{path}",
            "tracking_state_sha256": sha(state_path),
            "local_learning_sha256": sha(history_path),
            "local_audit_sha256": sha(folder / "audit.json"),
        }
    return {
        "passed": True,
        "run_count": len(results),
        "analysis_records": sum(run["analysis_records"] for run in results.values()),
        "runs": results,
        "scientific_freeze_sha256": sha(base / "frozen/freeze.json"),
        "completion_manifest_sha256": sha(base / "completion-manifest.json"),
        "tracking_drain_sha256": sha(base / "tracking-completion.json"),
        "audit_recovery_receipt_sha256": sha(base / "audit-recovery-receipt.json"),
        "upload_float_precision": {
            "abs_tolerance": UPLOAD_ABS_TOL,
            "rel_tolerance": UPLOAD_REL_TOL,
        },
        "scientific_replay_comparison": read(base / "audit-recovery-receipt.json")[
            "comparison_contract"
        ],
        "read_only_cloud_api": True,
        "new_optimizer_updates": 0,
        "new_evaluations": 0,
        "verified_utc": utc(),
    }


def run_verifier(args):
    base = args.base
    folder = base / "cloud-verification"
    folder.mkdir(exist_ok=True)
    with (folder / "verifier.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = folder / "receipt.json"
        if receipt_path.exists():
            raise RuntimeError("Independent cloud receipt already exists; no automatic retry")
        try:
            state = wait_terminal(base, args.timeout_seconds, args.poll_seconds)
            require_local_endpoint(base, state)
            import wandb

            auth = netrc.netrc(str(CREDENTIALS)).authenticators("api.wandb.ai")
            if not auth or not auth[2]:
                raise ValueError("Local W&B credential is unavailable")
            api = wandb.Api(api_key=auth[2], timeout=60)
            result = verify_endpoint(base, api)
        except (ValueError, FileNotFoundError, TimeoutError) as error:
            result = {"passed": False, "error_type": type(error).__name__, "message": str(error)}
        except Exception as error:
            # SDK exceptions can contain request details; omit them from receipts.
            result = {
                "passed": False,
                "error_type": type(error).__name__,
                "message": "Read-only W&B endpoint query did not complete successfully",
            }
        result.update(
            verifier_script_sha256=sha(Path(__file__)),
            attempt_base=str(base),
            completed_utc=utc(),
        )
        write(receipt_path, result)
        print(json.dumps({"passed": result["passed"], "receipt": str(receipt_path)}), flush=True)
        if not result["passed"]:
            raise SystemExit(1)


def launch(args):
    base = args.base
    folder = base / "cloud-verification"
    folder.mkdir(exist_ok=True)
    with (folder / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "process.json").exists() or (folder / "receipt.json").exists():
            raise RuntimeError("Independent cloud verification has already been submitted")
        script = base / "orchestration" / SCRIPT_NAME
        if script.exists():
            if sha(script) != sha(Path(__file__)):
                raise ValueError("Installed independent cloud verifier differs")
        else:
            shutil.copy2(Path(__file__), script)
            script.chmod(0o444)
        command = [
            sys.executable,
            "-u",
            str(script),
            "run",
            "--base",
            str(base),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--poll-seconds",
            str(args.poll_seconds),
        ]
        env = {
            **os.environ,
            "LD_LIBRARY_PATH": "/lib64"
            + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
        }
        with (folder / "verifier.log").open("x") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
        write(
            folder / "process.json",
            {
                "pid": process.pid,
                "command": command,
                "script_sha256": sha(script),
                "read_only_cloud_api": True,
                "created_utc": utc(),
            },
        )
        print(json.dumps({"pid": process.pid, "log": str(folder / "verifier.log")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "run"))
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60 or args.timeout_seconds <= 0:
        parser.error("poll must be in (0,60], and timeout must be positive")
    launch(args) if args.action == "launch" else run_verifier(args)


if __name__ == "__main__":
    main()
