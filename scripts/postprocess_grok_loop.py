#!/usr/bin/env python3
"""Monitor registered training endpoints and evaluate them in isolated GPU processes.

The monitor imports only the standard library. Existing complete analysis outputs
are identity-checked; incomplete outputs are retained and cause a nonzero result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path("docs/development-artifacts/grok-loop-v1/postprocess")
SPLITS = ("test_composite", "ood_composite")
REPEATS = list(range(1, 9))


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_jobs(configs):
    jobs, seen = [], set()
    for config in configs:
        cfg = read(ROOT / config)
        for run, overrides in cfg["runs"].items():
            require(run not in seen, f"Duplicate registered run: {run}")
            require(Path(run).name == run, f"Invalid run name: {run}")
            seen.add(run)
            spec = {**cfg["base"], **overrides}
            jobs.append(
                {
                    "run": run,
                    "config": str(config),
                    "spec": spec,
                    "directory": str(ROOT / cfg["output_root"] / spec["phase"] / run),
                    "output_root": str(ROOT / cfg["output_root"]),
                }
            )
    require(bool(jobs), "No registered runs")
    return jobs


def training_complete(job):
    path = Path(job["directory"]) / "complete.json"
    if not path.exists():
        return False
    complete = read(path)
    require(complete["spec"] == job["spec"], f"Training spec mismatch: {path}")
    require(
        complete["endpoint"]["step"] == job["spec"]["steps"] and complete.get("finished_utc"),
        f"Training completion marker is incomplete: {path}",
    )
    return True


def validate_archive(path):
    with zipfile.ZipFile(path) as archive:
        require(bool(archive.namelist()), f"Empty prediction archive: {path}")
        require(archive.testzip() is None, f"Corrupt prediction archive: {path}")


def validate_audit(path, job):
    result = read(path)
    require(
        result.get("state") == "complete"
        and result.get("finished_utc")
        and result.get("passed") is True,
        f"Audit incomplete or failed: {path}",
    )
    require(
        result.get("run") == job["run"]
        and result.get("spec") == job["spec"]
        and result.get("training_complete_sha256") == job["training_complete_sha256"]
        and result.get("checkpoint_sha256") == job["checkpoint_sha256"],
        f"Audit identity mismatch: {path}",
    )
    require(
        result["audit"]["run"] == job["run"]
        and result["audit"]["passed"] is True
        and result["audit"]["reload_max_nll_absolute_error"] is not None,
        f"Independent GPU reload missing or failed: {path}",
    )
    require(
        result.get("auditor_source")
        and result.get("environment", {}).get("gpu")
        and result["environment"].get("device") == "cuda:0",
        f"Audit source or GPU provenance missing: {path}",
    )


def validate_mechanism(path, job, split):
    status, meta, summary = (
        read(path / name) for name in ("status.json", "metadata.json", "summary.json")
    )
    require(
        status.get("state") == "complete" and status.get("finished_utc"),
        f"Mechanism output incomplete: {path}",
    )
    provenance = meta["provenance"]
    require(
        meta["spec"] == job["spec"]
        and meta["split"] == summary["split"] == split
        and meta["step"] == job["spec"]["steps"]
        and Path(provenance["source_dir"]).resolve() == Path(job["directory"]).resolve()
        and provenance["checkpoint_sha256"] == job["checkpoint_sha256"],
        f"Mechanism identity mismatch: {path}",
    )
    for name in ("donors.npz", "predictions.npz"):
        validate_archive(path / name)


def validate_recurrence(path, job):
    result = read(path / "summary.json")
    require(result.get("finished_utc"), f"Recurrence output incomplete: {path}")
    require(result["repeats"] == REPEATS, f"Recurrence counts changed: {path}")
    require(len(result["runs"]) == 1, f"Unexpected recurrence run count: {path}")
    require(
        result.get("source")
        and result.get("training_source_lock", {}).get("verified") is True
        and result.get("analysis_source_lock", {}).get("verified") is True
        and result.get("environment", {}).get("gpu"),
        f"Recurrence source or GPU provenance missing: {path}",
    )
    run = result["runs"][0]
    require(
        run["run"] == job["run"]
        and run["spec"] == job["spec"]
        and run["checkpoint"] == {job["checkpoint"]: job["checkpoint_sha256"]},
        f"Recurrence identity mismatch: {path}",
    )
    require(
        run.get("provenance", {}).get("complete_sha256") == job["training_complete_sha256"]
        and all(
            run.get("native_repeats_audit", {}).get(split, {}).get("passed") is True
            for split in ("atomic", *SPLITS)
        ),
        f"Recurrence native-checkpoint verification missing or failed: {path}",
    )
    require(
        [item["repeats"] for item in run["measurements"]] == REPEATS
        and all(
            all(split in item for split in ("atomic", *SPLITS)) for item in run["measurements"]
        ),
        f"Recurrence measurements incomplete: {path}",
    )
    validate_archive(path / f"{job['run']}.npz")


def attempt_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = parent / f"attempt-{index:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def process_job(job, gpu):
    """Run one endpoint's stages sequentially; never mutate a training artifact."""
    artifact = ROOT / ARTIFACT / job["run"]
    attempt = attempt_directory(artifact)
    record = {
        "run": job["run"],
        "spec": job["spec"],
        "gpu": gpu,
        "started_utc": utc(),
        "stages": [],
    }

    def save():
        write(attempt / "status.json", record)
        write(artifact / "status.json", record)

    def execute(stage, command):
        log = attempt / f"{stage}.log"
        item = {"stage": stage, "state": "running", "command": command, "log": str(log)}
        record["stages"].append(item)
        save()
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        with log.open("x") as handle:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        item.update(
            exit_code=result.returncode, state="complete" if result.returncode == 0 else "failed"
        )
        save()
        require(result.returncode == 0, f"{stage} exited {result.returncode}; see {log}")

    try:
        record["state"] = "running"
        save()
        require(training_complete(job), f"Training incomplete: {job['run']}")
        checkpoint = Path(job["directory"]) / f"weights-{job['spec']['steps']:07d}.pt"
        job = {
            **job,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "training_complete_sha256": digest(Path(job["directory"]) / "complete.json"),
        }
        audit = artifact / "audit.json"
        if not audit.exists():
            execute(
                "audit",
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--audit-worker",
                    job["config"],
                    job["run"],
                    str(audit),
                ],
            )
        validate_audit(audit, job)
        if "sensitivity" not in job["spec"]["phase"]:
            root = Path(job["output_root"]) / f"mechanism-{job['spec']['phase']}"
            missing = []
            for split in SPLITS:
                path = root / job["run"] / f"step-{job['spec']['steps']:07d}-{split}"
                if path.exists():
                    validate_mechanism(path, job, split)
                else:
                    missing.append(split)
            if missing:
                command = [
                    sys.executable,
                    "scripts/analyze_grok_loop_mechanism.py",
                    job["directory"],
                    "--out",
                    str(root),
                    "--device",
                    "cuda:0",
                ]
                for split in missing:
                    command.extend(["--split", split])
                execute("mechanism", command)
                for split in missing:
                    validate_mechanism(
                        root / job["run"] / f"step-{job['spec']['steps']:07d}-{split}", job, split
                    )
            if job["spec"]["architecture"] in {"l1", "l2"}:
                root = Path(job["output_root"]) / f"recurrence-{job['spec']['phase']}-{job['run']}"
                if not root.exists():
                    execute(
                        "recurrence",
                        [
                            sys.executable,
                            "scripts/evaluate_grok_loop_recurrence.py",
                            "--config",
                            job["config"],
                            "--runs",
                            job["run"],
                            "--architectures",
                            "l1",
                            "l2",
                            "--out",
                            str(root),
                            "--device",
                            "cuda:0",
                            "--repeats",
                            *map(str, REPEATS),
                        ],
                    )
                validate_recurrence(root, job)
        record.update(state="complete", finished_utc=utc())
    except Exception as exc:
        record.update(state="failed", error=repr(exc), finished_utc=utc())
        (attempt / "error.txt").write_text(traceback.format_exc())
    save()
    return record


def audit_worker(config, run, output):
    """Only this subprocess imports torch or creates a GPU context."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from audit_grok_loop import audit_run

    from llm_memory_editability.grok_depth_bridge import environment

    jobs = [job for job in load_jobs([config]) if job["run"] == run]
    require(len(jobs) == 1, f"Unknown audit run: {run}")
    job = jobs[0]
    require(training_complete(job), f"Training incomplete: {run}")
    require(not Path(output).exists(), f"Refusing to overwrite audit: {output}")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device("cuda:0"))
    directory = Path(job["directory"])
    result = audit_run(directory, "cuda:0")
    passed = result["passed"]
    write(
        output,
        {
            "run": run,
            "spec": job["spec"],
            "passed": passed,
            "state": "complete" if passed else "failed",
            "finished_utc": utc(),
            "device": "cuda:0",
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "environment": environment("cuda:0"),
            "auditor_source": {
                str(path.relative_to(ROOT)): digest(path)
                for path in (
                    Path(__file__).resolve(),
                    ROOT / "scripts/audit_grok_loop.py",
                    ROOT / "src/llm_memory_editability/grok_loop_model.py",
                    ROOT / "src/llm_memory_editability/grok_multihop.py",
                    ROOT / "src/llm_memory_editability/grok_multihop_data.py",
                    ROOT / "src/llm_memory_editability/grok_depth.py",
                    ROOT / "src/llm_memory_editability/bios_model.py",
                )
            },
            "training_complete_sha256": digest(directory / "complete.json"),
            "checkpoint_sha256": digest(directory / f"weights-{job['spec']['steps']:07d}.pt"),
            "audit": result,
        },
    )
    return 0 if passed else 1


def monitor(jobs, gpus, poll_seconds):
    artifact = ROOT / ARTIFACT
    attempt = attempt_directory(artifact / "monitor")
    pending = {job["run"]: job for job in jobs}
    records, active, unreadable = {}, {}, {}
    free = list(gpus)

    def save():
        state = (
            "running"
            if pending or active
            else (
                "complete"
                if all(record["state"] == "complete" for record in records.values())
                else "failed"
            )
        )
        write(
            attempt / "status.json",
            {
                "state": state,
                "expected": len(jobs),
                "expected_runs": [job["run"] for job in jobs],
                "pending": list(pending),
                "active": [job["run"] for _, job in active.values()],
                "runs": records,
                "updated_utc": utc(),
            },
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        while pending or active:
            for future in list(active):
                if future.done():
                    gpu, job = active.pop(future)
                    free.append(gpu)
                    try:
                        records[job["run"]] = future.result()
                    except Exception as exc:
                        records[job["run"]] = {"state": "failed", "error": repr(exc)}
                    print(
                        json.dumps({"run": job["run"], "state": records[job["run"]]["state"]}),
                        flush=True,
                    )
            for run, job in list(pending.items()):
                try:
                    ready = training_complete(job)
                except json.JSONDecodeError as exc:
                    unreadable[run] = unreadable.get(run, 0) + 1
                    if unreadable[run] < 3:
                        continue
                    records[run] = {"state": "failed", "error": repr(exc)}
                    del pending[run]
                    continue
                except Exception as exc:
                    records[run] = {"state": "failed", "error": repr(exc)}
                    del pending[run]
                    continue
                unreadable.pop(run, None)
                if ready and free:
                    gpu = free.pop(0)
                    active[pool.submit(process_job, job, gpu)] = (gpu, job)
                    del pending[run]
            save()
            if active:
                concurrent.futures.wait(
                    active, timeout=poll_seconds, return_when=concurrent.futures.FIRST_COMPLETED
                )
            elif pending:
                time.sleep(poll_seconds)
    return all(record["state"] == "complete" for record in records.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument(
        "--audit-worker", nargs=3, metavar=("CONFIG", "RUN", "OUT"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.audit_worker:
        raise SystemExit(audit_worker(*args.audit_worker))
    if not args.configs or not 0 < args.poll_seconds <= 30:
        parser.error("--configs required; --poll-seconds must be in (0, 30]")
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU IDs must be unique and nonnegative")
    jobs = load_jobs(args.configs)
    artifact = ROOT / ARTIFACT
    artifact.mkdir(parents=True, exist_ok=True)
    with (artifact / "monitor.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not monitor(jobs, args.gpus, args.poll_seconds):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
