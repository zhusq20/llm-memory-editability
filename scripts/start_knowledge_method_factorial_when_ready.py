#!/usr/bin/env python3
"""Submit the authorized factorial batch once, after every engineering gate passes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "knowledge-method-factorial-1000-v1"
DEFAULT_PLAN = ROOT / "configs/knowledge-method-factorial-launch-v1.json"
SCIENTIFIC_CONFIG = "configs/knowledge-method-factorial-1000-v1.json"
JUDGE_MODEL = "Qwen/Qwen3-4B"
JUDGE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ARMS = (
    "pd_lora",
    "sft_lora",
    "pd_full",
    "sft_full",
    "pd_down",
    "sft_down",
    "pd_down_lora",
    "sft_down_lora",
)
EXPECTED_STAGES = {f"exercise-{arm}" for arm in ARMS} | {
    "reload-pd_full",
    "reload-pd_down",
    "reload-pd_lora",
    "reload-pd_down_lora",
    "pressure-pd_full",
}
CORE_FILES = (
    "src/llm_memory_editability/knowledge_method_factorial.py",
    "src/llm_memory_editability/knowledge_method_diagnosis.py",
)
MARKERS = {
    "full_size": "engineering/full-size-check.json",
    "training_core": "engineering/training-core-compatibility.json",
    "cpu": "engineering/fixed-image-contracts-final-v1/verification.json",
    "judge": "engineering/judge-smoke-qwen3-4b/verification.json",
    "download": "judge-download-complete.json",
}


class GateFailed(ValueError):
    """Existing evidence is failed, incomplete, or no longer matches the source."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise GateFailed(message)


def validate_plan(plan, root=ROOT):
    require(plan.get("batch") == BATCH, "Only the authorized factorial batch is allowed")
    require(plan.get("scientific_config") == SCIENTIFIC_CONFIG, "Unexpected scientific config")
    require(Path(plan["output"]).name == BATCH, "Unexpected batch output directory")
    require(set(plan.get("arms", [])) == set(ARMS), "The eight-arm matrix differs")
    config = json.loads((Path(root) / SCIENTIFIC_CONFIG).read_text())
    require(config.get("judge_model") == JUDGE_MODEL, "Judge model differs")
    require(config.get("judge_revision") == JUDGE_REVISION, "Judge revision differs")
    require(config.get("judge_enable_thinking") is False, "Judge thinking must be disabled")
    return config


def verify_hashes(values, root, label):
    require(isinstance(values, dict) and bool(values), f"{label}: missing source hashes")
    verified = {}
    root = Path(root).resolve()
    for relative, expected in values.items():
        require(isinstance(relative, str), f"{label}: invalid source path")
        path = root / relative
        require(
            not Path(relative).is_absolute() and path.resolve().is_relative_to(root),
            f"{label}: source path is not relative to the repository: {relative}",
        )
        require(path.is_file(), f"{label}: missing current source {relative}")
        actual = sha(path)
        require(actual == expected, f"{label}: stale source hash: {relative}")
        verified[relative] = actual
    return verified


def _passed(marker, label):
    require(marker.get("passed") is True, f"{label}: completed marker did not pass")


def check_readiness(plan, root=ROOT):
    """Read every available marker; absent evidence waits, failed/stale evidence stops."""
    root = Path(root)
    config = validate_plan(plan, root)
    batch = Path(plan["output"])
    missing, evidence = [], {}
    # Child failures can appear before the aggregate GPU report is written.
    for stage in sorted(EXPECTED_STAGES):
        path = batch / "engineering/results" / f"{stage}.json"
        if path.is_file():
            value = json.loads(path.read_text())
            _passed(value, str(path))
            require(value.get("exit_code", 0) == 0, f"{path}: nonzero child exit")
    for name, relative in MARKERS.items():
        path = batch / relative
        if not path.is_file():
            missing.append(relative)
            continue
        marker = json.loads(path.read_text())
        verified = {}
        if name == "training_core":
            require(
                marker.get("core_training_source_matches") is True,
                f"{relative}: training core does not match",
            )
        else:
            _passed(marker, relative)
        if name == "full_size":
            stages = marker.get("stage_results", {})
            require(
                isinstance(stages, dict) and set(stages) == EXPECTED_STAGES,
                f"{relative}: expected exactly all 13 engineering tasks",
            )
            for stage, result in stages.items():
                _passed(result, stage)
                require(result.get("exit_code") == 0, f"{stage}: missing or nonzero exit code")
            files = marker.get("source_manifest", {}).get("files", {})
            require(all(key in files for key in CORE_FILES), f"{relative}: missing core hashes")
            verified = verify_hashes({key: files[key] for key in CORE_FILES}, root, relative)
        elif name == "cpu":
            require(marker.get("exit_code") == 0, f"{relative}: nonzero or missing exit code")
            require(marker.get("source_frozen") is True, f"{relative}: source was not frozen")
            require(
                marker.get("working_repo_changed_since_freeze") == [],
                f"{relative}: repository changed after engineering freeze",
            )
            verified = verify_hashes(marker.get("source_sha256"), root, relative)
        elif name == "judge":
            require(marker.get("judge_model") == JUDGE_MODEL, f"{relative}: wrong judge model")
            require(
                marker.get("judge_revision") == JUDGE_REVISION,
                f"{relative}: wrong judge revision",
            )
            require(
                marker.get("judge_enable_thinking") is False,
                f"{relative}: judge thinking is enabled",
            )
            require(
                marker.get("generated_think_tag_count") == 0,
                f"{relative}: generated thinking tags",
            )
            checks = marker.get("checks")
            require(isinstance(checks, list) and len(checks) == 8, f"{relative}: expected 8 checks")
            require(
                all(
                    item.get("valid") is True
                    and item.get("matches") is True
                    and item.get("truncated") is False
                    for item in checks
                ),
                f"{relative}: judge correctness, coverage, or truncation check failed",
            )
            verified = verify_hashes(marker.get("source_sha256"), root, relative)
            require(
                marker.get("config_sha256") == sha(root / SCIENTIFIC_CONFIG),
                f"{relative}: stale scientific config hash",
            )
        elif name == "download":
            require(
                marker.get("model") == config["judge_model"]
                and marker.get("revision") == config["judge_revision"],
                f"{relative}: downloaded model identity differs",
            )
        evidence[name] = {
            "path": str(path.resolve()),
            "sha256": sha(path),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "verified_at": utc(),
            "verified_source_sha256": verified,
        }
    return {"passed": not missing, "missing": missing, "evidence": evidence, "checked_at": utc()}


def wait_until_ready(plan, root, on_state, *, timeout=86400, poll=30, clock=None, sleep=None):
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    deadline = clock() + timeout
    while True:
        result = check_readiness(plan, root)
        on_state(dict(result, state="ready" if result["passed"] else "waiting"))
        if result["passed"]:
            return result
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Engineering prerequisites remain missing after 24 hours")
        sleep(min(poll, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.plan.resolve() == DEFAULT_PLAN.resolve(), "Only the fixed launch plan is allowed")
    args.plan = args.plan.resolve()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, ROOT)
    if not args.execute:
        print(json.dumps(dict(check_readiness(plan, ROOT), execute=False), indent=2))
        return 0
    require(socket.gethostname() == plan["host"], "Host differs from the authorized resource map")
    batch = Path(plan["output"])
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / "readiness.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("readiness-process.json", "controller-process.json", "engineering-gate.json"):
            require(not (batch / name).exists(), f"Existing {name} is preserved; do not resubmit")
        initial_hashes = {
            str(args.plan.relative_to(ROOT)): sha(args.plan),
            SCIENTIFIC_CONFIG: sha(ROOT / SCIENTIFIC_CONFIG),
            str(Path(__file__).relative_to(ROOT)): sha(__file__),
        }
        write(
            batch / "readiness-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": utc(),
                "plan": str(args.plan),
                "source_sha256": initial_hashes,
                "poll_seconds": 30,
                "timeout_seconds": 86400,
            },
        )

        def state(value):
            verify_hashes(initial_hashes, ROOT, "readiness inputs")
            write(batch / "readiness-state.json", dict(value, updated_at=utc()))
            print(
                json.dumps({"state": value["state"], "missing": value.get("missing", [])}),
                flush=True,
            )

        try:
            result = wait_until_ready(plan, ROOT, state)
            require(
                not (batch / "controller-process.json").exists(),
                "The main controller was submitted while readiness was waiting",
            )
            gate = dict(result, created_at=utc(), readiness_pid=os.getpid(), batch=BATCH)
            write(batch / "engineering-gate.json", gate)
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/launch_knowledge_method_factorial.py"),
                "--plan",
                str(args.plan),
                "--execute",
            ]
            with (batch / "readiness-launcher.log").open("x") as log:
                state({"state": "launching", "command": command})
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                write(
                    batch / "readiness-state.json",
                    {"state": "launcher_running", "launcher_pid": process.pid, "command": command},
                )
                exit_code = process.wait()
            outcome = {
                "state": "complete" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "launcher_pid": process.pid,
                "finished_at": utc(),
            }
            write(batch / "readiness-state.json", outcome)
            write(batch / "readiness-exit.json", outcome)
            return exit_code
        except Exception as error:
            outcome = {
                "state": "needs_attention" if isinstance(error, TimeoutError) else "failed",
                "exit_code": 2 if isinstance(error, TimeoutError) else 1,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "finished_at": utc(),
            }
            write(batch / "readiness-state.json", outcome)
            write(batch / "readiness-exit.json", outcome)
            print(json.dumps(outcome), flush=True)
            return outcome["exit_code"]


if __name__ == "__main__":
    os.umask(0o022)
    raise SystemExit(main())
