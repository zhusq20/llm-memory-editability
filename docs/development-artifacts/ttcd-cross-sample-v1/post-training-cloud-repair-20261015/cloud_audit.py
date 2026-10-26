"""Read-only W&B verification, independent of the frozen failing cloud entrypoint.

Canonical uploaded wandb-summary.json preserves nested dictionaries and arrays.
HTTP Summary flattens dictionaries and can encode arrays differently, so compare
the canonical result in full and separately check every non-null HTTP scalar.
Nothing imports, trains, evaluates, updates or resumes an experimental model.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

STAGES = ("train", "train-audit", "calibrate", "evaluate", "audit")
ENTITY, PROJECT = "zhusq20", "llm-memory-editability"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def compare(expected, actual, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"Canonical dictionary differs at {path}")
        for key, value in expected.items():
            compare(value, actual[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"Canonical list differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, f"{path}/{i}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            expected, actual, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise AssertionError(f"Numeric value differs at {path}")
    elif expected != actual:
        raise AssertionError(f"Value differs at {path}")


def scalar_leaves(value, prefix="result"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalar_leaves(child, f"{prefix}.{key}")
    elif value is not None and not isinstance(value, list):
        yield prefix, value


def request(method, url, **kwargs):
    try:
        result = requests.request(method, url, timeout=60, **kwargs)
    except requests.RequestException as error:
        raise RuntimeError(f"Read-only cloud request failed: {type(error).__name__}") from None
    if not result.ok:
        raise RuntimeError(f"Read-only cloud request returned HTTP {result.status_code}")
    return result


def fetch_run(run_id, token):
    query = """query($entity:String!, $project:String!, $run:String!) {
      project(name:$project,entityName:$entity) { run(name:$run) {
        name state config summaryMetrics
        files(first:10,names:["wandb-summary.json"]) {
          edges { node { name url md5 sizeBytes } } pageInfo { hasNextPage }
        }
      } }
    }"""
    response = request(
        "POST",
        "https://api.wandb.ai/graphql",
        auth=("api", token),
        json={"query": query, "variables": {"entity": ENTITY, "project": PROJECT, "run": run_id}},
    ).json()
    if response.get("errors"):
        raise ValueError("W&B GraphQL returned a read-query error")
    run = response["data"]["project"]["run"]
    if run is None:
        raise ValueError("Original W&B run is unavailable")
    return run


def local_ready(batch, arm):
    directory = batch / arm
    paths = [directory / f"result-{stage}.json" for stage in STAGES]
    paths += [directory / "audit.json", directory / "training/audit.json"]
    paths += [directory / f"exit-{stage}.json" for stage in STAGES]
    if not all(path.is_file() for path in paths):
        return False
    for stage in STAGES:
        if json.loads((directory / f"exit-{stage}.json").read_text())["returncode"] != 0:
            raise ValueError(f"Original stage did not exit successfully: {arm}/{stage}")
    for relative in ("audit.json", "training/audit.json"):
        if not json.loads((directory / relative).read_text())["passed"]:
            raise ValueError(f"Local audit did not pass for {arm}/{relative}")
    return True


def verify_arm(config, batch, arm, out, token):
    if not local_ready(batch, arm):
        raise ValueError("All original local stages and independent audits are required")
    attempt = (
        out
        / arm
        / (
            datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    attempt.mkdir(parents=True, exist_ok=False)
    directory, rows = batch / arm, []
    original_paths = [directory / f"result-{stage}.json" for stage in STAGES]
    original_paths += [directory / f"wandb-{stage}.json" for stage in STAGES]
    original_paths += [
        p
        for p in (directory / "failure-cloud-audit.json", directory / "exit-cloud-audit.json")
        if p.exists()
    ]
    hashes = {str(path): sha(path) for path in original_paths}
    write(
        attempt / "started.json",
        {
            "utc": utc(),
            "arm": arm,
            "batch": config["batch"],
            "original_hashes": hashes,
            "validator_sha256": sha(__file__),
            "read_only": True,
            "GPU_used": False,
            "new_model_updates": 0,
        },
    )
    try:
        for stage in STAGES:
            result_path = directory / f"result-{stage}.json"
            expected = json.loads(result_path.read_text())
            metadata = json.loads((directory / f"wandb-{stage}.json").read_text())
            run = fetch_run(metadata["id"], token)
            api_config = json.loads(run["config"])
            plain_config = {
                k: v.get("value", v) if isinstance(v, dict) else v for k, v in api_config.items()
            }
            if (
                run["name"] != metadata["id"]
                or run["state"] != "finished"
                or plain_config.get("batch") != config["batch"]
                or plain_config.get("arm") != arm
                or plain_config.get("stage") != stage
            ):
                raise ValueError(f"Original cloud run identity or completion differs: {stage}")
            api_summary = json.loads(run["summaryMetrics"])
            files = run["files"]["edges"]
            if len(files) != 1 or run["files"]["pageInfo"]["hasNextPage"]:
                raise ValueError("Exactly one canonical wandb-summary.json is required")
            file = files[0]["node"]
            if file["name"] != "wandb-summary.json":
                raise ValueError("Unexpected canonical summary file name")
            host = urlparse(file["url"]).hostname or ""
            auth = (
                {"auth": ("api", token)} if host == "wandb.ai" or host.endswith(".wandb.ai") else {}
            )
            content = request("GET", file["url"], **auth).content
            if len(content) != int(file["sizeBytes"]):
                raise ValueError("Canonical cloud summary size differs")
            md5 = base64.b64encode(hashlib.md5(content).digest()).decode()
            if md5 != file["md5"]:
                raise ValueError("Canonical cloud summary MD5 differs")
            stage_dir = attempt / stage
            stage_dir.mkdir()
            (stage_dir / "wandb-summary.json").write_bytes(content)
            canonical = json.loads(content)
            compare(expected, canonical["result"])
            scalar_count = 0
            for key, value in scalar_leaves(expected):
                if key not in api_summary:
                    raise AssertionError(f"HTTP scalar missing: {key}")
                compare(value, api_summary[key], key)
                scalar_count += 1
            write(stage_dir / "api-summary-scalars.json", api_summary)
            if stage == "train":
                if api_summary.get("training/step") != config["models"][arm]["step"]:
                    raise ValueError("Original cloud terminal global step differs")
                if api_summary.get("exposure/base_parameter_updates") != config["parent"]["steps"]:
                    raise ValueError("Original cloud additional update count differs")
            row = {
                "stage": stage,
                "run_id": metadata["id"],
                "url": metadata["url"],
                "state": run["state"],
                "canonical_result_matches_original": True,
                "canonical_md5": md5,
                "canonical_sha256": hashlib.sha256(content).hexdigest(),
                "original_result_sha256": hashes[str(result_path)],
                "HTTP_scalar_fields_matched": scalar_count,
            }
            rows.append(row)
            write(stage_dir / "receipt.json", row)
        for path, digest in hashes.items():
            if sha(path) != digest:
                raise ValueError("An original local result changed during verification")
        result = {
            "passed": True,
            "arm": arm,
            "batch": config["batch"],
            "utc": utc(),
            "status": "original_cloud_stages_verified_with_canonical_files_and_HTTP_scalars",
            "stages": rows,
            "terminal_training_step": config["models"][arm]["step"],
            "additional_updates_verified": config["parent"]["steps"],
            "local_audit": json.loads((directory / "audit.json").read_text()),
            "original_hashes_preserved": True,
            "original_frozen_source_modified": False,
            "new_training_or_evaluation": False,
            "GPU_used": False,
            "repair_attempt": str(attempt),
        }
        write(attempt / "receipt.json", result)
        write(out / arm / "cloud-audit.json", result)
        return result
    except Exception:
        write(
            attempt / "failure.json",
            {
                "passed": False,
                "utc": utc(),
                "arm": arm,
                "completed_stages": rows,
                "traceback": traceback.format_exc(),
                "read_only": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("single", "stream"), required=True)
    parser.add_argument(
        "--token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("Polling interval must be between 1 and 60 seconds")
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"controller-{args.model}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = {
            "pid": os.getpid(),
            "arm": args.model,
            "utc": utc(),
            "validator_sha256": sha(__file__),
            "config_sha256": sha(args.config),
            "old_batch": str(args.batch),
            "new_results": str(args.out),
            "wait_for_local_independent_audit": args.wait,
            "GPU_used": False,
        }
        write(args.out / f"controller-{args.model}-process.json", process)
        try:
            if args.wait:
                write(
                    args.out / f"controller-{args.model}-state.json",
                    {**process, "state": "waiting_for_original_local_independent_audit"},
                )
                while not local_ready(args.batch, args.model):
                    for stage in STAGES:
                        exit_path = args.batch / args.model / f"exit-{stage}.json"
                        if exit_path.is_file() and json.loads(exit_path.read_text())["returncode"]:
                            raise ValueError("Original scientific stage failed; no automatic rerun")
                    time.sleep(args.poll_seconds)
            result = verify_arm(
                config, args.batch, args.model, args.out, args.token_file.read_text().strip()
            )
        except Exception:
            failure = {
                **process,
                "state": "failed",
                "passed": False,
                "utc": utc(),
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
            }
            write(
                args.out / f"controller-{args.model}-failure-{uuid.uuid4().hex[:8]}.json", failure
            )
            write(args.out / f"controller-{args.model}-state.json", failure)
            raise
        write(
            args.out / f"controller-{args.model}-state.json",
            {
                **process,
                "state": "complete",
                "passed": True,
                "finished_utc": utc(),
                "receipt": str(args.out / args.model / "cloud-audit.json"),
            },
        )
        print(
            json.dumps(
                {
                    "arm": args.model,
                    "passed": result["passed"],
                    "stages": len(result["stages"]),
                    "GPU_used": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
