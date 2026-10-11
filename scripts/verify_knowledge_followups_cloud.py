"""Read-only W&B endpoint verification after the six-lane development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import netrc
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def check(base):
    import wandb

    cfg = read(base / "frozen/config.json")
    api = wandb.Api(timeout=30)
    checks = []
    for task in cfg["execution"]["tasks"]:
        out = base / "runs" / task["name"]
        summary = read(out / "summary.json")
        is_edit = cfg["study_kind"] == "planned_U1_development_only"
        run_id = read(out / "run.json")["wandb_run_id"] if is_edit else summary["wandb"]["id"]
        remote = api.run(f"{cfg['tracking']['entity']}/{cfg['tracking']['project']}/{run_id}")
        if remote.state != "finished":
            raise ValueError(f"Cloud run is {remote.state}: {task['name']}")
        if remote.config.get("config_sha256") != sha(base / "frozen/config.json"):
            raise ValueError(f"Cloud config identity differs: {task['name']}")
        if is_edit:
            audit = read(out / "audit.json")
            if audit.get("passed") is not True or remote.summary.get("complete") is not True:
                raise ValueError(f"Missing successful endpoint: {task['name']}")
            if remote.summary.get("updates") != summary["updates"]:
                raise ValueError(f"Cloud update count differs: {task['name']}")
            compared = 2
        else:
            if read(out / "audit.json").get("passed") is not True:
                raise ValueError(f"Missing independent audit: {task['name']}")
            if remote.summary.get("optimizer_updates") != 0:
                raise ValueError("Unexpected new optimization updates")
            compared = 1
            for label in ("primary", "same_bridge"):
                for mode, kinds in summary[label].items():
                    for kind, values in kinds.items():
                        for metric in ("T", "R", "delta", "facts", "eligible_queries"):
                            value = values[metric]
                            if value is None:
                                continue
                            key = f"{label}/{kind}/{mode}/{metric}"
                            actual = remote.summary.get(key)
                            if actual is None or not math.isclose(
                                actual, value, rel_tol=1e-6, abs_tol=1e-8
                            ):
                                raise ValueError(f"Cloud metric differs: {key}")
                            compared += 1
        checks.append(
            {
                "task": task["name"],
                "run_id": run_id,
                "state": remote.state,
                "passed": True,
                "metrics_checked": compared,
                "url": remote.url,
            }
        )
    return {
        "passed": True,
        "checks": checks,
        "read_only_cloud_api": True,
        "new_evaluations": 0,
        "new_optimizer_updates": 0,
        "completion_manifest_sha256": sha(base / "completion-manifest.json"),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    credentials = netrc.netrc("/ossfs/workspace/.lm-memory-credentials/wandb.netrc").authenticators(
        "api.wandb.ai"
    )
    if credentials:
        os.environ["WANDB_API_KEY"] = credentials[2]
    deadline = time.monotonic() + 24 * 3600
    while not (base / "completion-manifest.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Local completion did not arrive")
        time.sleep(15)
    if read(base / "completion-manifest.json").get("passed") is not True:
        write(
            base / "cloud-verification.json",
            {"passed": False, "reason": "Local audit/completion failed; no cloud completion claim"},
        )
        return
    # Allow the native trackers to drain; these are read-only checks, no reruns.
    last_error = None
    for _ in range(20):
        try:
            result = check(base)
            write(base / "cloud-verification.json", result)
            return
        except Exception as error:
            last_error = repr(error)
            time.sleep(15)
    write(base / "cloud-verification.json", {"passed": False, "reason": last_error})
    raise RuntimeError("Independent cloud endpoint verification failed")


if __name__ == "__main__":
    main()
