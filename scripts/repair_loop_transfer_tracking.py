#!/usr/bin/env python3
"""Restore only empty configuration dictionaries omitted by W&B serialization.

The mutation writes configuration alone, never summary, history, or run state.
Frozen training and endpoint audit code remain unchanged.
"""

import argparse
import json
import os
import time
from pathlib import Path

from launch_loop_transfer import digest, utc, write


def empty_dictionary_omissions(expected, actual, path="experiment"):
    if expected == actual:
        return []
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError(f"Nonempty configuration differs at {path}")
    if set(actual) - set(expected):
        raise ValueError(f"Unexpected configuration keys at {path}")
    missing = []
    for key, value in expected.items():
        child = f"{path}.{key}"
        if key not in actual:
            if value != {}:
                raise ValueError(f"Nonempty configuration missing at {child}")
            missing.append(child)
        else:
            missing.extend(empty_dictionary_omissions(value, actual[key], child))
    return missing


def reconcile(base, task):
    import wandb

    job = task["job"]
    output = base / "cases" / job / "runs" / task["arm"]
    if not (output / "tracking.json").exists():
        return
    receipt = base / "tracking-repair" / f"{job}.json"
    if receipt.exists():
        return
    config_path = Path(task["source"]) / task["config"]
    assert digest(config_path) == task["config_sha256"]
    expected = json.loads(config_path.read_text())
    tracking = json.loads((output / "tracking.json").read_text())
    remote_path = (
        f"{expected['tracking']['entity']}/{expected['tracking']['project']}/{tracking['id']}"
    )
    api = wandb.Api(timeout=60)
    run = api.run(remote_path)
    before = run.config.get("experiment")
    omissions = empty_dictionary_omissions(expected, before)
    if omissions:
        write(
            base / "tracking-repair" / f"{job}-before.json",
            {
                "utc": utc(),
                "experiment": before,
                "omissions": omissions,
                "expected_config_sha256": task["config_sha256"],
            },
        )
        run.config["experiment"] = expected
        # Run.update() also writes summary, so deliberately use config-only mutation.
        run._exec(
            "mutation RestoreEmptyConfig($id: String!, $config: JSONString!) {"
            " upsertBucket(input: {id: $id, config: $config}) { bucket { id } } }",
            id=run.storage_id,
            config=run.json_config,
        )
    after = wandb.Api(timeout=60).run(remote_path)
    if after.config.get("experiment") != expected:
        raise RuntimeError(f"Exact configuration verification failed: {job}")
    write(
        receipt,
        {
            "utc": utc(),
            "passed": True,
            "url": tracking["url"],
            "restored_empty_dictionaries": omissions,
            "exact_configuration_match": True,
            "history_summary_state_modified": False,
            "frozen_scientific_files_modified": False,
        },
    )


def main():
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    tasks = json.loads((args.base / "tasks.json").read_text())["training_jobs"]
    while True:
        for task in tasks:
            try:
                reconcile(args.base, task)
            except wandb.Error as error:
                # Transient API outages must not discard future jobs' reconciliation.
                write(
                    args.base / "tracking-repair" / f"{task['job']}-api-error.json",
                    {
                        "utc": utc(),
                        "type": type(error).__name__,
                        "retry_after_seconds": 30,
                    },
                )
        state = json.loads((args.base / "controller-state.json").read_text())["state"]
        write(args.base / "tracking-repair/status.json", {"utc": utc(), "queue_state": state})
        if state in ("complete", "failed"):
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
