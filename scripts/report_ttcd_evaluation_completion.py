#!/usr/bin/env python3
"""Complete the existing one-step/four-step comparison after audited recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/ossfs/workspace/llm-memory-editability")
RECOVERY = ROOT / "evaluation-recovery-20261016"
TRAINED = ROOT / "ttcd-one-step-training-v1"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 48 * 3600
    while True:
        recovery = read(RECOVERY / "controller-state.json")
        if recovery["state"] == "complete":
            break
        if recovery["state"] == "needs_attention":
            raise RuntimeError("Evaluation recovery needs diagnosis before reporting")
        if not args.wait or time.monotonic() >= deadline:
            raise RuntimeError("Audited evaluation recovery has not completed")
        time.sleep(30)
    path = TRAINED / "transformer/source/scripts/launch_ttcd_one_step_training.py"
    spec = importlib.util.spec_from_file_location("frozen_completed_training_report", path)
    controller = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(controller)
    helpers = controller.resources
    plan = read(TRAINED / "transformer/source/configs/one-step-training.json")
    controller.validate_plan(plan)
    protected = {}
    for arm in controller.ARCHITECTURES:
        controller.transport.verify_snapshot(TRAINED, arm)
        state = read(TRAINED / arm / "controller-state.json")
        if state["state"] != "complete" or "cloud-audit" not in state["completed_stages"]:
            raise ValueError("All original one-step-trained arms must already be complete")
        for filename in ("training/parent.pt", "training/audit.json"):
            checkpoint = TRAINED / arm / "training-run" / filename
            protected[str(checkpoint)] = helpers.sha256(checkpoint)
    main_report = TRAINED / "report/result-aggregate.json"
    protected[str(main_report)] = helpers.sha256(main_report)
    receipt = controller.four_step_control_receipt(TRAINED)
    if receipt is None:
        raise ValueError("All three original four-step-trained controls require local/cloud audits")
    metadata = {
        "state": "running",
        "pid": os.getpid(),
        "repair": "evaluation-recovery-20261016",
        "protected": protected,
        "new_training_updates": 0,
    }

    def update(values):
        metadata.update(values)
        metadata["utc"] = helpers.utc_now()
        helpers.write_json(RECOVERY / "summary-state.json", metadata)

    update({})
    helpers.write_json(TRAINED / "four-step-control-receipt.json", receipt)
    controller.run_stage(
        plan,
        "transformer",
        "aggregate",
        TRAINED,
        Path.home() / ".cache/llm-memory-ttcd/wandb-token",
        update,
        report_directory="control-report",
        report_variant="control-aggregate",
    )
    output = TRAINED / "control-report/result-aggregate.json"
    result = read(output)
    if result["four_step_training_control"]["status"] != "complete":
        raise ValueError("The separate aggregate must contain the complete audited controls")
    for filename, expected in protected.items():
        if helpers.sha256(filename) != expected:
            raise ValueError("Existing training or original main report changed")
    completion = {
        "state": "complete",
        "utc": helpers.utc_now(),
        "new_base_parameter_updates": 30000,
        "new_training_updates_during_recovery": 0,
        "original_main_report_unchanged": True,
        "protected_evidence_unchanged": True,
        "main_report": {"path": str(main_report), "sha256": protected[str(main_report)]},
        "control_report": {"path": str(output), "sha256": helpers.sha256(output)},
        "all_three_arms_local_and_cloud_audited": True,
    }
    helpers.write_json(TRAINED / "controller-state.json", completion)
    helpers.write_json(TRAINED / "completion-manifest.json", completion)
    update({"state": "complete", **completion})
    print(json.dumps(completion))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
