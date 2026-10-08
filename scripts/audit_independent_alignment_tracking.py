"""Verify remote W&B nodes, scientific final step and independent reload markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json


def audit(config_path):
    import wandb

    config = json.loads(Path(config_path).read_text())
    root = Path(config["repository"]) / "results" / config["batch"]
    defaults = json.loads(
        (Path(config["source_root"]) / "configs/experiment-tracking-defaults.json").read_text()
    )
    api = wandb.Api(timeout=60)
    records = []
    for spec in config["specs"]:
        path = root / "tracking-wandb" / spec["name"] / "state.json"
        if not path.exists():
            records.append(dict(name=spec["name"], passed=False, reason="missing_tracking_state"))
            continue
        local = json.loads(path.read_text())
        remote = api.run(f"{defaults['entity']}/{defaults['project']}/{local['run_id']}")
        history = list(remote.scan_history(keys=["training/step"]))
        steps = sorted({int(r["training/step"]) for r in history})
        passed = (
            set(steps) == set(spec["nodes"])
            and remote.state == "finished"
            and remote.summary.get("independently_reloaded") is True
            and remote.summary.get("scientific_final_step") == spec["steps"]
        )
        records.append(
            dict(
                name=spec["name"],
                passed=passed,
                steps=steps,
                state=remote.state,
                independently_reloaded=remote.summary.get("independently_reloaded"),
                scientific_final_step=remote.summary.get("scientific_final_step"),
                url=local["url"],
            )
        )
    result = dict(passed=all(r["passed"] for r in records), runs=records, utc=utc())
    write_json(root / "tracking-cloud-audit.json", result)
    print(json.dumps(dict(passed=result["passed"], runs=len(records))))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    audit(parser.parse_args().config)
