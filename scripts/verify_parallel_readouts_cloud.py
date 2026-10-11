"""Independently check finished readout runs and their native summaries on W&B."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from track_parallel_readout import flatten


def metric_matches(expected, observed):
    """Cloud JSON may round the last binary64 digit; counts remain exact."""
    if isinstance(expected, bool):
        return isinstance(observed, bool) and expected == observed
    if isinstance(expected, int):
        return not isinstance(observed, bool) and expected == observed
    return (
        isinstance(observed, int | float)
        and not isinstance(observed, bool)
        and math.isclose(expected, observed, rel_tol=1e-12, abs_tol=1e-12)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    completion = json.loads((base / "completion-manifest.json").read_text())
    if completion.get("passed") is not True:
        raise RuntimeError("Native independent replay must complete before cloud verification")
    config_path = base / "frozen/config.json"
    config = json.loads(config_path.read_text())
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    import wandb

    api = wandb.Api(timeout=60)
    rows = []
    for task in config["execution"]["tasks"]:
        out = base / "runs" / task["name"]
        tracking = json.loads((out / "tracking.json").read_text())
        native = json.loads((out / "summary.json").read_text())
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['run_id']}")
        cloud = dict(run.summary)
        expected = {"exit_code": 0, "new_optimizer_updates": 0, **flatten(native, "readout")}
        mismatch = {
            key: {"native": value, "cloud": cloud.get(key)}
            for key, value in expected.items()
            if not metric_matches(value, cloud.get(key))
        }
        identity_pass = (
            run.config.get("config_sha256") == config_sha
            and run.config.get("task") == task["name"]
            and run.config.get("new_optimizer_updates") == 0
        )
        rows.append(
            {
                "task": task["name"],
                "run_id": run.id,
                "url": run.url,
                "state": run.state,
                "config_match": identity_pass,
                "metric_count": len(expected),
                "mismatches": mismatch,
                "passed": run.state == "finished" and identity_pass and not mismatch,
            }
        )
    receipt = {
        "passed": all(row["passed"] for row in rows),
        "batch": config["batch"],
        "new_optimizer_updates": 0,
        "native_completion_sha256": hashlib.sha256(
            (base / "completion-manifest.json").read_bytes()
        ).hexdigest(),
        "config_sha256": config_sha,
        "cloud_float_comparison": {
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "integers_and_booleans": "exact",
        },
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": rows,
    }
    (base / "cloud-verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "runs": len(rows),
                "metrics": sum(row["metric_count"] for row in rows),
            }
        )
    )
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
