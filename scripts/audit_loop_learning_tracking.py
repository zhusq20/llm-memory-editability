"""Check online W&B records against every frozen evaluation node and local audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def expected_steps(spec):
    return sorted(
        set(spec.get("evaluation_nodes", []))
        | {0, spec["stage_a_steps"], spec["stage_a_steps"] + spec["stage_b_steps"]}
    )


def check_run(api, root, spec, defaults):
    """Return evidence even for unfinished or absent runs, without exposing credentials."""
    root = Path(root)
    out = root / "runs" / spec["name"]
    local_path = root / "tracking-wandb" / spec["name"] / "state.json"
    result = {"name": spec["name"], "passed": False, "expected_steps": expected_steps(spec)}
    required = {
        "tracking_state": local_path,
        "learning": out / "learning.json",
        "independent_audit": out / "audit.json",
        "complete": out / "complete.json",
    }
    missing = [key for key, path in required.items() if not path.exists()]
    if missing:
        return {**result, "reason": "missing_local_artifacts", "missing": missing}
    local = read(local_path)
    local_steps = [int(row["step"]) for row in read(out / "learning.json")]
    audit = read(out / "audit.json")
    remote = api.run(f"{defaults['entity']}/{defaults['project']}/{local['run_id']}")
    remote_steps = [
        int(row["training/step"]) for row in remote.scan_history(keys=["training/step"])
    ]
    final_step = result["expected_steps"][-1]
    checks = {
        "local_online": local.get("mode") == "online",
        "entity_project_match": local.get("entity") == defaults["entity"]
        and local.get("project") == defaults["project"],
        "local_nodes_exact": local_steps == result["expected_steps"],
        "remote_nodes_exact": sorted(remote_steps) == result["expected_steps"],
        "remote_finished": remote.state == "finished",
        "local_independent_reload": audit.get("passed") is True,
        "remote_independent_reload": remote.summary.get("independently_reloaded") is True,
        "remote_final_step": remote.summary.get("scientific_final_step") == final_step,
        "local_cursor_final": local.get("last_logged_step") == final_step,
        "no_local_failure": not (out / "failure.json").exists(),
        "no_remote_failure": remote.summary.get("has_failure_record") is False,
    }
    return {
        **result,
        "passed": all(checks.values()),
        "checks": checks,
        "local_steps": local_steps,
        "remote_steps": remote_steps,
        "remote_state": remote.state,
        "url": local.get("url"),
    }


def audit(config_path, root=None, api=None):
    config = read(config_path)
    root = Path(root or config["results_root"])
    defaults = read(Path(config["source_root"]) / "configs/experiment-tracking-defaults.json")
    if api is None:
        import wandb

        api = wandb.Api(timeout=60)
    rows = [check_run(api, root, spec, defaults) for spec in config["specs"]]
    result = {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "expected_runs": len(config["specs"]),
        "runs": rows,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    path = root / "tracking-cloud-audit.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    result = audit(args.config, args.root)
    print(json.dumps({"passed": result["passed"], "runs": len(result["runs"])}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
