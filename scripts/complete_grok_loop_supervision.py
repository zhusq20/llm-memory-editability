#!/usr/bin/env python3
"""Check registered endpoints and audits, then write the completion manifest."""

import json
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grok_loop_supervision import digest

ART = Path("docs/development-artifacts/grok-loop-supervision-v1")
runs, files, configurations = [], {}, []
for phase in ("development", "followup"):
    config_path = Path(f"configs/grok-loop-supervision-{phase}-v1.json")
    config = json.loads(config_path.read_text())
    configurations.append(str(config_path))
    lock = json.loads(Path(config["source_lock"]).read_text())
    for path, sha in {**lock["files"], **lock["origins"]}.items():
        if digest(path) != sha:
            raise ValueError(f"Changed frozen dependency: {path}")
    decision = config.get("development_decision")
    if decision and digest(decision["plan"]) != decision["sha256"]:
        raise ValueError("Followup decision changed")
    for run_id, specific in config["runs"].items():
        directory = Path(config["output_root"]) / run_id
        complete = json.loads((directory / "complete.json").read_text())
        audit = json.loads((directory / "audit.json").read_text())
        learning = json.loads((directory / "learning.json").read_text())
        if complete["spec"] != {**config["base"], **specific}:
            raise ValueError("Specification mismatch")
        if [n["step"] for n in learning] != config["base"]["nodes"]:
            raise ValueError("Missing registered nodes")
        if not (audit["passed"] and complete["capture_restore_passed"]):
            raise ValueError("Audit failed")
        end = complete["endpoint"]
        runs.append(
            {
                "run": run_id,
                "phase": phase,
                "nodes": len(learning),
                "training_steps": end["step"],
                "training_seconds": complete["training_seconds"],
                "estimated_training_matmul_flops": end["estimated_training_matmul_flops"],
                "logical_examples": end["logical_examples"],
                "atomic_presentations": end["atomic_presentations"],
                "composite_presentations": end["composite_presentations"],
                "scoring_groups": audit["scoring_groups"],
                "reloaded_groups": len(audit["reload_groups"]),
                "max_cpu_nll_error": audit["max_cpu_nll_error"],
                "source": specific["source"],
            }
        )
        for name in ("latest.pt", "complete.json", "learning.json", "audit.json"):
            path = directory / name
            files[str(path)] = digest(path)
        for node in learning:
            path = directory / f"predictions-{node['step']:07d}.npz"
            if not path.exists():
                raise ValueError(f"Missing raw predictions: {path}")
    report = ART / f"{phase}-report" / "summary.json"
    summary = json.loads(report.read_text())
    if not summary["pair_initialization_and_exposure_audit"]:
        raise ValueError("Unpaired runs")
    files[str(report)] = digest(report)
dynamics = json.loads((ART / "dynamics" / "summary.json").read_text())
if dynamics["models"] != 18:
    raise ValueError("Missing finite-horizon diagnostic models")
dynamic_audit_path = ART / "dynamics" / "independent-audit.json"
dynamic_audit = json.loads(dynamic_audit_path.read_text())
if not dynamic_audit["passed"] or dynamic_audit["long_generation_groups"] != 36:
    raise ValueError("Missing long-loop independent scoring")
files[str(dynamic_audit_path)] = digest(dynamic_audit_path)
for path in sorted((ART / "dynamics").glob("w*-*.json")):
    item = json.loads(path.read_text())
    if len(item["native_audit"]) != 16:
        raise ValueError("Missing native prefix comparisons")
    if item["source_sha256"] != digest("scripts/analyze_grok_loop_supervision_dynamics.py"):
        raise ValueError("Dynamics source changed")
    if digest(item["checkpoint"]) != item["checkpoint_sha256"]:
        raise ValueError("Dynamics checkpoint changed")
    files[str(path)] = digest(path)
for name in (
    "report_grok_loop_supervision.py",
    "analyze_grok_loop_supervision_dynamics.py",
    "execute_grok_loop_supervision.py",
    "complete_grok_loop_supervision.py",
):
    path = Path("scripts") / name
    files[str(path)] = digest(path)
totals = {
    k: sum(r[k] for r in runs)
    for k in (
        "nodes",
        "training_steps",
        "training_seconds",
        "estimated_training_matmul_flops",
        "logical_examples",
        "atomic_presentations",
        "composite_presentations",
        "scoring_groups",
        "reloaded_groups",
    )
}
write_json(
    ART / "completion-manifest.json",
    {
        "completed_utc": utc(),
        "runs": len(runs),
        "development_runs": 2,
        "followup_runs": 12,
        "configurations": configurations,
        "totals": totals,
        "run_details": runs,
        "files": files,
        "dynamics_models": 18,
        "dynamics_independent_audit": dynamic_audit,
        "interpretation": (
            "Fixed-budget continuation; existing worlds; finite-horizon dynamics only."
        ),
    },
)
print(json.dumps(totals, indent=2))
