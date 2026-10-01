"""Record completion only after all frozen runs and reload audits are present."""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from llm_memory_editability.memory_reuse import sha, write_json

root = Path("docs/development-artifacts/memory-reuse-v1")
phases = {}
files = {}
records = []
for phase in ("development", "calibration", "confirmation"):
    config_path = Path(f"configs/memory-reuse-{phase}-v1.json")
    config = json.loads(config_path.read_text())
    artifact = Path(config["lock"]).parent
    audit = json.loads((artifact / "audit.json").read_text())
    report = json.loads((artifact / "report/summary.json").read_text())
    assert audit["status"] == "passed" and report["status"] == "complete"
    assert report["audit_sha256"] == sha(artifact / "audit.json")
    expected = list(itertools.product(config["worlds"], config["initializations"], config["arms"]))
    assert len(audit["runs"]) == len(expected) == report["runs"]
    for world, init, arm in expected:
        directory = Path(config["output_root"]) / f"w{world}-i{init}-{arm}"
        result = json.loads((directory / "summary.json").read_text())
        assert result["status"] == "complete"
        record = next(
            row
            for row in audit["runs"]
            if (row["world"], row["init"], row["arm"]) == (world, init, arm)
        )
        assert record["summary_sha256"] == sha(directory / "summary.json")
        metadata = json.loads((directory / "metadata.json").read_text())
        assert metadata["lock_sha256"] == sha(config["lock"])
        assert not (directory / "failure.json").exists()
        for path in directory.iterdir():
            if path.suffix in (".json", ".pt", ".npz"):
                files[str(path)] = sha(path)
        records.append(
            dict(
                phase=phase, run=directory.name, budget=result["budget"], seconds=result["seconds"]
            )
        )
    phases[phase] = dict(
        runs=len(expected),
        audit_predictions=audit["total_reload_predictions"],
        max_reload_error=audit["max_logit_error"],
        summary=str(artifact / "report/summary.json"),
        budget=report["budget"],
        means=report["means"],
    )
    for path in (
        config_path,
        Path(config["lock"]),
        artifact / "audit.json",
        artifact / "report/summary.json",
        artifact / "report/comparison.png",
        artifact / "report/comparison.pdf",
        config["plan"],
    ):
        files[str(path)] = sha(path)
files[str(root / "environment.json")] = sha(root / "environment.json")
for name in ("scripts/audit_memory_reuse.py", "scripts/report_memory_reuse.py", __file__):
    files[name] = sha(name)
budget_keys = (
    "memory_updates",
    "reused_memory_updates",
    "reader_updates",
    "atomic_exposures",
    "reader_input_tokens",
    "reader_supervised_answers",
    "approximate_matmul_flops",
)
manifest = dict(
    status="complete",
    utc=datetime.now(timezone.utc).isoformat(),
    phases=phases,
    runs=records,
    total_budget={k: sum(r["budget"].get(k, 0) for r in records) for k in budget_keys},
    total_process_seconds=sum(r["seconds"] for r in records),
    total_reload_predictions=sum(p["audit_predictions"] for p in phases.values()),
    unique_fitted_fact_mlps=28,
    fitted_readers=16,
    saved_reader_nodes=112,
    notes=[
        "Calibration reused the four development fact MLPs; reused updates are not new training.",
        "One pre-training launch was rejected by the absent lock; no scientific run failed.",
        "Process seconds include evaluation/save and concurrent jobs; not exclusive GPU time.",
        "FLOPs are approximate matrix-multiply estimates, not hardware profiling.",
    ],
    files=files,
)
write_json(root / "completion-manifest.json", manifest)
print(
    json.dumps(
        {
            k: manifest[k]
            for k in ("status", "total_budget", "total_process_seconds", "total_reload_predictions")
        },
        indent=2,
    )
)
