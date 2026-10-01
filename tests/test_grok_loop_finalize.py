"""CPU finalization boundaries: no partial completion, identities and native scans."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_grok_loop", ROOT / "scripts/finalize_grok_loop.py"
)
final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final)


def training_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    monkeypatch.setattr(final.post, "ROOT", tmp_path)
    spec = {
        "phase": "confirmation",
        "architecture": "l1",
        "steps": 2,
        "nodes": [0, 2],
        "weight_nodes": [0, 2],
        "batch_size": 256,
        "n_atomic_per_batch": 32,
        "hops": 2,
        "width": 8,
        "entities": 4,
        "relations": 2,
        "layers": 1,
        "repeats": 2,
    }
    directory = tmp_path / "results/confirmation/example"
    directory.mkdir(parents=True)
    source = tmp_path / "configs/fixture.json"
    final.post.write(source, {"fixture": True})
    sha = final.post.digest(source)
    final.post.write(directory / "source/configs/fixture.json", {"fixture": True})
    meta = {
        "spec": spec,
        "files": {str(source): sha},
        "visible_devices": "5",
        "started_utc": "2026-09-30T00:00:00+00:00",
    }
    history = []
    for step in spec["nodes"]:
        history.append(
            {
                "step": step,
                "counts": {"atomic": 32 * step, "composite": 224 * step},
                "examples": 256 * step,
                "supervised_tokens": 512 * step,
                "effective_input_tokens": step * (32 * 3 + 224 * 4),
                "estimated_training_flops": step * final.independent_flops_per_step(spec),
                "training_seconds": step / 2,
                "evaluation_seconds": (step + 1) / 10,
            }
        )
        np.savez(directory / f"predictions-{step:07d}.npz", fixture=[1])
        (directory / f"weights-{step:07d}.pt").write_bytes(b"not-loaded-confirmation-fixture")
    complete = {
        "spec": spec,
        "endpoint": history[-1],
        "finished_utc": "2026-09-30T00:00:05+00:00",
        "training_seconds": 1.0,
        "evaluation_seconds": 0.3,
    }
    for name, value in (
        ("metadata.json", meta),
        ("complete.json", complete),
        ("learning.json", history),
        ("environment.json", {"gpu": "fixture"}),
        ("world-metadata.json", {"dataset_sha256": "world"}),
    ):
        final.post.write(directory / name, value)
    job = {
        "run": "example",
        "spec": spec,
        "directory": str(directory),
        "output_root": str(tmp_path / "results"),
    }
    return job, {"files": {"configs/fixture.json": sha}}


def test_current_registration_has_exact_training_mechanism_and_scan_counts():
    configs = [json.loads((ROOT / path).read_text()) for path in final.CONFIGS]
    specs = [{**cfg["base"], **item} for cfg in configs for item in cfg["runs"].values()]
    assert len(specs) == 111
    assert sum(len(spec["nodes"]) for spec in specs) == 2997
    assert 2 * sum("sensitivity" not in spec["phase"] for spec in specs) == 210
    assert (
        sum(
            "sensitivity" not in spec["phase"] and spec["architecture"] in {"l1", "l2"}
            for spec in specs
        )
        == 42
    )


@pytest.mark.parametrize(
    "change", ["missing_prediction", "missing_weight", "node_order", "counts", "spec", "source"]
)
def test_training_rejects_missing_or_changed_registered_evidence(tmp_path, monkeypatch, change):
    job, lock = training_fixture(tmp_path, monkeypatch)
    verification = final.Verification()
    bound, cost = final.verify_training(job, lock, verification)
    assert cost["evaluation_nodes"] == 2 and bound["checkpoint_sha256"]
    directory = Path(job["directory"])
    if change == "missing_prediction":
        (directory / "predictions-0000000.npz").unlink()
    elif change == "missing_weight":
        (directory / "weights-0000000.pt").unlink()
    elif change in {"node_order", "counts"}:
        history = final.read(directory / "learning.json")
        if change == "node_order":
            history.reverse()
        else:
            history[0]["counts"]["atomic"] = 1
        final.post.write(directory / "learning.json", history)
    elif change == "spec":
        complete = final.read(directory / "complete.json")
        complete["spec"]["steps"] = 3
        final.post.write(directory / "complete.json", complete)
    else:
        (directory / "source/configs/fixture.json").write_text("changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        final.verify_training(job, lock, verification)


def test_old_analysis_lock_checks_snapshot_and_new_lock_checks_current(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    current = tmp_path / "scripts/scan.py"
    current.parent.mkdir()
    current.write_text("new implementation")
    lock = tmp_path / "analysis-lock.json"
    frozen = tmp_path / "analysis-lock-source/scripts/scan.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("old implementation")
    final.post.write(lock, {"files": {"scripts/scan.py": final.post.digest(frozen)}})
    assert final.verify_lock(lock, final.Verification(), current=False)
    with pytest.raises(ValueError, match="Locked source changed"):
        final.verify_lock(lock, final.Verification())
    frozen.write_text("tampered")
    with pytest.raises(ValueError, match="snapshot changed"):
        final.verify_lock(lock, final.Verification(), current=False)


def scan_fixture(tmp_path, monkeypatch):
    job, lock = training_fixture(tmp_path, monkeypatch)
    job["spec"]["phase"] = "development"
    job.update(
        checkpoint="weights.pt", checkpoint_sha256="checkpoint", training_complete_sha256="complete"
    )
    report = {"finished_utc": "now", "repeats": list(range(1, 9))}
    run = {
        "run": job["run"],
        "spec": job["spec"],
        "checkpoint": {"weights.pt": "checkpoint"},
        "measurements": [
            {
                "repeats": r,
                **{split: {"n": 2, "accuracy": 1.0} for split in ("atomic", *final.post.SPLITS)},
            }
            for r in range(1, 9)
        ],
    }
    predictions, native = {}, {}
    for split in ("atomic", *final.post.SPLITS):
        for name, value in {
            "answer": [2, 3],
            "stop": [1, 1],
            "target": [2, 3],
            "nll": [[0.1, 0.2], [0.3, 0.4]],
        }.items():
            native[split + "_" + name] = np.array(value)
            for r in range(1, 9):
                predictions[f"r{r}_{split}_{name}"] = np.array(value)
    out = tmp_path / "scan"
    out.mkdir()
    np.savez(out / "example.npz", **predictions)
    np.savez(Path(job["directory"]) / "predictions-0000002.npz", **native)
    return out / "summary.json", report, run, job, predictions


def test_legacy_scan_native_outputs_are_recounted_on_cpu(tmp_path, monkeypatch):
    path, report, run, job, predictions = scan_fixture(tmp_path, monkeypatch)
    result = final.verify_recurrence(path, report, run, job, final.Verification())
    assert result["passed"] and result["legacy_development_provenance"]
    assert all(item["passed"] for item in result["native_cpu_recount"].values())
    predictions["r2_test_composite_stop"][0] = 0
    np.savez(path.parent / "example.npz", **predictions)
    with pytest.raises(ValueError, match="accuracy|predictions"):
        final.verify_recurrence(path, report, run, job, final.Verification())


def test_formal_scan_cannot_use_legacy_provenance_or_omit_a_repeat(tmp_path, monkeypatch):
    path, report, run, job, _ = scan_fixture(tmp_path, monkeypatch)
    job["spec"]["phase"] = "confirmation"
    with pytest.raises(ValueError, match="historical development"):
        final.verify_recurrence(path, report, run, job, final.Verification())
    run["measurements"].pop()
    with pytest.raises(ValueError, match="Missing recurrence"):
        final.verify_recurrence(path, report, run, job, final.Verification())


def test_missing_and_failed_artifacts_are_distinct_and_all_retained(tmp_path):
    verification = final.Verification()
    assert verification.attempt("weights", "a", tmp_path / "absent", lambda: True) is None
    existing = tmp_path / "exists"
    existing.write_text("bad")
    assert (
        verification.attempt("audit", "b", existing, lambda: final.require(False, "failed check"))
        is None
    )
    assert verification.missing[0]["run"] == "a"
    assert verification.errors[0]["run"] == "b"


@pytest.mark.parametrize("check_only", [False, True])
def test_incomplete_matrix_never_writes_completion_manifest(tmp_path, monkeypatch, check_only):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["finalize", *(["--check-only"] if check_only else [])])
    result = {
        "state": "incomplete",
        "expected": {},
        "verified": {},
        "missing": [{"run": "pending"}],
        "errors": [],
    }
    monkeypatch.setattr(final, "inspect_completion", lambda: (result, {}))
    with pytest.raises(SystemExit) as error:
        final.main()
    assert error.value.code == 1
    assert not list(tmp_path.rglob("completion-manifest.json"))
    assert not list(tmp_path.rglob("*-endpoint-audit.json"))


def test_costs_sum_parallel_timers_but_use_outer_wall_span():
    rows = []
    for start, end in (("00:00:00", "00:00:05"), ("00:00:01", "00:00:06")):
        rows.append(
            {
                **dict.fromkeys(final.TOTAL_FIELDS, 10),
                "started_utc": f"2026-09-30T{start}+00:00",
                "finished_utc": f"2026-09-30T{end}+00:00",
            }
        )
    totals = final.total_cost(rows)
    assert totals["training_seconds"] == 20
    assert totals["run_utc_span_seconds"] == 6
    assert totals["run_count"] == 2


@pytest.mark.parametrize("check_only", [False, True])
def test_ready_check_only_does_not_write_and_final_write_preserves_development_audit(
    tmp_path, monkeypatch, check_only
):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["finalize", *(["--check-only"] if check_only else [])])
    development = tmp_path / final.ARTIFACT / "development-endpoint-audit.json"
    final.post.write(development, {"original": "must remain unchanged"})
    result = {
        "state": "ready",
        "expected": {"training_runs": 111},
        "verified": {"training_runs": 111},
        "missing": [],
        "errors": [],
        "verification_files": {},
    }
    aggregates = {name: {"passed": True} for name in final.CONFIGS}
    monkeypatch.setattr(final, "inspect_completion", lambda: (result, aggregates))
    final.main()
    assert final.read(development) == {"original": "must remain unchanged"}
    manifest = tmp_path / final.ARTIFACT / "completion-manifest.json"
    if check_only:
        assert not manifest.exists()
        assert not (manifest.parent / "confirmation-endpoint-audit.json").exists()
    else:
        saved = final.read(manifest)
        assert saved["state"] == "complete"
        for phase in ("sensitivity", "confirmation"):
            path = manifest.parent / f"{phase}-endpoint-audit.json"
            assert saved["verification_files"][str(path)] == final.post.digest(path)
