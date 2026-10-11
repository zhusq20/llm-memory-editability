"""Audit relaxation affects only finite cosine floats and volatile runtime config."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "cosine_audit_recovery",
    Path(__file__).resolve().parents[1] / "scripts/audit_path_overlap_tolerance.py",
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def ported():
    return {
        "layerwise_cosine": {"id": [[1.0, 0.25], [0.25, 1.0]]},
        "accuracy": 0.75,
        "prediction": [1, 2],
        "correct": True,
    }


def test_small_cosine_difference_is_recorded_as_tolerance_not_byte_exact():
    original, replay = ported(), ported()
    replay["layerwise_cosine"]["id"][0][1] += 1e-8
    stats = recovery.compare_ported(original, replay)
    assert stats["cosine_floats"] == 4
    assert stats["different_cosine_floats"] == 1
    assert stats["max_cosine_absolute_error"] == pytest.approx(1e-8)
    assert recovery.comparison_contract()["z1"] == "byte_exact"
    assert recovery.comparison_contract()["source_data_checkpoint_hashes"] == "exact"


@pytest.mark.parametrize("change", ["accuracy", "prediction", "type", "shape", "nan", "large"])
def test_noncosine_drift_and_invalid_or_large_cosine_drift_are_rejected(change):
    original, replay = ported(), ported()
    if change == "accuracy":
        replay["accuracy"] += 1e-8
    elif change == "prediction":
        replay["prediction"][0] = 3
    elif change == "type":
        replay["layerwise_cosine"]["id"][0][0] = 1
    elif change == "shape":
        replay["layerwise_cosine"]["id"][0].append(0.0)
    elif change == "nan":
        replay["layerwise_cosine"]["id"][0][0] = float("nan")
    else:
        replay["layerwise_cosine"]["id"][0][0] += 1e-3
    with pytest.raises(ValueError):
        recovery.compare_ported(original, replay)


@pytest.fixture
def audit_files(tmp_path):
    origin, attempt = tmp_path / "origin", tmp_path / "attempt"
    source_files = {}
    for name in (
        "src/llm_memory_editability/path_overlap.py",
        "scripts/run_zero_training_path_overlap.py",
        "scripts/read_path_hop_secondary.py",
    ):
        path = origin / "frozen/source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen scientific code\n")
        source_files[name] = recovery.sha(path)
    inputs = origin / "inputs"
    input_files = {}
    for name in ("data/world.npz", "data/panels.npz", "development/parent/checkpoint-0000000.pt"):
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen inputs")
        input_files[name] = {"sha256": recovery.sha(path)}
    recovery.write(
        origin / "frozen/freeze.json",
        {
            "contract": {"sha256": "original contract"},
            "source_files": source_files,
            "inputs": {"root": str(inputs), "files": input_files},
        },
    )
    metadata = {
        "complete": True,
        "run": "parent",
        "step": 0,
        "replication": {"failures": []},
        "contract_sha256": "original contract",
        "code_sha256": source_files,
        "world_sha256": input_files["data/world.npz"]["sha256"],
        "panels_sha256": input_files["data/panels.npz"]["sha256"],
        "checkpoint_sha256": input_files["development/parent/checkpoint-0000000.pt"]["sha256"],
        "tf32": False,
    }
    for root in (attempt, attempt / "audit-replay"):
        folder = root / "parent/0000000"
        recovery.write(folder / "z1.json", {"accuracy": 0.5, "cases": [1, 0]})
        recovery.write(folder / "ported.json", ported())
        block = deepcopy(metadata)
        block["outputs_sha256"] = {
            name: recovery.sha(folder / name) for name in ("z1.json", "ported.json")
        }
        recovery.write(folder / "hashes.json", block)
    legacy = SimpleNamespace(
        verify_checkpoint=lambda *args: recovery.sha(
            inputs / "development/parent/checkpoint-0000000.pt"
        )
    )
    return origin, attempt, legacy


def update_replay(attempt, name, value):
    folder = attempt / "audit-replay/parent/0000000"
    recovery.write(folder / name, value)
    block = recovery.read(folder / "hashes.json")
    block["outputs_sha256"][name] = recovery.sha(folder / name)
    recovery.write(folder / "hashes.json", block)


def test_real_node_audit_accepts_only_cosine_roundoff(audit_files):
    origin, attempt, legacy = audit_files
    value = ported()
    value["layerwise_cosine"]["id"][0][1] += 1e-8
    update_replay(attempt, "ported.json", value)
    result = recovery.audit_node(attempt, "parent", 0, origin, legacy)
    assert result["passed"] is True
    assert result["ported_byte_exact"] is False
    assert result["outputs_exact"] == ["z1.json"]
    assert result["different_cosine_floats"] == 1


@pytest.mark.parametrize("change", ["z1", "accuracy", "source", "checkpoint", "data", "metadata"])
def test_real_audit_rejects_scientific_and_provenance_drift(audit_files, change):
    origin, attempt, legacy = audit_files
    if change == "z1":
        update_replay(attempt, "z1.json", {"accuracy": 0.50000001, "cases": [1, 0]})
    elif change == "accuracy":
        value = ported()
        value["accuracy"] += 1e-8
        update_replay(attempt, "ported.json", value)
    elif change == "source":
        (origin / "frozen/source/scripts/run_zero_training_path_overlap.py").write_text("changed")
    elif change == "checkpoint":
        (origin / "inputs/development/parent/checkpoint-0000000.pt").write_bytes(b"changed")
    elif change == "data":
        (origin / "inputs/data/world.npz").write_bytes(b"changed")
    else:
        path = attempt / "audit-replay/parent/0000000/hashes.json"
        block = recovery.read(path)
        block["code_sha256"]["scripts/run_zero_training_path_overlap.py"] = "changed"
        recovery.write(path, block)
    with pytest.raises(ValueError):
        recovery.audit_node(attempt, "parent", 0, origin, legacy)


def test_runtime_config_update_does_not_authorize_scientific_mutation():
    calls = []
    proxy = recovery.RuntimeConfigProxy(
        SimpleNamespace(update=lambda values, **kw: calls.append(kw))
    )
    proxy.update({"container_runtime": {"host_pid": 101}})
    proxy.update({"parent_spec": {"same": True}})
    assert calls == [{"allow_val_change": True}, {}]
    with pytest.raises(ValueError, match="Scientific config"):
        proxy.update({"container_runtime": {}, "parent_spec": {}}, allow_val_change=True)
    with pytest.raises(ValueError, match="Scientific config"):
        proxy.update({"parent_spec": {}}, allow_val_change=True)


def test_prepare_inherits_all_nodes_and_keeps_old_failure_without_running_science(
    tmp_path, monkeypatch
):
    origin, science = tmp_path / "origin", tmp_path / "source-attempt"
    attempt = tmp_path / "audit-attempt"
    runs, nodes = [f"parent-{index}" for index in range(8)], list(range(14))
    recovery.write(origin / "frozen/freeze.json", {"immutable": True})
    (origin / "frozen/source").mkdir()
    (origin / "inputs/grokking-reproduction-v1").mkdir(parents=True)
    recovery.write(origin / "input-receipt.json", {"immutable": True})
    (origin / "validation").mkdir()
    recovery.write(
        science / "controller-state.json",
        {
            "state": "finished_with_failures",
            "active": [],
            "failed": runs,
        },
    )
    recovery.write(science / "recovery-receipt.json", {"mount_amendment": True})
    recovery.write(science / "selection-adoption.json", {"source_base": "old-v2"})
    recovery.write(science / "selection-stage/selection-summary.json", {"complete": True})
    for run in runs:
        for node in nodes:
            for name in ("z1.json", "ported.json", "hashes.json"):
                recovery.write(science / run / f"{node:07d}" / name, {"source_node": node})
            recovery.write(
                science / "selection-stage" / run / f"{node:07d}/selection.json",
                {
                    "complete": True,
                },
            )
        recovery.write(
            science / "runs" / run / "learning.json",
            [
                {
                    "step": step,
                    "optimizer_updates": 0,
                    "metrics": {"node": {"seconds": 0.01}},
                }
                for step in range(1, 29)
            ],
        )
        recovery.write(science / "runs" / run / "science-process.json", {"host_pid": 100})
        recovery.write(science / "runs" / run / "failure.json", {"preserved": True})
    checks = []
    legacy = SimpleNamespace(
        verify_freeze=lambda base, **kw: checks.append((base, kw)),
        contract=lambda base: {"batch": "unchanged-v3", "source": {"runs": runs, "nodes": nodes}},
        utc=lambda: "UTC",
    )
    monkeypatch.setattr(recovery, "build_legacy", lambda args: (legacy, None))
    monkeypatch.setattr(recovery, "verify_node", lambda *args: None)
    args = SimpleNamespace(origin_base=origin, science_base=science, attempt_base=attempt)
    recovery.prepare(args)
    assert checks == [(origin, {"all_inputs": True})]
    receipt = recovery.read(attempt / "audit-recovery-receipt.json")
    assert len(receipt["inherited_files"]) == 448
    assert receipt["new_optimizer_updates"] == receipt["new_scientific_interventions"] == 0
    assert recovery.sha(attempt / "recovery-receipt.json") == recovery.sha(
        attempt / "audit-recovery-receipt.json"
    )
    assert recovery.sha(attempt / "frozen/freeze.json") == recovery.sha(
        origin / "frozen/freeze.json"
    )
    assert not (attempt / "audit-replay").exists()
    assert not (attempt / "runs" / runs[0] / "failure.json").exists()
    snapshot = science / "scheduling-intervention/audit-tolerance-v1/original-attempt"
    assert recovery.read(snapshot / "runs" / runs[0] / "failure.json")["preserved"] is True
    history = recovery.read(attempt / "runs" / runs[0] / "learning.json")
    assert len(history) == 28
    assert history[-1]["condition"] == "science_inherited_from_science_attempt"
    assert history[-1]["metrics"]["node"]["seconds"] == 0
    assert history[-1]["metrics"]["node"]["inherited_source_execution_seconds"] == 0.01
    recovery.verify(args)
    (attempt / runs[0] / "0000000/z1.json").write_text("corrupted")
    with pytest.raises(ValueError, match="evidence changed"):
        recovery.verify(args)
