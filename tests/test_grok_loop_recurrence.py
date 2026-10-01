"""Evidence checks for recurrence scans, without changing inference arithmetic."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.grok_multihop_data import audit_world, build_world


@pytest.fixture
def scan():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_grok_loop_recurrence.py"
    spec = importlib.util.spec_from_file_location("recurrence_evidence_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_lock_requires_registered_config_and_unchanged_dependencies(scan, tmp_path):
    config = tmp_path / "config.json"
    source = tmp_path / "model.py"
    config.write_text("{}")
    source.write_text("original source")
    lock = {"files": {"config.json": scan.digest(config), "model.py": scan.digest(source)}}
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    cfg = {"source_lock": "lock.json"}
    _, provenance = scan.verify_stage_lock(cfg, config, root=tmp_path)
    assert provenance["verified"]
    source.write_text("changed source")
    with pytest.raises(ValueError, match="differs from training stage lock"):
        scan.verify_stage_lock(cfg, config, root=tmp_path)
    del lock["files"]["config.json"]
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="configuration is absent"):
        scan.verify_stage_lock(cfg, config, root=tmp_path)


def test_analysis_lock_covers_all_code_dependencies_without_self_reference(scan, tmp_path):
    files = {}
    for name in scan.DEPENDENCIES:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(name)
        files[name] = scan.digest(source)
    path = tmp_path / "analysis-lock.json"
    path.write_text(json.dumps({"files": files}))
    assert scan.verify_analysis_lock(path, root=tmp_path)["verified"]
    assert "analysis-lock.json" not in files
    del files[scan.DEPENDENCIES[-1]]
    path.write_text(json.dumps({"files": files}))
    with pytest.raises(ValueError, match="omits"):
        scan.verify_analysis_lock(path, root=tmp_path)


def test_native_prediction_check_catches_answer_eos_and_likelihood_drift(scan):
    world = {"atomic": np.array([[2, 6, 3], [3, 6, 2]])}
    pred = {
        "answer": np.array([3, 2]),
        "stop": np.array([1, 1]),
        "target": np.array([3, 2]),
        "nll": np.zeros((2, 2), dtype=np.float32),
    }
    native = {"atomic_" + key: value.copy() for key, value in pred.items()}
    assert scan.verify_native_predictions(pred, native, world, "atomic")["passed"]
    for key in ("answer", "stop", "target", "nll"):
        changed = {k: value.copy() for k, value in pred.items()}
        changed[key].flat[0] += 1
        with pytest.raises(ValueError, match="changes native"):
            scan.verify_native_predictions(changed, native, world, "atomic")
    assert (
        scan.verify_native_predictions({}, {}, {"atomic": world["atomic"][:0]}, "atomic")["n"] == 0
    )


def test_numerical_environment_mismatch_is_rejected(scan, tmp_path):
    saved = {
        "torch": "2.8.0",
        "numpy": "2.2.6",
        "cuda": "12.8",
        "gpu": "test GPU",
        "tf32": True,
        "precision": "FP32 parameters, forward and optimizer; TF32 matmuls",
    }
    runtime = {
        "torch": "2.8.0",
        "numpy": "2.2.6",
        "cuda": "12.8",
        "gpu": "test GPU",
        "python": "3.11.0",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
        "parameter_dtype": "torch.float32",
    }
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(saved))
    (tmp_path / "metadata.json").write_text(json.dumps({"python": "3.11.0"}))
    assert scan.verify_training_environment(path, runtime)["numerical_environment_matches_training"]
    for key, replacement in (
        ("torch", "new"),
        ("gpu", "other"),
        ("python", "3.12.0"),
        ("cuda_matmul_allow_tf32", False),
        ("parameter_dtype", "torch.bfloat16"),
    ):
        with pytest.raises(ValueError, match="differs from training"):
            scan.verify_training_environment(path, {**runtime, key: replacement})


def test_training_artifact_check_verifies_world_and_both_source_copies(scan, tmp_path):
    directory = tmp_path / "results/toy"
    directory.mkdir(parents=True)
    spec = {"steps": 10}
    files = {}
    for name in scan.DEPENDENCIES:
        if not name.startswith("src/"):
            continue
        current = tmp_path / name
        snapshot = directory / "source" / name
        for path in (current, snapshot):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("frozen dependency")
        files[str(current)] = scan.digest(current)
    (directory / "metadata.json").write_text(json.dumps({"spec": spec, "files": files}))
    (directory / "complete.json").write_text(json.dumps({"spec": spec, "endpoint": {"step": 10}}))
    world = build_world(7, hops=2, entities=4, relations=2, degree=2, evaluation_size=8)
    np.savez(directory / "world.npz", **{k: v for k, v in world.items() if k != "metadata"})
    np.savez(directory / "predictions-0000010.npz", placeholder=np.zeros(1))
    (directory / "world-metadata.json").write_text(json.dumps(world["metadata"]))
    (directory / "data-audit.json").write_text(json.dumps(audit_world(world)))
    loaded, provenance = scan.verify_training_artifacts(directory, spec, root=tmp_path)
    assert provenance["current_and_snapshot_sources_match_training"]
    np.testing.assert_array_equal(loaded["atomic"], world["atomic"])
    first_source = Path(next(iter(files)))
    first_source.write_text("changed current dependency")
    with pytest.raises(ValueError, match="Current or frozen source differs"):
        scan.verify_training_artifacts(directory, spec, root=tmp_path)
    first_source.write_text("frozen dependency")
    changed = dict(world["metadata"], dataset_sha256="incorrect")
    (directory / "world-metadata.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="metadata hash"):
        scan.verify_training_artifacts(directory, spec, root=tmp_path)
