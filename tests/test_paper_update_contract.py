"""Reuse fails closed when the scientific comparison or baseline evidence changes."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.paper_update_contract import (
    BASELINE_FILES,
    BATCH,
    CORE_SOURCE_FILES,
    PANELS,
    file_sha256,
    validate_config,
    verify_baseline,
    verify_input,
)

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reference(tmp_path):
    prior = json.loads((ROOT / "configs/dual-gate-multihop-v1.json").read_text())
    old = tmp_path / "old"
    current_source = tmp_path / "current"
    run = old / "runs/paper"
    run.mkdir(parents=True)
    source_hashes = {}
    for name in CORE_SOURCE_FILES:
        for root in (old / "source", current_source):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# unchanged source {name}\n")
        source_hashes[name] = file_sha256(current_source / name)
    prior_path = old / "source/configs/dual-gate-multihop-v1.json"
    write(prior_path, prior)
    config_sha = file_sha256(prior_path)
    write(
        old / "freeze.json",
        {
            "image": prior["runtime"]["image"],
            "files": {f"source/{name}": value for name, value in source_hashes.items()},
        },
    )
    config = copy.deepcopy(prior)
    config["batch"] = BATCH
    config["arms"] = {
        f"{mode}-update": {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "update_mode": mode,
            "initial_update": 0.9,
        }
        for mode in ("fixed", "token")
    }
    for name in ("checkpoint-0000000.pt", "checkpoint-0300000.pt"):
        (run / name).write_bytes(name.encode())
    metrics = {
        p: {"n": 1000 if p.startswith("train_") else 5000, "answer_accuracy": 0.25} for p in PANELS
    }
    endpoint = {"step": 300000, "metrics": metrics, "examples": 40000}
    identity = {
        "arm": "paper",
        "config_sha256": config_sha,
        "world_sha256": prior["input_origin"]["world_sha256"],
        "initial_model_sha256": "initial-model",
    }
    complete = {
        **identity,
        "state": "complete",
        "step": 300000,
        "budget": 300000,
        "engineering_only": False,
        "model_sha256": "terminal-model",
        "endpoint": endpoint,
        "final_checkpoint_sha256": file_sha256(run / "checkpoint-0300000.pt"),
    }
    write(run / "complete.json", complete)
    write(run / "run.json", {**identity, "panels_sha256": prior["input_origin"]["panels_sha256"]})
    audit = {
        "passed": True,
        "exposure_match": True,
        "step": 300000,
        "arm": "paper",
        "panels": {p: {"passed": True, "answer_and_eos_ids_exact": True} for p in PANELS},
    }
    write(run / "audit.json", audit)
    grid = {
        **identity,
        "panels_sha256": prior["input_origin"]["panels_sha256"],
        "model_sha256": "terminal-model",
        "step": 300000,
        "training_repeats": 4,
        "unique_layers": 1,
        "metrics": {str(r): metrics for r in range(1, 5)},
    }
    # The grid adds common-premise diagnostics across R; endpoint metrics stay intact.
    grid = copy.deepcopy(grid)
    grid["metrics"]["4"]["ood_3"]["common_premise_n"] = 4990
    write(run / "grid.json", grid)
    grid_audit = {
        "passed": True,
        "model_sha256": "terminal-model",
        "grid_sha256": file_sha256(run / "grid.json"),
        "findings": {f"r{r}/{p}": True for r in range(1, 5) for p in PANELS},
    }
    write(run / "grid-audit.json", grid_audit)
    write(
        run / "learning.json", [{**endpoint, "step": step} for step in config["evaluation_nodes"]]
    )
    np.savez(run / "exposure.npz", counts=np.ones(40000, dtype=np.int64))
    write(
        run / "canonical-result.json",
        {
            "complete.json": complete,
            "audit.json": audit,
            "grid.json": grid,
            "grid-audit.json": grid_audit,
        },
    )
    write(
        run / "cloud-audit.json",
        {
            "passed": True,
            "canonical_sha256": file_sha256(run / "canonical-result.json"),
        },
    )
    config["baseline"] = {
        "root": str(old),
        "run": str(run),
        "config": str(prior_path),
        "initial_model_sha256": "initial-model",
        "model_sha256": "terminal-model",
        "files": {n: file_sha256(run / n) for n in BASELINE_FILES},
        "source_files": source_hashes,
    }
    return config, current_source, run


def test_complete_paper_reference_is_verified_without_writes(reference):
    config, source, run = reference
    before = {p: file_sha256(p) for p in run.iterdir()}
    result = verify_baseline(config, source_root=source)
    assert result["passed"] and result["full_checkpoint_hashes"]
    assert result["new_baseline_updates"] == result["new_model_evaluations"] == 0
    assert result["baseline_run"] == str(run)
    assert result["exposure_examples"] == 40000
    assert before == {p: file_sha256(p) for p in run.iterdir()}


@pytest.mark.parametrize("change", ("extra_arm", "mode", "initial", "budget", "source_pin"))
def test_rejects_extra_scientific_variables(reference, change):
    config, _, _ = reference
    if change == "extra_arm":
        config["arms"]["paper"] = {}
    elif change == "mode":
        config["arms"]["fixed-update"]["update_mode"] = "full"
    elif change == "initial":
        config["arms"]["token-update"]["initial_update"] = 0.99
    elif change == "budget":
        config["training"]["steps"] = 20000
    else:
        config["baseline"]["source_files"].pop(CORE_SOURCE_FILES[0])
    with pytest.raises(ValueError):
        validate_config(config)


def test_changed_training_recipe_is_not_called_compatible(reference):
    config, source, _ = reference
    config["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="recipe mismatch: training"):
        verify_baseline(config, source_root=source)


def test_remote_batch_changes_only_both_initial_updates(reference):
    config, source, _ = reference
    config["batch"] = "paper-update-gate-multihop-u95-v1"
    for arm in config["arms"].values():
        arm["initial_update"] = 0.95
    assert verify_baseline(config, source_root=source)["passed"]
    config["arms"]["fixed-update"]["initial_update"] = 0.9
    with pytest.raises(ValueError, match="matched fixed/token"):
        validate_config(config)


def test_replacement_runtime_needs_explicit_amendment_and_keeps_original_identity(reference):
    config, source, _ = reference
    original_image = config["runtime"]["image"]
    config["baseline"]["source_image"] = original_image
    config["runtime"]["image"] = "sha256:replacement-runtime"
    with pytest.raises(ValueError, match="explicit amendment"):
        verify_baseline(config, source_root=source)
    config["runtime"]["amendment"] = {
        "reason": "User requested direct training without short training or runtime replay; "
        "major dependencies match but runtime equivalence has not been verified.",
        "original_image": original_image,
        "actual_image": config["runtime"]["image"],
        "requires_baseline_replay": False,
    }
    receipt = verify_baseline(config, source_root=source)
    assert receipt["passed"]
    assert receipt["requires_baseline_runtime_replay"] is False
    assert receipt["runtime_equivalence_verified"] is False
    config["baseline"]["source_image"] = config["runtime"]["image"]
    with pytest.raises(ValueError, match="original image differs"):
        verify_baseline(config, source_root=source)


def test_untouched_native_source_is_required(reference):
    config, source, _ = reference
    (source / CORE_SOURCE_FILES[0]).write_text("changed training")
    with pytest.raises(ValueError, match="Unchanged source differs"):
        verify_baseline(config, source_root=source)


def test_missing_or_modified_checkpoints_cannot_pass_full_verification(reference):
    config, source, run = reference
    (run / "checkpoint-0300000.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_baseline(config, source_root=source)
    receipt = verify_baseline(config, source_root=source, full_hashes=False)
    assert receipt["full_checkpoint_hashes"] is False
    assert not any(name.endswith(".pt") for name in receipt["checked_files"])
    (run / "checkpoint-0300000.pt").unlink()
    with pytest.raises(ValueError, match="Missing baseline artifact"):
        verify_baseline(config, source_root=source, full_hashes=False)


def test_all_inference_rounds_need_independent_audit(reference):
    config, source, run = reference
    audit = json.loads((run / "grid-audit.json").read_text())
    del audit["findings"]["r3/ood_3"]
    write(run / "grid-audit.json", audit)
    config["baseline"]["files"]["grid-audit.json"] = file_sha256(run / "grid-audit.json")
    with pytest.raises(ValueError, match="R1-4 grid reload"):
        verify_baseline(config, source_root=source)


def test_exposure_must_match_endpoint_accounting(reference):
    config, source, run = reference
    np.savez(run / "exposure.npz", counts=np.zeros(40000, dtype=np.int64))
    config["baseline"]["files"]["exposure.npz"] = file_sha256(run / "exposure.npz")
    with pytest.raises(ValueError, match="exposure identity"):
        verify_baseline(config, source_root=source)


def test_input_checks_existing_bytes_and_panel_coverage(reference, tmp_path):
    config, _, _ = reference
    root = tmp_path / "input"
    root.mkdir()
    config["input_root"] = str(root)
    for name, key in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        (root / name).write_bytes(name.encode())
        config["input_origin"][key] = file_sha256(root / name)
    metadata = {
        **config["data"],
        "training_examples": 40000,
        "entity_offset": 3,
        "relation_offset": 1003,
        "pad_token": 0,
        "end_marker": 2,
        "vocab_size": 1013,
        "sequence_length": 6,
        "atomic_examples": 10000,
        "world_sha256": config["input_origin"]["world_sha256"],
        "panels_sha256": config["input_origin"]["panels_sha256"],
        "panels": {p: 1000 if p.startswith("train_") else 5000 for p in PANELS},
    }
    write(root / "complete.json", metadata)
    write(root / "audit.json", {"passed": True})
    assert verify_input(config)["evaluation_queries"] == 43000
    metadata["pad_token"] = 1
    write(root / "complete.json", metadata)
    with pytest.raises(ValueError, match="token/shape contract changed: pad_token"):
        verify_input(config)
    metadata["pad_token"] = 0
    write(root / "complete.json", metadata)
    (root / "panels.npz").write_bytes(b"another split")
    with pytest.raises(ValueError, match="Input changed: panels"):
        verify_input(config)
