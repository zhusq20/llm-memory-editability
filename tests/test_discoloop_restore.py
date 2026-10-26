"""A capable, replayed foundation is required before adding a writable state."""

import copy
import json
from pathlib import Path

import pytest

from llm_memory_editability.discoloop_restore import (
    INITIAL_MODEL_SHA256,
    PANELS,
    PARAMETERS,
    foundation_gate,
    validate_recipe,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/discoloop-success-restore-v1.json").read_text())


def endpoint():
    metrics = {name: {"accuracy": 0.999} for name in PANELS}
    complete = {
        "state": "complete",
        "step": 300000,
        "budget": 300000,
        "engineering_only": False,
        "parameters": PARAMETERS,
        "initial_model_sha256": INITIAL_MODEL_SHA256,
        "model_sha256": "model",
        "final_checkpoint_sha256": "file",
        "endpoint": {"metrics": metrics},
    }
    audit = {
        "passed": True,
        "step": 300000,
        "findings": {
            "panels": {name: {"passed": True, "metrics": panel} for name, panel in metrics.items()}
        },
    }
    return complete, audit


def test_original_recipe_and_replayed_capable_endpoint_pass(config):
    validate_recipe(config)
    complete, audit = endpoint()
    assert foundation_gate(config, complete, audit, "file")["passed"]


@pytest.mark.parametrize(
    "key,value",
    [("unique_layers", 2), ("steps", 20000), ("batch_size", 128), ("phi", 2.0), ("lr", 1e-3)],
)
def test_shrunken_or_changed_recipe_rejected(config, key, value):
    config["spec"][key] = value
    with pytest.raises(ValueError, match="recipe differs"):
        validate_recipe(config)


@pytest.mark.parametrize("change", ["checkpoint", "initial", "audit", "step", "smoke", "tokens"])
def test_gate_requires_bound_checkpoint_and_independent_full_replay(config, change):
    complete, audit = endpoint()
    if change == "checkpoint":
        complete["final_checkpoint_sha256"] = "wrong"
    elif change == "initial":
        complete["initial_model_sha256"] = "wrong"
    elif change == "audit":
        audit["passed"] = False
    elif change == "step":
        complete["step"] = 299999
    elif change == "smoke":
        complete["engineering_only"] = True
    else:
        audit["findings"]["panels"]["test_oi"]["passed"] = False
    assert not foundation_gate(config, complete, audit, "file")["passed"]


@pytest.mark.parametrize("name,value", [("atomic_ood", 0.97), ("test_oi", 0.95), ("test_oo", 0.89)])
def test_gate_preserves_foundation_capability(config, name, value):
    complete, audit = endpoint()
    complete = copy.deepcopy(complete)
    complete["endpoint"]["metrics"][name]["accuracy"] = value
    audit["findings"]["panels"][name]["metrics"]["accuracy"] = value
    result = foundation_gate(config, complete, audit, "file")
    assert not result["passed"]
    assert not result["capability"][name]["passed"]


def test_thresholds_cannot_be_relaxed_after_observing_training(config):
    config["capability_gate"]["minimum_accuracy"]["test_oi"] = 0.5
    with pytest.raises(ValueError, match="thresholds"):
        validate_recipe(config)
