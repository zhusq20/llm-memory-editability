"""Authorized architecture matrix and isolated-stage execution contracts."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from test_ttcd_architecture_eval import make_reader

from llm_memory_editability import ttcd_architecture as architecture

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-architecture-comparison-v1"
PANEL_SHA = "ade07fa92e528f1b19bbc10965d984f747ed7f7daf66f5eafa254971ac7d08f6"


def config(name=""):
    suffix = f"-{name}" if name else ""
    return json.loads((ROOT / f"configs/{BATCH}{suffix}.json").read_text())


@pytest.fixture
def foundation(monkeypatch):
    # A normal `python scripts/...` invocation already has this sibling path.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    path = ROOT / "scripts/run_ttcd_architecture_foundation.py"
    spec = importlib.util.spec_from_file_location("architecture_foundation_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorized_matrix_and_budget_reuse_discoloop_without_restarting_it():
    matrix = config()
    assert matrix["architectures"] == ["transformer", "loop", "discoloop"]
    assert matrix["new_training_architectures"] == ["transformer", "loop"]
    assert matrix["new_scientific_updates"] == 640000
    assert matrix["per_new_architecture"] == {
        "foundation": 300000,
        "common": 10000,
        "stream": 10000,
    }
    assert matrix["new_scientific_updates"] == 2 * sum(matrix["per_new_architecture"].values())
    assert matrix["reuse"]["discoloop"]["training_restarted"] is False
    assert matrix["reuse"]["discoloop"]["arm"] == "stream"
    assert matrix["formal_panel_sha256"] == PANEL_SHA
    assert matrix["runtime"]["docker_context"] == "lm-memory"
    assert matrix["runtime"]["gpus"] == {"transformer": 8, "loop": 9}
    assert not set(matrix["runtime"]["cpu_sets"]["transformer"]).intersection(
        matrix["runtime"]["cpu_sets"]["loop"]
    )


@pytest.mark.parametrize("name,layers,repeats", [("transformer", 8, 1), ("loop", 4, 2)])
def test_same_native_recipe_only_changes_independent_depth_and_repeats(
    foundation, name, layers, repeats
):
    native = config(name + "-foundation")
    foundation.validate(native)
    assert native["spec"]["unique_layers"] == layers
    assert native["spec"]["repeats"] == repeats
    assert native["spec"]["initialization"] == 791101
    assert native["spec"]["stream_seed"] == 792101
    assert native["spec"]["steps"] == 300000
    assert native["dropout_seed"] == 20261012
    assert set(native["arms"]) == {name}
    assert native["arms"][name]["discoloop"] is False
    assert native["arms"][name]["first_round_kv"] is False
    assert native["evaluation_nodes"][-1] == 300000
    adaptation = config(name)
    assert set(adaptation["models"]) == {"stream"}
    assert adaptation["common"]["steps"] == adaptation["training"]["steps"] == 10000
    assert adaptation["training"]["correction_batch_size"] == 128
    assert adaptation["training"]["native_batch_size"] == 512
    assert adaptation["training"]["micro_facts"] == 32
    assert adaptation["continual"]["train_length"] == 8
    assert adaptation["continual"]["eval_length"] == adaptation["continual"]["rotations"] == 32
    assert f"/{name}/foundation/{name}/" in adaptation["foundation"]["checkpoint"]
    assert f"/{name}/stream/" in adaptation["models"]["stream"]["checkpoint"]


@pytest.mark.parametrize(
    "change",
    ["new_discoloop", "steps", "duplicate_initialization", "repeat_depth", "KV", "dropout"],
)
def test_foundation_rejects_unapproved_recipe_or_matrix_changes(foundation, change):
    native = config("transformer-foundation")
    if change == "new_discoloop":
        native["architecture_name"] = "discoloop"
    elif change == "steps":
        native["spec"]["steps"] += 1
    elif change == "duplicate_initialization":
        native["initial_model_sha256"] = "ba0af284"  # Cannot claim Loop initialization as D8.
    elif change == "repeat_depth":
        native["spec"].update(unique_layers=4, repeats=2)
    elif change == "KV":
        native["arms"]["transformer"]["first_round_kv"] = True
    else:
        native["dropout_seed"] += 1
    with pytest.raises(ValueError):
        foundation.validate(native)


def native_receipt_material(foundation, tmp_path):
    native = config("loop-foundation")
    science = tmp_path / "loop"
    science.mkdir()
    checkpoint = science / "checkpoint-0300000.pt"
    checkpoint.write_bytes(b"engineering receipt fixture; not a model checkpoint")
    complete = {
        "state": "complete",
        "step": 300000,
        "engineering_only": False,
        "final_checkpoint_sha256": foundation.sha256(checkpoint),
        "initial_model_sha256": native["initial_model_sha256"],
        "model_sha256": "engineering-model-digest",
        "parameters": native["parameters"],
        "endpoint": {
            "metrics": {
                name: {"accuracy": 0.0}
                for name in (
                    "atomic_id",
                    "atomic_ood",
                    "train_ii",
                    "test_ii",
                    "test_io",
                    "test_oi",
                    "test_oo",
                )
            }
        },
    }
    (science / "complete.json").write_text(json.dumps(complete))
    (science / "audit.json").write_text(json.dumps({"passed": True, "step": 300000}))
    return native, science, complete


def test_native_identity_is_numerical_and_never_filters_low_capability(foundation, tmp_path):
    native, science, _ = native_receipt_material(foundation, tmp_path)
    receipt = foundation.numerical_receipt(native, tmp_path, science)
    assert receipt["passed"]
    assert receipt["capability_thresholds"] is None
    assert receipt["gate_semantics"] == "identity_and_independent_numerical_replay_only"
    assert all(value["accuracy"] == 0.0 for value in receipt["native_metrics"].values())


@pytest.mark.parametrize("change", ["checkpoint", "failed_audit", "step", "initial", "smoke"])
def test_foundation_receipt_requires_the_original_numerical_endpoint(foundation, tmp_path, change):
    native, science, complete = native_receipt_material(foundation, tmp_path)
    if change == "checkpoint":
        (science / "checkpoint-0300000.pt").write_bytes(b"changed engineering bytes")
    elif change == "failed_audit":
        (science / "audit.json").write_text(json.dumps({"passed": False, "step": 300000}))
    else:
        if change == "step":
            complete["step"] -= 1
        elif change == "initial":
            complete["initial_model_sha256"] = "changed-initialization"
        else:
            complete["engineering_only"] = True
        (science / "complete.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError):
        foundation.numerical_receipt(native, tmp_path, science)


def test_capacity_accounting_keeps_double_fast_capacity_visible():
    ordinary = architecture.architecture_accounting(make_reader("d8", dtype=torch.float32))
    loop = architecture.architecture_accounting(make_reader("loop4x2", dtype=torch.float32))
    assert ordinary["physical_layers"] == ordinary["fast_matrices"] == 8
    assert loop["physical_layers"] == loop["fast_matrices"] == 4
    assert ordinary["effective_block_executions"] == loop["effective_block_executions"] == 8
    assert ordinary["fast_bytes"] == 2 * loop["fast_bytes"]
    assert ordinary["fast_elements"] == 2 * loop["fast_elements"]
    assert ordinary["base_parameters"] > loop["base_parameters"]
    assert ordinary["equal_compute_or_capacity_claim"] is False
    assert loop["equal_compute_or_capacity_claim"] is False
