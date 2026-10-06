"""Conclusion-changing contracts for the data-size and knowledge-load comparisons."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.composition_curves import (
    anchors_pass,
    budget_nodes,
    calibrated_choice,
    confirmation_specs,
    construct,
    load_specs,
    prepare_world,
)
from llm_memory_editability.grok_depth import EpochStream

ROOT = Path(__file__).resolve().parents[1]


def spec(phi=1, entities=16):
    return {
        "world_seed": 311,
        "entities": entities,
        "max_entities": 32,
        "relations": 8,
        "degree": 4,
        "phi": phi,
        "id_fraction": 0.75,
        "test_fraction": 0.1,
        "evaluation_size": 128,
        "batch_size": 16,
        "target_exposures": 17,
        "compute_steps": 64,
        "width": 16,
        "layers": 2,
        "heads": 2,
        "dropout": 0.1,
        "initialization": 981,
    }


def test_distinct_supports_are_nested_and_never_change_world_or_test_panels():
    previous = set()
    identity = None
    reference = None
    for phi in (0, 0.5, 1, 2):
        arrays, panels, meta = prepare_world(spec(phi))
        current = {tuple(row[:-1]) for row in arrays["train_composite"]}
        assert previous <= current
        assert len(current) == meta["composition_examples"]
        assert not current & {tuple(row[:-1]) for row in arrays["test_composite"]}
        if identity is None:
            identity, reference = meta["world_sha256"], panels
        assert meta["world_sha256"] == identity
        for key in ("atomic", "II", "OO"):
            np.testing.assert_array_equal(panels[key], reference[key])
        previous = current


def test_truth_and_ood_roles_survive_fixed_vocabulary_remapping():
    arrays, _, meta = prepare_world(spec(2))
    atomic = {tuple(row[:2]): int(row[2]) for row in arrays["atomic"]}
    ood = {tuple(row[:2]) for row in arrays["ood_atomic"]}
    for head, r1, r2, answer in arrays["train_composite"]:
        bridge = atomic[head, r1]
        assert atomic[bridge, r2] == answer
        assert (head, r1) not in ood and (bridge, r2) not in ood
    for head, r1, r2, answer in arrays["ood_composite"]:
        bridge = atomic[head, r1]
        assert atomic[bridge, r2] == answer
        assert (head, r1) in ood and (bridge, r2) in ood
    assert meta["role_coverage"]["ood_composite"]["both_facts_seen_in_required_roles"] in (0, None)
    assert arrays["atomic"][:, 1].min() >= spec()["max_entities"] + 2


def test_same_model_parameters_and_initial_weights_at_every_load():
    import torch

    small, cfg = construct(spec(1, 8), "cpu")
    large, large_cfg = construct(spec(1, 32), "cpu")
    assert cfg == large_cfg
    for key, value in small.state_dict().items():
        assert torch.equal(value, large.state_dict()[key])
    for n in (8, 16, 32):
        arrays, _, meta = prepare_world(spec(1, n))
        assert meta["knowledge_bits"] == len(arrays["atomic"]) * math.log2(n)
        assert meta["composition_additional_independent_bits"] == 0


def test_requests_never_cap_or_silently_repeat_distinct_compositions():
    with pytest.raises(ValueError, match="exceeds"):
        prepare_world(spec(100))
    with pytest.raises(ValueError, match="vocabulary"):
        prepare_world(spec(1, 33))


def test_world_creation_does_not_touch_global_numpy_rng():
    np.random.seed(775)
    expected = np.random.random(8)
    np.random.seed(775)
    prepare_world(spec())
    np.testing.assert_array_equal(np.random.random(8), expected)


def test_matched_exposure_budget_gives_every_record_the_declared_exposure():
    item = spec()
    total = 39
    nodes, budgets = budget_nodes(item, total)
    assert budgets["exposure"] == math.ceil(17 * total / 16)
    assert budgets["exposure"] in nodes and budgets["compute"] in nodes
    stream = EpochStream(total, 763)
    seen = np.bincount(stream.take(budgets["exposure"] * 16), minlength=total)
    assert seen.min() == 17 and seen.max() == 18
    resumed = EpochStream(total, 1)
    resumed.load_state_dict(copy.deepcopy(stream.state_dict()))
    np.testing.assert_array_equal(stream.take(120), resumed.take(120))


def test_fixed_development_budget_does_not_pretend_to_reach_exposure_endpoint():
    item = {**spec(), "fixed_steps": 64}
    nodes, budgets = budget_nodes(item, 1000)
    assert budgets["end"] == 64
    assert nodes[-1] == 64
    assert budgets["exposure"] not in nodes


def metrics(atomic=1, train=1, test=0.9):
    return {
        "atomic": {"accuracy": atomic},
        "train_composition": {"accuracy": train},
        "II": {"accuracy": test},
    }


def test_development_gate_does_not_reward_best_training_fit_without_generalization():
    thresholds = {"atomic": 0.95, "train_composition": 0.95, "II": 0.8}
    two, four = {"layers": 2}, {"layers": 4}
    assert calibrated_choice([(two, metrics(test=0.02)), (four, metrics())], thresholds) == four
    assert calibrated_choice([(four, metrics()), (two, metrics())], thresholds) == two
    assert calibrated_choice([(two, metrics(atomic=0.94))], thresholds) is None


def test_load_gate_requires_all_distinct_registered_worlds():
    thresholds = {"atomic": 0.95, "train_composition": 0.95, "II": 0.8}
    rows = [({"world_seed": n}, metrics()) for n in (1, 2, 3)]
    assert anchors_pass(rows, [1, 2, 3], thresholds)
    assert not anchors_pass(rows[:2], [1, 2, 3], thresholds)
    assert not anchors_pass([rows[0], rows[0], rows[2]], [1, 2, 3], thresholds)
    rows[1] = ({"world_seed": 2}, metrics(test=0.79))
    assert not anchors_pass(rows, [1, 2, 3], thresholds)


def test_complete_registered_matrix_reuses_low_load_without_duplicate_training():
    config = json.loads((ROOT / "configs/composition-data-curves-v1.json").read_text())
    selected = {**config["base_spec"], "layers": 2}
    support = confirmation_specs(config, selected)
    load = load_specs(config, selected)
    assert len(support) == 21 and len(load) == 15
    assert len({row["name"] for row in support + load}) == 36
    for world in config["confirmation_worlds"]:
        assert {row["phi"] for row in support if row["world_seed"] == world} == set(
            config["support_phi"]
        )
        anchors = [
            row
            for row in support
            if row["world_seed"] == world and row["phi"] == config["anchor_phi"]
        ]
        assert len(anchors) == 1 and anchors[0]["entities"] == 128
        assert all(row["entities"] != 128 for row in load)
    assert all(row["phase"] == "confirmation" for row in support + load)
    assert config["development_world"] not in config["confirmation_worlds"]


def load_executor():
    module_spec = importlib.util.spec_from_file_location(
        "curve_executor", ROOT / "scripts/execute_composition_curves.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_container_isolation_uses_real_repository_and_frozen_source(tmp_path):
    executor = load_executor()
    config = {
        "batch": "example",
        "repository": "/repo",
        "source_root": "/repo/frozen/source",
        "runtime": {"image": "sha256:fixed", "python": "/python", "cpus": 2, "memory": "12g"},
    }
    name, command = executor.container_command(
        config, Path("/repo/frozen/support-config.json"), {"name": "phi1"}, tmp_path, 1, 1
    )
    assert command[:4] == ["docker", "--context", "lm-memory", "run"]
    assert "type=bind,src=/repo,dst=/repo,readonly" in command
    assert "PYTHONPATH=/repo/frozen/source/src" in command
    assert "device=1" in command and "--read-only" in command and "--network=none" in command
    other, _ = executor.container_command(
        config, Path("/repo/frozen/support-config.json"), {"name": "phi1"}, tmp_path, 1, 2
    )
    assert name != other
