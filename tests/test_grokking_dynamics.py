"""Registered contrasts, fixed LR prefixes, truth and paired initialization."""

import importlib.util
from pathlib import Path

import numpy as np

from llm_memory_editability.latent_scaling import build_world, construct, model_digest
from llm_memory_editability.storage_composition import audit_world, data_digest
from llm_memory_editability.storage_frontier import learning_rate


def runner():
    path = Path(__file__).parents[1] / "scripts/run_grokking_dynamics.py"
    spec = importlib.util.spec_from_file_location("grokking_dynamics_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_matrix_has_unique_names_and_budget_independent_lr():
    module = runner()
    specs = module.specifications()
    assert len(specs) == len({module.run_name(s) for s in specs}) == 12
    assert {s["weight_decay"] for s in specs} == {0, 0.01, 0.1}
    assert {s["layers"] * s["repeats"] for s in specs} == {2}
    for s in specs:
        assert s["nodes"] == sorted(set(s["nodes"]))
        assert s["nodes"][-1] == s["steps"] == 512000
        assert set(s["checkpoint_nodes"]) <= set(s["nodes"])
        for step in [1, 2000, 8000, 128000]:
            assert learning_rate(s, step) == learning_rate(dict(s, steps=256000), step)


def test_weight_decay_does_not_change_data_or_initial_parameters():
    module = runner()
    specs = module.specifications()
    reference = build_world(specs[0])
    audit_world(reference)
    hashes = {}
    for s in specs:
        world = build_world(s)
        assert data_digest(world) == data_digest(reference)
        for key in ["common_atomic", "anchor_atomic", "train_composite"]:
            np.testing.assert_array_equal(world[key], reference[key])
        key = s["initialization"], s["layers"]
        digest = model_digest(construct(s, "cpu"))
        assert hashes.setdefault(key, digest) == digest
    assert hashes[781101, 1] != hashes[781102, 1]


def test_calibration_has_separate_world_and_no_late_data_selection():
    module = runner()
    calibration = module.specifications("calibration")
    main = module.specifications()
    assert len(calibration) == 6
    assert {s["world"] for s in calibration}.isdisjoint({s["world"] for s in main})
    assert all(s["steps"] == 16000 for s in calibration)
