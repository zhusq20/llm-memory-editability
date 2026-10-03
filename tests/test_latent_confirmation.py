"""New-world matrix and data isolation contracts for the support/recurrence comparison."""

import importlib.util
from pathlib import Path

import numpy as np

from llm_memory_editability.latent_scaling import build_world, run_name
from llm_memory_editability.storage_composition import audit_world


def runner():
    path = Path(__file__).parents[1] / "scripts/run_latent_confirmation.py"
    spec = importlib.util.spec_from_file_location("latent_confirmation_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_paired_new_world_matrix_and_locked_budget():
    specs = runner().specifications()
    assert len(specs) == len({run_name(s) for s in specs}) == 48
    assert {s["world"] for s in specs} == {740101, 740102, 740103}
    assert not {s["world"] for s in specs} & {730001, 730011}
    for world in {s["world"] for s in specs}:
        for initialization in {s["initialization"] for s in specs}:
            paired = [
                s for s in specs if s["world"] == world and s["initialization"] == initialization
            ]
            assert len(paired) == 8
            assert len({s["stream_seed"] for s in paired}) == 1
            assert {(s["composition_count"], s["layers"], s["repeats"]) for s in paired} == {
                (count, layers, repeat)
                for count in (256, "all")
                for layers, repeat in ((1, 1), (1, 2), (1, 3), (2, 1))
            }
    for spec in specs:
        assert spec["nodes"][0] == 0
        assert spec["steps"] == spec["nodes"][-1] == 128000
        assert set(spec["repeat_nodes"]) <= set(spec["checkpoint_nodes"]) <= set(spec["nodes"])
        assert spec["test_repeats"] == ([1, 2, 3, 4] if spec["layers"] == 1 else [1])


def test_support_changes_nested_composition_only_on_every_new_world():
    specs = runner().specifications()
    hashes = []
    for seed in {s["world"] for s in specs}:
        spec = next(s for s in specs if s["world"] == seed)
        sparse, dense = build_world(spec), build_world(dict(spec, composition_count="all"))
        audit_world(sparse)
        audit_world(dense)
        np.testing.assert_array_equal(sparse["train_composite"], dense["train_composite"][:256])
        for key in sparse:
            if key != "train_composite":
                np.testing.assert_array_equal(sparse[key], dense[key])
        assert len(sparse["common_atomic"]) + len(sparse["anchor_atomic"]) == 1024
        hashes.append(sparse["common_atomic"].tobytes())
    assert len(set(hashes)) == 3
