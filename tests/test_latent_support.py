"""Scientific contracts for graph treatment, fixed facts, and exact role exposure."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.grok_depth import EpochStream
from llm_memory_editability.latent_scaling import build_world as scaling_world
from llm_memory_editability.latent_support import (
    build_world,
    construct,
    data_digest,
    exposure_signature,
    model_digest,
    run_name,
    support_components,
    validate_nodes,
)


def runner():
    path = Path(__file__).parents[1] / "scripts/run_latent_support.py"
    spec = importlib.util.spec_from_file_location("latent_support_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_twelve_paired_runs_preserve_historical_budget():
    specs = runner().specifications()
    assert len(specs) == len({run_name(s) for s in specs}) == 12
    assert {s["world"] for s in specs} == {740101, 740102, 740103}
    assert {s["initialization"] for s in specs} == {741101, 741102}
    for world in {s["world"] for s in specs}:
        for initialization in {s["initialization"] for s in specs}:
            pair = [
                s for s in specs if (s["world"], s["initialization"]) == (world, initialization)
            ]
            assert {s["support"] for s in pair} == {"connected", "split"}
            assert len({s["stream_seed"] for s in pair}) == 1
    for spec in specs:
        validate_nodes(spec)
        assert (spec["layers"], spec["repeats"], spec["width"]) == (1, 2, 128)
        assert spec["steps"] == spec["nodes"][-1] == 128000
        assert len(spec["nodes"]) == 12
        assert spec["repeat_nodes"] == [32000, 128000]
        assert spec["test_repeats"] == [1, 2, 3, 4]


def test_support_changes_only_legal_combinations_with_matched_roles_and_marginals():
    specs = runner().specifications()
    for world in {s["world"] for s in specs}:
        pair = [s for s in specs if s["world"] == world and s["initialization"] == 741101]
        connected, split = (build_world(s) for s in pair)
        assert exposure_signature(connected) == exposure_signature(split)
        assert support_components(connected)["components"] < support_components(split)["components"]
        assert support_components(connected)["isolated_roles"] == 0
        assert support_components(split)["isolated_roles"] == 0
        for key in connected:
            if key != "train_composite":
                np.testing.assert_array_equal(connected[key], split[key])
        historical = scaling_world(dict(pair[0], composition_count="all"))
        for key in connected:
            if key != "train_composite":
                np.testing.assert_array_equal(connected[key], historical[key])
        np.testing.assert_array_equal(
            connected["train_composite"],
            historical["available_composite"][pair[0]["composition_indices"]],
        )
        assert len(set(map(tuple, connected["train_composite"]))) == 512


def test_sampler_matches_all_measurement_exposures_and_initial_weights():
    specs = runner().specifications()
    for world in {s["world"] for s in specs}:
        pair = [s for s in specs if s["world"] == world and s["initialization"] == 741101]
        worlds = [build_world(s) for s in pair]
        streams = [EpochStream(512, s["stream_seed"] + 1) for s in pair]
        counts = [np.zeros(512, dtype=np.int64) for _ in pair]
        previous = 0
        for node in pair[0]["nodes"][1:]:
            # Sampling rows is identical; semantic roles match at every full epoch.
            drawn = [stream.take((node - previous) * 64) for stream in streams]
            np.testing.assert_array_equal(*drawn)
            for tally, indices in zip(counts, drawn, strict=True):
                tally += np.bincount(indices, minlength=512)
                assert np.all(tally == node * 64 // 512)
            assert exposure_signature(worlds[0], counts[0]) == exposure_signature(
                worlds[1], counts[1]
            )
            previous = node
    first_pair = specs[:2]
    assert model_digest(construct(first_pair[0], "cpu")) == model_digest(
        construct(first_pair[1], "cpu")
    )


@pytest.mark.parametrize("corruption", ["duplicate", "out_of_range", "boolean", "wrong_count"])
def test_invalid_frozen_support_fails_without_reroll(corruption):
    spec = dict(runner().specifications()[0])
    indices = spec["composition_indices"].copy()
    if corruption == "duplicate":
        indices[-1] = indices[-2]
    elif corruption == "out_of_range":
        indices[-1] = 100000
    elif corruption == "boolean":
        indices[0] = True
    else:
        spec["composition_count"] = 256
    spec["composition_indices"] = indices
    with pytest.raises(ValueError):
        build_world(spec)


def test_frozen_data_digest_and_whole_epoch_contract_are_enforced():
    spec = runner().specifications()[0]
    frozen = dict(spec, frozen_data_sha256=data_digest(build_world(spec)))
    assert data_digest(build_world(frozen)) == frozen["frozen_data_sha256"]
    with pytest.raises(RuntimeError, match="frozen digest"):
        build_world(dict(frozen, frozen_data_sha256="changed"))
    with pytest.raises(ValueError, match="complete"):
        validate_nodes(dict(spec, nodes=[0, 9, 128000]))
    validate_nodes(runner().engineering_spec(frozen))
