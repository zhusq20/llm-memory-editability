"""Check the scientific comparisons in the four-block extension."""

import importlib.util
import json
from pathlib import Path

import torch

from llm_memory_editability.latent_scaling import build_world, construct, data_digest, model_digest


def specs():
    root = Path(__file__).resolve().parents[1]
    module = importlib.util.spec_from_file_location(
        "loop_extension", root / "scripts/run_loop_block_depth.py"
    )
    runner = importlib.util.module_from_spec(module)
    module.loader.exec_module(runner)
    parent = json.loads((root / "configs/latent-scaling-v1.json").read_text())
    return runner.specifications(parent), parent


def test_four_block_training_preserves_initial_state_data_and_native_R():
    cases, parent = specs()
    reference = next(
        s
        for s in parent["specs"]
        if s["width"] == 128
        and s["layers"] == 1
        and s["repeats"] == 2
        and s["composition_count"] == "all"
    )
    expected_data = data_digest(build_world(reference))
    hashes = set()
    for case in cases:
        assert data_digest(build_world(case)) == expected_data
        assert case["steps"] == 128000
        assert set(case["repeat_nodes"]) <= set(case["checkpoint_nodes"]) <= set(case["nodes"])
        if case["layers"] == 4:
            model = construct(case, "cpu")
            hashes.add(model_digest(model))
            assert len(list(model.iter_blocks())) == 4 * case["repeats"]
            assert len({id(b) for b in model.iter_blocks()}) == 4
            assert case["repeats"] in case["test_repeats"]
    assert len(hashes) == 1


def test_compute_control_matches_budget_and_contains_same_depth_comparators():
    cases, _ = specs()
    assert len(cases) == 9
    assert {(c["layers"], c["repeats"]) for c in cases} >= {(4, 1), (2, 2), (4, 2), (2, 4), (8, 1)}
    for case in cases:
        assert (
            0
            <= case["reference_compute_budget"]
            - case["matched_compute_step"] * case["per_step_flops"]
            < case["per_step_flops"]
        )


def test_four_block_native_forward_equals_explicit_override_without_parameter_mutation():
    cases, _ = specs()
    case = dict(cases[0], width=16, heads=2)
    model = construct(case, "cpu").eval()
    tokens = torch.tensor([[2, 25, 3, 13, 4]])
    original = model_digest(model)
    with torch.no_grad():
        torch.testing.assert_close(
            model(tokens), model(tokens, repeats=case["repeats"]), rtol=0, atol=0
        )
        model(tokens, repeats=1)
    assert model_digest(model) == original
