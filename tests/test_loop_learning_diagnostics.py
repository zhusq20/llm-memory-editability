"""Current-weight gradient diagnostics preserve the scientific training state."""

import copy
import json

import pytest
import torch

from llm_memory_editability.loop_learning import ARMS, construct
from llm_memory_editability.loop_learning_diagnostics import _gradient_geometry, diagnose
from llm_memory_editability.realworld_composition import example_losses, pack


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def spec(arm="shared"):
    return {
        "arm": arm,
        "initialization": 31,
        "base_unique_blocks": 2,
        "repeats": 3,
        "model": {
            "vocab_size": 31,
            "positions": 32,
            "hidden_size": 8,
            "attention_heads": 2,
            "dropout": 0.0,
        },
    }


def records():
    return [
        {"id": "train-a", "encoded": {"prefix": [1, 2], "target": [3, 30], "input": [1, 2, 3]}},
        {
            "id": "train-b",
            "encoded": {"prefix": [4, 5, 6], "target": [7, 30], "input": [4, 5, 6, 7]},
        },
    ]


@pytest.mark.parametrize("arm", ARMS)
def test_same_current_weight_reconstruction_and_actual_sharing_are_reported(arm):
    model = construct(spec(arm), "cpu").double()
    # Break initial occurrence equality where the architecture permits it.
    with torch.no_grad():
        model.transformer.h[-1].attn.c_proj.weight.add_(0.03)
        model.transformer.h[-1].mlp.c_proj.weight.mul_(0.9)
    result = diagnose(model, records(), spec(arm), "cpu", 30)
    json.dumps(result, allow_nan=False)
    assert result["forward_maximum_absolute_error"] == 0
    assert result["loss_absolute_error"] == 0
    assert result["gradient_reconstruction"]["maximum_absolute_error"] < 1e-12
    assert result["record_ids"] == ["train-a", "train-b"]
    assert len(result["groups"]) == 4
    for group in result["groups"]:
        tied = arm == "shared" or (arm == "mlp_shared" and group["component"] == "mlp")
        assert group["actually_shared_across_occurrences"] == tied
        error = group["shared_gradient_maximum_absolute_error"]
        assert (error is not None and error < 1e-12) if tied else error is None
        assert group["summed_gradient_squared_norm"] == pytest.approx(
            group["sum_individual_squared_norms"] + group["twice_pairwise_inner_product_sum"],
            rel=1e-11,
            abs=1e-12,
        )
        assert len(group["gradient_cosines"]) == (1 if arm == "shallow" else 3)
        if arm == "shallow":
            assert group["squared_norm_ratio"] == pytest.approx(1.0)
    copied = copy.deepcopy(model).eval()
    tokens, positions, labels = pack(records(), 30, "cpu")
    loss = example_losses(copied(tokens, positions), labels).mean()
    assert result["reference_loss"] == pytest.approx(float(loss.detach()), abs=1e-12)


@pytest.mark.parametrize("training", [True, False])
def test_diagnostics_leave_source_weights_gradients_modes_and_rng_unchanged(training):
    run_spec = spec("mlp_shared")
    run_spec["model"]["dropout"] = 0.1
    model = construct(run_spec, "cpu").train(training)
    # Preserve even mixed submodule modes and an existing partially populated gradient set.
    model.transformer.h[0].attn.eval()
    for index, parameter in enumerate(model.parameters()):
        if index % 2:
            parameter.grad = torch.full_like(parameter, 0.125)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = {
        name: (parameter.grad, None if parameter.grad is None else parameter.grad.clone())
        for name, parameter in model.named_parameters()
    }
    modes = {name: module.training for name, module in model.named_modules()}
    rng = torch.get_rng_state().clone()
    result = diagnose(model, records(), run_spec, "cpu", 30)
    assert not result["dropout_enabled"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert {name: module.training for name, module in model.named_modules()} == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        original, snapshot = gradients[name]
        assert parameter.grad is original
        if snapshot is not None:
            torch.testing.assert_close(parameter.grad, snapshot, rtol=0, atol=0)


def test_gradient_geometry_distinguishes_reinforcement_cancellation_and_zero():
    vector = torch.tensor([3.0, 4.0], dtype=torch.float64)
    reinforced = _gradient_geometry([vector, vector])
    canceled = _gradient_geometry([vector, -vector])
    zero = _gradient_geometry([torch.zeros_like(vector), vector])
    assert reinforced["occurrence_gradient_norms"] == [5.0, 5.0]
    assert reinforced["summed_gradient_norm"] == 10
    assert reinforced["squared_norm_ratio"] == 2
    assert canceled["summed_gradient_norm"] == 0
    assert canceled["squared_norm_ratio"] == 0
    assert canceled["gradient_cosines"][0][1] == -1
    assert zero["gradient_cosines"][0] == [None, None]
    assert zero["gradient_cosines"][1] == [None, 1.0]
    json.dumps(zero, allow_nan=False)


def test_empty_batch_and_spec_mismatch_are_rejected():
    model = construct(spec(), "cpu")
    with pytest.raises(ValueError, match="nonempty"):
        diagnose(model, [], spec(), "cpu", 30)
    with pytest.raises(ValueError, match="differs"):
        diagnose(model, records(), spec("untied"), "cpu", 30)
