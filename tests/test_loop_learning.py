"""Scientific contracts for same-function, different-sharing Loop training."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability import sequential_transfer as st
from llm_memory_editability.loop_learning import (
    ARMS,
    architecture_manifest,
    construct,
    estimate_training_flops,
    execution_digest,
)
from llm_memory_editability.realworld_composition import model_digest, optimizer_for, update


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def spec(arm="shared", repeats=3, dropout=0.0):
    return {
        "arm": arm,
        "initialization": 37,
        "base_unique_blocks": 2,
        "repeats": repeats,
        "model": {
            "vocab_size": 31,
            "positions": 32,
            "hidden_size": 8,
            "attention_heads": 2,
            "dropout": dropout,
        },
        "microbatch_size": 2,
    }


def records():
    return [
        {"encoded": {"prefix": [1, 2], "target": [3, 30], "input": [1, 2, 3]}},
        {"encoded": {"prefix": [4, 5, 6], "target": [7, 30], "input": [4, 5, 6, 7]}},
    ]


@pytest.mark.parametrize("repeats", [2, 4])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_same_function_initialization_preserves_full_forward_and_random_stream(repeats, dropout):
    models = [construct(spec(arm, repeats, dropout), "cpu") for arm in ARMS[:3]]
    assert len({execution_digest(model) for model in models}) == 1
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    outputs, states = [], []
    for model in models:
        model.train()
        torch.manual_seed(73)
        outputs.append(model(tokens))
        states.append(torch.get_rng_state())
    for output, state in zip(outputs[1:], states[1:], strict=True):
        torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)
        assert torch.equal(state, states[0])


def test_selected_readout_and_causality_hold_in_every_arm():
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    positions = torch.tensor([[1, 2], [0, 2]])
    for arm in ARMS:
        model = construct(spec(arm), "cpu").eval()
        logits = model(tokens)
        selected = logits[torch.arange(2)[:, None], positions]
        torch.testing.assert_close(model(tokens, positions), selected)
        modified = tokens.clone()
        modified[:, -1] = 30
        torch.testing.assert_close(model(modified)[:, :-1], logits[:, :-1], rtol=0, atol=0)


def test_reference_initialization_does_not_depend_on_arm_or_repeat_depth():
    original = construct(spec(), "cpu")
    for arm in ARMS:
        varied = spec(arm, repeats=4)
        varied["base_unique_blocks"] = 4
        model = construct(varied, "cpu")
        for index in range(2):
            for name, tensor in original.transformer.h[index].state_dict().items():
                torch.testing.assert_close(
                    model.transformer.h[index].state_dict()[name], tensor, rtol=0, atol=0
                )
        for module in ("wte", "wpe", "ln_f"):
            for name, tensor in getattr(original.transformer, module).state_dict().items():
                torch.testing.assert_close(
                    getattr(model.transformer, module).state_dict()[name], tensor, rtol=0, atol=0
                )
        assert architecture_manifest(model)["initialization_reference_layers"] == 8


def test_storage_sharing_and_executed_depth_are_counted_separately():
    models = {arm: construct(spec(arm), "cpu") for arm in ARMS}
    reports = {arm: architecture_manifest(model) for arm, model in models.items()}
    shared, untied, partial, shallow = [reports[arm] for arm in ARMS]
    assert shared["unique_parameters"] == shallow["unique_parameters"]
    assert shared["unique_parameters"] < partial["unique_parameters"] < untied["unique_parameters"]
    assert [reports[arm]["executed_blocks"] for arm in ARMS] == [6, 6, 6, 2]
    assert [reports[arm]["unique_mlp_modules"] for arm in ARMS] == [2, 6, 2, 2]
    assert [reports[arm]["unique_attention_modules"] for arm in ARMS] == [2, 6, 6, 2]
    blocks = models["mlp_shared"].transformer.h
    assert blocks[0].mlp is blocks[2].mlp is blocks[4].mlp
    assert blocks[0].ln_2 is not blocks[2].ln_2
    assert blocks[0].attn is not blocks[2].attn
    for model in models.values():
        optimizer = optimizer_for(model, 0.001, 0.1)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(parameters) == len({id(parameter) for parameter in parameters})
        assert (
            sum(p.numel() for p in parameters) == architecture_manifest(model)["unique_parameters"]
        )


@pytest.mark.parametrize("arm", ["shared", "mlp_shared"])
def test_shared_gradient_equals_sum_over_independent_occurrences(arm):
    shared = construct(spec(arm), "cpu").double()
    untied = construct(spec("untied"), "cpu").double()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = torch.tensor([[2, 3, 4, 30], [6, 7, 8, 30]])
    for model in (shared, untied):
        F.cross_entropy(model(tokens).flatten(0, 1), labels.flatten()).backward()
    for block_index, block in enumerate(shared.transformer.h):
        for name, parameter in block.named_parameters():
            occurrences = (
                [block_index % 2 + 2 * repeat for repeat in range(3)]
                if arm == "shared" or name.startswith("mlp.")
                else [block_index]
            )
            expected = sum(
                dict(untied.transformer.h[index].named_parameters())[name].grad
                for index in occurrences
            )
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-11, atol=1e-12)
    independent_parameters = dict(untied.named_parameters())
    for name, parameter in shared.named_parameters():
        if not name.startswith("transformer.h."):
            torch.testing.assert_close(
                parameter.grad, independent_parameters[name].grad, rtol=1e-11, atol=1e-12
            )


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_restores_sharing_and_exact_optimizer_rng_continuation(tmp_path, arm):
    run_spec = spec(arm, dropout=0.1)
    model = construct(run_spec, "cpu")
    optimizer = optimizer_for(model, 0.001, 0.1)
    update(model, optimizer, records(), run_spec, "cpu", 30)
    checkpoint = tmp_path / "resume.pt"
    manifest = {"plan_sha256": "same-plan"}
    st.save_checkpoint(
        checkpoint, model, optimizer, 1, np.ones(2, dtype=np.int64), {}, [], manifest
    )
    before = architecture_manifest(model)
    update(model, optimizer, records(), run_spec, "cpu", 30)
    restored = construct(run_spec, "cpu")
    restored_optimizer = optimizer_for(restored, 0.001, 0.1)
    st.restore_checkpoint(checkpoint, restored, restored_optimizer, manifest)
    assert architecture_manifest(restored) == before
    update(restored, restored_optimizer, records(), run_spec, "cpu", 30)
    assert model_digest(restored) == model_digest(model)
    if arm == "mlp_shared":
        assert restored.transformer.h[0].mlp is restored.transformer.h[2].mlp
    if arm == "untied":
        assert restored.transformer.h[0].mlp is not restored.transformer.h[2].mlp
        assert not torch.equal(
            restored.transformer.h[0].mlp.c_proj.weight,
            restored.transformer.h[2].mlp.c_proj.weight,
        )


def test_flops_use_executed_depth_and_match_existing_training_accounting():
    estimates = {}
    for arm in ARMS:
        model = construct(spec(arm), "cpu")
        optimizer = optimizer_for(model, 0.001, 0.1)
        result = update(model, optimizer, records(), spec(arm), "cpu", 30)
        estimates[arm] = estimate_training_flops(model, 16, 4, 128)
        assert estimates[arm] == result["estimated_matmul_training_flops"]
    assert estimates["shared"] == estimates["untied"] == estimates["mlp_shared"]
    assert estimates["shallow"] < estimates["shared"]


@pytest.mark.parametrize(
    "change",
    [
        {"arm": "unknown"},
        {"base_unique_blocks": 0},
        {"repeats": 0},
        {"initialization_reference_layers": 1},
    ],
)
def test_invalid_intervention_is_rejected(change):
    run_spec = copy.deepcopy(spec())
    run_spec.update(change)
    with pytest.raises(ValueError):
        construct(run_spec, "cpu")
