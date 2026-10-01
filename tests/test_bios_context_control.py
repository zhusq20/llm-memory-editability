"""Scientific contracts for the conditional document-context intervention."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability import bios_cross_train as frozen_trainer
from llm_memory_editability.bios_context_control import (
    FactIsolatedCausalLM,
    fact_causal_mask,
    isolated_model_factory,
    validate_study,
)
from llm_memory_editability.bios_cross import documents, render
from llm_memory_editability.bios_model import CausalLM, ModelConfig


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def models(vocab_size=128):
    config = ModelConfig(vocab_size=vocab_size, width=16, layers=2, heads=4)
    torch.manual_seed(813)
    reference = CausalLM(config)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(813)
    isolated = FactIsolatedCausalLM(config)
    assert torch.equal(rng, torch.get_rng_state())
    return reference, isolated


def document_positions(batch):
    return (torch.arange(10)[:, None] * 6 + torch.tensor([3, 4])).flatten()[None].expand(batch, -1)


def test_same_initialized_parameters_rng_and_checkpoint_schema():
    reference, isolated = models()
    assert reference.config_dict() == isolated.config_dict()
    assert reference.state_dict().keys() == isolated.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(value, isolated.state_dict()[key])
    assert sum(p.numel() for p in reference.parameters()) == sum(
        p.numel() for p in isolated.parameters()
    )
    reference.load_state_dict(isolated.state_dict(), strict=True)
    assert not any("fact_mask" in key for key in isolated.state_dict())


def test_mask_has_no_cross_fact_or_future_edges():
    mask = fact_causal_mask()
    assert mask.dtype == torch.bool and mask.shape == (60, 60)
    assert int(mask.sum()) == 210
    for query in range(60):
        for key in range(60):
            assert bool(mask[query, key]) == (query // 6 == key // 6 and key <= query)
    with pytest.raises(ValueError):
        fact_causal_mask(59)


def test_document_causal_reachability_and_original_positive_control():
    reference, isolated = models()
    tokens = torch.arange(60)[None]
    positions = document_positions(1)
    for model, blocked in ((reference, False), (isolated, True)):
        activations = []

        def capture(_module, _inputs, output, activations=activations):
            output.retain_grad()
            activations.append(output)

        handle = model.token.register_forward_hook(capture)
        model.train()
        model(tokens, positions)[0, 2, 77].backward()  # Query position 9, fact 1.
        handle.remove()
        gradient = activations[0].grad[0]
        assert bool((gradient[6:10].abs().sum(dim=-1) > 0).all())
        assert torch.equal(gradient[10:], torch.zeros_like(gradient[10:]))
        if blocked:
            assert torch.equal(gradient[:6], torch.zeros_like(gradient[:6]))
        else:
            assert gradient[:6].abs().sum() > 0
    changed = tokens.clone()
    changed[:, :6] += 61
    with torch.no_grad():
        baseline = isolated(tokens, positions)
        perturbed = isolated(changed, positions)
    torch.testing.assert_close(baseline[:, 2:], perturbed[:, 2:], rtol=0, atol=0)
    assert not torch.equal(baseline[:, :2], perturbed[:, :2])


def test_QA_training_and_all_inference_use_exact_original_forward():
    reference, isolated = models()
    for training, length in ((True, 6), (False, 5), (False, 6), (False, 60)):
        reference.train(training)
        isolated.train(training)
        tokens = torch.arange(length)[None].repeat(2, 1)
        positions = torch.tensor([[length - 2, length - 1]]).expand(2, -1)
        a, b = reference(tokens, positions), isolated(tokens, positions)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if training:
            target = torch.tensor([[21, 2], [22, 2]])
            F.cross_entropy(a.flatten(0, 1), target.flatten()).backward()
            F.cross_entropy(b.flatten(0, 1), target.flatten()).backward()
            for old, new in zip(reference.parameters(), isolated.parameters(), strict=True):
                torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
    assert all(not block.attention.document_attention_isolated for block in isolated.blocks)


def test_invalid_training_dispatch_or_document_supervision_fails_closed():
    _, model = models()
    model.train()
    with pytest.raises(ValueError, match="training input length"):
        model(torch.zeros((1, 12), dtype=torch.long))
    with pytest.raises(ValueError, match="twenty supervised"):
        model(torch.zeros((1, 60), dtype=torch.long))
    wrong_positions = document_positions(1).clone()
    wrong_positions[0, 0] = 2
    with pytest.raises(ValueError, match="supervision positions"):
        model(torch.zeros((1, 60), dtype=torch.long), wrong_positions)


def test_fakeworld_documents_keep_absolute_positions_and_joint_loss():
    from test_bios_cross import cross_world

    world = cross_world.__wrapped__()
    arrays = render(world, documents(world, "company")[:2])
    preserved = {key: value.copy() for key, value in arrays.items()}
    batch = {key: torch.as_tensor(value.copy()) for key, value in arrays.items()}
    reference, isolated = models(world.vocab_size)
    qa = frozen_trainer.tensor_queries(world, torch.device("cpu"))
    ids = torch.as_tensor(world.train_ids[:, :2].ravel())
    isolated.train()
    actual = isolated(batch["tokens"], batch["positions"])
    # Independent facts use the SAME absolute offsets, not six-token position resets.
    pieces = []
    for start in range(0, 60, 6):
        x = reference.token(batch["tokens"][:, start : start + 6])
        x = x + reference.position(torch.arange(start, start + 6))
        for block in reference.blocks:
            x = block(x)
        x = reference.ln_final(x)[:, 3:5]
        pieces.append(F.linear(x, reference.token.weight))
    expected = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)
    loss = F.cross_entropy(actual.flatten(0, 1), batch["labels"].flatten())
    reference_loss = F.cross_entropy(expected.flatten(0, 1), batch["labels"].flatten())
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=1e-6)
    (0.8 * loss).backward()
    (0.8 * reference_loss).backward()
    for model in (isolated, reference):
        logits = model(qa["tokens"][ids], qa["positions"][ids])
        (0.2 * F.cross_entropy(logits.flatten(0, 1), qa["labels"][ids].flatten())).backward()
    for old, new in zip(reference.parameters(), isolated.parameters(), strict=True):
        torch.testing.assert_close(old.grad, new.grad, rtol=3e-4, atol=2e-6)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], preserved[key])
        np.testing.assert_array_equal(batch[key].numpy(), preserved[key])


def test_factory_restores_original_trainer_after_success_and_error():
    hashes = frozen_trainer.source_hashes()
    with isolated_model_factory():
        assert frozen_trainer.CausalLM is FactIsolatedCausalLM
        with pytest.raises(RuntimeError, match="modified model factory"), isolated_model_factory():
            pass
    assert frozen_trainer.CausalLM is CausalLM
    with pytest.raises(RuntimeError, match="simulated"), isolated_model_factory():
        raise RuntimeError("simulated")
    assert frozen_trainer.CausalLM is CausalLM
    assert hashes == frozen_trainer.source_hashes()


def test_full_conditional_matrix_and_config(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_bios_context_control.py"
    spec = importlib.util.spec_from_file_location("context_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = root / "configs/bios-context-control-v1.json"
    study = json.loads(config_path.read_text())
    validate_study(study)
    matrix = module.jobs(tmp_path, config_path, python="python")
    assert len(matrix) == len({row["output"] for row in matrix}) == 12
    assert {(row["world"], row["seed"], row["condition"]) for row in matrix} == {
        (world, seed, condition)
        for world in (0, 1)
        for seed in (0, 1)
        for condition in ("company", "project", "neither")
    }
    assert all("--condition" in row["command"] for row in matrix)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="document_QA_weights"):
        validate_study({**study, "document_QA_weights": [0.5, 0.5]})
