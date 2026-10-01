import copy
from types import SimpleNamespace

import numpy as np
import pytest

from llm_memory_editability.hebbian_learning import calibration_trajectory, q_auc


def test_discrete_matrix_sgd_has_exact_two_modes():
    rho = 0.99
    phi = np.array([[1, rho], [0, np.sqrt(1 - rho**2)], [0, 0]])
    audit, norms, predicted, _ = calibration_trajectory(phi, 10000, 0.05, 4, [0, 16])
    assert audit["pass"]
    assert audit["crossings"][-1]["observed_step"] == 9209
    np.testing.assert_allclose(norms[:100], predicted[:100], atol=1e-13)


def test_auc_uses_linear_steps_and_nonzero_baseline():
    assert q_auc([0.2, 0.4, 0.8], [0, 1, 4]) == pytest.approx(0.525)


@pytest.fixture
def tiny_engine():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from llm_memory_editability.hebbian_model import QwenExperiment

    torch.manual_seed(123)
    engine = QwenExperiment.__new__(QwenExperiment)
    engine.device = torch.device("cpu")
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    engine.model = Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    engine.layer = engine.model.model.layers[1]
    engine.mlp = engine.layer.mlp
    engine.tokenizer = SimpleNamespace(pad_token_id=0)
    engine.base_local = {n: p.detach().clone() for n, p in engine.mlp.named_parameters()}
    return engine


def encoded(ids, start):
    return {
        "input_ids": ids,
        "answer_start": start,
        "loss_mask": [0] * start + [1] * (len(ids) - start),
    }


def test_all_position_gradient_reconstruction(tiny_engine):
    audit = tiny_engine.verify_hooks_and_gradients(encoded([1, 2, 3, 4, 5, 6], 4))
    assert audit["pass"]
    assert audit["earlier_prompt_delta_norm"] > 0


def test_batch_ce_is_mean_of_sequence_means(tiny_engine):
    engine = tiny_engine
    items = [encoded([1, 2, 3, 4], 2), encoded([3, 4, 5, 6, 7, 8], 3)]
    batched = engine.ce(items).detach().numpy()
    singles = np.array([float(engine.ce([x])[0]) for x in items])
    np.testing.assert_allclose(batched, singles, rtol=1e-6)
    engine.reference = copy.deepcopy(engine.model)
    assert float(engine.kl(items).abs().max()) < 1e-6


def test_derangement_changes_groups_without_changing_truths():
    torch = pytest.importorskip("torch")
    from llm_memory_editability.hebbian_train import contrastive_loss, grouping

    records = [{"target_id": str(i), "relation_id": str(i % 2), "answer": str(i)} for i in range(8)]
    before = copy.deepcopy(records)
    groups = grouping(records, 0)
    assert sorted(i for g in groups for i in g) == list(range(24))
    assert all(g[0] // 3 != g[1] // 3 and g[0] // 3 != g[2] // 3 for g in groups)
    assert records == before
    features = torch.randn(24, 16, requires_grad=True)
    loss, counts = contrastive_loss(
        features, [r["target_id"] for r in records for _ in range(3)], groups
    )
    loss.backward()
    assert counts["positive_pairs"] == 48
    assert features.grad.isfinite().all()


def test_checkpoint_resume_repeats_adam_exactly(tiny_engine, tmp_path):
    torch = pytest.importorskip("torch")
    from llm_memory_editability.hebbian_train import optimizer, restore_checkpoint, save_checkpoint

    engine = tiny_engine
    parameters = engine.configure_trainable("adapt")
    opt = optimizer(parameters, 1e-3)
    items = [encoded([1, 2, 3, 4], 2)]
    frozen = {
        n: p.detach().clone() for n, p in engine.model.named_parameters() if not p.requires_grad
    }

    def step():
        opt.zero_grad(set_to_none=True)
        engine.ce(items).mean().backward()
        opt.step()

    step()
    path = tmp_path / "state.pt"
    save_checkpoint(path, engine, opt, 1, {"test": True}, {"tokens": 4})
    step()
    expected = engine.mlp.down_proj.weight.detach().clone()
    restore_checkpoint(path, engine, opt, {"test": True})
    step()
    assert torch.equal(expected, engine.mlp.down_proj.weight)
    assert all(
        torch.equal(p, frozen[n]) for n, p in engine.model.named_parameters() if not p.requires_grad
    )
    with pytest.raises(RuntimeError, match="contract differs"):
        restore_checkpoint(path, engine, opt, {"test": False})


def test_ridge_and_exact_paired_inference():
    from llm_memory_editability.hebbian_statistics import fit_ridge, paired_inference, predict

    x = np.arange(40).reshape(10, 4).astype(float)
    y = np.arange(10) / 10
    model = fit_ridge(x, y, 0.1)
    assert np.max(np.abs(predict(model, x) - y)) < 0.002
    result = paired_inference(np.ones(8))
    assert result["exact_two_sided_sign_p"] == 2 / 256


def test_chunked_geometry_matches_whole_batch(tiny_engine):
    import torch
    import torch.nn.functional as functional

    from llm_memory_editability.hebbian_train import geometry_rows

    records = [
        {
            "case_id": i,
            "target_id": str(i),
            "encoded": [encoded([1, 2, 3 + i, 4, 5], 2 + view) for view in range(3)],
        }
        for i in range(9)
    ]
    items = [e for r in records for e in r["encoded"]]
    full = tiny_engine.features(items).double().reshape(9, 3, -1)
    normalized = functional.normalize(full, dim=-1)
    expected = (normalized[:, 1:] * normalized[:, :1]).sum(-1).mean(-1)
    actual = geometry_rows(tiny_engine, records)
    torch.testing.assert_close(
        torch.tensor([actual[i]["same_fact_cos"] for i in range(9)], dtype=torch.float64),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
