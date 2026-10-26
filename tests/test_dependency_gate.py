"""Scientific contracts for the dependency-task adapter, without GPU training."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_dependency_gate as runner  # noqa: E402


def configuration(task="listops", tiny=False):
    root = json.loads((ROOT / "configs/sequence-dependency-gate-v1.json").read_text())
    config = runner.resolve_config(root, task)
    if tiny:
        config["model"] = {
            **config["model"],
            "n_embd": 12,
            "n_head": 3,
            "n_inner": 48,
            "n_positions": 16,
            "n_ctx": 16,
            "attn_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "resid_pdrop": 0.0,
        }
        config["repeats"] = 4
        config["arms"] = {key: {**value, "repeats": 4} for key, value in config["arms"].items()}
    return config


def metadata():
    return {
        "task": "listops",
        "vocab_size": 23,
        "pad_token": 0,
        "end_marker": 2,
        "sequence_length": 7,
        "bos_token": 1,
    }


def panel():
    return {
        "tokens": np.array(
            [[1, 8, 9, 4, 0, 0, 0], [1, 8, 9, 5, 0, 0, 0], [1, 8, 10, 9, 6, 0, 0]], dtype=np.int64
        ),
        "positions": np.array([[2, 3], [2, 3], [3, 4]], dtype=np.int64),
        "labels": np.array([[4, 2], [5, 2], [6, 2]], dtype=np.int64),
        "length": np.array([4, 4, 5]),
        "depth": np.array([1, 1, 2]),
        "tree_depth": np.array([2, 2, 3]),
        "branch_imbalance": np.array([0, 1, 1]),
        "computed_branch_imbalance": np.array([0, 0, 1]),
        "join_count": np.array([0, 1, 2]),
        "ids": np.array([b"a", b"b", b"c"], dtype="S64"),
    }


@pytest.mark.parametrize("task", tuple(runner.TASKS))
def test_registered_recipe_resolves_without_mutation_and_keeps_test_out_of_training(task):
    root = json.loads((ROOT / "configs/sequence-dependency-gate-v1.json").read_text())
    before = copy.deepcopy(root)
    config = runner.resolve_config(root, task)
    runner.validate_config(config)
    assert root == before
    assert config["batch"] == root["batch"]
    assert set(config["arms"]) == set(runner.ARMS)
    assert config["repeats"] in config["inference_repeats"]
    changed = copy.deepcopy(config)
    changed["training_panels"] = [*runner.TRAIN_PANELS, "test_iid"]
    with pytest.raises(ValueError, match="Test panels"):
        runner.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["inference_repeats"] = [1, 4]
    with pytest.raises(ValueError, match="inference grid"):
        runner.validate_config(changed)
    changed = copy.deepcopy(root)
    changed["arms"]["only-u"]["initial_update"] = 0.5
    with pytest.raises(ValueError, match="architecture"):
        runner.resolve_config(changed, task)


def test_four_arm_initial_backbones_rng_objectives_and_costs_are_paired():
    config = configuration(tiny=True)
    states, rngs, costs = {}, {}, {}
    for name, (injection, mode) in runner.ARMS.items():
        model, spec, arm, _ = runner.initialize(config, name, metadata(), torch.device("cpu"))
        states[name] = {
            n: p.clone() for n, p in model.state_dict().items() if not n.startswith("gate_")
        }
        rngs[name] = torch.get_rng_state().clone()
        costs[name] = runner.accounting(model, spec, arm, metadata())
        assert arm["kind"] == "none" and arm["coefficient"] == 0
        assert arm["injection"] == injection and arm["update_mode"] == mode
        assert not any(n.startswith("gate_paper_") for n in model.state_dict())
        assert costs[name]["update_gate_parameters"] == (13 if mode == "token" else 0)
        assert costs[name]["channel_calls_per_forward"] == (3 if injection == "fixed1" else 0)
        assert costs[name]["padded_sequence_length"] == 7
    for name in runner.ARMS:
        assert torch.equal(rngs["loop"], rngs[name])
        assert states["loop"].keys() == states[name].keys()
        assert all(torch.equal(p, states[name][n]) for n, p in states["loop"].items())
    assert (
        costs["loop"]["estimated_matmul_training_flops_per_step"]
        == costs["only-u"]["estimated_matmul_training_flops_per_step"]
    )
    assert (
        costs["disco"]["estimated_matmul_training_flops_per_step"]
        == costs["fixed-q-plus-u"]["estimated_matmul_training_flops_per_step"]
    )


@pytest.mark.parametrize("name", tuple(runner.ARMS))
def test_real_model_causality_blocks_teacher_answers_and_right_padding_in_all_four_arms(name):
    model, *_ = runner.initialize(configuration(tiny=True), name, metadata(), torch.device("cpu"))
    model = model.double().eval()
    model.repeats = 8
    original = torch.tensor(panel()["tokens"])
    changed = original.clone()
    positions = torch.tensor(panel()["positions"])
    for row, position in enumerate(positions[:, 1]):
        changed[row, position:] = torch.arange(len(changed[row, position:])) + 11
    with torch.no_grad():
        first = model(original, positions[:, :1])
        second = model(changed, positions[:, :1])
        padded = model(F_pad(original), positions[:, :1])
    assert torch.equal(first, second)
    torch.testing.assert_close(first, padded, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(first).all()


def F_pad(tokens):
    return torch.cat((tokens, torch.zeros(len(tokens), 2, dtype=tokens.dtype)), dim=1)


class GeneratedAnswerEOS(torch.nn.Module):
    """Wrong answer 7; EOS is correct iff the evaluator actually feeds 7 back."""

    def forward(self, tokens, positions):
        result = torch.zeros(len(tokens), 2, 23)
        result[:, 0, 7] = 8.0
        generated = tokens[torch.arange(len(tokens)), positions[:, 1]] == 7
        result[:, 1, 1] = (~generated).float() * 8
        result[:, 1, 2] = generated.float() * 8
        return result


def test_generated_eos_uses_predicted_answer_and_preserves_full_denominators_and_data():
    rows = panel()
    before = {key: value.copy() for key, value in rows.items()}
    model = GeneratedAnswerEOS().train()
    metrics, prediction = runner.evaluate(model, rows, metadata(), torch.device("cpu"), 2)
    assert model.training
    assert np.all(prediction["answer"] == 7)
    assert np.all(prediction["stop"] == 2)
    assert metrics["n"] == 3 and metrics["correct"] == 0
    assert metrics["answer_accuracy"] == metrics["accuracy"] == 0
    assert prediction["nll"].shape == (3, 2)
    assert metrics["by_join_presence"]["0"]["n"] == 1
    assert metrics["by_join_presence"]["ge1"]["n"] == 2
    for feature in (
        "depth",
        "tree_depth",
        "length",
        "branch_imbalance",
        "computed_branch_imbalance",
    ):
        assert sum(item["n"] for item in metrics[f"by_{feature}"].values()) == 3
    assert all(np.array_equal(before[key], rows[key]) for key in rows)
    assert np.array_equal(prediction["ids"], rows["ids"])


def test_empty_join_stratum_is_retained_without_nan():
    rows = panel()
    rows["join_count"][:] = 0
    metrics, _ = runner.evaluate(GeneratedAnswerEOS(), rows, metadata(), torch.device("cpu"), 2)
    assert metrics["by_join_presence"]["ge1"]["n"] == 0
    assert metrics["by_join_presence"]["ge1"]["answer_accuracy"] is None
    json.dumps(metrics, allow_nan=False)


def test_training_measurement_never_accesses_test_panels():
    class GuardedPanels(dict):
        def __getitem__(self, key):
            if key in runner.TEST_PANELS:
                raise AssertionError("test accessed during training")
            return super().__getitem__(key)

    panels = GuardedPanels({name: panel() for name in runner.ALL_PANELS})
    measured, predicted = runner.measure_panels(
        GeneratedAnswerEOS(), {}, panels, metadata(), torch.device("cpu"), 2
    )
    assert set(measured) == set(predicted) == set(runner.TRAIN_PANELS)


def test_endpoint_sweep_covers_all_panels_and_restores_state_rng_and_repeats():
    config = configuration(tiny=True)
    model, *_ = runner.initialize(config, "only-u", metadata(), torch.device("cpu"))
    model.train()
    before = runner.model_digest(model)
    rng = torch.get_rng_state().clone()
    panels = {name: panel() for name in runner.ALL_PANELS}
    metrics, predictions = runner.sweep(
        model, {}, panels, metadata(), torch.device("cpu"), 2, [1, 4, 8]
    )
    assert set(metrics) == set(predictions) == {"1", "4", "8"}
    assert all(set(item) == set(runner.ALL_PANELS) for item in metrics.values())
    assert model.training and model.repeats == 4
    assert runner.model_digest(model) == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_endpoint_rejects_nonterminal_checkpoint_before_loading_weights(tmp_path, monkeypatch):
    out = tmp_path / "runs" / "loop"
    out.mkdir(parents=True)
    (out / "complete.json").write_text(json.dumps({"state": "complete", "step": 75000}))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("weights loaded"))
    with pytest.raises(ValueError, match="fixed registered terminal"):
        runner.endpoint(
            configuration(), "loop", tmp_path, ROOT / "configs/sequence-dependency-gate-v1.json"
        )


def test_install_restores_callbacks_and_preserves_original_training_and_step():
    changed = (
        "initialize",
        "accounting",
        "optimizer_for",
        "load",
        "make_table",
        "exposure_summary",
        "measure_panels",
    )
    preserved = ("train", "AlignmentStep", "MergedBatchStream", "learning_rate")
    before = {key: getattr(runner.native, key) for key in (*changed, *preserved)}
    with pytest.raises(RuntimeError, match="fixture"):
        with runner.installed():
            assert runner.native.measure_panels is runner.measure_panels
            assert all(getattr(runner.native, key) is before[key] for key in preserved)
            raise RuntimeError("fixture")
    assert all(getattr(runner.native, key) is value for key, value in before.items())
