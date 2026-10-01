"""Truth, information, weighting, and exact-restart contracts for P1."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability import bios_mechanism_edit as mechanism
from llm_memory_editability.bios_cross import edit_pair
from llm_memory_editability.bios_cross_train import evaluate, tensor_queries
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_organization_train import state_hash


@pytest.fixture(scope="module")
def world():
    from test_bios_cross import cross_world

    return cross_world.__wrapped__()


@pytest.mark.parametrize("chain", [0, 1])
def test_counterfactual_factorial_and_recomputed_truth(world, chain):
    pair = edit_pair(world, chain)
    specs = {
        arm: mechanism.make_arm(world, chain, arm)
        for arm in (*mechanism.ARMS, "coherent", "exception")
    }
    rehearsal = specs["old-fact-rehearsal"]
    root, actual, conflict = specs["root-only"], specs["actual-only"], specs["exception"]
    np.testing.assert_array_equal(rehearsal["target"], world.answers)
    assert len(rehearsal["E_changed"]) == len(rehearsal["D"]) == 0
    assert (len(root["E_changed"]), len(actual["E_changed"]), len(conflict["E_changed"])) == (
        3,
        90,
        93,
    )
    assert len(actual["D"]) == 0 and len(root["D"]) == len(conflict["D"]) == 96
    np.testing.assert_array_equal(
        actual["target"][actual["S_actual"]], conflict["target"][actual["S_actual"]]
    )
    np.testing.assert_array_equal(
        actual["target"][actual["S_root"]], world.answers[actual["S_root"]]
    )
    np.testing.assert_array_equal(root["target"][root["S_actual"]], world.answers[root["S_actual"]])
    np.testing.assert_array_equal(
        conflict["target"] - world.answers, root["target"] + actual["target"] - 2 * world.answers
    )
    for arm, spec in specs.items():
        np.testing.assert_array_equal(spec["S"], pair["E"])
        np.testing.assert_array_equal(spec["replay"], pair["replay"])
        changed = np.flatnonzero(spec["target"] != world.answers)
        np.testing.assert_array_equal(changed, np.union1d(spec["E_changed"], spec["D"]))
        np.testing.assert_array_equal(np.union1d(changed, spec["U"]), np.arange(len(world.answers)))
        assert not np.intersect1d(changed, spec["U"]).size
        np.testing.assert_array_equal(spec["target"][spec["replay"]], world.answers[spec["replay"]])
        for task in range(2):
            np.testing.assert_array_equal(
                spec["target"][world.derived_ids[task]],
                spec["target"][world.root_ids[task, world.memberships[task]]],
            )
        if arm in ("coherent", "exception"):
            np.testing.assert_array_equal(spec["target"], pair[arm])
            np.testing.assert_array_equal(spec["strata"], pair["strata"])
            np.testing.assert_array_equal(spec["heldout"], pair["heldout"])


@pytest.mark.parametrize("chain", [0, 1])
def test_new_unchanged_queries_are_retained_without_supervised_leakage(world, chain):
    root = mechanism.make_arm(world, chain, "root-only")
    actual = mechanism.make_arm(world, chain, "actual-only")
    rehearsal = mechanism.make_arm(world, chain, "old-fact-rehearsal")
    assert len(root["S_unchanged"]) == 90
    assert len(actual["S_unchanged"]) == 3
    assert len(rehearsal["S_unchanged"]) == 93
    np.testing.assert_array_equal(actual["D_probe_unchanged"], actual["D_probe"])
    assert np.isin(actual["D_probe"], actual["heldout"]).all()
    for spec in (root, actual, rehearsal):
        assert np.isin(spec["S_unchanged"], spec["U"]).all()
        for pool in ("heldout", "unseen"):
            assert not np.intersect1d(spec[pool], np.union1d(spec["S"], spec["replay"])).size
    before = np.ones(len(world.answers), dtype=bool)
    current = before.copy()
    current[actual["D_probe"][0]] = False
    metrics = mechanism.arm_metrics(world, actual, {"correct": current}, before)
    assert metrics["D"]["accuracy"] is None
    assert metrics["D"]["n"] == 0
    assert metrics["U_D_probe_unchanged"]["broken"] == 1
    assert metrics["U_heldout"]["broken"] == 1
    empty = mechanism.arm_metrics(world, rehearsal, {"correct": current}, np.zeros_like(before))
    assert empty["E_changed"]["accuracy"] is None
    assert empty["U_full"]["rate"] is None and empty["U_full"]["known"] == 0


def test_class_weights_have_unit_expectation_without_batch_renormalization(world):
    spec = mechanism.make_arm(world, 0, "class-balanced-exception")
    root = spec["S_root_mask"]
    weights = spec["S_weights"]
    np.testing.assert_allclose(weights[root], 15.5)
    np.testing.assert_allclose(weights[~root], 93 / 180)
    assert weights.mean() == pytest.approx(1)
    assert weights[root].sum() == pytest.approx(weights[~root].sum())
    logits = torch.zeros((93, 2, 3), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[root, :, 0] = -2
        logits[~root, :, 0] = 1
    labels = torch.zeros((93, 2), dtype=torch.long)
    loss, terms = mechanism.weighted_supervision(logits, labels, torch.tensor(weights))
    expected = (terms[root].mean() + terms[~root].mean()) / 2
    torch.testing.assert_close(loss, expected.double())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # A batch with no root samples retains their absent mass instead of being renormalized.
    observed, losses = mechanism.weighted_supervision(
        logits[~root], labels[~root], torch.tensor(weights[~root])
    )
    torch.testing.assert_close(observed, losses.mean().double() * (93 / 180))


def test_sampling_is_shared_and_optional_scope_freezes_tied_embedding(world):
    first = mechanism.sampling_streams(world, 0)
    second = mechanism.sampling_streams(world, 0)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.shape == (512, 128)
    model = CausalLM(ModelConfig(32, width=8, layers=8, heads=1))
    selected = mechanism.select_edit_parameters(model, mechanism.OPTIONAL_ARM)
    assert not model.token.weight.requires_grad
    assert model.position.weight.requires_grad
    assert all(id(p) != id(model.token.weight) for p in selected)
    assert len(selected) == len(list(model.parameters())) - 1


def _small_run_fixture(monkeypatch):
    """Small real Transformer optimization isolates restart mechanics from data size."""
    n, vocab = 12, 24
    answers = np.arange(n) % 6 + 8
    prompts = np.zeros((n, 5), dtype=np.int64)
    prompts[:, :4] = np.column_stack(
        [np.ones(n, dtype=int), np.arange(n) + 4, np.full(n, 5), np.full(n, 2)]
    )
    world = SimpleNamespace(
        answers=answers,
        prompts=prompts,
        lengths=np.full(n, 4),
        train_ids=np.array([[8], [9]]),
        heldout_ids=np.array([[10], [11]]),
    )
    target = answers.copy()
    target[:4] = (target[:4] - 8 + 1) % 6 + 8
    spec = {
        "target": target,
        "S": np.arange(4),
        "S_root": np.array([0]),
        "S_actual": np.arange(1, 4),
        "S_root_mask": np.array([True, False, False, False]),
        "S_weights": np.ones(4),
        "E_changed": np.arange(4),
        "E_changed_root": np.array([0]),
        "E_changed_actual": np.arange(1, 4),
        "D": np.array([], dtype=int),
        "D_conflict": np.array([], dtype=int),
        "D_probe": np.arange(8, 12),
        "D_probe_conflict": np.arange(10, 12),
        "D_probe_unchanged": np.arange(8, 12),
        "U": np.arange(4, 12),
        "strata": np.array([-1] * 4 + [0, 1, 2, 3] * 2),
        "heldout": np.arange(8, 12),
        "unseen": np.arange(8, 12),
        "S_unchanged": np.array([], dtype=int),
        "replay": np.arange(4, 8),
        "other_chain": np.array([9, 11]),
        "independent": np.array([], dtype=int),
        "groups": np.arange(3),
    }
    monkeypatch.setattr(mechanism, "make_arm", lambda *_args: spec)
    rng = np.random.default_rng(9)
    streams = (rng.integers(4, size=(2, 128)), rng.integers(4, size=(2, 128)))
    monkeypatch.setattr(mechanism, "sampling_streams", lambda *_args: streams)
    torch.manual_seed(17)
    model = CausalLM(ModelConfig(vocab, width=8, layers=8, heads=1))
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    data = tensor_queries(world, torch.device("cpu"))
    old = evaluate(model, data, answers)
    study = copy.deepcopy(mechanism.DEFAULT_STUDY)
    study.update(edit_steps=2, edit_checkpoints=[0, 1, 2], save_every=1, gradient_steps=[0, 1])
    return world, model, state, data, old, study


def test_restart_restores_optimizer_rng_and_full_state(tmp_path, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        world, model, baseline, data, old, study = _small_run_fixture(monkeypatch)
        args = (model, baseline, world, data, old, 0, "root-only")
        complete = mechanism.run_case(*args, tmp_path / "continuous", study, {"fixture": True})
        continuous = {key: value.detach().clone() for key, value in model.state_dict().items()}
        real_save = mechanism.atomic_torch_save

        def interrupt_after_checkpoint(value, path):
            real_save(value, path)
            if path.name == "resume.pt" and value["step"] == 1:
                raise RuntimeError("simulated interruption")

        monkeypatch.setattr(mechanism, "atomic_torch_save", interrupt_after_checkpoint)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            mechanism.run_case(*args, tmp_path / "resumed", study, {"fixture": True})
        monkeypatch.setattr(mechanism, "atomic_torch_save", real_save)
        resumed = mechanism.run_case(*args, tmp_path / "resumed", study, {"fixture": True})
        assert (
            resumed["final_model_sha256"]
            == complete["final_model_sha256"]
            == state_hash(continuous)
        )
        for key, value in model.state_dict().items():
            assert torch.equal(value, continuous[key])
        state = torch.load(tmp_path / "resumed/resume.pt", weights_only=False)
        assert state["step"] == len(state["update_stats"]) == 2
        assert len(state["optimizer"]["state"]) > 0
        assert [point["step"] for point in state["timeline"]] == [0, 1, 2]
        changed = {**study, "edit_lr": 1e-4}
        with pytest.raises(ValueError, match="Frozen contract mismatch"):
            mechanism.run_case(*args, tmp_path / "resumed", changed, {"fixture": True})
    finally:
        torch.set_num_threads(previous_threads)


def _summary_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_mechanism_edit.py"
    spec = importlib.util.spec_from_file_location("p1_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_response_probes_do_not_confuse_own_truth_or_overlapping_labels(world):
    summary = _summary_module()
    pair = edit_pair(world, 0)
    prediction = world.answers.copy()
    prediction[pair["D"]] = pair["exception"][world.actual_ids[0, world.person[pair["D"]]]]
    arrays = {"prediction": prediction, "ended": np.ones(len(prediction), dtype=bool)}
    metrics = summary.response_probes(world, pair, arrays)
    assert metrics["probe_conflict_heldout_new_actual"]["accuracy"] == 1
    assert metrics["probe_conflict_heldout_new_default"]["accuracy"] == 0
    assert metrics["probe_conflict_heldout_old_default"]["accuracy"] == 0
    assert metrics["probe_labels"]["ambiguous_hits"] >= 45
    assert metrics["probe_conflict_labels"]["ambiguous_hits"] == 0


def test_paired_metrics_keep_nonexistent_propagation_undefined():
    summary = _summary_module()
    rows = []
    for arm, response, propagation in (
        ("old-fact-rehearsal", 0.0, None),
        ("root-only", 0.8, 0.8),
        ("actual-only", 0.1, None),
        ("exception", 0.3, 0.3),
        ("class-balanced-exception", 0.6, 0.6),
    ):
        rows.append(
            {
                "width": 256,
                "world": 0,
                "seed": 1,
                "condition": "company",
                "chain": "project",
                "step": 512,
                "arm": arm,
                "D_heldout_accuracy": propagation,
                "probe_conflict_heldout_new_default_accuracy": response,
            }
        )
    contrasts = {row["contrast"]: row for row in summary.paired_rows(rows)}
    assert contrasts["balanced-minus-uniform"]["delta_D_heldout_accuracy"] == pytest.approx(0.3)
    assert contrasts["actual-minus-rehearsal"]["delta_D_heldout_accuracy"] is None
    interaction = contrasts["root-by-actual-interaction"]
    assert interaction["delta_probe_conflict_heldout_new_default_accuracy"] == pytest.approx(-0.6)
    assert "delta_D_heldout_accuracy" not in interaction


def test_prediction_audit_rejects_stale_correctness(tmp_path, world):
    summary = _summary_module()
    path = tmp_path / "predictions.npz"
    prediction = world.answers.copy()
    correct = np.ones(len(prediction), dtype=bool)
    correct[0] = False
    np.savez(
        path,
        prediction=prediction,
        correct=correct,
        ended=np.ones_like(correct),
        value_nll=np.zeros(len(prediction)),
    )
    with pytest.raises(ValueError, match="Stored correctness differs"):
        summary._read_predictions(path, world.answers)
