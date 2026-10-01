"""E39 sampling, full-answer/coverage scoring, and exact short-trajectory resume."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_cross_continue import tree_hash
from llm_memory_editability.bios_model import CausalLM, ModelConfig, select_parameters
from llm_memory_editability.bios_shortcut_control import high_exception_world
from llm_memory_editability.bios_shortcut_edit import (
    STUDY,
    capture_state,
    edit_metrics,
    edit_update,
    restore_state,
    sampling_stream,
    score_arrays,
    validate_study,
)
from llm_memory_editability.bios_shortcut_matched_edit import make_matched_edit_pair


@pytest.fixture(scope="module")
def pair():
    from test_bios_cross import cross_world

    low = cross_world.__wrapped__()
    high, _, _, _ = high_exception_world(low)
    return low, high, make_matched_edit_pair(low, high, 0)


def test_sampling_is_shared_across_all_other_factors():
    reference = sampling_stream(0, 1)
    assert reference[0].shape == reference[1].shape == (512, 128)
    assert reference[0].min() >= 0 and reference[0].max() < 39
    assert reference[1].min() >= 0 and reference[1].max() < 4096
    for _phase in ("low", "high"):
        for _seed in (0, 1):
            for _condition in ("company", "project", "neither"):
                for _kind in ("coherent", "exception"):
                    current = sampling_stream(0, 1)
                    for a, b in zip(reference, current, strict=True):
                        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(reference[0], sampling_stream(1, 1)[0])
    assert not np.array_equal(reference[0], sampling_stream(0, 0)[0])


def test_query_scoring_uses_EOS_and_fixed_reference_denominators(pair):
    low, _high, sets = pair
    target = sets["low_exception"]
    arrays = {"prediction": target.copy(), "ended": np.ones(len(target), dtype=bool)}
    arrays["prediction"][sets["E_roots"][0]] = -1
    arrays["ended"][sets["exception_conflict_D_heldout"][0]] = False
    arrays["correct"] = (arrays["prediction"] == target) & arrays["ended"]
    metrics = edit_metrics(sets, "low", "exception", arrays, np.ones(len(low.answers), bool))
    assert metrics["E"]["correct"] == 38 and metrics["E"]["n"] == 39
    assert metrics["E_roots"]["accuracy"] == 2 / 3
    assert metrics["E_actual"]["accuracy"] == 1
    assert metrics["paired_reference_D_heldout"]["correct"] == 8
    assert metrics["exception_conflict_D_heldout"]["n"] == 9
    arrays["correct"][sets["exception_conflict_D_heldout"][0]] = True
    with pytest.raises(AssertionError):
        score_arrays(arrays, target)
    coherent = sets["high_coherent"]
    arrays = dict(
        prediction=coherent.copy(),
        ended=np.ones(len(coherent), bool),
        correct=np.ones(len(coherent), bool),
    )
    metrics = edit_metrics(sets, "high", "coherent", arrays, np.ones(len(coherent), bool))
    assert "exception_conflict_D" not in metrics
    assert metrics["paired_reference_D"]["n"] == 18


def test_retention_uses_each_parent_old_correct_coverage(pair):
    _low, _high, sets = pair
    target = sets["high_exception"]
    old_correct = np.zeros(len(target), dtype=bool)
    known = sets["U_heldout"][0]
    old_correct[known] = True
    arrays = dict(
        prediction=target.copy(),
        ended=np.ones(len(target), bool),
        correct=np.ones(len(target), bool),
    )
    arrays["ended"][known] = arrays["correct"][known] = False
    metrics = edit_metrics(sets, "high", "exception", arrays, old_correct)
    assert metrics["U_full"]["known"] == metrics["U_heldout"]["known"] == 1
    assert metrics["U_full"]["broken"] == metrics["U_heldout"]["broken"] == 1
    assert metrics["U_full"]["damage"] == metrics["U_heldout"]["damage"] == 1
    group = str(sets["U_strata"][known])
    assert metrics["U_full"]["strata"][group]["damage"] == 1
    old_correct[:] = False
    assert edit_metrics(sets, "high", "exception", arrays, old_correct)["U_full"]["damage"] is None


def test_short_resume_matches_uninterrupted_MLP_weights_optimizer_and_rng(tmp_path):
    torch.set_num_threads(1)
    device = torch.device("cpu")

    def setup():
        torch.manual_seed(804)
        model = CausalLM(ModelConfig(vocab_size=32, width=8, layers=8, heads=1))
        selected = select_parameters(model, "mlp", 3)
        optimizer = torch.optim.AdamW(selected, lr=3e-5, weight_decay=0.1)
        tokens = torch.randint(0, 32, (8, 6))
        data = dict(
            tokens=tokens, positions=torch.tensor([[3, 4]]).repeat(8, 1), labels=tokens[:, 4:6]
        )
        new = copy.deepcopy(data)
        new["tokens"][:4, 4:6] = (new["tokens"][:4, 4:6] + 1) % 32
        new["labels"] = new["tokens"][:, 4:6]
        with torch.no_grad():
            references = model(data["tokens"][4:], data["positions"][4:]).detach()
        return model, selected, optimizer, data, new, references

    e, r = sampling_stream(19, 0, steps=7, batch=4, e_size=4, r_size=4)

    def update(parts, step):
        model, selected, optimizer, data, new, references = parts
        # The production stream is precomputed. This extra draw also makes the
        # checkpoint RNG restoration observable independently of model dropout.
        torch.rand(1)
        edit_update(
            model,
            optimizer,
            selected,
            data,
            new,
            references,
            torch.as_tensor(e[step]),
            torch.as_tensor(r[step] + 4),
            torch.as_tensor(r[step]),
            device,
        )

    parts = setup()
    original = copy.deepcopy(parts[0].state_dict())
    for step in range(3):
        update(parts, step)
    path = tmp_path / "resume.pt"
    torch.save(capture_state(parts[0], parts[2], device, step=3), path)
    for step in range(3, 7):
        update(parts, step)
    expected = tree_hash(capture_state(parts[0], parts[2], device, step=7))
    restored = setup()
    restore_state(restored[0], restored[2], torch.load(path, weights_only=False), device)
    for step in range(3, 7):
        update(restored, step)
    assert tree_hash(capture_state(restored[0], restored[2], device, step=7)) == expected
    updated = []
    for name, value in parts[0].state_dict().items():
        if not torch.equal(value, original[name]):
            updated.append(name)
            assert any(name.startswith(f"blocks.{layer}.mlp.") for layer in (3, 4, 5))
    assert updated


def test_prepared_runner_has_all_24_parents_and_96_cases(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_bios_shortcut_edit.py"
    spec = importlib.util.spec_from_file_location("E39_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = root / "configs/bios-shortcut-edit-v1.json"
    validate_study(json.loads(config.read_text()))
    matrix = runner.jobs(tmp_path / "low", tmp_path / "high", tmp_path / "output", config, "python")
    assert len(matrix) == len({row["output"] for row in matrix}) == 24
    assert {(r["phase"], r["world"], r["seed"], r["condition"]) for r in matrix} == {
        (phase, world, seed, condition)
        for phase in ("low", "high")
        for world in (0, 1)
        for seed in (0, 1)
        for condition in ("company", "project", "neither")
    }
    assert STUDY["edit_cases"] == 4 * len(matrix) == 96
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        validate_study({**STUDY, "scope": "all"})


def test_summary_pairs_world_seed_organization_chain_and_never_invents_conflicts():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/summarize_bios_shortcut_edit.py"
    spec = importlib.util.spec_from_file_location("E39_summary_test", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    rows = []
    for world in (0, 1):
        for phase in ("low", "high"):
            for kind in ("coherent", "exception"):
                row = dict(
                    phase=phase,
                    world=world,
                    seed=0,
                    condition="company",
                    chain="project",
                    kind=kind,
                    step=512,
                    paired_reference_D_heldout_accuracy=world / 2,
                    U_full_damage=None if phase == "low" else 0.1,
                )
                row["paired_reference_D_heldout_accuracy"] += 0.25 * (phase == "high")
                if kind == "exception":
                    row["exception_conflict_D_heldout_accuracy"] = row[
                        "paired_reference_D_heldout_accuracy"
                    ]
                rows.append(row)
    phase_pairs = summary.pair_rows(rows, "phase", ("low", "high"))
    assert len(phase_pairs) == 4
    assert all(
        row["high_minus_low_paired_reference_D_heldout_accuracy"] == 0.25 for row in phase_pairs
    )
    assert all(row["high_minus_low_U_full_damage"] is None for row in phase_pairs)
    kind_pairs = summary.pair_rows(rows, "kind", ("coherent", "exception"))
    assert len(kind_pairs) == 4
    assert not any("conflict" in key for row in kind_pairs for key in row)
