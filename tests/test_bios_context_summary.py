"""Tests for causal-control denominators, EOS scoring and paired organization effects."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.bios_cross import make_cross_world


@pytest.fixture
def summary(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "context_summary_test", scripts / "summarize_bios_context_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_error_counts_include_all_exception_people_and_require_eos(summary):
    world = make_cross_world(0)
    prediction = world.answers.copy()
    ids = world.derived_ids[0]
    prediction[ids] = world.answers[world.actual_ids[0]]
    ended = np.ones(len(prediction), dtype=bool)
    failed = ids[np.flatnonzero(world.exceptions[0])[0]]
    ended[failed] = False
    arrays = summary.score_direct(
        dict(
            prediction=prediction,
            ended=ended,
            correct=ended & (prediction == world.answers),
            value_nll=np.zeros(len(prediction)),
        ),
        world.answers,
    )
    rows = summary.path_rows({}, world, 0, "isolated", arrays, "direct")
    exception = next(r for r in rows if r["split"] == "all" and r["cohort"] == "original_exception")
    ordinary = next(r for r in rows if r["split"] == "all" and r["cohort"] == "ordinary")
    assert exception["n"] == 128
    assert exception["correct"] == 0
    assert exception["wrong_actual"] == 127
    assert exception["other_error"] == exception["termination_error"] == 1
    assert ordinary["accuracy"] == 1.0
    assert ordinary["wrong_actual"] == ordinary["conflict_n"] == 0
    assert ordinary["overlap_n"] == ordinary["n"] == 1920


def test_basic_facts_keep_wrong_qa_people_and_measure_their_actual_roots(summary):
    world = make_cross_world(1)
    correct = np.ones(len(world.answers), dtype=bool)
    correct[world.derived_ids] = False
    correct[world.membership_ids[0, :100]] = False
    arrays = {"correct": correct, "value_nll": np.arange(len(correct), dtype=float)}
    rows = summary.component_rows({}, world, 0, "isolated", arrays)
    all_row = next(r for r in rows if r["split"] == "all" and r["cohort"] == "all")
    assert all_row["n"] == 2048
    assert all_row["membership_accuracy"] == (2048 - 100) / 2048
    assert all_row["root_accuracy"] == all_row["actual_accuracy"] == 1.0
    assert all_row["root_nll"] == np.mean(world.root_ids[0, world.memberships[0]])


def test_crossed_organization_effects_do_not_substitute_matching_for_mean_benefit(summary):
    rows = []
    for phase in summary.PHASES:
        for condition in ("company", "project", "neither"):
            for chain in ("company", "project"):
                value = 0.5
                if phase == "unrestricted" and condition != "neither":
                    value += 0.3 if condition == chain else 0.1
                rows.append(
                    dict(
                        world=0,
                        seed=0,
                        step=15360,
                        phase=phase,
                        method="direct",
                        split="heldout",
                        cohort="all",
                        condition=condition,
                        chain=chain,
                        accuracy=value,
                    )
                )
    effects, interactions, missing = summary.organization_rows(rows)
    assert not missing
    old = next(r for r in effects if r["phase"] == "unrestricted")
    assert old["matching_effect"] == pytest.approx(0.2)
    assert old["company_unmatched_benefit"] == pytest.approx(0.1)
    assert old["company_mean_benefit"] == pytest.approx(0.2)
    assert interactions[0]["isolated_minus_unrestricted_matching_effect"] == pytest.approx(-0.2)
    _, _, incomplete = summary.organization_rows(rows[:-1])
    assert incomplete


def test_two_step_rejects_correct_run_with_mismatched_checkpoint_predictions(summary, tmp_path):
    original, path = tmp_path / "original", tmp_path / "two"
    path.mkdir()
    world = make_cross_world(0)
    (path / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "oracle_bridging": False,
                "identity": {
                    "run": str(original),
                    "config_sha256": "conf",
                    "direct_predictions_sha256": "wrong-checkpoint",
                    "answer_sha256": summary.array_hash(world.answers),
                },
            }
        )
    )
    ledger = {
        str(original / "config.json"): "conf",
        str(original / "predictions-15360.npz"): "right-checkpoint",
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        summary.verify_two_step_source(path, original, world, 15360, ledger)


def test_prior_p1_audit_requires_all_cases_and_weight_verification(summary):
    audit = dict(
        state="complete",
        passed_available_predictions=True,
        missing=[],
        complete_models=24,
        complete_new_edits=192,
        complete_reference_edits=96,
        prediction_checkpoints_recomputed=1152,
        weights_and_optimizer_states_verified=192,
    )
    assert summary.prior_audit_complete(audit)
    assert not summary.prior_audit_complete({**audit, "weights_and_optimizer_states_verified": 191})
    assert not summary.prior_audit_complete({"complete": True, "passed": False})
