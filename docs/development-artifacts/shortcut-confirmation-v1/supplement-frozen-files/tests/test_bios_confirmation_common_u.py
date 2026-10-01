"""Supplement tests use tiny arrays / existing development world0 only."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "confirmation_common_u_test", scripts / "summarize_bios_confirmation_common_u.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def arrays(prediction, truth, ended=None):
    prediction = np.asarray(prediction).copy()
    ended = np.ones(len(truth), dtype=bool) if ended is None else np.asarray(ended)
    return {
        "prediction": prediction,
        "ended": ended,
        "correct": (prediction == truth) & ended,
    }


@pytest.fixture
def synthetic_case():
    low = np.arange(10, dtype=np.int64)
    high = low.copy()
    high[5] = 55  # Both know a fact whose original truth differs by prevalence.
    truths = {"low": low, "high": high}
    baseline, sets, predictions = {}, {}, {}
    for phase, truth in truths.items():
        old = truth.copy()
        old[8 if phase == "low" else 9] = -1
        baseline[phase] = arrays(old, truth)
        target = truth.copy()
        target[:2] += 100
        sets[phase] = {
            "E": np.array([0]),
            "D": np.array([1]),
            "R": np.array([2]),
            "U_full": np.arange(2, 10),
            "U_heldout": np.array([3, 4, 5, 6]),
            "U_strata": np.array([-1, -1, 3, 0, 0, 1, 2, 4, 4, 4]),
            f"{phase}_exception": target,
            "old_correct": baseline[phase]["correct"].copy(),
        }
        now = target.copy()
        now[[2, 3, 5] if phase == "low" else [4, 7]] = -1
        ended = np.ones(10, dtype=bool)
        if phase == "high":
            ended[6] = False  # Correct value without EOS must be broken.
        predictions[phase] = arrays(now, target, ended)
    identity = dict(
        world=0, seed=0, condition="company", chain="company", support=0, kind="exception", step=512
    )
    return identity, truths, sets, baseline, predictions


def select(rows, control="same_truth_common_known", pool="U_full", stratum=-1):
    return next(
        r for r in rows if (r["control"], r["pool"], r["stratum"]) == (control, pool, stratum)
    )


def test_joint_baseline_truth_eos_and_unseen_are_separate_contracts(module, synthetic_case):
    rows = module.case_rows(*synthetic_case)
    assert len(rows) == 36
    row = select(rows)
    assert (row["pool_n"], row["eligible_n"], row["known"]) == (8, 7, 5)
    assert (row["low_baseline_known"], row["high_baseline_known"]) == (6, 6)
    assert (row["low_broken"], row["high_broken"]) == (2, 3)
    assert row["low_damage"] == 2 / 5 and row["high_damage"] == 3 / 5
    assert row["joint_known_coverage"] == 5 / 7
    assert row["joint_known_fraction_of_pool"] == 5 / 8
    differing = select(rows, control="common_known")
    assert (differing["known"], differing["low_broken"], differing["high_broken"]) == (6, 3, 3)
    unseen = select(rows, pool="U_unseen")
    assert (unseen["known"], unseen["low_broken"], unseen["high_broken"]) == (4, 1, 3)
    local = select(rows, pool="U_unseen", stratum=0)
    assert (local["eligible_n"], local["known"], local["low_broken"], local["high_broken"]) == (
        2,
        2,
        1,
        1,
    )
    empty = select(rows, stratum=1)
    assert empty["pool_n"] == 1 and empty["eligible_n"] == empty["known"] == 0
    assert empty["low_damage"] is empty["high_damage"] is None


@pytest.mark.parametrize(
    "mutation",
    ["changed_u_truth", "wrong_old_correct", "incorrect_score", "duplicate_u", "replay_in_heldout"],
)
def test_invalid_support_or_scoring_rejected(module, synthetic_case, mutation):
    identity, truths, sets, baseline, predictions = copy.deepcopy(synthetic_case)
    if mutation == "changed_u_truth":
        sets["low"]["low_exception"][3] = 12345
    elif mutation == "wrong_old_correct":
        sets["high"]["old_correct"][9] = True
    elif mutation == "incorrect_score":
        predictions["high"]["correct"][6] = True
    elif mutation == "duplicate_u":
        sets["low"]["U_full"][0] = 3
    else:
        sets["low"]["R"] = np.array([3])
    with pytest.raises((ValueError, AssertionError)):
        module.case_rows(identity, truths, sets, baseline, predictions)


def test_macro_pooled_and_zero_denominator_are_not_conflated(module, synthetic_case):
    base = select(module.case_rows(*synthetic_case))
    one = {
        **base,
        "known": 1,
        "low_broken": 1,
        "high_broken": 0,
        "low_damage": 1.0,
        "high_damage": 0.0,
        "high_minus_low_damage": -1.0,
    }
    many = {
        **base,
        "seed": 1,
        "known": 9,
        "low_broken": 0,
        "high_broken": 3,
        "low_damage": 0.0,
        "high_damage": 1 / 3,
        "high_minus_low_damage": 1 / 3,
    }
    empty = {
        **base,
        "support": 1,
        "known": 0,
        "low_broken": 0,
        "high_broken": 0,
        "low_damage": None,
        "high_damage": None,
        "high_minus_low_damage": None,
    }
    row = module.aggregate_rows([one, many, empty], True)[0]
    assert row["paired_cases"] == 3 and row["low_damage_valid_cases"] == 2
    assert row["low_damage_case_macro"] == 0.5
    assert row["low_damage_pooled"] == 0.1
    assert row["high_damage_case_macro"] == 1 / 6
    assert row["high_damage_pooled"] == 0.3


def test_nonempty_pool_without_joint_knowledge_is_undefined(module, synthetic_case):
    identity, truths, sets, baseline, predictions = copy.deepcopy(synthetic_case)
    baseline["low"]["prediction"][3:5] = -1
    baseline["low"]["correct"][3:5] = False
    sets["low"]["old_correct"][3:5] = False
    row = select(module.case_rows(identity, truths, sets, baseline, predictions), stratum=0)
    assert row["eligible_n"] == 2 and row["known"] == 0
    assert row["joint_known_coverage"] == 0
    for field in ("low_damage", "high_damage", "high_minus_low_damage"):
        assert row[field] is None
    aggregate = module.aggregate_rows([row], True)[0]
    assert aggregate["paired_cases"] == 1 and aggregate["low_damage_valid_cases"] == 0
    assert aggregate["low_damage_case_macro"] is aggregate["low_damage_pooled"] is None


def test_unseen_excludes_replay_from_either_phase(module, synthetic_case):
    identity, truths, sets, baseline, predictions = copy.deepcopy(synthetic_case)
    sets["high"]["R"] = np.array([7])
    row = select(module.case_rows(identity, truths, sets, baseline, predictions), pool="U_unseen")
    assert row["known"] == 3  # IDs 3, 4, 6; both R={2,7}, differing truth5 excluded.
    assert (row["low_broken"], row["high_broken"]) == (1, 2)


def test_world_macro_preserves_equal_world_weight_despite_undefined_cases(module, synthetic_case):
    base = select(module.case_rows(*synthetic_case))
    rows = [
        {**base, "world": 0, "low_damage": 0.0},
        {**base, "world": 0, "seed": 1, "low_damage": 0.0},
        {**base, "world": 1, "low_damage": 1.0},
        {**base, "world": 1, "seed": 1, "low_damage": None},
    ]
    row = module.aggregate_rows(rows, False)[0]
    assert row["low_damage_case_macro"] == 1 / 3
    assert row["low_damage_world_macro"] == 0.5
    assert row["low_damage_valid_cases"] == 3 and row["low_damage_valid_worlds"] == 2


def test_development_supports_at_zero_have_no_damage_and_original_actual_truths_match(module):
    low = module.primary.make_cross_world(0)
    high, *_ = module.primary.high_exception_world(low)
    truths = {"low": low.answers, "high": high.answers}
    baselines = {phase: arrays(truth, truth) for phase, truth in truths.items()}
    for chain in (0, 1):
        for support in (0, 1):
            pair = module.primary.make_confirmation_edit_pair(low, high, chain, support)
            for kind in module.primary.KINDS:
                sets = {
                    phase: {**pair, "old_correct": baselines[phase]["correct"]}
                    for phase in module.primary.PHASES
                }
                predictions = {
                    phase: arrays(truths[phase], pair[f"{phase}_{kind}"])
                    for phase in module.primary.PHASES
                }
                identity = dict(
                    world=0,
                    seed=0,
                    condition="company",
                    chain=module.primary.CHAINS[chain],
                    support=support,
                    kind=kind,
                    step=0,
                )
                rows = module.case_rows(identity, truths, sets, baselines, predictions)
                assert all(
                    row[f"{phase}_broken"] == 0 for row in rows for phase in module.primary.PHASES
                )
                assert select(rows, stratum=0)["known"] == 6
                assert select(rows)["known"] < select(rows, control="common_known")["known"]
                assert select(rows, pool="U_unseen")["known"] == select(rows)["known"] - 4096


def test_incomplete_gate_never_materializes_world_or_reads_predictions(
    module, monkeypatch, tmp_path
):
    source = tmp_path / "raw"
    monkeypatch.setattr(
        module.primary, "validate_lock", lambda *args: ({}, {"output_root": str(source)})
    )
    monkeypatch.setattr(module, "validate_receipt", lambda *args: {})

    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete matrix must not read any world, archive, or prediction")

    monkeypatch.setattr(module.primary, "make_cross_world", forbidden)
    monkeypatch.setattr(module.primary, "audit_archive", forbidden)
    monkeypatch.setattr(module.primary.Sources, "arrays", forbidden)
    with pytest.raises(ValueError, match="All 96 completed"):
        module.summarize(
            tmp_path / "config",
            tmp_path / "lock",
            None,
            tmp_path / "summary",
            tmp_path / "archive",
            tmp_path / "receipt",
            tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_receipt_binds_source_and_never_overwrites(module, monkeypatch, tmp_path):
    config, lock, receipt = (
        tmp_path / name for name in ("config.json", "lock.json", "receipt.json")
    )
    config.write_text("{}")
    lock.write_text("{}")
    monkeypatch.setattr(module.primary, "validate_lock", lambda *args: ({}, {}))
    module.freeze_receipt(config, lock, receipt)
    module.validate_receipt(config, lock, receipt, module.primary.Sources())
    with pytest.raises(FileExistsError):
        module.freeze_receipt(config, lock, receipt)
    saved = json.loads(receipt.read_text())
    saved["supplemental_sources"][module.SUPPLEMENT_FILES[0]] = "changed"
    receipt.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="source identity changed"):
        module.validate_receipt(config, lock, receipt, module.primary.Sources())


def test_primary_gate_rejects_incomplete_audit_before_reading_tables(module, tmp_path):
    directory = tmp_path / "summary"
    directory.mkdir()
    (directory / "audit.json").write_text('{"complete": false}')
    paths = [tmp_path / name for name in ("lock", "config", "archive")]
    for path in paths:
        path.write_text("{}")
    with pytest.raises(ValueError, match="Complete frozen primary summary"):
        module.audit_primary_summary(
            directory, paths[0], paths[1], tmp_path / "raw", paths[2], module.primary.Sources()
        )
