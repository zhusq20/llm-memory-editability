"""Continuous-only contracts and the fixed-architecture paired ablation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_ce_cross_sample_summary import SETTINGS, fixture, get

from llm_memory_editability.ce_cross_sample_ablation_summary import (
    compare_with_control,
    summarize_streams,
)
from llm_memory_editability.ce_cross_sample_summary import summarize

SPEC = importlib.util.spec_from_file_location(
    "_ablation_report",
    Path(__file__).resolve().parents[1] / "scripts/report_ce_cross_sample_ablation.py",
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def continuous_fixture():
    panel, records = fixture()
    return panel, [r for r in records if r["stream"] != "isolated"]


def test_no_isolated_records_are_needed_or_synthesized():
    panel, records = continuous_fixture()
    result = summarize_streams(records, panel, SETTINGS)
    assert result["records"] == 9
    assert result["primary_endpoint"]["double_new"]["accuracy"] == 1.0
    assert "isolated_double_new" not in result["primary_endpoint"]
    assert "continuous_minus_isolated" not in result
    assert len(result["fact_endpoint_values"]) == 4
    assert json.loads(json.dumps(result)) == result


def test_write_geometry_is_carried_without_reinterpreting_or_mutating_it():
    panel, records = continuous_fixture()
    info = {
        "maximum_absolute_parameter_change": 0.0005,
        "constraint_saturated_elements": 42,
        "delta_cosine_to_previous_write": -0.1,
        "history": [{"step": 20, "gradient_norm_before_clip": 3.0}],
    }
    get(records, "forward-1")["write_info"] = copy.deepcopy(info)
    result = summarize_streams(records, panel, SETTINGS)
    assert result["streams"]["forward"]["trajectory"][0]["write_info"] == info
    assert get(records, "forward-1")["write_info"] == info


def test_recomputed_control_matches_original_primary_age_and_old_statistics():
    panel, full = fixture()
    get(full, "forward-4")["readout"]["double_new"]["final_correct"][0] = False
    original = summarize(full, panel, SETTINGS)
    recomputed = summarize_streams([r for r in full if r["stream"] != "isolated"], panel, SETTINGS)
    for metric in (
        "double_new",
        "both_atoms",
        "both_atoms_and_composition",
        "double_new_answer",
        "just_closed_double_new",
        "composition_given_both_atoms",
    ):
        assert recomputed["primary_endpoint"][metric] == original["primary_endpoint"][metric]
    assert recomputed["age_retention"] == original["age_retention"]
    for stream in ("forward", "reverse"):
        assert (
            recomputed["streams"][stream]["endpoint"]["old"]
            == original["streams"][stream]["endpoint"]["old"]
        )


def test_initially_correct_new_atom_retention_has_pair_ratio_bootstrap():
    panel, records = continuous_fixture()
    get(records, "forward-1")["readout"]["atoms"]["final_correct"][0] = False
    get(records, "forward-4")["readout"]["atoms"]["final_correct"][2] = False
    result = summarize_streams(records, panel, SETTINGS)
    primary = result["primary_endpoint"]
    assert primary["just_written_atoms"]["correct"] == 7
    assert primary["all_written_atoms"]["correct"] == 7
    retained = primary["initially_correct_new_atoms_retained"]
    assert retained["correct"] == 6 and retained["n"] == 7
    assert retained["accuracy"] == 6 / 7
    assert retained["pair_bootstrap"]["mean"] == 6 / 7
    rows = {r["fact_id"]: r for r in result["fact_endpoint_values"]}
    assert not rows["a0"]["orders"]["forward"]["just_written_correct"]
    assert rows["a0"]["orders"]["forward"]["endpoint_correct"]


def test_unclosed_target_hits_do_not_enter_composition_or_age_retention():
    panel, records = continuous_fixture()
    get(records, "forward-1")["readout"]["double_new"]["final_correct"] = [True, True]
    result = summarize_streams(records, panel, SETTINGS)
    first = result["streams"]["forward"]["trajectory"][0]
    assert first["closed_pairs"] == 0
    assert first["final_correct"]["double_new"]["n"] == 0
    assert first["closed_composition_retention"]["originally_correct_retained"]["accuracy"] is None
    assert [(r["age"], r["accuracy"]["n"]) for r in result["age_retention"]["compositions"]] == [
        (0, 4),
        (1, 2),
    ]


def test_common_atomic_prerequisites_keep_zero_na_and_old_denominators_separate():
    panel, control = continuous_fixture()
    variant = copy.deepcopy(control)
    for stream in ("forward", "reverse"):
        get(variant, f"{stream}-4")["readout"]["atoms"]["final_correct"] = [False] * 4
    # Different old baseline coverage is visible; it must not become 100% retention.
    for record in variant:
        record["readout"]["old"]["atomic_id"]["final_correct"] = [False, False]
    result = compare_with_control(control, {"lower-lr": variant}, panel, SETTINGS)
    comparison = result["variants"]["lower-lr"]
    assert not comparison["baseline_old_coverage_identical"]
    common = comparison["common_atomic_prerequisites"]
    assert common["control"]["coverage_n"] == common["variant"]["coverage_n"] == 0
    assert common["variant"]["accuracy"] is None
    assert (
        comparison["common_prerequisite_paired_differences"]["variant_minus_control"]["mean"]
        is None
    )
    old = comparison["old_retention_by_architecture"]["variant"]["forward"]["pools"]
    assert old["atomic_id"]["final_correct"]["originally_correct_retained"]["accuracy"] is None


def test_fixed_variant_minus_control_sign_and_pair_unit():
    panel, control = continuous_fixture()
    variant = copy.deepcopy(control)
    get(variant, "forward-4")["readout"]["double_new"]["final_correct"][0] = False
    result = compare_with_control(control, {"carry-adam": variant}, panel, SETTINGS)
    delta = result["variants"]["carry-adam"]["paired_differences"]["variant_minus_control"]
    assert delta["double_new"]["mean"] == -0.25
    assert delta["double_new"]["ci95_pair_bootstrap"] == [-0.5, 0.0]
    assert delta["double_new"]["pairs"] == 2
    assert delta["all_written_atoms"]["mean"] == 0.0


def test_terminal_two_step_reference_is_preserved_and_pair_clustered():
    panel, records = continuous_fixture()
    for stream in ("forward", "reverse"):
        get(records, f"{stream}-4")["two_step"] = [
            {"pair_id": "p0", "strict_correct": True},
            {"pair_id": "p1", "strict_correct": False},
        ]
    result = summarize_streams(records, panel, SETTINGS)
    assert result["primary_endpoint"]["explicit_two_step_strict_correct"]["accuracy"] == 0.5
    assert len(result["streams"]["forward"]["endpoint"]["explicit_two_step"]) == 2


@pytest.mark.parametrize("failure", ["missing", "isolated", "wrong_order", "duplicate", "old_rows"])
def test_incomplete_or_mismatched_continuous_panel_is_rejected(failure):
    panel, records = continuous_fixture()
    if failure == "missing":
        records.pop()
    elif failure == "isolated":
        records[-1]["stream"] = "isolated"
    elif failure == "wrong_order":
        get(records, "forward-2")["written"].reverse()
    elif failure == "duplicate":
        records[-1]["key"] = records[0]["key"]
    elif failure == "old_rows":
        panel["old_eval_pools"] = {
            name: copy.deepcopy(pool["rows"]) for name, pool in records[0]["readout"]["old"].items()
        }
        panel["old_eval_pools"]["atomic_id"][0][-1] += 1
    with pytest.raises(ValueError):
        summarize_streams(records, panel, SETTINGS)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stage_fixture(tmp_path, valid):
    root, prepare = tmp_path / "arm/output", tmp_path / "prepare/output"
    gate = {
        "passed": True,
        "gate_valid": valid,
        "eligible": valid,
        "scientific_optimizer_updates": 0,
        "development_optimizer_updates": 640,
        "candidates": [{"candidate": 1}],
    }
    save_json(root / "gate/result.json", gate)
    save_json(
        root / "audit-gate/result.json",
        {
            "passed": True,
            "gate_valid": valid,
            "scientific_optimizer_updates": 0,
            "replayed_states": 32,
            "summary_recomputed": True,
            "gate_result_sha256": reporter.sha(root / "gate/result.json"),
        },
    )
    save_json(
        prepare / "prepare/result.json",
        {
            "passed": True,
            "scientific_optimizer_updates": 0,
            "engineering_optimizer_updates": 46,
            "restored_theta0": True,
        },
    )
    stages = ["prepare", "gate", "audit-gate"]
    if valid:
        stages += ["science", "audit"]
        save_json(
            root / "science/result.json",
            {
                "passed": True,
                "gate_valid": True,
                "scientific_optimizer_updates": 2560,
                "gate_result_sha256": reporter.sha(root / "gate/result.json"),
            },
        )
        save_json(
            root / "audit/result.json",
            {
                "passed": True,
                "gate_valid": True,
                "scientific_optimizer_updates": 0,
                "replayed_states": 129,
                "summary_recomputed": True,
                "science_result_sha256": reporter.sha(root / "science/result.json"),
            },
        )
    save_json(
        root / "cloud/result.json",
        {
            "passed": True,
            "gate_valid": valid,
            "canonical_full_json_exact": True,
            "runs": [
                {
                    "stage": stage,
                    "result_sha256": reporter.sha(
                        (prepare if stage == "prepare" else root) / stage / "result.json"
                    ),
                }
                for stage in stages
            ],
        },
    )
    return root, prepare


def test_failed_write_gate_is_an_explicit_unrun_outcome_not_zero(tmp_path):
    root, prepare = stage_fixture(tmp_path, False)
    outcome = reporter.verify_new_arm(root, prepare)
    assert outcome["status"] == "write_gate_failed_science_not_run"
    assert outcome["science"] is None and outcome["audit"] is None
    assert outcome["scientific_optimizer_updates"] == 0
    assert "accuracy" not in outcome
    (root / "science").mkdir()
    with pytest.raises(ValueError, match="must not enter science"):
        reporter.verify_new_arm(root, prepare)


def test_valid_arm_requires_2560_updates_and_129_readonly_replays(tmp_path):
    root, prepare = stage_fixture(tmp_path, True)
    assert reporter.verify_new_arm(root, prepare)["status"] == "complete"
    audit = json.loads((root / "audit/result.json").read_text())
    audit["replayed_states"] = 128
    save_json(root / "audit/result.json", audit)
    with pytest.raises(ValueError, match="129-state"):
        reporter.verify_new_arm(root, prepare)


def test_cloud_must_cover_prepare_and_every_completed_stage_with_exact_hashes(tmp_path):
    root, prepare = stage_fixture(tmp_path, False)
    cloud = json.loads((root / "cloud/result.json").read_text())
    cloud["runs"] = [r for r in cloud["runs"] if r["stage"] != "prepare"]
    save_json(root / "cloud/result.json", cloud)
    with pytest.raises(ValueError, match="exactly the completed stages"):
        reporter.verify_new_arm(root, prepare)


def test_gate_audit_cannot_silently_change_gate_validity(tmp_path):
    root, _ = stage_fixture(tmp_path, False)
    audit = json.loads((root / "audit-gate/result.json").read_text())
    audit["gate_valid"] = True
    save_json(root / "audit-gate/result.json", audit)
    with pytest.raises(ValueError, match="disagrees"):
        reporter.verify_gate(root)
