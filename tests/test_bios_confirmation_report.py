"""Checks for the unfrozen presentation layer; uses fictitious counts only."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report_module():
    path = Path(__file__).resolve().parents[1] / "scripts/report_bios_shortcut_confirmation.py"
    spec = importlib.util.spec_from_file_location("confirmation_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_step_final_and_first_hop_truth_are_separate(report_module):
    tables = report_module.demo_tables()
    checks = report_module.collect_checks(tables)
    final = report_module.points(checks, "learning_original_two_step", "high")
    bridge = report_module.points(checks, "learning_original_bridge_correct", "high")
    assert (final > bridge).all()
    for row in checks:
        if row["metric"] == "learning_direct_original_exception":
            assert row["cases"] == 12 and row["denominator_total"] == 12 * 64


def test_zero_known_retention_is_undefined_and_coverage_is_preserved(report_module):
    tables = report_module.demo_tables()
    for row in tables["editing-all-nodes.csv"]:
        if row["world"] == 100 and row["phase"] == "high" and row["kind"] == "exception":
            row.update(
                U_full_strata_0_known=0,
                U_full_strata_0_coverage=0,
                U_full_strata_0_damage=None,
                U_full_strata_0_broken=0,
            )
    checks = report_module.collect_checks(tables)
    selected = {r["metric"]: r for r in checks if r["world"] == 100 and r["phase"] == "high"}
    damaged = selected["exception_local0_damage"]
    assert damaged["value"] is None and damaged["available_cases"] == 0
    assert damaged["undefined_cases"] == 24 and damaged["known_total"] == 0
    assert damaged["denominator_total"] == 24 * 6
    assert selected["exception_local0_coverage"]["value"] == 0


def test_incomplete_formal_gate_stops_before_table_loading(report_module, monkeypatch, tmp_path):
    summary, statistics = tmp_path / "summary", tmp_path / "statistics"
    summary.mkdir()
    statistics.mkdir()
    (summary / "audit.json").write_text(json.dumps(dict(complete=False)))
    (statistics / "statistics-audit.json").write_text(json.dumps(dict(complete=False)))

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete audit must stop before outcome tables")

    monkeypatch.setattr(report_module, "read_csv", forbidden)
    with pytest.raises(ValueError, match="complete summary/statistics"):
        report_module.load_audited(summary, statistics)


def test_missing_world_is_not_silently_dropped(report_module):
    tables = report_module.demo_tables()
    tables["learning-strata.csv"] = [r for r in tables["learning-strata.csv"] if r["world"] != 107]
    with pytest.raises(ValueError, match="omits worlds/phases"):
        report_module.collect_checks(tables)


def test_organization_contrasts_do_not_conflate_match_and_unmatched(report_module):
    values = {
        ("company", "company"): 0.9,
        ("company", "project"): 0.4,
        ("project", "company"): 0.3,
        ("project", "project"): 0.8,
        ("neither", "company"): 0.5,
        ("neither", "project"): 0.6,
    }
    effects = report_module.org_block(values)
    assert effects["matching_interaction"] == pytest.approx(0.5)
    assert effects["company_matched_vs_neither"] == pytest.approx(0.4)
    assert effects["company_unmatched_vs_neither"] == pytest.approx(-0.2)
    assert effects["company_both_chains_vs_neither"] == pytest.approx(0.1)
    assert effects["project_matched_vs_neither"] == pytest.approx(0.2)
    assert effects["project_unmatched_vs_neither"] == pytest.approx(-0.2)
    assert effects["project_both_chains_vs_neither"] == pytest.approx(0)


def test_organization_all_worlds_and_phase_changes_pair_without_learning_supports(report_module):
    rows = report_module.org_effects(report_module.demo_tables())
    assert len(rows) == 16 * 8 * 3
    for endpoint in {r["endpoint"] for r in rows}:
        for world in report_module.WORLDS:
            values = {
                r["phase"]: r for r in rows if r["endpoint"] == endpoint and r["world"] == world
            }
            assert set(values) == {"low", "high", "high_minus_low"}
            assert {r["nested_blocks"] for r in values.values()} == {
                2 if endpoint.startswith("learning") else 4
            }
            for metric in report_module.ORG_CONTRASTS:
                assert values["high_minus_low"][metric] == pytest.approx(
                    values["high"][metric] - values["low"][metric]
                )


def test_duplicate_or_missing_organization_cells_reject(report_module):
    tables = report_module.demo_tables()
    tables["learning-strata.csv"].append(dict(tables["learning-strata.csv"][0]))
    with pytest.raises(ValueError, match="Duplicate organization cell"):
        report_module.org_effects(tables)
    tables = report_module.demo_tables()
    tables["editing-all-nodes.csv"].pop()
    with pytest.raises(ValueError, match="complete six-cell block"):
        report_module.org_effects(tables)


def test_retention_case_macro_and_pooled_are_distinct(report_module):
    tables = report_module.demo_tables()
    selected = [
        r
        for r in tables["editing-all-nodes.csv"]
        if r["world"] == 100 and r["phase"] == "low" and r["kind"] == "exception"
    ]
    for i, row in enumerate(selected):
        known, broken = (1, 1) if i < 12 else (6, 0)
        row.update(
            U_full_strata_0_known=known,
            U_full_strata_0_broken=broken,
            U_full_strata_0_damage=broken / known,
            U_full_strata_0_coverage=known / 6,
        )
    checks = report_module.collect_checks(tables)
    row = next(
        r
        for r in checks
        if r["world"] == 100 and r["phase"] == "low" and r["metric"] == "exception_local0_damage"
    )
    assert row["value"] == pytest.approx(0.5)
    assert row["pooled_value"] == pytest.approx(1 / 7)
    assert row["numerator_total"] == 12
    assert row["pooled_denominator_total"] == row["known_total"] == 84
    assert row["denominator_total"] == 144


def test_common_u_full_matrix_zero_known_and_counts(report_module):
    rows = report_module.demo_common_u()
    assert len(rows) == 55296
    for row in rows:
        if (
            row["world"] == 100
            and row["kind"] == "exception"
            and row["step"] == 512
            and row["stratum"] == 0
        ):
            row.update(
                known=0,
                low_broken=0,
                high_broken=0,
                low_damage=None,
                high_damage=None,
                high_minus_low_damage=None,
                joint_known_coverage=0,
                joint_known_fraction_of_pool=0,
            )
    checks = report_module.common_u_checks(rows)
    assert len(checks) == 2304
    selected = [
        r
        for r in checks
        if r["world"] == 100 and r["kind"] == "exception" and r["step"] == 512 and r["stratum"] == 0
    ]
    assert len(selected) == 6
    for row in selected:
        assert row["known"] == 0 and row["eligible_n"] > 0
        assert row["low_damage_case_macro"] is None
        assert row["high_damage_pooled"] is None
        assert row["low_damage_valid_cases"] == 0
        assert row["joint_known_coverage_pooled"] == 0
    # A same-length duplicate must not substitute for an omitted world/case.
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate or unexpected"):
        report_module.common_u_checks(rows)


def test_common_u_stored_rates_must_reproduce_counts(report_module):
    rows = report_module.demo_common_u()
    rows[0]["low_damage"] = 0.2
    with pytest.raises(ValueError, match="stored rate disagrees"):
        report_module.common_u_checks(rows)


def test_common_u_requires_matching_verified_audit_and_hashes(report_module, tmp_path):
    primary, common = tmp_path / "primary", tmp_path / "common"
    primary.mkdir()
    common.mkdir()
    (primary / "audit.json").write_text(
        json.dumps(
            dict(archive_index_sha256="a" * 64, lock_sha256="c" * 64, config_sha256="d" * 64)
        )
    )
    (common / "paired-common-u.csv").write_text("world,known\n100,0\n")
    implementation = tmp_path / "synthetic-implementation.py"
    implementation.write_text("# synthetic source identity only\n")
    source_identity = {str(implementation): report_module.sha(implementation)}
    receipt_path = tmp_path / "synthetic-preinspection-receipt.json"
    receipt = dict(
        protocol="confirmation-common-U-supplement-v1",
        status="frozen_before_confirmation_effect_inspection",
        frozen_at_utc="2026-09-27T00:00:00+00:00",
        config_sha256="d" * 64,
        lock_sha256="c" * 64,
        supplemental_sources=source_identity,
        checkpoints=[0, 32, 128, 512],
        controls=list(report_module.COMMON_CONTROLS),
        pools=list(report_module.COMMON_POOLS),
    )
    receipt_path.write_text(json.dumps(receipt))
    (common / "sources.json").write_text(
        json.dumps({str(receipt_path): report_module.sha(receipt_path)})
    )
    audit = dict(
        complete=True,
        weight_archive_verified=True,
        protocol="confirmation-common-U-supplement-v1",
        parent_models=96,
        edit_cases=768,
        paired_cases=384,
        paired_checkpoints=1536,
        rows=55296,
        declared_weights=2208,
        lock_sha256="c" * 64,
        primary_audit_sha256=report_module.sha(primary / "audit.json"),
        archive_index_sha256="a" * 64,
        receipt_sha256=report_module.sha(receipt_path),
        sources_sha256=report_module.sha(common / "sources.json"),
        supplementary_sources=source_identity,
        outputs_sha256={"paired-common-u.csv": report_module.sha(common / "paired-common-u.csv")},
    )
    (common / "audit.json").write_text(json.dumps(audit))
    assert report_module.load_common_u(common, primary, {}) == [{"world": "100", "known": "0"}]
    # A syntactically valid but unbound digest is insufficient.
    audit["receipt_sha256"] = "b" * 64
    (common / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="receipt is absent, ambiguous, or changed"):
        report_module.load_common_u(common, primary, {})
    audit["receipt_sha256"] = report_module.sha(receipt_path)
    (common / "audit.json").write_text(json.dumps(audit))
    receipt_path.write_text(json.dumps(receipt) + " ")
    with pytest.raises(ValueError, match="receipt is absent, ambiguous, or changed"):
        report_module.load_common_u(common, primary, {})
    receipt_path.write_text(json.dumps(receipt))
    (common / "paired-common-u.csv").write_text("world,known\n100,1\n")
    with pytest.raises(ValueError, match="outcome table changed"):
        report_module.load_common_u(common, primary, {})
    audit["weight_archive_verified"] = False
    (common / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="complete verified audit"):
        report_module.load_common_u(common, primary, {})
