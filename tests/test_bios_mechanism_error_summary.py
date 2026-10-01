"""Pending P0 summaries cannot silently become evidence over changed source tables."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def error_summary():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_mechanism_errors.py"
    spec = importlib.util.spec_from_file_location("error_summary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_analysis_rejects_missing_source_audit(error_summary, tmp_path):
    with pytest.raises(ValueError, match="complete original source audit"):
        error_summary.checked_source_identity(tmp_path, tmp_path, require_complete=True)


def test_changed_source_csv_cannot_replace_pending_identity(error_summary, tmp_path):
    for name in error_summary.CSV_INPUTS:
        (tmp_path / name).write_text("unchanged frozen table\n")
    audit, hashes = error_summary.checked_source_identity(tmp_path, tmp_path)
    assert not audit["complete"]
    (tmp_path / "h2-analysis.json").write_text(json.dumps({"complete": False, "inputs": hashes}))
    (tmp_path / "learning-strata.csv").write_text("altered table\n")
    with pytest.raises(ValueError, match="Source CSV identity changed"):
        error_summary.checked_source_identity(tmp_path, tmp_path)


def test_incorrect_completed_source_audit_rejected(error_summary, tmp_path):
    (tmp_path / "audit.json").write_text(json.dumps({"complete": True, "models": 47}))
    with pytest.raises(ValueError, match="incomplete or has errors"):
        error_summary.checked_source_identity(tmp_path, tmp_path, require_complete=True)
