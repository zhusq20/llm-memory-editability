"""Reject accidentally loading either adapted endpoint in the bare-base arm."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "foundation_addition", SCRIPTS / "run_ttcd_foundation_novel_facts.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def test_only_exact300k_checkpoint_and_zero_adaptation_are_accepted():
    provenance = {"step": 300000, "checkpoint_sha256": "original"}
    design = {
        "parent": {
            "step": 300000,
            "checkpoint_sha256": "original",
            "common_steps_loaded": 0,
            "writer_steps_loaded": 0,
        }
    }
    experiment.check_foundation(provenance, design)
    with pytest.raises(ValueError, match="300k foundation"):
        experiment.check_foundation({**provenance, "step": 320000}, design)
    with pytest.raises(ValueError, match="differs"):
        experiment.check_foundation({**provenance, "checkpoint_sha256": "adapted"}, design)
    for key in ("common_steps_loaded", "writer_steps_loaded"):
        wrong = {"parent": {**design["parent"], key: 10000}}
        with pytest.raises(ValueError, match="No adaptation checkpoint"):
            experiment.check_foundation(provenance, wrong)
