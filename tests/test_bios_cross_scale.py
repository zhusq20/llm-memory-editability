"""Guard scientific comparability and retention coverage across model sizes."""

from copy import deepcopy

import numpy as np
import pytest

from llm_memory_editability.bios_cross_scale import (
    retention_coverage,
    validate_run_grid,
    validate_size_study,
)


@pytest.mark.parametrize("field,value", [("lr", 0.0003), ("steps", 30720), ("layers", 4)])
def test_size_reuse_rejects_training_protocol_changes(field, value):
    reference = {"width": 768, "heads": 12, "layers": 8, "lr": 0.0001, "steps": 15360}
    study = {**reference, "width": 64, "heads": 1}
    validate_size_study(reference, study, 64, 1)
    study[field] = value
    with pytest.raises(ValueError, match="Non-size"):
        validate_size_study(reference, study, 64, 1)


def test_duplicate_units_cannot_hide_missing_seed():
    study = {"worlds": [0], "seeds": [0, 1], "conditions": ["company"]}
    run = {"world": 0, "seed": 0, "condition": "company"}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_run_grid([run, deepcopy(run)], study, require_complete=True)
    with pytest.raises(ValueError, match="Missing"):
        validate_run_grid([run], study, require_complete=True)
    validate_run_grid([run], study, require_complete=False)
    validate_run_grid([run, {**run, "seed": 1}], study, require_complete=True)


def test_unknown_knowledge_is_not_counted_as_retained():
    pair = {
        "strata": np.array([0, 0, 1, 1, 2, 2, 3, 3, -1, -1]),
        "heldout": np.array([1, 3, 5, 7]),
        "E": np.array([8]),
        "D": np.array([9]),
    }
    old_correct = np.array([True, False] * 5)
    row = {}
    for group in range(4):
        row[f"U_full_{group}_known"] = 1
        row[f"U_full_{group}_broken"] = int(group == 0)
        row[f"U_heldout_{group}_known"] = 0
        row[f"U_heldout_{group}_broken"] = 0
    metrics = retention_coverage(pair, old_correct, row)
    assert metrics["U_full_total"] == 8
    assert metrics["U_full_coverage"] == 0.5
    assert metrics["U_full_micro_damage"] == 0.25
    assert metrics["U_heldout_micro_damage"] is None
    assert metrics["U_heldout_coverage"] == 0
    assert metrics["E_old_accuracy"] == 1
    assert metrics["D_old_accuracy"] == 0
    row["U_full_0_known"] = 2
    with pytest.raises(ValueError, match="denominator"):
        retention_coverage(pair, old_correct, row)
