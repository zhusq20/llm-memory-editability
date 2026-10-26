"""Parameter geometry and age indexing for the read-only post-hoc inspection."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "forgetting_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_ce_cross_sample_forgetting.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_boundary_flip_and_projection_use_previous_saved_state():
    base = {"x": torch.zeros(4)}
    old = {"x": torch.tensor([1.0, -1.0, 0.5, -0.5])}
    new = {"x": torch.tensor([-1.0, 1.0, 0.25, -0.25])}
    result = analysis.parameter_metrics(base, old, new, 1.0)
    assert result["opposite_boundary_elements"] == 2
    assert result["opposite_boundary_fraction_previous_boundary"] == 1
    assert result["previous_offset_projection_retained"] == pytest.approx(-0.7)
    assert result["fact_update_l2"] == pytest.approx(math.sqrt(8.125))
    assert result["maximum_fact_update"] == 2
    assert torch.equal(old["x"], torch.tensor([1.0, -1.0, 0.5, -0.5]))
    assert torch.equal(new["x"], torch.tensor([-1.0, 1.0, 0.25, -0.25]))


def test_first_write_has_no_previous_direction():
    base = {"x": torch.zeros(2)}
    result = analysis.parameter_metrics(base, base, {"x": torch.tensor([0.25, -0.25])}, 1)
    assert result["previous_offset_cosine"] is None
    assert result["previous_offset_projection_retained"] is None
    assert result["opposite_boundary_fraction_previous_boundary"] is None


def test_fully_cancelled_update_has_zero_retained_projection():
    base = {"x": torch.zeros(2)}
    old = {"x": torch.ones(2)}
    result = analysis.parameter_metrics(base, old, base, 1)
    assert result["previous_offset_projection_retained"] == 0
    assert result["previous_offset_cosine"] is None
    assert result["update_previous_offset_cosine"] == pytest.approx(-1)


def test_original_anchor_bound_is_checked():
    with pytest.raises(ValueError, match="bound was violated"):
        analysis.parameter_metrics(
            {"x": torch.tensor([10.0])},
            {"x": torch.tensor([10.5])},
            {"x": torch.tensor([11.1])},
            1,
        )


def test_silent_dtype_change_is_rejected():
    with pytest.raises(ValueError, match="FP32"):
        analysis.parameter_metrics(
            {"x": torch.zeros(2)}, {"x": torch.zeros(2)}, {"x": torch.zeros(2).double()}, 1
        )


def test_age_retention_and_token_offset_are_not_confused():
    ids = [f"f{i}" for i in range(64)]
    panel = {
        "pairs": [
            {
                "split": "evaluation",
                "A": {"fact_id": ids[2 * i]},
                "B": {"fact_id": ids[2 * i + 1]},
            }
            for i in range(32)
        ],
        "streams": {"forward": ids, "reverse": ids[::-1]},
    }
    records = {}
    for stream, order in panel["streams"].items():
        rows = []
        for step, fact in enumerate(order, 1):
            index = ids.index(fact)
            rows.append(
                {
                    "written": order[:step],
                    "readout": {
                        "atoms": {
                            "expected": list(range(1000, 1064)),
                            "answer": [1000 + index] * 64,
                            "final_correct": [i == index for i in range(64)],
                            "answer_correct": [i == index for i in range(64)],
                        }
                    },
                    "write_info": {
                        "constraint_saturated_elements": 1,
                        "trainable_parameters": 2,
                        "history": [
                            {"step": 1, "new_ce_before": 10},
                            {"step": 2, "new_ce_before": 0},
                        ],
                    },
                }
            )
        records[stream] = rows
    result = analysis.behavioral_summary(records, panel)
    ages = {row["age"]: row for row in result["age_retention"]}
    assert ages[0]["correct"] == ages[0]["n"] == 128
    assert ages[1]["correct"] == 0 and ages[1]["n"] == 126
    assert all(row["previously_correct_lost"] == 1 for row in result["answer_replacements"])
    assert all(row["lost_now_current_tail"] == 1 for row in result["answer_replacements"])
