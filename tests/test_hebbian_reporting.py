import importlib.util

import numpy as np
import pytest

from llm_memory_editability.hebbian_learning import ROOT


def report_module():
    pytest.importorskip("matplotlib")
    path = ROOT / "scripts/report_hebbian_learning.py"
    spec = importlib.util.spec_from_file_location("hebbian_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_entity_sensitivity_keeps_episode_weights(monkeypatch):
    report = report_module()
    audit = {
        "pools": {
            "C_eval": {
                "cross_episode_subjects": {
                    "a": [0, 6],
                    "b": [1, 7],
                    "c": [2, 4],
                }
            }
        }
    }
    monkeypatch.setattr(report, "read_json", lambda path: audit)
    result = report.subject_component_sensitivity(np.ones(8), "C_eval")
    assert result["components"] == [[0, 6], [1, 7], [2, 4], [3], [5]]
    assert result["component_sign_p"] == 2 / 32
    assert result["ci95"] == [1.0, 1.0]
    uneven = report.subject_component_sensitivity(np.arange(8), "C_eval")
    assert uneven["mean"] == 3.5
