"""Check that withheld roles and atomic supervision match the paper's comparison."""

import copy
from pathlib import Path

import pytest

from llm_memory_editability.second_hop_reproduction import prepare, read_json, validate_roles

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "docs/development-artifacts/implicit-reasoning-paper-reproduction-v1"
    / "upstream/notebooks/data_preparation.ipynb"
)


def test_second_hop_pools_and_full_atomic_arm(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    out = tmp_path / "roles"
    audit = prepare(NOTEBOOK, out, entities=50, relations=30, degree=10)
    only = read_json(out / "only_ii/train.json")
    full = read_json(out / "full_atomic/train.json")
    assert len(only) == round(7.2 * 475)
    assert full[:500] == read_json(out / "atomic-truth.json")
    assert full[500:] == only
    assert read_json(out / "only_ii/test.json") == read_json(out / "full_atomic/test.json")
    assert audit["restricted_second_hop_training_queries"] == 0
    assert audit["restricted_facts_used_as_first_hop"] > 0
    assert audit["evaluation_counts"]["Test-II-SR"] > 0
    assert audit["evaluation_counts"]["Test-II"] > 0


def test_restricted_second_hop_cannot_enter_training():
    atomics = [
        {"input_text": "<e_0><r_0>", "target_text": "<e_0><r_0><e_1></a>"},
        {"input_text": "<e_1><r_1>", "target_text": "<e_1><r_1><e_2></a>"},
    ]
    row = {"input_text": "<e_0><r_0><r_1>", "target_text": "<e_0><r_0><r_1><e_2></a>"}
    restriction = {("<e_1>", "<r_1>")}
    assert validate_roles(atomics, [], restriction, [], {"Test-II-SR": [row]})["passed"]
    with pytest.raises(AssertionError):
        validate_roles(atomics, [], restriction, [copy.deepcopy(row)], {})
