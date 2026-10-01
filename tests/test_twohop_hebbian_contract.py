"""Checks for data semantics and train/evaluation information separation."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_literal_values_are_not_collapsed_to_placeholder_id():
    module = load_script("prepare_twohop_hebbian_benchmarks")
    row = dict(
        type="compositional",
        evidences_id=[["Q1", "parent", "Q2"], ["Q2", "birth", "date_information"]],
        evidences=[["a", "parent", "b"], ["b", "birth", "1 January 1900"]],
    )
    first = module.wiki_chain(row)
    row["evidences"][1][2] = "2 January 1900"
    second = module.wiki_chain(row)
    assert first != second
    assert first[0][2] == first[1][0] == "Q2"


def test_disconnected_or_comparison_evidence_is_not_twohop_composition():
    module = load_script("prepare_twohop_hebbian_benchmarks")
    row = dict(
        type="compositional",
        evidences_id=[["Q1", "r", "Q2"], ["Q3", "s", "Q4"]],
        evidences=[["a", "r", "b"], ["c", "s", "d"]],
    )
    assert module.wiki_chain(row) is None
    row["evidences_id"][1][0] = "Q2"
    row["type"] = "comparison"
    assert module.wiki_chain(row) is None


def test_memory_fit_does_not_depend_on_evaluation_chain_labels(tmp_path):
    module = load_script("run_twohop_hebbian_module")
    module.RAW = tmp_path
    facts = [("a", "r", "b"), ("b", "s", "c"), ("a", "s", "d"), ("d", "r", "c")]
    entities, relations = ["a", "b", "c", "d"], ["r", "s"]
    cfg = dict(
        entity_dimension=8, relation_dimension=4, ridge=1e-6, interfaces=["raw", "unit_norm"]
    )
    one = [[facts[0], facts[1]]]
    two = [[facts[2], facts[3]]]
    module.run(cfg, 123, 16, facts, one, entities, relations)
    path = tmp_path / "seed-123-width-16-memory.npz"
    with np.load(path) as checkpoint:
        first = {name: checkpoint[name].copy() for name in checkpoint.files}
    module.run(cfg, 123, 16, facts, two, entities, relations)
    with np.load(path) as checkpoint:
        for name, value in first.items():
            np.testing.assert_array_equal(value, checkpoint[name])
