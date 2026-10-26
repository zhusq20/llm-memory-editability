"""Contracts for answer truth, sequence dependency and evaluation isolation."""

import copy
import json

import numpy as np
import pytest

from llm_memory_editability import dependency_data as data


@pytest.fixture(scope="module", params=("listops", "arithmetic"))
def dataset(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("dependency") / request.param
    data.prepare(
        {
            "task": request.param,
            "train_samples": 20,
            "evaluation_samples": 6,
            "training_panel_size": 8,
            "seed": 71,
        },
        root,
    )
    return data.load(root)


@pytest.mark.parametrize(
    "task, expression, answer, depth, tree_depth, imbalance, branches",
    [
        ("listops", "[SM [MIN 7 9 ] [MAX 2 3 ] ]", 0, 3, 3, 0, 1),
        ("listops", "[MIN 0 [SM [SM 1 2 ] 3 ] ]", 0, 2, 4, 2, 0),
        ("listops", "[MED 1 2 8 9 ]", 5, 2, 2, 0, 0),
        ("arithmetic", "((4*7)+2)", 0, 3, 3, 1, 0),
        ("arithmetic", "((2+3)*(4+5))", 5, 3, 3, 0, 1),
        ("arithmetic", "(3*(2+(1*4)))", 8, 4, 4, 2, 0),
    ],
)
def test_independent_interpreter(task, expression, answer, depth, tree_depth, imbalance, branches):
    result = data.analyze_expression(expression, task)
    assert (
        result.answer,
        result.depth,
        result.tree_depth,
        result.branch_imbalance,
        result.branch_nodes,
    ) == (answer, depth, tree_depth, imbalance, branches)


@pytest.mark.parametrize(
    "task,expression",
    [
        ("listops", "[SM 1 ]"),
        ("listops", "[SM 1 2"),
        ("listops", "[SM 1 2 ] 3"),
        ("arithmetic", "(1-2)"),
        ("arithmetic", "(1+2"),
        ("arithmetic", "(1+2)3"),
        ("arithmetic", "3"),
    ],
)
def test_parser_rejects_malformed_expression(task, expression):
    with pytest.raises(ValueError):
        data.analyze_expression(expression, task)


def test_truth_depths_and_splits(dataset):
    world, panels, metadata = dataset
    audit = data.audit_data(world, panels, metadata)
    assert audit["passed"]
    assert set(panels) == set(data.PANELS)
    assert metadata["sequence_length"] == 53
    assert np.all((world["depth"] >= 2) & (world["depth"] <= 6))
    assert set(panels["valid_depth"]["depth"].tolist()) == {7}
    assert set(panels["test_depth"]["depth"].tolist()) == {8, 9}
    assert metadata["structural_metadata_used_as_model_input"] is False
    assert metadata["training_examples"] == (20 if metadata["task"] == "listops" else 220)
    if metadata["task"] == "arithmetic":
        assert np.all(world["author_depth"][:200] == 1)
        assert np.all(world["depth"][:200] == 2)
        assert np.all(world["tree_depth"][:200] == 2)
        assert np.all(panels["valid_iid"]["depth"] >= 3)


def test_only_final_answer_and_eos_are_targets(dataset):
    world, panels, metadata = dataset
    for rows in (world, *panels.values()):
        assert rows["labels"].shape[1] == 2
        assert np.all(rows["labels"][:, 1] == metadata["end_marker"])
        assert np.all(rows["positions"][:, 0] == rows["length"] - 2)
        assert np.all(rows["positions"][:, 1] == rows["length"] - 1)
        assert np.all(rows["length"] == rows["expression_length"] + 3)
        assert np.all(rows["author_length"] <= 50)
        for tokens, positions, labels in zip(
            rows["tokens"], rows["positions"], rows["labels"], strict=True
        ):
            assert tokens[positions[0]] == metadata["separator_token"]
            assert tokens[positions[1]] == labels[0]
            assert metadata["end_marker"] not in tokens


def test_listops_author_length_preserves_original_filter():
    result = data.analyze_expression("[SM [MIN 7 9 ] [MAX 2 3 ] ]", "listops")
    assert result.author_length == 13
    assert len("[SM [MIN 7 9 ] [MAX 2 3 ] ]".split()) == 10


def test_detects_heldout_overlap(dataset):
    world, panels, metadata = copy.deepcopy(dataset)
    for field in data.FIELDS:
        panels["test_iid"][field][0] = panels["valid_iid"][field][0]
    with pytest.raises(ValueError, match="leakage"):
        data.audit_data(world, panels, metadata)


def test_detects_corrupt_truth_and_padding(dataset):
    world, panels, metadata = copy.deepcopy(dataset)
    position = world["positions"][0, 1]
    world["labels"][0, 0] = world["tokens"][0, position] = (
        4 + (int(world["labels"][0, 0]) - 4 + 1) % 10
    )
    with pytest.raises(ValueError, match="truth"):
        data.audit_data(world, panels, metadata)


def test_cpu_training_table_has_only_three_inputs_and_masked_dummy(dataset):
    world, _, metadata = dataset
    table = data.make_table(world, metadata, "cpu")
    assert len(table) == 3
    assert tuple(table[0].shape) == (metadata["training_examples"] + 1, 53)
    assert np.all(table[2][-1].numpy() == -100)
    assert tuple(table[1][-1].tolist()) == (0, 1)
    assert np.all(table[0][-1].numpy() == metadata["pad_token"])


def test_exposure_counts_partial_batch_correctly(dataset):
    world, _, metadata = dataset
    batch = 7
    count = len(world["ids"])
    steps = (count + batch - 1) // batch
    result = data.exposure_summary(np.ones(count, dtype=np.int64), world, metadata, steps, batch)
    assert result["examples"] == count
    assert result["masked_batch_slots"] == steps * batch - count
    assert result["effective_input_tokens"] == int(world["length"].sum())
    assert result["supervised_tokens"] == 2 * count


def test_worker_count_does_not_change_data(tmp_path):
    spec = {"task": "arithmetic", "train_samples": 12, "evaluation_samples": 4, "seed": 123}
    for workers in (1, 2):
        data.prepare(spec, tmp_path / str(workers), workers=workers)
    first, second = [data.load(tmp_path / str(workers)) for workers in (1, 2)]
    for key in data.FIELDS:
        np.testing.assert_array_equal(first[0][key], second[0][key])
        for split in data.PANELS:
            np.testing.assert_array_equal(first[1][split][key], second[1][split][key])
    assert first[2]["world_sha256"] == second[2]["world_sha256"]
    assert first[2]["panels_sha256"] == second[2]["panels_sha256"]


def test_file_hashes_enforce_frozen_data(tmp_path):
    root = tmp_path / "data"
    data.prepare({"task": "arithmetic", "train_samples": 4, "evaluation_samples": 2}, root)
    manifest = json.loads((root / "complete.json").read_text())
    assert manifest["source_commit"] == data.SOURCE_COMMIT
    (root / "audit.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load(root)
