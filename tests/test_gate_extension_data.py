"""Contracts that distinguish replication, composition transfer and unseen hops."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability import gate_extension_data as data
from llm_memory_editability import loop_hop_data as original


@pytest.fixture
def spec():
    return {
        "data": {
            "kind": "depth_extrapolation",
            "world_seed": 41,
            "entities_per_graph": 8,
            "relations": 3,
            "train_compositions_per_hop": 12,
            "evaluation_templates": 2,
            "training_panel_size": 7,
        }
    }


@pytest.fixture
def task(tmp_path, spec):
    data.prepare(spec, tmp_path / "task")
    return data.load(tmp_path / "task")


def test_confirmation_exactly_reuses_original_generator(tmp_path, spec):
    spec["data"]["kind"] = "permutation_confirmation"
    data.prepare(spec, tmp_path / "adapted")
    original.prepare(spec, tmp_path / "original")
    for name in ("world.npz", "panels.npz"):
        assert (tmp_path / "adapted" / name).read_bytes() == (
            tmp_path / "original" / name
        ).read_bytes()
    world, panels, metadata = data.load(tmp_path / "adapted")
    assert set(panels) == set(original.panel_names())
    assert metadata["training_groups"] == ["atoms", "train_2", "train_3", "train_4"]
    assert data.audit_data(world, panels, metadata)["passed"]
    counts = np.ones(metadata["training_examples"], dtype=np.int64)
    expected = original.exposure_summary(counts, world, metadata, 7, 13)
    assert data.exposure_summary(counts, world, metadata, 7, 13) == expected


def test_depth_has_no_long_chain_training_and_shared_balanced_tests(task):
    world, panels, metadata = task
    assert set(world) == {"atoms", "permutations", "train_2"}
    assert list(panels) == list(data.DEPTH_PANELS)
    assert metadata["training_examples"] == 60
    assert metadata["training_hops"] == [1, 2]
    assert metadata["sequence_length"] == 6
    assert metadata["required_position_encoding"] == "none"
    assert set(map(tuple, panels["id_2"][:, :-1])).isdisjoint(map(tuple, world["train_2"][:, :-1]))
    for hop in (2, 3, 4):
        np.testing.assert_array_equal(panels[f"id_{hop}"][:, 1:-1], panels[f"ood_{hop}"][:, 1:-1])
        assert panels[f"id_{hop}"].shape == (16, hop + 2)
    audit = data.audit_data(world, panels, metadata)
    assert audit["three_or_four_hop_training_examples"] == 0
    assert audit["conditional_evaluation_requires_all_atoms"] == 48


@pytest.mark.parametrize("hop", [1, 2, 3, 4])
def test_objective_targets_only_answer_and_eos_at_every_length(task, hop):
    world, panels, metadata = task
    rows = world["atoms"] if hop == 1 else panels[f"ood_{hop}"]
    x, positions, labels = data.pack_rows(rows, metadata)
    assert x.shape == (len(rows), 6)
    np.testing.assert_array_equal(positions, np.tile([hop, hop + 1], (len(rows), 1)))
    np.testing.assert_array_equal(labels[:, 0], rows[:, -1] + metadata["entity_offset"])
    assert np.all(labels[:, 1] == metadata["end_marker"])
    assert np.all(x[:, hop + 2 :] == metadata["pad_token"])


def test_all_premise_indices_cover_actual_atomic_facts(task):
    world, panels, metadata = task
    for rows in panels.values():
        nodes, edges = data.path_details(rows, metadata, world)
        np.testing.assert_array_equal(world["atoms"][edges, 0], nodes[:, :-1])
        np.testing.assert_array_equal(world["atoms"][edges, 1], rows[:, 1:-1])
        np.testing.assert_array_equal(world["atoms"][edges, 2], nodes[:, 1:])
        np.testing.assert_array_equal(data.premise_edges(rows, metadata, world), edges)


def test_rejects_long_hop_training_contamination(task):
    world, panels, metadata = copy.deepcopy(task)
    world["train_3"] = panels["id_3"][:1]
    with pytest.raises(ValueError, match="only atoms and two-hop"):
        data.audit_data(world, panels, metadata)


def test_rejects_heldout_two_hop_leak(task):
    world, panels, metadata = copy.deepcopy(task)
    world["train_2"][0] = panels["id_2"][0]
    with pytest.raises(ValueError, match="leaked"):
        data.audit_data(world, panels, metadata)


def test_rejects_invalid_truth_and_nonpermutations(task):
    world, panels, metadata = copy.deepcopy(task)
    panels["ood_3"][0, -1] ^= 1
    with pytest.raises(ValueError):
        data.audit_data(world, panels, metadata)
    world, panels, metadata = copy.deepcopy(task)
    world["permutations"][0, 0, 0] = world["permutations"][0, 0, 1]
    with pytest.raises(ValueError, match="not a permutation"):
        data.audit_data(world, panels, metadata)


def test_exposure_counts_include_partial_epoch_and_only_trained_hops(task):
    world, _, metadata = task
    counts = np.ones(60, dtype=np.int64)
    summary = data.exposure_summary(counts, world, metadata, steps=5, batch_size=13)
    assert summary["counts"] == {"atoms": 48, "train_2": 12}
    assert summary["effective_input_tokens"] == 48 * 3 + 12 * 4
    assert summary["supervised_tokens"] == 120
    assert summary["executed_input_tokens"] == 390
    assert summary["masked_batch_slots"] == 5
    counts[:13] += 1
    assert data.exposure_summary(counts, world, metadata, 6, 13)["examples"] == 73
    counts[13] += 1
    with pytest.raises(AssertionError, match="Exposure does not match"):
        data.exposure_summary(counts, world, metadata, 6, 13)


def test_training_table_excludes_longer_hops_and_masks_padding(task):
    torch = pytest.importorskip("torch")
    world, _, metadata = task
    x, positions, labels = data.make_table(world, metadata, "cpu")
    assert x.shape == (61, 6)
    assert torch.all(labels[-1] == -100)
    assert torch.all(positions[:-1, 0] <= 2)
    assert torch.all(x[:-1, 4:] == metadata["pad_token"])


def test_reloading_detects_changed_bytes_and_preparation_never_overwrites(tmp_path, spec):
    destination = tmp_path / "task"
    data.prepare(spec, destination)
    with pytest.raises(FileExistsError):
        data.prepare(spec, destination)
    with (destination / "world.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load(destination)


@pytest.fixture
def archived_wang(tmp_path):
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path("/inputs/wang-original"),
        root / "data/discoloop-success-restore-v1/original/data",
    ]
    source = next((path for path in candidates if (path / "complete.json").exists()), None)
    if source is None:
        pytest.skip("Original Wang data must be mounted for archived-data integration checks")
    config = {"data": {"kind": "wang", "source_directory": str(source), "phi": 7.2}}
    destination = tmp_path / "wang"
    data.prepare(config, destination)
    return source, destination, data.load(destination)


def test_wang_preserves_original_data_support_vocab_scoring_and_pool_order(archived_wang):
    source, destination, (world, panels, metadata) = archived_wang
    for name in ("world.npz", "panels.npz"):
        assert (source / name).read_bytes() == (destination / name).read_bytes()
    assert metadata["vocab_size"] == 52463
    assert metadata["primary_score"] == "accuracy"
    assert metadata["sequence_length"] == 4
    assert metadata["training_examples"] == 313600
    assert len(world["atoms"]) == 40000
    assert len(world["train_2"]) == 273600
    assert list(panels) == list(data.WANG_PANELS)
    assert metadata["panels"]["test_oo"] == 2004
    np.testing.assert_array_equal(world["train_2"], world["chains"][world["train_order"][:273600]])
    audit = json.loads((destination / "audit.json").read_text())
    assert audit["conditional_evaluation_requires_all_atoms"] == 40000
    assert metadata["panels"]["atomic_id"] + metadata["panels"]["atomic_ood"] < 40000
    for name, kind in (("test_ii", 0), ("test_io", 1), ("test_oi", 2), ("test_oo", 3)):
        edges = data.premise_edges(panels[name], metadata, world)
        assert np.all(world["ood"][edges[:, 0]] * 2 + world["ood"][edges[:, 1]] == kind)


def test_wang_objective_matches_original_token_layout(archived_wang):
    _, _, (world, _, metadata) = archived_wang
    for group, hop in (("atoms", 1), ("train_2", 2)):
        rows = world[group][:10]
        x, positions, labels = data.pack_rows(rows, metadata)
        assert x.shape == (10, 4)
        np.testing.assert_array_equal(x[:, 0], rows[:, 0] + 50257)
        np.testing.assert_array_equal(x[:, 1 : hop + 1], rows[:, 1:-1] + 52257)
        np.testing.assert_array_equal(x[:, hop + 1], rows[:, -1] + 50257)
        assert np.all(positions == [hop, hop + 1])
        assert np.all(labels[:, 1] == 52460)


def test_wang_rejects_changed_training_support(archived_wang):
    _, _, (world, panels, metadata) = archived_wang
    world["train_2"][0] = panels["test_oo"][0]
    with pytest.raises(ValueError, match="training support"):
        data.audit_data(world, panels, metadata)
