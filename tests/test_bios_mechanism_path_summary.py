"""The world-1 gate must fail closed if candidate discovery changes."""

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def summary_module():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_mechanism_paths.py"
    spec = importlib.util.spec_from_file_location("mechanism_summary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replication_cannot_open_world_one_without_lock(summary_module, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("World 1 should not be accessed without a valid lock")

    monkeypatch.setattr(summary_module, "causal_effect_rows", forbidden)
    with pytest.raises(FileNotFoundError):
        summary_module.replication(tmp_path, tmp_path / "output", tmp_path / "absent.json")


def test_changed_discovery_or_raw_source_prevents_world_one_access(
    summary_module, monkeypatch, tmp_path
):
    discovery = tmp_path / "discovery"
    discovery.mkdir()
    source = tmp_path / "raw-node.npz"
    source.write_bytes(b"frozen world zero diagnostic")
    (discovery / "world-0-audit.json").write_text(
        json.dumps(
            {
                "complete": True,
                "worlds_read": [0],
                "primary_positions": [2, 3],
                "primary_known_only": False,
            }
        )
    )
    (discovery / "world-0-sources.json").write_text(
        json.dumps({str(source): summary_module.file_sha256(source)})
    )
    ranks = discovery / "world-0-candidate-ranks.csv"
    ranks.write_text("mechanism,layer,position,score\nactual_path,3,3,0.2\n")
    lock = tmp_path / "candidate.json"
    summary_module.lock_candidate(discovery, lock, "actual_path", 3, 3, "world zero candidate")

    def forbidden(*args, **kwargs):
        raise AssertionError("World 1 should not be read after source tampering")

    monkeypatch.setattr(summary_module, "causal_effect_rows", forbidden)
    source.write_bytes(b"changed world zero diagnostic")
    with pytest.raises(ValueError, match="source changed"):
        summary_module.replication(tmp_path, tmp_path / "output", lock)
    source.write_bytes(b"frozen world zero diagnostic")
    ranks.write_text("mechanism,layer,position,score\nactual_path,3,3,0.9\n")
    with pytest.raises(ValueError, match="discovery changed"):
        summary_module.replication(tmp_path, tmp_path / "output", lock)


def test_discovery_requests_world_zero_only(summary_module, monkeypatch, tmp_path):
    requested = []

    def fixture(source, world, candidate=None):
        requested.append(world)
        if world != 0:
            raise AssertionError("Discovery cannot request world 1")
        return [], {}, ["not complete"], []

    monkeypatch.setattr(summary_module, "causal_effect_rows", fixture)
    summary_module.discovery(tmp_path / "source", tmp_path / "output")
    assert requested == [0]
    audit = json.loads((tmp_path / "output/world-0-audit.json").read_text())
    assert not audit["complete"]


def test_clean_donor_strata_preserve_overlap_errors_and_all_cases(summary_module):
    candidates = np.array(
        [[1, 2, 10, 20], [1, 2, 10, 20], [1, 2, 10, 10], [1, 2, 10, 20], [1, 2, 10, 20]]
    )
    prediction = np.array([10, 20, 10, 99, 20])
    ended = np.array([True, True, True, True, False])
    strata = summary_module.donor_prediction_strata(prediction, ended, candidates)
    assert list(strata) == list(summary_module.DONOR_STRATA)
    assert sum(np.sum(strata == label) for label in summary_module.DONOR_STRATA) == 5


def test_answer_state_is_retained_but_cannot_rank_or_lock(summary_module, monkeypatch, tmp_path):
    base = {
        "mechanism": "actual_path",
        "known_only": False,
        "n": 8,
        "donor_minus_random": 0.2,
        "donor_minus_sham": 0.2,
    }
    rows = [
        {**base, "layer": 3, "position": 3},
        {**base, "layer": 6, "position": 4, "donor_minus_random": 1.0},
        {**base, "layer": 7, "position": 2, "donor_minus_random": 1.0},
    ]
    # This tests the causal-order rule independently of any real discovery values.
    monkeypatch.setattr(
        summary_module, "causal_effect_rows", lambda *args, **kwargs: (rows, {}, [], [])
    )
    output = tmp_path / "discovery"
    summary_module.discovery(tmp_path / "source", output)
    with (output / "world-0-effects.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    with (output / "world-0-candidate-ranks.csv").open() as stream:
        ranks = list(csv.DictReader(stream))
    assert [(int(row["layer"]), int(row["position"])) for row in ranks] == [(3, 3)]
    audit_path = output / "world-0-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["complete"] = True
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="positions 2/3"):
        summary_module.lock_candidate(
            output,
            tmp_path / "bad-lock.json",
            "actual_path",
            6,
            4,
            "answer position cannot identify routing",
        )
    assert not (tmp_path / "bad-lock.json").exists()


def test_primary_rank_uses_all_pairs_not_correct_subset(summary_module):
    rows = [
        {
            "mechanism": "actual_path",
            "layer": layer,
            "position": 3,
            "known_only": known,
            "n": 8,
            "donor_minus_random": score,
            "donor_minus_sham": score,
        }
        for layer, known, score in (
            (2, False, 0.1),
            (3, False, 0.2),
            (2, True, 0.9),
            (3, True, 0.1),
        )
    ]
    assert summary_module.rank_candidates(rows)[0]["layer"] == 3
    assert summary_module.rank_candidates(rows, known_only=True)[0]["layer"] == 2


def test_modified_artifact_is_rejected_against_producer_hash(summary_module, tmp_path):
    path = tmp_path / "node.npz"
    np.savez(path, prediction=np.array([1]))
    sealed = {"node.npz": summary_module.file_sha256(path)}
    np.savez(path, prediction=np.array([2]))
    with pytest.raises(ValueError, match="Producer artifact hash mismatch"):
        summary_module.load_verified_causal_arrays(tmp_path, "node.npz", sealed, {})


def test_wrong_checkpoint_identity_fails_before_reading_nodes(summary_module, tmp_path):
    run = {
        "source": tmp_path / "original",
        "width": 256,
        "world": 0,
        "seed": 0,
        "condition": "company",
    }
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "node_files": 144,
                "identity": {
                    "width": 256,
                    "world": 0,
                    "seed": 0,
                    "condition": "company",
                    "step": 10240,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="producer identity mismatch"):
        summary_module.verify_causal_producer(tmp_path, run, None, {})


@pytest.mark.parametrize("corrupted", ["candidate_tokens", "baseline_correct", "prediction"])
def test_candidate_baseline_and_sham_rescored_from_world(summary_module, corrupted):
    world = SimpleNamespace(
        answers=np.array([10, 20, 10, 30]),
        root_ids=np.array([[0, 1]]),
        actual_ids=np.array([[2, 3]]),
        memberships=np.array([[0, 1]]),
        derived_ids=np.array([[4, 5]]),
    )
    selected = {"recipient": np.array([0]), "donor": np.array([1])}
    clean = {
        "people": np.array([0, 1]),
        "prediction": np.array([10, 20]),
        "ended": np.array([True, True]),
    }
    candidates = np.array([[10, 10, 20, 30]])
    classes, bits = summary_module.classify_answers(np.array([10]), np.array([True]), candidates)
    arrays = {
        **selected,
        "recipient_query": np.array([4]),
        "donor_query": np.array([5]),
        "candidate_tokens": candidates,
        "baseline_prediction": np.array([10]),
        "baseline_ended": np.array([True]),
        "baseline_correct": np.array([True]),
        "prediction": np.array([10]),
        "ended": np.array([True]),
        "correct": np.array([True]),
        "class_code": classes,
        "matched_role_bits": bits,
    }
    summary_module.verify_causal_node(arrays, selected, world, 0, clean, "sham")
    arrays[corrupted] = arrays[corrupted].copy()
    arrays[corrupted].flat[0] = False if corrupted == "baseline_correct" else 99
    with pytest.raises(ValueError, match="scoring/sham mismatch"):
        summary_module.verify_causal_node(arrays, selected, world, 0, clean, "sham")
