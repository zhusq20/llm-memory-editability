"""V2 changes only numerical identity/baselines; interventions stay frozen."""

import inspect
import json

import numpy as np
import pytest
import torch

from llm_memory_editability import bios_mechanism_causal as v1
from llm_memory_editability import bios_mechanism_causal_v2 as v2
from llm_memory_editability.bios_model import CausalLM, ModelConfig


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_interventions_and_scientific_matrix_are_unchanged():
    for name in (
        "SOURCE",
        "DOWNSTREAM",
        "RANK",
        "FAMILIES",
        "ARMS",
        "TRANSFER_TYPES",
        "TRANSFER_ARMS",
    ):
        assert getattr(v1, name) == getattr(v2, name)
    for name in (
        "generate_queries",
        "lesion_operations",
        "transfer_operations",
        "make_probe_plan",
        "make_transfer_plan",
        "wrong_source_indices",
        "score_rows",
        "paired_score_rows",
    ):
        assert inspect.getsource(getattr(v1, name)) == inspect.getsource(getattr(v2, name))


def test_archive_disagreements_are_retained_without_query_exclusion(tmp_path):
    archived = {"prediction": np.array([4, 5, 6]), "ended": np.array([True, True, True])}
    ids = np.array([2, 0, 2])
    local = {"prediction": np.array([9, 4, 9]), "ended": np.array([True, False, True])}
    path = tmp_path / "differences.npz"
    v2.record_archive_comparison(path, local, archived, ids)
    with np.load(path) as recorded:
        np.testing.assert_array_equal(recorded["query_id"], ids)
        assert recorded["prediction_changed"].tolist() == [True, False, True]
        assert recorded["ended_changed"].tolist() == [False, True, False]
        np.testing.assert_array_equal(recorded["local_prediction"], local["prediction"])
    # The full-order identity gate still rejects the same discrepancy.
    with pytest.raises(ValueError, match="differs from archive"):
        v2._check_predictions(local, archived, ids)
    v2.record_archive_comparison(path, local, archived, ids)
    with pytest.raises(ValueError, match="plan changed"):
        v2.record_archive_comparison(
            path, {**local, "prediction": np.array([6, 4, 6])}, archived, ids
        )


def test_full_original_replay_is_strict_cached_and_costed(tmp_path):
    from types import SimpleNamespace

    torch.manual_seed(81)
    model = CausalLM(ModelConfig(vocab_size=24, width=12, layers=4, heads=1)).eval()
    world = SimpleNamespace(
        prompts=np.array([[1, 5, 6, 2, 0], [1, 8, 6, 7, 2]]),
        lengths=np.array([4, 5]),
        answers=np.array([1, 1]),
    )
    device = torch.device("cpu")
    archive = v1.generate_queries(model, world.prompts, world.lengths, device, 512)
    v2.verify_original_replay(model, world, archive, tmp_path, device)
    report = json.loads((tmp_path / "original-replay.json").read_text())
    assert report["prediction_differences"] == report["ended_differences"] == 0
    assert report["batch_size"] == 512
    assert report["extra_identity_gate_cost"]["forward_examples"] == 4
    bad = {**archive, "prediction": (archive["prediction"] + 1) % 24}
    with pytest.raises(ValueError, match="differs from archive"):
        v2.verify_original_replay(model, world, bad, tmp_path, device)


def test_v2_seal_includes_numerical_evidence_and_excludes_scheduler_logs(tmp_path):
    (tmp_path / "company").mkdir()
    for name in (
        "contract.json",
        "original-replay.npz",
        "original-replay.json",
        "company/archive-probe-clean.npz",
        "company/transfer-local-clean.npz",
        "company/archive-discrepancies.json",
        "company/transfer-reference-drift.npz",
    ):
        (tmp_path / name).write_bytes(b"scientific evidence")
    (tmp_path / "queue-worker.log").write_text("still mutable")
    hashes = v2.scientific_outputs(tmp_path)
    assert len(hashes) == 7
    assert "original-replay.npz" in hashes and "company/transfer-local-clean.npz" in hashes
    assert "queue-worker.log" not in hashes
