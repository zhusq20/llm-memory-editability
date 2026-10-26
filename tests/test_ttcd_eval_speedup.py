"""I/O-only TTCD amendment: complete states and exact independent replay."""

import copy
import json

import pytest
import test_ttcd_continual_eval
import test_ttcd_double_new_eval
import torch
from test_ttcd_architecture_eval import ARCHITECTURES, make_reader

from llm_memory_editability import ttcd_continual_eval as ce
from llm_memory_editability import ttcd_double_new_eval as dn
from llm_memory_editability import ttcd_editing as editing
from llm_memory_editability import ttcd_eval_speedup as speedup
from llm_memory_editability import ttcd_experiment


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def material(tmp_path):
    return test_ttcd_double_new_eval.material.__wrapped__(tmp_path)


@pytest.fixture
def experiment(tmp_path):
    return test_ttcd_continual_eval.experiment.__wrapped__(tmp_path)


def fixture_identity(*args):
    return {"immutable_fixture": True}


def assert_payload_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "fast":
            assert actual[key].keys() == expected[key].keys()
            for layer, tensor in actual[key].items():
                other = expected[key][layer]
                assert tensor.dtype == other.dtype and tensor.shape == other.shape
                assert torch.equal(tensor, other)
        else:
            assert actual[key] == expected[key]


@pytest.mark.parametrize("architecture", tuple(ARCHITECTURES))
def test_three_frozen_architectures_preserve_payload_reads_and_write_transitions(
    tmp_path, architecture
):
    reader = make_reader(architecture)
    fast = {k: torch.randn_like(v) * 0.0001 for k, v in reader.empty_fast().items()}
    provenance = {"key": "independent-fixture", "indices": [0], "read": True}
    original_save, original_load, original_digest = (
        editing.save_fast,
        editing.load_fast,
        editing.tensor_digest,
    )
    original_path, optimized_path = [tmp_path / label / "fast.pt" for label in ("old", "new")]
    expected_digest = original_save(original_path, reader, fast, provenance)
    expected_fast, expected_provenance = original_load(original_path, reader)
    packet = torch.tensor([[3, 36, 5, 2] + [0] * 17 + [3, 36]])
    expected_update, expected_history = reader.write(
        packet, expected_fast, teacher_window=32, student_window=16, learning_rate=0.001
    )
    query, positions = torch.tensor([[3, 36, 0, 0]]), torch.tensor([[1, 2]])
    with torch.no_grad():
        expected_logits = reader(query, positions, fast=expected_fast)
    compute_before = dict(reader.compute)
    undo, stats = speedup.install()
    try:
        assert ce.tensor_digest is original_digest and dn.tensor_digest is original_digest
        assert ce.save_fast == editing.save_fast == dn.save_fast == ttcd_experiment.save_fast
        assert ce.load_fast == editing.load_fast == dn.load_fast == ttcd_experiment.load_fast
        actual_digest = ce.save_fast(optimized_path, reader, fast, provenance)
        assert reader.compute == compute_before
        actual_fast, actual_provenance = dn.load_fast(optimized_path, reader)
        assert actual_digest == expected_digest
        assert actual_provenance == expected_provenance == provenance
        assert_payload_equal(
            torch.load(optimized_path, weights_only=True),
            torch.load(original_path, weights_only=True),
        )
        assert optimized_path.read_bytes() == original_path.read_bytes()
        assert all(torch.equal(actual_fast[k], expected_fast[k]) for k in fast)
        actual_update, actual_history = reader.write(
            packet, actual_fast, teacher_window=32, student_window=16, learning_rate=0.001
        )
        assert actual_history == expected_history
        assert all(torch.equal(actual_update[k], expected_update[k]) for k in fast)
        with torch.no_grad():
            assert torch.equal(reader(query, positions, fast=actual_fast), expected_logits)
        assert stats["registered_backbones"] == stats["backbone_cache_misses"] == 1
        assert stats["backbone_cache_hits"] >= 2
        assert stats["cpu_fast_snapshots"] == 1
        assert stats["scientific_training_updates"] == 0
        speedup.verify()
    finally:
        undo()
    assert stats["verification_passed"] and stats["restored"]
    assert editing.save_fast is original_save and editing.load_fast is original_load
    assert editing.tensor_digest is original_digest and ce.tensor_digest is original_digest


def test_parameter_version_invalidates_cache_and_old_payload_is_rejected(tmp_path):
    reader = make_reader("loop4x2")
    original_digest = editing.tensor_digest
    undo, stats = speedup.install()
    try:
        path = tmp_path / "fast.pt"
        editing.save_fast(path, reader, reader.empty_fast(), {})
        before = original_digest(reader.backbone.state_dict())
        parameter = next(reader.backbone.parameters())
        version = parameter._version
        with torch.no_grad():
            parameter.add_(0.25)
        assert parameter._version > version
        assert editing.tensor_digest(reader.backbone.state_dict()) != before
        with pytest.raises(ValueError, match="different backbone"):
            editing.load_fast(path, reader)
        assert stats["backbone_signature_invalidations"] == 1
        assert stats["backbone_cache_misses"] == 2
        with pytest.raises(AssertionError, match="changed frozen backbone"):
            undo()
    finally:
        undo(verify=False)
    assert stats["restored"] and stats["verification_passed"] is False
    assert editing.tensor_digest is original_digest


def test_complete_exit_hash_detects_mutation_even_without_version_increment(tmp_path):
    reader = make_reader("d8")
    undo, stats = speedup.install()
    try:
        editing.save_fast(tmp_path / "fast.pt", reader, reader.empty_fast(), {})
        parameter = next(reader.backbone.parameters())
        version = parameter._version
        parameter.data.add_(0.5)
        assert parameter._version == version
        with pytest.raises(AssertionError, match="changed frozen backbone"):
            speedup.verify()
    finally:
        undo(verify=False)
    assert stats["verification_passed"] is False and stats["restored"]


def test_fast_digest_and_shape_checks_remain_complete(tmp_path):
    reader = make_reader("disco4x2")
    undo, stats = speedup.install()
    try:
        fast, path = reader.empty_fast(), tmp_path / "fast.pt"
        before = editing.tensor_digest(fast)
        fast[0].add_(1)
        assert editing.tensor_digest(fast) != before
        editing.save_fast(path, reader, fast, {"provenance": "preserved"})
        payload = torch.load(path, weights_only=True)
        payload["fast"][0].add_(1)
        torch.save(payload, path)
        with pytest.raises(ValueError, match="hash mismatch"):
            editing.load_fast(path, reader)
        payload["fast"][0] = payload["fast"][0][:1]
        payload["fast_digest"] = editing.tensor_digest(payload["fast"])
        torch.save(payload, path)
        with pytest.raises(ValueError, match="shape/device"):
            editing.load_fast(path, reader)
        assert stats["uncached_digest_calls"] >= 5
    finally:
        undo()


def test_trainable_backbone_is_never_registered_and_install_is_reversible(tmp_path):
    reader = make_reader("loop4x2")
    next(reader.backbone.parameters()).requires_grad_(True)
    original_save = editing.save_fast
    undo, stats = speedup.install()
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            speedup.install()
        editing.save_fast(tmp_path / "fast.pt", reader, reader.empty_fast(), {})
        assert stats["registered_backbones"] == stats["backbone_cache_hits"] == 0
    finally:
        undo()
    assert editing.save_fast is original_save
    assert undo() is stats


def assert_all_states_equal(original, optimized):
    expected = sorted((original / "states").rglob("record.json"))
    actual = sorted((optimized / "states").rglob("record.json"))
    assert len(actual) == len(expected)
    for old, new in zip(expected, actual, strict=True):
        assert old.relative_to(original) == new.relative_to(optimized)
        # Includes every exact token, EOS, label, rank, score, margin, metrics,
        # provenance, write history and the unchanged complete fast digest.
        assert json.loads(old.read_text()) == json.loads(new.read_text())
        assert_payload_equal(
            torch.load(old.parent / "fast.pt", weights_only=True),
            torch.load(new.parent / "fast.pt", weights_only=True),
        )


def scientific_summary(value):
    return {k: v for k, v in value.items() if k not in ("wall_seconds", "compute", "utc")}


def test_existing_81_state_double_new_evaluation_and_fresh_replay_are_unchanged(material, tmp_path):
    config, settings, world, initial = material
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(dn.build_panel(config, settings, world)))
    results, audits = [], []
    stats = None
    for optimized in (False, True):
        reader, fresh = copy.deepcopy(initial), copy.deepcopy(initial)
        out = tmp_path / ("optimized" if optimized else "original")
        out.mkdir()
        undo = None
        if optimized:
            undo, stats = speedup.install()
        try:
            result = dn.evaluate(
                config,
                settings,
                "single",
                out,
                panel_path,
                "cpu",
                reader_loader=lambda *args, current=reader: (current, world),
                identity_factory=fixture_identity,
            )
            audit = dn.audit(
                config,
                settings,
                "single",
                out,
                panel_path,
                "cpu",
                reader_loader=lambda *args, current=fresh: (current, world),
                identity_factory=fixture_identity,
            )
            results.append(result)
            audits.append(audit)
        finally:
            if undo:
                undo()
    assert results[0]["states"] == audits[0]["states"] == 81
    assert audits[0]["passed"] and audits[1]["passed"]
    assert scientific_summary(results[0]) == scientific_summary(results[1])
    assert results[0]["compute"] == results[1]["compute"]
    assert scientific_summary(audits[0]) == scientific_summary(audits[1])
    assert_all_states_equal(tmp_path / "original", tmp_path / "optimized")
    assert stats["registered_backbones"] == stats["backbone_cache_misses"] == 2
    assert stats["backbone_cache_hits"] > 300
    assert stats["verification_passed"]


def test_original_29_state_revision_branch_and_exact_replay_are_unchanged(experiment, tmp_path):
    config, world, initial = experiment
    results, audits = [], []
    for optimized in (False, True):
        reader, fresh = copy.deepcopy(initial), copy.deepcopy(initial)
        out = tmp_path / ("optimized" if optimized else "original")
        out.mkdir()
        undo = None
        if optimized:
            undo, _ = speedup.install()
        try:
            ce.calibrate(
                config,
                "single",
                out,
                "cpu",
                reader_loader=lambda *args, current=reader: (current, world),
                identity_factory=fixture_identity,
            )
            results.append(
                ce.evaluate(
                    config,
                    "single",
                    out,
                    "cpu",
                    reader_loader=lambda *args, current=reader: (current, world),
                    identity_factory=fixture_identity,
                )
            )
            audits.append(
                ce.audit(
                    config,
                    "single",
                    out,
                    "cpu",
                    reader_loader=lambda *args, current=fresh: (current, world),
                    identity_factory=fixture_identity,
                )
            )
        finally:
            if undo:
                undo()
    assert results[0]["states"] == audits[0]["states"] == 29
    assert scientific_summary(results[0]) == scientific_summary(results[1])
    assert results[0]["compute"] == results[1]["compute"]
    assert scientific_summary(audits[0]) == scientific_summary(audits[1])
    assert_all_states_equal(tmp_path / "original", tmp_path / "optimized")
