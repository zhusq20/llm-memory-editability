"""CPU-only orchestration using development world0 and synthetic checkpoint bytes.

No reserved world is generated, loaded, trained, or scored. CUDA readiness is
mocked only so the producer's call graph can be exercised without GPU execution.
"""

import copy
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import bios_cross_train, bios_shortcut_edit_v2
from llm_memory_editability import bios_shortcut_confirmation as worker
from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_organization_train import state_hash
from llm_memory_editability.bios_shortcut_confirmation_sets import make_confirmation_edit_pair
from llm_memory_editability.bios_shortcut_control import high_exception_world

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def development_world():
    world = make_cross_world(0)
    assert world.seed == 0
    return world


def test_each_phase_dispatches_eight_same_parent_edits_with_support_bound_identity(
    tmp_path, monkeypatch, development_world
):
    assert worker.learning is bios_cross_train.learning
    assert worker.run_case is bios_shortcut_edit_v2.run_case
    config = json.loads((ROOT / "configs/bios-shortcut-confirmation-v1.json").read_text())
    config_path, lock_path = tmp_path / "config.json", tmp_path / "lock.json"
    config_path.write_text(json.dumps(config))
    lock_path.write_text('{"synthetic_unit_test": true}')
    base = tmp_path / "prepared-development-world0"
    base.mkdir()
    receipt = {"files_sha256": {"world-0/world.npz": "synthetic-unit-test"}}
    (base / "preparation-audit.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(worker, "validate_lock", lambda *args: (copy.deepcopy(config), {}))
    monkeypatch.setattr(worker, "validate_prepared_world", lambda *args: (base, receipt))
    monkeypatch.setattr(worker, "confirmation_sources", lambda: {"unit-test": "unchanged"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    materialized = []

    def only_development_world(seed, source_root):
        assert seed == 0 and source_root == base
        materialized.append(seed)
        return development_world

    monkeypatch.setattr(worker, "make_cross_world", only_development_world)
    models, calls, learning_calls = [], [], []

    def fake_learning(args, world, out, study, device):
        assert args.world == 0 and world.seed == 0 and device.type == "cuda"
        assert study["steps"] == 15360 and study["checkpoints"] == list(worker.LEARNING_CHECKPOINTS)
        assert study["lr"] == 1e-4 and study["document_QA_weights"] == [0.8, 0.2]
        assert study["supports"] == [0, 1] and study["phase"] == args.phase
        model = torch.nn.Linear(2, 2)  # Tiny CPU placeholder; never trained or evaluated.
        with torch.no_grad():
            model.weight.fill_(0.25)
            model.bias.fill_(-0.125)
        models.append(model)
        baseline = {key: value.detach().clone() for key, value in model.state_dict().items()}
        learning_calls.append((args.phase, baseline, world.answers.copy()))
        np.savez(
            out / "predictions-15360.npz",
            prediction=world.answers,
            ended=np.ones(len(world.answers), dtype=bool),
            correct=np.ones(len(world.answers), dtype=bool),
            value_nll=np.zeros(len(world.answers)),
        )
        (out / "learning-complete.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model_sha256": state_hash(model.state_dict()),
                }
            )
        )
        for step in worker.LEARNING_CHECKPOINTS:
            (out / f"model-{step}.pt").write_bytes(b"synthetic, not a model checkpoint")
        (out / "resume.pt").write_bytes(b"synthetic optimizer/RNG placeholder")
        return model, {"unit_phase": args.phase}

    def fake_two_step(model, world, chain, device):
        assert world.seed == 0 and device.type == "cuda"
        return {
            "query_id": world.derived_ids[chain],
            "prediction": world.answers[world.derived_ids[chain]],
        }

    def fake_edit(
        model,
        baseline,
        data,
        world,
        pair,
        phase,
        kind,
        chain,
        output,
        identity_hash,
        old_arrays,
        device,
    ):
        support = int(output.name.split("-")[-1])
        assert model is models[-1] and data["unit_phase"] == phase and device.type == "cuda"
        expected = learning_calls[-1][1]
        assert set(baseline) == set(expected)
        for key in baseline:
            assert baseline[key].device.type == "cpu"
            assert torch.equal(baseline[key], expected[key])
            assert baseline[key].data_ptr() != model.state_dict()[key].data_ptr()
        # Mirror the frozen E39 reset, then perturb the toy model to catch any
        # alias between the baseline clone and a previous edit's mutable weights.
        model.load_state_dict(baseline)
        with torch.no_grad():
            model.weight.add_(10 + len(calls))
        np.testing.assert_array_equal(old_arrays["prediction"], world.answers)
        assert old_arrays["correct"].all()
        parent_hash = worker.file_hash(output.parent.parent / "launch-contract.json")
        assert identity_hash == worker.case_identity(parent_hash, chain, support, kind, pair)
        calls.append(
            dict(
                phase=phase,
                support=support,
                chain=chain,
                kind=kind,
                identity=identity_hash,
                E=pair["E"].copy(),
                D=pair["D"].copy(),
            )
        )
        dest = output / f"{worker.CHAINS[chain]}-{kind}-mlp"
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ("model-final.pt", "resume.pt"):
            (dest / filename).write_bytes(b"synthetic, not an editing outcome")

    monkeypatch.setattr(worker, "learning", fake_learning)
    monkeypatch.setattr(worker, "autonomous_two_step", fake_two_step)
    monkeypatch.setattr(worker, "run_case", fake_edit)
    for phase in ("low", "high"):
        args = Namespace(
            world=0,
            seed=0,
            condition="company",
            phase=phase,
            config=config_path,
            lock=lock_path,
            output=tmp_path / phase,
            device="cuda",
            threads=1,
        )
        worker.run(args)
        phase_calls = [call for call in calls if call["phase"] == phase]
        assert len(phase_calls) == len({call["identity"] for call in phase_calls}) == 8
        assert {(c["support"], c["chain"], c["kind"]) for c in phase_calls} == {
            (support, chain, kind)
            for support in (0, 1)
            for chain in (0, 1)
            for kind in ("coherent", "exception")
        }
        completion = json.loads((args.output / "complete.json").read_text())
        assert completion["edit_cases"] == 8
        weights = [name for name in completion["artifacts"] if name.endswith(".pt")]
        assert len(weights) == 23
        # Rerunning a sealed complete worker does not retrain or dispatch edits.
        worker.run(args)
        assert len([call for call in calls if call["phase"] == phase]) == 8
    assert materialized == [0, 0] and len(learning_calls) == 2
    assert state_hash(learning_calls[0][1]) == state_hash(learning_calls[1][1])
    for chain in (0, 1):
        for kind in ("coherent", "exception"):
            for support in (0, 1):
                paired = [
                    c
                    for c in calls
                    if (c["chain"], c["kind"], c["support"]) == (chain, kind, support)
                ]
                for key in ("E", "D"):
                    np.testing.assert_array_equal(paired[0][key], paired[1][key])


def test_support_identity_binds_support_kind_chain_and_complete_pair_contract(development_world):
    high, _, _, _ = high_exception_world(development_world)
    pair = make_confirmation_edit_pair(development_world, high, 0, 0)
    expected = worker.case_identity("parent", 0, 0, "exception", pair)
    assert worker.case_identity("parent", 0, 0, "exception", pair) == expected
    assert worker.case_identity("parent", 0, 1, "exception", pair) != expected
    assert worker.case_identity("parent", 1, 0, "exception", pair) != expected
    assert worker.case_identity("parent", 0, 0, "coherent", pair) != expected
    assert worker.case_identity("different-parent", 0, 0, "exception", pair) != expected
    changed = copy.deepcopy(pair)
    changed["contract"]["array_sha256"]["E"] = "different-support-data"
    assert worker.case_identity("parent", 0, 0, "exception", changed) != expected
    # Frozen E39 resumes compare this exact passed identity before restoring
    # optimizer/model/RNG, so equal case names across supports cannot accept it.
    saved_resume = {"case": "company-exception-mlp", "identity_sha256": expected}
    assert saved_resume["identity_sha256"] != worker.case_identity(
        "parent", 0, 1, "exception", pair
    )


def test_every_one_of_23_weight_artifacts_is_required_for_producer_seal(tmp_path):
    (tmp_path / "launch-contract.json").write_text("{}")
    (tmp_path / "manipulation.npz").write_bytes(b"synthetic")
    required = worker.expected_weight_artifacts()
    assert len(required) == len(set(required)) == 23
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint")
    manifest = worker.scientific_artifacts(tmp_path)
    assert sum(name.endswith(".pt") for name in manifest) == 23
    for relative in required:
        path = tmp_path / relative
        original = path.read_bytes()
        path.unlink()
        with pytest.raises(ValueError, match="Required confirmation weight artifact"):
            worker.scientific_artifacts(tmp_path)
        path.write_bytes(original)


@pytest.mark.parametrize("cuda_available,bf16_available", [(False, True), (True, False)])
def test_unusable_cuda_rejected_before_any_world_or_model_is_materialized(
    tmp_path, monkeypatch, cuda_available, bf16_available
):
    monkeypatch.setattr(worker, "validate_lock", lambda *args: ({}, {}))
    monkeypatch.setattr(worker, "validate_prepared_world", lambda *args: (tmp_path, {}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_available)

    def forbidden(*args, **kwargs):
        pytest.fail("World/model construction must not occur without CUDA BF16")

    monkeypatch.setattr(worker, "make_cross_world", forbidden)
    monkeypatch.setattr(worker, "learning", forbidden)
    args = Namespace(
        config=tmp_path / "unused",
        lock=tmp_path / "unused-lock",
        world=0,
        output=tmp_path / "output",
        threads=1,
    )
    with pytest.raises(ValueError, match="CUDA device with BF16"):
        worker.run(args)
    assert not (args.output / "launch-contract.json").exists()


@pytest.fixture
def synthetic_preparation(tmp_path, monkeypatch):
    """Seal tiny development-ID byte fixtures; never construct any world."""
    monkeypatch.setattr(worker, "WORLDS", (0, 1))
    root = tmp_path / "prepared"
    root.mkdir()
    config = {"base_world_root": str(root)}
    config_path, lock_path = tmp_path / "config.json", tmp_path / "lock.json"
    config_path.write_text(json.dumps(config))
    sources = {"data/bios-source/manifest.json": "synthetic-source-sha256"}
    lock_path.write_text(json.dumps({"data_source_files": sources}))
    files = {}
    for world in (0, 1):
        directory = root / f"world-{world}"
        directory.mkdir()
        (directory / "world.npz").write_bytes(b"synthetic; not a world archive")
        (directory / "metadata.json").write_text(
            json.dumps({"seed": world, "token_labels": [], "names": {}})
        )
        (directory / "audit.json").write_text('{"synthetic": true}')
        for filename in ("world.npz", "metadata.json", "audit.json"):
            path = directory / filename
            files[str(path.relative_to(root))] = worker.file_hash(path)
    receipt = {
        "complete": True,
        "config_sha256": worker.file_hash(config_path),
        "lock_sha256": worker.file_hash(lock_path),
        "worlds": [0, 1],
        "source_files_sha256": sources,
        "files_sha256": files,
    }
    (root / "preparation-audit.json").write_text(json.dumps(receipt))
    return config, config_path, lock_path, root, receipt


def test_prepared_world_accepts_exact_locked_manifest_and_metadata(synthetic_preparation):
    config, config_path, lock_path, root, receipt = synthetic_preparation
    for world in (0, 1):
        assert worker.validate_prepared_world(config, config_path, lock_path, world) == (
            root,
            receipt,
        )
    with pytest.raises(ValueError, match="outside the prospective preparation matrix"):
        worker.validate_prepared_world(config, config_path, lock_path, 2)


@pytest.mark.parametrize("mutation", ["missing", "different", "extra"])
def test_prepared_world_rejects_nonexact_source_identity(synthetic_preparation, mutation):
    config, config_path, lock_path, root, receipt = synthetic_preparation
    if mutation == "missing":
        del receipt["source_files_sha256"]
    elif mutation == "different":
        receipt["source_files_sha256"]["data/bios-source/manifest.json"] = "different"
    else:
        receipt["source_files_sha256"]["extra-file"] = "extra"
    (root / "preparation-audit.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="source files differ"):
        worker.validate_prepared_world(config, config_path, lock_path, 0)


@pytest.mark.parametrize("mutation", ["missing", "missing_other_world", "extra"])
def test_prepared_world_rejects_nonexact_file_manifest(synthetic_preparation, mutation):
    config, config_path, lock_path, root, receipt = synthetic_preparation
    if mutation == "missing":
        del receipt["files_sha256"]
    elif mutation == "missing_other_world":
        del receipt["files_sha256"]["world-1/audit.json"]
    else:
        receipt["files_sha256"]["world-2/metadata.json"] = "extra"
    (root / "preparation-audit.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="file manifest differs"):
        worker.validate_prepared_world(config, config_path, lock_path, 0)


@pytest.mark.parametrize("seed", [0, "1", True, None])
def test_prepared_world_rejects_wrong_metadata_identity_even_with_matching_hash(
    synthetic_preparation, seed
):
    config, config_path, lock_path, root, receipt = synthetic_preparation
    path = root / "world-1/metadata.json"
    metadata = json.loads(path.read_text())
    if seed is None:
        del metadata["seed"]
    else:
        metadata["seed"] = seed
    path.write_text(json.dumps(metadata))
    receipt["files_sha256"]["world-1/metadata.json"] = worker.file_hash(path)
    (root / "preparation-audit.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="metadata seed differs"):
        worker.validate_prepared_world(config, config_path, lock_path, 1)
