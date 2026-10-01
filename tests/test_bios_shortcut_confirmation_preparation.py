"""Preparation uses synthetic source files and only development worlds 0 and 1."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from llm_memory_editability.bios_data import REVISION, SOURCE_FOLDER, make_world


@pytest.fixture(scope="module")
def preparation_module():
    path = Path(__file__).parents[1] / "scripts/prepare_bios_shortcut_confirmation.py"
    spec = importlib.util.spec_from_file_location("confirmation_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source(tmp_path_factory, preparation_module):
    repo = tmp_path_factory.mktemp("preparation-dev-fixture")
    source = repo / "data/bios-source"
    folder = source / SOURCE_FOLDER / "fields"
    folder.mkdir(parents=True)
    entries = []
    counts = {
        "first_name": 400,
        "middle_name": 400,
        "last_name": 1000,
        "city": 200,
        "company": 263,
        "university": 300,
        "field": 100,
    }
    for name, count in counts.items():
        path = folder / f"{name}.txt"
        path.write_text("\n".join(f"{name}_{i}" for i in range(count)))
        entries.append(
            {
                "path": str(path.relative_to(source)),
                "bytes": path.stat().st_size,
                "sha256": preparation_module.digest(path),
            }
        )
    for relative in ["LICENSE", f"{SOURCE_FOLDER}/Capo-bioS-bioR.py"]:
        path = source / relative
        path.write_text("Synthetic testing fixture; never used for confirmation.\n")
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": preparation_module.digest(path),
            }
        )
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({"revision": REVISION, "files": entries}))
    frozen = {
        str(p.relative_to(repo)): preparation_module.digest(p)
        for p in [manifest, *(source / entry["path"] for entry in entries)]
    }
    # The only actual generator calls in this test suite are these two dev seeds.
    worlds = {seed: make_world(seed, source) for seed in (0, 1)}
    return repo, source, frozen, worlds


def prepare_args(source, base):
    repo, source_root, frozen, worlds = source

    def factory(seed, requested_source):
        assert seed in (0, 1), "Unit tests must never generate confirmation worlds"
        assert requested_source == source_root
        return worlds[seed]

    return {
        "worlds": [0, 1],
        "base_root": base,
        "source_root": source_root,
        "repo_root": repo,
        "frozen_sources": frozen,
        "config_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
        "world_factory": factory,
    }


def test_development_world_bytes_and_receipts_are_immutable_on_resume(
    tmp_path, source, preparation_module
):
    module = preparation_module
    args = prepare_args(source, tmp_path / "base")
    first = module._prepare_worlds(**args)
    assert first["complete"] and first["worlds"] == [0, 1]
    assert len(first["files_sha256"]) == 6
    assert first["source_files_sha256"] == source[2]
    for seed in (0, 1):
        assert {p.name for p in (args["base_root"] / f"world-{seed}").iterdir()} == {
            "world.npz",
            "metadata.json",
            "audit.json",
        }
    snapshots = {name: (args["base_root"] / name).read_bytes() for name in first["files_sha256"]}
    assert module._prepare_worlds(**args) == first
    assert all(
        (args["base_root"] / name).read_bytes() == value for name, value in snapshots.items()
    )


def test_crash_resume_keeps_committed_prefix_and_regenerates_remaining(
    tmp_path, source, preparation_module
):
    module = preparation_module
    args = prepare_args(source, tmp_path / "base")
    factory = args["world_factory"]

    def interrupted(seed, root):
        if seed == 1:
            raise RuntimeError("Simulated interruption")
        return factory(seed, root)

    with pytest.raises(RuntimeError, match="interruption"):
        module._prepare_worlds(**{**args, "world_factory": interrupted})
    assert (args["base_root"] / "world-0/world.npz").exists()
    assert not (args["base_root"] / "preparation-audit.json").exists()
    before = module.digest(args["base_root"] / "world-0/world.npz")
    receipt = module._prepare_worlds(**args)
    assert receipt["complete"]
    assert before == module.digest(args["base_root"] / "world-0/world.npz")


def test_existing_mismatched_world_is_rejected_without_overwriting(
    tmp_path, source, preparation_module
):
    module = preparation_module
    args = prepare_args(source, tmp_path / "base")
    module._prepare_worlds(**args)
    damaged = args["base_root"] / "world-0/world.npz"
    damaged.write_bytes(b"Deliberately changed fixture")
    before = damaged.read_bytes()
    with pytest.raises(ValueError, match="differs from deterministic"):
        module._prepare_worlds(**args)
    assert damaged.read_bytes() == before


def test_exact_source_manifest_and_entry_lock_are_required(source, preparation_module):
    repo, root, frozen, _ = source
    module = preparation_module
    assert module.verified_sources(repo, root, frozen) == frozen
    omitted = dict(frozen)
    omitted.pop(next(iter(omitted)))
    with pytest.raises(ValueError, match="every entry exactly"):
        module.verified_sources(repo, root, omitted)
    with pytest.raises(ValueError, match="every entry exactly"):
        module.verified_sources(repo, root, {**frozen, "extra": "0" * 64})
    altered = dict(frozen)
    altered[next(iter(altered))] = "0" * 64
    with pytest.raises(ValueError, match="every entry exactly"):
        module.verified_sources(repo, root, altered)


def test_resume_rejects_different_lock_before_generation(tmp_path, source, preparation_module):
    args = prepare_args(source, tmp_path / "base")
    preparation_module._prepare_worlds(**args)

    def forbidden(*_):
        raise AssertionError("Existing lock conflict must be rejected before generation")

    with pytest.raises(ValueError, match="different frozen lock"):
        preparation_module._prepare_worlds(
            **{**args, "lock_sha256": "e" * 64, "world_factory": forbidden}
        )


def test_source_bytes_corruption_stops_before_generation(tmp_path, preparation_module):
    repo, root = tmp_path, tmp_path / "source"
    root.mkdir()
    (root / "entry.txt").write_text("changed")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "files": [{"path": "entry.txt", "sha256": hashlib.sha256(b"original").hexdigest()}],
            }
        )
    )
    with pytest.raises(ValueError, match="entry changed"):
        preparation_module.verified_sources(repo, root, {})


def test_protected_development_root_and_receipt_conflict_rejected(
    tmp_path, source, preparation_module
):
    module = preparation_module
    args = prepare_args(source, source[0] / "data/bios-organization-v1")
    with pytest.raises(ValueError, match="isolated"):
        module._prepare_worlds(**args)
    receipt = tmp_path / "receipt.json"
    module.immutable_json(receipt, {"lock": "first"})
    with pytest.raises(ValueError, match="receipt mismatch"):
        module.immutable_json(receipt, {"lock": "second"})
    assert json.loads(receipt.read_text()) == {"lock": "first"}


def test_formal_entry_guard_runs_before_any_preparation(tmp_path, preparation_module, monkeypatch):
    from llm_memory_editability import bios_shortcut_confirmation as runtime

    def not_locked(*_):
        raise ValueError("Prospective lock missing")

    def forbidden(*_, **__):
        raise AssertionError("Preparation must never start without the prospective lock")

    monkeypatch.setattr(runtime, "validate_frozen_design", not_locked)
    monkeypatch.setattr(preparation_module, "_prepare_worlds", forbidden)
    with pytest.raises(ValueError, match="lock missing"):
        preparation_module.prepare(tmp_path / "config.json", tmp_path / "lock.json")
