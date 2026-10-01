"""Prospective gates use synthetic archives and reused development worlds only."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def summary(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "confirmation_summary_test", scripts / "summarize_bios_shortcut_confirmation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_matrix_never_generates_reserved_world(summary, monkeypatch, tmp_path):
    config = summary.ROOT / "configs/bios-shortcut-confirmation-v1.json"
    analysis = {
        name: summary.sha(summary.ROOT / name)
        for name in (
            "scripts/summarize_bios_shortcut_confirmation.py",
            "scripts/summarize_bios_shortcut_control.py",
            "scripts/summarize_bios_shortcut_edit_v2.py",
            "scripts/archive_bios_mechanism_stage.py",
        )
    }
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            dict(
                status="frozen_before_any_confirmation_model_or_outcome",
                config_sha256=summary.sha(config),
                sources=summary.confirmation_sources(),
                analysis_sources=analysis,
                output_root=str(tmp_path / "raw"),
                environment={
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "numpy": np.__version__,
                },
                data_source_files={
                    "configs/bios-shortcut-confirmation-v1.json": summary.sha(config)
                },
            )
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete matrix must not materialize any reserved world")

    monkeypatch.setattr(summary, "make_cross_world", forbidden)
    audit = summary.summarize(config, lock, None, tmp_path / "out", tmp_path / "not-yet.json")
    assert not audit["complete"] and len(audit["missing"]) == 96
    assert not audit["reserved_worlds_generated"]
    assert not list((tmp_path / "out").glob("*.csv"))
    with pytest.raises(ValueError, match="96 missing parents"):
        summary.summarize(config, lock, None, tmp_path / "out", tmp_path / "not-yet.json", True)


def endpoint_fixture(summary):
    learning, editing = [], []
    for job in summary.matrix():
        for chain in summary.CHAINS:
            base = {k: job[k] for k in ("world", "seed", "condition")}
            base.update(chain=chain, prevalence=job["phase"], split="heldout")
            learning.append(
                {**base, "step": 15360, "cohort": "original_exception", "n": 64, "correct": 32}
            )
            for support in (0, 1):
                for kind in summary.KINDS:
                    editing.append(
                        {
                            **base,
                            "support": support,
                            "kind": kind,
                            "step": 512,
                            "n": 9,
                            "correct": 4,
                        }
                    )
    return learning, editing


def test_endpoint_grid_rejects_duplicate_support_and_learning_replication(summary):
    learning, editing = endpoint_fixture(summary)
    summary.validate_endpoint_rows(learning, editing)
    assert len(learning) == 192 and len(editing) == 768
    with pytest.raises(ValueError, match="Editing endpoint matrix"):
        summary.validate_endpoint_rows(learning, [editing[0], *editing[:-1]])
    with pytest.raises(ValueError, match="Learning endpoint matrix"):
        summary.validate_endpoint_rows([*learning, learning[0]], editing)
    learning[0]["support"] = 0
    with pytest.raises(ValueError, match="denominator / cohort"):
        summary.validate_endpoint_rows(learning, editing)


def test_valid_parent_seal_archive_and_all_23_weight_files(summary, tmp_path):
    path = summary.ROOT / "scripts/archive_bios_mechanism_stage.py"
    spec = importlib.util.spec_from_file_location("tiny_archive_test", path)
    archiver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archiver)
    source = tmp_path / "raw"
    identity = dict(phase="low", world=100, seed=0, condition="company")
    run = summary.run_path(source, identity)
    for name in summary.required_artifacts():
        dest = run / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"synthetic file bytes; no model or new-world data")
    env = dict(torch="test", numpy="test", cuda="test")
    (run / "launch-contract.json").write_text(
        json.dumps(
            {
                **identity,
                **env,
                "protocol": summary.PROTOCOL,
                "preregistration_lock_sha256": "lock",
                "sources": summary.confirmation_sources(),
            }
        )
    )
    artifacts = {
        name: {"bytes": (run / name).stat().st_size, "sha256": summary.sha(run / name)}
        for name in summary.required_artifacts()
    }
    complete = dict(
        status="complete",
        protocol=summary.PROTOCOL,
        learning_steps=15360,
        edit_cases=8,
        edit_steps=512,
        lock_sha256="lock",
        artifacts=artifacts,
    )
    (run / "complete.json").write_text(json.dumps(complete))
    status = tmp_path / "status.json"
    status.write_text(json.dumps(dict(state="complete", active=[], failed=[])))
    archiver.archive(source, tmp_path / "verified.tar", [status])
    index_path = tmp_path / "verified.tar.json"
    index = summary.audit_archive(index_path, {"output_root": str(source)}, summary.Sources())
    _, weights = summary.audit_completion(
        run, identity, "lock", summary.Sources(), index, source, env
    )
    assert len(weights) == 23
    weight = "learning/model-15360.pt"
    prefix = str(run.relative_to(source))
    changed = {**artifacts[weight], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Producer seal differs"):
        summary.require_archived(index, f"{prefix}/{weight}", changed)
    del complete["artifacts"][weight]
    (run / "complete.json").write_text(json.dumps(complete))
    index["entries"][f"{prefix}/complete.json"] = {
        "bytes": (run / "complete.json").stat().st_size,
        "sha256": summary.sha(run / "complete.json"),
    }
    with pytest.raises(ValueError, match="omits required"):
        summary.audit_completion(run, identity, "lock", summary.Sources(), index, source, env)
    index["every_member_verified"] = False
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="independently verified"):
        summary.audit_archive(index_path, {"output_root": str(source)}, summary.Sources())


def test_initialization_matches_all_six_phase_organization_cells(summary):
    seen = {}
    config = dict(model={"width": 256}, initial_sha256="same", torch="x", numpy="y", precision="z")
    for phase in summary.PHASES:
        for condition in summary.CONDITIONS:
            identity = dict(world=100, seed=0, phase=phase, condition=condition)
            summary.audit_shared_initialization(seen, identity, config)
    assert len(seen) == 1
    with pytest.raises(ValueError, match="across phase/organization"):
        summary.audit_shared_initialization(seen, identity, {**config, "initial_sha256": "changed"})


def test_development_supports_preserve_e_u_budgets_and_disjointness(summary):
    low = summary.make_cross_world(0)  # Never generate reserved worlds 100..107 in tests.
    high, *_ = summary.high_exception_world(low)
    for chain in (0, 1):
        pairs = [
            summary.make_confirmation_edit_pair(low, high, chain, support) for support in (0, 1)
        ]
        for key in ("groups", "E", "D", "selected_people"):
            assert np.intersect1d(pairs[0][key], pairs[1][key]).size == 0
        for pair in pairs:
            counts = [len(pair[k]) for k in ("E", "D", "R", "paired_reference_D_heldout")]
            assert counts == [39, 96, 4096, 9]
            assert len(pair["U_full"]) == len(low.answers) - 39 - 96
            for phase, world in (("low", low), ("high", high)):
                for kind in summary.KINDS:
                    np.testing.assert_array_equal(
                        pair[f"{phase}_{kind}"][pair["U_full"]], world.answers[pair["U_full"]]
                    )
            np.testing.assert_array_equal(low.answers[pair["R"]], high.answers[pair["R"]])
