"""Evaluation recovery preserves evidence, private modes and the training boundary."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluation_recovery",
    Path(__file__).resolve().parents[1] / "scripts/resume_ttcd_evaluations.py",
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_mount_repair_preserves_every_scientific_argument_and_readonly_flag():
    command = [
        "docker",
        "create",
        "--mount",
        "type=bind,source=/private,target=/output",
        "--mount",
        "type=bind,source=/old,target=/parent-evaluation,readonly",
        "image",
        "python",
        "evaluate",
        "--writer-steps",
        "1",
    ]
    fixed = recovery.redirect_mounts(
        command, {"/output": "/transport", "/parent-evaluation": "/audited"}
    )
    assert fixed[3] == "type=bind,source=/transport,target=/output"
    assert fixed[5] == "type=bind,source=/audited,target=/parent-evaluation,readonly"
    assert fixed[6:] == command[6:]
    with pytest.raises(ValueError, match="exactly once"):
        recovery.redirect_mounts(command, {"/unknown": "/transport"})


def test_public_output_is_separate_from_existing_private_evidence(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    transport = tmp_path / "transport"
    recovery.public_directory(transport / "arm/evaluation", transport)
    assert private.stat().st_mode & 0o777 == 0o700
    assert transport.stat().st_mode & 0o777 == 0o755
    with pytest.raises(ValueError, match="escaped"):
        recovery.public_directory(private, transport)


def test_publish_reuses_large_state_storage_and_refuses_overwrite(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    (source / "fast.pt").write_bytes(b"immutable fast state")
    recovery.link_evidence(source, target)
    assert os.path.samefile(source / "fast.pt", target / "fast.pt")
    (source / "summary.json").write_text("new")
    (target / "summary.json").write_text("old")
    with pytest.raises(ValueError, match="different evidence"):
        recovery.link_evidence(source, target)
    assert (target / "summary.json").read_text() == "old"


@pytest.mark.parametrize("stage", ["train", "train-audit", "foundation-train", "common-train"])
def test_recovery_never_calls_a_training_stage(stage):
    with pytest.raises(ValueError, match="training is prohibited"):
        recovery.run_stage(None, None, None, None, stage, None, None, None)


def test_successful_exit_without_exact_result_hash_cannot_be_reused(tmp_path):
    batch = tmp_path / "batch"
    stage = recovery.stage_directory(batch, "loop", "audit")
    stage.mkdir(parents=True)
    (stage / "exit.json").write_text(json.dumps({"returncode": 0}))
    (stage / "result.json").write_text(json.dumps({"sha256": "original"}))
    output = recovery.canonical_output(batch, "loop", "audit")
    output.mkdir(parents=True)
    (output / "result-audit.json").write_text(json.dumps({"passed": True}))
    controller = SimpleNamespace(
        read_json=lambda p: json.loads(p.read_text()), sha256=lambda _: "changed"
    )
    with pytest.raises(ValueError, match="canonical evidence"):
        recovery.stage_complete(controller, batch, "loop", "audit")
