"""Only zero-update failed Docker-create claims may receive the mount repair."""

import importlib.util
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "one_step_training_launch_repair", ROOT / "scripts/resume_ttcd_one_step_training_launch.py"
)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def failed(tmp_path):
    batch = tmp_path / repair.BATCH
    batch.mkdir(mode=0o700)
    plan_path = batch / "transformer/source/configs/one-step-training.json"
    write(plan_path, {"same": "frozen science"})
    manifest = batch / "source-manifest.json"
    write(manifest, {"unchanged": "source hashes"})
    write(
        batch / "controller-process.json",
        {
            "pid": repair.OLD_PID,
            "source_manifest_sha256": repair.sha256(manifest),
            "config_sha256": repair.sha256(plan_path),
        },
    )
    write(
        batch / "controller-state.json",
        {
            "state": "needs_attention",
            "outcomes": {},
            "failures": {arm: "create failed" for arm in repair.ARMS},
        },
    )
    write(
        batch / "launch-create-diagnostic.json",
        {
            "returncode": 1,
            "model_started": False,
            "stderr": f"stat {batch / 'transformer'}: permission denied",
        },
    )
    for arm in repair.ARMS:
        (batch / arm / "evaluation").mkdir(parents=True, mode=0o700)
        (batch / arm).chmod(0o700)
        source = batch / arm / "source"
        source.mkdir(exist_ok=True)
        source.chmod(0o700)
        write(
            batch / arm / "controller-state.json",
            {
                "state": "needs_attention",
                "completed_stages": [],
                "stage": "prepare",
            },
        )
        write(
            batch / arm / "stages/prepare/transport.json",
            {
                "state": "creating",
                "container": f"{repair.BATCH}-{arm}-prepare",
            },
        )
    controller = SimpleNamespace(
        transport=SimpleNamespace(verify_snapshot=lambda *_: None),
        resources=SimpleNamespace(
            inspect_container=lambda _: None, running_containers=lambda _: []
        ),
        validate_plan=lambda _: None,
    )
    return batch, controller


def test_known_zero_update_create_failure_is_eligible(failed):
    batch, controller = failed
    evidence = repair.inspect_failure(batch, controller, is_running=lambda _: False)
    assert not evidence["model_started"]
    assert evidence["base_parameter_updates_before_repair"] == 0
    assert len(evidence["permission_targets"]) == 7


def test_live_controller_is_refused_before_any_mutation(failed):
    batch, controller = failed
    with pytest.raises(RuntimeError, match="must have exited"):
        repair.inspect_failure(batch, controller, is_running=lambda _: True)
    assert not (batch / repair.REPAIR).exists()


@pytest.mark.parametrize("artifact", ["checkpoint", "result", "container", "advanced_stage"])
def test_any_started_science_or_container_is_refused(failed, artifact):
    batch, controller = failed
    if artifact == "checkpoint":
        path = batch / "loop/training-run/training/parent.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"model")
    elif artifact == "result":
        write(batch / "loop/evaluation/result-prepare.json", {"passed": True})
    elif artifact == "container":
        controller.resources.inspect_container = lambda _: {"Id": "already-created"}
    else:
        (batch / "loop/stages/train").mkdir()
    with pytest.raises(ValueError):
        repair.inspect_failure(batch, controller, is_running=lambda _: False)


def test_permissions_only_change_seven_explicit_directories(failed):
    batch, _ = failed
    original = {
        path: stat.S_IMODE(path.stat().st_mode) for path in batch.rglob("*") if path.is_dir()
    }
    records = repair.repair_permissions(batch)
    allowed = set(repair.permission_directories(batch))
    assert len(records) == len(allowed) == 7
    for path, mode in original.items():
        assert stat.S_IMODE(path.stat().st_mode) == (0o755 if path in allowed else mode)
    assert all(
        stat.S_IMODE((batch / arm / "source").stat().st_mode) == 0o700 for arm in repair.ARMS
    )


def test_permission_targets_reject_symlink_escape(failed, tmp_path):
    batch, _ = failed
    output = batch / "loop/evaluation"
    output.rmdir()
    output.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        repair.permission_directories(batch)


def test_archive_retains_failed_claims_and_never_overwrites(failed):
    batch, _ = failed
    original = (batch / "controller-process.json").read_bytes()
    archive = repair.archive_failed_attempt(batch)
    assert (archive / "controller-process.json").read_bytes() == original
    assert all((archive / arm / "stages/prepare/transport.json").exists() for arm in repair.ARMS)
    assert all(not (batch / arm / "stages/prepare").exists() for arm in repair.ARMS)
    with pytest.raises(FileExistsError):
        repair.archive_failed_attempt(batch)
    assert (archive / "controller-process.json").read_bytes() == original
