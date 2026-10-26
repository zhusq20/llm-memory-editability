"""Direct-launch resource admission and scope; no Docker calls or training."""

import importlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
launcher = importlib.import_module("launch_paper_update_gate")


def rows():
    return {
        gpu: {"uuid": f"GPU-{gpu}", "used": 0, "free": 80000, "utilization": 0} for gpu in (1, 2)
    }


def test_admission_requires_two_distinct_idle_unreserved_gpus():
    assert launcher.admit([1, 2], rows(), {1: [], 2: []}) == {1: "GPU-1", 2: "GPU-2"}
    with pytest.raises(ValueError):
        launcher.admit([1, 1], rows(), {1: [], 2: []})
    with pytest.raises(ValueError, match="occupied or reserved"):
        launcher.admit([1, 2], rows(), {1: ["foreign task"], 2: []})


@pytest.mark.parametrize(("key", "value"), [("used", 513), ("utilization", 6), ("free", 17999)])
def test_admission_rejects_live_use_or_insufficient_memory(key, value):
    observed = rows()
    observed[2][key] = value
    with pytest.raises(ValueError):
        launcher.admit([1, 2], observed, {1: [], 2: []})


@pytest.fixture
def controller(tmp_path):
    config = {
        "batch": "paper-update-launch-test",
        "arms": {"fixed-update": {}, "token-update": {}},
        "runtime": {"memory_gib": 16, "image": "sha256:test", "gpu_memory_fraction": 0.16},
    }
    (tmp_path / "freeze.json").write_text(json.dumps({"image": "sha256:test", "files": {}}))
    return launcher.Controller(
        config,
        tmp_path,
        tmp_path / "mirror",
        "configs/selection.json",
        {1: "GPU-1", 2: "GPU-2"},
        tmp_path / "private-token",
    )


def test_container_command_pins_gpu_and_keeps_private_content_out_of_arguments(controller):
    label, command, cpus = controller.command("fixed-update", "train", 1, "GPU-1")
    assert command[:4] == ["docker", "--context", "lm-memory", "create"]
    assert command[command.index("--gpus") + 1] == "device=GPU-1"
    assert command[command.index("--cpuset-cpus") + 1] == ",".join(map(str, cpus))
    assert len(cpus) == 2 and label.endswith("fixed-update-train")
    assert "sha256:test" in command
    assert "configs/selection.json" in command
    assert str(controller.token_file) not in " ".join(command)
    assert "scripts/run_paper_update_gate.py" in command
    assert any("target=/project,readonly" in part for part in command)
    assert any(f"target={launcher.DATA_TARGET},readonly" in part for part in command)
    assert not any(
        stage in command for stage in ("preflight", "smoke", "audit", "grid-audit", "cloud")
    )


@pytest.mark.parametrize("skip_grid", (False, True))
def test_worker_only_runs_user_requested_training_and_optional_grid(
    controller, monkeypatch, skip_grid
):
    observed = []
    controller.skip_grid = skip_grid
    monkeypatch.setattr(controller, "execute", lambda *args: observed.append(args))
    controller.worker("fixed-update", 1, "GPU-1")
    assert [stage for _, stage, _, _ in observed] == (["train"] if skip_grid else ["train", "grid"])
    assert all(gpu == 1 and uuid == "GPU-1" for _, _, gpu, uuid in observed)
    assert (
        controller.state["arms"]["fixed-update"]["state"] == "training_complete_audit_not_requested"
    )


def test_failed_arm_is_not_retried_and_other_arm_finishes(controller, monkeypatch):
    observed = []

    def execute(name, stage, gpu, uuid):
        observed.append((name, stage, gpu, uuid))
        if name == "fixed-update":
            raise RuntimeError("recorded stage failure")

    monkeypatch.setattr(controller, "execute", execute)
    monkeypatch.setattr(controller, "mirror_output", lambda force=False: None)
    assert controller.run() is False
    assert [(name, stage) for name, stage, _, _ in observed].count(("fixed-update", "train")) == 1
    assert ("fixed-update", "grid") not in [(name, stage) for name, stage, _, _ in observed]
    assert (
        controller.state["arms"]["token-update"]["state"] == "training_complete_audit_not_requested"
    )
    assert controller.state["state"] == "needs_attention"
    assigned = {name: gpu for name, _, gpu, _ in observed}
    assert assigned == {"fixed-update": 1, "token-update": 2}


def test_private_credential_archive_checks_permissions(controller):
    path = controller.token_file
    path.write_bytes(b"synthetic-unit-test-value\n")
    path.chmod(0o600)
    with tarfile.open(fileobj=io.BytesIO(launcher.credential_archive(path))) as archive:
        member = archive.getmember("run/secrets/wandb")
        assert member.mode == 0o400
        assert archive.extractfile(member).read() == b"synthetic-unit-test-value"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private W&B credential"):
        launcher.credential_archive(path)


def test_source_archive_preserves_bytes_and_is_mirrored_as_one_file(tmp_path):
    source, mirror = tmp_path / "source", tmp_path / "mirror"
    source.mkdir()
    mirror.mkdir()
    (source / "config.json").write_bytes(b'{"frozen":true}\n')
    (source / "subdir").mkdir()
    (source / "subdir/code.py").write_bytes(b"value = 1\n")
    receipt = launcher.archive_source(tmp_path, mirror)
    assert receipt["sha256"] == launcher.sha(mirror / "source.tar.gz")
    assert sorted(path.name for path in mirror.iterdir()) == ["source.tar.gz"]
    with tarfile.open(mirror / "source.tar.gz") as archive:
        assert archive.extractfile("source/config.json").read() == b'{"frozen":true}\n'
        assert archive.extractfile("source/subdir/code.py").read() == b"value = 1\n"
    with pytest.raises(FileExistsError):
        launcher.archive_source(tmp_path, mirror)
