"""Bounded GPU sharing rejects outside jobs and changed registered runtimes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluation_sharing",
    Path(__file__).resolve().parents[1] / "scripts/share_ttcd_evaluation_recovery.py",
)
sharing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sharing)


class Unavailable(RuntimeError):
    pass


@pytest.fixture
def registered(tmp_path, monkeypatch):
    monkeypatch.setattr(sharing, "ROOT", tmp_path)
    batch, arm, stage = "ttcd-one-step-second-hop-v1", "loop", "evaluate"
    image = "sha256:frozen"
    command = ["docker", "create", "--env", "FROZEN_ID=original", image, "python", "evaluate"]
    path = tmp_path / batch / arm / "stages" / stage / "command.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"argv": command}))
    container = {
        "Name": f"/{batch}-{arm}-{stage}",
        "Id": "original-id",
        "State": {"Pid": 41},
        "Config": {
            "User": "1015:1016",
            "Image": image,
            "Env": ["FROZEN_ID=original"],
            "Cmd": command[5:],
        },
    }
    row = {"uuid": "GPU7", "free_mib": 95000}
    processes = [{"uuid": "GPU7", "pid": 41}]
    helpers = SimpleNamespace(
        gpu_snapshot=lambda: ({7: row}, processes),
        running_containers=lambda _: [container],
        container_uses_gpu=lambda *_: True,
        docker=lambda *_: SimpleNamespace(stdout="PID\n41\n"),
        ResourceUnavailable=Unavailable,
    )
    controller = SimpleNamespace(read_json=lambda p: json.loads(p.read_text()))
    return (
        helpers,
        {batch: controller},
        {batch: {"runtime": {"image": image}}},
        container,
        row,
        processes,
    )


def test_exact_registered_container_is_admitted(registered):
    helpers, controllers, plans, container, _, _ = registered
    _, own = sharing.verify_cohort(helpers, controllers, plans)
    assert own[0]["id"] == container["Id"]


def test_outside_container_is_rejected(registered):
    helpers, controllers, plans, container, _, _ = registered
    container["Name"] = "/other-project"
    with pytest.raises(Unavailable, match="outside"):
        sharing.verify_cohort(helpers, controllers, plans)


def test_outside_cuda_process_is_rejected(registered):
    helpers, controllers, plans, _, _, processes = registered
    processes.append({"uuid": "GPU7", "pid": 99})
    with pytest.raises(Unavailable, match="outside CUDA"):
        sharing.verify_cohort(helpers, controllers, plans)


def test_changed_frozen_runtime_is_rejected(registered):
    helpers, controllers, plans, container, _, _ = registered
    container["Config"]["Env"] = ["FROZEN_ID=changed"]
    with pytest.raises(ValueError, match="identity changed"):
        sharing.verify_cohort(helpers, controllers, plans)


def test_insufficient_memory_is_rejected(registered):
    helpers, controllers, plans, _, row, _ = registered
    row["free_mib"] = 10000
    with pytest.raises(Unavailable, match="capacity"):
        sharing.verify_cohort(helpers, controllers, plans)


def test_fast_publish_keeps_existing_hardlinks_and_refuses_different_evidence(tmp_path):
    import os

    source, destination = tmp_path / "source", tmp_path / "destination"
    (source / "states").mkdir(parents=True)
    (source / "states/fast.pt").write_bytes(b"unchanged fast state")
    sharing.publish_evidence(source, destination)
    sharing.publish_evidence(source, destination)
    assert os.path.samefile(source / "states/fast.pt", destination / "states/fast.pt")
    (source / "summary.json").write_text("new")
    (destination / "summary.json").write_text("original")
    with pytest.raises(ValueError, match="different evidence"):
        sharing.publish_evidence(source, destination)
    assert (destination / "summary.json").read_text() == "original"


def test_fast_publish_never_follows_output_symlinks(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    source.joinpath("redirect").symlink_to(tmp_path, target_is_directory=True)
    sharing.publish_evidence(source, destination)
    assert not (destination / "redirect").exists()


def test_a_new_handoff_preserves_existing_monitor_log(tmp_path):
    old = tmp_path / "handoff-monitor.log"
    old.write_text("original monitor evidence")
    with sharing.monitor_log(tmp_path).open("x") as stream:
        stream.write("new amendment monitor")
    assert old.read_text() == "original monitor evidence"
    assert sharing.HANDOFF in sharing.monitor_log(tmp_path).name


def test_receipt_committed_after_full_publish_allows_new_audit_files_without_rescanning_states(
    tmp_path, monkeypatch
):
    import hashlib
    import os

    monkeypatch.setattr(sharing, "ROOT", tmp_path)
    monkeypatch.setattr(sharing, "TRANSPORT", tmp_path / "transport")
    relative = Path("ttcd-one-step-second-hop-v1/loop/evaluation")
    canonical, output = sharing.ROOT / relative, sharing.TRANSPORT / relative
    canonical.mkdir(parents=True)
    (output / "states").mkdir(parents=True)
    (output / "states/fast.pt").write_bytes(b"immutable state")
    result = output / "result-evaluate.json"
    result.write_text('{"states":1313}')
    sharing.publish_evidence(output, canonical)
    receipt = canonical.parent / "stages/evaluate/result.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"passed": True, "sha256": hashlib.sha256(result.read_bytes()).hexdigest()})
    )
    assert sharing.completed_publication(canonical, output) == (True, True)
    assert sharing.completed_publication(output, canonical) == (False, True)
    original_scandir = os.scandir

    def guarded_scandir(path):
        assert Path(path).name != "states"
        return original_scandir(path)

    monkeypatch.setattr(sharing.os, "scandir", guarded_scandir)
    (output / "result-audit.json").write_text('{"passed":true}')
    sharing.publish_evidence(output, canonical)
    assert (canonical / "result-audit.json").read_text() == '{"passed":true}'
    assert os.path.samefile(canonical / "states/fast.pt", output / "states/fast.pt")
    result.write_text("tampered")
    with pytest.raises(ValueError, match="receipt/result changed"):
        sharing.completed_publication(output, canonical)
