"""Scheduling, gate isolation and immutable transport for the six ablation arms."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ablation_controller", ROOT / "scripts/launch_ce_cross_sample_ablation.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.transport, "INPUT_ROOT", tmp_path)
    parent, repository = tmp_path / "parent", tmp_path / "repository"
    parent.mkdir()
    repository.mkdir()
    (parent / "old.py").write_text("original science\n")
    controller.write(
        parent / "source-manifest.json", {"files": {"old.py": controller.sha(parent / "old.py")}}
    )
    extras = [
        "scripts/launch_ce_cross_sample_ablation.py",
        "scripts/run_ce_cross_sample_ablation.py",
        "scripts/report_ce_cross_sample_ablation.py",
        f"configs/{controller.BATCH}.json",
        f"docs/development-artifacts/{controller.BATCH}/protocol.md",
        "tests/test_ce_cross_sample_ablation_controller.py",
    ]
    for relative in extras:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new ablation file\n")
    batch = tmp_path / controller.BATCH
    recipe = {
        "batch": controller.BATCH,
        "arms": {arm: {"architecture": arm.split("-", 1)[0]} for arm in controller.ARMS},
        "execution": {
            "parent_source": str(parent),
            "parent_source_manifest_sha256": controller.sha(parent / "source-manifest.json"),
            "parent_source_file_count": 1,
            "extra_files": extras,
            "persistent_root": str(batch),
            "image": controller.IMAGE,
            "timeout_seconds_per_arm": 100,
            "resources": {
                "memory": "48g",
                "cpu_sets": {
                    **{arm: f"{80 + 4 * i}-{83 + 4 * i}" for i, arm in enumerate(controller.ARMS)},
                    "loop-prepare": "80-83",
                    "discoloop-prepare": "92-95",
                },
            },
        },
    }
    return recipe, repository, batch


def test_matrix_has_exactly_six_single_variable_arms(plan):
    recipe, _, _ = plan
    assert len(controller.arms_for(recipe)) == 6
    del recipe["arms"]["loop-no-bound"]
    with pytest.raises(ValueError, match="six authorized"):
        controller.arms_for(recipe)


def test_arm_architecture_cannot_change(plan):
    recipe, _, _ = plan
    recipe["arms"]["loop-no-bound"]["architecture"] = "discoloop"
    with pytest.raises(ValueError, match="does not match"):
        controller.arms_for(recipe)


def test_freeze_reuses_full_parent_without_overwriting_science(plan):
    recipe, repository, batch = plan
    source, manifest = controller.transport.freeze(recipe, batch, repository=repository)
    assert (source / "old.py").read_text() == "original science\n"
    assert controller.transport.verify_source(source, manifest) == manifest
    assert controller.transport.freeze(recipe, batch, repository=repository) == (source, manifest)
    recipe["execution"]["extra_files"].append("old.py")
    recipe["execution"]["source_revision"] = "source-v2"
    with pytest.raises(ValueError):
        controller.transport.freeze(recipe, batch, repository=repository)


def test_transport_permissions_and_traversable_parent_mount(plan):
    recipe, repository, batch = plan
    source, manifest = controller.transport.freeze(recipe, batch, repository=repository)
    with tarfile.open(fileobj=io.BytesIO(controller.transport.archive(source))) as archive:
        for member in archive:
            assert (member.uid, member.gid) == (0, 0)
            assert member.mode == (0o555 if member.isdir() else 0o444)
    _, arguments = controller.transport.container_arguments(
        recipe, batch, manifest, "loop-no-bound", "science", {"uuid": "GPU-4"}
    )
    assert f"type=bind,source={controller.transport.INPUT_ROOT},target=/persist" in arguments
    assert (
        f"type=bind,source={controller.transport.INPUT_ROOT},target={controller.transport.INPUT_ROOT},readonly"
        in arguments
    )
    assert (
        f"type=bind,source={controller.transport.TOKEN},target=/run/secrets/wandb,readonly"
        in arguments
    )
    assert not any(value.startswith("type=volume") or "target=/out" in value for value in arguments)
    assert "--read-only" not in arguments
    assert arguments[arguments.index("--user") + 1] == "1015:1016"
    assert arguments[arguments.index("--memory") + 1] == "48g"


@pytest.mark.parametrize("architecture", controller.ARCHITECTURES)
def test_prepare_has_architecture_without_arm(plan, architecture):
    recipe, _, _ = plan
    command = controller.command_for(recipe, "prepare", f"{architecture}-prepare")
    assert command[command.index("--architecture") + 1] == architecture
    assert "--arm" not in command
    assert (
        command[command.index("--root") + 1]
        == f"/persist/{controller.BATCH}/{architecture}-prepare/output"
    )


@pytest.mark.parametrize("stage", ["gate", "audit-gate", "science", "audit", "cloud"])
def test_arm_interface_and_devices(plan, stage):
    recipe, _, _ = plan
    command = controller.command_for(recipe, stage, "discoloop-carry-adam")
    assert command[:3] == ["python", "scripts/run_ce_cross_sample_ablation.py", stage]
    assert command[command.index("--arm") + 1] == "discoloop-carry-adam"
    assert command[-1] == ("cpu" if stage == "cloud" else "cuda:0")


def test_report_writes_only_own_runtime_batch(plan):
    recipe, _, _ = plan
    command = controller.command_for(recipe, "report", "report")
    assert command[command.index("--batch") + 1] == f"/persist/{controller.BATCH}"
    assert command[command.index("--out") + 1] == f"/persist/{controller.BATCH}/report/output"


@pytest.mark.parametrize("arm", ["../old-batch", "/old", "transformer-no-bound", "loop"])
def test_output_cannot_escape_or_address_old_batch(plan, arm):
    recipe, _, batch = plan
    with pytest.raises(ValueError, match="escaped"):
        controller.private_output(batch, arm)
    with pytest.raises(ValueError, match="escaped"):
        controller.command_for(recipe, "science", arm)


def test_symlink_cannot_redirect_output(plan, tmp_path):
    _, _, batch = plan
    batch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (batch / "loop-no-bound").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink escaped"):
        controller.private_output(batch, "loop-no-bound")


class FakeLock:
    def __init__(self, events, gpu):
        self.events, self.gpu = events, gpu

    def close(self):
        self.events.append(("released", self.gpu))


def test_each_stage_obtains_a_new_dynamic_gpu_lease(plan, monkeypatch):
    recipe, _, batch = plan
    events, assignments = [], iter([{"uuid": "GPU-4"}, {"uuid": "GPU-6"}])

    def acquire(*args):
        gpu = next(assignments)
        events.append(("acquired", gpu["uuid"]))
        return gpu, FakeLock(events, gpu["uuid"])

    def stage(*args, **kwargs):
        events.append(("running", kwargs["gpu"]["uuid"]))
        return 10

    monkeypatch.setattr(controller.transport, "acquire_gpu", acquire)
    monkeypatch.setattr(controller.transport, "run_stage", stage)
    for stage in ("gate", "audit-gate"):
        assert (
            controller.gpu_stage(recipe, batch, Path("source"), "hash", "loop-no-bound", stage, 100)
            == 10
        )
    assert events == [
        ("acquired", "GPU-4"),
        ("running", "GPU-4"),
        ("released", "GPU-4"),
        ("acquired", "GPU-6"),
        ("running", "GPU-6"),
        ("released", "GPU-6"),
    ]
    history = json.loads((batch / "loop-no-bound/resource-history.json").read_text())
    assert history[-1]["previous_gpu_uuid"] == "GPU-4"
    assert history[-1]["gpu"]["uuid"] == "GPU-6"
    amendment = json.loads((batch / "loop-no-bound/resource-amendment-audit-gate.json").read_text())
    assert amendment["previous_gpu_uuid"] == "GPU-4"
    assert amendment["gpu"]["uuid"] == "GPU-6"


def mock_results(monkeypatch, *, eligible=True, gate_audit=True, cloud=True):
    results = {
        "gate": {"passed": True, "eligible": eligible, "development_optimizer_updates": 640},
        "science": {"scientific_optimizer_updates": 2560},
        "audit-gate": {"passed": gate_audit},
        "audit": {"passed": True},
        "cloud": {"passed": cloud},
    }
    monkeypatch.setattr(controller, "stage_result", lambda batch, arm, stage: results[stage])
    return results


def test_ineligible_gate_is_audited_and_cloud_verified_without_science(plan, monkeypatch):
    recipe, _, batch = plan
    stages = []
    mock_results(monkeypatch, eligible=False)
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: stages.append(args[5]) or 10)
    monkeypatch.setattr(
        controller.transport, "run_stage", lambda *args, **kwargs: stages.append(args[5]) or 1
    )
    result = controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-no-bound")
    assert stages == ["gate", "audit-gate", "cloud"]
    assert result["state"] == "ineligible"
    assert result["passed"] is True
    assert result["scientific_updates"] == 0
    assert result["development_optimizer_updates"] == 640
    assert result["optimizer_updates"] == 640


def test_eligible_arm_preserves_total_gpu_budget_and_clears_lease_for_cloud(plan, monkeypatch):
    recipe, _, batch = plan
    stages = []
    mock_results(monkeypatch)
    monkeypatch.setattr(
        controller, "gpu_stage", lambda *args: stages.append((args[5], args[6])) or 10
    )
    monkeypatch.setattr(
        controller.transport,
        "run_stage",
        lambda *args, **kwargs: stages.append((args[5], kwargs.get("gpu"))) or 1,
    )
    result = controller.execute_arm(recipe, batch, Path("source"), "hash", "discoloop-carry-adam")
    assert stages == [
        ("gate", 100),
        ("audit-gate", 90),
        ("science", 80),
        ("audit", 70),
        ("cloud", None),
    ]
    assert result["scientific_updates"] == 2560
    assert result["development_optimizer_updates"] == 640
    assert result["optimizer_updates"] == 3200
    assert result["full_cloud_json_verified"] is True


def test_failed_gate_replay_prevents_science(plan, monkeypatch):
    recipe, _, batch = plan
    stages = []
    mock_results(monkeypatch, gate_audit=False)
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: stages.append(args[5]) or 10)
    with pytest.raises(ValueError, match="gate replay"):
        controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-carry-adam")
    assert stages == ["gate", "audit-gate"]
    state = json.loads((batch / "loop-carry-adam/controller-state.json").read_text())
    assert state["automatic_retry"] is False
    assert state["state"] == "needs_attention"


def test_missing_explicit_eligibility_cannot_start_science(plan, monkeypatch):
    recipe, _, batch = plan
    results = mock_results(monkeypatch)
    results["gate"] = {"passed": True}
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: 1)
    with pytest.raises(ValueError, match="explicitly declare"):
        controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-lower-lr")


def test_cloud_failure_cannot_mark_arm_complete(plan, monkeypatch):
    recipe, _, batch = plan
    mock_results(monkeypatch, eligible=False, cloud=False)
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: 1)
    monkeypatch.setattr(controller.transport, "run_stage", lambda *args, **kwargs: 1)
    with pytest.raises(ValueError, match="cloud audit"):
        controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-lower-lr")
    assert not (batch / "loop-lower-lr/completion-manifest.json").exists()


def test_existing_failed_science_stage_is_never_repeated(plan, monkeypatch):
    recipe, _, batch = plan
    directory = batch / "loop-no-bound"
    controller.write(directory / "launch-science.json", {"source_sha256": "hash"})
    controller.write(directory / "exit-science.json", {"ExitCode": 1})
    monkeypatch.setattr(
        controller, "lease_gpu", lambda *args: pytest.fail("must not acquire a GPU")
    )
    with pytest.raises(FileExistsError, match="never repeat"):
        controller.gpu_stage(recipe, batch, Path("source"), "hash", "loop-no-bound", "science", 100)


def test_gpu_lease_released_on_stage_failure(plan, monkeypatch):
    recipe, _, batch = plan
    events = []
    monkeypatch.setattr(
        controller, "lease_gpu", lambda *args: ({"uuid": "GPU-4"}, FakeLock(events, "GPU-4"))
    )

    def failed(*args, **kwargs):
        raise RuntimeError("preserved failure")

    monkeypatch.setattr(controller.transport, "run_stage", failed)
    with pytest.raises(RuntimeError, match="preserved failure"):
        controller.gpu_stage(recipe, batch, Path("source"), "hash", "loop-no-bound", "science", 100)
    assert events == [("released", "GPU-4")]


def test_second_lower_lr_candidate_is_counted_in_development_budget(plan, monkeypatch):
    recipe, _, batch = plan
    results = mock_results(monkeypatch)
    results["gate"]["development_optimizer_updates"] = 1280
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: 1)
    monkeypatch.setattr(controller.transport, "run_stage", lambda *args, **kwargs: 1)
    result = controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-lower-lr")
    assert result["development_optimizer_updates"] == 1280
    assert result["optimizer_updates"] == 3840


def test_non_lr_arm_cannot_add_a_second_gate_candidate(plan, monkeypatch):
    recipe, _, batch = plan
    results = mock_results(monkeypatch)
    results["gate"]["development_optimizer_updates"] = 1280
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: 1)
    with pytest.raises(ValueError, match="candidate budget"):
        controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-carry-adam")


def test_incomplete_science_budget_cannot_be_marked_complete(plan, monkeypatch):
    recipe, _, batch = plan
    results = mock_results(monkeypatch)
    results["science"]["scientific_optimizer_updates"] = 2540
    monkeypatch.setattr(controller, "gpu_stage", lambda *args: 1)
    with pytest.raises(ValueError, match="authorized 2560"):
        controller.execute_arm(recipe, batch, Path("source"), "hash", "loop-carry-adam")
