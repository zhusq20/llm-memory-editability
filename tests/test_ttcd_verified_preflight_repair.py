"""A narrow, audited preflight repair cannot restart or silently alter science."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ttcd_verified_preflight_repair", ROOT / "scripts/repair_ttcd_verified_preflight.py"
)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def configuration():
    return json.loads((ROOT / "configs/ttcd-successful-base-v1.json").read_text())


@pytest.fixture
def material(tmp_path, monkeypatch):
    config, project = configuration(), tmp_path / "project"
    batch = tmp_path / repair.BATCH
    source = batch / "source"
    source.mkdir(parents=True)
    project.mkdir()
    for relative in repair.PATCHABLE_SCIENCE:
        old, new = source / relative, project / relative
        old.parent.mkdir(parents=True, exist_ok=True)
        new.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("original_reader = True\n")
        new.write_text("native_causal_attention = True\n")
    for relative in repair.ADDITIONS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# independent repair controller metadata\n")
    config_path = project / "configs/experiment.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))
    (source / "configs").mkdir()
    (source / "configs/experiment.json").write_bytes(config_path.read_bytes())
    checkpoint = repair.controller.result_host_path(batch, config["foundation"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"successful fixed300k foundation")
    gate = repair.controller.result_host_path(batch, config["foundation"]["gate_receipt"])
    gate.write_text(
        json.dumps({"passed": True, "checkpoint_sha256": repair.controller.sha256(checkpoint)})
    )
    foundation = repair.controller.verify_foundation(batch, config)
    inputs = {}
    for relative in repair.controller.raw_inputs(config) + repair.controller.prepared_inputs(
        config
    ):
        path = batch / "input-mount" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(relative).encode())
        inputs[str(relative)] = repair.controller.sha256(path)
    panel = repair.controller.result_host_path(batch, config["continual"]["panel_source_path"])
    panel.parent.mkdir(parents=True)
    panel.write_text('{"pure_graph_panel": true}\n')
    (source / "configs/continual-panel.json").write_bytes(panel.read_bytes())
    manifest = {
        "config_sha256": repair.controller.sha256(config_path),
        "foundation": foundation,
        "inputs": inputs,
        "panel_sha256": repair.controller.sha256(panel),
        "files": repair.controller.source_hashes(source),
    }
    (batch / "source-manifest.json").write_text(json.dumps(manifest))
    common = batch / "common"
    common.mkdir()
    (common / "failure-preflight.json").write_text('{"BF16_max_difference": 0.0625}')
    (common / "exit-preflight.json").write_text('{"returncode": 1}')
    (common / "claim-preflight.json").write_text('{"historical": true}')
    (batch / "controller-state.json").write_text('{"historical_failure": true}')
    evidence = tmp_path / "independent-engineering-results.json"
    evidence.write_text('{"passed": true,"scientific_updates": 0}')
    receipt_path = tmp_path / "engineering-receipt.json"
    receipt = {
        "passed": True,
        "checks": {"cpu_scientific_contracts": True, "trained_foundation_preflight": True},
        "original_source_manifest_sha256": repair.controller.sha256(batch / "source-manifest.json"),
        "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "scientific_updates": 0,
        "patched_files": {
            relative: repair.controller.sha256(project / relative)
            for relative in repair.PATCHABLE_SCIENCE
        },
        "evidence_files": {str(evidence): repair.controller.sha256(evidence)},
    }
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(repair, "ROOT", project)
    return batch, config, config_path, receipt_path, project


def test_default_plan_is_readonly_without_docker_gpu_or_output_access(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repair",
            "--config",
            str(path),
            "--engineering-receipt",
            str(tmp_path / "not-yet-passed.json"),
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not access Docker, GPUs or persistent output")

    monkeypatch.setattr(repair.controller, "persistent_batch", forbidden)
    monkeypatch.setattr(repair.subprocess, "run", forbidden)
    assert repair.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"] and not plan["gpu_sharing"] and not plan["automatic_retry"]
    assert not plan["foundation_retraining"] and not plan["configuration_changed"]
    assert plan["preflight_output"] == "common/preflight-v2"


def test_freeze_retains_old_source_config_inputs_panel_claims_and_failure(material):
    batch, config, config_path, receipt_path, project = material
    old_files = repair.controller.source_hashes(batch / "source")
    retained = {
        path: path.read_bytes()
        for path in (
            batch / "source-manifest.json",
            batch / "controller-state.json",
            batch / "common/claim-preflight.json",
            batch / "common/failure-preflight.json",
        )
    }
    frozen = repair.freeze_repair(batch, config, config_path, receipt_path)
    assert frozen["zero_updates_at_repair"] == {
        "common_updates": 0,
        "single_updates": 0,
        "stream_updates": 0,
    }
    assert repair.controller.source_hashes(batch / "source") == old_files
    assert all(path.read_bytes() == value for path, value in retained.items())
    assert (batch / "source-v2/configs/experiment.json").read_bytes() == config_path.read_bytes()
    assert "scripts/run_ttcd_continual.py" in frozen["controller_additions"]
    assert (batch / "source-v2/scripts/run_ttcd_continual.py").read_bytes() == (
        project / "scripts/run_ttcd_continual.py"
    ).read_bytes()
    assert (batch / "source-v2/configs/continual-panel.json").read_bytes() == (
        batch / "source/configs/continual-panel.json"
    ).read_bytes()
    for relative in repair.PATCHABLE_SCIENCE:
        assert (batch / "source-v2" / relative).read_bytes() == (project / relative).read_bytes()
        assert frozen["changes"][relative]["before_sha256"] == old_files[relative]
        assert Path(frozen["changes"][relative]["diff"]).is_file()
    repair.verify_repaired_source(batch, config, config_path, receipt_path)
    assert repair.freeze_repair(batch, config, config_path, receipt_path) == frozen


@pytest.mark.parametrize(
    "change",
    [
        "failed_cpu",
        "failed_preflight",
        "foundation",
        "config",
        "old_source",
        "input",
        "panel",
        "workspace",
        "evidence",
        "scope",
        "updates",
    ],
)
def test_repair_refuses_unvalidated_or_changed_lineage(material, change):
    batch, config, config_path, receipt_path, project = material
    receipt = json.loads(receipt_path.read_text())
    if change == "failed_cpu":
        receipt["checks"]["cpu_scientific_contracts"] = False
    elif change == "failed_preflight":
        receipt["checks"]["trained_foundation_preflight"] = False
    elif change == "foundation":
        repair.controller.result_host_path(batch, config["foundation"]["checkpoint"]).write_bytes(
            b"changed"
        )
    elif change == "config":
        config_path.write_text("{}")
    elif change == "old_source":
        (batch / "source" / next(iter(repair.PATCHABLE_SCIENCE))).write_text(
            "changed historical source"
        )
    elif change == "input":
        (batch / "input-mount" / repair.controller.prepared_inputs(config)[0]).write_bytes(
            b"changed"
        )
    elif change == "panel":
        repair.controller.result_host_path(
            batch, config["continual"]["panel_source_path"]
        ).write_text("{}")
    elif change == "workspace":
        (project / next(iter(repair.PATCHABLE_SCIENCE))).write_text("not the tested patch")
    elif change == "evidence":
        Path(next(iter(receipt["evidence_files"]))).write_text("unverified result")
    elif change == "scope":
        receipt["patched_files"]["configs/experiment.json"] = repair.controller.sha256(config_path)
    else:
        receipt["scientific_updates"] = 1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, RuntimeError)):
        repair.freeze_repair(batch, config, config_path, receipt_path)
    assert not (batch / "source-v2").exists()


@pytest.mark.parametrize(
    "phase,artifact",
    [("common", "training"), ("single", "claim-train.json"), ("stream", "result-train.json")],
)
def test_zero_update_guard_prevents_repeating_any_started_training(material, phase, artifact):
    batch, config, config_path, receipt_path, _ = material
    path = batch / phase / artifact
    path.parent.mkdir(exist_ok=True)
    path.write_text("already entered")
    with pytest.raises(ValueError, match="already entered"):
        repair.freeze_repair(batch, config, config_path, receipt_path)
    assert not (batch / "source-v2").exists()


def test_new_container_names_and_preflight_out_mount_only_repaired_source(material, tmp_path):
    batch, config, _, _, _ = material
    name, command = repair.build_command(
        config, "common", "preflight", batch, tmp_path / "token", {"uuid": "GPU-eight"}
    )
    assert name == "ttcd-successful-base-v1-common-preflight-v2"
    assert command[command.index("--out") + 1].endswith("common/preflight-v2")
    assert f"type=bind,source={batch / 'source-v2'},target=/frozen,readonly" in command
    assert not any(value.startswith(f"type=bind,source={batch / 'source'},") for value in command)
    assert "device=GPU-eight" in command
    name, command = repair.build_command(
        config, "stream", "train", batch, tmp_path / "token", {"uuid": "GPU-nine"}
    )
    assert name.endswith("stream-train-v2")
    assert command[command.index("--out") + 1].endswith("stream")
    assert (
        f"type=bind,source={batch / 'stream'},target=/project/results/{repair.BATCH}/stream"
        in command
    )


def test_common_failure_blocks_both_arms_without_repeating_preparation(material, monkeypatch):
    batch, config, config_path, receipt_path, _ = material
    called = []

    def run_phase(batch, config, config_path, receipt_path, phase, token):
        called.append(phase)
        return 3

    monkeypatch.setattr(repair, "run_phase", run_phase)
    assert repair.execute_pipeline(batch, config, config_path, receipt_path, Path("token")) == 3
    assert called == ["common"]
    assert (
        repair.controller.read_json(batch / "controller-state-v2.json")["paired_scientific_updates"]
        == 0
    )
    assert repair.controller.read_json(batch / "controller-state.json") == {
        "historical_failure": True
    }


def test_pair_runs_both_arms_after_common_gate_without_cancelling_other_arm(material, monkeypatch):
    batch, config, config_path, receipt_path, _ = material
    called = []
    monkeypatch.setattr(repair.controller, "require_common_gate", lambda *args: None)

    def run_phase(batch, config, config_path, receipt_path, phase, token):
        called.append(phase)
        return 1 if phase == "single" else 0

    monkeypatch.setattr(repair, "run_phase", run_phase)
    assert repair.execute_pipeline(batch, config, config_path, receipt_path, Path("token")) == 1
    assert set(called) == {"common", "single", "stream"}
    assert repair.controller.read_json(batch / "controller-state-v2.json")["exit_codes"] == {
        "single": 1,
        "stream": 0,
    }


def test_arm_cannot_bypass_common_gate_or_claim_gpu(material, monkeypatch):
    batch, config, config_path, receipt_path, _ = material

    def forbidden(*args, **kwargs):
        raise AssertionError("No GPU admission before a complete passing common gate")

    monkeypatch.setattr(repair.controller, "wait_gpu", forbidden)
    assert repair.run_phase(batch, config, config_path, receipt_path, "stream", Path("token")) == 1
    assert not (batch / "stream/claim-train.json").exists()
