import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_residual_gate_injection as launcher  # noqa: E402


@pytest.fixture
def config():
    return json.loads((ROOT / launcher.CONFIG).read_text())


def test_authorized_scope(config):
    launcher.validate_authorized(config)


@pytest.mark.parametrize("changed", ["steps", "injection", "gate", "u", "extra", "context"])
def test_scope_changes_rejected(config, changed):
    config = copy.deepcopy(config)
    if changed == "steps":
        config["training"]["steps"] += 1
    elif changed == "injection":
        config["arms"]["only-u"]["injection"] = "fixed1"
    elif changed == "gate":
        config["arms"]["only-u"]["update_mode"] = "fixed"
    elif changed == "u":
        config["arms"]["only-u"]["initial_update"] = 0.9
    elif changed == "extra":
        config["arms"]["third"] = config["arms"]["only-u"]
    else:
        config["runtime"]["docker_context"] = "default"
    with pytest.raises(ValueError):
        launcher.validate_authorized(config)


def test_preflight_must_match_exact_configuration(tmp_path, config):
    source = tmp_path / "source" / launcher.CONFIG
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(config))
    output = tmp_path / "execution"
    expected = launcher.common.sha(source)
    for name in config["arms"]:
        launcher.common.write(
            output / "preflight" / f"{name}.json",
            {"passed": True, "config_sha256": expected},
        )
    launcher.verify_preflights(tmp_path, output, config)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError):
        launcher.verify_preflights(tmp_path, output, config)


def test_failed_preflight_rejected(tmp_path, config):
    source = tmp_path / "source" / launcher.CONFIG
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(config))
    output = tmp_path / "execution"
    launcher.common.write(
        output / "preflight" / "only-u.json",
        {"passed": False, "config_sha256": launcher.common.sha(source)},
    )
    with pytest.raises(ValueError):
        launcher.verify_preflights(tmp_path, output, config)


def test_command_uses_new_runner_and_readonly_source(tmp_path, config):
    (tmp_path / "freeze.json").write_text("{}")
    control = launcher.Controller(config, tmp_path, tmp_path, launcher.CONFIG, {}, Path("secret"))
    label, command, _ = control.command("only-u", "train", 0, "test-uuid")
    assert "scripts/run_residual_gate_injection.py" in command
    assert "scripts/run_paper_update_gate.py" not in command
    assert "device=test-uuid" in command
    assert f"type=bind,source={tmp_path / 'source'},target=/project,readonly" in command
    assert command[1:3] == ["--context", "lm-memory"]
    assert label.startswith(config["batch"] + "-")


def test_parallel_freeze_check_detects_changed_bytes(tmp_path, config, monkeypatch):
    data = tmp_path / "input.bin"
    data.write_bytes(b"frozen data")
    launcher.common.write(
        tmp_path / "freeze.json",
        {"image": config["runtime"]["image"], "files": {"input.bin": launcher.common.sha(data)}},
    )
    monkeypatch.setattr(
        launcher.common,
        "docker",
        lambda *args: SimpleNamespace(stdout=json.dumps([{"Id": config["runtime"]["image"]}])),
    )
    launcher.verify_frozen(tmp_path, config)
    data.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        launcher.verify_frozen(tmp_path, config)


def test_parallel_freeze_check_rejects_wrong_image(tmp_path, config, monkeypatch):
    launcher.common.write(
        tmp_path / "freeze.json", {"image": config["runtime"]["image"], "files": {}}
    )
    monkeypatch.setattr(
        launcher.common, "docker", lambda *args: SimpleNamespace(stdout='[{"Id": "wrong"}]')
    )
    with pytest.raises(ValueError, match="Docker image identity differs"):
        launcher.verify_frozen(tmp_path, config)
