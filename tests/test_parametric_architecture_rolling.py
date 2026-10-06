"""Early runs must equal the original final selector, with identical evidence gates."""

import importlib.util
from pathlib import Path

import pytest
from test_parametric_architecture_controller import controller, fixture_config

SPEC = importlib.util.spec_from_file_location(
    "rolling", Path(__file__).parents[1] / "scripts/roll_parametric_architecture.py"
)
rolling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rolling)


def setup(tmp_path, monkeypatch):
    config, root = fixture_config(tmp_path)
    monkeypatch.setattr(rolling, "DEV", root)
    monkeypatch.setattr(rolling, "ART", tmp_path / "rolling")
    return config, root


def test_early_specs_equal_final_and_ignore_incomplete_unrelated_arm(tmp_path, monkeypatch):
    config, root = setup(tmp_path, monkeypatch)
    missing = root / "runs/HC8-0.0001/audit.json"
    missing.unlink()
    early, decision = rolling.early_config(controller, config)
    assert len(early["runs"]) == 3
    assert all(s["learning_rate"] == 5e-5 for s in early["runs"])
    assert set(decision["choices"]) == {"M8", "W8", "IHC8"}
    controller.write(missing, {"passed": True})
    final = controller.read(controller.select_main(config, root))
    for spec in early["runs"]:
        assert spec == next(s for s in final["runs"] if s["name"] == spec["name"])
    for key in ("data_file", "data_sha256", "evaluation_nodes", "reporting"):
        assert early[key] == final[key]


@pytest.mark.parametrize("failure", ["unaudited", "wrong_node", "nonfinite"])
def test_early_rejects_bad_candidate(tmp_path, monkeypatch, failure):
    config, root = setup(tmp_path, monkeypatch)
    out = root / "runs/M8-0.0001"
    if failure == "unaudited":
        controller.write(out / "audit.json", {"passed": False})
    else:
        record = controller.read(out / "learning.json")
        if failure == "wrong_node":
            record[-1]["step"] = 16000
        else:
            record[-1]["metrics"]["atomic"]["nll"] = float("nan")
        (out / "learning.json").write_text(__import__("json").dumps(record))
    with pytest.raises(AssertionError):
        rolling.early_config(controller, config)
