"""The stage boundary must not tune on main outcomes or lose a candidate."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "architecture_controller",
    Path(__file__).resolve().parents[1] / "scripts/execute_parametric_architecture.py",
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def fixture_config(tmp_path):
    root = tmp_path / "development"
    root.mkdir()
    main_data = tmp_path / "main.json"
    controller.write(main_data, {"sentinel": "never used for LR selection"})
    old = tmp_path / "historical.json"
    controller.write(
        old,
        [{"step": 32000, "metrics": {"atomic": {"nll": 1.0}, "train_composition": {"nll": 1.0}}}],
    )
    config = {
        "repository": str(tmp_path),
        "tracking": {},
        "runs": [],
        "transition": {
            "batch": "main",
            "results_root": str(tmp_path / "main"),
            "data_file": str(main_data),
            "data_sha256": controller.digest(main_data),
            "historical_reuse_verified": True,
            "selection_rule": "fixed 32k fitting NLL",
            "historical_development": {
                "path": str(old),
                "sha256": controller.digest(old),
                "selection_score": 1.0,
            },
        },
    }
    for arm in ["M8", "W8", "IHC8", "HC8", "D8"]:
        for lr in [1e-4] if arm == "D8" else [5e-5, 1e-4]:
            name = f"{arm}-{lr}"
            config["runs"].append(
                {
                    "name": name,
                    "architecture": arm,
                    "learning_rate": lr,
                    "steps": 32000,
                    "microbatch_size": 128,
                }
            )
            out = root / "runs" / name
            controller.write(out / "audit.json", {"passed": True})
            score = 0.5 if lr == 5e-5 else 2.0
            controller.write(
                out / "learning.json",
                [
                    {
                        "step": 32000,
                        "metrics": {
                            "atomic": {"nll": score},
                            "train_composition": {"nll": score},
                            "test_oo": {"alias_em": 0.0 if lr == 5e-5 else 1.0},
                        },
                    }
                ],
            )
    controller.write(root / "frozen-config.json", config)
    return config, root


def test_selection_uses_fitting_panels_despite_opposite_oo_scores(tmp_path):
    config, root = fixture_config(tmp_path)
    main = controller.read(controller.select_main(config, root))
    assert len(main["runs"]) == 12
    assert all(s["learning_rate"] == 5e-5 for s in main["runs"])
    assert {s["initialization"] for s in main["runs"]} == {811201, 811202, 811203}
    assert all(s["steps"] == 300000 for s in main["runs"])


def test_changed_dense_recipe_gets_three_new_baselines(tmp_path):
    config, root = fixture_config(tmp_path)
    p = root / "runs/D8-0.0001/learning.json"
    value = controller.read(p)
    value[0]["metrics"]["atomic"]["nll"] = 0.0
    value[0]["metrics"]["train_composition"]["nll"] = 0.0
    controller.write(p, value)
    main = controller.read(controller.select_main(config, root))
    assert len(main["runs"]) == 15
    assert len([s for s in main["runs"] if s["architecture"] == "D8"]) == 3


@pytest.mark.parametrize("failure", ["missing", "unaudited", "wrong_node", "changed_history"])
def test_stage_gate_rejects_incomplete_or_changed_evidence(tmp_path, failure):
    config, root = fixture_config(tmp_path)
    if failure == "missing":
        config["runs"].pop()
    elif failure == "unaudited":
        controller.write(root / "runs/M8-5e-05/audit.json", {"passed": False})
    elif failure == "wrong_node":
        p = root / "runs/M8-5e-05/learning.json"
        records = controller.read(p)
        records[-1]["step"] = 16000
        controller.write(p, records)
    else:
        Path(config["transition"]["historical_development"]["path"]).write_text(json.dumps([]))
    with pytest.raises(AssertionError):
        controller.select_main(config, root)
    assert not (tmp_path / "docs/development-artifacts/main/frozen-config.json").exists()
