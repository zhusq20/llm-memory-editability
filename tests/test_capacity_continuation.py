"""Continuation must retain negative calibrations and run the entire original scan."""

import json
from pathlib import Path

import pytest
from test_capacity_scaling import ROOT, script


@pytest.fixture
def continuation(tmp_path):
    module = script("continue_capacity_scaling")
    config = json.loads((ROOT / "configs/capacity-scaling-development-v1.json").read_text())
    root = tmp_path / "results"
    config.update(results_root=str(root), gpus=[0, 1], predecessors=[])
    parent_path = tmp_path / "parent.json"
    module.write(parent_path, config)
    for stage in ("calibration", "calibration-long"):
        if stage == "calibration":
            specs = [
                module.capacity.make_spec(config, f"calibration-lr{lr:g}", learning_rate=lr)
                for lr in config["calibration_learning_rates"]
            ]
        else:
            specs = [
                module.capacity.make_spec(
                    config, "calibration-long", learning_rate=0.003, epochs=3000
                )
            ]
        module.write(root / stage / "frozen-config.json", {"runs": specs})
        module.write(
            root / stage / "controller-state.json",
            {"state": "complete", "completed": [s["name"] for s in specs]},
        )
        module.write(root / stage / "tracking-completion.json", {"passed": True})
        for spec in specs:
            out = root / stage / "runs" / spec["name"]
            module.write(out / "run.json", {"spec": spec})
            module.write(
                out / "audit.json",
                {
                    "passed": True,
                    "metrics": {
                        "atomic": {"accuracy": 1.0},
                        "train_composition": {"accuracy": 1.0},
                        "II": {"accuracy": 0.0068},
                    },
                },
            )
            for filename in ("latest.pt", "world.npz", "exposures.npz"):
                (out / filename).write_bytes(b"retained scientific artifact")
    module.write(
        root / "calibration-decision.json",
        {
            "passed": False,
            "selected": str(root / "calibration-long/runs/calibration-long"),
            "recipe": {"learning_rate": 0.003, "epochs": 3000},
            "thresholds": config["prerequisite_thresholds"],
        },
    )
    config["continuation"] = {
        "prerequisite_policy": "report_only",
        "authorization": "Continue all original probes and retain all results",
        "parent_config": str(parent_path),
        "parent_config_sha256": module.digest(parent_path),
        "retained_files": {str(p): module.digest(p) for p in root.rglob("*") if p.is_file()},
    }
    return module, config


def test_negative_calibration_runs_all_six_without_retraining_or_rewriting(
    continuation, monkeypatch
):
    module, config = continuation
    called, reported = [], []
    before = {p: Path(p).read_bytes() for p in config["continuation"]["retained_files"]}

    def run_stage(config, stage, specs):
        called.append((stage, specs))
        return {s["name"]: Path(config["results_root"]) / stage / "runs" / s["name"] for s in specs}

    def report(config, paths, selected, recipe):
        reported.extend(paths)
        return {"runs": [p.name for p in paths]}

    monkeypatch.setattr(module.capacity, "run_stage", run_stage)
    monkeypatch.setattr(module.capacity, "report", report)
    module.control_locked(config)
    assert [s for s, _ in called] == ["load-scan", "fixed-target"]
    assert [s["name"] for _, group in called for s in group] == [
        "load-h1024",
        "load-h4096",
        "load-h16384",
        "background-h1024",
        "background-h4096",
        "background-h16384",
    ]
    assert all(
        s["epochs"] == 3000 and s["learning_rate"] == 0.003 for _, group in called for s in group
    )
    assert all("support_heads_n" not in s for s in called[0][1])
    assert all(s["support_heads_n"] == 256 for s in called[1][1])
    assert len(reported) == 9
    assert {p: Path(p).read_bytes() for p in before} == before
    summary = module.read(Path(config["results_root"]) / "summary.json")
    assert summary["calibration_prerequisite"]["passed"] is False
    assert (
        module.read(Path(config["results_root"]) / "controller-state.json")["state"] == "complete"
    )


def test_modified_calibration_is_rejected(continuation):
    module, config = continuation
    decision = Path(config["results_root"]) / "calibration-decision.json"
    decision.write_text('{"passed":true}')
    with pytest.raises(AssertionError, match="Retained calibration changed"):
        module.reuse_calibration(config)


def test_continuation_cannot_silently_change_scientific_recipe(continuation):
    module, config = continuation
    config["base_spec"]["width"] = 128
    with pytest.raises(AssertionError, match="base_spec"):
        module.reuse_calibration(config)
