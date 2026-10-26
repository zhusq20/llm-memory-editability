"""Standard-library launcher checks; no torch, Docker or GPU execution."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "norm_launcher", ROOT / "scripts/launch_norm_handoff.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def config():
    return launcher.read(ROOT / launcher.CONFIG)


def test_exact_four_new_runs_budget_and_unchanged_scientific_recipe():
    current = config()
    launcher.validate(current)
    baseline = launcher.read(ROOT / "configs/loop-hop-unified-v1.json")
    for key in (
        "data",
        "model",
        "training",
        "dropout_seed",
        "input_origin",
        "evaluation_nodes",
    ):
        assert current[key] == baseline[key]
    assert current["budget"]["reused_fixed_baseline_updates"] == 0
    assert current["budget"]["novel_fact_updates"] == 0
    assert len(current["arms"]) * current["training"]["steps"] == 1200000
    assert current["diagnostic"]["checkpoint_steps"] == [0, 8000, 300000]


@pytest.mark.parametrize("change", ("fixed", "scale", "budget", "handoff"))
def test_no_silent_baseline_training_or_design_change(change):
    current = config()
    if change == "fixed":
        current["arms"]["F"] = copy.deepcopy(current["arms"]["N"])
    elif change == "scale":
        current["arms"]["R"]["residual_scale"] = 0.5
    elif change == "budget":
        current["training"]["steps"] = 600000
    else:
        current["handoff"]["channel_coefficient"] = 0.25
    with pytest.raises(ValueError):
        launcher.validate(current)


def test_plan_and_run_without_execute_do_not_touch_docker_or_outputs(tmp_path):
    script = ROOT / "scripts/launch_norm_handoff.py"
    plan = subprocess.run(
        [sys.executable, str(script), "plan"], capture_output=True, text=True, check=True
    )
    assert json.loads(plan.stdout)["executed"] is False
    output = tmp_path / "untouched"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "run",
            "--bundle",
            str(tmp_path / "missing-bundle"),
            "--output",
            str(output),
            "--gpus",
            "0",
            "--docker-context",
            "missing-context",
            "--wandb-key-file",
            str(tmp_path / "missing-secret"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["executed"] is False
    assert not output.exists()


@pytest.mark.parametrize("reservation", ("compute", "device", "all", "memory"))
def test_foreign_gpu_use_and_reservations_block_admission(monkeypatch, reservation):
    def checked(argv):
        if "--query-gpu=uuid,memory.used,memory.free,utilization.gpu" in argv:
            return "GPU-0, 1024, 20000, 0" if reservation == "memory" else "GPU-0, 0, 24000, 0"
        if "--query-compute-apps=gpu_uuid,pid" in argv:
            return "GPU-0, 1234" if reservation == "compute" else ""
        if "ps" in argv:
            return "container-id"
        return json.dumps(
            [
                {
                    "Name": "/foreign",
                    "HostConfig": {
                        "DeviceRequests": [
                            {
                                "Count": -1 if reservation == "all" else 0,
                                "DeviceIDs": ["0"] if reservation == "device" else [],
                            }
                        ]
                    },
                }
            ]
        )

    monkeypatch.setattr(launcher, "checked", checked)
    with pytest.raises(RuntimeError):
        launcher.gpu_available(0, ["chosen-context"])


def fake_baseline(tmp_path):
    current = config()
    baseline, source = tmp_path / "baseline", tmp_path / "source"
    run = baseline / "runs" / current["baseline"]["arm"]
    run.mkdir(parents=True)
    files = {}
    for relative in launcher.CORE:
        for root in (baseline / "source", source):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unchanged native core\n")
        files[f"source/{relative}"] = launcher.sha(source / relative)
    prior_path = baseline / "source/configs/loop-hop-unified-v1.json"
    launcher.write(prior_path, current)
    files["source/configs/loop-hop-unified-v1.json"] = launcher.sha(prior_path)
    launcher.write(baseline / "freeze.json", {"image": current["runtime"]["image"], "files": files})
    for name in current["baseline"]["files"]:
        launcher.write(
            run / name,
            {
                "passed": True,
                "step": 300000,
                "initial_model_sha256": current["baseline"]["initial_model_sha256"],
            },
        )
    current["baseline"]["files"] = {n: launcher.sha(run / n) for n in current["baseline"]["files"]}
    return current, baseline, source


def test_baseline_source_identity_and_recipe_binding(tmp_path):
    current, baseline, source = fake_baseline(tmp_path)
    assert launcher.verify_baseline(current, baseline, source)["new_baseline_updates"] == 0
    (source / launcher.CORE[0]).write_text("changed core")
    with pytest.raises(ValueError, match="Native source"):
        launcher.verify_baseline(current, baseline, source)


def test_baseline_missing_or_changed_never_falls_back_to_training(tmp_path):
    current, baseline, source = fake_baseline(tmp_path)
    checkpoint = baseline / "runs" / current["baseline"]["arm"] / "checkpoint-0300000.pt"
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        launcher.verify_baseline(current, baseline, source)


def test_claimed_or_failed_stage_is_not_retried(tmp_path):
    args = type(
        "Args",
        (),
        {
            "bundle": tmp_path / "bundle",
            "output": tmp_path / "output",
            "docker_context": "chosen",
            "reservation_context": [],
        },
    )()
    controller = launcher.Controller(args, config())
    folder = args.output / "execution" / "N" / "train"
    folder.mkdir(parents=True)
    launcher.write(folder / "exit.json", {"code": 1})
    with pytest.raises(FileExistsError):
        controller.stage("N", "train", 0)


def test_successful_stage_is_skipped_without_model_or_docker(tmp_path):
    args = type(
        "Args",
        (),
        {
            "bundle": tmp_path / "bundle",
            "output": tmp_path / "output",
            "docker_context": "chosen",
            "reservation_context": [],
        },
    )()
    controller = launcher.Controller(args, config())
    launcher.write(args.output / "execution/N/train/exit.json", {"code": 0})
    controller.stage("N", "train", 0)


def test_all_arm_engineering_and_baseline_replay_precede_science(tmp_path, monkeypatch):
    args = type(
        "Args",
        (),
        {
            "bundle": tmp_path / "bundle",
            "output": tmp_path / "output",
            "docker_context": "chosen",
            "reservation_context": [],
            "gpus": [0],
        },
    )()
    controller = launcher.Controller(args, config())
    events = []
    monkeypatch.setattr(controller, "stage", lambda *a: events.append(a))
    monkeypatch.setattr(controller, "wave", lambda stages: events.append(stages))
    monkeypatch.setattr(launcher, "read", lambda path: {"passed": True})
    monkeypatch.setattr(launcher, "sha", lambda path: "frozen-config")
    controller.run()
    assert events.index(("all", "baseline-replay", 0)) < events.index(launcher.ENGINEERING)
    assert events.index(launcher.ENGINEERING) < events.index(launcher.SCIENCE)
    barrier = json.loads((args.output / "engineering-complete.json").read_text())
    assert barrier["arms"] == ["R", "N", "A", "C"]
    assert barrier["config_sha256"] == "frozen-config"
