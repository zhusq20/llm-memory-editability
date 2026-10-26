"""Do not take over a GPU during gaps in an existing scientific queue."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "local_launcher", ROOT / "scripts/launch_norm_handoff.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_active_prior_lane_is_not_released(tmp_path):
    launcher.write(tmp_path / "handoff/lane-1.json", {"state": "running"})
    assert not launcher.prior_lane_released(tmp_path, 1)


def test_missing_prior_receipt_stops_instead_of_taking_over(tmp_path):
    with pytest.raises(FileNotFoundError):
        launcher.prior_lane_released(tmp_path, 1)


@pytest.mark.parametrize("failure", ("exit", "audit", "cloud", "none"))
def test_takeover_requires_finished_queue_and_both_audits(tmp_path, failure):
    launcher.write(
        tmp_path / "handoff/lane-1.json",
        {"state": "finished", "outcomes": {"old": 1 if failure == "exit" else 0}},
    )
    for name, kind in (("result-audit.json", "audit"), ("result-cloud-audit.json", "cloud")):
        launcher.write(tmp_path / "jobs/old" / name, {"passed": failure != kind})
    if failure == "none":
        assert launcher.prior_lane_released(tmp_path, 1)
    else:
        with pytest.raises(RuntimeError):
            launcher.prior_lane_released(tmp_path, 1)


@pytest.mark.parametrize("tamper", (None, "hostname", "bundle_sha256", "launcher_sha256", "gpus"))
def test_runtime_amendment_binds_launch_without_changing_frozen_config(tmp_path, tamper):
    bundle = tmp_path / "bundle"
    launcher.write(bundle / "bundle.json", {"test": True})
    args = SimpleNamespace(
        bundle=bundle,
        gpus=[1, 2],
        wait_for_prior_batch=tmp_path / "prior",
        runtime_amendment=tmp_path / "amendment.json",
    )
    config = {"runtime": {"image": "sha256:" + "a" * 64}}
    amendment = {
        "hostname": launcher.platform.node(),
        "bundle_sha256": launcher.sha(bundle / "bundle.json"),
        "original_image": config["runtime"]["image"],
        "actual_image": "sha256:" + "b" * 64,
        "gpus": [1, 2],
        "launcher_sha256": launcher.sha(ROOT / "scripts/launch_norm_handoff.py"),
        "prior_batch": str(args.wait_for_prior_batch.resolve()),
    }
    if tamper:
        amendment[tamper] = "changed"
    launcher.write(args.runtime_amendment, amendment)
    if tamper:
        with pytest.raises(ValueError):
            launcher.effective_runtime(args, config)
    else:
        effective = launcher.effective_runtime(args, config)
        assert effective["runtime"]["image"] == amendment["actual_image"]
        assert config["runtime"]["image"] == amendment["original_image"]
