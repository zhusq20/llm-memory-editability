import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "gpu4_dispatcher", Path(__file__).resolve().parents[1] / "scripts/resume_norm_handoff_gpu4.py"
)
dispatcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatcher)


def test_gpu4_does_not_wait_for_old_gpu12_queue():
    def old(*args):
        raise AssertionError("GPU4 has no old lane")

    module = SimpleNamespace(prior_lane_released=old, gpu_available=lambda *args: "GPU4")
    assert dispatcher.eligible(module, 4, None, ["lm-memory", "default"])


@pytest.mark.parametrize("gpu", (1, 2))
def test_old_queue_remains_reserved_even_during_idle_gaps(gpu):
    module = SimpleNamespace(prior_lane_released=lambda *args: False)
    assert not dispatcher.eligible(module, gpu, None, [])


def test_busy_gpu4_is_not_claimed():
    def busy(*args):
        raise RuntimeError("foreign process")

    assert not dispatcher.eligible(SimpleNamespace(gpu_available=busy), 4, None, [])


def test_failed_old_audit_is_not_treated_as_release():
    def failed(*args):
        raise RuntimeError("audit failed")

    with pytest.raises(RuntimeError, match="audit failed"):
        dispatcher.eligible(SimpleNamespace(prior_lane_released=failed), 1, None, [])


@pytest.mark.parametrize("fail", (False, True))
def test_single_free_gpu_runs_each_arm_once_and_never_requeues_failure(monkeypatch, tmp_path, fail):
    monkeypatch.setattr(dispatcher, "eligible", lambda module, gpu, *args: gpu == 4)
    seen = []

    def stage(arm, name, gpu):
        seen.append((arm, name, gpu))
        if fail:
            raise RuntimeError("preserve failed stage")

    controller = SimpleNamespace(
        config={"arms": dict.fromkeys(("R", "N", "A", "C"))},
        contexts=[],
        output=tmp_path,
        stage=stage,
    )
    module = SimpleNamespace(
        time=SimpleNamespace(sleep=lambda seconds: time.sleep(0.001)),
        write=lambda *args: None,
        utc=lambda: "test",
    )
    if fail:
        with pytest.raises(RuntimeError, match="preserve failed"):
            dispatcher.dynamic_wave(controller, ("preflight", "smoke"), module, None)
        assert seen == [("R", "preflight", 4)]
    else:
        dispatcher.dynamic_wave(controller, ("preflight", "smoke"), module, None)
        assert seen == [
            (arm, stage, 4) for arm in ("R", "N", "A", "C") for stage in ("preflight", "smoke")
        ]
