import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "pending",
    Path(__file__).resolve().parents[1] / "scripts/resume_ttcd_pending_stages.py",
)
pending = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pending)


def test_direct_launch_uses_copied_transport_without_global_patch():
    calls = []
    original = subprocess.run
    helpers = SimpleNamespace(
        docker=lambda *args, **kwargs: calls.append((args, kwargs)) or "created"
    )
    proxy = pending.direct_launch_proxy(helpers)
    command = [
        "docker",
        "--context",
        "lm-memory",
        "run",
        "--name",
        "audit",
        "image",
        "python",
        "old.py",
    ]
    assert proxy.run(command, capture_output=True, text=True, timeout=60) == "created"
    assert calls == [(tuple(command[3:]), {"check": False})]
    assert subprocess.run is original
    with pytest.raises(ValueError, match="explicit LM"):
        proxy.run(["docker", "run", "image"])


def test_completed_receipts_form_a_prefix(tmp_path):
    for stage in ("train", "audit"):
        (tmp_path / f"{stage}-exit").write_text(json.dumps({"returncode": 0}))
        (tmp_path / f"{stage}-result").write_text(json.dumps({"passed": True}))
    prefix = pending.successful_prefix(
        ["train", "audit", "evaluate"],
        lambda stage: tmp_path / f"{stage}-exit",
        lambda stage: tmp_path / f"{stage}-result",
    )
    assert prefix == ["train", "audit"]
    (tmp_path / "train-exit").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        pending.successful_prefix(
            ["train", "audit"],
            lambda stage: tmp_path / f"{stage}-exit",
            lambda stage: tmp_path / f"{stage}-result",
        )


def test_failed_numerical_result_is_not_completed(tmp_path):
    (tmp_path / "exit").write_text(json.dumps({"code": 0}))
    (tmp_path / "result").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="successful result"):
        pending.successful_prefix(
            ["audit"], lambda _: tmp_path / "exit", lambda _: tmp_path / "result"
        )


@pytest.fixture
def failed(tmp_path):
    (tmp_path / "exit-train-audit.json").write_text(
        json.dumps({"returncode": 126, "launch_failed": True})
    )
    (tmp_path / "command-train-audit.json").write_text(
        json.dumps({"argv": ["--name", "audit"]})
    )
    (tmp_path / "train-audit.log").write_text(
        f"invalid mount config {pending.SPEEDUP / 'source'} permission denied"
    )
    helpers = SimpleNamespace(
        read_json=lambda path: json.loads(path.read_text()),
        inspect_container=lambda _: None,
    )
    return tmp_path, helpers


def test_only_diagnosed_pre_execution_failure_is_recoverable(failed):
    out, helpers = failed
    files = pending.failed_launch_files(out, helpers)
    assert len(files) == 5 and all("train-audit" in file.name for file in files)
    helpers.inspect_container = lambda _: {"State": {"ExitCode": 126}}
    with pytest.raises(ValueError, match="created a model"):
        pending.failed_launch_files(out, helpers)


def test_executed_audit_is_never_retried(failed):
    out, helpers = failed
    (out / "result-train-audit.json").write_text("{}")
    with pytest.raises(ValueError, match="executed scientific audit"):
        pending.failed_launch_files(out, helpers)


def test_other_launch_failure_is_not_retried(failed):
    out, helpers = failed
    (out / "exit-train-audit.json").write_text(
        json.dumps({"returncode": 1, "launch_failed": True})
    )
    with pytest.raises(ValueError, match="diagnosed"):
        pending.failed_launch_files(out, helpers)


def test_known_import_failure_is_preserved_before_retry(failed):
    out, helpers = failed
    (out / "exit-train-audit.json").write_text(json.dumps({"returncode": 1}))
    (out / "train-audit.log").write_text(
        "eval_module = load_extension\nImportError: cannot import name 'ttcd_double_new_eval'"
    )
    (out / "runtime-acceleration-train-audit.json").write_text("{}")
    helpers.inspect_container = lambda _: {"State": {"Status": "exited", "ExitCode": 1}}
    files = pending.failed_launch_files(out, helpers)
    assert len(files) == 6 and files[-1].name == "runtime-acceleration-train-audit.json"


def test_exclusive_fallback_uses_idle_card_and_preserves_original_plan(
    tmp_path, monkeypatch
):
    class ResourceUnavailable(Exception):
        pass

    plan = {"runtime": {"gpus": {"loop": 9}}}
    ctx = SimpleNamespace(
        config=plan,
        helpers=SimpleNamespace(
            gpu_snapshot=lambda: ({4: {}, 8: {}, 9: {}}, []),
            ResourceUnavailable=ResourceUnavailable,
        ),
    )
    seen = []

    def admit(actual, arm):
        gpu = actual["runtime"]["gpus"][arm]
        seen.append(gpu)
        if gpu != 8:
            raise ResourceUnavailable("CUDA process still occupies card")
        return {"gpu": gpu, "uuid": "GPU8"}

    monkeypatch.setattr(pending.os, "getuid", lambda: f"pending-test-{tmp_path.name}")
    held, row = pending.exclusive_choice(
        ctx, "loop", time.monotonic() + 1, lambda _: None, admit
    )
    with held:
        assert row["gpu"] == 8 and seen == [9, 4, 8]
        assert plan["runtime"]["gpus"]["loop"] == 9


def test_no_idle_card_keeps_waiting_without_launching(tmp_path, monkeypatch):
    class ResourceUnavailable(Exception):
        pass

    now = [0.0]
    ctx = SimpleNamespace(
        config={"runtime": {"gpus": {"loop": 9}}},
        helpers=SimpleNamespace(
            gpu_snapshot=lambda: ({8: {}, 9: {}}, []),
            ResourceUnavailable=ResourceUnavailable,
        ),
    )
    events = []

    def occupied(*_):
        raise ResourceUnavailable("occupied")

    monkeypatch.setattr(pending.os, "getuid", lambda: f"pending-test-{tmp_path.name}")
    monkeypatch.setattr(pending.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        pending.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    with pytest.raises(TimeoutError, match="wait expired"):
        pending.exclusive_choice(ctx, "loop", 1, events.append, occupied)
    assert events[0]["state"] == "waiting_for_resources"
    assert set(events[0]["reasons"]) == {"8", "9"}


def test_new_parent_work_waits_when_external_cuda_process_arrives(monkeypatch):
    class ResourceUnavailable(Exception):
        pass

    now, events = [0.0], []

    def admission(*_):
        raise ResourceUnavailable(
            "GPU4 has CUDA processes outside the authorized cohort"
        )

    ctx = SimpleNamespace(
        policy=SimpleNamespace(admission=admission),
        helpers=SimpleNamespace(ResourceUnavailable=ResourceUnavailable),
        config={"runtime": {}},
    )
    monkeypatch.setattr(pending.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        pending.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    with pytest.raises(TimeoutError, match="parent phase deadline"):
        pending.wait_parent_admission(ctx, "single", 1, events.append)
    assert events[0]["state"] == "waiting_for_resources"
    assert "outside the authorized cohort" in events[0]["reason"]
