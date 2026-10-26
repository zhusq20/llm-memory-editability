import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "factorial_readiness", ROOT / "scripts/start_knowledge_method_factorial_when_ready.py"
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def ready(tmp_path):
    root = tmp_path / "repo"
    batch = tmp_path / M.BATCH
    plan = {
        "batch": M.BATCH,
        "scientific_config": M.SCIENTIFIC_CONFIG,
        "output": str(batch),
        "arms": list(M.ARMS),
        "host": M.socket.gethostname(),
    }
    save(
        root / M.SCIENTIFIC_CONFIG,
        {
            "judge_model": M.JUDGE_MODEL,
            "judge_revision": M.JUDGE_REVISION,
            "judge_enable_thinking": False,
        },
    )
    for name in (*M.CORE_FILES, "scripts/start_knowledge_method_factorial_when_ready.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen test source\n")
    hashes = {name: M.sha(root / name) for name in M.CORE_FILES}
    markers = {
        "full_size": {
            "passed": True,
            "stage_results": {name: {"passed": True, "exit_code": 0} for name in M.EXPECTED_STAGES},
            "source_manifest": {"files": hashes},
        },
        "training_core": {"core_training_source_matches": True},
        "cpu": {
            "passed": True,
            "exit_code": 0,
            "source_frozen": True,
            "working_repo_changed_since_freeze": [],
            "source_sha256": hashes,
        },
        "judge": {
            "passed": True,
            "judge_model": M.JUDGE_MODEL,
            "judge_revision": M.JUDGE_REVISION,
            "judge_enable_thinking": False,
            "generated_think_tag_count": 0,
            "checks": [{"valid": True, "matches": True, "truncated": False} for _ in range(8)],
            "source_sha256": hashes,
            "config_sha256": M.sha(root / M.SCIENTIFIC_CONFIG),
        },
        "download": {"passed": True, "model": M.JUDGE_MODEL, "revision": M.JUDGE_REVISION},
    }
    for name, value in markers.items():
        save(batch / M.MARKERS[name], value)
    return root, batch, plan, markers


def test_valid_exact_thirteen_stage_gate_and_missing_marker_waits(ready):
    root, batch, plan, _ = ready
    result = M.check_readiness(plan, root)
    assert result["passed"] and len(result["evidence"]) == 5
    assert len(M.EXPECTED_STAGES) == 13
    assert not (batch / "engineering-gate.json").exists()
    (batch / M.MARKERS["full_size"]).unlink()
    result = M.check_readiness(plan, root)
    assert not result["passed"] and result["missing"] == [M.MARKERS["full_size"]]


@pytest.mark.parametrize("failure", ["failed", "missing_stage", "extra_stage", "exit_code"])
def test_failed_or_incomplete_thirteen_stage_matrix_is_terminal(ready, failure):
    root, batch, plan, markers = ready
    full = markers["full_size"]
    if failure == "failed":
        full["stage_results"]["exercise-sft_down"]["passed"] = False
    elif failure == "missing_stage":
        del full["stage_results"]["reload-pd_down_lora"]
    elif failure == "extra_stage":
        full["stage_results"]["exercise-extra"] = {"passed": True, "exit_code": 0}
    else:
        full["stage_results"]["pressure-pd_full"]["exit_code"] = 1
    save(batch / M.MARKERS["full_size"], full)
    with pytest.raises(M.GateFailed):
        M.check_readiness(plan, root)


def test_missing_marker_cannot_hide_existing_failure(ready):
    root, batch, plan, markers = ready
    (batch / M.MARKERS["full_size"]).unlink()
    markers["judge"]["passed"] = False
    save(batch / M.MARKERS["judge"], markers["judge"])
    with pytest.raises(M.GateFailed, match="completed marker did not pass"):
        M.check_readiness(plan, root)
    (batch / M.MARKERS["judge"]).unlink()
    save(batch / "engineering/results/exercise-pd_full.json", {"passed": False})
    with pytest.raises(M.GateFailed, match="completed marker did not pass"):
        M.check_readiness(plan, root)


@pytest.mark.parametrize("target", ["core", "cpu", "judge", "config"])
def test_stale_hashes_are_terminal(ready, target):
    root, batch, plan, markers = ready
    if target == "core":
        (root / M.CORE_FILES[0]).write_text("# changed\n")
    elif target == "config":
        with (root / M.SCIENTIFIC_CONFIG).open("a") as handle:
            handle.write("\n")
    else:
        markers[target]["source_sha256"] = {M.CORE_FILES[0]: "0" * 64}
        save(batch / M.MARKERS[target], markers[target])
    with pytest.raises(M.GateFailed, match="stale"):
        M.check_readiness(plan, root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("judge_enable_thinking", True),
        ("generated_think_tag_count", 1),
        ("judge_revision", "wrong"),
    ],
)
def test_judge_contract_cannot_be_overridden_by_passed_flag(ready, field, value):
    root, batch, plan, markers = ready
    markers["judge"][field] = value
    save(batch / M.MARKERS["judge"], markers["judge"])
    with pytest.raises(M.GateFailed):
        M.check_readiness(plan, root)


def test_timeout_never_passes_missing_evidence(ready):
    root, batch, plan, _ = ready
    (batch / M.MARKERS["cpu"]).unlink()
    elapsed, states = [0], []

    def sleep(seconds):
        elapsed[0] += seconds

    with pytest.raises(TimeoutError):
        M.wait_until_ready(
            plan, root, states.append, timeout=60, clock=lambda: elapsed[0], sleep=sleep
        )
    assert elapsed[0] == 60
    assert all(row["state"] == "waiting" and not row["passed"] for row in states)
    assert not (batch / "engineering-gate.json").exists()


def test_dry_run_and_failed_execute_never_start_launcher(ready, monkeypatch):
    root, batch, plan, markers = ready
    plan_path = root / "configs/knowledge-method-factorial-launch-v1.json"
    save(plan_path, plan)
    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.setattr(M, "DEFAULT_PLAN", plan_path)
    monkeypatch.setattr(
        M, "__file__", str(root / "scripts/start_knowledge_method_factorial_when_ready.py")
    )
    monkeypatch.setattr(M.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not launch"))
    assert M.main([]) == 0
    assert not (batch / "readiness-process.json").exists()
    markers["cpu"]["passed"] = False
    save(batch / M.MARKERS["cpu"], markers["cpu"])
    assert M.main(["--execute"]) == 1
    assert json.loads((batch / "readiness-exit.json").read_text())["state"] == "failed"
    assert not (batch / "controller-process.json").exists()
    assert not (batch / "engineering-gate.json").exists()


def test_valid_execute_calls_launcher_once_after_evidence_and_records_exit(ready, monkeypatch):
    root, batch, plan, _ = ready
    plan_path = root / "configs/knowledge-method-factorial-launch-v1.json"
    save(plan_path, plan)
    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.setattr(M, "DEFAULT_PLAN", plan_path)
    monkeypatch.setattr(
        M, "__file__", str(root / "scripts/start_knowledge_method_factorial_when_ready.py")
    )
    calls = []

    class FakeProcess:
        pid = 123456

        def __init__(self, command, **kwargs):
            calls.append(command)
            assert json.loads((batch / "engineering-gate.json").read_text())["passed"]
            assert not (batch / "controller-process.json").exists()

        def wait(self):
            return 7

    monkeypatch.setattr(M.subprocess, "Popen", FakeProcess)
    assert M.main(["--execute"]) == 7
    assert len(calls) == 1 and calls[0][-1] == "--execute"
    assert json.loads((batch / "readiness-exit.json").read_text())["exit_code"] == 7
    with pytest.raises(M.GateFailed, match="Existing readiness-process"):
        M.main(["--execute"])
    assert len(calls) == 1
