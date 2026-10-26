"""Audit resource recovery preserves frozen arguments and rejects unaccounted occupancy."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit_resources",
    Path(__file__).resolve().parents[1] / "scripts/resume_ttcd_audit_resources.py",
)
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


def test_only_registered_unchanged_neighbors_are_filtered():
    rows = {7: {"uuid": "GPU7"}}
    registered = {99: {"pid": 99, "uid": 1047, "start_ticks": 100}}
    processes = [{"pid": 41, "uuid": "GPU7"}, {"pid": 99, "uuid": "GPU7"}]
    assert resources.filtered_snapshot((rows, processes), registered, registered.get) == (
        rows,
        [{"pid": 41, "uuid": "GPU7"}],
    )
    processes.append({"pid": 101, "uuid": "GPU7"})
    assert (
        resources.filtered_snapshot((rows, processes), registered, registered.get)[1][-1]["pid"]
        == 101
    )


@pytest.mark.parametrize("changed", [{"pid": 99, "uid": 1047, "start_ticks": 101}, None])
def test_reused_or_disappearing_neighbor_pid_is_never_exempted(changed):
    assert resources.filtered_snapshot(
        ({7: {}}, [{"pid": 99}]),
        {99: {"pid": 99, "uid": 1047, "start_ticks": 100}},
        lambda _: changed,
    )[1] == [{"pid": 99}]


def test_idle_fallback_never_uses_occupied_or_busy_gpu():
    row = {"uuid": "idle", "memory": 0, "utilization": 0, "free_mib": 96000}
    rows = {
        0: row,
        1: {**row, "uuid": "occupied"},
        2: {**row, "uuid": "busy", "utilization": 50},
        3: {**row, "uuid": "reserved", "memory": 2000},
        7: {**row, "uuid": "cohort"},
    }
    assert resources.idle_candidates(rows, [{"pid": 99, "uuid": "occupied"}]) == [0]


def test_memory_bootstrap_preserves_exact_frozen_script_and_arguments(monkeypatch):
    import sys
    from types import SimpleNamespace

    original = [
        "docker",
        "create",
        "sha256:original",
        "python",
        "/frozen/scripts/run_ttcd_merge_hops.py",
        "audit",
        "--device",
        "cuda:0",
        "--out",
        "/output",
    ]
    bounded = resources.bounded_command(original)
    assert bounded[:4] == original[:4]
    assert bounded[6:] == original[4:]
    assert "set_per_process_memory_fraction(0.06)" in bounded[5]
    assert "sys.argv=sys.argv[1:]" in bounded[5]
    assert original[4].endswith("run_ttcd_merge_hops.py")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(set_per_process_memory_fraction=lambda f: calls.append(f))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "runpy",
        SimpleNamespace(run_path=lambda p, run_name: calls.append((p, run_name, sys.argv[:]))),
    )
    monkeypatch.setattr(sys, "argv", ["-c", *original[4:]])
    monkeypatch.setattr(sys, "path", sys.path[:])
    exec(bounded[5], {})
    assert calls == [0.06, (original[4], "__main__", original[4:])]


def test_bootstrap_executes_sibling_imports_in_a_new_process(tmp_path):
    import os
    import subprocess
    import sys

    mocks = tmp_path / "mock_modules"
    mocks.mkdir()
    (mocks / "torch.py").write_text(
        "import os\n"
        "class cuda:\n"
        " @staticmethod\n"
        " def set_per_process_memory_fraction(f):\n"
        "  assert f == 0.06\n"
        "  os.environ['TEST_ALLOCATOR_LIMIT'] = str(f)\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sibling_helper.py").write_text("ORIGINAL_VALUE = 42\n")
    entry = scripts / "run_ttcd_merge_hops.py"
    entry.write_text(
        "import os,sys\n"
        "from sibling_helper import ORIGINAL_VALUE\n"
        "assert ORIGINAL_VALUE == 42\n"
        "assert os.environ['TEST_ALLOCATOR_LIMIT'] == '0.06'\n"
        "assert sys.argv[1:] == ['audit','--device','cuda:0']\n"
    )
    command = resources.bounded_command(
        ["python", "/frozen/scripts/run_ttcd_merge_hops.py", "audit", "--device", "cuda:0"]
    )
    env = {**os.environ, "PYTHONPATH": str(mocks)}
    subprocess.run(
        [sys.executable, "-c", command[2], str(entry), *command[4:]],
        env=env,
        cwd=tmp_path,
        check=True,
    )


def test_bootstrap_refuses_training():
    with pytest.raises(ValueError, match="evaluation entrypoints"):
        resources.bounded_command(["python", "/frozen/scripts/run_ttcd_one_step_training.py"])
