"""Static registration and scheduler contracts; no data construction or training."""

import importlib.util
import itertools
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "execute_grok_loop", ROOT / "scripts/execute_grok_loop.py"
)
scheduler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scheduler)


def test_confirmation_registration_is_complete_and_matches_development_baseline():
    dev = json.loads((ROOT / "configs/grok-loop-development-v1.json").read_text())
    cfg = json.loads((ROOT / "configs/grok-loop-confirmation-v1.json").read_text())
    jobs = scheduler.load_jobs(["configs/grok-loop-confirmation-v1.json"])
    expected = set(
        itertools.product(
            (145011, 145012, 145013), (14501, 14502), (2, 3, 4), ("c1", "c2", "cd", "l1", "l2")
        )
    )
    found = set()
    pairs = {}
    for job in jobs:
        spec = job["spec"]
        world, init, hops, arch = (
            spec[key] for key in ("world_seed", "initialization", "hops", "architecture")
        )
        found.add((world, init, hops, arch))
        original = {**dev["base"], **dev["runs"][f"dev-h{hops}-{arch}"]}
        for key in original.keys() - {"world_seed", "initialization", "stream_seed", "phase"}:
            assert spec[key] == original[key]
        assert spec["phase"] == "confirmation"
        assert spec["stream_seed"] == world * 1000 + hops * 10 + init - 14500
        pairs.setdefault((world, init, hops), []).append(spec)
    assert found == expected
    assert len(jobs) == len(found) == 90
    assert len(pairs) == 18
    assert len({group[0]["stream_seed"] for group in pairs.values()}) == 18
    for group in pairs.values():
        assert len({spec["stream_seed"] for spec in group}) == 1
        assert all(spec["batch_size"] == 256 and spec["n_atomic_per_batch"] == 32 for spec in group)
        by_arch = {spec["architecture"]: spec for spec in group}
        depth = 6 if group[0]["hops"] == 4 else 4
        for arch in ("cd", "l1", "l2"):
            assert by_arch[arch]["layers"] * by_arch[arch]["repeats"] == depth
        parameters = {}
        for arch, spec in by_arch.items():
            width = spec["width"]
            vocab = 2 + spec["entities"] + spec["relations"]
            parameters[arch] = (vocab + 10) * width + spec["layers"] * (12 * width**2 + 13 * width)
        assert parameters["c1"] == parameters["l1"] == 218240
        assert parameters["c2"] == parameters["l2"] == 416512
        assert parameters["cd"] == (1209600 if depth == 6 else 813056)
    assert sum(job["spec"]["steps"] for job in jobs) == 11520000
    assert sum(job["spec"]["steps"] * job["spec"]["batch_size"] for job in jobs) == 2949120000
    assert cfg["source_lock"] == "docs/development-artifacts/grok-loop-v1/confirmation-lock.json"
    assert cfg["extra_lock_files"] == [
        "docs/development-artifacts/grok-loop-v1/confirmation-plan.md",
        "src/llm_memory_editability/grok_loop_mechanism.py",
        "scripts/analyze_grok_loop_mechanism.py",
    ]


def make_config(root, name, runs):
    path = root / name
    cfg = {
        "experiment": Path(name).stem,
        "output_root": "results",
        "source_lock": name + "-lock.json",
        "base": {"phase": "test", "steps": 12},
        "runs": {run: {"identity": run} for run in runs},
    }
    path.write_text(json.dumps(cfg))
    return cfg


def test_job_loading_preserves_each_config_and_rejects_cross_config_duplicates(tmp_path):
    make_config(tmp_path, "first.json", ("a", "b"))
    make_config(tmp_path, "second.json", ("c", "d"))
    jobs = scheduler.load_jobs(["first.json", "second.json"], root=tmp_path)
    assert [job["run"] for job in jobs] == ["a", "b", "c", "d"]
    assert [job["config"] for job in jobs] == ["first.json"] * 2 + ["second.json"] * 2
    selected = scheduler.load_jobs(["first.json", "second.json"], ["d", "a"], root=tmp_path)
    assert [job["run"] for job in selected] == ["d", "a"]
    make_config(tmp_path, "second.json", ("a", "c"))
    with pytest.raises(ValueError, match="Duplicate run"):
        scheduler.load_jobs(["first.json", "second.json"], ["b"], root=tmp_path)


def test_attempt_logs_preserve_every_original_and_resume(tmp_path):
    for resume in (False, True):
        paths = []
        for i in range(3):
            path, log = scheduler.open_attempt_log(tmp_path, "example", resume)
            with log:
                log.write(str(i))
            paths.append(path)
        assert len(set(paths)) == 3
        assert [path.read_text() for path in paths] == ["0", "1", "2"]
    assert len(list(tmp_path.glob("*.log"))) == 6


def test_dynamic_queue_fills_free_slot_and_calls_each_original_config(tmp_path, monkeypatch):
    make_config(tmp_path, "first.json", ("a", "b"))
    make_config(tmp_path, "second.json", ("c", "d"))
    monkeypatch.setattr(scheduler, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "execute_grok_loop",
            "--config",
            "first.json",
            "--additional-configs",
            "second.json",
            "--gpus",
            "0",
            "1",
        ],
    )
    third_started = threading.Event()
    calls = []

    def fake_run(command, **kwargs):
        run = command[2]
        calls.append((run, command[-1], kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        if run == "a":
            assert third_started.wait(2), "Free worker did not dynamically consume next job"
        if run == "c":
            third_started.set()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    scheduler.main()
    by_run = {run: (config, gpu) for run, config, gpu in calls}
    assert set(by_run) == {"a", "b", "c", "d"}
    assert by_run["c"][1] == by_run["b"][1] != by_run["a"][1]
    assert by_run["a"][0] == "first.json"
    assert by_run["d"][0] == "second.json"
    artifact = tmp_path / "docs/development-artifacts/grok-loop-v1"
    records = [
        record
        for path in artifact.glob("*-queue-*.json")
        for record in json.loads(path.read_text())["runs"]
    ]
    assert len(records) == 4
    assert all(record["state"] == "complete" and record["config"] for record in records)


def test_completed_jobs_skip_and_repeated_resume_keeps_old_log(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, "first.json", ("a", "b"))
    for run in ("a", "b"):
        directory = tmp_path / "results/test" / run
        directory.mkdir(parents=True)
        if run == "a":
            (directory / "complete.json").write_text(
                json.dumps({"spec": {**cfg["base"], **cfg["runs"][run]}})
            )
        else:
            (directory / "latest.pt").write_bytes(b"dummy; never loaded by scheduler")
    artifact = tmp_path / "docs/development-artifacts/grok-loop-v1"
    artifact.mkdir(parents=True)
    (artifact / "b-resume.log").write_text("old evidence")
    monkeypatch.setattr(scheduler, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["execute_grok_loop", "--config", "first.json", "--gpus", "0"])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("new attempt")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    scheduler.main()
    assert len(calls) == 1 and calls[0][2] == "b" and "--resume" in calls[0]
    assert (artifact / "b-resume.log").read_text() == "old evidence"
    assert (artifact / "b-resume-attempt-2.log").read_text() == "new attempt"
    records = json.loads((artifact / "first-queue-0.json").read_text())["runs"]
    assert [record["state"] for record in records] == ["already_complete", "complete"]


def test_failed_job_retains_record_and_returns_failure(tmp_path, monkeypatch):
    make_config(tmp_path, "first.json", ("a",))
    monkeypatch.setattr(scheduler, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["execute_grok_loop", "--config", "first.json", "--gpus", "0"])
    monkeypatch.setattr(
        scheduler.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=2)
    )
    with pytest.raises(SystemExit) as error:
        scheduler.main()
    assert error.value.code == 1
    artifact = tmp_path / "docs/development-artifacts/grok-loop-v1"
    record = json.loads((artifact / "first-queue-0.json").read_text())["runs"][0]
    assert record["state"] == "failed" and record["exit_code"] == 2
    assert (tmp_path / record["log"]).exists()
