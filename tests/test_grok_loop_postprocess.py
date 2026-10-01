"""CPU-only orchestration contracts: identity, isolation, recovery and failures."""

import importlib.util
import json
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "postprocess_grok_loop", ROOT / "scripts/postprocess_grok_loop.py"
)
post = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(post)


def make_job(tmp_path, monkeypatch, run="example", architecture="l2", phase="confirmation"):
    monkeypatch.setattr(post, "ROOT", tmp_path)
    spec = {"phase": phase, "architecture": architecture, "steps": 12}
    config = tmp_path / f"{run}.json"
    post.write(config, {"base": spec, "runs": {run: {}}, "output_root": "results"})
    job = post.load_jobs([str(config)])[0]
    directory = Path(job["directory"])
    post.write(
        directory / "complete.json", {"spec": spec, "endpoint": {"step": 12}, "finished_utc": "now"}
    )
    checkpoint = directory / "weights-0000012.pt"
    checkpoint.write_bytes(b"CPU fixture, never loaded")
    return {
        **job,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": post.digest(checkpoint),
        "training_complete_sha256": post.digest(directory / "complete.json"),
    }


def audit_output(job):
    return {
        "state": "complete",
        "passed": True,
        "finished_utc": "now",
        **{
            key: job[key]
            for key in ("run", "spec", "checkpoint_sha256", "training_complete_sha256")
        },
        "audit": {"run": job["run"], "passed": True, "reload_max_nll_absolute_error": 0},
        "auditor_source": {"audit.py": "hash"},
        "environment": {"gpu": "fixture", "device": "cuda:0"},
    }


def archive(path):
    with zipfile.ZipFile(path, "w") as saved:
        saved.writestr("fixture.npy", b"zip-integrity-only fixture")


def mechanism_output(root, job, split):
    out = Path(root) / job["run"] / f"step-{job['spec']['steps']:07d}-{split}"
    post.write(out / "status.json", {"state": "complete", "finished_utc": "now"})
    post.write(
        out / "metadata.json",
        {
            "spec": job["spec"],
            "split": split,
            "step": job["spec"]["steps"],
            "provenance": {
                "source_dir": job["directory"],
                "checkpoint_sha256": job["checkpoint_sha256"],
            },
        },
    )
    post.write(out / "summary.json", {"split": split})
    for name in ("donors.npz", "predictions.npz"):
        archive(out / name)


def recurrence_output(root, job):
    post.write(
        Path(root) / "summary.json",
        {
            "finished_utc": "now",
            "repeats": post.REPEATS,
            "source": {"scan.py": "hash"},
            "training_source_lock": {"verified": True},
            "analysis_source_lock": {"verified": True},
            "environment": {"gpu": "fixture"},
            "runs": [
                {
                    "run": job["run"],
                    "spec": job["spec"],
                    "checkpoint": {job["checkpoint"]: job["checkpoint_sha256"]},
                    "provenance": {"complete_sha256": job["training_complete_sha256"]},
                    "native_repeats_audit": {
                        split: {"passed": True} for split in ("atomic", *post.SPLITS)
                    },
                    "measurements": [
                        {"repeats": r, "atomic": {}, **{s: {} for s in post.SPLITS}}
                        for r in post.REPEATS
                    ],
                }
            ],
        },
    )
    archive(Path(root) / f"{job['run']}.npz")


def install_fake_subprocess(monkeypatch, job, calls):
    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "5"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert str(ROOT / "src") not in kwargs["env"]["PYTHONPATH"].split(":")[:1]
        kwargs["stdout"].write("retained subprocess evidence")
        if "--audit-worker" in command:
            post.write(command[-1], audit_output(job))
        elif "scripts/analyze_grok_loop_mechanism.py" in command:
            root = command[command.index("--out") + 1]
            for i, token in enumerate(command):
                if token == "--split":
                    mechanism_output(root, job, command[i + 1])
        else:
            assert command[command.index("--runs") + 1] == job["run"]
            assert command[command.index("--architectures") + 1 : command.index("--out")] == [
                "l1",
                "l2",
            ]
            assert command[command.index("--repeats") + 1 :] == list(map(str, range(1, 9)))
            recurrence_output(command[command.index("--out") + 1], job)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(post.subprocess, "run", execute)


@pytest.mark.parametrize(
    "architecture,expected", [("c1", 2), ("c2", 2), ("cd", 2), ("l1", 3), ("l2", 3)]
)
def test_all_architectures_get_two_splits_and_only_loops_get_scan(
    tmp_path, monkeypatch, architecture, expected
):
    job = make_job(tmp_path, monkeypatch, architecture=architecture)
    calls = []
    install_fake_subprocess(monkeypatch, job, calls)
    assert post.process_job(job, 5)["state"] == "complete"
    assert len(calls) == expected
    assert post.process_job(job, 5)["state"] == "complete"
    assert len(calls) == expected  # all saved identities/completion markers were checked
    artifact = tmp_path / post.ARTIFACT / job["run"]
    assert len(list(artifact.glob("attempt-*"))) == 2
    assert len(list(artifact.glob("attempt-*/*.log"))) == expected


def test_sensitivity_only_audits(tmp_path, monkeypatch):
    job = make_job(tmp_path, monkeypatch, phase="development-sensitivity")
    calls = []
    install_fake_subprocess(monkeypatch, job, calls)
    assert post.process_job(job, 5)["state"] == "complete"
    assert len(calls) == 1 and "--audit-worker" in calls[0]


def test_partial_mechanism_is_preserved_and_reported_failed(tmp_path, monkeypatch):
    job = make_job(tmp_path, monkeypatch)
    post.write(tmp_path / post.ARTIFACT / job["run"] / "audit.json", audit_output(job))
    partial = (
        tmp_path / "results/mechanism-confirmation" / job["run"] / "step-0000012-test_composite"
    )
    partial.mkdir(parents=True)
    (partial / "evidence.txt").write_text("do not replace")
    monkeypatch.setattr(
        post.subprocess, "run", lambda *a, **k: pytest.fail("Partial artifact was rerun")
    )
    assert post.process_job(job, 5)["state"] == "failed"
    assert (partial / "evidence.txt").read_text() == "do not replace"
    assert list((tmp_path / post.ARTIFACT / job["run"]).glob("attempt-*/error.txt"))


def test_saved_audit_must_match_spec_checkpoint_and_gpu_completion(tmp_path, monkeypatch):
    job = make_job(tmp_path, monkeypatch)
    path = tmp_path / "audit.json"
    original = audit_output(job)
    for key, value in (
        ("spec", {}),
        ("checkpoint_sha256", "other"),
        ("passed", False),
        ("environment", {}),
    ):
        post.write(path, {**original, key: value})
        with pytest.raises(ValueError):
            post.validate_audit(path, job)


def test_subprocess_failure_retains_log_and_stops_dependent_stages(tmp_path, monkeypatch):
    job = make_job(tmp_path, monkeypatch)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write("GPU failure evidence")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(post.subprocess, "run", fail)
    record = post.process_job(job, 5)
    assert record["state"] == "failed" and len(calls) == 1
    assert record["stages"][0]["exit_code"] == 7
    assert Path(record["stages"][0]["log"]).read_text() == "GPU failure evidence"


def test_monitor_fills_free_gpu_without_more_than_one_worker_per_gpu(tmp_path, monkeypatch):
    jobs = [make_job(tmp_path, monkeypatch, run=run) for run in ("a", "b", "c")]
    third_started = threading.Event()
    calls = []

    def fake(job, gpu):
        calls.append((job["run"], gpu))
        if job["run"] == "a":
            assert third_started.wait(2)
        if job["run"] == "c":
            third_started.set()
        return {"state": "complete"}

    monkeypatch.setattr(post, "process_job", fake)
    assert post.monitor(jobs, [2, 5], 0.001)
    assert dict(calls) == {"a": 2, "b": 5, "c": 5}


def test_monitor_retries_partial_json_then_processes_endpoint(tmp_path, monkeypatch):
    job = make_job(tmp_path, monkeypatch)
    path = Path(job["directory"]) / "complete.json"
    valid = path.read_text()
    path.write_text("{")
    monkeypatch.setattr(post.time, "sleep", lambda _: path.write_text(valid))
    monkeypatch.setattr(post, "process_job", lambda *a: {"state": "complete"})
    assert post.monitor([job], [5], 0.001)


def test_monitor_retains_failure_and_processes_other_registered_runs(tmp_path, monkeypatch):
    jobs = [make_job(tmp_path, monkeypatch, run=run) for run in ("a", "b")]
    calls = []

    def fake(job, gpu):
        calls.append(job["run"])
        return {"state": "failed" if job["run"] == "a" else "complete"}

    monkeypatch.setattr(post, "process_job", fake)
    assert not post.monitor(jobs, [5], 0.001)
    assert calls == ["a", "b"]


def test_duplicate_registration_and_wrong_completed_spec_fail(tmp_path, monkeypatch):
    job = make_job(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        post.load_jobs([job["config"], job["config"]])
    path = Path(job["directory"]) / "complete.json"
    data = json.loads(path.read_text())
    data["spec"] = {"wrong": "identity"}
    post.write(path, data)
    with pytest.raises(ValueError, match="spec mismatch"):
        post.training_complete(job)
