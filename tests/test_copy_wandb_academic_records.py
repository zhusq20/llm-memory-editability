"""Guard source immutability and exact sparse curve copying."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "academic_copy", Path(__file__).resolve().parents[1] / "scripts/copy_wandb_academic_records.py"
)
copy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(copy)


def test_sparse_history_preserves_steps_timestamps_and_values():
    rows = [
        {"_step": 0, "_timestamp": 1000.5, "loss": 1.0},
        {"_step": 8000, "_timestamp": 2000.25, "score": 0.75},
    ]
    chunks = list(copy.history_chunks(rows, max_bytes=1))
    assert [offset for offset, _ in chunks] == [0, 1]
    assert [json.loads(line) for _, lines in chunks for line in lines] == rows


def test_missing_fields_are_not_zero_or_changed_values():
    assert copy.history_digest([{"_step": 1, "metric": None}]) == copy.history_digest(
        [{"_step": 1}]
    )
    assert copy.history_digest([{"_step": 1, "metric": 0}]) != copy.history_digest([{"_step": 1}])
    assert copy.history_digest([{"_step": 1, "metric": 0.5}]) != copy.history_digest(
        [{"_step": 1, "metric": 0.500000001}]
    )


def test_active_source_is_deferred_without_destination_write(tmp_path, monkeypatch):
    paths = []

    class FakeCloud:
        def __init__(self, key):
            self.api = SimpleNamespace(run=self.run)

        def run(self, path):
            paths.append(path)
            return SimpleNamespace(state="running")

    monkeypatch.setattr(copy, "Cloud", FakeCloud)
    assert copy.copy_one("abc", tmp_path, "test-key")["state"] == "waiting_for_source"
    assert paths == [f"{copy.SOURCE}/abc"]


def test_unrelated_destination_is_never_overwritten(tmp_path, monkeypatch):
    class FakeCloud:
        def __init__(self, key):
            self.api = SimpleNamespace(
                run=lambda path: SimpleNamespace(state="finished", url="source-url"),
                runs=lambda *args, **kwargs: [SimpleNamespace(notes="unrelated run")],
            )

    monkeypatch.setattr(copy, "Cloud", FakeCloud)
    receipt = copy.copy_one("abc", tmp_path, "test-key")
    assert receipt["state"] == "error"
    assert "refusing overwrite" in receipt["error"]


def test_mutation_is_scoped_to_academic_destination():
    cloud = object.__new__(copy.Cloud)
    calls = []
    cloud.post = lambda url, payload: calls.append(payload)
    cloud.upsert(
        {
            "id": "abc",
            "name": "historical",
            "url": "original",
            "created_at": "old",
            "raw_config": {},
            "summary": {},
            "notes": "",
            "group": "g",
            "job_type": "train",
            "tags": [],
            "state": "failed",
        },
        None,
    )
    variables = calls[0]["variables"]
    assert variables["entity"] == "zsqzz"
    assert variables["project"] == "llm-memory-editability"
    assert variables["state"] is None
    assert "source-state-failed" in variables["tags"]
