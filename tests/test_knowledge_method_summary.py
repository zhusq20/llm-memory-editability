import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "kmd_summary", Path(__file__).parents[1] / "scripts/summarize_knowledge_method_diagnosis.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_incomplete_runs_are_not_complete(tmp_path):
    summary, report = module.collect(tmp_path)
    assert not summary["complete"]
    assert all(row["status"] == "incomplete" for row in summary["arms"].values())
    assert "pending" in report


def test_failed_reload_cannot_enter_scientific_summary(tmp_path):
    root = tmp_path / "runs/pd_lora"
    for name, value in (
        ("eval/evaluate.json", {}),
        ("audit/audit.json", {"passed": False}),
        ("complete.json", {"smoke": False}),
        ("manifest.json", {}),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="audit did not pass"):
        module.collect(tmp_path)
