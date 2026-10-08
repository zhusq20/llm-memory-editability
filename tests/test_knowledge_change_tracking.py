import json

from llm_memory_editability.interface_tracking import drain_runs


def test_tracking_drains_when_development_stops_before_confirmation(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "controller-state.json").write_text(
        json.dumps({"state": "development_prerequisite_not_met", "completed": []})
    )

    def unexpected_sleep(_seconds):
        raise AssertionError("A terminal development batch must not leave its tracker running")

    monkeypatch.setattr("llm_memory_editability.interface_tracking.time.sleep", unexpected_sleep)
    result = drain_runs(None, tmp_path, runs, {"poll_seconds": 15})
    assert result["passed"] and result["registered_runs"] == 0
    assert (tmp_path / "tracking-completion.json").exists()
