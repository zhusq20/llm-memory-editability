"""Cloud endpoint verification must decode W&B's non-iterable nested summaries."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ttcd_continual_cli", ROOT / "scripts/run_ttcd_continual.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

STAGES = ("train", "train-audit", "calibrate", "evaluate", "audit")


class FakeSummarySubDict:
    """W&B 0.30.0 supports items/get/keys/getitem, but has no iterator.

    Nested dictionaries become SummarySubDict instances when accessed; lists
    retain ordinary dictionaries, matching the public Summary._decode method.
    """

    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        value = self.values[key]
        return FakeSummarySubDict(value) if isinstance(value, dict) else value

    def get(self, key, default=None):
        return self[key] if key in self.values else default

    def keys(self):
        return self.values.keys()

    def items(self):
        return ((key, self[key]) for key in self.values)


def prepare_cloud(tmp_path, monkeypatch):
    config = {
        "batch": "ttcd-cross-sample-v1",
        "models": {"single": {"step": 30000}},
        "parent": {"steps": 10000},
    }
    runs, requests = {}, []
    for stage in STAGES:
        result = {
            "stage": stage,
            "passed": True,
            "counts": {"base_parameter_updates": 10000},
            "nested": [{"values": [0.25, None, True, "exact"]}],
        }
        (tmp_path / f"result-{stage}.json").write_text(json.dumps(result))
        (tmp_path / f"wandb-{stage}.json").write_text(
            json.dumps(
                {
                    "id": stage,
                    "url": f"https://wandb.ai/zhusq20/llm-memory-editability/runs/{stage}",
                }
            )
        )
        runs[stage] = SimpleNamespace(
            state="finished",
            config={"arm": "single", "batch": config["batch"]},
            summary=FakeSummarySubDict(
                {
                    "result": result,
                    "training/step": 30000,
                    "exposure/base_parameter_updates": 10000,
                }
            ),
        )

    class FakeAPI:
        def run(self, path):
            requests.append(path)
            return runs[path.rsplit("/", 1)[1]]

    def api(**kwargs):
        assert kwargs == {"timeout": 60}
        return FakeAPI()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=api))
    return config, runs, requests


def test_plain_summary_recursively_unwraps_noniterable_wandb_values():
    expected = {
        "result": {
            "counts": {"steps": 10000},
            "nested": [{"values": [0.25, None, True, "exact"]}],
        }
    }
    summary = FakeSummarySubDict(expected)
    with pytest.raises(KeyError):
        set(summary["result"])
    actual = cli.plain_summary(summary)
    assert isinstance(actual, dict)
    assert isinstance(actual["result"], dict)
    assert actual == expected
    assert json.loads(json.dumps(actual)) == expected


def test_cloud_audit_compares_every_nested_result_and_stage_in_order(tmp_path, monkeypatch):
    config, _, requests = prepare_cloud(tmp_path, monkeypatch)
    result = cli.cloud_audit(config, "single", tmp_path)
    assert result["passed"]
    assert [record["stage"] for record in result["runs"]] == list(STAGES)
    assert requests == [f"zhusq20/llm-memory-editability/{stage}" for stage in STAGES]
    assert json.loads((tmp_path / "cloud-audit.json").read_text()) == result


@pytest.mark.parametrize("field", ["state", "arm", "batch", "step", "updates", "nested"])
def test_cloud_audit_rejects_identity_state_step_count_and_nested_mismatches(
    tmp_path, monkeypatch, field
):
    config, runs, _ = prepare_cloud(tmp_path, monkeypatch)
    run = runs["train"]
    if field == "state":
        run.state = "running"
    elif field == "arm":
        run.config["arm"] = "stream"
    elif field == "batch":
        run.config["batch"] = "another-batch"
    elif field == "step":
        run.summary.values["training/step"] = 29999
    elif field == "updates":
        run.summary.values["exposure/base_parameter_updates"] = 9999
    else:
        run.summary.values["result"]["nested"][0]["values"][0] = 0.75
    with pytest.raises(ValueError):
        cli.cloud_audit(config, "single", tmp_path)
    assert not (tmp_path / "cloud-audit.json").exists()


def test_cloud_audit_checks_the_last_independent_audit_state(tmp_path, monkeypatch):
    config, runs, requests = prepare_cloud(tmp_path, monkeypatch)
    runs["audit"].state = "failed"
    with pytest.raises(ValueError, match="identity/state mismatch for audit"):
        cli.cloud_audit(config, "single", tmp_path)
    assert len(requests) == len(STAGES)
    assert not (tmp_path / "cloud-audit.json").exists()
