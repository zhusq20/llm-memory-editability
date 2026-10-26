import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "one_step_training_cli",
    Path(__file__).resolve().parents[1] / "scripts/run_ttcd_one_step_training.py",
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_frozen_source_mutation_rejected_before_model_execution(tmp_path, monkeypatch):
    code = tmp_path / "science.py"
    code.write_text("original = 1\n")
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps({"files": {"science.py": cli.digest(code)}}))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setenv("LM_TTCD_SOURCE_MANIFEST_SHA256", cli.digest(manifest))
    cli.verify_runtime()
    code.write_text("modified = 2\n")
    with pytest.raises(ValueError, match="Frozen runtime bytes changed"):
        cli.verify_runtime()


def test_runtime_manifest_must_match_and_paths_stay_contained(tmp_path, monkeypatch):
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps({"files": {"../outside.py": "untrusted"}}))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setenv("LM_TTCD_SOURCE_MANIFEST_SHA256", "wrong")
    with pytest.raises(ValueError, match="source manifest"):
        cli.verify_runtime()
    monkeypatch.setenv("LM_TTCD_SOURCE_MANIFEST_SHA256", cli.digest(manifest))
    with pytest.raises(ValueError, match="escaped"):
        cli.verify_runtime()


@pytest.mark.parametrize("wrong", ["writer", "updates", "result", "none"])
def test_cloud_audit_binds_one_step_budget_and_canonical_json(tmp_path, monkeypatch, wrong):
    result = tmp_path / "result-train.json"
    result.write_text(json.dumps({"step": 10000}))
    (tmp_path / "wandb-train.json").write_text(json.dumps({"id": "fixed", "url": "url"}))
    config = {
        "batch": "ttcd-one-step-training-v1",
        "architecture": "loop",
        "training_writer_steps": 1,
        "evaluation_writer_steps": 1,
    }
    summary = {
        "result_sha256": cli.digest(result),
        "terminal_step": 10000,
        "base_parameter_updates": 10000,
        "fast_state_updates": 1280000,
    }
    if wrong == "writer":
        config["training_writer_steps"] = 4
    elif wrong == "updates":
        summary["fast_state_updates"] *= 4
    elif wrong == "result":
        summary["result_sha256"] = "wrong"

    def download(root, replace):
        path = Path(root) / result.name
        path.write_bytes(result.read_bytes())
        return SimpleNamespace(name=str(path))

    run = SimpleNamespace(
        state="finished",
        config=config,
        summary=summary,
        file=lambda _: SimpleNamespace(download=download),
    )
    monkeypatch.setitem(
        sys.modules, "wandb", SimpleNamespace(Api=lambda **_: SimpleNamespace(run=lambda _: run))
    )
    task = {"batch": config["batch"], "architecture_name": "loop"}
    if wrong == "none":
        assert cli.cloud_audit(task, tmp_path, ("train",))["canonical_json_exact"] is True
    else:
        with pytest.raises(ValueError):
            cli.cloud_audit(task, tmp_path, ("train",))
