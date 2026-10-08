"""Keep Loop comparisons paired and retain actual endpoint/panel denominators."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_script("report_loop_learning")
cloud = load_script("audit_loop_learning_tracking")


def spec(arm="shared", **changes):
    return {
        "name": arm,
        "arm": arm,
        "model": {"hidden_size": 256},
        "base_unique_blocks": 2,
        "repeats": 2,
        "data_sha256": "same-data",
        "initialization": 11,
        "sampling_seed": 12,
        "stage_a_steps": 2,
        "stage_b_steps": 2,
        "evaluation_nodes": [0, 1, 2, 3, 4],
        "batch_size": 4,
        "history": "sequential_composition",
        "replay_source": "full_stage_a",
        **changes,
    }


def predictions(bits=(1, 1, 0, 0)):
    return {
        pool: [{"id": f"{pool}-{i}", "alias_em": bit} for i, bit in enumerate(bits)]
        for pool in report.POOLS
    }


def record(step, raw):
    return {
        "step": step,
        "metrics": {
            pool: {
                "n": len(rows),
                "answer_accuracy": sum(row["alias_em"] for row in rows) / len(rows),
            }
            for pool, rows in raw.items()
        },
        "supervised_tokens": 10 * step,
        "estimated_matmul_training_flops": 1000 * step,
        "training_seconds": 0.1 * step,
        "wall_seconds": 0.2 * step,
    }


def make_run(root, selected, *, final=True, completed=True):
    out = root / "runs" / selected["name"]
    out.mkdir(parents=True)
    before = predictions()
    after = predictions((0, 1, 1, 0))
    after["BB"] = predictions((1, 1, 1, 0))["BB"]
    history = []
    for step in [0, 1, 2, 3] + ([4] if final else []):
        raw = before if step == 2 else after if step == 4 else predictions((0, 1))
        history.append(record(step, raw))
        report.write(out / f"predictions-{step:07d}.json", raw)
    report.write(out / "learning.json", history)
    report.write(
        out / "run.json", {"spec": selected, "parameters": 100, "initial_model_sha256": "p"}
    )
    report.write(out / "status.json", {"state": "complete" if completed and final else "training"})
    report.write(out / "sampling-plan.json", {"plan_sha256": "plan", "multiset_sha256": "multiset"})
    if final:
        report.write(out / "endpoint.json", history[-1]["metrics"])
    if completed and final:
        report.write(out / "audit.json", {"passed": True})
        report.write(out / "complete.json", {})
    return out


def test_full_boundary_and_panel_denominators_are_separate(tmp_path):
    selected = spec()
    make_run(tmp_path, selected)
    summary = report.summarize({"specs": [selected]}, tmp_path)
    rows = [row for row in summary["curves"] if row["pool"] == "AA"]
    assert [(row["step"], row["evaluation_scope"], row["n"]) for row in rows] == [
        (0, "panel", 2),
        (1, "panel", 2),
        (2, "full", 4),
        (3, "panel", 2),
        (4, "full", 4),
    ]
    assert summary["state"] == "complete"
    run = summary["runs"][0]
    assert run["stage_a"]["BB"]["answer_accuracy"] == 0.5
    assert run["endpoint"]["BB"]["answer_accuracy"] == 0.75
    assert run["change_during_B"]["BB"] == 0.25
    assert run["parameters"] == 100
    assert run["base_blocks"] == 2


def test_retention_does_not_confuse_net_change_with_preservation(tmp_path):
    selected = spec()
    make_run(tmp_path, selected)
    run = report.summarize({"specs": [selected]}, tmp_path)["runs"][0]
    counts = run["transitions"]["AA"]
    assert run["change_during_B"]["AA"] == 0
    assert counts["stage_a_correct_coverage"] == 0.5
    assert counts["retention_of_stage_a_correct"] == 0.5
    assert counts["retained_correct"] == 1
    assert counts["lost_correct"] == counts["gained_correct"] == 1


def test_missing_endpoint_is_null_and_training_output_without_audit_is_partial(tmp_path):
    selected = spec()
    make_run(tmp_path, selected, final=False)
    summary = report.summarize({"specs": [selected]}, tmp_path)
    row = summary["runs"][0]
    assert summary["state"] == "partial"
    assert row["latest_step"] == 3
    assert row["endpoint"]["BB"] is None
    assert row["transitions"]["AA"] is None
    assert row["change_during_B"]["BB"] is None
    other = spec("untied")
    make_run(tmp_path, other, completed=False)
    summary = report.summarize({"specs": [other]}, tmp_path)
    assert summary["runs"][0]["endpoint"]["BB"] is not None
    assert summary["state"] == "partial"


def test_pairing_requires_identical_world_width_seed_and_budget(tmp_path):
    specs = [spec(), spec("untied"), spec("mlp_shared", initialization=22)]
    specs += [spec("shallow", data_sha256="different-data")]
    for selected in specs:
        make_run(tmp_path, selected)
    summary = report.summarize({"specs": specs}, tmp_path)
    assert len(summary["paired_comparisons"]) == 1
    pair = summary["paired_comparisons"][0]
    assert pair["contrast"] == "shared_minus_untied"
    assert pair["sampling_plan_identical"] is True
    assert pair["endpoint"]["AA"] == 0
    changed = copy.deepcopy(specs[1])
    changed["model"]["hidden_size"] = 128
    assert report.identity(changed) != report.identity(specs[0])
    changed = copy.deepcopy(specs[1])
    changed["stage_a_steps"] = 3
    assert report.identity(changed) != report.identity(specs[0])


def test_bad_prediction_counts_or_duplicate_ids_are_rejected():
    raw = predictions()
    value = record(2, raw)
    value["metrics"]["AA"]["answer_accuracy"] = 0.75
    with pytest.raises(ValueError, match="accuracy disagrees"):
        report.metric(value, "AA", raw)
    repeated = copy.deepcopy(raw)
    repeated["AA"][-1]["id"] = repeated["AA"][0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        report.transitions(raw, repeated, "AA")
    mismatched = copy.deepcopy(raw)
    mismatched["AA"][-1]["id"] = "different"
    with pytest.raises(ValueError, match="IDs differ"):
        report.transitions(raw, mismatched, "AA")


def test_empty_parent_correct_retention_has_null_denominator():
    counts = report.transitions(predictions((0, 0)), predictions((1, 0)), "AA")
    assert counts["retention_of_stage_a_correct"] is None
    assert counts["stage_a_correct_coverage"] == 0
    assert counts["gained_correct"] == 1


def test_partial_report_exports_csv_without_inventing_endpoints(tmp_path):
    selected = spec()
    make_run(tmp_path, selected, final=False)
    summary = report.summarize({"specs": [selected]}, tmp_path)
    out = tmp_path / "report"
    out.mkdir()
    report.export_csv(summary, out)
    text = report.render(summary)
    assert "**partial**" in text
    assert "not new knowledge worlds" in text
    assert "BB before" in text
    assert (out / "curves.csv").exists()
    assert (out / "endpoints.csv").exists()


def prepare_cloud(root, selected):
    make_run(root, selected)
    report.write(
        root / "tracking-wandb" / selected["name"] / "state.json",
        {
            "run_id": "run-id",
            "mode": "online",
            "entity": "team",
            "project": "project",
            "last_logged_step": 4,
            "url": "https://wandb.ai/team/project/runs/run-id",
        },
    )
    remote = SimpleNamespace(
        state="finished",
        summary={
            "independently_reloaded": True,
            "scientific_final_step": 4,
            "has_failure_record": False,
        },
        scan_history=lambda keys: [{"training/step": n} for n in [0, 1, 2, 3, 4]],
    )
    api = SimpleNamespace(run=lambda name: remote)
    return api, remote


def test_cloud_requires_exact_nodes_online_mode_and_independent_reload(tmp_path):
    selected = spec()
    api, remote = prepare_cloud(tmp_path, selected)
    defaults = {"entity": "team", "project": "project"}
    assert cloud.check_run(api, tmp_path, selected, defaults)["passed"] is True
    remote.scan_history = lambda keys: [{"training/step": n} for n in [0, 1, 2, 4]]
    assert cloud.check_run(api, tmp_path, selected, defaults)["passed"] is False
    remote.scan_history = lambda keys: [{"training/step": n} for n in [0, 1, 2, 3, 4, 4]]
    assert cloud.check_run(api, tmp_path, selected, defaults)["passed"] is False
    remote.scan_history = lambda keys: [{"training/step": n} for n in [0, 1, 2, 3, 4]]
    path = tmp_path / "tracking-wandb" / selected["name"] / "state.json"
    state = json.loads(path.read_text())
    report.write(path, {**state, "mode": "offline"})
    result = cloud.check_run(api, tmp_path, selected, defaults)
    assert result["passed"] is False
    assert result["checks"]["local_online"] is False


def test_cloud_missing_artifacts_never_pass(tmp_path):
    api = SimpleNamespace(run=lambda name: pytest.fail("Should not query an absent run"))
    result = cloud.check_run(api, tmp_path, spec(), {"entity": "team", "project": "project"})
    assert result["passed"] is False
    assert result["reason"] == "missing_local_artifacts"


def test_cloud_audit_writes_all_run_evidence(tmp_path):
    selected = spec()
    api, _ = prepare_cloud(tmp_path, selected)
    source = tmp_path / "source"
    report.write(
        source / "configs/experiment-tracking-defaults.json",
        {"entity": "team", "project": "project"},
    )
    path = tmp_path / "frozen-config.json"
    report.write(
        path, {"results_root": str(tmp_path), "source_root": str(source), "specs": [selected]}
    )
    result = cloud.audit(path, api=api)
    assert result["passed"] is True
    assert json.loads((tmp_path / "tracking-cloud-audit.json").read_text())["passed"] is True
