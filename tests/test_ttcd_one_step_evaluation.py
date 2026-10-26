"""New one-step endpoints, shared hop states and independent full replay."""

import copy
import json

import numpy as np
import pytest
import torch
from test_ttcd_architecture_eval import make_reader
from test_ttcd_continual_eval import experiment as material

from llm_memory_editability import ttcd_continual_eval as original
from llm_memory_editability import ttcd_one_step_evaluation as evaluation
from llm_memory_editability.ttcd_editing import tensor_digest
from llm_memory_editability.ttcd_merge_hop_panel import build_panel

experiment = material


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def settings():
    return {
        "batch": evaluation.BATCH,
        "writer_steps": 1,
        "training_writer_steps": 1,
        "methods": ["single", "continuous", "independent_merge"],
        "roles": ["first", "second"],
        "first_hop_panel": {"seed": 20261015, "max_queries_per_fact": 8},
    }


def task_for(config):
    return {
        "architecture_name": "transformer",
        "training_method": "stream",
        "training_config": config,
    }


def test_rejects_four_step_training_or_test_view_before_loading(experiment, settings):
    config, _, _ = experiment
    config = copy.deepcopy(config)
    config["writer"]["steps"] = 1
    task = task_for(config)
    evaluation.validate_settings(task, settings)
    task["training_config"]["writer"]["steps"] = 4
    with pytest.raises(ValueError, match="one-step"):
        evaluation.validate_settings(task, settings)
    config["writer"]["steps"] = 1
    changed = {**settings, "training_writer_steps": 4}
    with pytest.raises(ValueError, match="one-step"):
        evaluation.validate_settings(task, changed)
    changed = {**settings, "methods": ["continuous", "independent_merge"]}
    with pytest.raises(ValueError, match="one-step"):
        evaluation.validate_settings(task, changed)
    changed = {**task, "training_method": "single"}
    with pytest.raises(ValueError, match="one-step"):
        evaluation.validate_settings(changed, settings)


def test_formal_matrix_preserves_all_three_methods_and_shared_hop_states():
    panel = {"facts": [{}] * 32, "orders": [list(range(i, 32)) + list(range(i)) for i in range(32)]}
    plans = evaluation.state_plan(panel)
    assert len(plans) == 2081
    assert sum(p["kind"] == "single" for p in plans.values()) == 32
    for method in ("continuous", "independent_merge"):
        assert sum(p["kind"] == method for p in plans.values()) == 1024
        assert plans[f"{method}/r031/t032"]["indices"] == panel["orders"][31]
    assert not any("first" in key or "second" in key for key in plans)


def engineer(experiment, settings, tmp_path, monkeypatch, architecture):
    config, world, _ = experiment
    config = copy.deepcopy(config)
    config["writer"]["steps"] = 1
    world.lookup = np.full((world.entities, world.metadata["relations"]), -1, dtype=np.int64)
    world.lookup[world.atoms[:, 0], world.atoms[:, 1]] = np.arange(len(world.atoms))
    panel = build_panel(original.build_panel(config, world), world, settings, config["phi"])
    reader = make_reader(architecture, vocab_size=world.metadata["vocab_size"], dtype=torch.float32)
    out = tmp_path / "evaluation"
    out.mkdir()
    (out / "panel.json").write_text(json.dumps(panel))
    task = task_for(config)
    test_identity = {"engineering_only": True, "writer_steps": 1, "architecture": architecture}
    # Real Transformer writers/readouts are used; only external checkpoint I/O
    # is replaced with this explicitly tiny engineering endpoint.
    monkeypatch.setattr(evaluation, "source_context", lambda *_: (reader, world, config, panel))
    monkeypatch.setattr(evaluation, "identity", lambda *_: test_identity)
    return task, config, world, reader, panel, out


@pytest.mark.parametrize("architecture", ("d8", "loop4x2", "disco4x2"))
def test_three_methods_both_hops_and_all_states_independently_replay(
    architecture, experiment, settings, tmp_path, monkeypatch
):
    task, _, _, reader, panel, out = engineer(
        experiment, settings, tmp_path, monkeypatch, architecture
    )
    before = tensor_digest(reader.backbone.state_dict())
    assert evaluation.preflight(task, settings, out, "cpu")["formal_model_queries"] == 0
    prepared = (out / "panel.json").read_bytes()
    calibration = evaluation.calibrate(task, settings, out, "cpu")
    assert calibration["passed"] and calibration["formal_D_used"] is False
    assert calibration["model_or_fact_selection"] is False
    assert (out / "panel.json").read_bytes() == prepared
    summary = evaluation.evaluate(task, settings, out, "cpu")
    assert summary["schema"] == evaluation.BATCH
    assert summary["states"] == 37
    assert summary["independent_donors"] == 4
    assert summary["continuous_transitions"] == summary["merged_states"] == 16
    assert len(list((out / "states").rglob("fast.pt"))) == 37
    assert summary["training_writer_steps"] == summary["evaluation_writer_steps"] == 1
    assert summary["formal_updated_two_hop_supervision"] is False
    for method in ("continuous", "independent_merge"):
        assert set(summary["roles"]["second"][method]) == {"1", "2", "3", "4"}
        assert (
            summary["roles"]["first"][method]["4"]["all"]["n"]
            == 4 * panel["first_hop_coverage"]["queries_total"]
        )
        # Atomic age measurements include facts with zero first-hop coverage.
        assert summary["E_by_age"][method]["0"]["all"]["n"] == 16
    for rotation, order in enumerate(panel["orders"]):
        single = json.loads((out / f"states/single/f{order[0]:03d}/record.json").read_text())
        for method in ("continuous", "independent_merge"):
            record = json.loads(
                (out / f"states/{method}/r{rotation:03d}/t001/record.json").read_text()
            )
            assert record["fast_digest"] == single["fast_digest"]
            assert record["readout"] == single["readout"]
            assert record["first"] == single["first"]
    audit = evaluation.audit(task, settings, out, "cpu")
    assert audit["passed"] and audit["states"] == 37
    assert audit["one_step_writes_rebuilt"] == 20
    assert audit["independent_donors_rewritten"] == 4
    assert audit["all_continuous_trajectories_rebuilt_from_A0"]
    assert before == tensor_digest(reader.backbone.state_dict())
    assert not any(p.requires_grad for p in reader.parameters())


def test_audit_rejects_coherently_forged_merge_continuous_tokens_and_history(
    experiment, settings, tmp_path, monkeypatch
):
    task, _, _, _, _, out = engineer(experiment, settings, tmp_path, monkeypatch, "d8")
    evaluation.calibrate(task, settings, out, "cpu")
    evaluation.evaluate(task, settings, out, "cpu")
    for method in ("continuous", "independent_merge"):
        directory = out / f"states/{method}/r000/t002"
        fast_bytes = (directory / "fast.pt").read_bytes()
        record_bytes = (directory / "record.json").read_bytes()
        payload = torch.load(directory / "fast.pt", weights_only=True)
        if method == "independent_merge":
            payload["fast"] = {key: value / 2 for key, value in payload["fast"].items()}
        else:
            next(iter(payload["fast"].values()))[0, 0] += 0.01
        payload["fast_digest"] = tensor_digest(payload["fast"])
        torch.save(payload, directory / "fast.pt")
        record = json.loads(record_bytes)
        record["fast_digest"] = payload["fast_digest"]
        (directory / "record.json").write_text(json.dumps(record))
        with pytest.raises(AssertionError, match="independently rebuilt"):
            evaluation.audit(task, settings, out, "cpu")
        (directory / "fast.pt").write_bytes(fast_bytes)
        (directory / "record.json").write_bytes(record_bytes)
    path = out / "states/single/f000/record.json"
    original_bytes = path.read_bytes()
    record = json.loads(original_bytes)
    record["readout"]["predictions"]["D"]["answer"][0] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="token/label"):
        evaluation.audit(task, settings, out, "cpu")
    record = json.loads(original_bytes)
    record["write_history"]["updates"] *= 2
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="lengths"):
        evaluation.audit(task, settings, out, "cpu")
    path.unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.audit(task, settings, out, "cpu")


def test_calibration_does_not_filter_an_endpoint_with_low_development_accuracy(
    experiment, settings, tmp_path, monkeypatch
):
    task, config, _, _, _, out = engineer(experiment, settings, tmp_path, monkeypatch, "d8")
    # This threshold is deliberately unattainable; it must remain a reported
    # development diagnostic rather than a reason to omit this model.
    config["reading_gate"]["minimum_accuracy"] = 1.1
    receipt = evaluation.calibrate(task, settings, out, "cpu")
    assert receipt["passed"] is True
    assert receipt["accuracy_gate_passed"] is False
    assert receipt["accuracy_gate_advisory_only"] is True
    assert evaluation._require_calibration(task, settings, out) == receipt["identity"]


def test_eval_identity_uses_new_one_step_adapter_not_old_evaluation_interfaces(
    experiment, settings, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    config, _, _ = experiment
    config["writer"]["steps"] = 1
    task = task_for(config)
    (tmp_path / "panel.json").write_text(json.dumps({"engineering_panel": True}))
    calls = []

    def endpoint_identity(received, phase, panel):
        calls.append((received, phase, panel))
        return {"new_endpoint_sha256": "one-step-trained", "writer_steps": 1}

    monkeypatch.setattr(
        evaluation,
        "_engine",
        lambda received: SimpleNamespace(evaluation_identity=endpoint_identity),
    )
    result = evaluation.identity(task, settings, tmp_path)
    assert calls == [(config, "stream", {"engineering_panel": True})]
    assert result["training_endpoint"]["new_endpoint_sha256"] == "one-step-trained"
    assert result["training_writer_steps"] == result["evaluation_writer_steps"] == 1
    assert result["architecture"] == "transformer"


def test_training_control_pairs_query_order_prefix_and_reports_joint_baseline_coverage():
    old = [
        {
            "query": [1, 2, 3, 4],
            "rotation": 0,
            "t": 32,
            "baseline_correct": True,
            "new_correct": False,
        },
        {
            "query": [1, 2, 3, 4],
            "rotation": 1,
            "t": 32,
            "baseline_correct": False,
            "new_correct": True,
        },
    ]
    new = copy.deepcopy(old)
    new[0]["new_correct"] = True
    new[1]["new_correct"] = False
    new[1]["baseline_correct"] = True
    report = evaluation.paired_training_difference(new, old)
    assert report["all"]["one_step_minus_four_step_training_pp"] == 0
    assert report["both_baseline_correct"]["one_step_minus_four_step_training_pp"] == 100
    assert report["both_baseline_correct"]["coverage"] == 0.5
    with pytest.raises(ValueError, match="identical"):
        evaluation.paired_training_difference(new, old[:1])
    new[1]["rotation"] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.paired_training_difference(new, old)


def test_incomplete_four_step_control_does_not_block_new_main_report(tmp_path):
    paths = {}
    for architecture in ("transformer", "loop", "discoloop"):
        out = tmp_path / architecture
        out.mkdir()
        paths[architecture] = out
        (out / "summary.json").write_text(json.dumps({"schema": evaluation.BATCH}))
        paired = {
            "baseline": {role: [] for role in evaluation.ROLES},
            "roles": {
                role: {method: [] for method in evaluation.METHODS} for role in evaluation.ROLES
            },
        }
        (out / "paired-queries.json").write_text(json.dumps(paired))
    report = evaluation.aggregate(paths, tmp_path / "aggregate")
    assert report["schema"] == evaluation.BATCH
    assert report["four_step_training_control"]["status"] == "pending"
    assert len(report["all_results"]) == 3
    assert report["roles"]["second"]["common_baseline_correct"] == 0
    for control in report["four_step_training_control"]["architectures"].values():
        assert control["status"] == "pending"
        assert len(control["missing"]) == 4
