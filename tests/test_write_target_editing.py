"""Same-label, causal-state and exact-reload contracts of the write comparison."""

import copy
import json

import numpy as np
import pytest
import torch

from llm_memory_editability import interface_editing as ie
from llm_memory_editability import write_target_editing as wt
from llm_memory_editability.latent_scaling import build_world, model_digest
from llm_memory_editability.representation_alignment import new_model


def small_spec():
    return dict(
        world=771001,
        initialization=772001,
        stream_seed=773001,
        heads_n=32,
        bridges_n=32,
        tails_n=16,
        familiar_n=8,
        strict_n=4,
        anchor_n=4,
        holdout_fraction=0.25,
        low_extra="anchors",
        composition_count=32,
        width=16,
        heads=2,
        layers=1,
        repeats=2,
        dropout=0.0,
    )


def setup_case():
    torch.set_num_threads(1)
    spec = small_spec()
    model = new_model(spec, "cpu").eval()
    cases, _ = wt.selected_cases(spec, {"per_cell": 1, "replay_n": 4})
    case = cases[0]
    tasks, _ = ie.case_tasks(build_world(spec), case)
    target = ie.answer_batch(tasks["E_new"], "cpu")
    replay = ie.answer_batch(tasks["R_atomic"], "cpu")
    with torch.no_grad():
        reference = model(replay[0], positions=replay[1]).log_softmax(-1)
    return model, case, tasks, target, replay, reference


def test_final_baseline_loss_and_gradients_identical_to_existing_editor():
    model, _case, _tasks, target, replay, reference = setup_case()
    baseline = copy.deepcopy(model)
    loss, _ = wt.objective(model, target, replay, reference, "final_ce", {})
    old_loss, _, _ = ie.edit_objective(baseline, target, replay, reference)
    loss.backward()
    old_loss.backward()
    torch.testing.assert_close(loss, old_loss, atol=0, rtol=0)
    for p, q in zip(model.parameters(), baseline.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)


def test_added_supervision_is_same_entity_and_early_state_cannot_see_it():
    model, case, _tasks, target, replay, reference = setup_case()
    old_target = ie.answer_batch(np.asarray([case["old_fact"]]), "cpu")
    _, new_state = model(target[0], return_bridge=True)
    _, old_state = model(old_target[0], return_bridge=True)
    torch.testing.assert_close(new_state, old_state, atol=0, rtol=0)
    labels = target[2][:, 0]
    assert labels.tolist() == [case["new_fact"][2]]
    parts = []
    for kind in wt.OBJECTIVES:
        loss, components = wt.objective(model, target, replay, reference, kind, {})
        parts.append((loss, components))
    torch.testing.assert_close(parts[1][0] - parts[0][0], 0.3 * parts[0][1]["early_ce"])
    torch.testing.assert_close(parts[2][0] - parts[1][0], 0.3 * parts[0][1]["alignment"])
    with pytest.raises(ValueError, match="detached"):
        wt.objective(model, target, replay, reference.requires_grad_(), "early_ce", {})


def test_same_shared_mlp_is_only_updated_scope():
    model, _case, _tasks, target, replay, reference = setup_case()
    selected = ie.editable_parameters(model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss, _ = wt.objective(model, target, replay, reference, "early_ce_align", {})
    loss.backward()
    optimizer.step()
    changed = {
        key for key, value in model.state_dict().items() if not torch.equal(value, before[key])
    }
    assert changed and changed <= set(selected)
    assert model.token.weight.grad is None


def test_paired_matrix_exposure_accounting_and_independent_reload(tmp_path):
    spec = small_spec()
    model = new_model(spec, "cpu").eval()
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    torch.save({"model": model.state_dict(), "spec": spec}, parent_dir / "model.pt")
    out = tmp_path / "run"
    result = wt.run(
        {
            "phase": "engineering",
            "parent_dir": str(parent_dir),
            "per_cell": 1,
            "replay_n": 4,
            "nodes": [0, 1],
        },
        out,
        "cpu",
    )
    assert result["updates"] == 6 and len(result["branches"]) == 6
    assert {b["objective"] for b in result["branches"]} == set(wt.OBJECTIVES)
    initial = model_digest(model)
    for branch in result["branches"]:
        history = json.loads((out / branch["path"] / "learning.json").read_text())
        assert history[0]["model_sha256"] == initial
        assert branch["role"] == "first" and branch["stratum"] == "strict"
    audit = wt.audit(out, "cpu")
    assert audit["passed"] and audit["nodes"] == 12 and audit["max_float_error"] == 0
    decision = wt.development_decision([out])
    assert len(decision["records"]) == 6
    assert all(not any(key.startswith("D") for key in row) for row in decision["records"])
    learning_path = out / "learning.json"
    learning = json.loads(learning_path.read_text())
    assert learning[-1]["early_target_presentations"] == 4
    assert learning[-1]["alignment_target_presentations"] == 2
    learning[-1]["early_target_presentations"] = 5
    learning_path.write_text(json.dumps(learning))
    with pytest.raises(AssertionError, match="exposure accounting"):
        wt.audit(out, "cpu")
    with pytest.raises(FileExistsError):
        wt.run({}, out, "cpu")


def test_selection_uses_existing_strict_graph_cases_and_rejects_other_roles():
    spec = small_spec()
    settings = {"candidate_seed": 15, "per_cell": 2, "replay_n": 4}
    cases, _ = wt.selected_cases(spec, settings)
    all_cases, _ = ie.graph_cases(spec, seed=15, per_cell=2, replay_n=4)
    assert cases == [case for case in all_cases if case["case_id"].startswith("first_strict")]
    with pytest.raises(ValueError, match="strict first-hop"):
        wt.selected_cases(spec, {**settings, "case_ids": ["second_strict-00"]})


def test_development_decision_is_invariant_to_all_downstream_scores(tmp_path):
    directory = tmp_path / "development"
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps({"spec": {"phase": "development"}}))
    (directory / "audit.json").write_text(json.dumps({"passed": True}))
    branches = []
    for kind in wt.OBJECTIVES:
        for arm in ("edit", "sham"):
            metrics = {
                key: {"accuracy": 1.0} for key in ["E_new", "E_old", "U_atomic", "necessary_atomic"]
            }
            metrics["early"] = {"new_entity_accuracy": 1.0, "old_entity_accuracy": 1.0}
            metrics["D_first_strict"] = {"accuracy": 0.0}
            branches.append(
                {"objective": kind, "arm": arm, "case_id": "first_strict-00", "metrics": metrics}
            )
    (directory / "complete.json").write_text(json.dumps({"branches": branches}))
    before = wt.development_decision([directory])
    for branch in branches:
        branch["metrics"]["D_first_strict"] = {"accuracy": 1.0, "unexpected": "not consumed"}
    (directory / "complete.json").write_text(json.dumps({"branches": branches}))
    after = wt.development_decision([directory])
    assert before["records"] == after["records"]
    assert before["operational_gate_passed"] and after["operational_gate_passed"]


def test_historical_diagnostics_reload_without_training_or_mutation(tmp_path):
    spec = small_spec()
    model = new_model(spec, "cpu").eval()
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    torch.save({"model": model.state_dict(), "spec": spec}, parent_dir / "model.pt")
    old_run = tmp_path / "existing-run"
    ie.run(
        {
            "phase": "engineering",
            "parent_dir": str(parent_dir),
            "per_cell": 1,
            "replay_n": 4,
            "nodes": [0, 1],
            "case_ids": ["first_strict-00"],
        },
        old_run,
        "cpu",
    )
    before = {str(path): path.read_bytes() for path in old_run.rglob("*") if path.is_file()}
    result = wt.diagnostics(old_run, tmp_path / "diagnostics.json", "cpu")
    after = {str(path): path.read_bytes() for path in old_run.rglob("*") if path.is_file()}
    assert before == after
    assert result["training_updates"] == 0
    assert len(result["records"]) == 4
    assert all(row["early"]["state_delta_l2"] == 0 for row in result["records"] if row["step"] == 0)
