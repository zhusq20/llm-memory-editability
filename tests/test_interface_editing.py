"""Contracts for graph-selected edits, held-out propagation and exact reloads."""

import json

import numpy as np
import pytest
import torch

from llm_memory_editability.interface_editing import (
    affected,
    answer_batch,
    audit,
    case_tasks,
    composition_pools,
    edit_objective,
    editable_parameters,
    graph_cases,
    learned_atoms,
    run,
)
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


def test_graph_cases_have_no_model_dependency_and_pair_across_parent_arms():
    spec = small_spec()
    cases, counts = graph_cases(spec, seed=14, per_cell=2, replay_n=4)
    other, other_counts = graph_cases(
        {**spec, "initialization": 991, "arm": "aligned"},
        seed=14,
        per_cell=2,
        replay_n=4,
    )
    assert cases == other and counts == other_counts
    assert len(cases) == 8
    assert all(v["eligible_addresses"] >= 2 for v in counts.values())
    assert {c["stratum"] for c in cases} == {"familiar", "strict"}
    assert {c["role"] for c in cases} == {"first", "second"}


def test_truth_all_affected_untrained_paths_partition_and_retention_exclusions():
    spec = small_spec()
    world = build_world(spec)
    cases, _ = graph_cases(spec, seed=19, per_cell=2, replay_n=4)
    pools = composition_pools(world)
    trained = set(map(tuple, world["train_composite"]))
    atoms = learned_atoms(world)
    for case in cases:
        tasks, originals = case_tasks(world, case)
        old, new, role = case["old_fact"], case["new_fact"], case["role"]
        lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
        lookup[tuple(new[:2])] = new[2]
        assert tuple(old) not in set(map(tuple, tasks["R_atomic"]))
        assert not set(map(tuple, tasks["R_atomic"])) & set(map(tuple, tasks["U_atomic"]))
        for stratum in ("familiar", "strict", "registered_familiar"):
            dname = f"D_{role}_{stratum}"
            same = f"S_same_answer_{role}_{stratum}"
            rows = pools[stratum]
            expected = rows[affected(rows, old, role)]
            actual = np.concatenate([originals[dname], originals[same]])
            assert set(map(tuple, actual)) == set(map(tuple, expected))
            assert not set(map(tuple, actual)) & trained
            assert np.all(tasks[dname][:, -1] != originals[dname][:, -1])
            assert np.all(tasks[same][:, -1] == originals[same][:, -1])
            assert not affected(tasks[f"U_{stratum}"], old, role).any()
            for h, r1, b, r2, t in np.concatenate([tasks[dname], tasks[same]]):
                assert lookup[int(h), int(r1)] == b
                assert lookup[int(b), int(r2)] == t
        assert len(tasks[f"D_{role}_{case['stratum']}"]) > 0


def test_strict_bridge_substitutions_remain_without_composition_experience():
    spec = small_spec()
    world = build_world(spec)
    cases, _ = graph_cases(spec, per_cell=2, replay_n=4)
    used_bridges = set(world["train_composite"][:, 2])
    for case in cases:
        if case["role"] == "first" and case["stratum"] == "strict":
            assert case["new_fact"][2] not in used_bridges


def test_answer_loss_excludes_prefix_and_parameter_scope_freezes_everything_else():
    torch.set_num_threads(1)
    spec = small_spec()
    model = new_model(spec, "cpu").eval()
    world = build_world(spec)
    cases, _ = graph_cases(spec, per_cell=1, replay_n=4)
    tasks, _ = case_tasks(world, cases[0])
    target = answer_batch(tasks["E_new"], "cpu")
    replay = answer_batch(tasks["R_atomic"], "cpu")
    assert target[1].tolist() == [[4, 5, 6]]
    assert target[2].tolist() == [[cases[0]["new_fact"][2], 5, 1]]
    selected = editable_parameters(model, "mlp")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        parent_log = model(replay[0], positions=replay[1]).log_softmax(-1)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss, _ce, kl = edit_objective(model, target, replay, parent_log)
    assert abs(float(kl)) < 1e-6
    loss.backward()
    optimizer.step()
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and changed <= set(selected)
    with pytest.raises(ValueError):
        edit_objective(model, target, replay, parent_log.requires_grad_())


def test_run_sham_parent_reset_exact_saved_tensors_and_independent_reload(tmp_path):
    spec = small_spec()
    parent = new_model(spec, "cpu").eval()
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    torch.save({"model": parent.state_dict(), "spec": spec}, parent_dir / "model.pt")
    out = tmp_path / "run"
    result = run(
        {
            "phase": "engineering",
            "parent_dir": str(parent_dir),
            "candidate_seed": 81,
            "per_cell": 1,
            "replay_n": 4,
            "nodes": [0, 1],
            "lr": 1e-4,
            "case_ids": ["first_strict-00", "second_familiar-00"],
        },
        out,
        "cpu",
    )
    assert result["updates"] == 4
    assert len(result["branches"]) == 4
    learning_path = out / "learning.json"
    root_learning = json.loads(learning_path.read_text())
    # A reset node zero is still a new evaluation. A tracking consumer that
    # accepts only strictly newer step values must retain every branch node.
    assert len(root_learning) == 8
    assert [row["step"] for row in root_learning] == list(range(8))
    assert [row["optimizer_updates"] for row in root_learning] == [0, 1, 1, 2, 2, 3, 3, 4]
    for row in root_learning:
        assert row["target_supervised_tokens"] == row["optimizer_updates"] * 3
        assert row["replay_distillation_positions"] == row["optimizer_updates"] * 4 * 3
        assert "supervised_tokens" not in row
    initial = model_digest(parent)
    for branch in result["branches"]:
        directory = out / branch["path"]
        history = json.loads((directory / "learning.json").read_text())
        assert history[0]["model_sha256"] == initial
        assert len(json.loads((directory / "losses.json").read_text())) == 1
        assert set(branch["changed_tensors"]) <= {
            n for n in parent.state_dict() if n.startswith("blocks.0.mlp.")
        }
    audit_result = audit(out, "cpu")
    assert audit_result["passed"] and audit_result["nodes"] == 8
    assert audit_result["max_nll_error"] == 0
    root_learning[1]["step"] = 0
    learning_path.write_text(json.dumps(root_learning))
    with pytest.raises(AssertionError, match="evaluation indices"):
        audit(out, "cpu")
    root_learning[1]["step"] = 1
    root_learning[1]["target_supervised_tokens"] = 15
    learning_path.write_text(json.dumps(root_learning))
    with pytest.raises(AssertionError, match="exposure accounting"):
        audit(out, "cpu")
    root_learning[1]["target_supervised_tokens"] = 3
    learning_path.write_text(json.dumps(root_learning))
    with pytest.raises(FileExistsError):
        run({"parent_dir": str(parent_dir)}, out, "cpu")


def test_formal_runs_cannot_search_learning_rates(tmp_path):
    with pytest.raises(ValueError, match="development-only"):
        run({"phase": "confirmation", "learning_rates": [1e-5, 1e-4]}, tmp_path / "bad", "cpu")
