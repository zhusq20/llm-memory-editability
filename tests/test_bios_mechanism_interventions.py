"""P4 operator tests use synthetic states and tiny CPU models exclusively."""

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_mechanism_interventions import (
    Site,
    attach_equal_branch,
    attach_state_operation,
    calibration_plan,
    downstream_routing_site,
    fit_predicted_group_bases,
    match_delta_norm,
    query_site_roles,
    remove_equal_branch,
    remove_projection,
    restore_projection,
    select_branch_parameters,
    valid_predicted_groups,
    validate_restoration_sites,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_states():
    rng = np.random.default_rng(37)
    labels = np.repeat(np.arange(4), 16)
    centers = rng.standard_normal((4, 12))
    x = centers[labels] + 0.15 * rng.standard_normal((len(labels), 12))
    return x, labels


def test_calibration_excludes_nonactual_and_unpermitted_queries():
    plan = calibration_plan(
        actual_query_ids=np.arange(6, 12),
        person_by_query=np.array([-1] * 6 + list(range(6))),
        membership_query_ids=np.arange(6),
        supervised=np.array([0, 6, 8]),
        replay=np.array([2, 8, 10]),
    )
    np.testing.assert_array_equal(plan["actual_query_ids"], [6, 8, 10])
    np.testing.assert_array_equal(plan["membership_query_ids"], [0, 2, 4])
    assert plan["derived_calibration_queries"] == 0
    assert plan["logical_additional_membership_queries"] == 3
    labels, valid = valid_predicted_groups(
        np.array([10, 99, 11]), np.array([True, True, False]), np.array([10, 11])
    )
    np.testing.assert_array_equal(labels, [10, -1, -1])
    np.testing.assert_array_equal(valid, [True, False, False])


def test_three_bases_have_equal_rank_variance_and_orthogonal_complement():
    x, groups = synthetic_states()
    valid = np.ones(len(x), dtype=bool)
    bases, report = fit_predicted_group_bases(x, groups, valid, rank=2)
    assert report["available"] and report["used_rows"] == len(x)
    assert not report["uses_group_truth"] and not report["uses_derived_calibration"]
    for basis in bases.values():
        assert basis.rank == 2
        np.testing.assert_allclose(basis.q.T @ basis.q, np.eye(2), atol=1e-6)
        normalized = (x @ basis.q - basis.mean) / basis.scale
        assert np.mean(np.sum(normalized**2, axis=1)) == pytest.approx(1, abs=1e-6)
    np.testing.assert_allclose(bases["target"].q.T @ bases["complement"].q, 0, atol=1e-6)
    repeated, _ = fit_predicted_group_bases(x, groups, valid, rank=2)
    for name in bases:
        np.testing.assert_array_equal(bases[name].q, repeated[name].q)


def test_rank_failure_is_unavailable_without_random_padding():
    x, groups = synthetic_states()
    bases, report = fit_predicted_group_bases(x, groups % 2, np.ones(len(x), dtype=bool), rank=2)
    assert bases is None and not report["available"]
    assert report["target_numerical_rank"] == 1
    assert "Target" in report["reason"]
    centers = np.random.default_rng(2).standard_normal((4, 12))
    bases, report = fit_predicted_group_bases(
        centers[groups], groups, np.ones(len(x), dtype=bool), rank=2
    )
    assert bases is None and report["complement_numerical_rank"] == 0


def test_answer_centering_is_explicit_and_shared_by_all_controls():
    x, groups = synthetic_states()
    valid = np.ones(len(x), dtype=bool)
    with pytest.raises(ValueError, match="requires model answers"):
        fit_predicted_group_bases(x, groups, valid, rank=2, variant="answer_centered_diagnostic")
    bases, report = fit_predicted_group_bases(
        x,
        groups,
        valid,
        rank=2,
        variant="answer_centered_diagnostic",
        predicted_answers=groups % 2,
        answer_valid=valid,
    )
    assert report["available"] and report["answer_span_rank"] == 1
    assert set(bases) == {"target", "complement", "random"}
    for check in report["normalization"].values():
        assert check["answer_span_overlap"] < 1e-6
    # If the entire group-centroid span also codes the predicted answer, it is not relabeled
    # as organization signal by silently filling the missing dimensions with random axes.
    bases, report = fit_predicted_group_bases(
        x,
        groups,
        valid,
        rank=2,
        variant="answer_centered_diagnostic",
        predicted_answers=groups,
        answer_valid=valid,
    )
    assert bases is None and report["target_numerical_rank"] == 0


def test_projection_removal_and_later_restoration_leave_other_components_intact():
    x = torch.tensor([[3.0, 4.0, 7.0, 2.0]])
    q, mean = torch.eye(4)[:, :2], torch.zeros(2)
    lesioned, trace = remove_projection(x, q, mean)
    torch.testing.assert_close(lesioned, torch.tensor([[0.0, 0.0, 7.0, 2.0]]))
    norms = torch.linalg.vector_norm(trace["delta"], dim=-1)
    control, control_trace = remove_projection(x, torch.eye(4)[:, 2:], mean, norms)
    torch.testing.assert_close(torch.linalg.vector_norm(control - x, dim=-1), norms)
    assert control_trace["valid"].all()
    later = torch.tensor([[0.0, 0.0, 99.0, -9.0]])
    restored, _ = restore_projection(later, x, q)
    torch.testing.assert_close(restored, torch.tensor([[3.0, 4.0, 99.0, -9.0]]))
    _, invalid = match_delta_norm(torch.zeros_like(x), norms)
    assert not invalid.any()
    zero, valid = match_delta_norm(torch.zeros_like(x), torch.zeros(1))
    assert valid.all() and not zero.any()


def test_rescue_guards_reject_same_point_unreachable_and_terminal_shortcuts():
    source = Site(0, 2)
    assert downstream_routing_site(source, 8, [3, 4]) == Site(1, 2)
    assert not validate_restoration_sites(source, Site(1, 2), 8, [3, 4])["readout_control"]
    with pytest.raises(ValueError, match="strictly later"):
        validate_restoration_sites(source, source, 8, [4])
    with pytest.raises(ValueError, match="causally before"):
        validate_restoration_sites(source, Site(1, 1), 8, [4])
    with pytest.raises(ValueError, match="no nonterminal"):
        downstream_routing_site(Site(6, 2), 8, [4])
    with pytest.raises(ValueError, match="cannot affect"):
        validate_restoration_sites(Site(6, 2), Site(7, 2), 8, [4])
    with pytest.raises(ValueError, match="readout control"):
        validate_restoration_sites(Site(6, 2), Site(7, 4), 8, [4])
    assert validate_restoration_sites(Site(6, 2), Site(7, 4), 8, [4], allow_readout=True)[
        "readout_control"
    ]
    np.testing.assert_array_equal(query_site_roles([4, 5], 2), ["first_relation", "first_relation"])
    np.testing.assert_array_equal(query_site_roles([4, 5], 3), ["answer_marker", "second_relation"])
    with pytest.raises(ValueError, match="future answer"):
        query_site_roles([4, 5], 4)


def test_zero_initialized_branches_preserve_function_and_common_mlp_scope():
    x, groups = synthetic_states()
    bases, _ = fit_predicted_group_bases(x, groups, np.ones(len(x), dtype=bool), rank=2)
    torch.manual_seed(10)
    model = CausalLM(ModelConfig(24, width=12, layers=8, heads=1))
    tokens = torch.tensor([[1, 8, 6, 2, 0], [1, 9, 6, 7, 2]])
    positions = torch.tensor([[3], [4]])
    with torch.no_grad():
        clean = model(tokens, positions)
    counts, initial_a = [], []
    for basis in bases.values():
        branch, handle = attach_equal_branch(model, basis, Site(0, 2), bottleneck=3)
        with torch.no_grad():
            assert torch.equal(model(tokens, positions), clean)
        selected = select_branch_parameters(model, branch, train_common_mlp=True)
        expected_mlp = [p for block in model.blocks[3:6] for p in block.mlp.parameters()]
        assert {id(p) for p in selected} == {
            id(p) for p in expected_mlp + list(branch.parameters())
        }
        assert not model.token.weight.requires_grad
        counts.append(sum(p.numel() for p in branch.parameters()))
        initial_a.append(branch.a.detach().clone())
        branch_only = select_branch_parameters(model, branch, train_common_mlp=False)
        assert {id(p) for p in branch_only} == {id(p) for p in branch.parameters()}
        assert all(not p.requires_grad for block in model.blocks for p in block.parameters())
        remove_equal_branch(model, handle)
    assert counts == [3 * (12 + 2)] * 3
    assert all(torch.equal(initial_a[0], value) for value in initial_a)
    assert not hasattr(model, "edit_branch")


def test_hook_is_removed_and_early_last_layer_position_is_a_negative_control():
    torch.manual_seed(19)
    model = CausalLM(ModelConfig(24, width=12, layers=8, heads=1))
    tokens = torch.tensor([[1, 8, 6, 7, 2]])
    position = torch.tensor([[4]])
    with torch.no_grad():
        clean = model(tokens, position)
        handle, traces = attach_state_operation(
            model, Site(7, 2), lambda value: (torch.zeros_like(value), {})
        )
        assert torch.equal(model(tokens, position), clean)
        assert len(traces) == 1
        handle.remove()
        assert torch.equal(model(tokens, position), clean)
