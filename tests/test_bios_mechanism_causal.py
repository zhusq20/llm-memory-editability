"""CPU-only locked causal runner tests; no trained checkpoint or discovery result is read."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_mechanism_causal import (
    ARMS,
    DOWNSTREAM,
    FAMILIES,
    SOURCE,
    generate_queries,
    lesion_operations,
    make_probe_plan,
    paired_score_rows,
    scientific_outputs,
    score_rows,
    transfer_operations,
    validate_lock,
    wrong_source_indices,
)
from llm_memory_editability.bios_mechanism_interventions import (
    Basis,
    Site,
    projection_component,
    restore_projection,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_path_diagnostics import free_generate_values


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def toy_model():
    torch.manual_seed(81)
    return CausalLM(ModelConfig(vocab_size=24, width=12, layers=4, heads=1)).eval()


def toy_queries():
    return np.array(
        [[1, 5, 6, 2, 0], [1, 8, 6, 7, 2], [1, 9, 6, 2, 0], [1, 10, 6, 7, 2]]
    ), np.array([4, 5, 4, 5])


def toy_bases():
    q = np.eye(12, dtype=np.float32)
    return {
        label: Basis(
            q[:, start : start + 2], np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
        )
        for label, start in (("target", 0), ("random", 2), ("complement", 4))
    }


def test_candidate_guard_allows_only_predeclared_relation_site():
    lock = {
        "mechanism": "default_path",
        "layer": 0,
        "position": 2,
        "known_only": False,
        "locked_at_utc": "frozen-before-tests",
    }
    validate_lock(lock)
    for key, value in (
        ("position", 4),
        ("layer", 1),
        ("mechanism", "actual_path"),
        ("known_only", True),
        ("locked_at_utc", ""),
    ):
        with pytest.raises(ValueError):
            validate_lock({**lock, key: value})


def test_completion_hashes_exclude_mutable_scheduler_artifacts(tmp_path):
    (tmp_path / "company").mkdir()
    (tmp_path / "contract.json").write_text("{}")
    (tmp_path / "company/clean.npz").write_bytes(b"scientific")
    for name in ("queue-worker.log", "queue-launch.json", ".run.lock", "company/queue-worker.log"):
        (tmp_path / name).write_text("mutable")
    first = scientific_outputs(tmp_path)
    assert set(first) == {"contract.json", "company/clean.npz"}
    (tmp_path / "queue-worker.log").write_text("still writing after scientific completion")
    assert scientific_outputs(tmp_path) == first


def test_mixed_query_generation_matches_frozen_convention_and_caches_preanswer_states():
    model = toy_model()
    prompts, lengths = toy_queries()
    expected = free_generate_values(model, prompts, lengths, torch.device("cpu"), batch_size=2)
    actual = generate_queries(
        model, prompts, lengths, torch.device("cpu"), 2, capture_sites=(SOURCE, DOWNSTREAM)
    )
    for key in ("prediction", "ended"):
        np.testing.assert_array_equal(expected[key], actual[key])
    assert actual["forward_calls"] == 4 and actual["forward_examples"] == 8
    assert actual["padded_token_positions"] == 44
    assert actual["logical_token_positions"] == 40
    assert actual["state_0_2"].shape == actual["state_1_2"].shape == (4, 12)
    np.testing.assert_allclose(actual["state_0_2_pass_drift"], 0, atol=1e-7)
    # The same pre-answer site cannot depend on a generated answer in pass 2.
    np.testing.assert_allclose(actual["state_1_2_pass_drift"], 0, atol=1e-7)
    with pytest.raises(ValueError, match="future answer"):
        generate_queries(model, prompts, lengths, torch.device("cpu"), capture_sites=(Site(0, 4),))


def test_real_cpu_model_all_nine_conditions_and_equal_norm_controls():
    model = toy_model()
    prompts, lengths = toy_queries()
    device = torch.device("cpu")
    clean = generate_queries(model, prompts, lengths, device, 2, capture_sites=(SOURCE, DOWNSTREAM))
    bases = toy_bases()
    wrong = wrong_source_indices(np.array([0, 0, 1, 1]))
    generated = {}
    for arm in ARMS:
        ops = lesion_operations(arm, bases, bases, clean["state_1_2"], wrong, 42, device)
        generated[arm] = generate_queries(model, prompts, lengths, device, 2, operations=ops)
    for arm in ("clean", "sham", "rescue_only"):
        np.testing.assert_array_equal(generated[arm]["prediction"], clean["prediction"])
        np.testing.assert_array_equal(generated[arm]["ended"], clean["ended"])
    for arm in ("random_remove", "complement_remove"):
        np.testing.assert_allclose(
            generated[arm]["source_delta_norm"],
            generated["target_remove"]["source_delta_norm"],
            atol=1e-7,
        )
    for arm in ("remove_wrong_source_rescue", "remove_random_rescue"):
        np.testing.assert_allclose(
            generated[arm]["downstream_delta_norm"],
            generated["remove_rescue"]["downstream_delta_norm"],
            atol=1e-7,
        )
    for result in generated.values():
        for key, value in result.items():
            if key.endswith("_valid"):
                assert value.all()
    for block in model.blocks:
        assert not block._forward_hooks
    with pytest.raises(ValueError, match="unavailable"):
        lesion_operations("remove_rescue", bases, None, clean["state_1_2"], wrong, 42, device)
    # No-Q sham remains available when the fixed-rank calibration fails.
    assert len(lesion_operations("sham", None, None, None, None, 42, device)(0, 2)) == 1


def test_control_randomness_is_paired_across_batching():
    model = toy_model()
    prompts, lengths = toy_queries()
    device = torch.device("cpu")
    clean = generate_queries(model, prompts, lengths, device, capture_sites=(DOWNSTREAM,))
    bases, wrong = toy_bases(), np.array([1, 0, 3, 2])
    results = []
    for batch_size in (1, 4):
        factory = lesion_operations(
            "remove_random_rescue", bases, bases, clean["state_1_2"], wrong, 9, device
        )
        results.append(
            generate_queries(model, prompts, lengths, device, batch_size, operations=factory)
        )
    np.testing.assert_array_equal(results[0]["prediction"], results[1]["prediction"])
    np.testing.assert_allclose(
        results[0]["downstream_delta_norm"], results[1]["downstream_delta_norm"], atol=1e-6
    )


def test_fp32_projection_is_not_silently_autocast_inside_model_hook():
    torch.manual_seed(6)
    x, clean = torch.randn(3, 12), torch.randn(3, 12)
    q = torch.linalg.qr(torch.randn(12, 3)).Q
    mean = torch.randn(3)
    expected = projection_component(x, q, mean)
    expected_restored, _ = restore_projection(x, clean, q)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = projection_component(x, q, mean)
        restored, _ = restore_projection(x, clean, q)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)


def test_probe_families_keep_roots_separate_and_wrong_sources_do_not_use_truth():
    n = 4
    # Two chains' membership/actual/derived IDs, four independent relations,
    # and three root IDs per chain are sufficient to test plan construction.
    membership = np.arange(8).reshape(2, n)
    actual = np.arange(8, 16).reshape(2, n)
    derived = np.arange(16, 24).reshape(2, n)
    relation = np.full(46, -1)
    person = np.full(46, -1)
    for offset, rel in enumerate((3, 4, 5, 6)):
        relation[24 + offset * n : 28 + offset * n] = rel
        person[24 + offset * n : 28 + offset * n] = np.arange(n)
    world = SimpleNamespace(
        derived_ids=derived,
        actual_ids=actual,
        membership_ids=membership,
        root_ids=np.arange(40, 46).reshape(2, 3),
        relation=relation,
        person=person,
        answers=np.arange(46),
        exceptions=np.array([[False, True, False, False]] * 2),
    )
    plan = make_probe_plan(world, 0, np.array([0, 1]))
    assert len(plan["query_id"]) == 9 * 2 + 3
    root = plan["family"] == FAMILIES.index("group_root")
    np.testing.assert_array_equal(plan["query_id"][root], [40, 41, 42])
    assert (plan["person"][root] == -1).all()
    assert (plan["recipient_old_exception"][root] == -1).all()
    wrong = wrong_source_indices(plan["family"])
    assert (wrong != np.arange(len(wrong))).all()
    np.testing.assert_array_equal(plan["family"][wrong], plan["family"])
    np.testing.assert_array_equal(wrong, wrong_source_indices(plan["family"]))


def test_missing_controls_and_root_population_have_explicit_denominators():
    baseline = {"prediction": np.array([5, 6, 7]), "ended": np.array([True, True, True])}
    generated = {
        "prediction": np.array([5, 0, 7]),
        "ended": np.array([True, True, True]),
        "source_valid": np.array([[True, True], [False, False], [True, True]]),
    }
    plan = {
        "answer": np.array([5, 6, 7]),
        "family": np.array([0, 0, 1]),
        "recipient_old_exception": np.array([0, 1, -1]),
    }
    rows = score_rows(generated, baseline, plan, "test", labels=("default", "group_root"))
    default = next(row for row in rows if row["family"] == "default" and row["population"] == "all")
    assert default["requested"] == 2 and default["available"] == 1 and default["unavailable"] == 1
    assert default["delta_accuracy"] == 0
    for row in rows:
        if row["family"] == "group_root" and row["population"] != "all":
            assert row["available"] == 0 and row["accuracy"] is None
    paired = paired_score_rows(
        generated, baseline, baseline, plan, "control", "clean", labels=("default", "group_root")
    )
    all_default = next(
        row for row in paired if row["family"] == "default" and row["population"] == "all"
    )
    assert all_default["common_available"] == 1
    assert all_default["left_unavailable"] == 1
    assert all_default["accuracy_difference"] == 0


def test_donor_projection_controls_match_target_delta_and_unavailable_is_not_no_effect():
    bases = toy_bases()
    device = torch.device("cpu")
    recipient = np.zeros((2, 12), dtype=np.float32)
    donor = np.ones((2, 12), dtype=np.float32)
    donor[1, 2:6] = 0  # target nonzero, both projected controls unavailable on this row
    results = {}
    for arm in (
        "sham",
        "full_donor",
        "full_norm_random",
        "projected_target",
        "projected_random",
        "projected_complement",
    ):
        operation = transfer_operations(arm, bases, recipient, donor, 8, device)(0, 2)[0][2]
        results[arm] = operation(torch.from_numpy(recipient))
    target_norm = torch.linalg.vector_norm(results["projected_target"][1]["delta"], dim=-1)
    for arm in ("projected_random", "projected_complement"):
        norms = torch.linalg.vector_norm(results[arm][1]["delta"], dim=-1)
        torch.testing.assert_close(norms[:1], target_norm[:1])
        assert results[arm][1]["valid"].tolist() == [True, False]
    full_norm = torch.linalg.vector_norm(results["full_donor"][1]["delta"], dim=-1)
    torch.testing.assert_close(
        full_norm, torch.linalg.vector_norm(results["full_norm_random"][1]["delta"], dim=-1)
    )


def test_failed_intervention_still_cleans_up_hooks():
    model = toy_model()
    prompts, lengths = toy_queries()

    def fail(_value):
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        generate_queries(
            model,
            prompts,
            lengths,
            torch.device("cpu"),
            operations=lambda _b, _e: [("source", SOURCE, fail)],
        )
    assert all(not block._forward_hooks for block in model.blocks)


def test_chain_smoke_writes_raw_predictions_and_resumes_without_model_calls(tmp_path, monkeypatch):
    import csv

    import llm_memory_editability.bios_mechanism_causal as causal

    model = toy_model()
    n, size = 4, 46
    relation, person = np.full(size, -1), np.full(size, -1)
    for offset, rel in enumerate((3, 4, 5, 6)):
        relation[24 + offset * n : 28 + offset * n] = rel
        person[24 + offset * n : 28 + offset * n] = np.arange(n)
    prompts = np.tile(np.array([1, 5, 6, 2, 0]), (size, 1))
    prompts[:, 1] = 5 + np.arange(size) % 4
    lengths = np.full(size, 4)
    prompts[16:24] = np.column_stack(
        [np.ones(8), 5 + np.arange(8) % 4, np.full(8, 6), np.full(8, 7), np.full(8, 2)]
    )
    lengths[16:24] = 5
    world = SimpleNamespace(
        seed=0,
        prompts=prompts,
        lengths=lengths,
        derived_ids=np.arange(16, 24).reshape(2, n),
        actual_ids=np.arange(8, 16).reshape(2, n),
        membership_ids=np.arange(8).reshape(2, n),
        root_ids=np.arange(40, 46).reshape(2, 3),
        relation=relation,
        person=person,
        answers=np.arange(size) % 24,
        exceptions=np.array([[False, True, False, True]] * 2),
    )
    recipients, donors = np.array([0, 1] * 3), np.array([2, 3] * 3)
    transfer = {
        "case_id": np.arange(6),
        "recipient": recipients,
        "donor": donors,
        "pair_type": np.repeat(np.arange(3), 2),
        "candidate_count": np.ones(6),
        "recipient_old_exception": world.exceptions[0, recipients],
        "same_actual": np.zeros(6, dtype=bool),
        "same_default": np.zeros(6, dtype=bool),
        "recipient_default": world.answers[world.derived_ids[0, recipients]],
        "recipient_actual": world.answers[world.actual_ids[0, recipients]],
        "donor_default": world.answers[world.derived_ids[0, donors]],
        "donor_actual": world.answers[world.actual_ids[0, donors]],
    }
    monkeypatch.setattr(
        causal, "_fit_chain", lambda *_args: {"source": toy_bases(), "downstream": toy_bases()}
    )
    monkeypatch.setattr(causal, "make_transfer_plan", lambda *_args: transfer)
    device = torch.device("cpu")
    archived = generate_queries(model, world.prompts, world.lengths, device, 2)
    first = causal.run_chain(model, world, 0, {}, tmp_path, device, 2, 1, archived)
    assert all(status == "complete" for status in first["arms"].values())
    assert all(status == "complete" for status in first["transfer_arms"].values())
    with (tmp_path / "necessity.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(ARMS) * len(FAMILIES) * 3
    with np.load(tmp_path / "remove_rescue.npz") as result:
        assert result["prediction"].shape == (21,)
        assert result["source_valid"].shape == result["downstream_valid"].shape == (21, 2)

    def fail(*_args):
        raise RuntimeError("Resume must use the complete cached node")

    monkeypatch.setattr(model, "forward", fail)
    second = causal.run_chain(model, world, 0, {}, tmp_path, device, 2, 1, archived)
    assert first == second
