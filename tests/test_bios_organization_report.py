"""Reports must retain missing outcomes, audit every checkpoint and respect models."""

import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.bios_data import (
    N_BASE,
    N_QUERIES,
    REVISION,
    SOURCE_FOLDER,
    make_world,
    paired_edit,
)

spec = importlib.util.spec_from_file_location(
    "organization_report", Path(__file__).parents[1] / "scripts/summarize_bios_organization.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def world(tmp_path):
    source = tmp_path / "source"
    fields = source / SOURCE_FOLDER / "fields"
    fields.mkdir(parents=True)
    files = []
    for name, count in {
        "first_name": 400,
        "middle_name": 400,
        "last_name": 1000,
        "city": 200,
        "company": 263,
        "university": 300,
        "field": 100,
    }.items():
        path = fields / f"{name}.txt"
        path.write_text("\n".join(f"{name}_{i}" for i in range(count)))
        files.append(
            {
                "path": str(path.relative_to(source)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json(source / "manifest.json", {"revision": REVISION, "files": files})
    return make_world(0, source)


@pytest.fixture
def edit_artifacts(tmp_path, world):
    pytest.importorskip("torch")
    from llm_memory_editability.bios_edit import score_edit

    answers = world.answers
    pair = paired_edit(world, support=0, k=1)
    target = pair["exception"]
    sets = dict(
        **{key: pair[key] for key in ("E", "D", "strata", "heldout", "replay")},
        target=target,
        old_correct=np.ones(N_QUERIES, dtype=bool),
    )
    directory = tmp_path / "run/edits/support-0-exception-mlp"
    directory.mkdir(parents=True)
    baseline = dict(
        prediction=answers,
        ended=np.ones(N_QUERIES, dtype=bool),
        correct=np.ones(N_QUERIES, dtype=bool),
        value_nll=np.full(N_QUERIES, 0.1),
    )
    np.savez(directory.parent.parent / "predictions-14336.npz", **baseline)
    config = {
        "steps": 512,
        "scope": "mlp",
        "kind": "exception",
        "support": 0,
        "lr": 3e-5,
        "retention_kl_weight": 1.0,
        "window_zero_based": [3, 4, 5],
        "branch": None,
        "branch_parameters": 0,
        "selection": "prespecified_unmatched",
    }
    write_json(directory / "config.json", config)
    np.savez(directory / "sets.npz", **sets)
    trajectory = []
    for step in report.EDIT_CHECKPOINTS:
        prediction = answers.copy() if step == 0 else target.copy()
        arrays = dict(
            prediction=prediction,
            ended=np.ones(N_QUERIES, dtype=bool),
            correct=prediction == target,
            value_nll=np.full(N_QUERIES, 0.1),
        )
        np.savez(directory / f"predictions-{step}.npz", **arrays)
        trajectory.append(
            {"step": step, **score_edit(world, sets, target, arrays, sets["old_correct"])}
        )
    write_json(directory / "trajectory.json", trajectory)
    write_json(
        directory / "complete.json",
        {
            "status": "complete",
            **config,
            "final": trajectory[-1],
            "first_observed_joint_step": 1,
            "joint_evaluable": True,
        },
    )
    return directory, world, sets


def test_audit_recomputes_intermediate_checkpoints(edit_artifacts):
    directory, world, _ = edit_artifacts
    result = report.audit_case(directory, world, [])
    assert result["passed"]
    assert len(result["checkpoints"]) == 11
    with np.load(directory / "predictions-1.npz") as source:
        arrays = dict(source)
    arrays["prediction"][2048] += 7  # Final checkpoint is intact; intermediate is corrupted.
    np.savez(directory / "predictions-1.npz", **arrays)
    result = report.audit_case(directory, world, [])
    assert not result["passed"]
    assert any("step 1" in error for error in result["errors"])
    assert any("correctness" in error for error in result["errors"])


def test_missing_retention_denominator_is_unevaluable(edit_artifacts):
    directory, world, sets = edit_artifacts
    from llm_memory_editability.bios_edit import score_edit

    sets["old_correct"][sets["strata"] == 0] = False
    arrays = dict(
        prediction=sets["target"],
        ended=np.ones(N_QUERIES, dtype=bool),
        correct=np.ones(N_QUERIES, dtype=bool),
        value_nll=np.zeros(N_QUERIES),
    )
    metrics = score_edit(world, sets, sets["target"], arrays, sets["old_correct"])
    assert metrics["joint_pass"] is None
    assert metrics["U_heldout_destruction"]["0"] == {"known": 0, "broken": 0, "rate": None}
    # JSON metric comparisons must not turn an unavailable joint result into False.
    assert report.matches(metrics, metrics)
    assert not report.matches(None, False)


def test_partial_study_preserves_expected_denominators(tmp_path):
    root = tmp_path / "results"
    directory = root / "world-0-seed-0-A"
    write_json(
        directory / "config.json", {"condition": "A", "world_seed": 0, "seed": 0, "steps": 14336}
    )
    write_json(
        directory / "learning.json",
        [
            {
                "step": 896,
                "base_accuracy": 0.8,
                "derived_accuracy": 0.6,
                "value_nll": 0.5,
                "learning_threshold_passed": False,
                "strata": {"actual_old_exception": 0.5},
            }
        ],
    )
    result = report.run(
        Namespace(root=root, world_root=tmp_path / "missing-worlds", output=None, no_plots=True)
    )
    assert result["counts"]["expected_old_models"] == 12
    assert result["counts"]["expected_edits"] == 96
    assert result["counts"]["completed_old_models"] == 0
    assert result["learning"][0]["status"] == "in_progress"
    assert result["learning"][0]["base_accuracy"] == 0.8
    assert result["learning_aggregates"][0]["mean_base_accuracy"] is None
    assert all(pair["old_knowledge_quality_eligible"] is None for pair in result["learning_pairs"])
    assert (root / "report/report.md").exists()
    assert "NaN" not in (root / "report/summary.json").read_text()


def test_quality_gate_rejects_low_quality_and_unequal_nll():
    good = dict(base_accuracy=0.999, derived_accuracy=0.999, value_nll=0.01)
    assert report.quality_pair(good, good)
    assert not report.quality_pair(good, {**good, "derived_accuracy": 0.989})
    assert not report.quality_pair(good, {**good, "value_nll": 0.12})
    assert report.quality_pair(good, {}) is None


def test_model_aggregation_does_not_select_partial_supports():
    def case(seed, support, accuracy):
        return dict(
            condition="A",
            scope="mlp",
            kind="exception",
            world=0,
            seed=seed,
            support=support,
            complete=True,
            audit_passed=True,
            joint_final_passed=None,
            joint_ever_passed=None,
            **{field: accuracy for field in report.EDIT_FIELDS},
            final={
                "U_heldout_destruction": {
                    g: {"known": 0, "broken": 0, "rate": None} for g in report.RETENTION_GROUPS
                }
            },
        )

    # One complete model scores 0.2; another has only its first (perfect) support.
    cases = [case(0, 0, 0.1), case(0, 1, 0.3), case(1, 0, 1.0)]
    aggregates, models = report.aggregate_edits(cases)
    result = next(
        row
        for row in aggregates
        if (row["condition"], row["scope"], row["kind"]) == ("A", "mlp", "exception")
    )
    assert result["completed_cases"] == 3
    assert result["complete_two_support_models"] == 1
    assert result["mean_E"] == pytest.approx(0.2)
    assert result["joint_final_evaluable"] == 0
    assert result["joint_final_passed"] == 0
    assert result["mean_U0_damage"] is None
    assert len(models) == 1


@pytest.fixture
def learning_artifacts(tmp_path, world):
    from llm_memory_editability.bios_data import array_hash, rng_for
    from llm_memory_editability.bios_organization import make_documents

    directory = tmp_path / "study/world-0-seed-0-A"
    directory.mkdir(parents=True)
    plan = report.study_manifest(tmp_path / "absent", [])[0]
    write_json(directory.parent / "study-config.json", plan)
    documents = make_documents(world, "A", 0)
    rng = rng_for(0, 702)
    derived = np.concatenate([rng.permutation(2048) + N_BASE for _ in range(196)]).reshape(-1, 28)
    np.savez(directory / "schedule.npz", documents=documents, derived=derived)
    config = {
        "world_seed": 0,
        "seed": 0,
        "condition": "A",
        "steps": 14336,
        "run_kind": "full",
        "protocol": "v2.4",
        "organization_seed": 0,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 16,
        "documents_per_step": 16,
        "derived_queries_per_step": 28,
        "facts_per_document": 7,
        "cross_document_attention": False,
        "document_weight": 0.8,
        "derived_weight": 0.2,
        "checkpoints": list(report.TRAIN_CHECKPOINTS),
        "planned_supervised_tokens": 14336 * 280,
        "planned_input_tokens_including_padding": 14336 * 840,
        "planned_total_presentations": 14336 * 140,
        "planned_train_matmul_flops_estimate": 1e15,
        "model": {
            "width": 768,
            "layers": 8,
            "heads": 12,
            "context": 128,
            "vocab_size": world.vocab_size,
        },
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "lengths_sha256": array_hash(world.lengths),
        "model_initial_sha256": "a" * 64,
        "core_source_sha256": {"fixture.py": "b" * 64},
        "documents_sha256": array_hash(documents),
        "derived_schedule_sha256": array_hash(derived),
    }
    write_json(directory / "config.json", config)
    timeline = []
    for step in report.TRAIN_CHECKPOINTS:
        exposure = np.full(N_QUERIES, step // 128, dtype=np.int64)
        exposure[:N_BASE][world.relation[:N_BASE] == 1] *= 32
        exposure[N_BASE:] = np.bincount(derived[:step].ravel() - N_BASE, minlength=2048)
        arrays = dict(
            prediction=world.answers,
            ended=np.ones(N_QUERIES, dtype=bool),
            correct=np.ones(N_QUERIES, dtype=bool),
            value_nll=np.full(N_QUERIES, 0.1),
            exposure=exposure,
            slot_exposure=np.repeat(exposure[:N_BASE, None] // 7, 7, axis=1),
            lr_weighted_exposure=exposure.astype(float) * 1e-4,
        )
        np.savez(directory / f"predictions-{step}.npz", **arrays)
        # Learning audit never loads model weights; only trained run does.
        (directory / f"model-{step}.pt").touch()
        metrics, _ = report.score_learning(world, arrays)
        timeline.append(
            {
                "step": step,
                **metrics,
                **{
                    f"{field}_sha256": array_hash(arrays[field])
                    for field in ("exposure", "slot_exposure", "lr_weighted_exposure")
                },
            }
        )
    write_json(directory / "learning.json", timeline)
    complete = {
        "step": 14336,
        "status": "complete",
        "final": timeline[-1],
        "exposure_sha256": timeline[-1]["exposure_sha256"],
        "slot_exposure_sha256": timeline[-1]["slot_exposure_sha256"],
        "model_final_sha256": "c" * 64,
    }
    write_json(directory / "complete.json", complete)
    return directory, world, config, timeline


def test_learning_requires_full_identity_and_budget(learning_artifacts):
    directory, world, config, _ = learning_artifacts
    row, _ = report.inspect_learning(directory, 0, 0, "A", [], truth=world)
    assert row["complete"], row["audit_errors"]
    config["steps"] = 2
    config["run_kind"] = "smoke"
    write_json(directory / "config.json", config)
    row, _ = report.inspect_learning(directory, 0, 0, "A", [], truth=world)
    assert not row["complete"]
    assert row["audit_passed"] is False
    assert "config missing/mismatched steps" in row["audit_errors"]
    (directory / "config.json").unlink()
    row, _ = report.inspect_learning(directory, 0, 0, "A", [], truth=world)
    assert not row["complete"]
    assert "config missing/mismatched condition" in row["audit_errors"]


def test_learning_recomputes_saved_predictions_and_completion(learning_artifacts):
    directory, world, _, _ = learning_artifacts
    complete = json.loads((directory / "complete.json").read_text())
    complete["final"]["base_accuracy"] = 0.5
    write_json(directory / "complete.json", complete)
    with np.load(directory / "predictions-896.npz") as source:
        arrays = dict(source)
    arrays["prediction"][0] += 1
    np.savez(directory / "predictions-896.npz", **arrays)
    row, _ = report.inspect_learning(directory, 0, 0, "A", [], truth=world)
    assert not row["complete"]
    assert any("completion final differs" in error for error in row["audit_errors"])
    assert any("step 896: saved correctness" in error for error in row["audit_errors"])


def test_low_learning_accuracy_is_completed_not_failed(learning_artifacts):
    directory, world, _, timeline = learning_artifacts
    with np.load(directory / "predictions-14336.npz") as source:
        arrays = dict(source)
    arrays["ended"][:] = False
    arrays["correct"][:] = False
    np.savez(directory / "predictions-14336.npz", **arrays)
    metrics, _ = report.score_learning(world, arrays)
    timeline[-1].update(metrics)
    write_json(directory / "learning.json", timeline)
    complete = json.loads((directory / "complete.json").read_text())
    complete["final"] = timeline[-1]
    write_json(directory / "complete.json", complete)
    row, _ = report.inspect_learning(directory, 0, 0, "A", [], truth=world)
    assert row["complete"], row["audit_errors"]
    assert row["base_accuracy"] == 0
    assert row["status"] == "complete"


def test_missing_edit_checkpoint_in_both_sources_fails(edit_artifacts):
    directory, world, _ = edit_artifacts
    (directory / "predictions-0.npz").unlink()
    trajectory = json.loads((directory / "trajectory.json").read_text())[1:]
    write_json(directory / "trajectory.json", trajectory)
    audit = report.audit_case(directory, world, [])
    assert not audit["passed"]
    assert sum("prescribed checkpoint set" in message for message in audit["errors"]) == 2


def test_edit_metadata_and_canonical_target_are_verified(edit_artifacts):
    directory, world, sets = edit_artifacts
    config = json.loads((directory / "config.json").read_text())
    config["lr"] = 1e-2
    write_json(directory / "config.json", config)
    sets["target"][sets["E"][0]] += 1
    np.savez(directory / "sets.npz", **sets)
    audit = report.audit_case(directory, world, [])
    assert not audit["passed"]
    assert "edit config missing/mismatched lr" in audit["errors"]
    assert "saved edit target differs from canonical paired edit" in audit["errors"]


def test_edit_baseline_is_linked_to_old_endpoint(edit_artifacts):
    directory, world, _ = edit_artifacts
    endpoint = directory.parent.parent / "predictions-14336.npz"
    with np.load(endpoint) as source:
        arrays = dict(source)
    arrays["prediction"][0] += 1
    np.savez(endpoint, **arrays)
    audit = report.audit_case(directory, world, [])
    assert not audit["passed"]
    assert "step 0: prediction differs from the old-model endpoint" in audit["errors"]


def test_cross_condition_initial_and_exposure_hashes(learning_artifacts):
    directory, _, config, timeline = learning_artifacts
    root = directory.parent
    rows, trajectories = [], {}
    for world in report.WORLDS:
        for seed in report.SEEDS:
            for condition in report.CONDITIONS:
                rows.append(
                    dict(
                        run=f"world-{world}-seed-{seed}-{condition}",
                        world=world,
                        seed=seed,
                        condition=condition,
                        reported_complete=False,
                        audit_errors=[],
                    )
                )
                trajectories[world, seed, condition] = []
    other = dict(
        config,
        condition="B",
        documents_sha256="b" * 64,
        model_initial_sha256="different-initial-weights",
    )
    write_json(root / "world-0-seed-0-B/config.json", other)
    trajectories[0, 0, "A"] = [timeline[0]]
    trajectories[0, 0, "B"] = [dict(timeline[0], slot_exposure_sha256="different-slots")]
    result = report.cross_condition_audit(root, rows, trajectories, [])
    pair = next(p for p in result if p["configs_available"])
    assert pair["passed"] is False
    assert any("model_initial_sha256" in error for error in pair["errors"])
    assert any("slot_exposure_sha256" in error for error in pair["errors"])


def test_quality_subset_keeps_model_pair_denominators():
    pairs = []
    for world, quality, deltas in ((0, True, (0.1, 0.3)), (1, False, (0.8, 1.0))):
        for support, delta in enumerate(deltas):
            pairs.append(
                dict(
                    world=world,
                    seed=0,
                    left="A",
                    right="B",
                    scope="mlp",
                    kind="exception",
                    support=support,
                    identical_edit_sets=True,
                    old_knowledge_quality_eligible=quality,
                    **{f"left_minus_right_{f}": delta for f in report.EDIT_FIELDS},
                )
            )
    models, aggregates = report.aggregate_paired_edits(pairs)
    assert len(models) == 2
    selected = {
        r["subset"]: r
        for r in aggregates
        if (r["left"], r["right"], r["scope"], r["kind"]) == ("A", "B", "mlp", "exception")
    }
    assert selected["all"]["n_model_pairs"] == 2
    assert selected["all"]["mean_left_minus_right_E"] == pytest.approx(0.55)
    assert selected["quality_matched"]["n_model_pairs"] == 1
    assert selected["quality_matched"]["mean_left_minus_right_E"] == pytest.approx(0.2)


def test_edit_execution_provenance_cannot_reuse_other_scope_run(edit_artifacts):
    directory, world, _ = edit_artifacts
    sources = {
        f"src/llm_memory_editability/{name}": "frozen-hash"
        for name in ("bios_edit.py", "bios_model.py", "bios_data.py", "bios_train.py")
    }
    write_json(directory.parent.parent / "config.json", {"source_sha256": sources})
    execution = dict(
        world_step=14336,
        steps=512,
        window=3,
        lr=3e-5,
        retention=1.0,
        supports=[0, 1],
        scopes=["mlp", "all"],
        branches=False,
        checkpoint=str(directory.parent.parent / "model-14336.pt"),
        source_sha256=sources,
    )
    write_json(directory.parent / "run.json", execution)
    meta = dict(scope="mlp", support=0, kind="exception")
    case, _ = report.inspect_edit(directory, meta, world, [])
    assert case["audit_passed"], case["audit_errors"]
    execution["scopes"] = ["down"]
    write_json(directory.parent / "run.json", execution)
    case, _ = report.inspect_edit(directory, meta, world, [])
    assert case["audit_passed"] is False
    assert "editing run metadata missing/mismatched scopes" in case["audit_errors"]
