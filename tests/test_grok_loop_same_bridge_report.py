"""Independent recounts, nested world weighting, complete matrices and evidence hashes."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def reporter():
    path = Path(__file__).resolve().parents[1] / "scripts/report_grok_loop_same_bridge.py"
    spec = importlib.util.spec_from_file_location("same_bridge_report_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spec(world=1, seed=1, phase="confirmation"):
    return {
        "world_seed": world,
        "initialization": seed,
        "phase": phase,
        "hops": 2,
        "architecture": "l1",
        "layers": 1,
        "repeats": 4,
        "steps": 100,
    }


def record(reporter, world, seed, value, phase="confirmation", n=10):
    return {
        **reporter.identity(
            {"run": f"w{world}-s{seed}", "phase": phase, "spec": spec(world, seed, phase)}
        ),
        "kind": "score",
        "group": "paired",
        "condition": "id_full",
        "component": "full",
        **{metric: value for metric in reporter.METRICS},
        "n_recipients": n,
        "selected_recipients": n,
        "total_recipients": n,
        "n_donor_evaluations": n * 3,
        "coverage": 1.0 if n else 0.0,
    }


def expected(worlds=(1, 2), seeds=(1, 2)):
    return [
        {"run": f"w{world}-s{seed}", "phase": "confirmation", "spec": spec(world, seed)}
        for world in worlds
        for seed in seeds
    ]


def test_donors_and_pairs_receive_one_vote_per_recipient(reporter):
    indices = np.array([0, 0, 0, 1])
    values = {metric: np.array([0.0, 0.0, 0.0, 1.0]) for metric in reporter.METRICS}
    scored = reporter.balanced_summary(values, indices, np.array([True, True, True]))
    assert scored["complete_accuracy"] == 0.5
    assert scored["n_recipients"] == 2 and scored["n_donor_evaluations"] == 4
    assert scored["coverage"] == pytest.approx(2 / 3)
    absent = reporter.balanced_summary(values, indices, np.array([False, False, True]))
    assert absent["complete_accuracy"] is None and absent["selected_recipients"] == 1
    assert absent["n_recipients"] == 0 and absent["coverage"] == 0.0


def test_initializations_nest_within_equal_worlds_and_never_weight_by_queries(reporter):
    rows = [record(reporter, 1, seed, value) for seed, value in enumerate((0.0, 0.0, 1.0))]
    rows += [record(reporter, 2, seed, value, n=1000) for seed, value in enumerate((0.8, 1.0))]
    registered = [
        {
            "run": row["run"],
            "phase": row["phase"],
            "spec": spec(row["world_seed"], row["initialization"]),
        }
        for row in rows
    ]
    worlds, groups = reporter.aggregate(rows, registered)
    assert len(worlds) == 2 and len(groups) == 1
    metric = groups[0]["metrics"]["complete_accuracy"]
    assert metric["mean"] == pytest.approx((1 / 3 + 0.9) / 2)
    assert metric["min"] == pytest.approx(1 / 3) and metric["max"] == pytest.approx(0.9)
    assert groups[0]["registration_complete"] is True


def test_missing_initializations_and_empty_denominators_remain_explicit(reporter):
    rows = [record(reporter, 1, 1, 0.7), record(reporter, 2, 1, None, n=0)]
    _, groups = reporter.aggregate(rows, expected())
    group = groups[0]
    assert group["registration_complete"] is False
    assert group["missing_world_initializations"] == [(1, 2), (2, 2)]
    assert group["metrics"]["complete_accuracy"]["mean"] == 0.7
    assert group["metrics"]["complete_accuracy"]["worlds_with_denominator"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        reporter.aggregate([rows[0], copy.deepcopy(rows[0])], expected())


def test_phases_architectures_and_component_effects_never_merge(reporter):
    base = record(reporter, 1, 1, 0.2)
    rows = [
        base,
        {**base, "phase": "development"},
        {**base, "architecture": "l2"},
        {**base, "component": "mlp"},
        {**base, "kind": "contrast"},
    ]
    _, groups = reporter.aggregate(rows, expected())
    assert len(groups) == 5


def synthetic_report(reporter):
    n = 3
    arrays = {
        "pair_id_indices": np.array([0, 1, 2, 2]),
        "pair_ood_indices": np.array([0, 0, 1, 2]),
        "pair_recipient_indices": np.array([0, 0, 1, 1]),
    }
    masks = {group: np.array([True, True, False]) for group in reporter.GROUPS}
    masks["all"] = np.ones(n, dtype=bool)
    masks["neither"] = np.array([False, False, True])
    masks["id_only"] = masks["ood_only"] = np.zeros(n, dtype=bool)
    masks["paired_all_atomics_correct"] = np.zeros(n, dtype=bool)
    arrays.update({"subset_" + group: mask for group, mask in masks.items()})
    values, indices = {}, {}
    for condition in reporter.CONDITIONS:
        ix = (
            np.array([0, 0, 1])
            if condition.startswith("id_")
            else (np.array([0, 1, 1]) if condition.startswith("ood_") else np.arange(n))
        )
        indices[condition] = ix
        value = 0.8 if condition == "id_full" else 0.3 if condition == "ood_full" else 0.2
        values[condition] = {metric: np.full(len(ix), value) for metric in reporter.METRICS}
        arrays[condition + "_recipient_indices"] = ix
        arrays.update(
            {condition + "_" + metric: vector for metric, vector in values[condition].items()}
        )
    scores = {
        group: {
            condition: reporter.balanced_summary(values[condition], indices[condition], mask)
            for condition in reporter.CONDITIONS
        }
        for group, mask in masks.items()
    }
    contrasts = {}
    pid, pod, pi = (
        arrays[key] for key in ("pair_id_indices", "pair_ood_indices", "pair_recipient_indices")
    )
    for component in reporter.COMPONENTS:
        raw = {
            metric: values["id_" + component][metric][pid] - values["ood_" + component][metric][pod]
            for metric in reporter.METRICS
        }
        adjusted = {
            metric: raw[metric]
            - values["id_baseline"][metric][pid]
            + values["ood_baseline"][metric][pod]
            for metric in reporter.METRICS
        }
        for group, mask in masks.items():
            contrasts[group + ":" + component] = {
                "id_minus_ood": reporter.balanced_summary(raw, pi, mask),
                "baseline_adjusted_id_minus_ood": reporter.balanced_summary(adjusted, pi, mask),
            }
    report = {
        "n_original_queries": n,
        "n_common_queries": 2,
        "n_pairs": 4,
        "conditions": {name: {} for name in reporter.CONDITIONS},
        "scores": scores,
        "contrasts": contrasts,
        "donor_audit": {},
        "engineering": {
            "all_self_and_original_prefix_conditions_equal_baseline": True,
            "both_equals_full_first_execution_layer": True,
            "parameters_unchanged": True,
            "historical_answer_eos_reproduced": True,
            "eos_uses_generated_answer": True,
            "training_updates": 0,
            "mlp_recomputed_after_mixing": False,
            "model_state_sha256_before": "same",
            "model_state_sha256_after": "same",
        },
    }
    return report, arrays


def test_recount_rejects_changed_scores_missing_controls_and_donor_weighting(reporter, tmp_path):
    report, arrays = synthetic_report(reporter)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(path, **arrays)
    reporter.verify_raw_scores(path, report)
    changed = copy.deepcopy(report)
    changed["scores"]["paired"]["id_full"]["complete_accuracy"] += 0.1
    with pytest.raises(ValueError, match="disagree"):
        reporter.verify_raw_scores(path, changed)
    incomplete = copy.deepcopy(report)
    del incomplete["conditions"]["self_full"]
    with pytest.raises(ValueError, match="Incomplete"):
        reporter.verify_raw_scores(path, incomplete)
    contrast = copy.deepcopy(report)
    contrast["contrasts"]["paired:full"]["id_minus_ood"]["target_probability"] = 0.7
    with pytest.raises(ValueError, match="disagree"):
        reporter.verify_raw_scores(path, contrast)


def test_matched_baseline_effects_and_all_conditions_survive_flattening(reporter):
    report, _ = synthetic_report(reporter)
    entry = {"run": "toy", "phase": "confirmation", "spec": spec()}
    rows = reporter.endpoint_records(report, entry)
    assert {row["condition"] for row in rows if row["kind"] == "score"} == set(reporter.CONDITIONS)
    primary = next(
        row
        for row in rows
        if row["group"] == "paired"
        and row["component"] == "full"
        and row["condition"] == "id_minus_ood"
    )
    assert primary["complete_accuracy"] == pytest.approx(0.5)
    effects = {
        row["condition"]: row
        for row in rows
        if row["group"] == "paired"
        and row["component"] == "full"
        and row["condition"].endswith("_minus_baseline")
    }
    assert effects["id_minus_baseline"]["complete_accuracy"] == pytest.approx(0.6)
    assert effects["ood_minus_baseline"]["complete_accuracy"] == pytest.approx(0.1)
    empty = next(
        row
        for row in rows
        if row["group"] == "paired_all_atomics_correct" and row["condition"] == "id_full"
    )
    assert empty["complete_accuracy"] is None and empty["n_recipients"] == 0


def evidence_fixture(reporter, tmp_path):
    root = tmp_path
    config = {
        "experiment": "same-bridge-test",
        "historical_configs": {},
        "expected_runs": {"development": 1, "confirmation": 1},
        "output_root": "results/new",
        "locks": {"development": "development-lock-v3.json"},
    }
    for phase in ("development", "confirmation"):
        historical = root / (phase + ".json")
        run = phase + "-toy"
        reporter.save_json(historical, {"base": spec(phase=phase), "runs": {run: {}}})
        config["historical_configs"][phase] = historical.name
    config_path = root / "followup.json"
    reporter.save_json(config_path, config)
    artifact = root / "docs/development-artifacts" / config["experiment"]
    artifact.mkdir(parents=True)
    phase, run = "development", "development-toy"
    historical_dir = root / "historical"
    historical_dir.mkdir()
    (historical_dir / "weights-0000100.pt").write_bytes(b"frozen-checkpoint")
    donor = root / "donors.npz"
    donor.write_bytes(b"frozen-donors")
    audit = root / "donors.json"
    audit.write_text("{}")
    lock = {
        "phase": phase,
        "config_sha256": reporter.digest(config_path),
        "analysis_source_hashes": {},
        "runs": [
            {
                "run": run,
                "spec": spec(phase=phase),
                "directory": str(historical_dir),
                "inputs": {
                    "weights-0000100.pt": reporter.digest(historical_dir / "weights-0000100.pt")
                },
                "donors_file": donor.name,
                "donors_sha256": reporter.digest(donor),
                "donor_audit_file": audit.name,
                "donor_audit_sha256": reporter.digest(audit),
            }
        ],
    }
    lock_path = artifact / config["locks"][phase]
    reporter.save_json(lock_path, lock)
    out = root / config["output_root"] / phase / run
    out.mkdir(parents=True)
    report, arrays = synthetic_report(reporter)
    report.update({"run": run, "phase": phase, "spec": spec(phase=phase)})
    metadata = {
        "run": run,
        "phase": phase,
        "spec": spec(phase=phase),
        "lock_sha256": reporter.digest(lock_path),
        "analysis_source_hashes": {},
        "donors_sha256": reporter.digest(donor),
    }
    reporter.save_json(out / "metadata.json", metadata)
    reporter.save_json(out / "report.json", report)
    np.savez_compressed(out / "predictions.npz", **arrays)
    reporter.save_json(
        out / "complete.json",
        {
            "run": run,
            "passed": True,
            "lock_sha256": reporter.digest(lock_path),
            "output_hashes": {
                name: reporter.digest(out / name)
                for name in ("metadata.json", "report.json", "predictions.npz")
            },
        },
    )
    return root, config, config_path, out


def test_final_collection_requires_complete_registration_and_checks_output_hashes(
    reporter, tmp_path
):
    root, config, config_path, out = evidence_fixture(reporter, tmp_path)
    with pytest.raises(ValueError, match="Incomplete registered matrix"):
        reporter.collect(config, config_path, root=root)
    rows, registered, manifest, matrix = reporter.collect(
        config, config_path, root=root, allow_partial=True
    )
    assert rows and len(registered) == 2 and len(manifest) == 1
    assert [row["state"] for row in matrix] == ["complete", "missing"]
    assert matrix[1]["reason"] == "phase_not_frozen"
    (out / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="Completed output changed"):
        reporter.collect(config, config_path, root=root, allow_partial=True)


def test_historical_input_hashes_are_verified_even_for_partial_calibration(reporter, tmp_path):
    root, config, config_path, _ = evidence_fixture(reporter, tmp_path)
    (root / "historical/weights-0000100.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Historical input changed"):
        reporter.collect(config, config_path, root=root, allow_partial=True)


def test_all_scientific_figures_write_png_and_pdf_without_changing_scores(reporter, tmp_path):
    pytest.importorskip("matplotlib")
    report, _ = synthetic_report(reporter)
    entries = [
        {"run": phase, "phase": phase, "spec": spec(phase=phase)}
        for phase in ("development", "confirmation")
    ]
    rows = [row for entry in entries for row in reporter.endpoint_records(report, entry)]
    _, groups = reporter.aggregate(rows, entries)
    before = json.dumps(groups, sort_keys=True)
    paths = reporter.plot(groups, tmp_path)
    assert len(paths) == 6
    assert {Path(path).suffix for path in paths} == {".png", ".pdf"}
    assert all(Path(path).stat().st_size > 1000 for path in paths)
    assert json.dumps(groups, sort_keys=True) == before
