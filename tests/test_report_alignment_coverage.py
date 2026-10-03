"""Guard pairing, exposure, complete matrices, and independent-world aggregation."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_alignment_coverage", REPOSITORY / "scripts/report_alignment_coverage.py"
)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write(path, value):
    path.write_text(json.dumps(value))


def fixture_matrix(tmp_path, phase="development"):
    config = json.loads((REPOSITORY / f"configs/alignment-coverage-{phase}-v1.json").read_text())
    config_path = tmp_path / "config.json"
    write(config_path, config)
    root = tmp_path / "batch"
    for spec in config["specs"]:
        path = root / "runs" / spec["name"]
        path.mkdir(parents=True)
        identity = {
            "spec": spec,
            "source": {"coverage.py": "source-hash"},
            "implementation_sha256": "source-hash",
            "world_sha256": str(spec["world"]),
            "initial_model_sha256": str(spec["initialization"]),
        }
        coverage, normalization = spec["alignment_coverage"], spec["alignment_normalization"]
        selected = {"none": 0, "composition_only": 64, "all_atomic_and_composition": 192}[coverage]
        denominator = 64 if normalization == "selected_mean" else 192
        unique = {
            name: int(
                coverage == "all_atomic_and_composition"
                or (coverage == "composition_only" and name == "train_composite")
            )
            * 3
            for name in reporter.STRATA
        }
        unique["total"] = sum(unique.values())
        write(
            path / "run.json",
            {
                **identity,
                "strata": dict.fromkeys(reporter.STRATA, 3),
                "bridge_ce_coverage": "all_atomic_and_composition",
                "geometry_normalization": normalization,
                "geometry_unique_examples": unique,
            },
        )
        accuracy = {
            "none": 0.1,
            "composition_batch": 0.3,
            "composition_selected": 0.5,
            "all_batch": 0.9,
        }[spec["arm"]]
        metrics = {
            split: {
                "n": 3,
                "accuracy": accuracy,
                "answer_accuracy": accuracy,
                "answer_nll": 1.0 - accuracy,
            }
            for split in reporter.SPLITS
        }
        for split in ("familiar_test", "strict_test"):
            metrics[split].update(
                atomic_correct_coverage=1.0, conditional_accuracy=accuracy, autonomous_two_calls=1.0
            )
        history, checkpoints, audited = [], [], []
        (path / "model.pt").write_bytes(b"synthetic endpoint checkpoint; never executed")
        model_hash = reporter.sha256(path / "model.pt")
        for step in spec["nodes"]:
            counts = np.full(3, step * 64 // 3, dtype=np.int64)
            counts[: step * 64 % 3] += 1
            arrays = {f"stratum{j}": counts for j in range(3)}
            np.savez(path / f"exposures-{step:06d}.npz", **arrays)
            presentations = {k: n // 3 * step * 64 for k, n in unique.items()}
            row = {
                "step": step,
                "metrics": metrics,
                "examples": step * 192,
                "supervised_tokens": step * 64 * 23,
                "auxiliary_target_presentations": step * 192,
                "composition_epochs": step * 64 / 3,
                "estimated_matmul_training_flops": step * 123,
                "sample_stream_sha256": f"stream-{spec['world']}-{step}",
                "geometry_target_presentations": presentations,
                "alignment_normalization": normalization,
                "alignment_weight": 0.3,
                "alignment_selected_count": selected if step else None,
                "alignment_denominator": denominator if step else None,
                "alignment_selected_mean": float(selected > 0) if step else None,
                "alignment_batch_mean": selected / 192 if step else None,
                "alignment_loss": selected / denominator if step else None,
                "alignment_all_mean": 1.0 if step else None,
                "text_ce": 0.1 if step else None,
                "bridge_ce": 0.2 if step else None,
            }
            history.append(row)
            checkpoints.append({"step": step, "sha256": model_hash, "file": f"model-{step:06d}.pt"})
            audited.append(
                {"step": step, "passed": True, "predictions_recomputed": step == spec["steps"]}
            )
        np.savez(path / "exposures.npz", **arrays)
        write(path / "learning.json", history)
        write(path / "checkpoints.json", checkpoints)
        final = {
            k: history[-1][k]
            for k in ("metrics", "sample_stream_sha256", "geometry_target_presentations")
        }
        write(
            path / "complete.json",
            {
                **identity,
                **final,
                "model_sha256": model_hash,
                "nodes_saved": spec["nodes"],
                "training_seconds": 1.0,
            },
        )
        write(
            path / "audit.json",
            {**final, "passed": True, "nodes": audited, "world_sha256": identity["world_sha256"]},
        )
    return root, config_path


@pytest.mark.parametrize("phase,count,worlds", [("development", 4, 1), ("confirmation", 24, 3)])
def test_complete_endpoints_preserve_coverage_and_paired_contrasts(tmp_path, phase, count, worlds):
    root, config = fixture_matrix(tmp_path, phase)
    summary = reporter.report(root, phase, config)
    assert summary["runs"] == count and summary["independent_worlds"] == worlds
    endpoints = {row["arm"]: row for row in summary["endpoints"]}
    assert endpoints["composition_batch"]["geometry"]["total_coefficient"] == pytest.approx(0.1)
    assert endpoints["composition_selected"]["geometry"]["total_coefficient"] == pytest.approx(0.3)
    assert endpoints["all_batch"]["geometry"]["total_coefficient"] == pytest.approx(0.3)
    assert endpoints["none"]["geometry"]["target_presentations"]["total"] == 0
    contrasts = {row["contrast"]: row for row in summary["contrasts"]}
    assert contrasts["all_batch_minus_composition_batch"]["values"][
        "strict_test.accuracy"
    ] == pytest.approx(0.6)
    assert contrasts["all_batch_minus_composition_selected"]["per_world"][0]["values"][
        "strict_test.accuracy"
    ] == pytest.approx(0.4)
    assert (root / f"report-{phase}/summary.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_run",
        "missing_audit",
        "no_reload",
        "initialization",
        "stream",
        "ce_exposure",
        "geometry_denominator",
        "geometry_targets",
        "sample_counts",
        "checkpoint",
        "audit_metrics",
    ],
)
def test_broken_pair_or_audit_never_emits_summary(tmp_path, damage):
    root, config = fixture_matrix(tmp_path)
    path = sorted((root / "runs").iterdir())[0]
    if damage in ("missing_run", "missing_audit"):
        (path / ("run.json" if damage == "missing_run" else "audit.json")).unlink()
    elif damage == "checkpoint":
        (path / "model.pt").write_bytes(b"modified after audit")
    elif damage == "audit_metrics":
        data = reporter.read_json(path / "audit.json")
        data["metrics"]["strict_test"]["accuracy"] += 0.1
        write(path / "audit.json", data)
    elif damage == "sample_counts":
        with np.load(path / "exposures.npz") as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["stratum0"][0] -= 1
        arrays["stratum0"][1] += 1
        for name in ("exposures.npz", "exposures-016000.npz"):
            np.savez(path / name, **arrays)
    elif damage == "initialization":
        for filename in ("run.json", "complete.json"):
            data = reporter.read_json(path / filename)
            data["initial_model_sha256"] = "unpaired initialization"
            write(path / filename, data)
    elif damage == "no_reload":
        data = reporter.read_json(path / "audit.json")
        data["nodes"][-1]["predictions_recomputed"] = False
        write(path / "audit.json", data)
    else:
        data = reporter.read_json(path / "learning.json")
        last = data[-1]
        if damage == "stream":
            last["sample_stream_sha256"] = "unpaired stream"
            for filename in ("complete.json", "audit.json"):
                other = reporter.read_json(path / filename)
                other["sample_stream_sha256"] = last["sample_stream_sha256"]
                write(path / filename, other)
        elif damage == "ce_exposure":
            last["auxiliary_target_presentations"] -= 1
        elif damage == "geometry_denominator":
            last["alignment_denominator"] *= 3
        else:
            last["geometry_target_presentations"]["total"] += 1
        write(path / "learning.json", data)
    with pytest.raises((ValueError, FileNotFoundError)):
        reporter.report(root, "development", config)
    assert not (root / "report-development/summary.json").exists()


def test_confirmation_requires_all_predeclared_worlds():
    config = reporter.read_json(REPOSITORY / "configs/alignment-coverage-confirmation-v1.json")
    reporter.planned_specs(config, "confirmation")
    smaller = copy.deepcopy(config)
    smaller["specs"] = [s for s in smaller["specs"] if s["world"] != 810103]
    with pytest.raises(ValueError, match="Incomplete planned matrix"):
        reporter.planned_specs(smaller, "confirmation")


def test_worlds_are_equal_and_undefined_conditions_are_not_dropped():
    rows = [
        {"world": 1, "initialization": 1, "values": {"accuracy": 1.0, "conditional": 0.8}},
        {"world": 1, "initialization": 2, "values": {"accuracy": 1.0, "conditional": None}},
        {"world": 2, "initialization": 1, "values": {"accuracy": 0.0, "conditional": 0.2}},
    ]
    result = reporter.hierarchical_mean(rows)
    assert result["values"]["accuracy"] == 0.5
    assert result["values"]["conditional"] is None
    assert result["per_world"][0]["defined_initializations"]["conditional"] == 1
    assert result["defined_worlds"]["conditional"] == 1
