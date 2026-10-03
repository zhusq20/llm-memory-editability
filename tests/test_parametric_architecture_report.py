"""Report contracts: audited completion, seed inference, and prerequisite coverage."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "parametric_report", Path(__file__).parents[1] / "scripts/report_parametric_architecture.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study(tmp_path):
    data = {
        "atoms": [{"id": name} for name in ("a1", "a2", "a3")],
        "train_compositions": [{"id": "train1"}],
        "evaluation_compositions": [
            {"id": "c1", "role": "II", "atom_ids": ["a1", "a2"]},
            {"id": "c2", "role": "OO", "atom_ids": ["a2", "a3"]},
        ],
    }
    data_path = tmp_path / "data.json"
    write(data_path, data)
    data_hash = report.digest(data_path)

    def make_run(root, arm, seed):
        name = f"{arm}-{seed}"
        spec = {
            "name": name,
            "architecture": arm,
            "initialization": seed,
            "sampling_seed": seed + 1000,
            "dropout_seed": seed + 2000,
            "steps": 300000,
            "learning_rate": 5e-5,
        }
        path = root / name
        atom_values = [1, 1, 0] if arm == "M8" else [0, 1, 1] if arm == "W8" else [1, 1, 1]
        test_values = (
            {811201: [1, 0], 811202: [1, 1], 811203: [0, 0]}[seed] if arm == "M8" else [0, 0]
        )
        raw = {
            "atomic": [
                {"id": row["id"], "alias_em": value}
                for row, value in zip(data["atoms"], atom_values, strict=True)
            ],
            "train_composition": [{"id": "train1", "alias_em": 1}],
            "test_all": [
                {"id": row["id"], "alias_em": value}
                for row, value in zip(data["evaluation_compositions"], test_values, strict=True)
            ],
        }
        metrics = {
            group: {
                "n": len(rows),
                "alias_em": sum(r["alias_em"] for r in rows) / len(rows),
                "nll": 0.5,
            }
            for group, rows in raw.items()
        }
        metrics["test_oo"] = {"n": 1, "alias_em": test_values[1], "nll": 0.5}
        identity = {"model_sha256": name, "checkpoint_sha256": name + "-checkpoint"}
        write(
            path / "complete.json",
            {"state": "complete", "step": 300000, "independently_reloaded": True, **identity},
        )
        write(
            path / "audit.json", {"passed": True, "pid": 12, "source_training_pid": 11, **identity}
        )
        write(path / "run.json", {"spec": spec, "world_sha256": data_hash, "parameters": 100})
        write(path / "endpoint.json", metrics)
        write(path / "endpoint-predictions.json", raw)
        return spec

    main_root, old_root = tmp_path / "main", tmp_path / "old"
    specs = [
        make_run(main_root / "runs", arm, seed) for arm in report.NEW_ARMS for seed in report.SEEDS
    ]
    old_specs = [
        make_run(old_root / "confirmation", arm, seed)
        for arm in ("standard4", "standard8", "loop4x2")
        for seed in report.SEEDS
    ]
    historic_path = tmp_path / "historical.json"
    write(
        historic_path,
        {
            "runs": old_specs,
            "results_root": str(old_root),
            "phase": "confirmation",
            "data_sha256": data_hash,
        },
    )
    config_path, lock_path = tmp_path / "config.json", tmp_path / "lock.json"
    config = {
        "phase": "main",
        "repository": str(tmp_path),
        "runs": specs,
        "results_root": str(main_root),
        "data_file": str(data_path),
        "data_sha256": data_hash,
        "execution_lock": str(lock_path),
        "reporting": {
            "historical_config": str(historic_path),
            "historical_config_sha256": report.digest(historic_path),
            "historical_reuse_verified": True,
        },
    }

    def freeze():
        write(config_path, config)
        write(lock_path, {"config_sha256": report.digest(config_path)})

    freeze()
    return config_path, tmp_path / "report", config, freeze


def test_reports_seed_statistics_and_keeps_empty_subset_denominators(study):
    config_path, output, _, _ = study
    result = report.build_report(config_path, output)
    effect = next(
        r
        for r in result["paired_summary"]
        if r["treatment"] == "M8"
        and r["reference"] == "W8"
        and r["pool"] == "test_all"
        and r["metric"] == "alias_em"
    )
    assert effect["n_seed_values"] == 3
    assert effect["mean"] == 0.5 and effect["sample_std"] == 0.5
    subset = next(
        r
        for r in result["common_prerequisites"]
        if r["treatment"] == "M8" and r["reference"] == "W8" and r["pool"] == "test_all"
    )
    assert subset["full_pool_n"] == 2
    assert subset["treatment_prerequisite_n"] == subset["reference_prerequisite_n"] == 1
    assert subset["common_prerequisite_n"] == 0
    assert subset["paired_common_subset_difference"] is None
    assert result["independent_training_initializations"] == 3
    assert result["independent_data_worlds"] == 1
    assert sum(r["alias_of"] == "D8_hist" for r in result["runs"]) == 3
    assert {p.name for p in output.iterdir()} == {
        "summary.json",
        "runs.csv",
        "paired-effects.csv",
        "common-prerequisites.csv",
    }


@pytest.mark.parametrize("corruption", ["missing_audit", "unfinished", "raw_score", "wrong_seed"])
def test_incomplete_or_corrupt_study_fails_before_writing(study, corruption):
    config_path, output, config, _ = study
    path = Path(config["results_root"]) / "runs" / config["runs"][0]["name"]
    if corruption == "missing_audit":
        (path / "audit.json").unlink()
    elif corruption == "unfinished":
        value = report.read(path / "complete.json")
        value["step"] = 32000
        write(path / "complete.json", value)
    elif corruption == "raw_score":
        value = report.read(path / "endpoint.json")
        value["test_all"]["alias_em"] = 0.9
        write(path / "endpoint.json", value)
    else:
        value = report.read(path / "run.json")
        value["spec"]["sampling_seed"] = 9
        write(path / "run.json", value)
    with pytest.raises(ValueError):
        report.build_report(config_path, output)
    assert not output.exists()


def test_historical_alias_requires_verified_compatibility(study):
    config_path, output, config, freeze = study
    config["reporting"]["historical_reuse_verified"] = False
    freeze()
    with pytest.raises(ValueError, match="cannot alias"):
        report.build_report(config_path, output)
    assert not output.exists()


def test_report_rejects_development_and_changed_frozen_config(study):
    config_path, output, config, freeze = study
    config["phase"] = "development"
    freeze()
    with pytest.raises(ValueError, match="main phase"):
        report.build_report(config_path, output)
    config["phase"] = "main"
    write(config_path, config)
    with pytest.raises(ValueError, match="freeze mismatch"):
        report.build_report(config_path, output)
