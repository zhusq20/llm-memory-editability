"""Report contracts: audited identities, full coverage, and unchanged exposure."""

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.grokking_reproduction import digest
from llm_memory_editability.loop_hop_data import panel_names

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
reporter = importlib.import_module("report_paper_update_gate")


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def report_files(tmp_path, monkeypatch):
    config = {
        "batch": "report-fixture",
        "primary": "Graph B three-hop answer accuracy at R4 and 300k",
        "training": {"steps": 300000},
        "input_origin": {"world_sha256": "world", "panels_sha256": "panels"},
        "arms": {
            "fixed-update": {"update_mode": "fixed", "repeats": 4, "unique_layers": 1},
            "token-update": {"update_mode": "token", "repeats": 4, "unique_layers": 1},
        },
    }
    config_path = tmp_path / "config.json"
    write(config_path, config)
    base = tmp_path / "results"
    baseline = tmp_path / "paper"
    for name, accuracy in (("paper-reused", 0.372), ("fixed-update", 0.4), ("token-update", 0.45)):
        folder = baseline if name == "paper-reused" else base / "runs" / name
        folder.mkdir(parents=True)
        metrics = {
            panel: {
                "n": 8,
                "answer_accuracy": accuracy,
                "accuracy": accuracy / 2,
                "answer_nll": 1.0,
                "nll": 1.5,
                **(
                    {"premise_correct_n": 0, "answer_accuracy_given_premises": None}
                    if panel.startswith(("train", "id", "ood"))
                    else {}
                ),
            }
            for panel in panel_names()
        }
        checkpoint = folder / "checkpoint-0300000.pt"
        checkpoint.write_bytes(name.encode())
        complete = {
            "state": "complete",
            "arm": name,
            "step": 300000,
            "budget": 300000,
            "engineering_only": False,
            "config_sha256": digest(config_path),
            "world_sha256": "world",
            "initial_model_sha256": "initial-" + name,
            "model_sha256": "model-" + name,
            "final_checkpoint_sha256": digest(checkpoint),
            "endpoint": {"step": 300000, "metrics": metrics},
        }
        audit = {
            "passed": True,
            "arm": name,
            "step": 300000,
            "exposure_match": True,
            "panels": {
                panel: {"passed": True, "answer_and_eos_ids_exact": True, "metrics": values}
                for panel, values in metrics.items()
            },
        }
        grid = {
            "arm": name,
            "step": 300000,
            "training_repeats": 4,
            "unique_layers": 1,
            "engineering_only": False,
            "config_sha256": digest(config_path),
            "world_sha256": "world",
            "panels_sha256": "panels",
            "initial_model_sha256": complete["initial_model_sha256"],
            "model_sha256": complete["model_sha256"],
            "metrics": {str(repeats): copy.deepcopy(metrics) for repeats in range(1, 5)},
        }
        write(folder / "complete.json", complete)
        write(folder / "audit.json", audit)
        write(folder / "grid.json", grid)
        write(
            folder / "grid-audit.json",
            {
                "passed": True,
                "model_sha256": complete["model_sha256"],
                "grid_sha256": digest(folder / "grid.json"),
                "findings": {
                    f"r{repeats}/{panel}": True
                    for repeats in range(1, 5)
                    for panel in panel_names()
                },
            },
        )
        np.savez_compressed(folder / "exposure.npz", counts=np.array([3, 4, 3]))
    monkeypatch.setattr(
        reporter,
        "verify_baseline",
        lambda config, baseline_dir=None: {"passed": True, "baseline_run": str(baseline)},
    )
    return config_path, base, baseline


def test_report_keeps_all_panels_rounds_and_fixed_learned_differences(report_files):
    config_path, base, baseline = report_files
    report, rows = reporter.build_report(config_path, base)
    assert len(rows) == 3 * 11 * 4
    assert set(row["panel"] for row in rows) == set(panel_names())
    assert set(row["inference_repeats"] for row in rows) == {1, 2, 3, 4}
    assert report["new_scientific_updates"] == 600000
    assert report["baseline_new_updates"] == 0
    assert report["primary_difference_percentage_points"] == pytest.approx(
        {"fixed_minus_paper": 2.8, "token_minus_fixed": 5.0, "token_minus_paper": 7.8}
    )
    assert report["conditions"]["paper-reused"]["reused"] is True
    assert (
        report["conditions"]["token-update"]["grid.json"]["metrics"]["4"]["ood_3"][
            "answer_accuracy_given_premises"
        ]
        is None
    )
    assert not (base / "comparison.json").exists()
    assert not (baseline / "comparison.json").exists()
    reporter.write_report(base, report, rows)
    assert len((base / "comparison.csv").read_text().splitlines()) == 133
    with pytest.raises(FileExistsError):
        reporter.write_report(base, report, rows)


@pytest.mark.parametrize(
    ("filename", "key", "value"),
    [
        ("complete.json", "engineering_only", True),
        ("complete.json", "budget", 20000),
        ("complete.json", "config_sha256", "changed"),
        ("complete.json", "world_sha256", "changed"),
        ("audit.json", "passed", False),
        ("audit.json", "step", 20000),
        ("grid.json", "panels_sha256", "changed"),
        ("grid.json", "model_sha256", "changed"),
        ("grid-audit.json", "grid_sha256", "changed"),
    ],
)
def test_report_rejects_changed_identity_or_failed_audit(report_files, filename, key, value):
    config_path, base, _ = report_files
    path = base / "runs" / "token-update" / filename
    payload = reporter.read(path)
    payload[key] = value
    write(path, payload)
    with pytest.raises(ValueError, match="Unverified token-update"):
        reporter.build_report(config_path, base)


@pytest.mark.parametrize("change", ("exposure", "checkpoint", "missing-audit", "metric"))
def test_report_requires_exact_exposure_checkpoint_and_complete_audits(report_files, change):
    config_path, base, _ = report_files
    folder = base / "runs" / "fixed-update"
    if change == "exposure":
        np.savez_compressed(folder / "exposure.npz", counts=np.array([4, 3, 3]))
    elif change == "checkpoint":
        (folder / "checkpoint-0300000.pt").write_bytes(b"different checkpoint")
    elif change == "missing-audit":
        (folder / "audit.json").unlink()
    else:
        audit = reporter.read(folder / "audit.json")
        audit["panels"]["ood_3"]["metrics"]["answer_accuracy"] = 0.9
        write(folder / "audit.json", audit)
    with pytest.raises((ValueError, FileNotFoundError)):
        reporter.build_report(config_path, base)


def test_report_rejects_changed_config_bytes_and_incomplete_grid(report_files):
    config_path, base, _ = report_files
    original = config_path.read_text()
    config_path.write_text(original + "\n")
    with pytest.raises(ValueError, match="endpoint identity"):
        reporter.build_report(config_path, base)
    config_path.write_text(original)
    folder = base / "runs" / "token-update"
    grid = reporter.read(folder / "grid.json")
    del grid["metrics"]["2"]["ood_4"]
    write(folder / "grid.json", grid)
    audit = reporter.read(folder / "grid-audit.json")
    audit["grid_sha256"] = digest(folder / "grid.json")
    write(folder / "grid-audit.json", audit)
    with pytest.raises(ValueError, match="missing panel"):
        reporter.build_report(config_path, base)


def test_report_propagates_baseline_verification_failure(report_files, monkeypatch):
    config_path, base, _ = report_files

    def reject(*args, **kwargs):
        raise ValueError("baseline source changed")

    monkeypatch.setattr(reporter, "verify_baseline", reject)
    with pytest.raises(ValueError, match="baseline source changed"):
        reporter.build_report(config_path, base)
