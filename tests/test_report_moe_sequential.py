"""Scientific recounts must preserve losses, pairing and missing evidence."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_moe_sequential as report  # noqa: E402


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    config = {
        "results_root": str(root),
        "specs": [],
        "reused_runs": [],
        "analysis": {"unit": "one graph, two initializations"},
    }
    for seed in (1, 2):
        for arm in report.ARMS:
            name = f"{arm}-{seed}"
            spec = {
                "name": name,
                "architecture": arm,
                "initialization": seed,
                "stage_a_steps": 8,
                "stage_b_steps": 4,
            }
            out = root / "runs" / name
            config["specs"].append(spec)
            save(out / "run.json", {"spec": spec})
            save(out / "sampling-plan.json", {"plan_sha256": str(seed)})
            save(
                out / "architecture.json",
                {"total_parameters": 100, "nominal_parameters_selected_per_token": 80},
            )
            save(out / "complete.json", {"state": "complete"})
            save(out / "audit.json", {"passed": True})
            history = []
            for step, scores in ((8, [1, 1, 0, 0]), (12, [1, 0, 1, 0])):
                raw = {
                    pool: [
                        {"id": f"{pool}-{i}", "alias_em": score} for i, score in enumerate(scores)
                    ]
                    for pool in report.POOLS
                }
                if arm == "M4" and step == 12:
                    raw["BB"][-1]["alias_em"] = seed == 1
                metrics = {
                    pool: {"n": 4, "alias_em": sum(p["alias_em"] for p in values) / 4}
                    for pool, values in raw.items()
                }
                node = {
                    "step": step,
                    "stage": "A" if step == 8 else "B",
                    "metrics": metrics,
                    **{key: step * 100 for key in report.COSTS},
                }
                history.append(node)
                save(out / f"predictions-{step:07d}.json", raw)
            save(out / "learning.json", history)
    return config


def test_recount_does_not_replace_retention_with_net_accuracy(tmp_path):
    summary = report.summarize(fixture(tmp_path))
    assert summary["complete"]
    ordinary = summary["means"]["D4"]
    assert ordinary["stage_a"]["AA"] == ordinary["endpoint"]["AA"] == 0.5
    assert ordinary["AA_retention"] == 0.5
    assert summary["means"]["M4"]["BB_gain"] == 0.125
    assert len(summary["paired_differences"]) == 4


def test_missing_completed_run_and_sample_stream_mismatch_are_visible(tmp_path):
    config = fixture(tmp_path)
    (tmp_path / "runs/M4-1/complete.json").unlink()
    assert not report.summarize(config)["complete"]
    save(tmp_path / "runs/M4-2/sampling-plan.json", {"plan_sha256": "different"})
    with pytest.raises(ValueError, match="sample streams"):
        report.summarize(config)
