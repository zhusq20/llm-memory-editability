import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_knowledge_factorial_20261025.py"
SPEC = importlib.util.spec_from_file_location("knowledge_factorial_analysis", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def test_invalid_grades_never_become_false_or_change_retention_denominator():
    before = {"a": True, "b": True, "c": False, "d": False}
    after = {"a": True, "b": None, "c": True, "d": False}
    result = A.paired_metrics(before, after, replicates=100)
    assert result["after"]["accuracy_valid"] == 2 / 3
    assert result["accuracy_change_pp"] == pytest.approx(100 / 6)
    assert result["correct_fraction_all_change_pp"] == 0
    assert result["retained_fraction"] == 0.5
    assert result["lost_to_false_n"] == 0
    assert result["initially_correct_after_invalid_n"] == 1
    assert result["new_from_false_n"] == 1


def test_paired_bootstrap_resamples_shared_cases():
    values = {str(i): i % 2 == 0 for i in range(100)}
    result = A.paired_metrics(values, values, replicates=1000)
    assert result["ci95"]["accuracy_change_pp"] == [0, 0]
    assert result["ci95"]["retained_fraction"] == [1, 1]
    with pytest.raises(ValueError, match="coverage"):
        A.paired_metrics(values, {"wrong": True})


def test_training_clock_includes_restarted_discarded_segments():
    records = [{"seconds_this_process": t} for t in [1, 20, 2, 30]]
    assert A.curve_seconds(records) == {"seconds": 50, "segments": [20, 30]}


def test_correlated_retention_templates_stay_in_the_same_bootstrap_unit():
    records = [
        {"id": "ret/capital_rev/Warsaw", "gold": "Poland"},
        {"id": "ret/currency/Poland", "gold": "zloty"},
        {"id": "nat2/0/Warsaw", "gold": "zloty"},
    ]
    clusters = A.retention_clusters(records)
    assert set(clusters.values()) == {"Poland"}
    before = {key: True for key in clusters}
    after = {key: i == 0 for i, key in enumerate(clusters)}
    result = A.paired_metrics(before, after, clusters=clusters, replicates=100)
    assert result["bootstrap_units_n"] == 1
    assert result["ci95"]["retained_fraction"] == [1 / 3, 1 / 3]


def test_incomplete_arm_is_excluded_even_with_training_complete(tmp_path):
    run = tmp_path / "runs/pd_lora"
    run.mkdir(parents=True)
    (run / "complete.json").write_text(json.dumps({"exposure": 1489200, "step": 1}))
    result, observations = A.collect_arm(tmp_path, "pd_lora", 100, 1)
    assert result["status"] == "incomplete"
    assert result["training_complete"]
    assert not observations


def test_cost_uses_post_stop_audit_and_separates_recomputation(tmp_path):
    original = tmp_path / "runs/sft_full"
    continuation = tmp_path / "batchsize-20261022/runs/sft_full"
    handoff = tmp_path / "batchsize-20261022/handoff"
    for path in (original, continuation, handoff):
        path.mkdir(parents=True)
    selection = {
        "source_step": 25000,
        "source_exposure": 100000,
        "observed_recomputed_updates": 1500,
        "batch_size": 16,
    }
    (continuation / "selection.json").write_text(json.dumps(selection))
    (handoff / "sft_full-post-stop-audit.json").write_text(
        json.dumps({"observed_recomputed_updates": 1800})
    )
    curve = [
        {
            "step": 25000,
            "seconds_this_process": 100,
            "answer_tokens": 10,
            "exposure": 100000,
            "forward_tokens": 20,
        },
        {
            "step": 26800,
            "seconds_this_process": 110,
            "answer_tokens": 13,
            "exposure": 107200,
            "forward_tokens": 25,
        },
    ]
    (original / "curve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in curve))
    (continuation / "curve.jsonl").write_text(json.dumps({"seconds_this_process": 200}) + "\n")
    cost = A.arm_cost(tmp_path, "sft_full", {"exposure": 1489200})
    assert cost["recomputed_updates_logged"] == 1800
    assert cost["recomputed_tokens_logged"]["exposure"] == 7200
    assert cost["training_clock_hours_logged"] == 310 / 3600
    assert cost["teacher_forward_sample_passes"] == 0
    assert cost["flops"] is None


def test_cloud_completion_requires_matching_ids_config_counters_and_all_eight(tmp_path):
    frozen = tmp_path / "source/configs/knowledge-method-factorial-1000-v1.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text(json.dumps({"batch": "knowledge-method-factorial-1000-v1", "seed": 1234}))
    (tmp_path / "tracking").mkdir()
    local, remote = {}, {}
    counts = {"correct_n": 3, "valid_n": 4, "invalid_n": 0}
    for arm in A.ARMS:
        local[arm] = {
            "status": "complete",
            "cost": {
                "exposure": 1489200,
                "optimizer_updates": 1,
                "answer_tokens": 10,
                "forward_tokens_student_plus_teacher_if_pd": 20,
            },
            "groups": {
                group: {phase: counts for phase in ("before", "after")} for group in A.GROUPS
            },
            "old_retention": {"retained_n": 3},
        }
        for stage in ("train", "paper-evaluate"):
            key = f"{arm}-{stage}"
            identity = {"id": key, "url": f"https://example.test/{key}"}
            (tmp_path / f"tracking/{key}.json").write_text(json.dumps(identity))
            summary = {
                "exposure": 1489200,
                "optimizer_updates": 1,
                "answer_tokens": 10,
                "forward_tokens": 20,
                "retention.retained_n": 3,
            }
            summary.update(
                {
                    f"{group}.{phase}.{name}": value
                    for group in A.GROUPS
                    for phase in ("before", "after")
                    for name, value in counts.items()
                }
            )
            remote[key] = {
                **identity,
                "state": "finished",
                "summary": summary,
                "config": {
                    "batch": {"value": "knowledge-method-factorial-1000-v1"},
                    "seed": {"value": 1234},
                },
            }
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"utc": "2026-10-25T00:00:00Z", "runs": remote}))
    assert A.audit_cloud(tmp_path, local, snapshot)["complete"]
    remote["pd_lora-train"]["summary"]["exposure"] -= 1
    snapshot.write_text(json.dumps({"utc": "2026-10-25T00:00:00Z", "runs": remote}))
    bad = A.audit_cloud(tmp_path, local, snapshot)
    assert not bad["complete"]
    assert "summary/exposure" in bad["arms"]["pd_lora"]["stages"]["train"]["mismatches"]
    remote["pd_lora-train"]["id"] = "another-experiment"
    snapshot.write_text(json.dumps({"utc": "2026-10-25T00:00:00Z", "runs": remote}))
    assert (
        "run_id"
        in A.audit_cloud(tmp_path, local, snapshot)["arms"]["pd_lora"]["stages"]["train"][
            "mismatches"
        ]
    )
    assert not A.audit_cloud(tmp_path, local, tmp_path / "missing.json")["complete"]
