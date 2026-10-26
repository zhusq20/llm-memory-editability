import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_knowledge_use_coverage", ROOT / "scripts/analyze_knowledge_use_coverage.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def cfg():
    return {
        "batch": "knowledge-use-coverage-v1",
        "model": "Qwen/Qwen3-1.7B",
        "model_revision": "frozen-model-revision",
        "scope": "down",
        "max_sequence_tokens": 2048,
        "development_pairs": 2,
        "formal_pairs": 8,
        "development_token_nodes": [100, 200],
        "formal_token_nodes": [400, 800],
        "rewrite_token_fraction": 0.25,
        "calibration": {"atomic_recall_target": 0.8},
        "natural_validation_cases": [32, 96],
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def node_summary(batch, node, arm, correct, *, unseen=0):
    p = batch / "runs" / f"development-pd-{arm}-addition" / f"node-{node}/evaluation/summary.json"
    write(
        p,
        {
            "complete": True,
            "groups": {
                "atomic_seen/direct": {"n": 4, "correct": correct, "accuracy": correct / 4},
                "atomic_unseen/direct": {"n": 8, "correct": unseen, "accuracy": unseen / 8},
            },
        },
    )
    return p


def row(identity, correct, cluster=None, category="atomic_seen", mode="direct", **extra):
    return {
        "id": identity,
        "correct": correct,
        "cluster_id": cluster or identity,
        "category": category,
        "mode": mode,
        "question": identity,
        "gold": "yes",
        **extra,
    }


def test_selection_waits_for_all_exact_development_nodes_and_ignores_unseen(tmp_path):
    for node in (100, 200):
        for arm in "ABC":
            if (node, arm) != (200, "C"):
                node_summary(tmp_path, node, arm, 4, unseen=0)
    assert M.select_development_node(tmp_path, cfg())["status"] == "incomplete"
    # A nearby token checkpoint cannot substitute for the exact configured node.
    node_summary(tmp_path, 199, "C", 4)
    assert M.select_development_node(tmp_path, cfg())["status"] == "incomplete"
    node_summary(tmp_path, 200, "C", 4)
    selected = M.select_development_node(tmp_path, cfg())
    assert selected["status"] == "complete" and selected["target_met"]
    assert selected["development_selected_tokens"] == 100
    assert selected["formal_selected_tokens"] == 400
    assert selected["rewrite_selected_tokens"] == 100
    assert selected["selection_group"] == "atomic_seen/direct"


def test_selection_fallback_is_largest_planned_not_best_formal(tmp_path):
    for node in (100, 200):
        for arm in "ABC":
            node_summary(tmp_path, node, arm, 2, unseen=8)
    write(
        tmp_path / "runs/formal-pd-A-addition/node-400/evaluation/summary.json",
        {"complete": True, "groups": {"atomic_seen/direct": {"accuracy": 1}}},
    )
    selected = M.select_development_node(tmp_path, cfg())
    assert selected["status"] == "complete" and not selected["target_met"]
    assert selected["development_selected_tokens"] == 200
    p = node_summary(tmp_path, 200, "A", 4)
    broken = M.read(p)
    broken["groups"]["atomic_seen/direct"]["accuracy"] = 0.25
    write(p, broken)
    assert M.select_development_node(tmp_path, cfg())["status"] == "invalid"


def test_cluster_bootstrap_resamples_rewrite_blocks_not_individual_questions():
    result = M.cluster_ratio_ci([("swap-0", 1, 1), ("swap-0", -1, 1)], repeats=300)
    assert result["clusters"] == 1 and result["value"] == 0
    assert result["ci95"] == [0, 0]
    missing = M.cluster_ratio_ci([], repeats=300)
    assert missing["value"] is None and missing["ci95"] is None


def test_pairing_preserves_gain_loss_despite_zero_net_change():
    before = [row("one", True), row("two", False)]
    after = [row("one", False), row("two", True), row("missing", True)]
    result = M.paired_statistics(before, after, repeats=100)["atomic_seen/direct"]
    assert result["paired_n"] == 2 and result["unpaired_or_unknown_n"] == 1
    assert result["gained_n"] == result["lost_n"] == 1
    assert result["accuracy_difference"]["value"] == 0
    assert result["initially_correct_success"]["value"] == 0
    assert result["paired_coverage"] == pytest.approx(2 / 3)


def test_direct_step_and_chain_never_collapse_into_one_observation():
    records = [
        row("same", correct, mode=mode)
        for correct, mode in (
            (True, "direct"),
            (False, "step_by_step"),
            (True, "self_generated_chain"),
        )
    ]
    assert len({M.row_key(r) for r in records}) == 3
    groups = M.summarize_rows(records, repeats=20)
    assert len(groups) == 3
    assert groups["atomic_seen/step_by_step"]["accuracy"] == 0


def test_rewrite_pairing_tracks_changed_answers_without_calling_them_preservation():
    before = row("formal/pair-0/addition/evaluation/use/direct", True, gold="Paris")
    after = row(
        "formal/pair-0/rewrite/evaluation/use/direct", True, gold="Berlin", answer_changed=True
    )
    result = M.paired_statistics([before], [after], repeats=20)["atomic_seen/direct"]
    assert result["paired_n"] == 1 and not result["same_answer_preservation"]
    # Comparing two arms at the same rewrite state remains a same-answer comparison.
    assert M.paired_statistics([after], [after], 20)["atomic_seen/direct"][
        "same_answer_preservation"
    ]


def test_unknown_judgment_is_not_a_wrong_answer():
    rows = [row("a", True, category="natural_closed"), row("b", None, category="natural_closed")]
    group = M.summarize_rows(rows, repeats=20)["natural_closed/direct"]
    assert group["n"] == 2 and group["valid_n"] == 1 and group["unknown_n"] == 1
    assert group["accuracy"] == 1 and group["coverage"] == 0.5


def test_prerequisites_require_the_actual_rewrite_question_and_gold():
    use = row(
        "use",
        True,
        category="double_new_composition",
        question="City?",
        gold="Berlin",
        fact_ids=["new/a"],
        background_fact_ids=["background/city"],
    )
    atom = row("atom", True, fact_ids=["new/a"])
    background = row(
        "background", True, category="retention_related_background", fact_ids=["background/city"]
    )
    wrong_state_oracle = row(
        "use", True, category="double_new_composition", question="City?", gold="Paris"
    )
    result = M.prerequisite_subsets([use, atom], [background], [wrong_state_oracle], repeats=20)
    group = result["source_present"]["double_new_composition/direct"]
    assert group["observed"] == 0 and group["conditional"] is None
    assert result["background"]["double_new_composition/direct"]["eligible"] == 1


def test_cost_uses_exact_node_and_never_later_exposure(tmp_path):
    write(tmp_path / "learning.json", [{"supervised_tokens": 100, "training_seconds": 3.0}])
    write(
        tmp_path / "complete.json",
        {
            "supervised_tokens": 200,
            "training_seconds": 8.0,
            "elapsed_total_seconds": 12.0,
            "smoke": False,
            "cost_limitations": ["FLOPs unmeasured"],
        },
    )
    write(
        tmp_path / "exposure.json",
        {
            "student_tokens": 999,
            "main/q": 200,
            "main_student_tokens": 700,
            "replay_student_tokens": 299,
            "teacher_padded_tokens": 1111,
            "target_fact_exposures/fact-1": 3,
        },
    )
    first = M.node_cost(tmp_path, 100)
    assert first["training_seconds"] == 3.0 and first["complete_counters"] is None
    assert first["elapsed_total_seconds"] is None
    unknown = M.node_cost(tmp_path, 150)
    assert unknown["supervised_tokens"] is None and not unknown["exact_node_observed"]
    last = M.node_cost(tmp_path, 200)
    assert last["complete_counters"]["student_tokens"] == 999
    assert last["elapsed_total_seconds"] == 12.0
    assert last["cost_limitations"] == ["FLOPs unmeasured"]
    assert last["complete_counters"]["target_fact_exposures/fact-1"] == 3
    assert last["complete_counters"]["main_student_tokens"] == 700
    assert last["fact_row_supervised_tokens"] == {"q": 200}


def test_missing_experiments_produce_incomplete_reports_not_zero_scores(tmp_path):
    config = tmp_path / "config.json"
    write(config, cfg())
    output = tmp_path / "report"
    validation = M.write_outputs(tmp_path / "empty-batch", config, output, repeats=20)
    summary = M.read(output / "summary.json")
    assert validation["status"] == "incomplete" and not validation["complete"]
    assert summary["development"]["baselines"]["closed"]["metrics"] == {}
    assert summary["formal"].get("runs") is None
    assert (output / "report.md").exists()


def test_complete_matrix_requires_cloud_identity_and_exact_cost_evidence(tmp_path):
    config = cfg()
    config["natural_validation_cases"] = [0, 2]

    def jsonl(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

    def controlled(directory, cohort, stage):
        records = [
            row(
                f"{cohort}/pair-{i // 2}/{stage}/evaluation/atom-{i}/direct",
                True,
                cluster=f"pair-{i // 2}",
                fact_ids=[f"new/{i}"],
            )
            for i in range(4)
        ]
        jsonl(directory / "predictions.jsonl", records)
        write(
            directory / "summary.json",
            {
                "complete": True,
                "groups": {"atomic_seen/direct": {"n": 4, "correct": 4, "accuracy": 1}},
            },
        )
        return records

    def natural(directory):
        records = [
            {
                "case_id": str(i),
                "group": group,
                "valid_grade": True,
                "grade": True,
                "question": str(i),
                "gold": "yes",
            }
            for i in range(2)
            for group in ("closed", "oracle_context")
        ]
        jsonl(directory / "judge/grading.jsonl", records)
        jsonl(
            directory / "retention.jsonl",
            [
                {"id": f"ret/city_country/City-{i}", "gold": f"Country-{i % 91}", "strict": True}
                for i in range(317)
            ],
        )
        write(directory / "summary.json", {"complete": True})

    for cohort in ("development", "formal"):
        records = controlled(tmp_path / "baseline" / cohort / "closed", cohort, "addition")
        controlled(tmp_path / "baseline" / cohort / "context", cohort, "addition")
        rewritten = controlled(
            tmp_path / "baseline" / cohort / "rewrite-context", cohort, "rewrite"
        )
        for stage, rs in (("addition", records), ("rewrite", rewritten)):
            jsonl(tmp_path / "data" / cohort / f"evaluation-{stage}.jsonl", rs)
            jsonl(tmp_path / "data" / cohort / f"retention-{stage}.jsonl", [])
    natural(tmp_path / "baseline/natural")
    specs = [("development", "pd", arm, "addition", [100, 200]) for arm in "ABC"]
    specs += [
        (cohort, objective, arm, stage, [node])
        for cohort, stage, node in (
            ("formal", "addition", 400),
            ("formal", "rewrite", 100),
            ("natural", "addition", 400),
        )
        for objective in ("pd", "sft")
        for arm in "ABC"
    ]
    cloud_paths = []
    for cohort, objective, arm, stage, nodes in specs:
        name = f"{cohort}-{objective}-{arm}-{stage}"
        run = tmp_path / "runs" / name
        write(run / "learning.json", [{"supervised_tokens": n} for n in nodes])
        write_run_contract(
            run,
            config,
            cohort,
            objective,
            arm,
            stage,
            nodes,
            parent_node=400 if stage == "rewrite" else None,
        )
        write(tmp_path / "tracking-wandb" / name / "state.json", {"run_id": name})
        for node in nodes:
            destination = run / f"node-{node}"
            if cohort == "natural":
                natural(destination / "evaluation")
            else:
                controlled(destination / "evaluation", cohort, stage)
            write(destination / "audit.json", {"passed": True})
            cloud = destination / "cloud-verification.json"
            cloud_paths.append(cloud)
            write(
                cloud,
                {
                    "passed": True,
                    "config_equal": True,
                    "node_summary_equal": True,
                    "state": "finished",
                    "run_id": name,
                },
            )
    result = M.analyze(tmp_path, config, repeats=0)
    assert result["complete"], result["missing"]
    old = result["natural_application"]["baseline"]["metrics"]["natural_old_retention/direct"]
    assert old["clusters"] == 91 and old["unknown_cluster_rows"] == 0
    assert all(r["status"] == "complete" for r in result["abc_comparisons"].values())
    wrong = M.read(cloud_paths[-1])
    wrong["run_id"] = "another-run"
    write(cloud_paths[-1], wrong)
    broken = M.analyze(tmp_path, config, repeats=0)
    assert not broken["complete"]
    assert any("cloud run identity" in " ".join(r["problems"]) for r in broken["missing"])


def write_run_contract(run, config, cohort, objective, arm, stage, nodes, parent_node=None):
    parent = (
        f"/batch/runs/{cohort}-{objective}-{arm}-addition/node-{parent_node}/resume.pt"
        if stage == "rewrite"
        else None
    )
    write(
        run / "run.json",
        {
            "spec": {
                "config": M.resolved_run_config(config, cohort),
                "cohort": cohort,
                "objective": objective,
                "arm": arm,
                "stage": stage,
                "nodes": nodes,
                "parent": parent,
            },
            "model": config["model"],
            "initial_model_sha256": config["model_revision"],
            "tracking_group": config["batch"],
            "parameters": {
                "trainable_names": ["model.layers.0.mlp.down_proj.weight"],
                "trainable_parameters": 10,
                "total_parameters": 100,
            },
        },
    )
    write(
        run / "complete.json",
        {
            "supervised_tokens": max(nodes),
            "nodes": nodes,
            "smoke": False,
            "parent_optimizer_reset": bool(parent),
            "teacher_frozen": True,
        },
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("arm", "B"),
        ("nodes", [401]),
        ("parent", "/batch/runs/wrong/resume.pt"),
        ("objective", "sft"),
        ("cohort", "development"),
        ("stage", "rewrite"),
    ],
)
def test_actual_run_spec_must_match_requested_trajectory(tmp_path, key, value):
    config = cfg()
    write_run_contract(tmp_path, config, "formal", "pd", "A", "addition", [400])
    assert (
        M.validate_run_contract(tmp_path, config, "formal", "pd", "A", "addition", [400])["status"]
        == "complete"
    )
    record = M.read(tmp_path / "run.json")
    record["spec"][key] = value
    write(tmp_path / "run.json", record)
    result = M.validate_run_contract(tmp_path, config, "formal", "pd", "A", "addition", [400])
    assert result["status"] == "incomplete"
    assert any(f"spec.{key}" in p for p in result["problems"])


def test_natural_config_uses_real_runtime_resolver_not_controlled_config(tmp_path):
    config = cfg()
    write_run_contract(tmp_path, config, "natural", "pd", "A", "addition", [400])
    record = M.read(tmp_path / "run.json")
    resolved = record["spec"]["config"]
    assert resolved["max_sequence_tokens"] == 8192
    assert resolved["judge_model_path"] == "/factorial/models/qwen3-4b"
    assert resolved["eval_max_new_tokens"] == 500
    assert (
        M.validate_run_contract(tmp_path, config, "natural", "pd", "A", "addition", [400])["status"]
        == "complete"
    )
    record["spec"]["config"] = config
    write(tmp_path / "run.json", record)
    assert "spec.config" in " ".join(
        M.validate_run_contract(tmp_path, config, "natural", "pd", "A", "addition", [400])[
            "problems"
        ]
    )


def test_engineering_real_stages_and_named_failures_preserve_unknown_costs(tmp_path):
    for objective in ("pd", "sft"):
        write(
            tmp_path / "engineering/formal" / objective / "exercise.json",
            {"passed": True, "seconds": 2.5, "engineering_optimizer_updates": 3},
        )
        write(tmp_path / "engineering/formal" / objective / "reload.json", {"passed": True})
    write(
        tmp_path / "engineering/formal/preflight.json",
        {"passed": True, "engineering_optimizer_updates": 6},
    )
    write(tmp_path / "engineering/formal/failure-exercise-pd.json", {"traceback": "failed probe"})
    report = M.analyze(tmp_path, cfg(), repeats=0)
    cost = report["cost"]["engineering"]
    assert len(cost["records"]) == 5
    assert cost["observed_exercise_seconds"] == 5.0
    assert cost["total_engineering_seconds"] is None
    assert all(r["seconds"] is None for r in cost["records"] if r["stage"] != "exercise")
    assert report["failures"][0]["path"].endswith("failure-exercise-pd.json")


def test_unknown_country_cluster_does_not_fall_back_to_question_id():
    rows = [
        row("q1", True, category="natural_old_retention"),
        row("q2", False, category="natural_old_retention"),
    ]
    for r in rows:
        r["cluster_id"] = None
    metric = M.summarize_rows(rows, 100)["natural_old_retention/direct"]
    assert metric["accuracy"] == 0.5 and metric["accuracy_ci95"] is None
    assert metric["unknown_cluster_rows"] == 2 and metric["clusters"] == 0
    pair = M.paired_statistics(rows, rows, 100)["natural_old_retention/direct"]
    assert pair["accuracy_difference"]["ci95"] is None


def test_controlled_old_retention_recomputes_country_blocks_from_preserved_source(tmp_path):
    records = [
        row(
            "formal/old-1",
            True,
            category="natural_old_retention",
            original_source_id="ret/capital_rev/Paris",
            gold="France",
        ),
        row(
            "formal/old-2",
            False,
            category="natural_old_retention",
            original_source_id="nat2short/Paris",
            gold="euro",
        ),
        row(
            "formal/old-3",
            True,
            category="natural_old_retention",
            original_source_id="ret/currency/France",
            gold="euro",
        ),
    ]
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    write(tmp_path / "summary.json", {"complete": True})
    loaded = M._load_controlled(tmp_path, records)
    assert loaded["status"] == "complete"
    assert len({r["cluster_id"] for r in loaded["rows"]}) == 1
    assert M.summarize_rows(loaded["rows"], 100)["natural_old_retention/direct"]["clusters"] == 1
