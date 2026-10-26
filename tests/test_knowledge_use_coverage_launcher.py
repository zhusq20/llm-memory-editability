import copy
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "coverage_launcher", ROOT / "scripts/launch_knowledge_use_coverage.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "coverage_launcher_analysis", ROOT / "scripts/analyze_knowledge_use_coverage.py"
)
A = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(A)


@pytest.fixture
def setup(tmp_path):
    config = json.loads((ROOT / "configs/knowledge-use-coverage-v1.json").read_text())
    source = tmp_path / "new/source"
    config_path = "configs/knowledge-use-coverage-v1.json"
    M.write(source / config_path, config)
    for name in (
        "run_knowledge_use_coverage.py",
        "run_knowledge_use_coverage_natural.py",
        "check_knowledge_use_coverage_cuda.py",
        "analyze_knowledge_use_coverage.py",
    ):
        path = source / "scripts" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            (ROOT / "scripts" / name).read_text()
            if name
            in {"analyze_knowledge_use_coverage.py", "run_knowledge_use_coverage_natural.py"}
            else "# frozen fixture\n"
        )
    archive = tmp_path / "new/source.tar"
    with tarfile.open(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file():
                handle.add(path, arcname=str(path.relative_to(source)))
    plan = {
        "batch": config["batch"],
        "host": "test-host",
        "scope": config["scope"],
        "output": str(tmp_path / "new"),
        "factorial_batch": str(tmp_path / "old"),
        "factorial_analysis": str(tmp_path / "analysis/summary.json"),
        "factorial_collector_receipt": str(tmp_path / "analysis/collector-state.json"),
        "scope_decision_receipt": str(tmp_path / "new/scope-decision.json"),
        "source_tar": str(archive),
        "source_tar_sha256": M.sha(archive),
        "source_dir": str(source),
        "scientific_config": config_path,
        "scientific_config_sha256": M.sha(source / config_path),
        "wandb_token_file": str(tmp_path / "secret"),
        "lock_directory": str(tmp_path / "locks"),
        "lanes": [{"gpu": gpu, "gpu_uuid": f"GPU-{gpu}", "cpus": str(gpu)} for gpu in (4, 6, 8)],
        "runtime": {
            "image": M.IMAGE,
            "hf_cache": "/hf-host",
            "deps": "/deps-host",
            "upstream": "/upstream-host",
        },
    }
    return plan, config


def endpoints(plan):
    weights = {arm: "weights-" + arm for arm in M.FACTORIAL_ARMS}
    collector = {
        "state": "complete",
        "local_and_cloud_verified": True,
        "training_processes_modified": False,
        "container_exits": {arm: {"status": "exited", "exit_code": 0} for arm in weights},
        "arms": {
            arm: {
                "exposure": 1489200,
                "train_complete": True,
                "audit_passed": True,
                "evaluation_complete": True,
            }
            for arm in weights
        },
    }
    M.write(plan["factorial_collector_receipt"], collector)
    cloud_path = Path(plan["factorial_analysis"]).with_name("cloud-snapshot.json")
    M.write(cloud_path, {"arms": list(weights)})
    summary = {
        "complete": True,
        "status": "complete",
        "complete_arms": list(weights),
        "cloud": {
            "complete": True,
            "snapshot_sha256": M.sha(cloud_path),
            "arms": {arm: {"terminal_verified": True} for arm in weights},
        },
        "arms": {arm: {"source": {"weights_sha256": weight}} for arm, weight in weights.items()},
    }
    M.write(plan["factorial_analysis"], summary)
    for arm, weight in weights.items():
        directory = Path(plan["factorial_batch"]) / "runs" / arm
        M.write(
            directory / "complete.json",
            {"exposure": 1489200, "smoke": False, "weights_sha256": weight},
        )
        M.write(directory / "audit/audit.json", {"passed": True})
        M.write(
            directory / "paper-eval/evaluate.json",
            {"complete": True, "source": {"weights_sha256": weight}},
        )
    return weights


def approve(plan, weights):
    M.write(
        plan["scope_decision_receipt"],
        {
            "decision": "approved",
            "scope": plan["scope"],
            "scientific_config_sha256": plan["scientific_config_sha256"],
            "factorial_endpoint_weights": weights,
            "analysis_sha256": M.sha(plan["factorial_analysis"]),
            "collector_sha256": M.sha(plan["factorial_collector_receipt"]),
            "cloud_snapshot_sha256": M.sha(
                Path(plan["factorial_analysis"]).with_name("cloud-snapshot.json")
            ),
            "decision_rationale": "Synthetic fixture review of all eight endpoints",
        },
    )


def controller(setup):
    plan, _ = setup
    config, payload, manifest = M.validate(plan)
    return M.Controller(plan, config, payload, manifest)


def test_incomplete_gate_never_queries_resources_or_creates_cuda(setup, monkeypatch):
    control = controller(setup)
    monkeypatch.setattr(
        M.resources, "snapshot", lambda: pytest.fail("GPU queried before endpoint gate")
    )
    monkeypatch.setattr(M, "call", lambda *_: pytest.fail("Docker called before endpoint gate"))
    assert control.tick() == "waiting_factorial"
    assert all(job["state"] == "pending" for job in control.state["jobs"].values())


def test_gate_requires_collector_cloud_identity_and_reviewed_scope(setup):
    plan, _ = setup
    weights = endpoints(plan)
    assert M.factorial_gate(plan)["state"] == "waiting_scope_decision"
    approve(plan, weights)
    assert M.factorial_gate(plan)["passed"]
    path = Path(plan["factorial_analysis"]).with_name("cloud-snapshot.json")
    path.write_text("{}")
    assert M.factorial_gate(plan)["state"] == "needs_attention"
    weights = endpoints(plan)
    weights["pd_lora"] = "another-weight"
    approve(plan, weights)
    assert M.factorial_gate(plan)["state"] == "needs_attention"
    receipt = M.read(plan["factorial_collector_receipt"])
    receipt["container_exits"]["pd_lora"]["status"] = "running"
    M.write(plan["factorial_collector_receipt"], receipt)
    assert M.factorial_gate(plan)["state"] == "waiting_factorial"


def explicit_full_start(plan):
    plan.update(start_mode="user_authorized_full_before_factorial", scope="full")
    evidence = Path(plan["output"]) / "start-evidence"
    arms = set(M.FACTORIAL_ARMS) - {"pd_lora"}
    M.write(evidence / "analysis.json", {"complete": False, "complete_arms": sorted(arms)})
    M.write(
        evidence / "collector.json",
        {
            "state": "waiting_for_endpoints",
            "failed_arms": [],
            "training_processes_modified": False,
            "arms": {
                arm: {
                    key: arm in arms
                    for key in ("train_complete", "audit_passed", "evaluation_complete")
                }
                for arm in M.FACTORIAL_ARMS
            },
        },
    )
    M.write(evidence / "cloud.json", {"complete": False})
    receipt = {
        "decision": "approved",
        "authorization_source": "user_message",
        "authorization_text": "立即开始，固定全参数",
        "supersedes_complete_factorial_prerequisite": True,
        "factorial_complete_at_decision": False,
        "batch": plan["batch"],
        "scope": "full",
        "gpu_lanes": [4, 6, 8],
        "scientific_config_sha256": plan["scientific_config_sha256"],
        "source_tar_sha256": plan["source_tar_sha256"],
        "utc": "2026-10-25T02:20:00Z",
        "decision_rationale": "Explicit user choice",
        "evidence": {
            name: {
                "path": str(evidence / f"{name}.json"),
                "sha256": M.sha(evidence / f"{name}.json"),
            }
            for name in ("analysis", "collector", "cloud")
        },
    }
    M.write(plan["scope_decision_receipt"], receipt)
    return receipt


def test_explicit_full_start_preserves_incomplete_factorial(setup):
    plan, _ = setup
    explicit_full_start(plan)
    result = M.factorial_gate(plan)
    assert result["passed"] and result["factorial_complete_at_decision"] is False
    assert result["mode"] == "user_authorized_full_before_factorial"
    # Later changes to the live collector cannot rewrite the preserved decision.
    M.write(plan["factorial_collector_receipt"], {"state": "complete"})
    assert M.factorial_gate(plan)["passed"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("scope", "down"),
        ("authorization_source", "inferred"),
        ("authorization_text", "start"),
        ("scientific_config_sha256", "different"),
        ("source_tar_sha256", "different"),
        ("factorial_complete_at_decision", True),
        ("supersedes_complete_factorial_prerequisite", False),
        ("gpu_lanes", [3, 6, 8]),
    ],
)
def test_explicit_full_start_rejects_changed_authority_or_recipe(setup, key, value):
    plan, _ = setup
    receipt = explicit_full_start(plan)
    receipt[key] = value
    M.write(plan["scope_decision_receipt"], receipt)
    assert M.factorial_gate(plan)["state"] == "needs_attention"


def test_explicit_start_requires_preserved_evidence_and_never_implicitly_activates(setup):
    plan, _ = setup
    receipt = explicit_full_start(plan)
    Path(receipt["evidence"]["analysis"]["path"]).write_text("{}")
    assert not M.factorial_gate(plan)["passed"]
    explicit_full_start(plan)
    plan.pop("start_mode")
    assert M.factorial_gate(plan)["state"] == "waiting_factorial"
    plan["start_mode"] = "ignore_prerequisites"
    assert M.factorial_gate(plan)["state"] == "needs_attention"


def test_parallel_preparation_keeps_training_and_quality_dependencies():
    config = json.loads((ROOT / "configs/knowledge-use-coverage-full-v1.json").read_text())
    jobs = M.pipeline(config)
    assert {name for name, job in jobs.items() if not job["dependencies"]} == {
        "development-prepare",
        "formal-prepare",
        "natural-calibrate",
    }
    assert jobs["formal-preflight"]["dependencies"] == ["formal-prepare", "select-budget"]
    assert jobs["natural-prepare"]["dependencies"][0] == "natural-quality-review"
    assert len(jobs["natural-prepare"]["dependencies"]) == 7
    assert all("rewrite-evaluate" in name for name in jobs["natural-prepare"]["dependencies"][1:])
    assert jobs["development-pd-A-addition-train"]["dependencies"] == ["development-baseline"]
    assert config["scope"] == "full" and config["arms"] == ["pd_full", "sft_full"]


def test_gate_retries_receipts_changed_while_sampling_sha(setup, monkeypatch):
    plan, _ = setup
    approve(plan, endpoints(plan))
    original = M._factorial_gate

    def replace_after_read(plan, read, sha):
        result = original(plan, read, sha)
        summary = M.read(plan["factorial_analysis"])
        summary["created_utc"] = "new analysis generation"
        M.write(plan["factorial_analysis"], summary)
        return result

    monkeypatch.setattr(M, "_factorial_gate", replace_after_read)
    result = M.factorial_gate(plan)
    assert result["state"] == "waiting_factorial" and not result["passed"]


def test_prior_passed_state_cannot_bypass_current_factorial_gate(setup, monkeypatch):
    control = controller(setup)
    control.state["gate"] = {"passed": True}
    monkeypatch.setattr(M.resources, "snapshot", lambda: pytest.fail("stale gate admitted GPU"))
    assert control.tick() == "waiting_factorial"


def resource_snapshot():
    return {
        "rows": {
            gpu: {"uuid": f"GPU-{gpu}", "used": 0, "free": 140000, "utilization": 0}
            for gpu in range(10)
        },
        "containers": [],
        "process_details": [],
        "processes": [],
    }


def reservation(gpu=4, identity=M.RELEASED_GPU4):
    return {
        "id": identity,
        "name": "reserved",
        "context": "lm-memory",
        "state": {
            "Status": "created",
            "Pid": 0,
            "Running": False,
            "StartedAt": "0001-01-01T00:00:00Z",
        },
        "restart": {"Name": "no"},
        "requests": [{"Capabilities": [["gpu"]], "DeviceIDs": [str(gpu)]}],
    }


def test_external_cuda_and_reservations_block_with_one_exact_exception(setup):
    plan, _ = setup
    lane = plan["lanes"][0]
    snap = resource_snapshot()
    snap["containers"] = [reservation()]
    assert M.admissible(snap, lane, plan["runtime"]) == []
    for field, value in (("Pid", 12), ("Running", True), ("StartedAt", "2026-10-25")):
        invalid = copy.deepcopy(snap)
        invalid["containers"][0]["state"][field] = value
        assert M.admissible(invalid, lane, plan["runtime"])
    snap["containers"][0]["restart"]["Name"] = "always"
    assert M.admissible(snap, lane, plan["runtime"])
    snap["containers"] = [reservation(identity="unrelated-created")]
    assert M.admissible(snap, lane, plan["runtime"])
    snap["containers"] = []
    snap["process_details"] = [{"uuid": "GPU-4", "pid": 25}]
    assert M.admissible(snap, lane, plan["runtime"])
    snap["process_details"] = []
    assert M.admissible(
        snap, lane, plan["runtime"], owned_uuids={"GPU-0", "GPU-1", "GPU-2", "GPU-3"}
    )
    # External work can appear while this stage's created container holds GPU4.
    assert M.admissible(
        snap, lane, plan["runtime"], owned_uuids={"GPU-0", "GPU-1", "GPU-2", "GPU-3", "GPU-4"}
    )


def test_readonly_mounts_cannot_be_reached_through_writable_batch(setup):
    plan, _ = setup
    plan["factorial_batch"] = str(Path(plan["output"]) / "old")
    with pytest.raises(ValueError, match="separate"):
        M.validate(plan)


def test_materialization_must_precede_controller_creation(setup):
    plan, _ = setup
    plan["materialize_command"] = ["python", "prepare_knowledge_use_coverage.py"]
    with pytest.raises(ValueError, match="before starting"):
        M.validate(plan)


def test_four_card_amendment_retains_resource_and_lane_limits(setup):
    plan, _ = setup
    plan["lanes"].append({"gpu": 3, "gpu_uuid": "GPU-3", "cpus": "3"})
    M.validate(plan)
    plan["lanes"].append({"gpu": 7, "gpu_uuid": "GPU-7", "cpus": "7"})
    with pytest.raises(ValueError, match="four authorized"):
        M.validate(plan)
    plan["lanes"].pop(0)
    with pytest.raises(ValueError, match="four authorized"):
        M.validate(plan)


def test_cuda25_amendment_rejects_mixed_or_unapproved_lanes(setup):
    plan, _ = setup
    plan["lanes"] = [{"gpu": gpu, "gpu_uuid": f"GPU-{gpu}", "cpus": str(gpu)} for gpu in (2, 5)]
    M.validate(plan)
    expanded = copy.deepcopy(plan)
    expanded["lanes"].append({"gpu": 6, "gpu_uuid": "GPU-6", "cpus": "6"})
    M.validate(expanded)
    for extra in (3, 7):
        for base in (plan, expanded):
            changed = copy.deepcopy(base)
            changed["lanes"].append({"gpu": extra, "gpu_uuid": f"GPU-{extra}", "cpus": str(extra)})
            with pytest.raises(ValueError, match="authorized"):
                M.validate(changed)


def test_project_gpu_cap_includes_other_created_reservations(setup, monkeypatch):
    control = controller(setup)
    monkeypatch.setattr(M.resources, "owned_gpu_uuids", lambda _: {"GPU-3"})
    snap = resource_snapshot()
    snap["containers"] = [reservation(gpu=i, identity=f"other-{i}") for i in (0, 1, 2)] + [
        reservation()
    ]
    owned = control.own_uuids(snap)
    assert owned == {"GPU-0", "GPU-1", "GPU-2", "GPU-3"}
    assert M.admissible(snap, control.plan["lanes"][0], control.plan["runtime"], owned_uuids=owned)


def test_archive_and_container_keep_source_and_old_results_readonly(setup):
    plan, _ = setup
    _, payload, manifest = M.validate(plan)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for item in archive:
            assert item.uid == item.gid == 0
            assert item.mode == (0o555 if item.isdir() else 0o444)
    assert plan["scientific_config"] in manifest
    _, argv = M.container_command(
        plan,
        "development-prepare",
        {"cohort": "development", "action": "prepare"},
        plan["lanes"][0],
        {},
    )
    assert argv[:3] == ["docker", "--context", "lm-memory"]
    assert f"type=bind,source={plan['factorial_batch']},target=/factorial,readonly" in argv
    assert "type=bind,source=/upstream-host,target=/factorial/upstream,readonly" in argv
    assert not any("WANDB_API_KEY=" in item for item in argv)
    assert "--restart" in argv and argv[argv.index("--restart") + 1] == "no"
    path = Path(plan["source_tar"])
    with tarfile.open(path, "w") as archive:
        item = tarfile.TarInfo("scripts/escape")
        item.type, item.linkname = tarfile.SYMTYPE, "/etc/passwd"
        archive.addfile(item)
    with pytest.raises(ValueError, match="Unsafe"):
        M.source_archive(path, M.sha(path))


def test_pipeline_budget_preflights_and_manual_quality_review(setup):
    plan, config = setup
    jobs = M.pipeline(config)
    assert [name for name, job in jobs.items() if job["action"] == "preflight"] == [
        "development-preflight",
        "formal-preflight",
        "natural-preflight",
    ]
    assert (
        len(
            [
                job
                for job in jobs.values()
                if job.get("cohort") == "development" and job["action"] == "train"
            ]
        )
        == 3
    )
    assert len(jobs["select-budget"]["dependencies"]) == 6
    assert jobs["natural-prepare"]["dependencies"] == ["natural-quality-review"]
    assert jobs["natural-quality-review"]["dependencies"] == ["natural-calibrate"]
    assert all("-pd-" in name for name in jobs["formal-sft-A-addition-train"]["dependencies"])
    assert len(jobs["natural-calibrate"]["dependencies"]) == 6
    selection = {"formal_selected_tokens": 800768, "rewrite_selected_tokens": 200192}
    argv = M.runner_command(plan, jobs["formal-pd-A-rewrite-train"], selection)
    assert argv[-1] == "/batch/runs/formal-pd-A-addition/node-800768/resume.pt"
    argv = M.runner_command(plan, jobs["natural-pd-A-addition-train"], selection)
    assert "--stage" not in argv and "--cohort" not in argv
    assert argv[-1] == "800768"


def test_missing_quality_receipt_waits_without_rerunning_calibration(setup):
    control = controller(setup)
    control.cpu("natural-quality-review")
    assert control.state["state"] == "waiting_quality_review"
    assert control.state["jobs"]["natural-quality-review"]["state"] == "pending"


def test_frozen_analysis_loads_on_cpu_host_and_keeps_container_config_paths(setup):
    control = controller(setup)
    module = control.load_analysis()
    assert Path(module.__file__).parent == Path(control.plan["source_dir"]) / "scripts"
    resolved = module.resolved_run_config(control.config, "natural")
    assert resolved["judge_model_path"] == "/factorial/models/qwen3-4b"
    assert resolved["max_sequence_tokens"] == 8192


def test_preflight_requires_both_independent_reload_receipts(setup):
    plan, config = setup
    batch = Path(plan["output"])
    root = batch / "engineering/development"
    spec = {"cohort": "development", "action": "preflight"}
    M.write(
        root / "preflight.json",
        {"status": "passed", "passed": True, "config": config, "scientific_updates": 0},
    )
    with pytest.raises(ValueError, match="engineering exercise/reload"):
        M.verify_outputs(batch, spec, {}, config, A)
    for objective in ("pd", "sft"):
        directory = root / objective
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "resume.pt").write_bytes(b"engineering fixture")
        M.write(
            directory / "exercise.json",
            {
                "passed": True,
                "restoration_exact": True,
                "teacher_unchanged": True,
                "weights_sha256": M.sha(directory / "resume.pt"),
            },
        )
        M.write(
            directory / "reload.json",
            {
                "passed": True,
                "full_trainable_weights_exact": True,
                "logits_and_greedy_exact": True,
                "new_process": True,
            },
        )
    assert M.verify_outputs(batch, spec, {}, config, A)
    (root / "sft/reload.json").unlink()
    with pytest.raises(ValueError, match="engineering exercise/reload"):
        M.verify_outputs(batch, spec, {}, config, A)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def evaluation_fixture(batch, directory, stage="addition", *, context=False, correct=True):
    row = {
        "id": f"development/pair-1/{stage}/evaluation/atomic",
        "mode": "direct",
        "category": "atomic_seen",
        "question": "Original question",
        "gold": "answer",
    }
    write_rows(batch / f"data/development/evaluation-{stage}.jsonl", [row])
    write_rows(batch / f"data/development/retention-{stage}.jsonl", [])
    write_rows(
        directory / "predictions.jsonl", [{**row, "correct": correct, "parse_status": "parsed"}]
    )
    M.write(
        directory / "summary.json",
        {
            "complete": True,
            "with_context": context,
            "groups": {
                "atomic_seen/direct": {
                    "n": 1,
                    "correct": int(correct),
                    "unparsed": 0,
                    "accuracy": float(correct),
                }
            },
        },
    )


def trajectory_fixture(plan, config):
    batch = Path(plan["output"])
    run = batch / "runs/development-pd-A-addition"
    nodes = config["development_token_nodes"]
    M.write(batch / "data/development/manifest.json", {"source": "fixture"})
    M.write(
        run / "run.json",
        {
            "spec": {
                "config": config,
                "cohort": "development",
                "objective": "pd",
                "arm": "A",
                "stage": "addition",
                "nodes": nodes,
                "parent": None,
            },
            "model": config["model"],
            "initial_model_sha256": config["model_revision"],
            "tracking_group": config["batch"],
            "world_sha256": M.sha(batch / "data/development/manifest.json"),
            "parameters": {
                "trainable_names": ["model.layers.0.mlp.down_proj.weight"],
                "trainable_parameters": 10,
                "total_parameters": 100,
            },
        },
    )
    records = [{"step": n // 512, "supervised_tokens": n, "replay_tokens": n // 4} for n in nodes]
    M.write(run / "learning.json", records)
    M.write(
        run / "complete.json",
        {
            **records[-1],
            "nodes": nodes,
            "smoke": False,
            "parent_optimizer_reset": False,
            "teacher_frozen": True,
        },
    )
    for n in nodes:
        node = run / f"node-{n}"
        node.mkdir(parents=True)
        (node / "resume.pt").write_bytes(b"checkpoint fixture")
        (node / "reload-reference.pt").write_bytes(b"reference fixture")
        M.write(node / "audit.json", {"passed": True, "weights_sha256": M.sha(node / "resume.pt")})
        evaluation_fixture(batch, node / "evaluation")
    return run


def test_node_completion_requires_finished_cloud_identity(setup):
    plan, config = setup
    batch = Path(plan["output"])
    run = "development-pd-A-addition"
    trajectory_fixture(plan, config)
    node = batch / "runs" / run / "node-200192"
    M.write(batch / "tracking-wandb" / run / "state.json", {"run_id": "original"})
    cloud = {
        "passed": True,
        "state": "finished",
        "config_equal": True,
        "node_summary_equal": True,
        "run_id": "different",
    }
    M.write(node / "cloud-verification.json", cloud)
    spec = {
        "cohort": "development",
        "objective": "pd",
        "arm": "A",
        "stage": "addition",
        "action": "evaluate",
        "nodes": [200192],
    }
    with pytest.raises(ValueError, match="cloud identity"):
        M.verify_outputs(batch, spec, {}, config, A)
    cloud["run_id"] = "original"
    M.write(node / "cloud-verification.json", cloud)
    assert M.verify_outputs(batch, spec, {}, config, A)
    (node / "resume.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="reload"):
        M.verify_outputs(batch, spec, {}, config, A)


def test_baseline_requires_rewrite_context_raw_predictions(setup):
    plan, config = setup
    batch = Path(plan["output"])
    base = batch / "baseline/development"
    spec = {"cohort": "development", "action": "baseline"}
    evaluation_fixture(batch, base / "closed")
    evaluation_fixture(batch, base / "context", context=True)
    with pytest.raises(ValueError, match="Evaluation contract incomplete"):
        M.verify_outputs(batch, spec, {}, config, A)
    evaluation_fixture(batch, base / "rewrite-context", "rewrite", context=True)
    assert M.verify_outputs(batch, spec, {}, config, A)
    (base / "rewrite-context/predictions.jsonl").unlink()
    with pytest.raises(ValueError, match="Evaluation contract incomplete"):
        M.verify_outputs(batch, spec, {}, config, A)


def test_raw_atomic_results_must_match_selection_summary(setup):
    plan, config = setup
    batch = Path(plan["output"])
    directory = batch / "evaluation"
    evaluation_fixture(batch, directory, correct=False)
    summary = M.read(directory / "summary.json")
    summary["groups"]["atomic_seen/direct"].update(correct=1, accuracy=1.0)
    M.write(directory / "summary.json", summary)
    with pytest.raises(ValueError, match="counts differ"):
        M.verified_evaluation(batch, directory, "development", "addition", config, A)


def test_train_completion_rejects_wrong_arm_and_replay_budget(setup):
    plan, config = setup
    batch = Path(plan["output"])
    run = trajectory_fixture(plan, config)
    spec = {
        "cohort": "development",
        "objective": "pd",
        "arm": "A",
        "stage": "addition",
        "action": "train",
        "nodes": config["development_token_nodes"],
    }
    assert M.verify_outputs(batch, spec, {}, config, A)
    metadata = M.read(run / "run.json")
    metadata["spec"]["arm"] = "B"
    M.write(run / "run.json", metadata)
    with pytest.raises(ValueError, match="trajectory contract"):
        M.verify_outputs(batch, spec, {}, config, A)
    metadata["spec"]["arm"] = "A"
    M.write(run / "run.json", metadata)
    done = M.read(run / "complete.json")
    done["replay_tokens"] -= 1
    M.write(run / "complete.json", done)
    with pytest.raises(ValueError, match="main/replay"):
        M.verify_outputs(batch, spec, {}, config, A)


def test_recovery_of_failed_container_never_recreates_or_restarts(setup, monkeypatch):
    control = controller(setup)
    name = "development-prepare"
    job = control.state["jobs"][name]
    job.update(state="running", lane=control.plan["lanes"][0], container_id="original")
    (control.batch / "execution" / name).mkdir(parents=True)
    monkeypatch.setattr(
        control,
        "inspect",
        lambda _: {"id": "original", "state": {"Status": "exited", "ExitCode": 1}},
    )
    monkeypatch.setattr(
        M.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        control, "lease", lambda _: pytest.fail("Exited container must not reacquire GPU lease")
    )
    monkeypatch.setattr(M, "call", lambda *_: pytest.fail("No restart or duplicate allowed"))
    assert control.tick() == "needs_attention"
    assert job["container_id"] == "original" and job["state"] == "needs_attention"


def test_interrupted_cpu_stage_stops_for_review(setup):
    control = controller(setup)
    control.state["jobs"]["select-budget"]["state"] = "cpu_running"
    assert control.tick() == "needs_attention"
    assert "Interrupted CPU stage" in control.state["jobs"]["select-budget"]["error"]
