"""Execution guarantees affecting the zero-training scientific interpretation."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "z1_execution", PROJECT / "scripts/execute_path_overlap.py"
)
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


@pytest.fixture
def frozen(tmp_path):
    base = tmp_path / "batch"
    root = execution.input_root(base)
    run, node = "loop-parent", 1000
    files = {
        "data/world.npz": b"world",
        "data/panels.npz": b"panels",
        "data/complete.json": b"{}",
        f"development/{run}/learning.json": b"[]",
        f"development/{run}/checkpoint-{node:07d}.pt": b"checkpoint",
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    cfg = {
        "batch": "zero-training-path-overlap-v2",
        "source": {
            "runs": [run],
            "nodes": [node],
            "world_sha256": execution.sha(root / "data/world.npz"),
        },
    }
    execution.write(execution.source(base) / execution.CONTRACT_NAME, cfg)
    execution.write(
        base / "input-receipt.json",
        {
            "checkpoints": [
                {
                    "run": run,
                    "step": node,
                    "sha256": execution.sha(root / f"development/{run}/checkpoint-{node:07d}.pt"),
                }
            ],
            "files": [
                {
                    "path": str(Path("inputs/grokking-reproduction-v1") / name),
                    "sha256": execution.sha(root / name),
                }
                for name in files
                if "checkpoint-" not in name
            ],
        },
    )
    execution.write(
        base / "frozen/freeze.json",
        {
            "source_files": {
                execution.CONTRACT_NAME: execution.sha(
                    execution.source(base) / execution.CONTRACT_NAME
                )
            },
            "contract": {
                "path": execution.CONTRACT_NAME,
                "sha256": execution.sha(execution.source(base) / execution.CONTRACT_NAME),
            },
            "inputs": {
                "root": str(root),
                "receipt_sha256": execution.sha(base / "input-receipt.json"),
                "files": {
                    name: {
                        "sha256": execution.sha(root / name),
                        "bytes": len(data),
                        "resolved_path": str((root / name).resolve()),
                    }
                    for name, data in files.items()
                },
            },
            "image_sha256": execution.IMAGE,
            "optimizer_updates": 0,
        },
    )
    return base, run, node


def test_checkpoint_checked_against_prior_receipt_before_load(frozen):
    base, run, node = frozen
    assert execution.verify_checkpoint(base, run, node)
    path = execution.input_root(base) / f"development/{run}/checkpoint-{node:07d}.pt"
    path.write_bytes(b"different weight")
    with pytest.raises(ValueError, match="prior source receipt"):
        execution.verify_checkpoint(base, run, node)


def test_task_verifies_metadata_and_full_verification_verifies_weights(frozen):
    base, run, node = frozen
    execution.verify_freeze(base, run)
    path = execution.input_root(base) / f"development/{run}/checkpoint-{node:07d}.pt"
    path.write_bytes(b"different weight")
    # The worker checks weights immediately before each model load, while its
    # initial task check verifies metadata. A full launch check verifies both.
    execution.verify_freeze(base, run)
    with pytest.raises(ValueError, match="Checkpoint changed"):
        execution.verify_freeze(base, all_inputs=True)
    (execution.input_root(base) / "data/complete.json").write_text('{"bad":1}')
    with pytest.raises(ValueError, match="Frozen input changed"):
        execution.verify_freeze(base, run)


def test_source_and_receipt_are_immutable(frozen):
    base, run, _ = frozen
    path = execution.source(base) / execution.CONTRACT_NAME
    path.write_text("{}")
    with pytest.raises(ValueError, match="Frozen source changed"):
        execution.verify_freeze(base, run)
    receipt = execution.read(base / "input-receipt.json")
    receipt["checkpoints"][0]["sha256"] = "bad"
    execution.write(base / "input-receipt.json", receipt)
    with pytest.raises(ValueError, match="input receipt changed"):
        execution.verify_freeze(base, run)


def feasibility(loop, standard, secondary=False):
    return {
        "groups": {
            name: {"state": "complete", "expected_inconclusive": value}
            for name, value in (
                ("loop_ood", loop),
                ("standard_id", standard),
                ("loop_id_secondary", secondary),
            )
        }
    }


@pytest.mark.parametrize(
    "loop,standard,allowed",
    [(True, True, False), (False, True, True), (True, False, True), (False, False, True)],
)
def test_fixed_feasibility_gate_uses_main_groups_only(loop, standard, allowed):
    assert execution.selection_decision(feasibility(loop, standard)) is allowed


def test_incomplete_selection_does_not_authorize_science():
    summary = feasibility(False, False)
    summary["groups"]["loop_ood"]["state"] = "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        execution.selection_decision(summary)


def make_audit_nodes(base, run, node):
    frozen = execution.read(base / "frozen/freeze.json")
    for prefix in (base, base / "audit-replay"):
        folder = prefix / run / f"{node:07d}"
        execution.write(folder / "z1.json", {"value": 0.123456789})
        execution.write(folder / "ported.json", {"value": [1, 2, 3]})
        execution.write(
            folder / "hashes.json",
            {
                "complete": True,
                "run": run,
                "step": node,
                "replication": {"failures": []},
                "checkpoint_sha256": execution.verify_checkpoint(base, run, node),
                "world_sha256": frozen["inputs"]["files"]["data/world.npz"]["sha256"],
                "panels_sha256": frozen["inputs"]["files"]["data/panels.npz"]["sha256"],
                "contract_sha256": frozen["contract"]["sha256"],
                "code_sha256": frozen["source_files"],
                "tf32": False,
                "outputs_sha256": {
                    name: execution.sha(folder / name) for name in ("z1.json", "ported.json")
                },
            },
        )


def test_independent_replay_is_exact_and_retains_three_node_files(frozen):
    base, run, node = frozen
    make_audit_nodes(base, run, node)
    assert execution.audit_node(base, run, node)["passed"]
    replay = base / "audit-replay" / run / f"{node:07d}"
    execution.write(replay / "z1.json", {"value": 0.123456788})
    hashes = execution.read(replay / "hashes.json")
    hashes["outputs_sha256"]["z1.json"] = execution.sha(replay / "z1.json")
    execution.write(replay / "hashes.json", hashes)
    with pytest.raises(ValueError, match="Exact independent replay"):
        execution.audit_node(base, run, node)
    (replay / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly the inherited three files"):
        execution.audit_node(base, run, node)


def test_equal_outputs_cannot_hide_provenance_mismatch(frozen):
    base, run, node = frozen
    make_audit_nodes(base, run, node)
    replay_hashes = base / "audit-replay" / run / f"{node:07d}/hashes.json"
    value = execution.read(replay_hashes)
    value["checkpoint_sha256"] = "bad"
    execution.write(replay_hashes, value)
    with pytest.raises(ValueError, match="provenance differs"):
        execution.audit_node(base, run, node)


def test_analysis_indices_are_not_optimizer_steps():
    nodes = [
        0,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
        256000,
        512000,
        700000,
        1000000,
        1500000,
    ]
    assert execution.analysis_index("select", 1500000, nodes) == 14
    assert execution.analysis_index("science", 0, nodes) == 15
    assert execution.analysis_index("audit", 1500000, nodes) == 42


def test_gpu_cpu_container_isolation_and_readonly_symlink_sources(frozen):
    base, run, _ = frozen
    root = execution.input_root(base)
    value = execution.read(base / "frozen/freeze.json")
    value["inputs"]["files"][f"development/{run}/checkpoint-0001000.pt"]["resolved_path"] = (
        "/outside/original/weights/checkpoint.pt"
    )
    execution.write(base / "frozen/freeze.json", value)
    name, command = execution.container_command(base, run, "science", 4, list(range(24, 32)))
    assert name.endswith("science")
    assert command[:3] == [execution.DOCKER, "--context", "lm-memory"]
    assert command[command.index("--gpus") + 1] == "device=4"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--memory") + 1] == "48g"
    assert command[command.index("--cpuset-cpus") + 1] == "24,25,26,27,28,29,30,31"
    assert (
        "type=bind,source=/outside/original/weights,target=/outside/original/weights,readonly"
        in command
    )
    assert "WANDB_API_KEY" not in " ".join(command)
    assert command[-2:] == ["--physical-gpu", "4"]
    assert root == base / "inputs/grokking-reproduction-v1"


def test_cpu_slots_never_overlap_running_container():
    groups = execution.cpu_allocations(range(96), 6, occupied=range(32, 40))
    assert len(set(sum(groups, []))) == 48
    assert not set(sum(groups, [])) & set(range(32, 40))
    with pytest.raises(RuntimeError, match="Insufficient"):
        execution.cpu_allocations(range(40), 6)
    assert execution.parse_cpuset("2-4,7,11-12") == [2, 3, 4, 7, 11, 12]


def test_science_worker_cannot_bypass_selection_gate(frozen):
    base, run, _ = frozen
    execution.write(base / "selection-stage/selection-summary.json", feasibility(True, True))
    args = argparse.Namespace(
        base=base, run=run, stage="science", steps=None, gpu=None, physical_gpu=None
    )
    with pytest.raises(RuntimeError, match="selection prerequisite"):
        execution.worker(args)
    assert not (base / run).exists()


def test_worker_refuses_existing_stage_outputs(frozen):
    base, run, node = frozen
    (base / "selection-stage" / run / f"{node:07d}").mkdir(parents=True)
    args = argparse.Namespace(
        base=base, run=run, stage="select", steps=None, gpu=None, physical_gpu=None
    )
    with pytest.raises(RuntimeError, match="Existing stage output"):
        execution.worker(args)


def test_host_credentials_stay_outside_controller_environment(frozen, monkeypatch):
    base, _, _ = frozen
    monkeypatch.setenv("WANDB_API_KEY", "test-secret")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/prior/lib")
    env = execution.host_environment(base)
    assert "WANDB_API_KEY" not in env
    assert env["LD_LIBRARY_PATH"] == "/lib64:/prior/lib"


def test_controller_runs_all_selection_before_any_science(frozen, monkeypatch):
    base, run, _ = frozen
    execution.write(
        base / "launch-resource-check.json", {"gpus": [2], "cpus": [list(range(24, 32))]}
    )
    events = []
    monkeypatch.setattr(execution, "verify_freeze", lambda *args, **kwargs: None)

    def monitor(base, runs, slots, stage, **options):
        events.append(stage)
        execution.write(execution.run_dir(base, run) / "learning.json", [])
        return [run], []

    monkeypatch.setattr(execution, "monitor_stage", monitor)

    def summarize(base, action):
        events.append(action)
        execution.write(base / "selection-stage/selection-summary.json", feasibility(True, True))

    monkeypatch.setattr(execution, "summary_command", summarize)
    execution.controller(SimpleNamespace(base=base))
    assert events == ["select", "select-summary"]
    assert (
        execution.read(base / "controller-state.json")["state"]
        == "development_prerequisite_not_met"
    )
    assert not (execution.run_dir(base, run) / "audit.json").exists()
    assert not (execution.run_dir(base, run) / "complete.json").exists()


def fake_runner_factory(base, calls):
    def load(base):
        runner = SimpleNamespace(write_json=execution.write)

        def checked(*args):
            calls.append("loaded")

        runner.checked_model = checked

        def run(args):
            frozen = execution.read(base / "frozen/freeze.json")
            for node in args.steps:
                runner.checked_model(args.root, args.run, node)
                folder = args.out / args.run / f"{node:07d}"
                runner.write_json(
                    folder / "z1.json",
                    {
                        "atomic_all_answer_accuracy": 0.9,
                        "primary_design": {"kinds": {}},
                        "fallback_design": {"kinds": {}},
                    },
                )
                runner.write_json(folder / "ported.json", {"values": [0.1, 0.2]})
                runner.write_json(
                    folder / "hashes.json",
                    {
                        "complete": True,
                        "run": args.run,
                        "step": node,
                        "seconds": {"total": 1.2},
                        "replication": {"failures": []},
                        "checkpoint_sha256": execution.verify_checkpoint(base, args.run, node),
                        "world_sha256": frozen["inputs"]["files"]["data/world.npz"]["sha256"],
                        "panels_sha256": frozen["inputs"]["files"]["data/panels.npz"]["sha256"],
                        "contract_sha256": frozen["contract"]["sha256"],
                        "code_sha256": frozen["source_files"],
                        "tf32": False,
                        "outputs_sha256": {
                            name: execution.sha(folder / name)
                            for name in ("z1.json", "ported.json")
                        },
                    },
                )

        runner.run = run
        return runner

    return load


def test_science_has_no_completion_until_fresh_process_replay(frozen, monkeypatch):
    base, run, _ = frozen
    execution.write(base / "selection-stage/selection-summary.json", feasibility(False, True))
    calls = []
    monkeypatch.setattr(execution, "load_runner", fake_runner_factory(base, calls))
    args = argparse.Namespace(
        base=base, run=run, stage="science", steps=None, gpu=None, physical_gpu=None
    )
    execution.worker(args)
    out = execution.run_dir(base, run)
    assert not (out / "complete.json").exists()
    assert not (out / "audit.json").exists()
    args.stage = "audit"
    with pytest.raises(RuntimeError, match="new process"):
        execution.worker(args)
    monkeypatch.setattr(execution.os, "getpid", lambda: 987654321)
    execution.worker(args)
    assert execution.read(out / "audit.json")["fresh_process"] is True
    assert execution.read(out / "complete.json")["state"] == "complete"
    assert [row["step"] for row in execution.read(out / "learning.json")] == [2, 3]
    assert all(row["optimizer_updates"] == 0 for row in execution.read(out / "learning.json"))
    assert calls == ["loaded", "loaded"]


def test_changed_checkpoint_stops_before_model_load_and_preserves_failure(frozen, monkeypatch):
    base, run, node = frozen
    execution.write(base / "selection-stage/selection-summary.json", feasibility(False, True))
    calls = []
    monkeypatch.setattr(execution, "load_runner", fake_runner_factory(base, calls))
    (execution.input_root(base) / f"development/{run}/checkpoint-{node:07d}.pt").write_bytes(
        b"wrong"
    )
    args = argparse.Namespace(
        base=base, run=run, stage="science", steps=None, gpu=None, physical_gpu=None
    )
    with pytest.raises(ValueError, match="prior source receipt"):
        execution.worker(args)
    assert calls == []
    assert execution.read(execution.run_dir(base, run) / "failure.json")["stage"] == "science"
    assert not (base / run).exists()


def test_monitor_never_retries_failed_run_and_replays_success_separately(tmp_path, monkeypatch):
    started = []

    def start(base, run, stage, gpu, cpus):
        started.append((run, stage))
        return {
            "process": SimpleNamespace(poll=lambda: 1 if run == "bad" else 0),
            "run": run,
            "stage": stage,
            "gpu": gpu,
            "cpus": cpus,
            "container": f"{run}-{stage}",
        }

    monkeypatch.setattr(execution, "resource_check", lambda *args, **kwargs: {})
    monkeypatch.setattr(execution, "start_container", start)
    monkeypatch.setattr(execution, "persist_exit", lambda base, item, code: code != 0)
    monkeypatch.setattr(execution.time, "sleep", lambda seconds: None)
    complete, failed = execution.monitor_stage(
        tmp_path, ["bad", "good"], [(2, list(range(16, 24)))], "science", audit_after=True
    )
    assert complete == ["good"]
    assert len(failed) == 1
    assert started == [("bad", "science"), ("good", "science"), ("good", "audit")]
    assert (execution.run_dir(tmp_path, "bad") / "failure.json").exists()


def test_launch_reserves_gpu_zero_and_one_without_touching_resources(tmp_path):
    for gpus in ([0, 1], [1, 2], [2, 2], list(range(1, 8))):
        with pytest.raises(ValueError, match="GPUs 0/1 are reserved"):
            execution.resource_check(tmp_path, gpus)


@pytest.fixture
def adoption_pair(tmp_path):
    previous, base = tmp_path / "previous", tmp_path / "current"
    runs = ["standard8-phi3.6-wd0.1"] + [f"parent-{index}" for index in range(7)]
    nodes = [
        0,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
        256000,
        512000,
        700000,
        1000000,
        1500000,
    ]
    old_cfg = {
        "batch": "zero-training-path-overlap-v2",
        "source": {"runs": runs, "nodes": nodes, "world_sha256": "world"},
        "z1": {"sample": {"count": 256, "controls": 0.9}},
        "s1_s2_readout": {},
        "outputs": {"path": "zero-training-path-overlap-v2"},
        "not_permitted": [],
        "withdrawn": {},
        "execution": {
            "code": "same-code",
            "numerics": "float32",
            "replication_check": (
                "agreement >= 0.95 on every panel at every node with step >= 16000, "
                "and >= 0.99 on test_ii at 1.5M. the node stops only below 0.50"
            ),
        },
    }
    old_core = {
        "src/llm_memory_editability/path_overlap.py": (
            '"""Version two."""\nBATCH="zero-training-path-overlap-v2"\n'
            "def science():\n    return 1\n"
        ),
        "scripts/run_zero_training_path_overlap.py": (
            'CONTRACT="configs/zero-training-path-overlap-v2.json"\n'
        ),
        "scripts/read_path_hop_secondary.py": "def secondary():\n    return 0\n",
        "src/llm_memory_editability/grokking_reproduction.py": "def forward():\n    return 2\n",
        "src/llm_memory_editability/grok_depth.py": "def depth():\n    return 8\n",
    }
    for destination, version in ((previous, 2), (base, 3)):
        cfg = execution.version_normalized(old_cfg) if version == 3 else old_cfg
        if version == 3:
            cfg["execution"]["replication_check"] = cfg["execution"]["replication_check"].replace(
                "and >= 0.99", "and >= 0.95"
            )
        path = execution.source(destination) / (
            execution.CONTRACT_NAME if version == 3 else execution.PREVIOUS_CONTRACT
        )
        execution.write(path, cfg)
        for name, contents in old_core.items():
            if version == 3:
                contents = contents.replace(
                    "zero-training-path-overlap-v2", "zero-training-path-overlap-v3"
                ).replace("Version two.", "Version three documentation.")
            target = execution.source(destination) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents)
        root = execution.input_root(destination)
        files = {}
        for name in ("data/world.npz", "data/panels.npz"):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(name.encode())
            files[name] = {
                "sha256": execution.sha(target),
                "bytes": target.stat().st_size,
                "resolved_path": str(target),
            }
        checkpoints = []
        for run in runs:
            for node in nodes:
                name = f"development/{run}/checkpoint-{node:07d}.pt"
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(name.encode())
                files[name] = {
                    "sha256": execution.sha(target),
                    "bytes": target.stat().st_size,
                    "resolved_path": str(target),
                }
                checkpoints.append({"run": run, "step": node, "sha256": execution.sha(target)})
        execution.write(destination / "input-receipt.json", {"checkpoints": checkpoints})
        source_files = {
            str(target.relative_to(execution.source(destination))): execution.sha(target)
            for target in execution.source(destination).rglob("*")
            if target.is_file()
        }
        execution.write(
            destination / "frozen/freeze.json",
            {
                "source_files": source_files,
                "contract": {
                    "path": str(path.relative_to(execution.source(destination))),
                    "sha256": execution.sha(path),
                },
                "inputs": {
                    "root": str(root),
                    "files": files,
                    "receipt_sha256": execution.sha(destination / "input-receipt.json"),
                },
                "image_sha256": execution.IMAGE,
                "optimizer_updates": 0,
            },
        )
    frozen = execution.read(previous / "frozen/freeze.json")
    for run in runs:
        for node in nodes:
            if run == runs[0] and node == 1500000:
                continue
            execution.write(
                previous / f"selection-stage/{run}/{node:07d}/selection.json",
                {
                    "complete": True,
                    "run": run,
                    "step": node,
                    "seconds": 7.5,
                    "replication": {"failures": []},
                    "contract_sha256": frozen["contract"]["sha256"],
                    "code_sha256": {
                        name: frozen["source_files"][name] for name in list(old_core)[:3]
                    },
                    "checkpoint_sha256": frozen["inputs"]["files"][
                        f"development/{run}/checkpoint-{node:07d}.pt"
                    ]["sha256"],
                    "designs": {
                        "primary_design": {"kinds": {"id": {"sampled": 256}}},
                        "fallback_design": {"kinds": {"ood": {"sampled": 256}}},
                    },
                },
            )
    return base, previous, runs, nodes


def test_v3_adopts_exactly_111_original_records_with_explicit_provenance(adoption_pair):
    base, previous, runs, nodes = adoption_pair
    execution.adopt_v2(base, previous)
    receipt = execution.verify_adoption(base)
    assert len(receipt["nodes"]) == 111
    assert receipt["missing_nodes"] == [{"run": runs[0], "step": 1500000}]
    original = previous / f"selection-stage/{runs[1]}/1500000/selection.json"
    adopted = base / f"selection-stage/{runs[1]}/1500000/selection.json"
    assert original.read_bytes() == adopted.read_bytes()
    assert (
        execution.read(adopted)["contract_sha256"]
        == execution.read(previous / "frozen/freeze.json")["contract"]["sha256"]
    )
    assert len(execution.read(execution.run_dir(base, runs[0]) / "learning.json")) == 13
    history = execution.read(execution.run_dir(base, runs[1]) / "learning.json")
    assert len(history) == 14
    assert all(
        row["condition"] == "select_adopted_v2"
        and row["metrics"]["node"]["seconds"] == 0
        and row["optimizer_updates"] == 0
        for row in history
    )
    assert not (base / f"selection-stage/{runs[0]}/1500000").exists()
    assert not (base / runs[0]).exists()
    with pytest.raises(RuntimeError, match="already exists"):
        execution.adopt_v2(base, previous)


def test_v3_selection_computes_only_the_one_missing_node(adoption_pair, monkeypatch):
    base, previous, runs, nodes = adoption_pair
    execution.adopt_v2(base, previous)
    calls = []

    def loader(base):
        runner = SimpleNamespace(write_json=execution.write, checked_model=lambda *args: None)

        def select(args):
            calls.extend(args.steps)
            for node in args.steps:
                value = execution.read(
                    previous / f"selection-stage/{runs[1]}/{node:07d}/selection.json"
                )
                value["run"] = args.run
                value["contract_sha256"] = execution.read(base / "frozen/freeze.json")["contract"][
                    "sha256"
                ]
                runner.write_json(
                    args.out / f"selection-stage/{args.run}/{node:07d}/selection.json", value
                )

        runner.select = select
        return runner

    monkeypatch.setattr(execution, "load_runner", loader)
    execution.worker(
        argparse.Namespace(
            base=base, run=runs[0], stage="select", steps=None, gpu=None, physical_gpu=None
        )
    )
    assert calls == [1500000]
    history = execution.read(execution.run_dir(base, runs[0]) / "learning.json")
    assert len(history) == 14
    assert history[-1]["condition"] == "select"
    assert history[-1]["step"] == 14
    assert all(row["condition"] == "select_adopted_v2" for row in history[:-1])


def test_v3_controller_does_not_start_seven_completed_selection_runs(adoption_pair, monkeypatch):
    base, previous, runs, nodes = adoption_pair
    execution.adopt_v2(base, previous)
    started = []

    def start(base, run, stage, gpu, cpus):
        started.append((run, stage))
        return {
            "process": SimpleNamespace(poll=lambda: 0),
            "run": run,
            "stage": stage,
            "gpu": gpu,
            "cpus": cpus,
            "container": "fake",
        }

    monkeypatch.setattr(execution, "resource_check", lambda *args, **kwargs: {})
    monkeypatch.setattr(execution, "start_container", start)
    monkeypatch.setattr(execution, "persist_exit", lambda *args: False)
    monkeypatch.setattr(execution.time, "sleep", lambda *args: None)
    completed, failed = execution.monitor_stage(base, runs, [(2, list(range(16, 24)))], "select")
    assert failed == []
    assert set(completed) == set(runs)
    assert started == [(runs[0], "select")]


def test_adoption_rejects_unrelated_scientific_or_input_changes(adoption_pair):
    base, previous, _, _ = adoption_pair
    path = execution.source(base) / "src/llm_memory_editability/path_overlap.py"
    path.write_text(path.read_text().replace("return 1", "return 2"))
    freeze = execution.read(base / "frozen/freeze.json")
    freeze["source_files"][str(path.relative_to(execution.source(base)))] = execution.sha(path)
    execution.write(base / "frozen/freeze.json", freeze)
    with pytest.raises(ValueError, match="Scientific computation changed"):
        execution.validate_adoption_sources(base, previous)


def test_adopted_evidence_has_no_historical_process_attachment(adoption_pair):
    base, previous, runs, _ = adoption_pair
    execution.adopt_v2(base, previous)
    execution.register_adopted_tracking(base)
    for run in runs:
        metadata = execution.read(execution.run_dir(base, run) / "run.json")
        assert "pid" not in metadata and "gpu" not in metadata
        assert metadata["selection_evidence"]["origin"] == "adopted_v2"
        assert execution.read(execution.run_dir(base, run) / "container-runtime.json") == {}
