"""Scientific contracts and scheduling barriers for the new capacity batch."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.capacity_scaling import (
    ANSWER_LENGTH,
    CONTEXT,
    VOCAB,
    digest_arrays,
    generate_world,
    grid_capacity,
    information_ledger,
    model_parameter_count,
    pack,
    prerequisite_pass,
)
from llm_memory_editability.grok_depth import SmallGPT

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def spec():
    return {
        "world_seed": 917,
        "heads_n": 64,
        "max_heads": 256,
        "values_n": 256,
        "relations": 4,
        "evaluation_per_pool": 2048,
    }


def test_truth_entropy_and_role_splits(spec):
    w = generate_world(spec)
    trained = set(map(tuple, w["train_composition"][:, :4]))
    used = set(w["train_composition"][:, 5:].ravel())
    seen_queries = set()
    for pool in ("II", "IO", "OI", "OO", "train_composition"):
        for row in w[pool]:
            _, h, r1, r2, answer, a, b = row
            bridge = w["first"][h, r1]
            assert answer == w["second"][bridge, r2]
            assert w["atomic"][a, 4] == bridge
            assert w["atomic"][b, 1] == bridge
            assert w["atomic"][b, 4] == answer
            if pool != "train_composition":
                assert tuple(row[:4]) not in trained
                assert tuple(row[:4]) not in seen_queries
                seen_queries.add(tuple(row[:4]))
                assert (a in used) == (pool[0] == "I")
                assert (b in used) == (pool[1] == "I")
    ledger = information_ledger(w, 1000, np.log(256))
    assert ledger["world_bits"] == (64 * 4 + 256 * 4) * 8
    assert abs(ledger["learned_bits_raw"]) < 1e-9
    assert ledger["composition_independent_bits"] == 0
    assert len(w["OO"]) > 0


def test_nested_world_and_fixed_target_identical(spec):
    low = generate_world(spec)
    high = generate_world({**spec, "heads_n": 256, "support_heads_n": 64})
    assert np.array_equal(low["first"], high["first"][:64])
    assert np.array_equal(low["second"], high["second"])
    for pool in ("train_composition", "II", "IO", "OI", "OO"):
        assert np.array_equal(low[pool][:, :5], high[pool][:, :5])
        # Global IDs differ because background atoms precede B->C atoms.
        for hop in (5, 6):
            assert np.array_equal(
                low["atomic"][low[pool][:, hop], :5], high["atomic"][high[pool][:, hop], :5]
            )
    assert digest_arrays(generate_world(spec)) == digest_arrays(low)


def test_packing_no_answer_in_query(spec):
    w = generate_world(spec)
    for pool in ("atomic", "train_composition"):
        rows = w[pool][:20].copy()
        x, pos, labels = pack(rows)
        changed = rows.copy()
        changed[:, 4] = (rows[:, 4] + 13) % 256
        other, _, other_labels = pack(changed)
        assert x.shape[1] == CONTEXT and labels.shape[1] == ANSWER_LENGTH
        assert x.max() < VOCAB and labels.max() < VOCAB
        for i in range(len(rows)):
            assert np.array_equal(x[i, : pos[i, 0] + 1], other[i, : pos[i, 0] + 1])
            assert not np.array_equal(labels[i], other_labels[i])
        assert np.array_equal(x[np.arange(len(x))[:, None], pos[:, 1:]], labels[:, :-1])


@pytest.mark.parametrize("width", [64, 96, 128])
def test_parameter_accounting(width):
    m = SmallGPT(ModelConfig(VOCAB, width, 2, 4, CONTEXT), dropout=0.0)
    assert sum(p.numel() for p in m.parameters()) == model_parameter_count(width)


def test_grid_bounds_and_nonmonotonicity():
    points = [{"bits": 10, "atomic": 1.0, "II": 0.9}, {"bits": 20, "atomic": 1.0, "II": 0.7}]
    result = grid_capacity(points, "II", 0.8, 0.95)
    assert result["largest_observed_passing_bits"] == 10
    assert result["next_observed_failing_bits"] == 20
    assert not result["nonmonotonic_grid"]
    points.append({"bits": 30, "atomic": 1.0, "II": 0.9})
    assert grid_capacity(points, "II", 0.8, 0.95)["nonmonotonic_grid"]
    assert grid_capacity(points, "II", 0.99, 0.95)["censoring"] == "below_grid"


def test_gate_does_not_use_oo():
    m = {
        "atomic": {"accuracy": 0.99},
        "train_composition": {"accuracy": 0.99},
        "II": {"accuracy": 0.85},
        "OO": {"accuracy": 0},
    }
    t = {"atomic": 0.95, "train_composition": 0.95, "II": 0.8}
    assert prerequisite_pass(m, t)
    m["II"]["accuracy"] = 0.79
    assert not prerequisite_pass(m, t)


def test_predecessor_requires_entire_matrix_and_audits(tmp_path):
    scheduler = script("execute_capacity_scaling")
    (tmp_path / "frozen-config.json").write_text(json.dumps({"runs": [{"name": "a"}]}))
    state = {"state": "complete", "completed": ["a"], "active": {}, "queued": [], "failed": []}
    path = tmp_path / "controller-state.json"
    path.write_text(json.dumps(state))
    assert not scheduler.predecessor_status([tmp_path])[0]
    out = tmp_path / "runs/a"
    out.mkdir(parents=True)
    (out / "audit.json").write_text('{"passed":true}')
    assert scheduler.predecessor_status([tmp_path])[0]
    state["active"] = {"2": {"name": "a"}}
    path.write_text(json.dumps(state))
    assert not scheduler.predecessor_status([tmp_path])[0]


def test_container_uses_frozen_worker_isolation_and_gpu(tmp_path):
    scheduler = script("execute_capacity_scaling")
    c = json.loads((ROOT / "configs/capacity-scaling-development-v1.json").read_text())
    c.update(repository=str(ROOT), source_root=str(tmp_path))
    name, cmd = scheduler.command_builder(
        c, tmp_path / "config.json", {"name": "test"}, tmp_path / "run", 3, 1
    )
    assert name.startswith("lm-capacity-")
    assert cmd[:3] == ["docker", "--context", "lm-memory"]
    assert cmd[cmd.index("--gpus") + 1] == "device=3"
    assert "--read-only" in cmd and "--network=none" in cmd
    assert str(tmp_path / "scripts/execute_capacity_scaling.py") in cmd
    assert str(tmp_path / "scripts/execute_parametric_architecture.py") not in cmd


def test_stage_provides_a_matching_execution_lock_to_shared_scheduler(tmp_path, monkeypatch):
    controller = script("execute_capacity_scaling")
    config = json.loads((ROOT / "configs/capacity-scaling-development-v1.json").read_text())
    config.update(
        results_root=str(tmp_path / "results"), source_root=str(tmp_path / "source"), gpus=[0, 1]
    )
    specs = [controller.make_spec(config, "calibration-test")]
    calls = []

    def check_scheduler(stage_config, path, root, command_builder):
        # These are the actual shared scheduler's required entry contracts.
        assert (
            controller.digest(path)
            == controller.read(stage_config["execution_lock"])["config_sha256"]
        )
        assert stage_config["gpus"] == [0, 1]
        assert stage_config["runs"] == specs
        calls.append(path)
        controller.write(root / "runs/calibration-test/audit.json", {"passed": True})
        controller.write(
            root / "controller-state.json", {"state": "complete", "completed": ["calibration-test"]}
        )
        controller.write(root / "tracking-completion.json", {"passed": True})

    monkeypatch.setattr(controller, "devices_free", lambda _: True)
    monkeypatch.setattr(controller.scheduler, "control_locked", check_scheduler)
    first = controller.run_stage(config, "calibration", specs)
    assert first == controller.run_stage(config, "calibration", specs)
    assert len(calls) == 1


def test_free_generation_not_teacher_forced(spec):
    runner = script("run_capacity_scaling")
    world = generate_world(spec)
    rows = world["II"][:8]
    altered = rows.copy()
    altered[:, 4] = (altered[:, 4] + 1) % 256
    torch.set_num_threads(1)
    m = SmallGPT(ModelConfig(VOCAB, 16, 2, 4, CONTEXT), dropout=0.0)
    p, _ = runner.evaluate_rows(m, rows, torch.device("cpu"), 8)
    q, _ = runner.evaluate_rows(m, altered, torch.device("cpu"), 8)
    assert np.array_equal(p, q)


def test_pipeline_selects_fit_and_stops_after_failed_long_calibration(tmp_path, monkeypatch):
    scheduler = script("execute_capacity_scaling")
    config = json.loads((ROOT / "configs/capacity-scaling-development-v1.json").read_text())
    config.update(results_root=str(tmp_path / "results"), source_files={})
    path = tmp_path / "frozen-config.json"
    path.write_text(json.dumps(config))
    (tmp_path / "execution-lock.json").write_text(
        json.dumps({"config_sha256": scheduler.digest(path)})
    )
    monkeypatch.setattr(scheduler, "predecessor_status", lambda _: (True, []))
    monkeypatch.setattr(scheduler, "devices_free", lambda _: True)
    called = []

    def fake_stage(config, stage, specs):
        called.append(stage)
        result = {}
        for spec in specs:
            # Better held-out performance must not win the LR selection.
            low_lr = spec["learning_rate"] == 0.001
            metrics = {
                "atomic": {"accuracy": 1.0, "answer_nll": 0.1 if low_lr else 0.2},
                "train_composition": {"accuracy": 1.0, "answer_nll": 0.1 if low_lr else 0.2},
                "II": {"accuracy": 0.1 if low_lr else 1.0},
            }
            out = tmp_path / stage / spec["name"]
            out.mkdir(parents=True)
            (out / "run.json").write_text(json.dumps({"spec": spec}))
            (out / "audit.json").write_text(json.dumps({"passed": True, "metrics": metrics}))
            result[spec["name"]] = out
        return result

    monkeypatch.setattr(scheduler, "run_stage", fake_stage)
    scheduler.controller(path)
    assert called == ["calibration", "calibration-long"]
    state = json.loads((tmp_path / "results/controller-state.json").read_text())
    assert state["state"] == "development_prerequisite_not_met"
    decision = json.loads((tmp_path / "results/calibration-decision.json").read_text())
    assert decision["recipe"]["learning_rate"] == 0.001
    assert decision["recipe"]["epochs"] == 3000


def test_interrupted_training_restores_weights_optimizer_and_exposure(tmp_path, monkeypatch):
    runner = script("run_capacity_scaling")
    config = json.loads((ROOT / "configs/capacity-scaling-development-v1.json").read_text())
    spec = {
        **config["base_spec"],
        "name": "resume-test",
        "heads_n": 4,
        "max_heads": 64,
        "width": 16,
        "epochs": 2,
        "batch_size": 1024,
        "evaluation_per_pool": 4,
        "evaluation_epochs": [0, 1, 2],
        "checkpoint_epochs": 1,
    }
    original_step = torch.optim.AdamW.step
    calls = 0

    def fail_after_checkpoint(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", fail_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train(config, spec, tmp_path / "resumed", torch.device("cpu"))
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    runner.train(config, spec, tmp_path / "resumed", torch.device("cpu"))
    runner.train(config, spec, tmp_path / "continuous", torch.device("cpu"))
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=False)
    continuous = torch.load(tmp_path / "continuous/latest.pt", weights_only=False)
    assert resumed["step"] == continuous["step"]
    assert np.array_equal(resumed["counts"], continuous["counts"])
    assert np.all(resumed["counts"] == 2)
    for key in resumed["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], atol=0, rtol=0)
