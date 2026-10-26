"""Scope and exclusive-lane scheduling checks; no Docker calls or training."""

import copy
import importlib
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
launcher = importlib.import_module("launch_paper_update_continuation")


def config():
    return {
        "batch": launcher.BATCH,
        "arms": dict.fromkeys(launcher.ARMS, {}),
        "continuation": {"start_step": 300000},
        "training": {"steps": 1500000, "lr": 1e-4},
        "budget": {"new_scientific_updates": 4800000},
        "evaluation_nodes": [300000, 600000, 1500000],
        "runtime": {
            "docker_context": "lm-memory",
            "tracking": "online",
            "memory_gib": 16,
            "cpu_threads": 2,
            "image": "sha256:test",
            "gpu_memory_fraction": 0.16,
        },
    }


@pytest.fixture
def controller(tmp_path):
    (tmp_path / "freeze.json").write_text(json.dumps({"image": "sha256:test", "files": {}}))
    return launcher.Controller(
        config(),
        tmp_path,
        tmp_path / "mirror",
        "configs/continuation.json",
        {1: "GPU-1", 2: "GPU-2"},
        tmp_path / "private-token",
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("continuation", "start_step", 0),
        ("training", "steps", 300000),
        ("training", "lr", 5e-5),
        ("budget", "new_scientific_updates", 6000000),
        ("runtime", "tracking", "offline"),
        ("runtime", "docker_context", "default"),
        ("runtime", "maximum_parallel", 4),
    ],
)
def test_config_rejects_changed_scope(section, key, value):
    selected = copy.deepcopy(config())
    selected[section][key] = value
    with pytest.raises(ValueError):
        launcher.validate_config(selected)


def test_four_arm_matrix_is_exact_and_insertion_order_is_fixed():
    selected = config()
    launcher.validate_config(selected)
    selected["arms"]["extra"] = {}
    with pytest.raises(ValueError, match="four ordered"):
        launcher.validate_config(selected)


def test_commands_bind_readonly_inputs_and_reuse_two_cpu_pairs(controller):
    allocations = {}
    for lane, names in enumerate(launcher.LANES):
        for name in names:
            gpu = lane + 1
            label, command, cpus = controller.command(name, "train", gpu, f"GPU-{gpu}")
            allocations[name] = cpus
            assert command[:4] == ["docker", "--context", "lm-memory", "create"]
            assert command[command.index("--gpus") + 1] == f"device=GPU-{gpu}"
            assert command[command.index("--cpuset-cpus") + 1] == ",".join(map(str, cpus))
            assert len(cpus) == 2
            assert command[command.index(launcher.RUNNER) + 1 :] == [
                "--stage",
                "train",
                "--config",
                "configs/continuation.json",
                "--arm",
                name,
                "--output",
                "/output",
            ]
            assert any("target=/inputs,readonly" in value for value in command)
            assert any("target=/project,readonly" in value for value in command)
            assert not any(f"target={launcher.original.DATA_TARGET}" in value for value in command)
            assert str(controller.token_file) not in " ".join(command)
            assert label.endswith(f"{name}-train")
    assert allocations["token-update"] == allocations["paper"]
    assert allocations["token-update-half-lr"] == allocations["fixed-update"]
    assert not set(allocations["token-update"]) & set(allocations["token-update-half-lr"])


def test_commands_reject_other_lane_and_unauthorized_stages(controller):
    with pytest.raises(ValueError, match="another lane"):
        controller.command("paper", "train", 2, "GPU-2")
    with pytest.raises(ValueError, match="another lane"):
        controller.command("paper", "train", 1, "GPU-foreign")
    with pytest.raises(ValueError, match="Only training"):
        controller.command("paper", "audit", 1, "GPU-1")


def test_two_gate_branches_start_together_and_each_lane_stays_serial(controller, monkeypatch):
    first_pair = threading.Barrier(2, timeout=5)
    observed, active = [], set()
    guard = threading.Lock()

    def execute(name, stage, gpu, uuid):
        with guard:
            assert gpu not in active
            active.add(gpu)
            observed.append((name, stage, gpu, uuid))
        if name in ("token-update", "token-update-half-lr") and stage == "train":
            first_pair.wait()
        with guard:
            active.remove(gpu)

    monkeypatch.setattr(controller, "execute", execute)
    monkeypatch.setattr(controller, "mirror_output", lambda force=False: None)
    assert controller.run()
    for gpu, names in zip((1, 2), launcher.LANES, strict=True):
        assert [(name, stage) for name, stage, chosen, _ in observed if chosen == gpu] == [
            (name, stage) for name in names for stage in ("train", "grid")
        ]
    assert len(observed) == 8
    assert controller.state["state"] == (
        "training_and_grid_complete_independent_audit_not_requested"
    )


def test_failed_arm_is_retained_without_retry_and_other_lane_finishes(controller, monkeypatch):
    observed = []

    def execute(name, stage, gpu, uuid):
        observed.append((name, stage, gpu, uuid))
        if name == "token-update":
            raise RuntimeError("recorded stage failure")

    monkeypatch.setattr(controller, "execute", execute)
    monkeypatch.setattr(controller, "mirror_output", lambda force=False: None)
    assert controller.run() is False
    pairs = [(name, stage) for name, stage, _, _ in observed]
    assert pairs.count(("token-update", "train")) == 1
    assert ("token-update", "grid") not in pairs
    assert controller.state["arms"]["token-update"]["state"] == "needs_attention"
    for name in ("paper", "fixed-update", "token-update-half-lr"):
        assert controller.state["arms"][name]["state"] == "training_complete_audit_not_requested"
    assert controller.state["state"] == "needs_attention"


def test_executor_does_not_ignore_own_reserved_gpu(controller, monkeypatch):
    monkeypatch.setattr(launcher.original, "verify_frozen", lambda *args: None)
    observed = []

    def snapshot(*args, **kwargs):
        observed.append((args, kwargs))
        return (
            {1: {"uuid": "GPU-1", "used": 0, "free": 80000, "utilization": 0}},
            {1: [f"/{launcher.BATCH}-own-active-container"]},
        )

    monkeypatch.setattr(launcher.original, "snapshot", snapshot)
    with pytest.raises(ValueError, match="no longer independently idle"):
        controller.execute("paper", "train", 1, "GPU-1")
    assert observed == [((), {})]
    assert not (controller.output / "execution").exists()
