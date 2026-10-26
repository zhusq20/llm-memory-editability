"""Scientific and scheduling contracts for the six higher-LR trajectories."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from llm_memory_editability.lr_control import (
    BASE_SPEC,
    BATCH,
    RATES,
    decay_contract,
    initialization_config,
    job_config,
    matched_decay,
    rate_at,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / f"configs/{BATCH}.json").read_text())


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", CONFIG["jobs"])
def test_only_lr_and_decay_change_in_native_recipe(name):
    original = copy.deepcopy(CONFIG)
    job = job_config(CONFIG, name)
    assert CONFIG == original
    assert {k: v for k, v in job["spec"].items() if k not in ("name", "lr", "weight_decay")} == {
        k: v for k, v in BASE_SPEC.items() if k not in ("lr", "weight_decay")
    }
    assert job["spec"]["steps"] == 300000
    assert job["arms"][name]["discoloop"] == name.startswith("discoloop-")
    assert job["arms"][name]["kind"] == "none"
    assert job["arms"][name]["coefficient"] == 0
    assert job["arms"][name]["first_round_kv"] is False
    assert job["dropout_seed"] == 20261012


@pytest.mark.parametrize("name", CONFIG["jobs"])
def test_source_spec_is_used_only_for_loading_initial_tensors(name):
    job = job_config(CONFIG, name)
    before = copy.deepcopy(job)
    initial = initialization_config(job)
    assert job == before
    assert initial["spec"]["lr"] == 1e-4
    assert initial["spec"]["weight_decay"] == 0.1
    assert job["spec"]["lr"] == CONFIG["jobs"][name]["lr"]
    assert initial["arms"] == job["arms"]
    assert initial["initial_checkpoint"] == job["initial_checkpoint"]


@pytest.mark.parametrize("rate,decay", [(2e-4, 0.05), (5e-4, 0.02), (1e-3, 0.01)])
def test_complete_decay_schedule_matches_original(rate, decay):
    assert matched_decay(rate) == decay
    receipt = decay_contract(rate)
    assert receipt["passed"]
    assert receipt["maximum_per_step_product_error"] < 1e-18
    assert receipt["reference_cumulative_multiplier"] == receipt["candidate_cumulative_multiplier"]
    assert receipt["reference_cumulative_multiplier"] == pytest.approx(0.050286937199844596)


@pytest.mark.parametrize("step", [1, 2, 999, 1000, 2000, 2001, 300000])
def test_warmup_matches_original_off_by_one_and_constant_tail(step):
    for rate in RATES:
        assert rate_at(rate, step) * matched_decay(rate) == pytest.approx(
            rate_at(1e-4, step) * 0.1, abs=1e-18
        )
    assert rate_at(1e-3, 1) == 0
    assert rate_at(1e-3, 2001) == 1e-3


@pytest.mark.parametrize(
    "key,value",
    [
        ("steps", 300001),
        ("initialization", 1),
        ("stream_seed", 1),
        ("batch_size", 256),
        ("warmup", 1000),
        ("phi", 12.6),
    ],
)
def test_reject_changed_base_recipe(key, value):
    config = copy.deepcopy(CONFIG)
    config["base_spec"][key] = value
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "decay", "rate", "architecture"])
def test_reject_changed_matrix_or_unmatched_decay(mutation):
    config = copy.deepcopy(CONFIG)
    name = next(iter(config["jobs"]))
    if mutation == "missing":
        del config["jobs"][name]
    elif mutation == "extra":
        config["jobs"]["loop-lr2.5e-5"] = {"lr": 2.5e-5}
    else:
        key = {"decay": "weight_decay", "rate": "lr", "architecture": "architecture"}[mutation]
        config["jobs"][name][key] = "transformer" if mutation == "architecture" else 0.1
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("key,value", [("lr", 0.0003), ("weight_decay", 0.1), ("steps", 64)])
def test_reject_unapproved_per_job_change_before_initialization(key, value):
    name = next(iter(CONFIG["jobs"]))
    job = job_config(CONFIG, name)
    job["spec"][key] = value
    with pytest.raises(ValueError):
        initialization_config(job)


def test_new_matrix_has_exact_symmetric_budget_and_no_retrained_control():
    validate(CONFIG)
    assert len(CONFIG["jobs"]) == 6
    assert 6 * CONFIG["base_spec"]["steps"] == 1800000
    for architecture in ("loop", "discoloop"):
        assert {
            job["lr"] for job in CONFIG["jobs"].values() if job["architecture"] == architecture
        } == set(RATES)
    assert all(job["lr"] != 1e-4 for job in CONFIG["jobs"].values())
    assert 6 * CONFIG["smoke_steps"] == 192


@pytest.mark.parametrize("reason", ["memory", "utilization", "pid", "reservation", "cpu"])
def test_occupied_gpu_is_not_admitted(reason, monkeypatch):
    launcher = load_script("launch_lr_control")
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda _: set(range(96)))
    name = next(iter(CONFIG["jobs"]))
    row = {
        "gpu": 4,
        "uuid": "GPU-test",
        "used_mib": 0,
        "free_mib": 97887,
        "total_mib": 97887,
        "utilization": 0,
    }
    snapshot = {
        "gpus": {4: row},
        "containers": [],
        "processes": [],
        "available_kib": 900 * 1024**2,
        "utc": "test",
    }
    assert launcher.admission(CONFIG, name, snapshot, 4) is not None
    if reason == "memory":
        row["used_mib"] = 2000
    elif reason == "utilization":
        row["utilization"] = 10
    elif reason == "pid":
        snapshot["processes"] = [{"uuid": "GPU-test", "pid": 123}]
    else:
        snapshot["containers"] = [
            {
                "Name": "other-project",
                "HostConfig": {
                    "DeviceRequests": [{"DeviceIDs": ["GPU-test"]}]
                    if reason == "reservation"
                    else [],
                    "CpusetCpus": "0-3" if reason == "cpu" else "",
                },
            }
        ]
    assert launcher.admission(CONFIG, name, snapshot, 4) is None


def test_all_visible_container_is_not_equivalent_to_owning_all_gpus():
    launcher = load_script("launch_lr_control")
    container = {"HostConfig": {"DeviceRequests": [{"Count": -1, "DeviceIDs": None}]}}
    assert launcher.explicit_gpu_reservation(container, 4, "GPU-test") is False


def test_stage_order_requires_short_training_and_independent_reload_before_science():
    launcher = load_script("launch_lr_control")
    assert launcher.STAGES == (
        "cuda-contracts",
        "preflight",
        "smoke",
        "smoke-audit",
        "train",
        "audit",
        "cloud-audit",
    )


def test_container_command_is_isolated_and_preserves_tracking_secret(tmp_path):
    launcher = load_script("launch_lr_control")
    name = next(iter(CONFIG["jobs"]))
    token = tmp_path / "private-token"
    token.write_text("never-print-this-token")
    row = {"gpu": 4, "uuid": "GPU-test", "cpu_set": [0, 1, 2, 3]}
    container, command = launcher.command(CONFIG, tmp_path, name, "train", token, row)
    assert command[:3] == ["docker", "--context", "lm-memory"]
    assert command[command.index("--gpus") + 1] == "device=GPU-test"
    assert "--detach" in command and "--privileged" not in command
    assert "never-print-this-token" not in json.dumps(command)
    assert container.startswith(BATCH)
    secret_mount = f"type=bind,source={token.resolve()},target=/run/secrets/wandb,readonly"
    assert secret_mount in command


def test_update_observation_preserves_weights_and_rng(monkeypatch):
    import torch

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runner = load_script("run_lr_control")
    name = next(iter(CONFIG["jobs"]))
    native = runner.load_native(job_config(CONFIG, name))
    torch.manual_seed(617)
    model = torch.nn.Linear(4, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    inputs = torch.ones(3, 4)

    def train_step(candidate, opt):
        opt.zero_grad()
        output = candidate(torch.nn.functional.dropout(inputs, p=0.25, training=True))
        output.square().mean().backward()
        opt.step()

    class Replay:
        def replay(self):
            train_step(model, optimizer)

    observed = object.__new__(native.AlignmentStep)
    observed.model = model
    observed.observed_calls = 999
    observed.update_measurement = {}
    observed.index = torch.zeros(3, dtype=torch.long)
    observed.graph = Replay()
    observed.loss = torch.tensor(0.0)
    rng = torch.get_rng_state()
    before = [p.detach().clone() for p in model.parameters()]
    observed(torch.arange(3))
    observed_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    train_step(reference, reference_optimizer)
    torch.testing.assert_close(torch.get_rng_state(), observed_rng, rtol=0, atol=0)
    for candidate, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
    delta2 = sum(
        float((p.detach() - old).square().double().sum())
        for p, old in zip(model.parameters(), before, strict=True)
    )
    assert observed.update_measurement["parameter_update_l2"] == pytest.approx(delta2**0.5)
    assert observed.update_measurement["measured_update_step"] == 1000
