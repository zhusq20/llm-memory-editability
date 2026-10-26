"""Identity of original Wang facts under compact encoding and current architecture."""

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_compact_wang as launcher  # noqa: E402
import run_compact_wang as runner  # noqa: E402

from llm_memory_editability import compact_wang_data as compact  # noqa: E402
from llm_memory_editability import gate_extension_data as original  # noqa: E402
from llm_memory_editability.joint_update_gate import JointUpdateGateGPT  # noqa: E402


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/compact-wang-joint-gate-v1.json").read_text())


def test_registered_model_and_scope(config):
    runner.validate_config(config)
    assert config["runtime"]["gpu"] == 2
    assert config["budget"]["new_scientific_updates"] == 1500000
    assert config["training"]["stream_seed"] == 792101


@pytest.mark.parametrize("change", ["depth", "repeats", "grid", "budget", "gate", "width"])
def test_cannot_accidentally_run_old_model(config, change):
    if change in {"depth", "repeats", "gate"}:
        key, value = {
            "depth": ("unique_layers", 4),
            "repeats": ("repeats", 2),
            "gate": ("update_mode", "fixed"),
        }[change]
        config["arms"]["joint-token"][key] = value
    elif change == "grid":
        config["inference_repeats"] = [1, 2]
    elif change == "budget":
        config["training"]["steps"] = 300000
    else:
        config["model"]["n_embd"] = 512
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_every_used_symbol_is_bijective():
    old = np.array([-100, 50256, 52460, *range(50257, 52457)])
    new = compact.remap(old)
    assert len(np.unique(new)) == len(new)
    assert np.array_equal(compact.remap(new, inverse=True), old)
    assert set(new) == {-100, 0, 2, *range(3, 2203)}


@pytest.mark.parametrize("token,inverse", [(1, False), (52457, False), (1, True), (2203, True)])
def test_unused_tokens_fail_visibly(token, inverse):
    with pytest.raises(ValueError):
        compact.remap(np.array([token]), inverse=inverse)


def test_answer_and_eos_positions_remain_distinct():
    meta = {**compact.VOCABULARY, "sequence_length": 4}
    rows = np.array([[1999, 199, 0]])
    x, pos, labels = original.pack_rows(rows, meta)
    assert x.tolist() == [[2002, 2202, 3, 0]]
    assert pos.tolist() == [[1, 2]]
    assert labels.tolist() == [[3, 2]]


def test_actual_model_size_and_terminal_readout(config, monkeypatch):
    meta = {**compact.VOCABULARY, "sequence_length": 4}
    monkeypatch.setattr(runner.joint.original, "load", lambda _: ({}, {"test_oo": []}, meta))
    with runner.joint.installed(config):
        model, spec, _, _ = runner.joint.native.initialize(config, "joint-token", meta, "cpu")
        assert isinstance(model, JointUpdateGateGPT)
        assert len(model.transformer.h) == 1 and spec["repeats"] == 4
        assert sum(p.numel() for p in model.parameters()) == 9569282
        assert model.transformer.wte.weight.shape == (2203, 768)
    small = copy.deepcopy(config)
    small["model"].update(n_embd=12, n_head=3, n_positions=8, n_ctx=8)
    with runner.joint.installed(small):
        model, _, _, _ = runner.joint.native.initialize(small, "joint-token", meta, "cpu")
        model.eval()
        tokens = torch.tensor([[3, 2003, 2004, 4]])
        logits, _, h = model.forward_with_states(tokens)
        assert torch.equal(logits, model.transformer.ln_f(h) @ model.transformer.wte.weight.T)
        changed = tokens.clone()
        changed[:, -1] = 6
        other, _, _ = model.forward_with_states(changed)
        assert torch.equal(logits[:, :-1], other[:, :-1])


def test_gpu2_command_uses_compact_runner_and_persistent_output(config, tmp_path):
    (tmp_path / "freeze.json").write_text("{}")
    controller = launcher.Controller(
        config,
        tmp_path,
        tmp_path,
        launcher.RELATIVE,
        {2: config["runtime"]["gpu_uuid"]},
        tmp_path / "unused-token",
    )
    controller.output = tmp_path / "persistent"
    _, command, cpus = controller.command("joint-token", "train", 2, config["runtime"]["gpu_uuid"])
    assert "scripts/run_compact_wang.py" in command
    assert command[:5] == ["docker", "--context", "lm-memory", "create", "--label"]
    assert command[command.index("--gpus") + 1] == f"device={config['runtime']['gpu_uuid']}"
    assert f"type=bind,source={tmp_path / 'data'},target=/inputs,readonly" in command
    assert cpus == [24, 25]


def test_full_archived_task_mapping(tmp_path):
    source = os.environ.get("COMPACT_WANG_SOURCE")
    if not source:
        pytest.skip("Full original Wang archive supplied by the frozen-container check")
    receipt = compact.prepare(source, tmp_path / "compact")
    assert receipt["passed"]
    world, panels, meta = original.load(tmp_path / "compact")
    assert original.audit_data(world, panels, meta)["passed"]
    assert len(world["atoms"]) == 40000
    assert len(world["train_2"]) == 273600
    assert sum(len(rows) for rows in panels.values()) == 19004
    assert original.path_details(panels["test_oo"], meta, world)[1].shape == (2004, 2)
