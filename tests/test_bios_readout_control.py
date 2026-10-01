"""H5 frozen tied readout, original E93 stream, architecture and resume contracts."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_cross import edit_pair, make_cross_world
from llm_memory_editability.bios_cross_continue import tree_hash
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_readout_control import (
    STUDY,
    capture_state,
    edit_update,
    restore_state,
    sampling_stream,
    select_control_parameters,
    validate_parent_config,
    validate_study,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tied_readout_and_only_token_weight_excluded(monkeypatch):
    model = CausalLM(ModelConfig(vocab_size=32, width=8, layers=8, heads=1))
    selected = select_control_parameters(model)
    assert len(selected) == len(list(model.parameters())) - 1
    assert {name for name, p in model.named_parameters() if not p.requires_grad} == {"token.weight"}
    assert all(p.requires_grad for p in selected)
    assert all(id(p) != id(model.token.weight) for p in selected)
    linear = torch.nn.functional.linear
    weights = []

    def spy(inputs, weight, bias=None):
        weights.append(weight)
        return linear(inputs, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", spy)
    result = model(torch.ones((2, 6), dtype=torch.int64))
    assert result.shape == (2, 6, 32)
    assert weights[-1] is model.token.weight
    assert weights[-1].data_ptr() == model.token.weight.data_ptr()
    assert model.position.weight.requires_grad


@pytest.mark.parametrize("world_seed", [0, 1])
def test_both_worlds_and_both_widths_match_real_parent_architecture_and_samples(world_seed):
    world = make_cross_world(world_seed)
    for width in (256, 768):
        base = ROOT / (
            "results/bios-cross-dev-v1"
            if width == 768
            else "results/bios-cross-scale-dev-v1/width-256"
        )
        parent = base / f"world-{world_seed}-seed-0-company"
        config_path = parent / "config.json"
        if not config_path.is_file():
            pytest.skip("Historical parent checks require local, untracked experiment results")
        config = json.loads(config_path.read_text())
        if config["torch"] == torch.__version__ and config["numpy"] == np.__version__:
            validate_parent_config(config, world)
        else:
            with pytest.raises(ValueError, match="numerical software differs from parent"):
                validate_parent_config(config, world)
        assert config["model"]["vocab_size"] == world.vocab_size
        invalid = copy.deepcopy(config)
        invalid["model"]["vocab_size"] += 1
        with pytest.raises(ValueError, match="architecture"):
            validate_parent_config(invalid, world)
        for chain, name in enumerate(("company", "project")):
            pair = edit_pair(world, chain)
            e, r = sampling_stream(world_seed, chain)
            assert (len(pair["E"]), len(pair["replay"])) == (93, 4096)
            for scope in ("mlp", "all"):
                with np.load(parent / "edits" / f"{name}-exception-{scope}" / "sets.npz") as actual:
                    np.testing.assert_array_equal(actual["edit_sampling"], e)
                    np.testing.assert_array_equal(actual["replay_sampling"], r)
                    for key in ("E", "D", "exception", "replay"):
                        np.testing.assert_array_equal(actual[key], pair[key])


def test_short_resume_exact_with_frozen_token_and_trainable_attention_position():
    torch.set_num_threads(1)
    device = torch.device("cpu")

    def setup():
        torch.manual_seed(224)
        model = CausalLM(ModelConfig(vocab_size=32, width=8, layers=8, heads=1))
        selected = select_control_parameters(model)
        optimizer = torch.optim.AdamW(selected, lr=3e-5, weight_decay=0.1)
        tokens = torch.randint(0, 32, (8, 6))
        data = dict(
            tokens=tokens, positions=torch.tensor([[3, 4]]).repeat(8, 1), labels=tokens[:, 4:6]
        )
        new = copy.deepcopy(data)
        new["tokens"][:4, 4:6] = (new["tokens"][:4, 4:6] + 1) % 32
        new["labels"] = new["tokens"][:, 4:6]
        with torch.no_grad():
            refs = model(data["tokens"][4:], data["positions"][4:]).detach()
        return model, selected, optimizer, data, new, refs

    e, r = sampling_stream(0, 0, steps=7, batch=4)
    e, r = e % 4, r % 4

    def step(parts, index):
        model, selected, optimizer, data, new, refs = parts
        torch.rand(1)
        edit_update(
            model,
            optimizer,
            selected,
            data,
            new,
            refs,
            torch.as_tensor(e[index]),
            torch.as_tensor(r[index] + 4),
            torch.as_tensor(r[index]),
            device,
        )

    parts = setup()
    original = copy.deepcopy(parts[0].state_dict())
    for i in range(3):
        step(parts, i)
    saved = copy.deepcopy(capture_state(parts[0], parts[2], device, step=3))
    for i in range(3, 7):
        step(parts, i)
    expected = tree_hash(capture_state(parts[0], parts[2], device, step=7))
    resumed = setup()
    restore_state(resumed[0], resumed[2], saved, device)
    for i in range(3, 7):
        step(resumed, i)
    assert tree_hash(capture_state(resumed[0], resumed[2], device, step=7)) == expected
    assert torch.equal(parts[0].token.weight, original["token.weight"])
    for name in ("position.weight", "blocks.0.attention.qkv.weight", "blocks.7.mlp.down.weight"):
        assert not torch.equal(parts[0].state_dict()[name], original[name])


def test_full_24_parent_48_edit_matrix_without_creating_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "h5_runner", ROOT / "scripts/run_bios_readout_control.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = ROOT / "configs/bios-readout-control-v1.json"
    validate_study(json.loads(config.read_text()))
    jobs = runner.jobs(
        tmp_path / "small", tmp_path / "large", tmp_path / "output", config, "python"
    )
    assert len(jobs) == len({job["id"] for job in jobs}) == 24
    assert {(j["width"], j["world"], j["seed"], j["condition"]) for j in jobs} == {
        (w, d, s, o)
        for w in (256, 768)
        for d in (0, 1)
        for s in (0, 1)
        for o in ("company", "project", "neither")
    }
    assert STUDY["edit_cases"] == len(jobs) * 2 == 48
    assert not list(tmp_path.iterdir())
