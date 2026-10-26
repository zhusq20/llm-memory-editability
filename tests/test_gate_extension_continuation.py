"""Check u95 continuation identity and uninterrupted optimizer/RNG/data behavior."""

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.paper_update_gate import PaperUpdateGateGPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_gate_extension_continuation")
SPEC = {"lr": 1e-4, "weight_decay": 0.1}


def config():
    return {
        "batch": "gate-generalization-extension-v1-continuation",
        "continuation": {
            "start_step": 300000,
            "new_warmup": False,
            "preserve_optimizer": True,
            "preserve_stream_and_rng": True,
        },
        "training": {**SPEC, "steps": 1500000, "warmup": 2000, "batch_size": 512},
        "evaluation_nodes": list(range(300000, 1500001, 100000)),
        "budget": {"new_scientific_updates": 2400000},
        "runtime": {"image": runner.IMAGE},
        "arms": {
            name: {
                "architecture": "discoloop",
                "unique_layers": 1,
                "repeats": 4,
                "update_mode": mode,
                "initial_update": 0.95,
                "parent": name,
                "update_gate_lr_factor": 1.0,
            }
            for name, mode in zip(runner.ARMS, ("fixed", "token"), strict=True)
        },
        "parents": {
            name: {
                "source_image": runner.IMAGE,
                "model_sha256": "a" * 64,
                "files": dict.fromkeys(runner.PARENT_FILES, "b" * 64),
            }
            for name in runner.ARMS
        },
    }


def test_frozen_actual_continuation_config_matches_the_authorized_runner():
    selected = json.loads((ROOT / "configs/gate-extension/continuation.json").read_text())
    runner.validate_config(selected)
    assert selected["input_root"] == "/inputs/continuation-world"
    assert all(
        parent["root"] == f"/inputs/parents/{name}" for name, parent in selected["parents"].items()
    )


def tiny_model(mode="token", dropout=0):
    torch.manual_seed(491)
    hf = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
    )
    return PaperUpdateGateGPT(
        hf, repeats=4, update_mode=mode, initial_update=0.95, discoloop=True
    ).double()


def assign_gradients(model, update):
    for index, parameter in enumerate(model.parameters()):
        values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
        parameter.grad = (values.remainder(11) + index + update + 1) / 100


@pytest.mark.parametrize("mode", ("fixed", "token"))
def test_restore_retains_group_order_adam_options_and_multiple_exact_updates(mode):
    parent = tiny_model(mode)
    original = runner.adapter.optimizer_for(parent, torch.tensor(SPEC["lr"]), 0.1)
    for update in range(3):
        assign_gradients(parent, update)
        original.step()
    candidate = copy.deepcopy(parent)
    restored, lr, gate_lr = runner.restore_optimizer(
        candidate, copy.deepcopy(original.state_dict()), SPEC, 1.0, torch.device("cpu")
    )
    assert len(restored.param_groups) == 2
    assert runner.inherited.parameter_groups(parent, original) == runner.inherited.parameter_groups(
        candidate, restored
    )
    assert all(group["lr"] is lr for group in restored.param_groups)
    assert gate_lr is (lr if mode == "token" else None)
    for before, after in zip(original.param_groups, restored.param_groups, strict=True):
        for key in before.keys() - {"params", "lr"}:
            assert after[key] == before[key], key
    for update in range(3, 8):
        assign_gradients(parent, update)
        assign_gradients(candidate, update)
        original.step()
        restored.step()
        assert runner.inherited.model_digest(candidate) == runner.inherited.model_digest(parent)
        assert runner.inherited.optimizer_digest(candidate, restored) == (
            runner.inherited.optimizer_digest(parent, original)
        )


@pytest.mark.parametrize("mode", ("fixed", "token"))
def test_serialized_resume_matches_dropout_gradients_samples_exposures_and_rng(tmp_path, mode):
    torch.set_num_threads(2)
    model = tiny_model(mode, dropout=0.2)
    optimizer = runner.adapter.optimizer_for(model, torch.tensor(SPEC["lr"]), 0.1)
    stream = runner.inherited.MergedBatchStream(10, 184)
    counts = np.zeros(10, dtype=np.int64)
    table = torch.arange(66).reshape(11, 6).remainder(23)
    positions = torch.tensor([[3, 4]] * 4)

    def step(current, opt, sampler, exposure):
        indices, _ = sampler.batch(4)
        exposure += np.bincount(indices, minlength=11)[:10]
        x = table[indices]
        opt.zero_grad(set_to_none=True)
        logits = current(x, positions)
        loss = F.cross_entropy(logits.flatten(0, 1), x[:, 4:6].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(current.parameters(), 1.0, foreach=True)
        opt.step()
        return float(loss.detach()), indices.copy()

    for _ in range(2):
        step(model, optimizer, stream, counts)
    path = tmp_path / "state.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "cpu_rng": torch.get_rng_state(),
        },
        path,
    )
    expected = [step(model, optimizer, stream, counts) for _ in range(3)]
    expected_rng = torch.get_rng_state()
    source = torch.load(path, weights_only=False)
    candidate = tiny_model(mode, dropout=0.2)
    candidate.load_state_dict(source["model"])
    restored, _, _ = runner.restore_optimizer(
        candidate, source["optimizer"], SPEC, 1.0, torch.device("cpu")
    )
    resumed = runner.inherited.MergedBatchStream(10, 0)
    resumed.load_state_dict(source["stream"])
    resumed_counts = source["counts"].copy()
    torch.set_rng_state(source["cpu_rng"])
    actual = [step(candidate, restored, resumed, resumed_counts) for _ in range(3)]
    for (a_loss, a_indices), (e_loss, e_indices) in zip(actual, expected, strict=True):
        assert a_loss == e_loss
        np.testing.assert_array_equal(a_indices, e_indices)
    np.testing.assert_array_equal(resumed_counts, counts)
    assert runner.inherited.model_digest(candidate) == runner.inherited.model_digest(model)
    assert runner.inherited.optimizer_digest(candidate, restored) == (
        runner.inherited.optimizer_digest(model, optimizer)
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert resumed.state_dict()["rng"] == stream.state_dict()["rng"]
    np.testing.assert_array_equal(resumed.remaining, stream.remaining)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("continuation", "start_step", 0),
        ("continuation", "new_warmup", True),
        ("continuation", "preserve_optimizer", False),
        ("continuation", "preserve_stream_and_rng", False),
        ("training", "steps", 300000),
        ("training", "lr", 5e-5),
        ("training", "batch_size", 256),
        ("budget", "new_scientific_updates", 4800000),
        ("runtime", "image", "sha256:other"),
    ],
)
def test_scope_rejects_changed_training_and_restore_contract(section, key, value):
    selected = config()
    runner.validate_config(selected)
    selected[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_config(selected)


@pytest.mark.parametrize(
    ("key", "value"),
    [("initial_update", 0.9), ("update_gate_lr_factor", 0.5), ("parent", "fixed-update")],
)
def test_scope_rejects_u90_lr_interventions_or_wrong_parent(key, value):
    selected = config()
    selected["arms"]["token-update"][key] = value
    with pytest.raises(ValueError, match="Changed u95"):
        runner.validate_config(selected)


def test_scope_requires_both_parents_all_hashes_and_absolute_nodes():
    for change in ("parent", "hash", "node", "batch"):
        selected = config()
        if change == "parent":
            del selected["parents"]["fixed-update"]
        elif change == "hash":
            del selected["parents"]["fixed-update"]["files"]["latest.pt"]
        elif change == "node":
            selected["evaluation_nodes"].remove(400000)
        else:
            selected["batch"] = "paper-update-gate-continuation-v1"
        with pytest.raises(ValueError):
            runner.validate_config(selected)


@pytest.mark.parametrize("change", ("rate", "decay", "group", "factor"))
def test_optimizer_rejects_modified_parent_recipe(change):
    model = tiny_model()
    saved = runner.adapter.optimizer_for(model, torch.tensor(SPEC["lr"]), 0.1).state_dict()
    factor = 1.0
    if change == "rate":
        saved["param_groups"][1]["lr"] = torch.tensor(5e-5)
    elif change == "decay":
        saved["param_groups"][1]["weight_decay"] = 0.1
    elif change == "group":
        saved["param_groups"].append(copy.deepcopy(saved["param_groups"][1]))
    else:
        factor = 0.5
    with pytest.raises(ValueError):
        runner.restore_optimizer(model, saved, SPEC, factor, torch.device("cpu"))


def parent_fixture(tmp_path):
    selected = config()
    selected.update(model={"n_embd": 12}, data={"world_seed": 491}, dropout_seed=14)
    selected["input_origin"] = {"world_sha256": "c" * 64, "panels_sha256": "d" * 64}
    name = "token-update"
    parent = selected["parents"][name]
    parent["root"] = str(tmp_path)
    original = copy.deepcopy(selected)
    original["batch"] = "paper-update-gate-multihop-u95-v1"
    original["training"]["steps"] = 300000
    original["arms"][name] = {
        k: v
        for k, v in selected["arms"][name].items()
        if k not in ("parent", "update_gate_lr_factor")
    }
    (tmp_path / "config.json").write_text(json.dumps(original))
    source = {
        "step": 300000,
        "world_sha256": selected["input_origin"]["world_sha256"],
        "config_sha256": runner.inherited.digest(tmp_path / "config.json"),
        "model_sha256": parent["model_sha256"],
        "spec": {**original["training"], **original["arms"][name], "name": name},
        **dict.fromkeys(("optimizer", "stream", "counts", "cpu_rng", "cuda_rng")),
        "training_seconds": 1.0,
    }
    torch.save(source, tmp_path / "latest.pt")
    (tmp_path / "complete.json").write_text(
        json.dumps({"step": 300000, "model_sha256": parent["model_sha256"], "arm": name})
    )
    for filename in ("run.json", "learning.json", "exposure.npz"):
        (tmp_path / filename).write_bytes(b"frozen fixture")
    parent["files"] = {key: runner.inherited.digest(tmp_path / key) for key in parent["files"]}
    return selected, source, original


@pytest.mark.parametrize("change", ("hash", "model", "arm", "image", "missing_state"))
def test_parent_checks_reject_corrupted_or_scientifically_different_parent(tmp_path, change):
    selected, source, original = parent_fixture(tmp_path)
    name = "token-update"
    runner.load_parent(selected, name, selected["input_origin"])
    if change == "hash":
        (tmp_path / "exposure.npz").write_bytes(b"modified")
    elif change == "model":
        selected["model"]["n_embd"] = 24
    else:
        if change == "arm":
            source["spec"]["initial_update"] = 0.9
        elif change == "missing_state":
            del source["cpu_rng"]
        else:
            original["runtime"]["image"] = "sha256:other"
            (tmp_path / "config.json").write_text(json.dumps(original))
            source["config_sha256"] = runner.inherited.digest(tmp_path / "config.json")
        torch.save(source, tmp_path / "latest.pt")
        for filename in ("latest.pt", "config.json"):
            selected["parents"][name]["files"][filename] = runner.inherited.digest(
                tmp_path / filename
            )
    with pytest.raises(ValueError):
        runner.load_parent(selected, name, selected["input_origin"])


def test_scoped_adapters_restore_every_inherited_module_on_exception():
    before = (
        runner.inherited.restore_optimizer,
        runner.inherited.load_parent,
        runner.native.train,
        runner.native.write_json,
        runner.native.initialize,
    )
    with pytest.raises(RuntimeError), runner.installed():
        assert runner.inherited.restore_optimizer is runner.restore_optimizer
        assert runner.inherited.load_parent is runner.load_parent
        raise RuntimeError("interrupted")
    assert before == (
        runner.inherited.restore_optimizer,
        runner.inherited.load_parent,
        runner.native.train,
        runner.native.write_json,
        runner.native.initialize,
    )
