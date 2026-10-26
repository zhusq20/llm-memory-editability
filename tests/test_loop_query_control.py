"""Contracts that distinguish a causal query controller from extra supervision."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import ReproductionGPT
from llm_memory_editability.loop_query_control import QueryControlGPT, control_accounting

ROOT = Path(__file__).resolve().parents[1]


def pair(dropout=0.0, repeats=3):
    config = GPT2Config(
        vocab_size=31,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=12,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
    )
    torch.manual_seed(123)
    base = ReproductionGPT(config, repeats)
    torch.manual_seed(123)
    candidate = QueryControlGPT(copy.deepcopy(config), repeats, 4, 456)
    return base, candidate


def test_zero_initialization_matches_logits_and_all_backbone_gradients_with_dropout():
    base, candidate = pair(0.1)
    tokens = torch.tensor([[2, 4, 7, 9], [3, 5, 8, 10]])
    pos = torch.tensor([[1, 2], [2, 3]])
    labels = torch.tensor([[7, 9], [10, 1]])
    values = []
    for model in (base, candidate):
        torch.manual_seed(789)
        logits = model(tokens, pos)
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
        values.append(logits.detach())
    assert torch.equal(*values)
    for name, parameter in base.named_parameters():
        assert torch.equal(parameter, dict(candidate.named_parameters())[name])
        assert torch.equal(parameter.grad, dict(candidate.named_parameters())[name].grad)
    assert candidate.control_query.weight.grad.abs().sum() > 0
    for name in ("control_input", "control_content", "control_recurrent"):
        assert getattr(candidate, name).weight.grad.count_nonzero() == 0


def test_active_controller_is_causal_in_logits_and_gradients():
    _, model = pair()
    model.eval().double()
    with torch.no_grad():
        model.control_query.weight.normal_(std=0.2)
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 8, 9]])
    torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3], atol=0, rtol=0)
    model(a)[:, 2, 6].sum().backward()
    assert model.transformer.wte.weight.grad[4:6].count_nonzero() == 0
    assert model.transformer.wpe.weight.grad[3:].count_nonzero() == 0


def test_control_enters_only_query_and_updates_only_at_boundaries(monkeypatch):
    _, model = pair(repeats=3)
    model.eval()
    with torch.no_grad():
        model.control_query.weight.normal_(std=0.1)
    actual, expected, projections, updates = [], [], [], []
    for block in model.transformer.h:
        block.attn.c_attn.register_forward_hook(
            lambda _m, _i, output: projections.append(
                tuple(t.detach().clone() for t in output.chunk(3, -1))
            )
        )
    model.control_query.register_forward_hook(
        lambda _m, _i, output: expected.append(output.detach().clone())
    )
    original_attention = F.scaled_dot_product_attention

    def attention(q, k, v, **kwargs):
        actual.append(tuple(t.transpose(1, 2).flatten(2).detach().clone() for t in (q, k, v)))
        return original_attention(q, k, v, **kwargs)

    original_update = model.update_control

    def update(*args):
        updates.append(True)
        return original_update(*args)

    monkeypatch.setattr(F, "scaled_dot_product_attention", attention)
    monkeypatch.setattr(model, "update_control", update)
    model(torch.tensor([[1, 2, 3, 4]]))
    assert len(actual) == 6 and len(expected) == 3 and len(updates) == 2
    for index, ((q, k, v), (q0, k0, v0)) in enumerate(zip(actual, projections, strict=True)):
        torch.testing.assert_close(q, q0 + expected[index // 2], atol=0, rtol=0)
        assert torch.equal(k, k0) and torch.equal(v, v0)


def test_all_controller_parameters_learn_and_state_resets_per_forward():
    _, model = pair(repeats=2)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tokens = torch.tensor([[2, 3, 4], [5, 6, 7]])
    for _ in range(2):
        optimizer.zero_grad()
        model(tokens).square().mean().backward()
        optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith("control_"):
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    alone = model(tokens[:1])
    model(torch.tensor([[10, 11, 12]]))
    torch.testing.assert_close(alone, model(tokens[:1]), atol=0, rtol=0)
    torch.testing.assert_close(alone, model(tokens)[:1], atol=1e-7, rtol=1e-6)
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(model(tokens), restored(tokens))


def test_parameter_and_compute_accounting_and_controller_rng():
    base, candidate = pair(repeats=2)
    accounting = control_accounting({"repeats": 2, "batch_size": 512}, 16, 4)
    assert (
        accounting["additional_parameters"]
        == (
            sum(p.numel() for p in candidate.parameters())
            - sum(p.numel() for p in base.parameters())
        )
        == 208
    )
    assert accounting["additional_training_matmul_flops_per_step"] == 3 * 512 * 4 * 2 * 272
    config = base.config
    torch.manual_seed(123)
    ReproductionGPT(config, 2)
    reference_rng = torch.get_rng_state()
    torch.manual_seed(123)
    QueryControlGPT(config, 2, 4, 456)
    assert torch.equal(reference_rng, torch.get_rng_state())


def test_frozen_budget_and_no_discoloop_or_auxiliary_loss():
    spec = importlib.util.spec_from_file_location(
        "query_runner", ROOT / "scripts/run_loop_query_control.py"
    )
    runner = importlib.util.module_from_spec(spec)
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(runner)
    config = json.loads((ROOT / "configs/loop-query-control-development-v1.json").read_text())
    runner.validate_config(config)
    for key, value in (
        ("discoloop", True),
        ("first_round_kv", True),
        ("coefficient", 0.3),
        ("control_dim", 64),
    ):
        changed = copy.deepcopy(config)
        changed["arms"]["query_control"][key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed)
    config["spec"]["steps"] = 300001
    with pytest.raises(ValueError):
        runner.validate_config(config)


def launcher():
    spec = importlib.util.spec_from_file_location(
        "query_launcher", ROOT / "scripts/launch_loop_query_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_is_explicit_lm_and_freezes_inputs_and_output_mounts(tmp_path):
    module = launcher()
    config = json.loads((ROOT / "configs/loop-query-control-development-v1.json").read_text())
    token = tmp_path / "credential-reference"
    _, command = module.command(config, tmp_path / "batch", "train", token, {"uuid": "GPU-free"})
    assert command[:3] == ["docker", "--context", "lm-memory"]
    assert command[command.index("--gpus") + 1] == "device=GPU-free"
    assert command[command.index("--cpuset-cpus") + 1] == "80,81,82,83"
    mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
    assert any("target=/frozen,readonly" in value for value in mounts)
    assert any("target=/inputs,readonly" in value for value in mounts)
    assert any("target=/output" in value and "readonly" not in value for value in mounts)
    assert command[-5:] == [
        "train",
        "--config",
        "/frozen/configs/loop-query-control-development-v1.json",
        "--out",
        "/output",
    ]


@pytest.mark.parametrize(
    "used,utilization,process", [(2000, 0, ""), (0, 90, ""), (0, 0, "GPU-owned, 123\n")]
)
def test_resource_admission_refuses_other_gpu_use(monkeypatch, used, utilization, process):
    from types import SimpleNamespace

    module = launcher()
    config = json.loads((ROOT / "configs/loop-query-control-development-v1.json").read_text())
    monkeypatch.setattr(module, "docker", lambda *_: SimpleNamespace(stdout=""))

    def command(args, **_):
        if "--query-compute-apps=gpu_uuid,pid" in args:
            return SimpleNamespace(stdout=process)
        return SimpleNamespace(stdout=f"4, GPU-owned, {used}, {utilization}\n")

    monkeypatch.setattr(module.subprocess, "run", command)
    assert module.resources(config) is None
