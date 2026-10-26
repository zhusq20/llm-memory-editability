"""Contracts for the parameter-near-matched ordinary stateless Q adapter."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import MergedBatchStream, ReproductionGPT
from llm_memory_editability.loop_query_adapter import QueryAdapterGPT, adapter_accounting
from llm_memory_editability.loop_query_control import control_accounting

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def pair(dropout=0.0):
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
    base = ReproductionGPT(config, 2)
    torch.manual_seed(123)
    candidate = QueryAdapterGPT(copy.deepcopy(config), 2, 6, 456)
    return base, candidate


def test_zero_init_preserves_backbone_logits_all_gradients_and_rng():
    base, adapter = pair(0.1)
    tokens = torch.tensor([[2, 4, 7, 9], [3, 5, 8, 10]])
    pos = torch.tensor([[1, 2], [2, 3]])
    labels = torch.tensor([[7, 9], [10, 1]])
    values = []
    for model in (base, adapter):
        torch.manual_seed(789)
        logits = model(tokens, pos)
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
        values.append(logits.detach())
    assert torch.equal(*values)
    for name, p in base.named_parameters():
        assert torch.equal(p, dict(adapter.named_parameters())[name])
        assert torch.equal(p.grad, dict(adapter.named_parameters())[name].grad)
    assert adapter.adapter_up.weight.grad.abs().sum() > 0
    assert adapter.adapter_down.weight.grad.count_nonzero() == 0
    torch.manual_seed(123)
    ReproductionGPT(base.config, 2)
    reference_rng = torch.get_rng_state()
    torch.manual_seed(123)
    QueryAdapterGPT(base.config, 2, 6, 456)
    assert torch.equal(reference_rng, torch.get_rng_state())


def test_active_adapter_causal_in_forward_and_backward():
    _, adapter = pair()
    adapter.eval().double()
    with torch.no_grad():
        adapter.adapter_up.weight.normal_(std=0.2)
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 8, 9]])
    torch.testing.assert_close(adapter(a)[:, :3], adapter(b)[:, :3], atol=0, rtol=0)
    adapter(a)[:, 2, 6].sum().backward()
    assert adapter.transformer.wte.weight.grad[4:6].count_nonzero() == 0
    assert adapter.transformer.wpe.weight.grad[3:].count_nonzero() == 0


def test_adapter_recomputed_from_current_ln_hidden_every_block_and_only_changes_q(monkeypatch):
    base, model = pair()
    model.eval()
    with torch.no_grad():
        model.adapter_up.weight.normal_(std=0.1)
    ln_outputs, adapter_inputs, projections, deltas, actual = [], [], [], [], []
    for block in model.transformer.h:
        block.ln_1.register_forward_hook(lambda _m, _i, o: ln_outputs.append(o.detach().clone()))
        block.attn.c_attn.register_forward_hook(
            lambda _m, _i, o: projections.append(tuple(t.detach().clone() for t in o.chunk(3, -1)))
        )
    model.adapter_down.register_forward_pre_hook(
        lambda _m, i: adapter_inputs.append(i[0].detach().clone())
    )
    model.adapter_up.register_forward_hook(lambda _m, _i, o: deltas.append(o.detach().clone()))
    original = F.scaled_dot_product_attention

    def attention(q, k, v, **kwargs):
        actual.append(tuple(t.transpose(1, 2).flatten(2).detach().clone() for t in (q, k, v)))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", attention)
    model(torch.tensor([[1, 2, 3, 4]]))
    assert len(deltas) == len(adapter_inputs) == len(ln_outputs) == 4
    for z, a, delta, (q, k, v), (q0, k0, v0) in zip(
        ln_outputs, adapter_inputs, deltas, actual, projections, strict=True
    ):
        assert torch.equal(z, a)
        torch.testing.assert_close(q, q0 + delta, atol=0, rtol=0)
        assert torch.equal(k, k0) and torch.equal(v, v0)
    assert not torch.equal(adapter_inputs[0], adapter_inputs[1])
    assert len([n for n, _ in model.named_parameters() if n.startswith("adapter_")]) == 2
    assert not any("control_" in n for n, _ in model.named_parameters())
    assert set(dict(model.named_buffers())) == set(dict(base.named_buffers()))
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, dict(base.named_buffers())[name])


def test_both_matrices_learn_no_cross_sample_state_and_strict_reload():
    _, model = pair()
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[2, 3, 4], [5, 6, 7]])
    for _ in range(2):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    for p in (model.adapter_down.weight, model.adapter_up.weight):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    before = model(x[:1])
    model(torch.tensor([[8, 9, 10]]))
    assert torch.equal(before, model(x[:1]))
    torch.testing.assert_close(before, model(x)[:1], atol=1e-7, rtol=1e-6)
    fresh = copy.deepcopy(model)
    fresh.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(model(x), fresh(x))


def test_parameter_and_compute_matching_counts_all_block_executions():
    base, model = pair()
    small = adapter_accounting({"unique_layers": 2, "repeats": 2, "batch_size": 512}, 16, 6)
    assert (
        small["additional_parameters"]
        == 192
        == sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
    )
    spec = {"unique_layers": 4, "repeats": 2, "batch_size": 512}
    adapter = adapter_accounting(spec, 768, 48)
    control = control_accounting(spec, 768, 32)
    assert adapter["additional_parameters"] == 73728
    assert control["additional_parameters"] == 74752
    assert control["additional_parameters"] - adapter["additional_parameters"] == 1024
    assert adapter["additional_training_matmul_flops_per_step"] == 7247757312
    assert control["additional_training_matmul_flops_per_step"] == 1220542464
    assert adapter["independent_control_state_elements"] == 0


def test_config_budget_recipe_and_sample_stream_match_parent():
    config = json.loads((ROOT / "configs/loop-query-adapter-development-v1.json").read_text())
    parent = json.loads((ROOT / "configs/loop-query-control-development-v1.json").read_text())
    runner = module("run_loop_query_adapter")
    runner.validate_config(config)
    assert all(value == parent["spec"][k] for k, value in config["spec"].items() if k != "name")
    for key in ("dropout_seed", "evaluation_nodes", "checkpoint_nodes", "evaluation_batch_size"):
        assert config[key] == parent[key]
    a = MergedBatchStream(313600, config["spec"]["stream_seed"])
    b = MergedBatchStream(313600, parent["spec"]["stream_seed"])
    for _ in range(1230):
        ai, an = a.batch(512)
        bi, bn = b.batch(512)
        assert an == bn and (ai == bi).all()
    for key, value in [
        ("discoloop", True),
        ("first_round_kv", True),
        ("coefficient", 0.3),
        ("adapter_dim", 32),
    ]:
        changed = copy.deepcopy(config)
        changed["arms"]["query_adapter"][key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed)


def fake_admission(monkeypatch, foreign_pid=False, peer_name=None):
    launcher = module("launch_loop_query_adapter")
    config = json.loads((ROOT / "configs/loop-query-adapter-development-v1.json").read_text())
    peer_name = peer_name or config["runtime"]["shared_gpu_peer_containers"][0]
    peer = {
        "Name": "/" + peer_name,
        "HostConfig": {
            "CpusetCpus": "80-83",
            "DeviceRequests": [{"Count": 0, "DeviceIDs": ["GPU-test"]}],
        },
    }

    def docker(*args):
        if args[0] == "ps":
            return SimpleNamespace(stdout="peer-id\n")
        if args[0] == "inspect":
            return SimpleNamespace(stdout=json.dumps([peer]))
        if args[0] == "top":
            return SimpleNamespace(stdout="PID\n111\n")
        raise AssertionError(args)

    def command(args, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in args:
            return SimpleNamespace(stdout="GPU-test, " + ("222" if foreign_pid else "111") + "\n")
        return SimpleNamespace(stdout="4, GPU-test, 4500, 93000, 100\n")

    monkeypatch.setattr(launcher, "docker", docker)
    monkeypatch.setattr(launcher.subprocess, "run", command)
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda _: set(range(96)))
    return launcher, config


def test_shared_gpu_admits_only_registered_peer_and_disjoint_cpus(monkeypatch):
    launcher, config = fake_admission(monkeypatch)
    admitted = launcher.resources(config)
    assert admitted["shared_with"] == [config["runtime"]["shared_gpu_peer_containers"][0]]
    assert admitted["cpus"] == [84, 85, 86, 87]
    assert admitted["peer_gpu_pids"] == [111]


def test_shared_gpu_rejects_foreign_gpu_process(monkeypatch):
    launcher, config = fake_admission(monkeypatch, foreign_pid=True)
    assert launcher.resources(config) is None


def test_shared_gpu_rejects_unregistered_gpu_owner(monkeypatch):
    launcher, config = fake_admission(monkeypatch, peer_name="other-project")
    assert launcher.resources(config) is None


def test_container_uses_frozen_science_and_only_own_output(tmp_path):
    launcher = module("launch_loop_query_adapter")
    config = json.loads((ROOT / "configs/loop-query-adapter-development-v1.json").read_text())
    launcher.validate(config)
    _, args = launcher.command(config, tmp_path, "train", {"uuid": "GPU-test"})
    assert args[:4] == ["docker", "--context", "lm-memory", "create"]
    assert args[args.index("--cpuset-cpus") + 1] == "84,85,86,87"
    assert args[args.index("--gpus") + 1] == "device=GPU-test"
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "--mount"]
    assert len(mounts) == 3
    assert any("target=/frozen,readonly" in mount for mount in mounts)
    assert any("target=/inputs,readonly" in mount for mount in mounts)
    assert any("target=/output" in mount and "readonly" not in mount for mount in mounts)
    assert args[-5:] == [
        "train",
        "--config",
        "/frozen/configs/loop-query-adapter-development-v1.json",
        "--out",
        "/output",
    ]
