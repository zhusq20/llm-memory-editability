#!/usr/bin/env python3
"""GPU engineering checks for the small-transformer composition experiment.

This uses arbitrary synthetic labels to verify execution only, not scientific results.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import (
    EpochStream,
    GraphStep,
    SmallGPT,
    make_optimizer,
    pack_rows,
    source_hash,
    utc,
    write_json,
)


def parameter_pairs(custom, hf):
    pairs = [
        (custom.token.weight, hf.transformer.wte.weight, False),
        (custom.position.weight, hf.transformer.wpe.weight, False),
        (custom.ln_final.weight, hf.transformer.ln_f.weight, False),
        (custom.ln_final.bias, hf.transformer.ln_f.bias, False),
    ]
    for block, target in zip(custom.blocks, hf.transformer.h, strict=True):
        for left, right, transposed in [
            (block.ln1, target.ln_1, False),
            (block.ln2, target.ln_2, False),
            (block.attention.qkv, target.attn.c_attn, True),
            (block.attention.proj, target.attn.c_proj, True),
            (block.mlp.up, target.mlp.c_fc, True),
            (block.mlp.down, target.mlp.c_proj, True),
        ]:
            pairs.extend([(left.weight, right.weight, transposed), (left.bias, right.bias, False)])
    return pairs


def hf_equivalence(device):
    from transformers import GPT2Config, GPT2LMHeadModel

    # Compare the mathematical implementation without TF32 attention rounding.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(41)
    cfg = ModelConfig(vocab_size=73, width=64, layers=2, heads=4, context=8)
    model = SmallGPT(cfg, dropout=0.1).to(device).eval()
    hf_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.context,
        n_ctx=cfg.context,
        n_embd=cfg.width,
        n_layer=cfg.layers,
        n_head=cfg.heads,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
        use_cache=False,
    )
    hf_config._attn_implementation = "eager"
    hf = GPT2LMHeadModel(hf_config).to(device).eval()
    pairs = parameter_pairs(model, hf)
    with torch.no_grad():
        for a, b, transpose in pairs:
            b.copy_(a.T if transpose else a)
    x = torch.randint(2, cfg.vocab_size, (16, 4), device=device)
    target = torch.randint(2, cfg.vocab_size, (16, 2), device=device)
    custom_logits, hf_logits = model(x), hf(x).logits
    logits_delta = (custom_logits - hf_logits).abs().max().item()
    F.cross_entropy(custom_logits[:, -2:].flatten(0, 1), target.flatten()).backward()
    F.cross_entropy(hf_logits[:, -2:].flatten(0, 1), target.flatten()).backward()
    grad_delta = max(
        ((a.grad.T if transpose else a.grad) - b.grad).abs().max().item()
        for a, b, transpose in pairs
    )
    torch.testing.assert_close(custom_logits, hf_logits, rtol=2e-5, atol=2e-6)
    for a, b, transpose in pairs:
        torch.testing.assert_close(a.grad.T if transpose else a.grad, b.grad, rtol=2e-4, atol=2e-6)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return {
        "passed": True,
        "tf32": False,
        "logits_max_abs_delta": logits_delta,
        "grad_max_abs_delta": grad_delta,
    }


def make_table(device, vocab=128):
    rng = np.random.default_rng(97)
    atomic = pack_rows(rng.integers(2, vocab, (127, 3)))
    composite = pack_rows(rng.integers(2, vocab, (129, 4)))
    return tuple(
        torch.as_tensor(np.concatenate([a, b]), device=device)
        for a, b in zip(atomic, composite, strict=True)
    )


def eager(model, optimizer, table, index):
    optimizer.zero_grad(set_to_none=False)
    x, pos, labels = (t[index] for t in table)
    loss = F.cross_entropy(model(x, pos).flatten(0, 1), labels.flatten())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    optimizer.step()
    return loss, norm


def parameter_delta(a, b):
    return max(
        (p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters(), strict=True)
    )


def optimizer_delta(a, b):
    return max(
        (av - bv).abs().max().item()
        for sa, sb in zip(a.state.values(), b.state.values(), strict=True)
        for av, bv in zip(sa.values(), sb.values(), strict=True)
    )


def graph_parity(device):
    torch.manual_seed(83)
    cfg = ModelConfig(vocab_size=128, width=64, layers=2, heads=4, context=8)
    a_model = SmallGPT(cfg, dropout=0.1).to(device)
    b_model = copy.deepcopy(a_model)
    a_lr = torch.tensor(1e-3, device=device)
    b_lr = a_lr.clone()
    a_opt, b_opt = make_optimizer(a_model, a_lr, 0.1), make_optimizer(b_model, b_lr, 0.1)
    table = make_table(device)
    stream = EpochStream(len(table[0]), 51)
    # Establish a nonempty optimizer state before capture.
    for _ in range(7):
        ix = torch.as_tensor(stream.take(128), device=device)
        state = torch.cuda.get_rng_state(device)
        eager(a_model, a_opt, table, ix)
        torch.cuda.set_rng_state(state, device)
        eager(b_model, b_opt, table, ix)
    before = {k: v.clone() for k, v in b_model.state_dict().items()}
    before_opt = copy.deepcopy(b_opt.state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    graph = GraphStep(b_model, b_opt, table, 128)
    capture_weights = all(torch.equal(v, b_model.state_dict()[k]) for k, v in before.items())
    capture_opt = all(
        torch.equal(v, b_opt.state_dict()["state"][i][k])
        for i, values in before_opt["state"].items()
        for k, v in values.items()
    )
    capture_rng = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(
        cuda_rng, torch.cuda.get_rng_state(device)
    )
    assert capture_weights and capture_opt and capture_rng
    loss_delta = norm_delta = 0.0
    rng_equal = True
    for i in range(100):
        ix = torch.as_tensor(stream.take(128), device=device)
        learning_rate = 1e-3 * (1 - i / 110)
        a_lr.fill_(learning_rate)
        b_lr.fill_(learning_rate)
        starting_rng = torch.cuda.get_rng_state(device)
        a_loss, a_norm = eager(a_model, a_opt, table, ix)
        ending_rng = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(starting_rng, device)
        b_loss = graph(ix)
        loss_delta = max(loss_delta, abs(a_loss.item() - b_loss.item()))
        norm_delta = max(norm_delta, abs(a_norm.item() - graph.grad_norm.item()))
        rng_equal &= torch.equal(ending_rng, torch.cuda.get_rng_state(device))
    weights_delta, states_delta = parameter_delta(a_model, b_model), optimizer_delta(a_opt, b_opt)
    b_lr.zero_()
    values = [graph(ix).item() for _ in range(5)]
    assert loss_delta < 1e-5 and norm_delta < 1e-5
    assert weights_delta < 1e-6 and states_delta < 1e-6 and rng_equal
    assert len(set(values)) == len(values)
    return {
        "passed": True,
        "steps": 100,
        "dropout": 0.1,
        "clip": 1.0,
        "capture_weights_unchanged": capture_weights,
        "capture_optimizer_unchanged": capture_opt,
        "capture_rng_unchanged": capture_rng,
        "per_step_rng_equal": rng_equal,
        "loss_max_abs_delta": loss_delta,
        "grad_norm_max_abs_delta": norm_delta,
        "parameter_max_abs_delta": weights_delta,
        "optimizer_max_abs_delta": states_delta,
        "fixed_weights_dropout_losses": values,
    }


def resume_equivalence(device):
    torch.manual_seed(59)
    cfg = ModelConfig(vocab_size=128, width=64, layers=2, heads=4, context=8)
    model = SmallGPT(cfg, dropout=0.1).to(device)
    lr = torch.tensor(1e-3, device=device)
    opt = make_optimizer(model, lr, 0.1)
    table = make_table(device)
    stream = EpochStream(len(table[0]), 13)
    graph = GraphStep(model, opt, table, 128)

    def advance(g, rate, s, start, count):
        losses = []
        for step in range(start, start + count):
            rate.fill_(1e-3 * (1 + (step % 7)) / 7)
            ix = torch.as_tensor(s.take(128), device=device)
            losses.append(g(ix).item())
        return losses

    advance(graph, lr, stream, 0, 30)
    with tempfile.TemporaryDirectory(prefix="grok-resume-") as tmp:
        path = Path(tmp) / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "stream": stream.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
            },
            path,
        )
        losses_a = advance(graph, lr, stream, 30, 50)
        final_rng_a = torch.cuda.get_rng_state(device)
        final_cpu_a = torch.get_rng_state()
        restored_model = SmallGPT(cfg, dropout=0.1).to(device)
        restored_opt = make_optimizer(restored_model, torch.tensor(1e-3, device=device), 0.1)
        state = torch.load(path, map_location=device, weights_only=False)
        restored_model.load_state_dict(state["model"])
        restored_opt.load_state_dict(state["optimizer"])
        restored_lr = restored_opt.param_groups[0]["lr"]
        for group in restored_opt.param_groups:
            group["lr"] = restored_lr
        restored_stream = EpochStream(len(table[0]), 999)
        restored_stream.load_state_dict(state["stream"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        restored_graph = GraphStep(restored_model, restored_opt, table, 128)
        losses_b = advance(restored_graph, restored_lr, restored_stream, 30, 50)
        weight_delta = parameter_delta(model, restored_model)
        state_delta = optimizer_delta(opt, restored_opt)
        rng_equal = torch.equal(final_rng_a, torch.cuda.get_rng_state(device)) and torch.equal(
            final_cpu_a, torch.get_rng_state()
        )
        stream_equal = np.array_equal(stream.take(777), restored_stream.take(777))
    assert losses_a == losses_b
    assert weight_delta == state_delta == 0.0 and rng_equal and stream_equal
    return {
        "passed": True,
        "before_checkpoint_steps": 30,
        "continued_steps": 50,
        "losses_identical": losses_a == losses_b,
        "parameter_max_abs_delta": weight_delta,
        "optimizer_max_abs_delta": state_delta,
        "rng_identical": rng_equal,
        "epoch_stream_identical": stream_equal,
    }


def benchmark(device, layers):
    cfg = ModelConfig(vocab_size=202, width=128, layers=layers, heads=4, context=8)
    model = SmallGPT(cfg, dropout=0.1).to(device)
    lr = torch.tensor(1e-3, device=device)
    opt = make_optimizer(model, lr, 0.1)
    table = make_table(device, vocab=202)
    graph = GraphStep(model, opt, table, 256)
    ix = torch.arange(256, device=device)
    for _ in range(20):
        graph(ix)
    counts = []
    for steps in [100, 1000]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            lr.fill_(1e-3)
            graph(ix)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        counts.append({"steps": steps, "seconds": elapsed, "steps_per_second": steps / elapsed})
    return {
        "layers": layers,
        "width": 128,
        "batch_size": 256,
        "sequence": 4,
        "vocab_size": 202,
        "dropout": 0.1,
        "clip": 1.0,
        "timing": counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="docs/development-artifacts/grok-depth-v1/preflight.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = {
        "started_utc": utc(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "tf32": True,
        "source_hash": source_hash(
            [
                "src/llm_memory_editability/grok_depth.py",
                "scripts/preflight_grok_depth.py",
                "tests/test_grok_depth_training.py",
            ]
        ),
    }
    checks = [
        ("hf_equivalence", hf_equivalence),
        ("graph_parity", graph_parity),
        ("resume_equivalence", resume_equivalence),
    ]
    try:
        for name, check in checks:
            result[name] = check(device)
            print(name, result[name], flush=True)
            write_json(args.out, result)
        result["throughput"] = [benchmark(device, layers) for layers in [2, 4]]
        result["passed"] = True
    except Exception as exc:
        result["passed"] = False
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_utc"] = utc()
        write_json(args.out, result)
    print(result["throughput"], flush=True)


if __name__ == "__main__":
    main()
