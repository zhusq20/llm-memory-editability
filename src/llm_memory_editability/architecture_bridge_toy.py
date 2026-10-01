"""A frozen small GQA / gated-delta mechanism experiment.

Both models use RMSNorm, SwiGLU and tied token embeddings. The hybrid is a
deliberately small mechanism model, NOT an implementation of any Qwen release.
Its delta mixer has a genuine recurrent matrix, causal q/k/v convolution,
normalized q/k, input-dependent decay/write gates, and a headwise output gate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from llm_memory_editability.twohop_depth import BOS, ENTITY, EOS


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x, offset):
    """Full rotary position embedding; head width is even."""
    width = x.shape[-1]
    positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=x.dtype)
    inverse = 10000.0 ** (-torch.arange(0, width, 2, device=x.device, dtype=x.dtype) / width)
    phase = positions[:, None] * inverse[None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (even * phase.cos() - odd * phase.sin(), even * phase.sin() + odd * phase.cos()),
        dim=-1,
    ).flatten(-2)


class GQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width, self.heads, self.kv_heads = cfg["width"], cfg["heads"], cfg["kv_heads"]
        self.dim = width // self.heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, self.kv_heads * self.dim, bias=False)
        self.v = nn.Linear(width, self.kv_heads * self.dim, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm = RMSNorm(self.dim, cfg["norm_eps"])
        self.k_norm = RMSNorm(self.dim, cfg["norm_eps"])

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        offset = 0 if cache is None else cache["k"].shape[2]
        q = self.q(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        q, k = rope(self.q_norm(q), offset), rope(self.k_norm(k), offset)
        if cache is not None:
            k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
        new_cache = {"k": k, "v": v}
        repeat = self.heads // self.kv_heads
        keys, values = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        scores = (q @ keys.transpose(-2, -1)) / math.sqrt(self.dim)
        allowed = (
            torch.arange(k.shape[2], device=x.device)[None, :]
            <= torch.arange(offset, offset + length, device=x.device)[:, None]
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        out = (scores.softmax(-1) @ values).transpose(1, 2).reshape(batch, length, width)
        return self.out(out), new_cache


class GatedDelta(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width, self.heads = cfg["width"], cfg["heads"]
        self.dim, self.kernel = width // self.heads, cfg["conv_kernel"]
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.kernel, 3 * width))
        self.decay = nn.Linear(width, self.heads, bias=True)
        self.write = nn.Linear(width, self.heads, bias=True)
        self.gate = nn.Linear(width, self.heads, bias=True)
        self.out_norm = RMSNorm(self.dim, cfg["norm_eps"])
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        raw = self.qkv(x)
        if cache is None:
            history = raw.new_zeros(batch, self.kernel - 1, 3 * width)
            state = raw.new_zeros(batch, self.heads, self.dim, self.dim)
        else:
            history, state = cache["conv"], cache["state"]
        full = torch.cat((history, raw), 1)
        convolved = sum(full[:, j : j + length] * self.conv_weight[j] for j in range(self.kernel))
        q, k, v = F.silu(convolved).chunk(3, -1)
        q = F.normalize(q.view(batch, length, self.heads, self.dim), dim=-1, eps=1e-6)
        k = F.normalize(k.view(batch, length, self.heads, self.dim), dim=-1, eps=1e-6)
        v = v.view(batch, length, self.heads, self.dim)
        alpha = torch.exp(-F.softplus(self.decay(x)))
        beta, gate = self.write(x).sigmoid(), self.gate(x).sigmoid()
        outputs = []
        for t in range(length):
            # S_t = alpha S_(t-1)(I-beta kk^T) + beta vk^T.
            state = alpha[:, t, :, None, None] * state
            read_at_key = (state * k[:, t, :, None, :]).sum(-1)
            error = v[:, t] - read_at_key
            state = state + beta[:, t, :, None, None] * (error[..., :, None] * k[:, t, :, None, :])
            outputs.append((state * q[:, t, :, None, :]).sum(-1))
        out = self.out_norm(torch.stack(outputs, 1)) * gate[..., None]
        new_cache = {"conv": full[:, -(self.kernel - 1) :], "state": state}
        return self.out(out.reshape(batch, length, width)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg["width"], cfg["mlp_hidden"], bias=False)
        self.gate = nn.Linear(cfg["width"], cfg["mlp_hidden"], bias=False)
        self.down = nn.Linear(cfg["mlp_hidden"], cfg["width"], bias=False)

    def forward(self, x):
        return self.down(self.up(x) * F.silu(self.gate(x)))


class Block(nn.Module):
    def __init__(self, kind, cfg):
        super().__init__()
        self.norm_mixer = RMSNorm(cfg["width"], cfg["norm_eps"])
        self.mixer = GQA(cfg) if kind == "gqa" else GatedDelta(cfg)
        self.norm_mlp = RMSNorm(cfg["width"], cfg["norm_eps"])
        self.mlp = SwiGLU(cfg)


def apply_patch(output, patch):
    if patch is None:
        return output
    result = output.clone()
    result[:, patch["positions"]] = patch["value"].to(output)
    return result


class ToyLM(nn.Module):
    def __init__(self, architecture, cfg):
        super().__init__()
        self.config = cfg
        self.architecture = architecture
        self.token = nn.Embedding(ENTITY + cfg["entities"] + cfg["relations"], cfg["width"])
        self.blocks = nn.ModuleList(Block(kind, cfg) for kind in architecture["mixers"])
        self.norm_final = RMSNorm(cfg["width"], cfg["norm_eps"])

    def forward(
        self, tokens, positions=None, *, cache=None, patches=None, capture=False, return_cache=False
    ):
        h, states, trace = self.token(tokens), [], {}
        patches = {} if patches is None else patches
        for layer, block in enumerate(self.blocks):
            past = None if cache is None else cache[layer]
            mixer, state = block.mixer(block.norm_mixer(h), past)
            mixer = apply_patch(mixer, patches.get(("mixer", layer)))
            h = h + mixer
            mlp = apply_patch(block.mlp(block.norm_mlp(h)), patches.get(("mlp", layer)))
            h = h + mlp
            states.append(state)
            if capture:
                trace[("mixer", layer)], trace[("mlp", layer)] = mixer, mlp
        h = self.norm_final(h)
        if positions is not None:
            h = h[torch.arange(len(h), device=h.device)[:, None], positions]
        logits = F.linear(h, self.token.weight)
        return (logits, states, trace) if capture or return_cache else logits


def initialize(model, seed):
    """Pair every common named and shaped parameter without RNG-order effects."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".bias"):
                parameter.zero_()
            elif "norm" in name and name.endswith(".weight"):
                parameter.fill_(1)
            else:
                key = hashlib.sha256(f"{seed}:{name}".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "little"))
                std = 0.02
                if name.endswith("mixer.out.weight") or name.endswith("mlp.down.weight"):
                    std /= math.sqrt(2 * len(model.blocks))
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)


def parameter_digest(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def exposure(cfg, steps):
    count = cfg["batch_per_kind"] * steps
    return {
        "steps": steps,
        "atomic_examples": count,
        "training_composite_examples": count,
        "valid_input_tokens": count * 11,
        "processed_tokens_including_padding": count * 12,
        "supervised_answer_eos_tokens": count * 4,
    }


def flops_estimate(model, cfg, steps):
    """Conventional 6*N*T estimate plus transparent small mixer correction."""
    params = sum(p.numel() for p in model.parameters())
    tokens = exposure(cfg, steps)["processed_tokens_including_padding"]
    n_gqa = model.architecture["mixers"].count("gqa")
    n_delta = model.architecture["mixers"].count("delta")
    length, width, heads = 6, cfg["width"], cfg["heads"]
    dim = width // heads
    mixing = tokens * (n_gqa * 12 * length * width + n_delta * 24 * heads * dim * dim)
    return {
        "parameter_count": params,
        "conventional_6_parameter_token_flops": 6 * params * tokens,
        "additional_mixing_flops_approx": mixing,
        "total_approx": 6 * params * tokens + mixing,
        "limitation": (
            "Rough training estimate, not profiler measurements; 6NT includes embeddings "
            "and excludes detailed normalization/activation costs."
        ),
    }


@torch.no_grad()
def generate(model, x, y, device, batch=128, patches=None):
    result = {key: [] for key in ("pred", "eos", "logp", "logits")}
    for start in range(0, len(x), batch):
        tokens = torch.as_tensor(x[start : start + batch], device=device)
        target = torch.as_tensor(y[start : start + batch], device=device)
        part = None
        if patches is not None:
            part = {
                key: {
                    "positions": value["positions"],
                    "value": value["value"][start : start + batch],
                }
                for key, value in patches.items()
            }
        logits = model(tokens, patches=part)[:, -1]
        prediction = logits.argmax(-1)
        stop = model(torch.cat((tokens, prediction[:, None]), 1), patches=part)[:, -1].argmax(-1)
        result["pred"].append(prediction.cpu().numpy())
        result["eos"].append(stop.cpu().numpy())
        result["logp"].append(logits.log_softmax(-1).gather(1, target[:, None]).cpu().numpy()[:, 0])
        result["logits"].append(logits.cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


@torch.no_grad()
def evaluate(model, world, cfg, device):
    model.eval()
    arrays, metrics = {}, {}
    for name, key, mask in (
        ("atomic", "atomic", slice(None)),
        ("train", "composite", world["train_mask"]),
        ("test", "composite", ~world["train_mask"]),
    ):
        x, y = world[key + "_x"][mask], world[key + "_y"][mask]
        output = generate(model, x, y, device, cfg["eval_batch"])
        for field, value in output.items():
            arrays[f"{name}_{field}"] = value
        correct = (output["pred"] == y) & (output["eos"] == EOS)
        metrics[name] = {
            "n": len(y),
            "answer_accuracy": float(np.mean(output["pred"] == y)),
            "exact_answer_eos": float(correct.mean()),
            "answer_nll": float(-output["logp"].mean()),
        }
    indices = np.flatnonzero(~world["train_mask"])
    a, r1, r2 = world["comps"][indices].T
    bridge = arrays["atomic_pred"][a * cfg["relations"] + r1]
    bridge_eos = arrays["atomic_eos"][a * cfg["relations"] + r1]
    second = np.c_[np.full(len(a), BOS), bridge, ENTITY + cfg["entities"] + r2]
    y = world["composite_y"][indices]
    output = generate(model, second, y, device, cfg["eval_batch"])
    arrays.update({f"two_call_{field}": value for field, value in output.items()})
    arrays["two_call_bridge"], arrays["two_call_bridge_eos"] = bridge, bridge_eos
    correct = (output["pred"] == y) & (output["eos"] == EOS) & (bridge_eos == EOS)
    metrics["two_call"] = {"n": len(y), "exact_answer_eos": float(correct.mean())}
    atom_ok = (arrays["atomic_pred"] == world["atomic_y"]) & (arrays["atomic_eos"] == EOS)
    gold_bridge = world["maps"][r1, a]
    both = atom_ok[a * cfg["relations"] + r1] & atom_ok[gold_bridge * cfg["relations"] + r2]
    direct_ok = (arrays["test_pred"] == y) & (arrays["test_eos"] == EOS)
    arrays["test_both_atoms_correct"] = both
    metrics["test_given_both_atoms"] = {
        "n": int(both.sum()),
        "coverage": float(both.mean()),
        "exact_answer_eos": float(direct_ok[both].mean()) if both.any() else None,
    }
    return metrics, arrays


def intervention_conditions(traces, cfg, device):
    """Same orthogonal transform for both donor norms; no fitted direction."""
    rng = np.random.default_rng(cfg["random_control_seed"])
    perm = torch.as_tensor(rng.permutation(cfg["width"]), device=device)
    signs = torch.as_tensor(rng.choice([-1.0, 1.0], cfg["width"]), device=device)
    sites = {
        "mlp": (cfg["mlp_layer"], cfg["mlp_position"]),
        "mixer": (cfg["mixer_layer"], cfg["mixer_position"]),
    }
    conditions, norms = {}, {}
    for target in cfg["patch_targets"]:
        kinds = ("mlp", "mixer") if target == "joint" else (target,)
        for donor in cfg["patch_donors"]:
            patches, measurements = {}, {}
            for kind in kinds:
                layer, position = sites[kind]
                base = traces["identity"][(kind, layer)][:, [position]]
                source = donor.replace("random_", "")
                delta = traces[source][(kind, layer)][:, [position]] - base
                if donor.startswith("random_"):
                    delta = delta[..., perm] * signs.to(delta)
                patches[(kind, layer)] = {"positions": [position], "value": base + delta}
                measurements[kind] = delta.flatten(1).norm(dim=-1).cpu().numpy()
            key = f"{target}_{donor}"
            conditions[key], norms[key] = patches, measurements
    return conditions, norms


@torch.no_grad()
def diagnostics(model, world, cfg, device):
    model.eval()
    cases = world["cases"]
    query = world["composite_x"][cases[:, 0]]
    correct = np.c_[
        np.full(len(cases), BOS),
        ENTITY + cases[:, 9],
        ENTITY + cfg["entities"] + cases[:, 10],
        ENTITY + cfg["entities"] + cases[:, 3],
    ]
    wrong = np.c_[
        np.full(len(cases), BOS),
        ENTITY + cases[:, 6],
        ENTITY + cfg["entities"] + cases[:, 2],
        ENTITY + cfg["entities"] + cases[:, 3],
    ]
    traces = {}
    for key, x in (("identity", query), ("correct", correct), ("wrong", wrong)):
        _, _, traces[key] = model(torch.as_tensor(x, device=device), capture=True)
    conditions, norms = intervention_conditions(traces, cfg, device)
    target, competitor = ENTITY + cases[:, 5], ENTITY + cases[:, 8]
    arrays = {
        "case_ids": cases[:, 0],
        "target": target,
        "wrong_target": competitor,
        "query": query,
        "correct_donor": correct,
        "wrong_donor": wrong,
    }
    baseline = generate(model, query, target, device, cfg["eval_batch"])
    base_ok = (baseline["pred"] == target) & (baseline["eos"] == EOS)
    metrics = {}
    for name, patches in {"baseline": None, **conditions}.items():
        output = (
            baseline
            if patches is None
            else generate(model, query, target, device, cfg["eval_batch"], patches)
        )
        for field, value in output.items():
            arrays[f"{name}_{field}"] = value
        row = np.arange(len(target))
        margin = output["logits"][row, target] - output["logits"][row, competitor]
        arrays[f"{name}_margin"] = margin
        exact = (output["pred"] == target) & (output["eos"] == EOS)
        wrong_ok = (output["pred"] == competitor) & (output["eos"] == EOS)
        metrics[name] = {
            "n": len(target),
            "correct_exact": float(exact.mean()),
            "wrong_exact": float(wrong_ok.mean()),
            "rescued": int((exact & ~base_ok).sum()),
            "damaged": int((~exact & base_ok).sum()),
            "baseline_correct_n": int(base_ok.sum()),
            "baseline_incorrect_n": int((~base_ok).sum()),
            "mean_correct_minus_wrong_margin": float(margin.mean()),
            "max_logit_change": float(np.max(np.abs(output["logits"] - baseline["logits"]))),
        }
        for kind, value in norms.get(name, {}).items():
            arrays[f"{name}_{kind}_norm"] = value
    # Functional interaction in the 2x2 channel intervention, not an attribution percentage.
    for donor in ("correct", "wrong", "random_correct", "random_wrong"):
        interaction = (
            arrays[f"joint_{donor}_margin"]
            - arrays[f"mlp_{donor}_margin"]
            - arrays[f"mixer_{donor}_margin"]
            + arrays["baseline_margin"]
        )
        arrays[f"interaction_{donor}"] = interaction
    return metrics, arrays


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
