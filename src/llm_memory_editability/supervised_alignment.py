"""Original-label intermediate objectives and two explicit Loop adaptations.

This revision is separate from the frozen self-decoded-target experiment. Its
auxiliary labels are exactly the original answer/end-marker training labels;
there is no bridge-label argument. The optional cache exposes only the same
position of the same first-round layer. The DiscoLoop adaptation adds a soft
embedding channel at nonfinal round boundaries, preserving the full residual.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .self_decode_alignment import SelfDecodeGPT, _loss_dtype, _readout


def supervised_alignment_loss(
    model,
    first_state,
    final_state,
    positions,
    labels,
    *,
    kind="alignment",
    scope="supervised",
    eps=1e-8,
):
    """Mean detached-p(label)-weighted loss at original supervised positions.

    ``alignment`` is 1-cos(raw first-round state, detached E[label]), matching
    roadmap equation (41), rather than squared unit-vector distance. ``label_ce``
    uses the identical target/confidence rule and differentiates through native
    ln_f and the tied readout. Both divide by the number of nonignored labels,
    never the confidence sum. Apply the auxiliary coefficient outside this API.

    Only the supplied supervision slots are decoded, including the original end
    marker. Ignored slots are safely gathered at zero then receive zero weight;
    graph-padding rows and trailing padding contribute no loss or gradient.
    Diagnostics have [batch, supervision_slots] shape. final_state is accepted
    for runner compatibility but never inspected beyond shape or differentiated.
    Fixed-shape operations also support CUDA graph capture with dummy rows.
    """
    if kind not in {"alignment", "label_ce"}:
        raise ValueError("kind must be 'alignment' or 'label_ce'")
    if scope != "supervised":
        raise ValueError("scope must be 'supervised'")
    if first_state.ndim != 3 or final_state.shape != first_state.shape:
        raise ValueError("first_state and final_state must have the same [batch, length, width]")
    if positions.ndim != 2 or positions.shape != labels.shape:
        raise ValueError("positions and labels must have the same [batch, targets] shape")
    if positions.shape[0] != first_state.shape[0]:
        raise ValueError("supervision and states must have the same batch size")
    if positions.shape[1] < 1 or first_state.shape[1] < 1:
        raise ValueError("supervision and sequence dimensions must be nonempty")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    valid = labels != -100
    safe_positions = torch.where(valid, positions, 0)
    safe_labels = torch.where(valid, labels, 0)
    selected_state = first_state[
        torch.arange(first_state.shape[0], device=first_state.device)[:, None], safe_positions
    ]
    # Alignment's decoder is only a target-weight measurement. CE reuses one
    # differentiable projection for both CE and its detached confidence weight.
    if kind == "alignment":
        with torch.no_grad():
            logits = _readout(model, selected_state)
    else:
        logits = _readout(model, selected_state)
    with torch.no_grad():
        confidence = _loss_dtype(logits).softmax(-1).gather(-1, safe_labels[..., None])[..., 0]
        confidence = torch.where(valid, confidence, 0)

    if kind == "alignment":
        target = model.transformer.wte.weight.detach()[safe_labels]
        state_direction = F.normalize(_loss_dtype(selected_state), dim=-1, eps=eps)
        target_direction = F.normalize(_loss_dtype(target), dim=-1, eps=eps)
        per_token = 1 - (state_direction * target_direction).sum(-1)
    else:
        per_token = F.cross_entropy(
            _loss_dtype(logits).flatten(0, 1), safe_labels.flatten(), reduction="none"
        ).view_as(labels)
    valid_count = valid.sum()
    loss = (per_token * confidence).sum() / valid_count.clamp_min(1)
    return loss, {
        "targets": labels.detach(),
        "confidence": confidence,
        "valid_mask": valid,
        "selected_mask": valid,
        "valid_positions": valid_count,
        "selected_positions": valid_count,
        "confidence_sum": confidence.sum(),
        "per_token_loss": torch.where(valid, per_token.detach(), 0),
    }


class SupervisedAlignmentGPT(SelfDecodeGPT):
    """GPT with diagonal first-round KV and an optional DiscoLoop channel.

    Each cached bank is same-layer first-round K/V with j == i; the current
    bank remains j <= i. One attention softmax normalizes both banks jointly.
    Caches stay gradient-connected and live only inside one forward. This does
    not exclude information already mixed into a same-position cached vector
    by earlier causal layers. No extra modules, parameters or buffers are added.

    The channel uses native GPT ln_f/tied decoding, a full-vocabulary softmax,
    and tied soft re-encoding. Affine-free RMS normalization is an explicitly
    fixed adaptation to the current GPT. It is added after every nonfinal round
    at every position during both training and inference. The returned first
    state is before the channel; there is no channel after the final round.
    """

    def __init__(
        self,
        config,
        repeats=2,
        first_round_kv=False,
        discoloop=False,
        discoloop_alpha=1.0,
        discoloop_temperature=1.0,
        discoloop_eps=1e-6,
    ):
        if not isinstance(discoloop, bool):
            raise ValueError("discoloop must be boolean")
        for name, value in (
            ("discoloop_temperature", discoloop_temperature),
            ("discoloop_eps", discoloop_eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(discoloop_alpha) or discoloop_alpha < 0:
            raise ValueError("discoloop_alpha must be finite and nonnegative")
        super().__init__(config, repeats=repeats, first_round_kv=first_round_kv)
        self.discoloop = discoloop
        self.discoloop_alpha = discoloop_alpha
        self.discoloop_temperature = discoloop_temperature
        self.discoloop_eps = discoloop_eps

    def _round(self, x, first_round_cache=None, capture_cache=None):
        if first_round_cache is None:
            return super()._round(x, capture_cache=capture_cache)
        cfg, t = self.config, self.transformer
        batch, length, _ = x.shape
        if len(first_round_cache) != len(t.h):
            raise ValueError("first-round cache must contain one K/V pair per layer")
        index = torch.arange(length, device=x.device)
        diagonal = index[:, None] == index[None, :]
        causal = index[None, :] <= index[:, None]
        shared_mask = torch.cat((diagonal, causal), dim=1)
        for layer_index, block in enumerate(t.h):
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
            shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
            q, k, v = (item.view(shape).transpose(1, 2) for item in (q, k, v))
            if capture_cache is not None:
                capture_cache.append((k, v))
            cached_k, cached_v = first_round_cache[layer_index]
            y = F.scaled_dot_product_attention(
                q,
                torch.cat((cached_k, k), dim=2),
                torch.cat((cached_v, v), dim=2),
                attn_mask=shared_mask,
                dropout_p=cfg.attn_pdrop if self.training else 0.0,
            )
            y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, cfg.n_embd))
            x = x + F.dropout(y, cfg.resid_pdrop, self.training)
            x = x + block.mlp(block.ln_2(x))
        return x

    def _discoloop_boundary(self, state):
        if self.discoloop_alpha == 0:
            return state
        logits = _readout(self, state)
        probability = (_loss_dtype(logits) / self.discoloop_temperature).softmax(-1)
        soft_embedding = F.linear(probability, self.transformer.wte.weight.T)
        channel = _loss_dtype(soft_embedding)
        channel = channel * torch.rsqrt(
            channel.square().mean(-1, keepdim=True) + self.discoloop_eps
        )
        return state + self.discoloop_alpha * channel.to(state.dtype)

    def forward_with_states(self, tokens, positions=None):
        if not self.discoloop or self.discoloop_alpha == 0:
            return super().forward_with_states(tokens, positions)
        if self.repeats < 1 or not len(self.transformer.h):
            raise ValueError("at least one round and one block are required")
        x = self._embedded(tokens)
        first_state = None
        first_round_cache = [] if self.first_round_kv else None
        for index in range(self.repeats):
            x = self._round(
                x,
                first_round_cache=first_round_cache if index else None,
                capture_cache=first_round_cache if index == 0 else None,
            )
            if index == 0:
                first_state = x
            if index < self.repeats - 1:
                x = self._discoloop_boundary(x)
        return _readout(self, x, positions), first_state, x
