"""Self-decoded intermediate objectives on the original reproduction GPT.

With first_round_kv=False the ordinary forward, parameters, tied readout and
state-dict format are unchanged. Optional same-layer first-round KV access
matches the existing FirstRoundKVGPT: both cached and current full prefixes are
causal and jointly normalized, and cached tensors retain gradients. Targets come
only from the model's own intermediate readout; graph
answers and bridge labels are not accepted. ``labels`` supplies the existing
supervision mask, never the auxiliary target identity.

Consistency compares first/final readouts at the *same* causal position in an
unintervened pass. It can select detached training targets or a fixed-weight
diagnostic, but must not be described as a free inference-time gate: the latter
also computes the original downstream rounds before replaying them.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .grokking_reproduction import ReproductionGPT


def valid_token_mask(positions, labels, sequence_length):
    """Mark the prefix ending at the last nonignored original supervised index.

    Original atomic rows have positions [1, 2] and one trailing pad; composition
    rows have [2, 3]. Entirely ignored graph-padding rows have no valid tokens.
    Only ``labels != -100`` is inspected, so changing answer identities cannot
    change this mask or the auxiliary objective.
    """
    if positions.ndim != 2 or positions.shape != labels.shape:
        raise ValueError("positions and labels must have the same [batch, targets] shape")
    if positions.shape[1] < 1 or sequence_length < 1:
        raise ValueError("supervision and sequence dimensions must be nonempty")
    last = torch.where(labels != -100, positions, -1).amax(dim=-1)
    return torch.arange(sequence_length, device=positions.device)[None, :] <= last[:, None]


def _readout(model, state, positions=None):
    x = model.transformer.ln_f(state)
    if positions is not None:
        x = x[torch.arange(len(state), device=state.device)[:, None], positions]
    return F.linear(x, model.transformer.wte.weight)


@torch.no_grad()
def detached_self_targets(model, state):
    """Original ln_f/tied readout, full vocabulary, detached own argmax and pmax."""
    logits = _readout(model, state)
    targets = logits.argmax(-1)
    probabilities = logits.float().softmax(-1)
    confidence = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return targets, confidence


def _loss_dtype(tensor):
    # Accumulate reduced-precision objectives in float32 without demoting double
    # in CPU numerical/gradient checks.
    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


def self_decode_loss(
    model,
    first_state,
    final_state,
    positions,
    labels,
    *,
    kind="alignment",
    scope="all",
    eps=1e-8,
):
    """Return an unweighted auxiliary scalar and detached diagnostics.

    ``alignment`` uses squared distance between unit raw intermediate state and
    unit detached input embedding of its own decoded token. ``self_ce`` uses the
    same target/confidence rule, but backpropagates through intermediate ln_f and
    the tied readout. In both cases the denominator is ALL valid positions, not
    confidence sum or number passing consistency. Apply lambda outside this API.

    With ``scope='consistent'``, pass final_state from the same ordinary forward;
    its argmax is used only for a detached same-position equality mask. No part
    of final_state contributes an auxiliary gradient. Empty valid masks produce
    a differentiable zero. This API does not mutate model parameters or states.
    """
    if kind not in {"alignment", "self_ce"}:
        raise ValueError("kind must be 'alignment' or 'self_ce'")
    if scope not in {"all", "consistent"}:
        raise ValueError("scope must be 'all' or 'consistent'")
    if first_state.ndim != 3 or final_state.shape != first_state.shape:
        raise ValueError("first_state and final_state must have the same [batch, length, width]")
    if positions.shape[0] != first_state.shape[0]:
        raise ValueError("supervision and states must have the same batch size")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    valid = valid_token_mask(positions, labels, first_state.shape[1])
    targets, confidence = detached_self_targets(model, first_state)
    selected = valid
    if scope == "consistent":
        with torch.no_grad():
            final_targets = _readout(model, final_state).argmax(-1)
            selected = valid & (targets == final_targets)

    if kind == "alignment":
        embedding = model.transformer.wte.weight.detach()[targets]
        state_direction = F.normalize(_loss_dtype(first_state), dim=-1, eps=eps)
        target_direction = F.normalize(_loss_dtype(embedding), dim=-1, eps=eps)
        per_token = (state_direction - target_direction).square().sum(-1)
    else:
        logits = _readout(model, first_state)
        per_token = F.cross_entropy(
            _loss_dtype(logits).flatten(0, 1), targets.flatten(), reduction="none"
        ).view_as(targets)

    valid_count = valid.sum()
    weights = confidence * selected
    loss = (per_token * weights).sum() / valid_count.clamp_min(1)
    info = {
        "pseudo_targets": targets,
        "confidence": confidence,
        "valid_mask": valid,
        "selected_mask": selected,
        "valid_positions": valid_count,
        "selected_positions": selected.sum(),
        "confidence_sum": weights.sum(),
        "per_token_loss": per_token.detach(),
    }
    return loss, info


def norm_matched_replacement(state, embedding, selected, alpha=1.0, eps=1e-8):
    """Interpolate toward detached token direction at the original hidden norm.

    At alpha=1 the selected state norms are preserved (for nonzero embeddings).
    At 0<alpha<1 the interpolation is deliberately not renormalized; its norm
    need not match either endpoint. alpha=0 returns the original tensor exactly.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between zero and one")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if state.shape != embedding.shape or selected.shape != state.shape[:-1]:
        raise ValueError("embedding and mask must match the state dimensions")
    if alpha == 0:
        return state
    replacement = state.norm(dim=-1, keepdim=True) * F.normalize(
        embedding.detach().to(state.dtype), dim=-1, eps=eps
    )
    mixed = (1 - alpha) * state + alpha * replacement
    return torch.where(selected.unsqueeze(-1), mixed, state)


class SelfDecodeGPT(ReproductionGPT):
    """Original GPT with access to the first complete round's raw residual state.

    A round executes every independently parameterized block, preserving the
    exact original SDPA, attention/MLP, residual, dropout and normalization calls.
    For the primary Loop 4x2 setting this exposes the state after all four blocks,
    before the second round. No new modules, parameters or buffers are added.

    first_round_kv=True adds the same layer's first-round K/V to each later
    round's attention. Both halves use j <= i in original token coordinates;
    concatenated-index causal attention alone would leak future cached tokens.
    This retains the full current prefix, matching the project's FirstRoundKVGPT
    adaptation and Monea's window=T case. It is not MoR's first-round-only KV
    variant: current-round K/V are still computed and participate in attention.
    Cache lifetime is one forward only, with no detach or cross-layer reuse.
    """

    def __init__(self, config, repeats=1, first_round_kv=False):
        if not isinstance(first_round_kv, bool):
            raise ValueError("first_round_kv must be boolean")
        super().__init__(config, repeats)
        self.first_round_kv = first_round_kv

    def _round(self, x, first_round_cache=None, capture_cache=None):
        cfg, t = self.config, self.transformer
        batch, length, _ = x.shape
        if first_round_cache is not None and len(first_round_cache) != len(t.h):
            raise ValueError("first-round cache must contain one K/V pair per layer")
        shared_mask = None
        if first_round_cache is not None:
            token_positions = torch.arange(length, device=x.device)
            causal = token_positions[None, :] <= token_positions[:, None]
            shared_mask = torch.cat((causal, causal), dim=1)
        for layer_index, block in enumerate(t.h):
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
            shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            if capture_cache is not None:
                capture_cache.append((k, v))
            if first_round_cache is None:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
                )
            else:
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

    def _embedded(self, tokens):
        t = self.transformer
        x = t.wte(tokens) + t.wpe(torch.arange(tokens.shape[1], device=tokens.device))
        return F.dropout(x, self.config.embd_pdrop, self.training)

    def forward_with_states(self, tokens, positions=None):
        """Return logits and full [B,T,H] first/final states, both before ln_f."""
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
        return _readout(self, x, positions), first_state, x

    def forward(self, tokens, positions=None):
        return self.forward_with_states(tokens, positions)[0]

    @torch.no_grad()
    def forward_with_replacement(
        self,
        tokens,
        positions=None,
        *,
        scope="none",
        alpha=1.0,
        valid_mask=None,
        r1_position=1,
        return_info=False,
    ):
        """Zero-training diagnostic; ordinary forward never calls this method.

        ``consistent`` first computes the unintervened final round, gates only by
        same-position first/final equality, then replays downstream computation.
        Because each comparison uses a causal state, it never sees future tokens.
        All-token scope includes all supplied tokens unless valid_mask is passed.
        r1 is the first relation index in the original head/relation token layout.
        """
        if self.training:
            raise ValueError("replacement is an inference diagnostic; call model.eval()")
        if scope not in {"none", "r1", "all", "consistent"}:
            raise ValueError("scope must be 'none', 'r1', 'all' or 'consistent'")
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and between zero and one")
        if self.repeats < 2:
            raise ValueError("replacement requires a downstream round (repeats >= 2)")
        if valid_mask is not None and (
            valid_mask.shape != tokens.shape or valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be boolean with the same shape as tokens")
        if scope == "r1" and not 0 <= r1_position < tokens.shape[1]:
            raise ValueError("r1_position must name a supplied token position")

        if (scope == "none" or alpha == 0) and not return_info:
            return self(tokens, positions)

        first_round_cache = [] if self.first_round_kv else None
        first_state = self._round(self._embedded(tokens), capture_cache=first_round_cache)
        targets, confidence = detached_self_targets(self, first_state)
        valid = torch.ones_like(tokens, dtype=torch.bool) if valid_mask is None else valid_mask
        selected = valid.clone()
        final_targets = None
        if scope == "none":
            selected.zero_()
        elif scope == "r1":
            selected &= torch.arange(tokens.shape[1], device=tokens.device) == r1_position
        elif scope == "consistent":
            original_final = first_state
            for _ in range(self.repeats - 1):
                original_final = self._round(original_final, first_round_cache=first_round_cache)
            final_targets = _readout(self, original_final).argmax(-1)
            selected &= targets == final_targets

        replacement = norm_matched_replacement(
            first_state, self.transformer.wte.weight[targets], selected, alpha
        )
        final = replacement
        for _ in range(self.repeats - 1):
            final = self._round(final, first_round_cache=first_round_cache)
        logits = _readout(self, final, positions)
        if not return_info:
            return logits
        return logits, {
            "first_state": first_state,
            "replacement_state": replacement,
            "final_state": final,
            "pseudo_targets": targets,
            "confidence": confidence,
            "selected_mask": selected,
            "original_final_targets": final_targets,
        }
