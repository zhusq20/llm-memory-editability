"""Four loop mechanisms and their feedback controls on the agreed shared GPT backbone.

InfiLoop is a single-stream adaptation of the authors' Connection, initialization
and raw-candidate recurrence, not their Sudoku H/L/grid architecture. Reference:
pixeli99/InfiLoop, commit 63716a9c46723d3592c8ad4fc5819855f22aaa47,
sudoku/model.py (MIT; license retained with the experiment's reference snapshot).
"""

import math

import torch
from torch.nn import functional as F

from .loop_native_handoff import LoopNativeHandoffGPT
from .loop_terminal_training import checked_depths, run_terminal_batch, supervised_token_count
from .self_decode_alignment import _loss_dtype, _readout


class InfiLoopGPT(LoopNativeHandoffGPT):
    """Original scored raw candidates and fixed random prior on a common backbone."""

    def __init__(self, config, repeats, *, state_seed, feedback_to_candidate=False, **options):
        super().__init__(
            config,
            repeats,
            handoff="content",
            feedback_to_candidate=feedback_to_candidate,
            **options,
        )
        self.handoff = "infiloop"
        # Author initialization: variance-corrected truncated N(0,1), fixed buffer.
        # Use a separate RNG so all four methods retain identical backbone weights.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(state_seed)
            bound = math.erf(2 / math.sqrt(2))
            scale = 1 / math.sqrt(1 - 4 * math.exp(-2) / (math.sqrt(2 * math.pi) * bound))
            initial = torch.empty(config.n_embd)
            initial.uniform_(-bound, bound).erfinv_().mul_(math.sqrt(2) * scale)
            initial.clamp_(-2 * scale, 2 * scale)
        self.register_buffer("handoff_initial_h", initial)

    def content_weight(self, x):
        # Keep the reference's float32 RMS statistics, dtype cast, clipping and exp.
        value = x.float()
        key = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
        return (
            torch.einsum("hd,btd->bth", self.handoff_query[None].to(x.dtype), key)
            .float()
            .clamp(-8, 8)
            .exp()
        )

    def initial_state(self, injection):
        high = self.handoff_initial_h.to(injection.dtype).expand_as(injection)
        # Matches the released outer initial_state: initial memory is not learned
        # through its reset. Every subsequent loop update remains differentiable.
        with torch.no_grad():
            weight = self.content_weight(high)
            numerator = weight * high.float()
        return high, numerator, weight, None, injection

    def aggregate_round(self, state, injection, *, first_round=False):
        high, numerator, denominator = state[:3]
        candidate = self._round(high + injection)
        if self.feedback_to_candidate:
            # The added control changes only the candidate H -> H + q(H).
            # InfiLoop still scores/aggregates it without normalizing the value.
            candidate = _loss_dtype(candidate) + self._decoded_feedback(candidate)[1]
        weight = self.content_weight(candidate)
        numerator = self.beta * numerator + weight * candidate.float()
        denominator = self.beta * denominator + weight
        high = (numerator / denominator).to(candidate.dtype)
        return (high, numerator, denominator, None, injection), weight / denominator


def terminal_logits(model, tokens, positions, depths):
    """Masked full-batch rollout; each method reads its own original endpoint."""
    depths = checked_depths(depths, len(tokens))
    if model.handoff == "invariant":
        return run_terminal_batch(model, tokens, positions, depths)[0]
    injection = _loss_dtype(model._embedded(tokens))
    limits = torch.as_tensor(depths, device=tokens.device)[:, None, None]
    if model.handoff == "infiloop":
        state = model.initial_state(injection)
        for index in range(int(depths.max())):
            proposed, _ = model.aggregate_round(state, injection)
            state = tuple(
                torch.where(limits > index, new, old)
                for new, old in zip(proposed[:3], state[:3], strict=True)
            ) + (None, injection)
        return _readout(model, state[0], positions)
    if model.handoff not in ("plain", "additive"):
        raise ValueError(f"Unknown comparison handoff: {model.handoff}")
    raw, carry = injection, injection
    for index in range(int(depths.max())):
        proposed = model._round(carry)
        raw = torch.where(limits > index, proposed, raw)
        if model.handoff == "plain":
            carry = raw
        elif index + 1 < int(depths.max()):
            # DiscoLoop's terminal alpha is zero: read raw H_T, not H_T + q(H_T).
            feedback_carry = model._discoloop_boundary(proposed)
            carry = torch.where(limits > index + 1, feedback_carry, carry)
    return _readout(model, raw, positions)


def endpoint_loss(model, tokens, positions, labels, depths):
    count = supervised_token_count(tokens, positions, labels, -100)
    logits = terminal_logits(model, tokens, positions, depths)
    return (
        F.cross_entropy(
            _loss_dtype(logits).flatten(0, 1), labels.flatten(), reduction="sum", ignore_index=-100
        )
        / count
    )
