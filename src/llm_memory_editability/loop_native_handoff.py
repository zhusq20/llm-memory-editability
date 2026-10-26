"""Loop-native handoff between rounds of a shared GPT block (InfiLoop-style carry).

All handoffs share the GPT-2 block and tied readout. By default, except for
``plain``, they use DiscoLoop feedback q(H) = RMSNorm(E^T softmax(E ln_f(H) / tau)).

``plain``      Ordinary Loop: pass raw H directly to the next shared block call.
``additive``   The existing DiscoLoop chain: a round passes H + q(H) to the next one.
               A call reads raw H after its last round; the carry is H + q(H).
``ema``        Normalized aggregation with score zero, i.e. a learned exponential average.
``content``    InfiLoop's score s = w^T RMSNorm(c) with one shared pseudo-query w.
``invariant``  s = psi(phi). phi holds only quantities that do not change when
               vocabulary items are relabeled: decode entropy, maximum probability,
               agreement with the previous decode, relative change and cosine to the
               previous state. The score cannot condition on which entity is decoded.
``state-mlp``  s = psi([h_{t-1}; c_t]), using the two raw states with no additional
               normalization or decoded-probability features.

Aggregation handoffs re-inject the raw GPT embedding e = wte + wpe every round:

    H_t = F(h_{t-1} + e)                      one pass of every shared block
    c_t = RMSNorm(H_t + q(H_t))
    N_t = beta N_{t-1} + exp(s_t) c_t
    D_t = beta D_{t-1} + exp(s_t)
    h_t = N_t / D_t,  gamma_t = exp(s_t) / D_t

with h_0 = N_0 = e, D_0 = 1 and readout ln_f(h_T). The injection keeps the native GPT
scale, as the additive DiscoLoop does: an RMS-normalized injection would dominate the
state at initialization and make the tied readout start by copying the input token
(engineering smoke, 2026-10-20: initial answer CE 13.6 versus 7.4). Every score starts
at zero, so the
four aggregation handoffs begin as the same exponential average. Nothing here defines
an objective: callers train with answer/EOS cross-entropy only, and no handoff reads
labels, hop counts, positions or graph identity.

The terminal-supervision recipe selects first_round_input="embedding": H_1 = F(e),
then the recurrence above for t >= 2. Its carry also retains the original embedding
(including embedding dropout), so continuing a trajectory never reinjects a new draw.
The default retains the historical recurrence for existing configs/checkpoints.

The feedback ablation sets feedback_to_candidate=False: c_t = RMSNorm(H_t).
It retains the decoded probabilities in the invariant score, but never constructs
or adds the decoded embedding. All historical configurations default to True.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .self_decode_alignment import _loss_dtype, _readout
from .supervised_alignment import SupervisedAlignmentGPT

HANDOFFS = ("additive", "ema", "content", "invariant", "plain")
# Keep the original matrix stable for the historical configs and their contracts.
SUPPORTED_HANDOFFS = (*HANDOFFS, "state-mlp")
FEATURES = ("entropy", "max_probability", "agreement", "relative_change", "cosine")


def rms_normalize(value, eps):
    value = _loss_dtype(value)
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)


class LoopNativeHandoffGPT(SupervisedAlignmentGPT):
    """Shared-block loop whose carry is one of ``SUPPORTED_HANDOFFS``.

    Handoff parameters are named ``handoff_*`` and are created after the backbone
    without consuming the global RNG, so the backbone matches a DiscoLoop built from
    the same seed. ``run`` accepts any number of rounds and an optional carried state,
    which is how deep supervision continues a previous call.
    """

    def __init__(
        self,
        config,
        repeats=4,
        *,
        handoff,
        initial_beta=0.55,
        score_hidden=16,
        score_clip=8.0,
        score_seed=0,
        eps=1e-6,
        first_round_input="state_plus_embedding",
        feedback_to_candidate=True,
    ):
        if handoff not in SUPPORTED_HANDOFFS:
            raise ValueError(f"Unknown handoff: {handoff}")
        if not 0 < initial_beta < 1:
            raise ValueError("initial_beta must be strictly between zero and one")
        if not math.isfinite(score_clip) or score_clip <= 0:
            raise ValueError("score_clip must be finite and positive")
        if first_round_input not in ("embedding", "state_plus_embedding"):
            raise ValueError("Unknown first-round input rule")
        if type(feedback_to_candidate) is not bool:
            raise ValueError("feedback_to_candidate must be a boolean")
        super().__init__(
            config,
            repeats,
            first_round_kv=False,
            discoloop=handoff != "plain",
            discoloop_alpha=1.0,
            discoloop_temperature=1.0,
            discoloop_eps=eps,
        )
        self.handoff, self.score_clip, self.eps = handoff, score_clip, eps
        self.first_round_input = first_round_input
        self.feedback_to_candidate = feedback_to_candidate
        if handoff in ("additive", "plain"):
            return
        self.handoff_beta_logit = nn.Parameter(
            torch.tensor(math.log(initial_beta / (1 - initial_beta)))
        )
        if handoff == "content":
            self.handoff_query = nn.Parameter(torch.zeros(config.n_embd))
        elif handoff in ("invariant", "state-mlp"):
            generator = torch.Generator().manual_seed(score_seed)
            score_inputs = 2 * config.n_embd if handoff == "state-mlp" else len(FEATURES)
            bound = 1 / math.sqrt(score_inputs)
            first = torch.empty(score_hidden, score_inputs)
            first.uniform_(-bound, bound, generator=generator)
            self.handoff_score_in = nn.Parameter(first)
            self.handoff_score_in_bias = nn.Parameter(torch.zeros(score_hidden))
            # Zero output layer: the score starts at zero, as InfiLoop's zero query.
            self.handoff_score_out = nn.Parameter(torch.zeros(score_hidden))
            self.handoff_score_out_bias = nn.Parameter(torch.zeros(()))

    @property
    def beta(self):
        return torch.sigmoid(_loss_dtype(self.handoff_beta_logit))

    def _decoded_probabilities(self, state):
        """Decode for scoring without projecting probabilities back to embeddings."""
        logits = _readout(self, state)
        return (_loss_dtype(logits) / self.discoloop_temperature).softmax(-1)

    def _decoded_feedback(self, state):
        """Return the full-vocabulary decode and q(state), both float32."""
        probability = self._decoded_probabilities(state)
        soft = F.linear(probability, self.transformer.wte.weight.T)
        return probability, rms_normalize(soft, self.discoloop_eps)

    def features(self, candidate, previous_state, probability, previous_probability):
        """Relabeling-invariant score inputs, [B, T, len(FEATURES)] in float32."""
        entropy = -torch.special.xlogy(probability, probability).sum(-1)
        entropy = entropy / math.log(probability.shape[-1])
        change = (candidate - previous_state).norm(dim=-1)
        relative = change / (previous_state.norm(dim=-1) + self.eps)
        cosine = F.cosine_similarity(candidate, previous_state, dim=-1, eps=self.eps)
        agreement = (probability * previous_probability).sum(-1)
        values = (entropy, probability.amax(-1), agreement, relative, cosine)
        return torch.stack(values, dim=-1)

    def score(self, candidate, previous_state, probability, previous_probability):
        if self.handoff == "ema":
            return candidate.new_zeros(candidate.shape[:-1])
        if self.handoff == "content":
            return (candidate * _loss_dtype(self.handoff_query)).sum(-1)
        if self.handoff == "state-mlp":
            phi = torch.cat((previous_state, candidate), dim=-1)
        else:
            phi = self.features(candidate, previous_state, probability, previous_probability)
        hidden = F.gelu(
            F.linear(
                phi, _loss_dtype(self.handoff_score_in), _loss_dtype(self.handoff_score_in_bias)
            )
        )
        return (hidden * _loss_dtype(self.handoff_score_out)).sum(-1) + _loss_dtype(
            self.handoff_score_out_bias
        )

    def initial_state(self, injection):
        state = (injection, injection, injection.new_ones((*injection.shape[:-1], 1)))
        probability = None
        if self.handoff == "invariant":
            probability = (
                self._decoded_feedback(injection)[0]
                if self.feedback_to_candidate
                else self._decoded_probabilities(injection)
            )
        state = (*state, probability)
        return (*state, injection) if self.first_round_input == "embedding" else state

    def aggregate_round(self, state, injection, *, first_round=False):
        """One round; returns the new state and gamma = exp(s) / D, both float32."""
        h, numerator, denominator, previous_probability = state[:4]
        block_input = (
            injection if first_round and self.first_round_input == "embedding" else h + injection
        )
        output = self._round(block_input)
        if self.feedback_to_candidate:
            probability, feedback = self._decoded_feedback(output)
            candidate = rms_normalize(_loss_dtype(output) + feedback, self.eps)
        else:
            probability = self._decoded_probabilities(output)
            candidate = rms_normalize(output, self.eps)
        weight = (
            self.score(candidate, h, probability, previous_probability)
            .clamp(-self.score_clip, self.score_clip)
            .exp()
            .unsqueeze(-1)
        )
        beta = self.beta
        numerator = beta * numerator + weight * candidate
        denominator = beta * denominator + weight
        new = (numerator / denominator, numerator, denominator)
        carried = probability if self.handoff == "invariant" else None
        return (*new, carried, *state[4:]), weight / denominator

    def _trajectory_round(self, state, injection, *, first_round):
        if self.first_round_input == "embedding":
            return self.aggregate_round(state, injection, first_round=first_round)
        # Historical diagnostic hooks accept exactly (state, injection).
        return self.aggregate_round(state, injection)

    def run(self, tokens, positions=None, rounds=None, state=None, carry=True):
        """Execute ``rounds`` rounds from ``state`` (fresh when None).

        Returns logits at ``positions`` and the carry for a further call (None when
        ``carry`` is False). This method never detaches; the historical segmented
        objective explicitly detaches in its caller.
        """
        rounds = self.repeats if rounds is None else rounds
        if rounds < 1 or not len(self.transformer.h):
            raise ValueError("at least one round and one block are required")
        if self.handoff in ("additive", "plain"):
            x = self._embedded(tokens) if state is None else state[0]
            for index in range(rounds):
                output = self._round(x)
                if index < rounds - 1 or carry:
                    x = output if self.handoff == "plain" else self._discoloop_boundary(output)
            return _readout(self, output, positions), ((x,) if carry else None)
        fresh = state is None
        if not fresh and self.first_round_input == "embedding":
            if len(state) != 5:
                raise ValueError("Embedding-first continuation requires its original injection")
            injection = state[4]
        else:
            injection = _loss_dtype(self._embedded(tokens))
        state = self.initial_state(injection) if fresh else state
        for index in range(rounds):
            state, _ = self._trajectory_round(state, injection, first_round=fresh and index == 0)
        return _readout(self, state[0], positions), (state if carry else None)

    def forward(self, tokens, positions=None):
        return self.run(tokens, positions, self.repeats, carry=False)[0]

    def forward_with_states(self, tokens, positions=None):
        raise NotImplementedError("Use run(); first/final states are not defined per handoff")

    @torch.no_grad()
    def trace(self, tokens, positions, rounds):
        """Per-round readout and carry diagnostics for one fresh call, in eval use.

        ``state`` is the carry after each round (additive: H + q(H); aggregation: h).
        Readout at round r equals a fresh call with r rounds. Returns logits
        [R, B, P, V], relative carry change [R, B, T] and gamma [R, B, T] (NaN for
        the additive handoff).
        """
        batch, length = tokens.shape
        logits, change, gamma = [], [], []
        if self.handoff in ("additive", "plain"):
            x = self._embedded(tokens)
            for _ in range(rounds):
                output = self._round(x)
                logits.append(_readout(self, output, positions))
                new = output if self.handoff == "plain" else self._discoloop_boundary(output)
                change.append(
                    _loss_dtype(new - x).norm(dim=-1) / (_loss_dtype(x).norm(dim=-1) + self.eps)
                )
                gamma.append(torch.full((batch, length), math.nan, device=tokens.device))
                x = new
        else:
            injection = _loss_dtype(self._embedded(tokens))
            state = self.initial_state(injection)
            for index in range(rounds):
                previous = state[0]
                state, accepted = self._trajectory_round(state, injection, first_round=index == 0)
                logits.append(_readout(self, state[0], positions))
                change.append(
                    (state[0] - previous).norm(dim=-1) / (previous.norm(dim=-1) + self.eps)
                )
                gamma.append(accepted.squeeze(-1))
        return torch.stack(logits), torch.stack(change), torch.stack(gamma)


def handoff_parameter_count(handoff, width, hidden=16):
    if handoff in ("additive", "plain"):
        return 0
    if handoff == "ema":
        return 1
    if handoff == "content":
        return width + 1
    if handoff == "state-mlp":
        return (2 * width + 2) * hidden + 2
    return hidden * len(FEATURES) + hidden + hidden + 1 + 1


def detach_state(state):
    if state is None:
        return None
    return tuple(None if value is None else value.detach() for value in state)


def sample_schedule(rng, first, second, segments):
    """Round counts per supervised segment, drawn uniformly from inclusive ranges.

    The ranges come from the configuration and never from a row's hop count; one
    schedule is shared by the whole batch.
    """
    if segments < 1:
        raise ValueError("at least one segment is required")
    ranges = [first] + [second] * (segments - 1)
    return [int(rng.integers(low, high + 1)) for low, high in ranges]


def deep_supervision_loss(model, tokens, positions, labels, schedule):
    """Mean answer/EOS cross-entropy at the end of every segment.

    The carry is detached between segments, so gradients run through all rounds of
    a segment and never across segments. No other loss term exists.
    """
    state, losses = None, []
    for index, rounds in enumerate(schedule):
        last = index == len(schedule) - 1
        logits, state = model.run(tokens, positions, rounds, state, carry=not last)
        losses.append(F.cross_entropy(_loss_dtype(logits).flatten(0, 1), labels.flatten()))
        state = detach_state(state)
    return torch.stack(losses).mean(), [loss.detach() for loss in losses]
