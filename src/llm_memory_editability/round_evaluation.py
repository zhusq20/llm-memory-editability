"""Descriptive, paired comparisons of fixed checkpoints across inference rounds."""

from __future__ import annotations

import numpy as np


def correctness_summary(correct, reference_round=4):
    """Counts use every item; first-correct rounds are retrospective, not halting rules."""
    correct = np.asarray(correct, dtype=bool)
    if correct.ndim != 2 or not len(correct) or not 1 <= reference_round <= correct.shape[1]:
        raise ValueError("Expected nonempty [items, rounds] and an available reference round")
    n, rounds = correct.shape
    ref = correct[:, reference_round - 1]
    retained = (ref[:, None] & correct).sum(0)
    gained = (~ref[:, None] & correct).sum(0)
    lost = (ref[:, None] & ~correct).sum(0)
    stable = np.logical_and.accumulate(correct[:, ::-1], axis=1)[:, ::-1]

    def first_hist(mask):
        first = np.where(mask.any(1), mask.argmax(1) + 1, 0)
        return {str(r): int((first == r).sum()) for r in range(rounds + 1)}

    return {
        "n": n,
        "correct_n": correct.sum(0).tolist(),
        "accuracy": correct.mean(0).tolist(),
        "reference_round": reference_round,
        "reference_correct_n": int(ref.sum()),
        "retained_vs_reference_n": retained.tolist(),
        "retention_vs_reference": (retained / ref.sum()).tolist() if ref.any() else [None] * rounds,
        "gained_vs_reference_n": gained.tolist(),
        "lost_vs_reference_n": lost.tolist(),
        "net_gain_vs_reference_n": (gained - lost).tolist(),
        "right_to_wrong_n": [0] + (correct[:, :-1] & ~correct[:, 1:]).sum(0).tolist(),
        "wrong_to_right_n": [0] + (~correct[:, :-1] & correct[:, 1:]).sum(0).tolist(),
        "first_correct_round_histogram": first_hist(correct),
        "first_correct_through_final_round_histogram": first_hist(stable),
        "histogram_zero": "Never correct / never continuously correct through the final round",
        "first_round_policy": "Uses labels retrospectively; not an executable stopping rule",
    }


def paired_summary(candidate, baseline):
    candidate, baseline = np.asarray(candidate, bool), np.asarray(baseline, bool)
    if candidate.shape != baseline.shape or candidate.ndim != 2 or not len(candidate):
        raise ValueError("Paired comparisons require the same items and rounds")
    gain = (candidate & ~baseline).sum(0)
    loss = (~candidate & baseline).sum(0)
    return {
        "n": len(candidate),
        "candidate_only_correct_n": gain.tolist(),
        "baseline_only_correct_n": loss.tolist(),
        "both_correct_n": (candidate & baseline).sum(0).tolist(),
        "neither_correct_n": (~candidate & ~baseline).sum(0).tolist(),
        "baseline_never_correct_n": int((~baseline.any(1)).sum()),
        "correct_when_baseline_never_correct_n": (candidate & ~baseline.any(1)[:, None])
        .sum(0)
        .tolist(),
        "difference_pp": (100 * (gain - loss) / len(candidate)).tolist(),
    }


def inference_matmul_flops(width, vocab, length, rounds, rule=None, layers=1):
    """Per item, two complete native forwards for answer then generated-answer EOS.

    Dense padded causal attention, GPT-2 MLP expansion 4, two readout positions.
    Excludes softmax, normalization, reductions, elementwise work and memory traffic.
    """
    block = 24 * length * width**2 + 4 * length**2 * width
    feedback = 4 * length * width * vocab
    extra = 0 if rule is None else 2 * length * width * vocab
    if rule == "readiness":
        extra += 6 * length * width**2 + 2 * length**2 * width
    readout = 4 * width * vocab
    return 2 * (rounds * layers * block + (rounds - 1) * (feedback + extra) + readout)


def round_logits(model, tokens, positions, max_rounds):
    """Read native raw-H outputs along one trajectory, preserving rule history."""
    from .self_decode_alignment import _loss_dtype, _readout

    x = model._embedded(tokens)
    previous_probability = previous_ready = None
    rule = getattr(model, "rule", None)
    for index in range(max_rounds):
        final = index == max_rounds - 1
        attention = model.round_attention(x) if rule == "readiness" and not final else None
        h = model._round(x)
        yield _readout(model, h, positions)
        if final:
            break
        if rule is None:
            x = model._discoloop_boundary(h)
        else:
            probability = _loss_dtype(_readout(model, h)).softmax(-1)
            update, previous_ready = model.rule_gates(
                tokens, probability, previous_probability, attention, previous_ready
            )
            previous_probability = probability
            x = model.handoff(x, h, update.unsqueeze(-1))
