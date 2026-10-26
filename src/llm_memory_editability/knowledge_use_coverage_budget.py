"""Deterministic token budgets for the authorized coverage comparison."""

from __future__ import annotations

import copy
import random


class TokenStream:
    """Cycle through shuffled examples; carry split answer prefixes into context."""

    def __init__(self, samples, seed):
        if not samples or any(not row["answer"] for row in samples):
            raise ValueError("Every stream must have nonempty answers")
        self.samples = samples
        self.rng = random.Random(seed)
        self.order = []
        self.position = 0
        self.pending = None
        self.offset = 0
        self.examples_started = 0
        self.examples_completed = 0
        self.tokens = 0

    def take(self, tokens):
        if tokens <= 0:
            raise ValueError("Token target must be positive")
        result = []
        while tokens:
            if self.pending is None:
                if self.position == len(self.order):
                    self.order = list(range(len(self.samples)))
                    self.rng.shuffle(self.order)
                    self.position = 0
                self.pending = self.samples[self.order[self.position]]
                self.position += 1
                self.offset = 0
                self.examples_started += 1
            row = self.pending
            take = min(tokens, len(row["answer"]) - self.offset)
            prefix = row["answer"][: self.offset]
            result.append(
                {
                    **row,
                    "student": row["student"] + prefix,
                    "teacher": row["teacher"] + prefix,
                    "answer": row["answer"][self.offset : self.offset + take],
                    "original_answer_offset": self.offset,
                    "original_answer_length": len(row["answer"]),
                }
            )
            self.offset += take
            self.tokens += take
            tokens -= take
            if self.offset == len(row["answer"]):
                self.pending = None
                self.examples_completed += 1
        return result

    def state_dict(self):
        return copy.deepcopy(
            {
                "rng": self.rng.getstate(),
                "order": self.order,
                "position": self.position,
                "pending": self.pending,
                "offset": self.offset,
                "examples_started": self.examples_started,
                "examples_completed": self.examples_completed,
                "tokens": self.tokens,
            }
        )

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        self.rng.setstate(state.pop("rng"))
        for key, value in state.items():
            setattr(self, key, value)


def microbatch_weight(objective, microbatch, all_samples):
    if objective == "pd":
        return len(microbatch) / len(all_samples)
    if objective == "sft":
        return sum(len(s["answer"]) for s in microbatch) / sum(
            len(s["answer"]) for s in all_samples
        )
    raise ValueError(objective)


def pressure_example(samples):
    """Synthetic token shape combining observed student/teacher/answer maxima.

    Prefixes are trimmed from actual token sequences; this is discarded
    engineering data, not a scientific QA or a new claim about source content.
    """
    answer_row = max(samples, key=lambda r: len(r["answer"]))
    answer = list(answer_row["answer"])
    result = {"id": "engineering-shape-bound", "answer": answer}
    provenance = {
        "kind": "synthetic_joint_shape_bound",
        "answer_source": answer_row["id"],
        "answer_tokens": len(answer),
    }
    for side in ("student", "teacher"):
        row = max(samples, key=lambda r: len(r[side]) + len(r["answer"]))
        total = len(row[side]) + len(row["answer"])
        prefix = total - len(answer)
        if prefix < 1:
            raise ValueError("Pressure example requires a nonempty prompt")
        result[side] = list(row[side][:prefix])
        if len(result[side]) != prefix:
            raise AssertionError("Source cannot supply pressure prefix")
        provenance[f"{side}_source"] = row["id"]
        provenance[f"{side}_total_tokens"] = total
    return result, provenance


def learning_rate_scale(tokens, maximum, warmup_fraction=0.02):
    if maximum <= 0 or tokens < 0 or tokens > maximum:
        raise ValueError("Invalid learning-rate token position")
    warmup = max(1, int(maximum * warmup_fraction))
    if tokens < warmup:
        return max(tokens, 1) / warmup
    return max(0.0, (maximum - tokens) / max(1, maximum - warmup))


def select_development_node(nodes, scores, target=0.8):
    """Only development atomic recall can choose a common budget."""
    for node in nodes:
        values = scores.get(str(node), {})
        if set(values) != {"A", "B", "C"}:
            raise ValueError("Incomplete development matrix")
        if all(values[a] >= target for a in ("A", "B", "C")):
            return {"tokens": node, "target_met": True}
    return {"tokens": nodes[-1], "target_met": False}
