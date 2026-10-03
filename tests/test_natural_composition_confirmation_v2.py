"""Scientific contracts: isolated facts, novel endpoints, matched atomic training."""

import torch

from llm_memory_editability import natural_composition_confirmation_v2 as exp


class CharTokenizer:
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]


def test_nested_support_role_isolation_and_atomic_exposure():
    cfg = dict(
        training_seed=971001,
        steps=16,
        atomic_per_step=16,
        compositions_per_step=4,
        arms=["low", "high"],
    )
    for seed in (970001, 970101, 970102, 970103):
        world = exp.build_world(seed)
        audit = exp.audit(world, exp.streams(world, cfg), exp.records(CharTokenizer(), world))
        assert audit["familiar_fact_role_coverage_high"] == 64
        assert audit["exposures"]["high"]["atomic"] == audit["exposures"]["low"]["atomic"]
        assert len(audit["exposures"]["high"]["atomic"]) == 128


def test_answer_scoring_and_heldout_questions():
    world = exp.build_world(970001)
    assert exp.base.grade("Paris.\nExtra text", "Paris")
    assert not exp.base.grade("Paris or Rome", "Paris")
    for row in world["chains"]:
        assert row["answer"] not in exp.prompt(row)
        assert row["bridge"] not in exp.prompt(row)
        assert exp.prompt(row, 4) not in [exp.prompt(row, v) for v in range(4)]


def test_atomic_stream_weight_is_independent_of_composition_length():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = torch.nn.Parameter(torch.randn(4, 8, 6))

        def forward(self, **kwargs):
            return type("Output", (), dict(logits=self.values))()

    torch.manual_seed(1)
    model = Toy()
    labels = torch.randint(0, 6, (4, 8))
    inputs = dict(input_ids=labels, attention_mask=torch.ones_like(labels), labels=labels)
    loss = exp.stream_loss(model, inputs, 2, 0.8)
    first = torch.autograd.grad(loss, model.values, retain_graph=True)[0][:2].clone()
    inputs["labels"] = labels.clone()
    inputs["labels"][2:, 4:] = -100
    loss = exp.stream_loss(model, inputs, 2, 0.8)
    second = torch.autograd.grad(loss, model.values)[0][:2]
    assert torch.equal(first, second)


def test_all_relation_programs_practiced_in_both_support_sets():
    for seed in (970001, 970101, 970102, 970103):
        world = exp.build_world(seed)
        for arm in ("low", "high"):
            counts = __import__("collections").Counter(
                (world["chains"][i]["first"], world["chains"][i]["second"])
                for i in world[arm + "_support"]
            )
            assert len(counts) == 4
            assert len(set(counts.values())) == 1
