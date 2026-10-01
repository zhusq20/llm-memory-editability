"""Contracts affecting causal interpretation of the full-LM follow-up."""

from collections import Counter

import torch

from llm_memory_editability import small_lm_composition as base
from llm_memory_editability import small_lm_followup as exp


class CharacterTokenizer:
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]


def config():
    cfg = base.read("configs/small-lm-composition-development-v2.json")
    return dict(cfg, steps=10, edit_steps=10)


def test_matched_training_information_and_rehearsal():
    cfg = config()
    world = base.build_world(cfg)
    records = base.corpus(CharacterTokenizer(), world)
    for batch in exp.pairing_stream(cfg, world):
        signatures = []
        for arm in cfg["arms"]:
            docs = exp.training_documents(records, batch, arm)
            signatures.append(
                (
                    Counter(t for d in docs for t in d["ids"]),
                    Counter(t for d in docs for t in d["labels"] if t != -100),
                )
            )
            tensors = base.batch_tensors(docs, 1, "cpu")
            assert torch.all(tensors["labels"][tensors["attention_mask"] == 0] == -100)
            assert torch.all(tensors["labels"][:, 0] == -100)
        assert signatures[0] == signatures[1] == signatures[2]
        assert sorted(batch["second_order"]) == list(range(8))
        for (i, _), j in zip(batch["pairs"], batch["second_order"], strict=True):
            k = batch["pairs"][j][0]
            assert i != k and world[i]["city"] != world[k]["city"]
    assert all(h < 2 or world[i]["split"] == "background" for i, h, _ in records)


def test_updates_change_only_designated_cities_and_never_train_holdout():
    cfg = config()
    world = base.build_world(cfg)
    edit = exp.edit_design(cfg, world)
    ids = set(edit["edited_ids"])
    assert len(ids) == 32
    assert Counter(world[i]["city"] for i in ids) == Counter(edit["world"][i]["city"] for i in ids)
    for before, after in zip(world, edit["world"], strict=True):
        assert before["head"] == after["head"] and before["bridge"] == after["bridge"]
        assert (before["city"] != after["city"]) == (before["id"] in ids)
    for batch in edit["stream"]:
        for i, hop, _ in batch:
            assert i in ids or world[i]["split"] == "background"
            assert hop < 2 or world[i]["split"] == "background"


def test_autonomous_call_uses_actual_output_including_wrong_bridges():
    world = base.build_world(config())
    first = dict(world[0], hop=0, view=0, text=" Wrong Person.\n", tokens=[1, 2], correct=False)
    rows = exp.autonomous_rows(world, [first])
    assert "Wrong Person" in rows[0]["prompt"]
    assert world[0]["bridge"] not in rows[0]["prompt"]
    assert rows[0]["answer"] == world[0]["city"]
    assert not rows[0]["first_correct"]


def test_new_gold_shared_by_replay_and_update_and_views_are_held_out():
    cfg = config()
    world = base.build_world(cfg)
    edit = exp.edit_design(cfg, world)
    for row in exp.evaluation_rows(edit["world"], extras=True):
        if row["view"] == 1:
            assert row["prompt"] not in [base.prompt(row, row["hop"], v) for v in range(4)]
        if row["hop"] == 2:
            assert row["city"] not in row["prompt"] and row["bridge"] not in row["prompt"]
        if row["id"] in edit["edited_ids"] and row["hop"] > 0:
            assert row["answer"] != world[row["id"]]["city"]
