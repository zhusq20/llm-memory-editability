"""Guard knowledge leakage and paired training exposure, not implementation details."""

from collections import Counter

import torch

from llm_memory_editability import small_lm_composition as exp


class CharacterTokenizer:
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]


def test_split_and_training_information():
    config = dict(target_triples=64, background_triples=16, world_seed=1)
    world = exp.build_world(config)
    names = [r[k] for r in world for k in ("head", "bridge")]
    assert len(set(names)) == len(names)
    records = exp.corpus(CharacterTokenizer(), world)
    assert all(hop < 2 or world[i]["split"] == "background" for i, hop, _ in records)
    for row in exp.evaluation_rows(world):
        if row["hop"] == 2:
            assert row["bridge"] not in row["prompt"] and row["city"] not in row["prompt"]
    for split in ("target", "background"):
        counts = Counter(r["city"] for r in world if r["split"] == split)
        assert len(set(counts.values())) == 1


def test_paired_inputs_labels_and_padding():
    cfg = dict(
        target_triples=4,
        background_triples=2,
        world_seed=4,
        training_seed=3,
        steps=4,
        atomic_pairs_per_step=2,
        background_compositions_per_step=1,
    )
    world = exp.build_world(cfg)
    records = exp.corpus(CharacterTokenizer(), world)
    for batch in exp.stream(cfg, world):
        separate = exp.documents(records, batch, "separate")
        linked = exp.documents(records, batch, "linked")
        assert [t for d in separate for t in d["ids"]] == [t for d in linked for t in d["ids"]]
        assert [t for d in separate for t in d["labels"]] == [
            t for d in linked for t in d["labels"]
        ]
        for docs in (separate, linked):
            x = exp.batch_tensors(docs, 1, "cpu")
            assert torch.all(x["labels"][x["attention_mask"] == 0] == -100)
            assert torch.all(x["labels"][:, 0] == -100)
            assert int(x["labels"][:, 1:].ne(-100).sum()) == sum(
                sum(v != -100 for v in d["labels"]) for d in docs
            )


def test_scoring_does_not_accept_answer_substrings_or_insert_a_bridge():
    assert exp.grade(" Paris.\n", "Paris")
    assert not exp.grade("Paris or London.", "Paris")
    assert not exp.grade("The answer is Paris.", "Paris")
    rows = [
        dict(id=i, split="target", hop=h, correct=ok)
        for i, values in enumerate([(True, True, False), (True, False, True)])
        for h, ok in enumerate(values)
    ]
    rows.append(dict(id=2, split="background", hop=2, correct=True))
    scores = exp.score(rows)
    assert scores["two_hop"] == 0.5
    assert scores["both_atomic_coverage"] == 0.5
    assert scores["two_hop_if_both_atomic"] == 0.0
