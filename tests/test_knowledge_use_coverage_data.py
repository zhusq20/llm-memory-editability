from collections import Counter

import pytest

from llm_memory_editability import knowledge_use_coverage_data as d


def test_natural_retention_groups_all_templates_by_country_without_predictions():
    records = [
        {"id": "ret/capital_rev/Paris", "gold": "France", "strict": False},
        {"id": "ret/city_country/Lyon", "gold": "France", "strict": True},
        {"id": "ret/capital/Paris", "gold": "Paris"},
        {"id": "ret/currency/France", "gold": "euro"},
        {"id": "ret/city_country/Berlin", "gold": "Germany"},
        {"id": "ret/currency/Germany", "gold": "euro"},
        {"id": "ret/capital/Unknown", "gold": "Unknown"},
        {"id": "unknown/0", "gold": "France"},
        {"id": "nat2/0/Paris", "gold": "euro"},
        {"id": "nat2/1/Paris", "gold": "euro"},
        {"id": "nat2short/Paris", "gold": "euro"},
    ]
    clusters = d.natural_retention_clusters(records)
    assert len({clusters[r["id"]] for r in records[:4]}) == 1
    assert clusters[records[4]["id"]] == clusters[records[5]["id"]]
    assert clusters[records[0]["id"]] != clusters[records[4]["id"]]
    assert clusters[records[6]["id"]] is None and clusters[records[7]["id"]] is None
    assert all(clusters[r["id"]] == clusters[records[0]["id"]] for r in records[8:])
    preserved = [{**r, "id": str(i), "original_source_id": r["id"]} for i, r in enumerate(records)]
    assert list(d.natural_retention_clusters(preserved).values()) == list(clusters.values())
    conflicting = records + [{"id": "ret/city_country/Paris", "gold": "Other"}]
    assert d.natural_retention_clusters(conflicting)["ret/capital/Paris"] is None


def test_worlds_separate_new_entities_and_share_only_known_background():
    dev = d.make_world("development", 32, 7)
    formal = d.make_world("formal", 128, 7)
    assert dev == d.make_world("development", 32, 7)
    dev_names = {name for p in dev.pairs for name in (p.organization, p.office)}
    formal_names = {name for p in formal.pairs for name in (p.organization, p.office)}
    assert len(dev_names) == 64 and len(formal_names) == 256
    assert not dev_names & formal_names
    assert not {p.id for p in dev.pairs} & {p.id for p in formal.pairs}
    assert {p.city for p in dev.pairs} & {p.city for p in formal.pairs}
    assert len(formal.manifest()["rewrite_permutation"]) == 128
    for world in (dev, formal):
        assert all(p.reference_city != p.city for p in world.pairs)
        assert all(p.reference_country == p.country for p in world.pairs)


@pytest.mark.parametrize("count", [0, 1, 3, 130])
def test_invalid_pair_counts_cannot_break_rewrite_permutation(count):
    with pytest.raises(ValueError):
        d.make_world(pair_count=count)


def test_training_generation_is_independent_of_evaluation(monkeypatch):
    world = d.make_world(pair_count=4)

    def forbidden(*args, **kwargs):
        raise AssertionError("training accessed evaluation")

    monkeypatch.setattr(d, "evaluation_rows", forbidden)
    monkeypatch.setattr(d, "TEST_ATOMIC", None)
    rows = {a: d.training_rows(world, a) for a in "ABC"}
    assert {a: len(r) for a, r in rows.items()} == {"A": 8, "B": 64, "C": 64}
    assert Counter(r["category"] for r in rows["C"]) == {
        "atomic": 32,
        "composition": 8,
        "judgment": 16,
        "comparison": 8,
    }
    for p in world.pairs:
        contexts = {r["context"] for arm in rows.values() for r in arm if r["pair_id"] == p.id}
        assert len(contexts) == 1
    for row in rows["C"]:
        new_dependencies = [f for f in row["fact_ids"] if not f.startswith("background/")]
        assert len(new_dependencies) == 1  # Dual-new composition is never supervised.
        assert set(row["fact_ids"]) <= set(row["context_fact_ids"])
        assert "assistant" not in row  # Gold is not mislabelled as a teacher trajectory.


def test_heldout_questions_and_all_modes_are_explicit():
    world = d.make_world(pair_count=4)
    audit = d.audit_world(world)
    assert audit["passed"]
    rows = d.evaluation_rows(world)
    assert len(rows) == 4 * 32
    counts = Counter(r["category"] for r in rows)
    assert counts == {
        "atomic_seen": 16,
        "atomic_unseen": 32,
        "seen_use_new_wording": 56,
        "double_new_composition": 8,
        "heldout_relation_composition": 8,
        "heldout_use_category": 8,
    }
    modes = Counter(r["variant"] for r in rows)
    assert modes["direct"] == modes["step_by_step"] == 64
    for row in rows:
        if row["category"] == "double_new_composition":
            assert len(row["fact_ids"]) == 2
            assert "{answer0}" in row["chain"][1]["question"]
            assert "gold" not in row["chain"][0]
            assert world.pairs[0].office not in row["chain"][1]["question"]


def test_rewrite_changes_only_assignments_and_dependent_answers():
    world = d.make_world(pair_count=4)
    rewrite = d.rewrite_training_rows(world)
    assert len(rewrite) == 4
    for i, row in enumerate(rewrite):
        assert row["gold"] == world.pairs[i ^ 1].office
        assert row["previous_gold"] == world.pairs[i].office
        assert row["answer_changed"]
        assert row["context_fact_ids"] == [world.pairs[i].assignment_id]
        assert "base city is" not in row["context"]
        assert "located in" not in row["context"]
        assert row["cluster_id"] == rewrite[i ^ 1]["cluster_id"]
    changed = d.evaluation_rows(world, "rewrite")
    for row in changed:
        i = next(i for i, p in enumerate(world.pairs) if p.id == row["pair_id"])
        if row["category"] == "double_new_composition":
            assert row["gold"] == world.pairs[i ^ 1].city
            assert row["previous_gold"] == world.pairs[i].city
            assert row["answer_changed"]
        if row["category"] == "heldout_relation_composition":
            assert row["gold"] == world.pairs[i ^ 1].country
            assert row["answer_changed"]
    retention = d.retention_rows(world, "rewrite")
    locations = [r for r in retention if r["category"] == "retention_unchanged_new_location"]
    assert len(locations) == 4 * 4
    assert all(not row["answer_changed"] for row in locations)


def test_replay_and_independent_retention_cannot_overlap_or_preserve_edits():
    replay, retention = d.replay_rows(), d.general_retention_rows()
    d.validate_retention_split(replay, retention)
    with pytest.raises(ValueError, match="overlap"):
        d.validate_retention_split(replay, [dict(replay[0])])
    with pytest.raises(ValueError, match="changed target"):
        d.validate_retention_split(replay, retention, target_fact_ids=retention[0]["fact_ids"])
    assert not {r["gold"] for r in replay} == {r["gold"] for r in retention}


def test_inverse_assignment_keeps_question_fixed_and_uses_new_owner_context():
    world = d.make_world(pair_count=4)
    by_stage = {
        stage: {
            (row["pair_id"], row["mode"]): row
            for row in d.evaluation_rows(world, stage)
            if row["category"] == "heldout_use_category"
        }
        for stage in ("addition", "rewrite")
    }
    assert by_stage["addition"].keys() == by_stage["rewrite"].keys()
    for i, original in enumerate(world.pairs):
        new_owner = world.pairs[i ^ 1]
        for mode in ("direct", "step_by_step"):
            before = by_stage["addition"][original.id, mode]
            after = by_stage["rewrite"][original.id, mode]
            assert before["question"] == after["question"]
            assert original.office in after["question"]
            assert new_owner.office not in after["question"]
            assert before["gold"] == after["previous_gold"] == original.organization
            assert after["gold"] == new_owner.organization and after["answer_changed"]
            assert after["fact_ids"] == [new_owner.assignment_id]
            assert set(after["fact_ids"]) <= set(after["context_fact_ids"])
            assert new_owner.assignment_id in after["context_fact_ids"]
            assert original.assignment_id not in after["context_fact_ids"]
            assert (
                f"{new_owner.organization}'s assigned office is {original.office}."
                in after["context"]
            )
            assert after["pair_id"] == original.id
            assert after["cluster_id"] == f"{world.cohort}/swap-{i // 2:03d}"


def test_all_evaluation_pairs_keep_questions_fixed_and_previous_gold_matches():
    world = d.make_world(pair_count=4)
    before = {
        row["id"].replace("/addition/", "/state/"): row
        for row in d.evaluation_rows(world, "addition")
    }
    after = {
        row["id"].replace("/rewrite/", "/state/"): row
        for row in d.evaluation_rows(world, "rewrite")
    }
    assert before.keys() == after.keys()
    assert {r["category"] for r in before.values()} == {
        "atomic_seen",
        "atomic_unseen",
        "seen_use_new_wording",
        "double_new_composition",
        "heldout_relation_composition",
        "heldout_use_category",
    }
    for identity, old in before.items():
        new = after[identity]
        assert new["question"] == old["question"]
        assert new["previous_gold"] == old["gold"]
        assert new["answer_changed"] == (new["gold"] != old["gold"])
        assert set(new["fact_ids"]) <= set(new["context_fact_ids"])


def test_c_judgments_cover_both_polarities_and_rewrite_evaluation_flips_only_assignment():
    world = d.make_world(pair_count=4)
    trained = [r for r in d.training_rows(world, "C") if r["category"] == "judgment"]
    for pair in world.pairs:
        for relation in ("assignment", "location"):
            examples = [
                r for r in trained if r["pair_id"] == pair.id and f"/{relation}-judge-" in r["id"]
            ]
            assert len(examples) == 2 and {r["gold"] for r in examples} == {"yes", "no"}
    for row in d.evaluation_rows(world, "rewrite"):
        if "/assignment-judge-" in row["id"]:
            true_candidate = "/assignment-judge-true/" in row["id"]
            assert row["gold"] == ("no" if true_candidate else "yes")
            assert row["previous_gold"] == ("yes" if true_candidate else "no")
            assert row["answer_changed"]
        elif "/location-judge-" in row["id"]:
            assert row["gold"] == row["previous_gold"]
            assert not row["answer_changed"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Paris.", True),
        ("PARIS", True),
        ("Answer: Paris", True),
        ("I considered Berlin.\nFinal answer: Paris", True),
        ("Paris is an intermediate city.\nFinal answer: Berlin", False),
        ("Final answer: Paris or Berlin", False),
        ("Paris\nBerlin", False),
        ("The correct answer is Paris.", False),
        ("", False),
    ],
)
def test_scorer_uses_final_answer_not_reasoning_mentions(text, expected):
    row = {"gold": "Paris", "aliases": ["Paris"]}
    assert d.score_answer(row, text)["correct"] is expected


def test_boolean_and_alias_scoring_has_no_substring_success():
    row = {"gold": "yes", "aliases": ["yes"]}
    assert d.score_answer(row, "Yes.")["correct"]
    assert not d.score_answer(row, "yesterday")["correct"]
    result = d.score_answer(row, "yes or no")
    assert not result["correct"] and result["conflicting_candidates"] == ["no"]
    row = {"gold": "Brasília", "aliases": ["Brasília", "Brasilia"]}
    assert d.score_answer(row, "final ANSWER: Brasilia")["correct"]
    assert not d.score_answer(row, "Brasilia or Paris")["correct"]
    answer, status = d.extract_answer(
        "My intermediate answer is wrong.\nFinal answer: Belalen Field Office"
    )
    assert answer == "Belalen Field Office" and status == "explicit_answer"


def test_formal_world_complete_audit():
    result = d.audit_world(d.make_world("formal", 128))
    assert result["unique_training_rows"] == {"A": 256, "B": 2048, "C": 2048}
    assert result["evaluation_rows_per_stage"] == 4096
