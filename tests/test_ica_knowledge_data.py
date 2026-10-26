from dataclasses import asdict, replace

import pytest

from llm_memory_editability import ica_knowledge_data as data
from llm_memory_editability import read_interface_data as old


@pytest.fixture
def facts():
    known = {q["qid"]: True for q in old.screening_queries()}
    return old.build_facts(known, seed=7)[0]


def test_selection_preserves_subject_groups_original_ids_and_splits(facts):
    selected = data.select_facts(facts)
    assert selected == data.select_facts(reversed(facts))
    assert len(selected.main) == 48
    assert len(selected.calibration) == 10
    main_subjects = {f.h for f in selected.main}
    assert not main_subjects & {f.h for f in selected.calibration}
    assert {f.fid for f in selected.main} == {f.fid for f in facts if f.h in main_subjects}
    originals = {f.fid: f for f in facts}
    assert all(f is originals[f.fid] for f in (*selected.main, *selected.calibration))
    assert {f.split for f in selected.main} == {"adapt", "test"}
    assert all(f.split == "calibration" for f in selected.calibration)
    assert selected.main != data.select_facts(facts, seed=data.DEFAULT_SEED + 1).main


def test_selection_does_not_depend_on_values_or_retention_queries(facts, monkeypatch):
    before = data.select_facts(facts)
    changed = [replace(f, v="Unused value", country_v="Unused country") for f in facts]
    after = data.select_facts(changed)
    assert [f.fid for f in before.main] == [f.fid for f in after.main]
    monkeypatch.setattr(old, "screening_queries", lambda: [])
    assert before == data.select_facts(facts)


def test_invalid_selection_and_calibration_subject_overlap_are_rejected(facts):
    with pytest.raises(ValueError, match="subject groups"):
        data.select_facts(facts, n=47)
    with pytest.raises(ValueError, match="subject groups"):
        data.select_facts(facts, n=10000)
    with pytest.raises(ValueError, match="IDs must be unique"):
        data.select_facts([*facts, facts[0]])
    overlap = replace(facts[-1], h=facts[0].h)
    with pytest.raises(ValueError, match="calibration and main"):
        data.select_facts([*facts[:-1], overlap])


def test_direct_questions_keep_historical_golds_and_grading(facts):
    f = facts[-1]
    rows = data.evaluation_rows(f)
    originals = {
        r["use"]: r for r in old.use_queries(f) if r["mode"] == "direct" and r["use"] in data.USES
    }
    assert tuple(r["use"] for r in rows) == data.USES
    for row in rows:
        assert row["fact_id"] == row["fid"] == f.fid
        assert row["split"] == f.split
        assert all(row[key] == value for key, value in originals[row["use"]].items())
        assert row["exclude"] == sorted({f.h, f.country_h})
        assert data.grade(row["gold"], row)["strict"]
    assert rows[2]["gold"] == f.country_v
    assert rows[3]["gold"] == f.v
    assert data.grade(f"{f.h}'s partner is {f.v}", rows[0])["strict"]


def test_retention_is_unrelated_and_separate_from_feature_selection(facts):
    screening = {r["qid"]: r for r in old.screening_queries()}
    for f in data.select_facts(facts).main:
        row = data.retention_row(f)
        assert row == data.retention_row(f)
        assert row["role"] == "retention"
        assert row["feature_selection_eligible"] is False
        assert row["qid"].startswith("city_country/")
        assert row["qid"].split("/", 1)[1] not in {f.h, f.v}
        assert row["gold"] not in {f.country_h, f.country_v}
        assert row["gold"] == screening[row["qid"]]["gold"]
        assert row["prompt"] == screening[row["qid"]]["prompt"] + old.DIRECT
        assert all(r["use"] != "retention" for r in data.evaluation_rows(f))


def test_wrong_value_changes_only_value_and_country_and_records_donor(facts):
    by_id = {f.fid: f for f in facts}
    for f in facts:
        wrong = data.wrong_value_fact(f, facts)
        assert wrong == data.wrong_value_fact(f, reversed(facts))
        source = by_id[wrong.source_fid]
        assert source.split != "calibration"
        assert source.relation == f.relation
        assert wrong.fact.v == source.v != f.v
        assert wrong.fact.country_v == source.country_v != f.country_v
        assert asdict(wrong.fact) == {**asdict(f), "v": source.v, "country_v": source.country_v}
        assert data.evaluation_rows(wrong.fact)[2]["gold"] != data.evaluation_rows(f)[2]["gold"]


def test_wrong_value_never_uses_different_city_in_same_country():
    f = old.Fact("a", "Paris", "France", "aurora", "Osaka", "Japan", "test")
    same_country = replace(f, fid="b", h="Berlin", v="Tokyo")
    with pytest.raises(ValueError, match="different country"):
        data.wrong_value_fact(f, [f, same_country])
    different_relation = replace(f, fid="c", relation="cobalt", v="Rome", country_v="Italy")
    assert data.wrong_value_fact(f, [f, different_relation]).source_fid == "c"


def test_context_pair_changes_only_target_value_with_three_shared_distractors(facts):
    by_id = {f.fid: f for f in facts}
    for f in facts:
        pair = data.controlled_contexts(f, facts)
        assert pair == data.controlled_contexts(f, reversed(facts))
        clean, corrupt = pair.clean.splitlines(), pair.corrupt.splitlines()
        assert len(clean) == len(corrupt) == 4
        assert len(pair.distractor_fids) == 3
        assert clean[pair.target_index] == data.line(f)
        assert corrupt[pair.target_index] == data.line(pair.wrong_fact)
        assert sum(a != b for a, b in zip(clean, corrupt, strict=True)) == 1
        distractors = [by_id[fid] for fid in pair.distractor_fids]
        assert all(d.relation == f.relation and d.split != "calibration" for d in distractors)
        assert len({f.h, *(d.h for d in distractors)}) == 4
        assert len({f.v, pair.wrong_fact.v, *(d.v for d in distractors)}) == 5
        assert [line for i, line in enumerate(clean) if i != pair.target_index] == [
            data.line(d) for d in distractors
        ]
        original = asdict(f)
        corrupt_fact = asdict(pair.wrong_fact)
        for key in original.keys() - {"v", "country_v"}:
            assert corrupt_fact[key] == original[key]


def test_context_pair_fails_instead_of_silently_dropping_distractors(facts):
    f = next(f for f in facts if f.split != "calibration")
    donor = next(
        g
        for g in facts
        if g.split != "calibration" and g.relation == f.relation and g.country_v != f.country_v
    )
    with pytest.raises(ValueError, match="three distinct"):
        data.controlled_contexts(f, [f, donor])
