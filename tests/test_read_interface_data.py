from llm_memory_editability import read_interface_data as d


def known_everything():
    return {q["qid"]: True for q in d.screening_queries()}


def test_grader_prefers_longest_alias_and_ignores_prompt_entities():
    g = d.graders()
    assert g["country"].grade("It is in North Korea.", "South Korea")["lenient"] is False
    assert g["country"].grade("South Korea", "South Korea")["strict"]
    assert g["city"].grade("Tokyo's partner is Lisbon.", "Lisbon", exclude={"Tokyo"})["strict"]
    assert not g["city"].grade("Tokyo's partner is Lisbon.", "Lisbon")["strict"]
    assert g["country"].grade("Let us think. It is Peru.", "Peru")["strict"]
    assert g["country"].grade("the US", "United States")["strict"]
    assert g["city"].grade("São Paulo", "São Paulo")["strict"]
    assert g["city"].grade("Sao Paulo", "São Paulo")["strict"]


def test_facts_are_additions_with_disjoint_subjects_and_values():
    facts, subjects, values = d.build_facts(known_everything(), seed=7)
    cities = d.city_table()
    assert not set(subjects) & set(values)
    assert len({f.fid for f in facts}) == len(facts)
    for f in facts:
        assert f.h in subjects and f.v in values
        assert cities[f.v]["country"] != f.country_h
    per_subject = {}
    for f in facts:
        per_subject.setdefault(f.h, []).append(f)
    assert all(len(v) == 2 and v[0].relation != v[1].relation for v in per_subject.values())
    splits = [f.split for f in facts]
    assert splits.count("calibration") == 10
    assert abs(splits.count("adapt") - splits.count("test")) <= 1
    assert max(sum(f.v == v for f in facts) for v in values) <= 2
    again, _, _ = d.build_facts(known_everything(), seed=7)
    assert again == facts


def test_held_out_prompts_never_appear_in_training_data():
    facts, _, _ = d.build_facts(known_everything(), seed=7)
    fact = facts[0]
    trained = {u for m in ("m1", "m2", "m3", "m5") for u, _ in d.training_examples(fact, m)}
    for row in d.use_queries(fact):
        if row["use"] == "U0":
            assert row["prompt"] in trained
        else:
            assert row["prompt"] not in trained
            stem = row["prompt"].replace(d.DIRECT, "").replace(d.COT, "")
            assert all(stem not in u for u in trained)
    assert len(d.training_examples(fact, "m2")) == 24


def test_roles_and_golds():
    facts, _, _ = d.build_facts(known_everything(), seed=7)
    rows = {(r["use"], r["mode"]): r for r in d.use_queries(facts[0])}
    f = facts[0]
    assert rows[("U2.0", "direct")]["gold"] == f.country_v
    assert rows[("U3.0", "cot")]["gold"] == f.v
    assert f.country_h in rows[("U3.0", "direct")]["prompt"]
    assert f.h not in rows[("U3.0", "direct")]["prompt"]
    assert f.h in rows[("U2.1", "cot")]["prompt"]


def test_replay_never_contains_evaluated_items():
    facts, subjects, values = d.build_facts(known_everything(), seed=7)
    evaluated = {r["prompt"] for f in facts for r in d.use_queries(f)}
    evaluated |= {r["prompt"] for r in d.retention_queries(subjects, values, facts)}
    evaluated |= {r["prompt"] for r in d.natural_two_hop(subjects)}
    replay = d.replay_queries(subjects, "direct") + d.replay_queries(subjects, "cot")
    assert replay and not set(replay) & evaluated
    assert all(f.h not in r for f in facts for r in d.replay_queries(subjects, "cot"))
