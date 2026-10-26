from llm_memory_editability import read_interface_bridge as b
from llm_memory_editability import read_interface_data as d


def facts():
    known = {q["qid"]: True for q in d.screening_queries()}
    out, _, _ = d.build_facts(known, seed=7)
    return out


def test_partners_never_share_subject_country_or_value():
    fs = facts()
    by = {f.fid: f for f in fs}
    partner = b.partners(fs)
    assert sum(p is not None for p in partner.values()) >= 0.95 * len(fs)
    for fid, gid in partner.items():
        if gid is None:
            continue
        f, g = by[fid], by[gid]
        assert f.relation == g.relation
        assert f.h != g.h and f.country_h != g.country_h and f.v != g.v
        assert f.v not in (g.h, g.v) and g.v != f.h
    assert b.partners(fs) == partner


def test_decode_hits_and_ambiguous_names():
    g = d.graders()
    assert b.name_hit("Buenos Aires", "Buenos Aires", g["city"])
    assert b.name_hit("Aires ->", "Buenos Aires", g["city"])
    assert not b.name_hit("Some city", "Mexico City", g["city"])
    assert b.name_hit("Japan", "Japan", g["country"])
    assert not b.name_hit("cat", "Tokyo", g["city"])
    f = d.Fact("r/x", "Kuwait City", "Kuwait", "aurora", "Lima", "Peru", "test")
    assert b.ambiguous(f)
    f = d.Fact("r/y", "Tokyo", "Japan", "aurora", "Lima", "Peru", "test")
    assert not b.ambiguous(f)


def test_bridge_prompts_hide_the_subject():
    for f in facts()[:20]:
        for r in b.bridge_rows([f]):
            assert f.h not in r["prompt"] and f.country_h in r["prompt"]
            assert r["subject"] == f.country_h
