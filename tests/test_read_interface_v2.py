import torch
from torch import nn

from llm_memory_editability import read_interface_data as d
from llm_memory_editability import read_interface_e3 as e3
from llm_memory_editability import read_interface_v2_data as v


def facts():
    known = {q["qid"]: True for q in d.screening_queries()}
    return d.build_facts(known, seed=7)


def test_attribute_tables_cover_every_subject_and_value_country():
    countries = {row["country"] for row in d.city_table().values()}
    for c in countries:
        assert v.continent_of(c), c
        assert v.LANGUAGES.get(c), c
    g = v.graders()
    assert v.grade_any(g["continent"], "Australia.", ["Oceania"])["strict"]
    assert v.grade_any(g["language"], "Mandarin Chinese", ["Chinese"])["strict"]
    assert v.grade_any(g["mountain"], "It is Mount Fuji.", ["Mount Fuji"])["strict"]
    assert v.grade_any(g["river"], "The Moscow River", ["Moskva"])["strict"]
    assert not v.grade_any(g["language"], "Malayalam", ["Malay"])["lenient"]


def test_distractor_context_has_target_and_three_same_relation_others():
    fs, _, _ = facts()
    systems = v.distractor_systems(fs)
    by_id = {f.fid: f for f in fs}
    for fid, entry in systems.items():
        f = by_id[fid]
        assert d.context_statement(f) in entry["system"]
        assert len(entry["distractors"]) == 3
        for g in map(by_id.get, entry["distractors"]):
            assert g.relation == f.relation and g.h != f.h and g.v != f.v
            assert d.context_statement(g) in entry["system"]
    assert v.distractor_systems(fs) == systems


def test_natural_reference_prompts_have_the_right_hops():
    known = {q["qid"]: True for q in d.screening_queries()}
    rows = v.natural_reference_rows(known)
    assert {r["shape"] for r in rows} == {"U2", "U3"}
    for r in rows:
        if r["role"].startswith("two_hop"):
            if r["shape"] == "U2":
                assert r["h"] in r["prompt"] and r["C"] not in r["prompt"]
            else:
                assert r["C"] in r["prompt"] and r["h"] not in r["prompt"]
        if r["role"] == "no_relation":
            assert "capital" not in r["prompt"]
        assert r["golds"]


def test_usage_practice_is_disjoint_from_evaluation_and_never_names_country_of_v():
    fs, _, _ = facts()
    g = v.graders()
    for f in fs[:30]:
        practice = v.practice_examples(f)
        para = v.paraphrase_examples(f)
        assert len(practice) == len(para) == 8
        trained = {u for u, _ in practice + para} | {u for u, _ in d.training_examples(f, "m2")}
        evaluated = [r["prompt"] for r in v.op_queries(f)] + [r["prompt"] for r in d.use_queries(f)]
        for prompt in evaluated:
            if prompt.startswith(d.fill(d.U0, f)):
                continue
            assert prompt not in trained
            stem = prompt.replace(d.DIRECT, "").replace(d.COT, "")
            assert all(stem not in u for u in trained)
        for user, reply in practice:
            plain = reply
            for continent in v.CONTINENT_ALIASES:
                plain = plain.replace(continent, "")
            assert f.country_v not in g["country"].mentions(plain)
            assert f.v in reply or user.endswith(d.DIRECT)
        for user, _ in para:
            assert f.h in user and d.RELATIONS[f.relation] in user


def test_counterfactual_country_control():
    fs, subjects, _ = facts()
    pairs = v.e1n_pairs(subjects)
    cities = d.city_table()
    for h, c in pairs.items():
        own = cities[h]["country"]
        assert c != own
        assert not d.currency_aliases(d.currency_of(own)) & d.currency_aliases(d.currency_of(c))
    train = {u for u, _ in v.E1N_TRAIN}
    assert not train & set(v.E1N_HELD_OUT)


class TinyMLP(nn.Module):
    def __init__(self, dim=6, inter=10):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def test_adapters_start_at_zero_and_contributions_match_masking():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    for kind in ("structural", "synaptic"):
        w = e3.Adapted(TinyMLP(), kind)
        x = torch.randn(2, 5, 6)
        base = w.mlp(x)
        assert torch.allclose(w(x), base)
        for _ in range(3):
            before = w(x)
            w.begin(0.3, gen)
            assert torch.allclose(w(x), before)
            with torch.no_grad():
                for p in w.current.values():
                    p.add_(torch.randn_like(p) * 0.5)
            w.commit()
        full = w(x)
        flat = x.reshape(-1, 6)
        for k in range(3):
            mask = torch.ones(2, 3)
            mask[:, k] = 0
            w.mask = mask
            without = w(x)
            w.mask = None
            ks = torch.full((len(flat),), k)
            got = e3.contributions(w, flat, ks, vectors=True)
            assert torch.allclose(got, (full - without).reshape(-1, 6), atol=1e-5), kind


def test_e2_facts_pair_twin_city_with_a_fictional_relation_and_avoid_priors():
    known = {q["qid"]: True for q in d.screening_queries()}
    v.register_e2_relations()
    order = v.e2_candidates(known)
    priors = {h: ["Tokyo", "Tokyo", "Tokyo"] if i < 3 else ["Osaka", None, "Kyoto"]
              for i, h in enumerate(order)}  # fmt: skip
    fs, subjects, _, stable = v.build_e2_facts(known, priors)
    assert set(stable) == set(order[:3]) and not set(stable) & set(subjects)
    per = {}
    for f in fs:
        per.setdefault(f.h, []).append(f)
        assert f.v not in priors[f.h]
        assert f.country_v != f.country_h
    for pair in per.values():
        assert [f.relation for f in pair][0] == "twin" and pair[1].relation in d.RELATIONS
        assert pair[1].relation != "twin"
    assert d.fill(d.U0, fs[0]).startswith("What is the twin city of")
