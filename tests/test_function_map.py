import torch

from llm_memory_editability import function_map as FM


def test_shares_partition_and_pure_factors():
    g = torch.Generator().manual_seed(0)
    E, Q, P, d = 6, 8, 3, 5
    a = torch.randn(E, 1, P, d, generator=g)
    b = torch.randn(1, Q, P, d, generator=g)
    s = FM.two_way_shares(a.expand(E, Q, P, d))
    assert torch.allclose(s["entity"], torch.ones(P, dtype=torch.float64))
    s = FM.two_way_shares(b.expand(E, Q, P, d) + 3.0)
    assert torch.allclose(s["question"], torch.ones(P, dtype=torch.float64))
    x = torch.randn(E, Q, P, d, generator=g)
    s = FM.two_way_shares(x)
    total = s["entity"] + s["question"] + s["joint"]
    assert torch.allclose(total, torch.ones(P, dtype=torch.float64))
    assert torch.allclose(s["relation"] + s["phrasing"], s["question"])
    # relation-only signal (same for both phrasings) lands in the relation part
    r = torch.randn(1, len(FM.RELATIONS), 1, P, d, generator=g).expand(1, -1, 2, -1, -1)
    s = FM.two_way_shares(r.reshape(1, Q, P, d).expand(E, Q, P, d))
    assert torch.allclose(s["relation"], torch.ones(P, dtype=torch.float64))


def test_shares_match_explicit_anova():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 8, 1, 3, generator=g, dtype=torch.float64)
    s = FM.two_way_shares(x)
    mu = x.mean((0, 1))
    tot = (x - mu).pow(2).sum()
    ss_e = sum(8 * (x[i].mean(0) - mu).pow(2).sum() for i in range(4))
    assert torch.allclose(s["entity"][0], ss_e / tot)


def test_constant_position_is_nan():
    x = torch.ones(3, 8, 2, 4)
    s = FM.two_way_shares(x)
    assert torch.isnan(s["entity"]).all()


def test_counterfactual_targets_differ():
    ents = list(FM.GOLD)[:20]
    t = FM.counterfactual_targets(ents)
    for (e, r), v in t.items():
        assert v.lower() not in {a.lower() for a in FM.GOLD[e][FM.RELATIONS.index(r)]}
    assert len(t) == len(ents) * len(FM.RELATIONS)


def test_similarity_matrix_diagonal():
    x = torch.randn(5, 7)
    m = FM.similarity_matrix(x)
    assert torch.allclose(m.diagonal(), torch.ones(5, dtype=torch.float64))
