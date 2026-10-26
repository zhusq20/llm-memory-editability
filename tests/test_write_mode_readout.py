import torch

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_v2_data as V
from llm_memory_editability import write_mode_readout as W


def toy_facts():
    cities = [f"City{i:02d}" for i in range(40)]
    facts = []
    for i in range(16):
        rel = ("aurora", "helix")[i % 2]
        facts.append(
            D.Fact(
                f"{rel}/{cities[i]}", cities[i], f"Land{i}", rel, cities[20 + i], f"Far{i}", "adapt"
            )  # fmt: skip
        )
    return facts


def test_control_systems_swap_only_the_target_line():
    facts = toy_facts()
    systems = V.distractor_systems(facts)
    controls = W.control_systems(facts, systems)
    assert controls == W.control_systems(facts, systems)
    for f in facts:
        sys_lines = systems[f.fid]["system"].split("\n")
        ctl_lines = controls[f.fid]["system"].split("\n")
        assert len(sys_lines) == len(ctl_lines) == 4
        changed = [i for i, (a, b) in enumerate(zip(sys_lines, ctl_lines, strict=True)) if a != b]
        assert len(changed) == 1 and sys_lines[changed[0]] == W.line(f)
        g = next(x for x in facts if x.fid == controls[f.fid]["replacement"])
        assert g.relation == f.relation and g.h != f.h and g.v != f.v
        assert f.h not in controls[f.fid]["system"] and f.v not in controls[f.fid]["system"]


def test_locate_spans_by_use():
    f = toy_facts()[0]
    rows = {r["use"]: r for r in W.use_rows(f, W.G3_USES)}
    system_line = W.line(f)
    for use, r in rows.items():
        text = f"<sys>{system_line}\nFact: other</sys><user>{r['prompt']}</user><asst>"
        spans = W.locate(text, r, f, system_line)
        assert text[slice(*spans["user"])] == r["prompt"]
        assert text[slice(*spans["record"])] == system_line
        assert text[slice(*spans["relation"])] == D.RELATIONS[f.relation]
        if use in ("U3.0", "U3.1", "shortcut"):
            assert spans["subject"] is None and text[slice(*spans["country"])] == f.country_h
        else:
            assert spans["country"] is None and text[slice(*spans["subject"])] == f.h
            assert spans["subject"][0] > spans["user"][0]


def test_specific_and_group_measures():
    cos = torch.tensor([[0.5, 0.1, 0.0]])
    assert torch.allclose(W.specific(cos, 0), torch.tensor([0.45]))
    g = torch.Generator().manual_seed(0)
    L, C, d = 3, 5, 16
    ent = torch.nn.functional.normalize(torch.randn(L, C, d, generator=g), dim=-1)
    ans = torch.nn.functional.normalize(torch.randn(C, d, generator=g), dim=-1)
    ref = torch.randn(L, 2, d, generator=g)
    delta = ent[:, 2:3].expand(L, 2, d) * 0.5
    rel, e, a = W.group_measures(delta, ref, ent, ans, 2)
    assert torch.allclose(rel, (0.5 / ref.norm(dim=-1)).mean(1))
    assert (e > 0.5).all()
    rel, e, a = W.group_measures(ans[1].expand(L, 2, d), ref, ent, ans, 1)
    assert (a > 0.5).all()


def test_knockout_mask_blocks_only_requested_pairs():
    b, t = 2, 6
    mask = W.knockout_mask(None, b, t, "cpu", [1], [[4, 5]], [[0, 2]])
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    assert torch.equal(mask[0, 0], causal)
    expect = causal.clone()
    for q in (4, 5):
        for k in (0, 2):
            expect[q, k] = False
    assert torch.equal(mask[1, 0], expect)
    additive = torch.zeros(b, 1, t, t)
    out = W.knockout_mask(additive, b, t, "cpu", [0], [[3]], [[1]])
    assert out[0, 0, 3, 1] == torch.finfo(torch.float32).min and out.sum() == out[0, 0, 3, 1]


def test_query_positions():
    pos = {"user": [5, 6, 7]}
    q = W.query_positions(pos, (12, 15), 15)
    assert q["question"] == list(range(5, 15)) and q["last"] == [11, 12, 13, 14]
