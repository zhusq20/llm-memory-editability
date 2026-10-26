"""CPU contracts of answer-gradient-address-v1: hook-captured delta/k reproduce the down_proj
weight gradient, Gram-based shares equal the materialized ones, left-padded rows are independent,
and the position groups and content points are what the protocol defines."""

from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import answer_gradient as ag
from llm_memory_editability import read_interface_data as d
from llm_memory_editability import read_interface_model as m

QWEN3_TOKENIZER = Path(m.MODELS["qwen3-1.7b"]["path"])


class PadTok:
    pad_token_id = 0


def tiny(kind, vocab=97):
    torch.manual_seed(0)
    if kind == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
        model = Qwen3ForCausalLM(cfg)
    else:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        cfg = Qwen3_5TextConfig(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
        model = Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():  # move norms and weights off their trivial initial values
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.double().eval()


def items(lengths, answers, vocab=97, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for n, k in zip(lengths, answers, strict=True):
        ids = torch.randint(1, vocab, (n,), generator=g).tolist()
        out.append({"ids": ids, "positions": [], "answer": (n - k, n)})
    return out


@pytest.mark.parametrize("kind", ["qwen3", "qwen3.5"])
def test_hooks_reproduce_down_proj_weight_gradient(kind):
    model = tiny(kind)
    for p in model.parameters():
        p.requires_grad_(False)
    params = [layer.mlp.down_proj.weight for layer in m.layers(model)]
    for p in params:
        p.requires_grad_(True)
    batch = items([11, 7, 9], [3, 2, 4])
    k, delta, pads, stats = ag.batch_gradients(model, PadTok(), batch)
    for i, p in enumerate(params):
        outer = torch.einsum("btd,btf->df", delta[i], k[i])
        assert torch.allclose(outer, p.grad, rtol=1e-9, atol=1e-12), i
    assert pads == [0, 4, 2]
    assert [s["n_tokens"] for s in stats] == [3, 2, 4]


@pytest.mark.parametrize("kind", ["qwen3", "qwen3.5"])
def test_left_padded_rows_are_independent(kind):
    model = tiny(kind)
    for p in model.parameters():
        p.requires_grad_(False)
    batch = items([10, 6, 8], [3, 2, 2], seed=3)
    k, delta, pads, stats = ag.batch_gradients(model, PadTok(), batch)
    for i, it in enumerate(batch):
        k1, d1, _, s1 = ag.batch_gradients(model, PadTok(), [it])
        # float32-level agreement: RoPE tables and the logits cast stay float32, and padding
        # shifts absolute positions; a leak through padding would differ at order one
        scale = float(d1.abs().max())
        assert float((delta[:, i, pads[i] :] - d1[:, 0]).abs().max()) < 1e-5 * scale
        assert torch.allclose(k[:, i, pads[i] :], k1[:, 0], rtol=1e-5, atol=1e-7)
        assert abs(stats[i]["loss"] - s1[0]["loss"]) < 1e-5
        assert float(delta[:, i, : pads[i]].abs().sum()) == 0.0
    # the last layer only reaches the loss through its own position's logits
    n, (a, b) = len(batch[0]["ids"]), batch[0]["answer"]
    last = delta[-1, 0]
    assert float(last[: a - 1].abs().max()) == 0.0 and float(last[n - 1].abs().max()) == 0.0
    assert float(last[a - 1 : b - 1].norm(dim=-1).min()) > 0


def test_gram_shares_equal_materialized_and_projection_shares_sum_to_one():
    g = torch.Generator().manual_seed(5)
    n_layers, n, f, dim = 3, 9, 12, 7
    k = torch.randn(n_layers, n, f, generator=g, dtype=torch.float64)
    delta = torch.randn(n_layers, n, dim, generator=g, dtype=torch.float64)
    label = np.array([3, 3, 1, 1, 2, 3, 0, 0, 0])
    masks = np.zeros((len(ag.GROUPS), n))
    masks[0], masks[1], masks[3], masks[4] = label == 0, label == 1, label == 2, label == 3
    masks[2, 3] = 1.0
    exact, proj, approx, total = ag.address_shares(k, delta, masks)
    for layer in range(n_layers):
        full = torch.einsum("pd,pf->df", delta[layer], k[layer])
        assert torch.isclose(total[layer], full.norm())
        for j in range(len(ag.GROUPS)):
            sel = torch.as_tensor(masks[j]).bool()
            part = torch.einsum("pd,pf->df", delta[layer, sel], k[layer, sel])
            assert torch.isclose(exact[layer, j], part.norm() / full.norm())
            assert torch.isclose(proj[layer, j], (part * full).sum() / full.norm() ** 2)
        partition = [0, 1, 3, 4]
        assert torch.isclose(proj[layer, partition].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.isclose(approx[layer, partition].sum(), torch.tensor(1.0, dtype=torch.float64))


def facts():
    known = {q["qid"]: True for q in d.screening_queries()}
    return d.build_facts(known, seed=7)[0]


def test_use_rows_and_gold_offsets():
    f = [x for x in facts() if x.split != "calibration"][0]
    targets = {"continent": "Europe", "language": "French"}
    rows = ag.use_rows(f, targets)
    assert [r["use"] for r in rows].count("U0") == 24
    assert {u: [r["use"] for r in rows].count(u) for u in ag.USES[1:]} == {
        "U2": 2,
        "continent": 2,
        "language": 2,
    }
    for r in rows:
        at = r.get("gold_at", 0)
        assert r["reply"][at : at + len(r["gold"])] == r["gold"]
        assert f.h in r["prompt"]
    assert len(ag.use_rows(f, {})) == 26
    screen = [
        {"v": "Paris", "op": "continent", "golds": ["Europe"], "strict": True, "first": "Europe"},
        {"v": "Paris", "op": "language", "golds": ["French"], "strict": False, "first": None},
        {"v": "Paris", "op": "currency", "golds": ["euro"], "strict": True, "first": "euro"},
    ]
    assert ag.screen_targets(screen) == {"Paris": {"continent": "Europe"}}


@pytest.mark.skipif(not QWEN3_TOKENIZER.exists(), reason="Qwen3 tokenizer not mounted")
def test_position_groups_and_content_points_with_the_real_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER)
    tok.padding_side = "left"
    f = [x for x in facts() if x.split != "calibration"][0]
    rows = ag.use_rows(f, {"continent": "Europe", "language": "French"})
    same = 0
    for r in rows:
        it = ag.encode_row(tok, r)
        n, (a, _) = len(it["ids"]), it["answer"]
        masks = it["masks"]
        partition = masks[[0, 1, 3, 4]]
        assert np.all(partition.sum(0) == 1)
        assert np.flatnonzero(masks[0]).tolist() == list(range(a - 1, n - 1))
        astar, a0, slast = it["points"]
        assert a0 == a - 1 and masks[1, slast] == 1 and masks[2].sum() == 1
        subject = "".join(tok.decode([it["ids"][p]]) for p in np.flatnonzero(masks[1]))
        assert subject.strip() == f.h
        first = tok.decode([it["target"]]).strip()
        assert first and r["gold"].startswith(first)
        assert it["ids"][astar + 1] == it["target"]
        if r["relation_text"] in r["prompt"]:
            rel = "".join(tok.decode([it["ids"][p]]) for p in np.flatnonzero(masks[3]))
            assert rel.strip() == r["relation_text"]
        if r["use"] == "U0":
            same += astar == a0
        else:
            assert astar == a0
    assert same == 7  # the seven M2 replies that start with v


@pytest.mark.skipif(not QWEN3_TOKENIZER.exists(), reason="Qwen3 tokenizer not mounted")
def test_fact_readouts_end_to_end_on_a_tiny_model():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER)
    tok.padding_side = "left"
    model = tiny("qwen3", vocab=len(tok)).float()
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers = len(m.layers(model))
    f = [x for x in facts() if x.split != "calibration"][0]
    cities = sorted(d.city_table())
    g = torch.Generator().manual_seed(0)
    entity = {i: torch.randn(len(cities), 32, generator=g) for i in range(n_layers)}
    refs = ag.Refs(model, entity, cities)
    rows = ag.use_rows(f, {"continent": "Europe", "language": "French"})
    err = ag.weight_gradient_check(model, tok, rows, [0, n_layers - 1])
    assert max(err.values()) < 1e-5
    recs, means, ex = ag.fact_readouts(model, tok, refs, f, rows, {("U0", 0)})
    assert len(recs) == len(rows) and set(ex) == {"U0.0"}
    for r in recs:
        assert r["exact"].shape == (n_layers, len(ag.GROUPS))
        assert r["exact"][-1, 0] == pytest.approx(1.0, abs=1e-6)  # only answer positions
        assert np.isnan(r["cos_e_v"][-1, 2]) and not np.isnan(r["cos_e_v"][-1, 0])
        assert np.all(r["rank_target"][:, 0] >= 0)
    sh = ag.sharing(means)
    assert {"U0~U2", "U0~continent", "U0~language", "U0_even~U0_odd"} <= set(sh)
    fl = ag.fact_level(recs)
    agg = ag.aggregate(fl, {f.fid: sh}, {})
    assert agg["U0"]["n_facts"] == 1
    pred = ag.predictions(agg, n_layers)
    assert set(pred) >= {"P1", "P2a", "P2b", "P3"}
    assert len(ex["U0.0"]["position_share"]) == n_layers
