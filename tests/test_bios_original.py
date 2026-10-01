"""Scientific-contract checks for original six-attribute natural-language bioS."""

import ast
import random

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_original_data import (
    ATTRS,
    SOURCE_CODE,
    BioStream,
    answer,
    date_key,
    edit_cases,
    make_people,
    render_sentences,
    source_templates,
    verify_source,
)
from llm_memory_editability.bios_original_model import GPT
from llm_memory_editability.bios_original_train import (
    pack_tokens,
    recognized_format,
    sequence_scores,
    supervised_batch,
)


def small_config():
    return dict(
        vocab_size=50257,
        width=32,
        layers=2,
        heads=2,
        context=512,
        rotary_fraction=0.25,
        rotary_base=10000,
        dropout=0.0,
    )


def test_upstream_hashes_and_renderer_equivalence():
    verify_source()
    templates = source_templates()
    tree = ast.parse(SOURCE_CODE.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_text_simple3"
    )
    ns = {"random": random}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "upstream", "exec"), ns)
    for p in make_people(142701, 200)[:25]:
        old = dict(
            p,
            first_name=p["name"].split()[0],
            middle_name=p["name"].split()[1],
            last_name=p["name"].split()[2],
            birthday=p["day"],
            birthyear=p["year"],
            birthmonth=__import__("calendar").month_name[p["month"]],
            company1city=p["workcity"],
            company1name=p["company"],
        )
        random.seed(p["id"])
        expected = ns["get_text_simple3"](old)
        rng = random.Random(p["id"])
        ids = [rng.randrange(len(t)) for t in templates]
        assert "".join(render_sentences(p, templates, ids)) == expected
        permuted = render_sentences(p, templates, ids, [5, 1, 3, 0, 4, 2])
        assert p["name"] in permuted[0]
        assert sum(p["name"] in sentence for sentence in permuted) == 1
        assert p["company"] in permuted[0]


def test_split_comparisons_and_edit_factorial_truth():
    people = make_people(142701, 200)
    assert len({p["name"] for p in people}) == 200
    for split, n in [("train", 100), ("dev", 20), ("test", 80)]:
        selected = [p for p in people if p["split"] == split]
        assert len(selected) == n
        assert sum(date_key(p) < date_key(people[p["partner"]]) for p in selected) == n // 2
        assert all(people[p["partner"]]["split"] == split for p in selected)
    cases = edit_cases(people, 142701)
    assert len(cases) == 16
    assert sum(c["old_month"] % 2 for c in cases) == 8
    for c in cases:
        p = people[c["person_id"]]
        assert p["split"] == "test"
        cells = [dict(p, **x) for x in c["cells"]]
        assert answer(cells[0], "year") == answer(cells[1], "year")
        assert answer(cells[2], "year") == answer(cells[3], "year")
        assert answer(cells[0], "parity") == answer(cells[2], "parity")
        assert answer(cells[1], "parity") == answer(cells[3], "parity")
        assert answer(cells[0], "parity") != answer(cells[1], "parity")
        for attr in ATTRS[1:]:
            assert len({answer(x, attr) for x in cells}) == 1


@pytest.mark.parametrize("length", [2, 512, 513, 514, 1028])
def test_packing_no_dropped_or_duplicated_targets(length):
    tokens = np.arange(length)
    x, y = pack_tokens(tokens)
    assert np.array_equal(y[y != -100].numpy(), tokens[1:])
    assert np.array_equal(x[y != -100].numpy(), tokens[:-1])


def test_biostream_exposure_and_s_m_mp_matching(tmp_path):
    tokens = np.empty((4, 5, 6, 2, 1), dtype=np.uint16)
    for p in range(4):
        for v in range(5):
            for a in range(6):
                tokens[p, v, a, :, 0] = 1000 + p * 100 + v * 10 + a
    np.save(tmp_path / "sentence-tokens.npy", tokens)
    np.save(tmp_path / "sentence-lengths.npy", np.ones((4, 5, 6, 2), dtype=np.int16))
    observed = {p: [] for p in range(4)}
    for epoch in range(10):
        m, _ = BioStream(tmp_path, 42, "M").pass_tokens(epoch)
        mp, _ = BioStream(tmp_path, 42, "MP").pass_tokens(epoch)
        assert sorted(m) == sorted(mp)
        assert sum(m == 50256) == 5
        for pid in range(4):
            values = [x for x in m if 1000 + pid * 100 <= x < 1100 + pid * 100]
            assert len(values) == 6
            observed[pid].append((values[0] - 1000 - pid * 100) // 10)
    assert all(sorted(v) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] for v in observed.values())
    s, _ = BioStream(tmp_path, 42, "S").pass_tokens(3)
    assert all((x - 1000) % 100 < 6 for x in s if x != 50256)


def test_causality_partial_rope_and_lora_freezing():
    torch.manual_seed(1)
    model = GPT(small_config()).eval()
    x = torch.randint(0, 100, (2, 8))
    a = model(x).detach()
    z = x.clone()
    z[:, 5:] = 99
    torch.testing.assert_close(model(z)[:, :5], a[:, :5])
    attn = model.blocks[0].attn
    probe = torch.randn(2, 2, 8, 16)
    rotated = attn.rotate(probe)
    torch.testing.assert_close(rotated[..., 4:], probe[..., 4:], rtol=0, atol=0)
    model.add_lora(2, 4)
    torch.testing.assert_close(model(x), a, rtol=0, atol=0)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(x, x).backward()
    opt.step()
    for n, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[n], value, rtol=0, atol=0)


def test_answer_eos_mask_and_complete_sequence_scoring():
    model = GPT(small_config()).eval()
    rows = [
        dict(ids=[1, 2, 3, 4, 50256], answer_start=3),
        dict(ids=[8, 9, 7, 50256], answer_start=2),
    ]
    x, y = supervised_batch(rows, "cpu")
    assert (y != -100).sum().item() == 4
    avg, full = sequence_scores(model, rows, "cpu")
    logp = model(x).log_softmax(-1)
    expected = logp[0, 2, 4] + logp[0, 3, 50256]
    torch.testing.assert_close(full[0], expected)
    torch.testing.assert_close(avg * 2, full)


def test_format_validity_separate_from_answer_correctness():
    vocab = dict(company={"example inc."}, workcity={"boston, ma"}, name={"alex b cole"})
    assert recognized_format("March 28, 2000", "date", vocab)
    assert not recognized_format("March 29, 2000", "date", vocab)
    assert not recognized_format("I am unsure", "comparison", vocab)
    assert recognized_format("no", "parity", vocab)
    assert not recognized_format("maybe", "parity", vocab)
