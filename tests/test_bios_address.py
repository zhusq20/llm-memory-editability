from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import bios_address as BA
from llm_memory_editability.bios_original_model import GPT

OLD = Path("/mnt/disk2_from_server2/siqizhu4/llm-memory")
DATA = OLD / "data/bios-original-v1"
SOURCE = OLD / "data/bios-source/data-synthetic-pretrain/Capo-bioS-bioR/Capo-bioS-bioR.py"
need_data = pytest.mark.skipif(not DATA.exists(), reason="bioS original data not mounted")


@pytest.fixture(scope="module")
def world():
    return BA.World(DATA / "world-142701", SOURCE, DATA / "tokenizer/tokenizer.json")


def stored(world_root, a, v, attr, form):
    tokens = np.load(world_root / "sentence-tokens.npy", mmap_mode="r")
    lengths = np.load(world_root / "sentence-lengths.npy", mmap_mode="r")
    i = BA.ATTRS.index(attr)
    return [int(t) for t in tokens[a, v, i, form, : lengths[a, v, i, form]]]


@need_data
def test_original_prompt_is_training_prefix(world):
    root = DATA / "world-142701"
    rows = [r for r in BA.read_prompts(world, [0, 1, 2]) if r["prompt"] == "original"]
    for r in rows:
        a, j = r["person"], r["j"]
        want = [BA.EOT] + stored(root, a, 0, BA.ATTRS[0], 0)
        for i in range(1, j):
            want += stored(root, a, 0, BA.ATTRS[i], 1)
        target = stored(root, a, 0, BA.ATTRS[j], 1)
        assert r["ids"] == want + target[: len(r["ids"]) - len(want)]
        rest = world.decode(target[len(r["ids"]) - len(want) :])
        assert rest.lstrip().startswith(r["a_value"])


@need_data
def test_conflict_changes_only_the_name(world):
    rows = BA.read_prompts(world, [5])
    by = {(r["attr"], r["prompt"]): r for r in rows}
    for attr in BA.ATTRS[1:]:
        o, c = by[(attr, "original")], by[(attr, "conflict")]
        p_a, p_b = world.people[o["person"]], world.people[o["partner"]]
        assert p_a["pronoun"] == p_b["pronoun"] and o["a_value"] != o["b_value"]
        assert world.decode(o["ids"]).replace(p_a["name"], p_b["name"]) == world.decode(c["ids"])
        u = world.decode(by[(attr, "context_only")]["ids"]).split(" entered")[0]
        assert p_a["name"] not in world.decode(by[(attr, "context_only")]["ids"]) or u


@need_data
def test_biography_rows_match_training_tokens(world):
    root = DATA / "world-142701"
    rows = BA.biography_rows(world, [3], [2], permute=False)
    for r in rows:
        k = BA.ATTRS.index(r["attr"])
        want = [BA.EOT] + stored(root, 3, 2, BA.ATTRS[0], 0)
        for i in range(1, k + 1):
            want += stored(root, 3, 2, BA.ATTRS[i], 1)
        if k == 0:
            want = [BA.EOT] + stored(root, 3, 2, BA.ATTRS[0], 0)
        s, e = r["targets"]
        assert r["ids"] == want[: len(r["ids"])]
        assert world.decode(r["ids"][s:e]).strip() == BA.value(world.people[3], r["attr"])
        lab = np.array(r["label"])
        assert (lab == 3).sum() == e - s and lab[s - 1] == 3
        assert (lab == 0).sum() == len(world.encode(" " + world.people[3]["name"]))


def tiny():
    torch.manual_seed(0)
    cfg = dict(vocab_size=50257, width=32, layers=2, heads=4, context=64,
               rotary_fraction=0.25, rotary_base=10000, dropout=0.0)  # fmt: skip
    return GPT(cfg).eval()


def test_group_shares_match_weight_gradient():
    model = tiny()
    rows = [
        dict(ids=[5, 6, 7, 8, 9, 10], label=[1, 0, 0, 2, 3, 3], targets=(4, 6),
             person=0, attr="x", slot=1),
        dict(ids=[5, 6, 11, 12], label=[1, 0, 3, 3], targets=(3, 4), person=1, attr="x", slot=0),
    ]  # fmt: skip
    model.requires_grad_(False)
    for b in model.blocks:
        b.down.weight.requires_grad_(True)
    k, d, lab = BA.write_batch(model, rows)
    for i, b in enumerate(model.blocks):
        outer = torch.einsum("btd,btf->df", d[i].double(), k[i].double())
        assert torch.allclose(outer, b.down.weight.grad.double(), rtol=1e-4, atol=1e-7)
    sh = BA.group_shares(k, d, lab)
    for row in range(2):
        for g in range(len(BA.GROUPS)):
            m = lab[row] == g
            dw = torch.einsum("td,tf->df", d[0, row][m].double(), k[0, row][m].double())
            full = torch.einsum("td,tf->df", d[0, row].double(), k[0, row].double())
            assert torch.allclose(sh[0, row, g], dw.norm() / full.norm(), atol=1e-6)


def test_cross_variant_cosine_identical_rows_is_one():
    model = tiny()
    r = dict(ids=[5, 6, 7, 8, 9, 10], label=[1, 0, 0, 2, 3, 3], targets=(4, 6),
             person=0, attr="x", slot=1)  # fmt: skip
    k, d, lab = BA.write_batch(model, [r, dict(r)])
    cos, n = BA.cross_variant_cosine(k, d, lab, [r, dict(r)])
    assert n == 1 and torch.allclose(cos[:-1], torch.ones_like(cos[:-1]), atol=1e-6)
    assert torch.isnan(cos[-1, 0]) and torch.allclose(cos[-1, 1], torch.ones((), dtype=cos.dtype))
