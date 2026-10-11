import copy
import random

import pytest

from llm_memory_editability.perk_writer_data import (
    build_pools,
    make_world,
    save_pools,
    validate_world,
    world_id,
)


def test_complete_truth_coverage_and_composition():
    for seed in range(50):
        row = make_world(random.Random(seed))
        validate_world(row)
        assert len(row["atomic"]) == 9 and len(row["compositional"]) == 3
        assert all(len(q["support"]) == 2 for q in row["compositional"])


def test_world_identity_does_not_count_text_order_as_new_world():
    row = make_world(random.Random(17))
    assert (
        world_id(dict(reversed(list(row["locations"].items()))), row["owners"]) == row["world_id"]
    )


def test_disjoint_worlds_and_reserved_bindings():
    counts = {"development": 8, "train": 40, "validation": 8, "test": 40}
    pools = build_pools(counts)
    assert pools == build_pools(counts)
    identities = [r["world_id"] for rows in pools.values() for r in rows]
    assert len(set(identities)) == sum(counts.values())
    for split, rows in pools.items():
        for row in rows:
            assert any(q["held_binding"] for q in row["atomic"]) == (split == "test")
            if split == "test":
                assert any(q["held_binding"] for q in row["compositional"])


@pytest.mark.parametrize("change", ["answer", "support", "world_id", "material"])
def test_corrupt_truth_or_material_is_rejected(change):
    row = copy.deepcopy(make_world(random.Random(13)))
    if change == "answer":
        row["compositional"][0]["answer"] = "wrong"
    elif change == "support":
        row["atomic"][0]["support"] = ["invented"]
    elif change == "world_id":
        row["world_id"] = "wrong"
    else:
        row["context"] = ["Answer: leaked"]
    with pytest.raises(ValueError):
        validate_world(row)


def test_data_is_write_once(tmp_path):
    target = tmp_path / "data"
    result = save_pools(target, {"development": 4, "test": 4})
    assert result["files"]["test"]["rows"] == 4
    with pytest.raises(FileExistsError):
        save_pools(target, {"development": 4})
