import copy

import numpy as np
import pytest

from llm_memory_editability.gated_depth_data import (
    BalancedStream,
    audit_data,
    generate,
    load,
    pack,
    prepare,
    query_key,
)


def config():
    return {
        "data": {
            "entities_per_graph": 8,
            "relations": 4,
            "slots": 20,
            "world_seed": 1,
            "layout_seed": 2,
            "evaluation_templates": 2,
            "train_templates": {"2": 4, "3": 4, "4": 4, "5": 4},
            "long_hops": [6, 8, 10, 12, 16, 20],
        }
    }


def test_truth_partition_balance_atoms_and_no_b_composition():
    pools, permutations, metadata, audit = generate(config())
    assert audit["passed"]
    assert len(pools["facts_train"]) == 64
    assert all(r["graph_id"] == "A" and r["hop_count"] <= 5 for r in pools["comp_A_train_k1to5"])
    groups = [
        {query_key(r) for r in pools[name] if r["hop_count"] > 1}
        for name in ("comp_A_train_k1to5", "comp_A_val_k1to5", "eval_A_short")
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    broken = copy.deepcopy(pools)
    broken["eval_A_long"][0]["answer_entity"] = 999
    with pytest.raises(AssertionError):
        audit_data(broken, permutations, metadata)


def test_no_answer_token_or_bridge_and_identity_padding():
    pools, _, metadata, _ = generate(config())
    rows = pools["eval_A_long"][:8]
    x, pos, labels = pack(rows, metadata)
    assert x.shape == (8, 22)
    assert np.all(pos == 21)
    for i, row in enumerate(rows):
        relations = x[i, 1:21]
        actual = relations[relations != metadata["noop_token"]] - metadata["relation_offset"]
        assert actual.tolist() == row["relation_ids"]
        assert labels[i] == row["answer_entity"] + metadata["entity_offset"]
        assert np.all(relations >= metadata["relation_offset"])
    different = copy.deepcopy(rows)
    for row in different:
        row["answer_entity"] = 999
        row["intermediate_entities"] = [999]
    assert np.array_equal(pack(different, metadata)[0], x)


def test_noop_controls_preserve_fact_and_hop():
    pools, _, metadata, _ = generate(config())
    rows = [r for r in pools["eval_A_short"] if r["hop_count"] == 3]
    for count in (8, 12, 20):
        for layout in ("left", "right", "random"):
            x, pos, _ = pack(rows, metadata, slots=count, layout=layout)
            assert np.all(pos == count + 1)
            assert np.all((x[:, 1 : count + 1] == metadata["noop_token"]).sum(1) == count - 3)
            assert np.all(x[:, count + 2 :] == metadata["pad_token"])


def test_paired_stream_and_rng_restore():
    pools, _, metadata, _ = generate(config())
    a, b = [BalancedStream(pools, metadata, 11) for _ in range(2)]
    for _ in range(5):
        ap, ar = a.batch(128)
        bp, br = b.batch(128)
        assert all(np.array_equal(x, y) for x, y in zip(ap, bp, strict=True))
        assert ar == br
    state = copy.deepcopy(a.rng.bit_generator.state)
    first, _ = a.batch(128)
    a.rng.bit_generator.state = state
    second, _ = a.batch(128)
    assert all(np.array_equal(x, y) for x, y in zip(first, second, strict=True))


def test_disk_hash_roundtrip_and_controls(tmp_path):
    prepare(config(), tmp_path / "data")
    pools, arrays, metadata = load(tmp_path / "data")
    assert metadata["vocab_size"] == 24
    assert len([key for key in arrays if key.startswith("noop_")]) == 27
    assert len(pools["facts_train"]) == 64
    (tmp_path / "data" / "facts_train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="Data changed"):
        load(tmp_path / "data")
