"""The cited APE formula must retain old nodes and causal GPT2 computation."""

from __future__ import annotations

import pytest

from llm_memory_editability.perk_data import sha256_file
from llm_memory_editability.perk_fticr import (
    _load_upstream,
    check_data,
    extend_gpt2_positions,
    interpolate_absolute_positions,
)


def test_upstream_encoder_fork_serialization_preserves_all_rows(tmp_path, monkeypatch):
    import sys

    import llm_memory_editability.perk_fticr as baseline

    datasets = pytest.importorskip("datasets")
    multiprocess = pytest.importorskip("multiprocess")
    if multiprocess.get_start_method() != "fork":
        pytest.skip("This contract exercises the Linux Dataset.map fork workers")
    source = tmp_path / "open_instruct" / "finetune.py"
    source.parent.mkdir()
    source.write_text("def encode(row):\n    return {'encoded': row['source_id'] * 2}\n")
    monkeypatch.setattr(baseline, "OPEN_INSTRUCT_FINETUNE_SHA256", sha256_file(source))
    monkeypatch.delitem(sys.modules, "perk_author_open_instruct", raising=False)
    module = _load_upstream(tmp_path)
    try:
        rows = datasets.Dataset.from_dict({"source_id": list(range(17))})
        encoded = rows.map(module.encode, num_proc=4, load_from_cache_file=False)
        assert encoded["source_id"] == list(range(17))
        assert encoded["encoded"] == [index * 2 for index in range(17)]
    finally:
        sys.modules.pop("perk_author_open_instruct", None)


def test_integer_position_interpolation_preserves_original_nodes_and_tail():
    torch = pytest.importorskip("torch")
    weight = torch.tensor([[1.0, 2.0], [5.0, 8.0], [9.0, 14.0]])
    result = interpolate_absolute_positions(weight, 12)
    assert torch.equal(result[::4], weight)
    assert torch.equal(result[1], torch.tensor([2.0, 3.5]))
    assert torch.equal(result[-4:], weight[-1:].expand(4, -1))
    with pytest.raises(ValueError, match="integral"):
        interpolate_absolute_positions(weight, 10)


def test_gpt2_long_context_forward_backward_and_causal_mask():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=1, n_head=2, n_embd=8, n_positions=4, vocab_size=20)
    )
    original = model.transformer.wpe.weight.detach().clone()
    extend_gpt2_positions(model, 12)
    assert torch.equal(model.transformer.wpe.weight[::3], original)
    assert model.config.n_positions == 12
    assert model.transformer.h[0].attn.bias.shape == (1, 1, 12, 12)
    assert not model.transformer.h[0].attn.bias[0, 0, 0, 1]
    inputs = torch.randint(0, 20, (1, 10))
    output = model(input_ids=inputs, labels=inputs)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert model.transformer.wpe.weight.grad is not None


def test_full_data_cannot_be_truncated_or_changed(tmp_path):
    import json

    manifest = {"outputs": {}, "audits": {}, "exact_sample_overlap": {"train/test": 0}}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("original data\n")
        manifest["outputs"][split] = {"fticr": {"path": str(path), "sha256": sha256_file(path)}}
        manifest["audits"][split] = {"measurements": {"fticr_tokens": {"max": 9149}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "data_manifest": str(manifest_path),
        "max_seq_length": 9216,
        "max_position_embeddings": 10240,
    }
    assert check_data(config) == manifest
    with pytest.raises(ValueError, match="truncate"):
        check_data(dict(config, max_seq_length=8192))
    with pytest.raises(ValueError, match="generation"):
        check_data(dict(config, max_position_embeddings=9216))
    (tmp_path / "test.jsonl").write_text("changed data\n")
    with pytest.raises(ValueError, match="Changed"):
        check_data(config)
