"""Architecture-agnostic TTCD states and readouts; CPU engineering only."""

import copy
import hashlib
import json
from functools import partial
from pathlib import Path

import pytest
import torch
from test_ttcd_double_new_eval import material as double_material
from transformers import GPT2Config

from llm_memory_editability import ttcd_double_new_eval as evaluation
from llm_memory_editability import ttcd_verified_base as verified
from llm_memory_editability.grokking_reproduction import model_digest
from llm_memory_editability.path_overlap import World, sha256
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import (
    TTCDReader,
    load_fast,
    local_ttcd_update,
    save_fast,
    tensor_digest,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {"d8": (8, 1, False), "loop4x2": (4, 2, False), "disco4x2": (4, 2, True)}
PANEL_FILE_SHA256 = "ade07fa92e528f1b19bbc10965d984f747ed7f7daf66f5eafa254971ac7d08f6"


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_reader(architecture, *, vocab_size=45, dtype=torch.float64):
    layers, repeats, discoloop = ARCHITECTURES[architecture]
    torch.manual_seed(731)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=40,
        n_embd=16,
        n_head=4,
        n_layer=layers,
        n_inner=64,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        use_cache=False,
    )
    model = SupervisedAlignmentGPT(config, repeats=repeats, discoloop=discoloop).to(dtype=dtype)
    return TTCDReader(model, native_causal_attention=True).eval()


@pytest.fixture(params=tuple(ARCHITECTURES))
def architecture_reader(request):
    return request.param, make_reader(request.param)


def test_fast_state_matches_independent_physical_layers(architecture_reader):
    name, reader = architecture_reader
    layers, repeats, _ = ARCHITECTURES[name]
    fast = reader.empty_fast()
    assert len(fast) == layers
    assert set(fast) == set(range(layers))
    assert len({value.data_ptr() for value in fast.values()}) == layers
    assert all(value.shape == (64, 16) for value in fast.values())
    assert sum(value.numel() for value in fast.values()) == layers * 64 * 16
    assert layers * repeats == 8
    assert not torch.equal(
        reader.backbone.transformer.h[0].mlp.c_fc.weight,
        reader.backbone.transformer.h[-1].mlp.c_fc.weight,
    )
    assert not any(parameter.requires_grad for parameter in reader.parameters())
    bad = dict(fast)
    bad.pop(layers - 1)
    with pytest.raises(ValueError, match="per physical layer"):
        reader.validate_fast(bad)


def test_native_A0_and_materialized_fast_weights_agree(architecture_reader):
    _, reader = architecture_reader
    tokens = torch.tensor([[2, 7, 8, 4, 11], [3, 8, 4, 1, 12]])
    positions = torch.tensor([[1, 3], [2, 4]])
    zero = reader.empty_fast()
    before = tensor_digest(reader.backbone.state_dict())
    torch.testing.assert_close(
        reader(tokens, positions, fast=zero), reader.backbone(tokens, positions), rtol=0, atol=0
    )
    fast = {layer: value.normal_(std=0.001) for layer, value in zero.items()}
    materialized = copy.deepcopy(reader.backbone)
    with torch.no_grad():
        for layer, block in enumerate(materialized.transformer.h):
            block.mlp.c_proj.weight.add_(fast[layer])
    torch.testing.assert_close(
        reader(tokens, positions, fast=fast),
        materialized(tokens, positions),
        rtol=1e-10,
        atol=1e-10,
    )
    assert tensor_digest(reader.backbone.state_dict()) == before


def test_loop_rounds_read_the_same_fast_matrix(architecture_reader, monkeypatch):
    name, reader = architecture_reader
    layers, repeats, _ = ARCHITECTURES[name]
    fast = reader.empty_fast()
    pointers = {value.data_ptr(): layer for layer, value in fast.items()}
    seen = []
    original = torch.matmul

    def traced(left, right, *args, **kwargs):
        if right.data_ptr() in pointers:
            seen.append(pointers[right.data_ptr()])
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", traced)
    reader(torch.tensor([[2, 7, 8, 4]]), fast=fast)
    assert seen == list(range(layers)) * repeats


def test_causality_and_one_committed_write_per_physical_layer(architecture_reader):
    name, reader = architecture_reader
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    changed = x.clone()
    changed[:, 5:] = torch.tensor([[19, 20]])
    fast = reader.empty_fast()
    captures = []
    for packet in (x, changed):
        capture = {"student_window": 2, "positions": torch.tensor([4])}
        reader(packet, fast=fast, window=7, capture=capture)
        captures.append(capture["features"])
    assert len(captures[0]) == ARCHITECTURES[name][0]
    for layer in captures[0]:
        for left, right in zip(captures[0][layer], captures[1][layer], strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    positions = torch.tensor([[1, 4]])
    torch.testing.assert_close(
        reader(x, positions, fast=fast), reader(changed, positions, fast=fast), rtol=0, atol=0
    )
    capture = {"student_window": 2, "positions": torch.tensor([6])}
    reader(x, fast=fast, window=7, capture=capture)
    before, base = tensor_digest(fast), tensor_digest(reader.backbone.state_dict())
    written, info = reader.write(x, fast, teacher_window=7, student_window=2, learning_rate=0.001)
    assert info["layers_written"] == ARCHITECTURES[name][0]
    for layer, features in capture["features"].items():
        torch.testing.assert_close(written[layer], local_ttcd_update(*features, 0.001))
    assert tensor_digest(fast) == before
    assert tensor_digest(reader.backbone.state_dict()) == base


def test_architecture_specific_fast_state_reload_is_exact(architecture_reader, tmp_path):
    name, reader = architecture_reader
    fast, _ = reader.write(
        torch.tensor([[2, 8, 11, 4, 3, 5, 6]]),
        reader.empty_fast(),
        teacher_window=7,
        student_window=2,
        learning_rate=0.001,
    )
    path = tmp_path / "fast.pt"
    save_fast(path, reader, fast, {"engineering_architecture": name})
    fresh = copy.deepcopy(reader)
    restored, provenance = load_fast(path, fresh)
    assert provenance == {"engineering_architecture": name}
    assert tensor_digest(restored) == tensor_digest(fast)
    query = torch.tensor([[1, 7, 5]])
    torch.testing.assert_close(
        reader(query, fast=fast), fresh(query, fast=restored), rtol=0, atol=0
    )
    other = make_reader("loop4x2" if name == "d8" else "d8")
    with pytest.raises(ValueError, match="different backbone"):
        load_fast(path, other)


@pytest.mark.parametrize("architecture", tuple(ARCHITECTURES))
def test_custom_loader_identity_hooks_support_all_architectures(
    tmp_path, monkeypatch, architecture
):
    config, settings, world, _ = double_material.__wrapped__(tmp_path)
    reader = make_reader(architecture, vocab_size=world.metadata["vocab_size"], dtype=torch.float32)
    checkpoint = tmp_path / "engineering-checkpoint.pt"
    torch.save({"model": reader.backbone.state_dict()}, checkpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("Architecture adapter must not call the old DiscoLoop-only loader")

    monkeypatch.setattr(verified, "load_reader", forbidden)
    monkeypatch.setattr(verified, "evaluation_identity", forbidden)

    def loader(config, name, device):
        assert name == architecture and str(device) == "cpu"
        fresh = make_reader(name, vocab_size=world.metadata["vocab_size"], dtype=torch.float32)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fresh.backbone.load_state_dict(saved["model"], strict=True)
        return fresh, world

    def parent_identity(config, name, panel):
        layers, repeats, disco = ARCHITECTURES[name]
        return {
            "checkpoint_sha256": sha256(checkpoint),
            "name": name,
            "physical_layers": layers,
            "repeats": repeats,
            "discoloop": disco,
            "engineering_fixture_only": True,
        }

    identity_factory = partial(evaluation.identity, parent_identity=parent_identity)
    panel = evaluation.build_panel(config, settings, world)
    panel_path = tmp_path / "supplement-panel.json"
    panel_path.write_text(json.dumps(panel))
    out = tmp_path / "evaluation"
    out.mkdir()
    base = model_digest(reader.backbone)
    result = evaluation.evaluate(
        config,
        settings,
        architecture,
        out,
        panel_path,
        "cpu",
        reader_loader=loader,
        identity_factory=identity_factory,
    )
    replay = evaluation.audit(
        config,
        settings,
        architecture,
        out,
        panel_path,
        "cpu",
        reader_loader=loader,
        identity_factory=identity_factory,
    )
    assert result["identity"]["parent"]["physical_layers"] == ARCHITECTURES[architecture][0]
    assert replay["passed"] and replay["write_states_exact"] and replay["states"] == 81
    assert result["primary"]["eligible_pair_readouts"] == 32
    assert "earlier_written_atoms_accuracy" in result["primary"]
    assert "earlier_atoms_retention_from_immediate" in result["primary"]
    assert set(result["primary"]["U"]) == {"U_atoms", "U_chains"}
    assert model_digest(reader.backbone) == base
    # Neither saved-state shape nor count may accidentally become the Loop layout.
    saved = torch.load(out / "states/main/r000/t008/fast.pt", weights_only=True)
    assert len(saved["fast"]) == ARCHITECTURES[architecture][0]


def test_existing_native_data_and_exact_supplement_panel_are_architecture_independent():
    config = json.loads((ROOT / "configs/ttcd-successful-base-v1.json").read_text())
    data = ROOT / config["data_dir"]
    required = [data / n for n in ("world.npz", "panels.npz", "training-split.json")]
    if not all(path.exists() for path in required):
        pytest.skip("Original native data are not installed in this checkout")
    config["data_dir"] = str(data)
    world = World(data)
    assert sha256(data / "world.npz") == (
        "e31ec9959ce0a7b25465191a1d9ef8e83ec42163ed967fa56a60605f6ac1c3b4"
    )
    assert sha256(data / "panels.npz") == (
        "8655858c0de4646cfb5c7ea1779b5374ef28b492e6ac59bb6eda9e89c7936bf9"
    )
    settings = json.loads((ROOT / "configs/ttcd-double-new-eval-v1.json").read_text())
    frozen = None
    for architecture, (layers, repeats, disco) in ARCHITECTURES.items():
        candidate = copy.deepcopy(config)
        candidate["architecture"] = architecture
        candidate["models"] = {
            architecture: {"unique_layers": layers, "repeats": repeats, "discoloop": disco}
        }
        panel = evaluation.build_panel(candidate, settings, world)
        payload = (json.dumps(panel, indent=2) + "\n").encode()
        assert hashlib.sha256(payload).hexdigest() == PANEL_FILE_SHA256
        assert len(panel["facts"]) == 32 and len(panel["pairs"]) == 16
        if frozen is None:
            frozen = panel
        else:
            assert panel == frozen
