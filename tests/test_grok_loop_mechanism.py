"""Contracts for prefix-only, component-specific loop interventions."""

import copy
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability import grok_loop_mechanism as mechanism
from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.grok_multihop_data import audit_world, build_world


def small_model(layers=2, repeats=2):
    torch.manual_seed(733)
    return LoopGPT(
        ModelConfig(vocab_size=32, width=16, layers=layers, heads=2, context=8),
        repeats=repeats,
        dropout=0.3,
    ).eval()


def true_candidates(world, row, split):
    """Independently enumerate candidates by graph traversal and complete query."""
    atoms = world["ood_atomic" if split == "ood_composite" else "id_atomic"]
    graph = {(int(h), int(r)): int(t) for h, r, t in atoms}
    trained = {tuple(map(int, r[:-1])) for r in world["train_composite"]}
    head, r1, target = int(row[0]), int(row[1]), int(row[-1])
    bridge = graph[head, r1]
    found = {"different": set(), "same": set()}
    for (dh, dr), db in graph.items():
        if dr != r1 or dh == head:
            continue
        tail = db
        for relation in row[2:-1]:
            tail = graph.get((tail, int(relation)))
            if tail is None:
                break
        if tail is None:
            continue
        if db == bridge and dh != target:
            found["same"].add((dh, dr, db, tail))
        if (
            db != bridge
            and tail != target
            and dh not in (target, tail)
            and (dh, *map(int, row[1:-1])) not in trained
        ):
            found["different"].add((dh, dr, db, tail))
    return found


@pytest.mark.parametrize("hops", [2, 3, 4])
@pytest.mark.parametrize("split", mechanism.SPLITS)
def test_donors_follow_same_relation_true_same_class_paths_and_never_train_query(hops, split):
    world = build_world(
        87,
        hops=hops,
        entities=12,
        relations=4,
        degree=3,
        phi=0.4,
        id_fraction=0.7,
        id_test_fraction=0.3,
        evaluation_size=24,
    )
    donors = mechanism.select_donors(world, 883, split)
    np.testing.assert_array_equal(donors["original_rows"], world[split])
    for i, row in enumerate(world[split]):
        candidates = true_candidates(world, row, split)
        for family in ("different", "same"):
            assert donors[family + "_candidate_count"][i] == len(candidates[family])
            assert donors[family + "_valid"][i] == bool(candidates[family])
            if not candidates[family]:
                np.testing.assert_array_equal(donors[family + "_donor"][i], -1)
                assert donors[family + "_reason"][i] != "eligible"
                continue
            donor = tuple(donors[family + "_donor"][i])
            cf = donors[family + "_counterfactual_rows"][i]
            assert (*donor, cf[-1]) in candidates[family]
            np.testing.assert_array_equal(cf[1:-1], row[1:-1])
            indices = donors[family + "_atomic_indices"][i]
            facts = world["atomic"][indices]
            assert facts[0, 0] == donor[0] and facts[-1, -1] == cf[-1]
            np.testing.assert_array_equal(facts[:, 1], cf[1:-1])
            np.testing.assert_array_equal(facts[:-1, -1], facts[1:, 0])
            category = donors[family + "_counterfactual_split"][i]
            if family == "different":
                assert category != "train"
            if split == "ood_composite":
                assert category == "ood"


def test_donor_selection_is_behavior_independent_and_deterministic():
    world = build_world(71, hops=2, entities=12, relations=4, degree=3, phi=0.4)
    before = copy.deepcopy(world)
    first = mechanism.select_donors(world, 7)
    np.random.seed(90)
    torch.manual_seed(81)
    np.random.random(300)
    second = mechanism.select_donors(world, 7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for key in world:
        if isinstance(world[key], np.ndarray):
            np.testing.assert_array_equal(world[key], before[key])
        else:
            assert world[key] == before[key]


@pytest.mark.parametrize("layers,repeats", [(1, 1), (1, 4), (2, 2)])
def test_trace_and_every_self_patch_match_native_forward_exactly(layers, repeats):
    model = small_model(layers, repeats)
    tokens = torch.tensor([[2, 20, 21, 5], [3, 21, 20, 6]])
    positions = torch.tensor([[2, 3], [2, 3]])
    expected = model(tokens, positions)
    logits, trace = mechanism.traced_forward(model, tokens, positions, return_trace=True)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert len(trace) == layers * repeats
    for layer in range(layers * repeats):
        for component in mechanism.COMPONENTS:
            identity = mechanism.traced_forward(
                model, tokens, positions, patch_layer=layer, component=component, identity=True
            )
            torch.testing.assert_close(identity, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hops", [2, 3, 4])
def test_prefix_trace_has_only_fixed_padding_and_cannot_see_any_suffix(hops):
    model = small_model()
    prefixes = torch.tensor([[2, 20], [3, 21]])
    expected = mechanism.prefix_trace(model, prefixes, hops + 2)
    arbitrary = torch.cat([prefixes, torch.randint(2, 32, (2, hops))], dim=1)
    _, full = mechanism.traced_forward(model, arbitrary, return_trace=True)
    for a, b in zip(expected, full, strict=True):
        for field in mechanism.TRACE_FIELDS:
            torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)


def test_only_first_layer_same_relation_both_patch_equals_full_residual():
    model = small_model()
    tokens = torch.tensor([[2, 20, 21, 5], [3, 21, 20, 6]])
    donor = mechanism.prefix_trace(model, torch.tensor([[8, 20], [9, 21]]), 4)
    both = mechanism.traced_forward(
        model, tokens, patch_layer=0, component="both", donor_trace=donor
    )
    full = mechanism.traced_forward(
        model, tokens, patch_layer=0, component="full", donor_trace=donor
    )
    torch.testing.assert_close(both, full, rtol=0, atol=0)
    both_later = mechanism.traced_forward(
        model, tokens, patch_layer=1, component="both", donor_trace=donor
    )
    full_later = mechanism.traced_forward(
        model, tokens, patch_layer=1, component="full", donor_trace=donor
    )
    assert (both_later[:, 2:] - full_later[:, 2:]).abs().max() > 1e-6


@pytest.mark.parametrize("component", mechanism.COMPONENTS)
def test_component_patch_matches_explicit_mix_without_recomputing_current_mlp(component):
    model = small_model()
    tokens = torch.tensor([[2, 20, 21, 5], [3, 21, 20, 6]])
    donor = mechanism.prefix_trace(model, torch.tensor([[8, 20], [9, 21]]), 4)
    original_tokens = tokens.clone()
    saved_donor = copy.deepcopy(donor)
    x = model.token(tokens) + model.position(torch.arange(4))
    block = model.blocks[0]
    attention = block.attention(block.ln1(x))
    mlp = block.mlp(block.ln2(x + attention))
    expected = (x + attention) + mlp
    if component == "full":
        replacement = donor[0]["residual"]
    else:
        a = donor[0]["attention"] if component in ("both", "attention") else attention[:, 1]
        m = donor[0]["mlp"] if component in ("both", "mlp") else mlp[:, 1]
        replacement = (x[:, 1] + a) + m
    expected[:, 1] = replacement
    for next_block in list(model.iter_blocks())[1:]:
        expected = next_block(expected)
    expected = F.linear(model.ln_final(expected), model.token.weight)
    calls = Counter()

    def count(module, inputs, output):
        calls[id(module)] += 1

    hooks = [block.mlp.register_forward_hook(count) for block in model.blocks]
    try:
        actual = mechanism.traced_forward(
            model, tokens, patch_layer=0, component=component, donor_trace=donor
        )
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(calls[id(block.mlp)] == model.repeats for block in model.blocks)
    torch.testing.assert_close(tokens, original_tokens, rtol=0, atol=0)
    for a, b in zip(donor, saved_donor, strict=True):
        for field in mechanism.TRACE_FIELDS:
            torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)


def test_final_execution_patch_cannot_change_later_answer_or_generated_eos():
    model = small_model()
    rows = np.array([[2, 20, 21, 5], [3, 21, 20, 6]], dtype=np.int64)
    prefixes = np.array([[8, 20], [9, 21]], dtype=np.int64)
    baseline, _ = mechanism.evaluate_condition(model, rows, "cpu", 4)
    for component in mechanism.COMPONENTS:
        changed, _ = mechanism.evaluate_condition(
            model,
            rows,
            "cpu",
            4,
            donor_prefixes=prefixes,
            patch_layer=3,
            component=component,
        )
        for key in ("answer", "stop", "answer_logits"):
            np.testing.assert_array_equal(changed[key], baseline[key])


def test_generated_answer_is_used_for_eos_and_identical_donor_trace_is_reapplied(monkeypatch):
    model = small_model()
    calls = []

    def forward(model, tokens, positions=None, **kwargs):
        calls.append((tokens.clone(), kwargs))
        logits = torch.full((len(tokens), positions.shape[1], 32), -10.0)
        logits[:, 0, 7] = 10.0
        if positions.shape[1] == 2:
            logits[torch.arange(len(tokens)), 1, (tokens[:, 3] == 7).long()] = 10.0
        trace = [{key: torch.ones((len(tokens), 16)) for key in mechanism.TRACE_FIELDS}] * 4
        return (logits, trace) if kwargs.get("return_trace") else logits

    monkeypatch.setattr(mechanism, "traced_forward", forward)
    rows = np.array([[2, 20, 21, 5], [3, 21, 20, 6]], dtype=np.int64)
    before = rows.copy()
    prefixes = np.array([[8, 20], [9, 21]], dtype=np.int64)
    result, _ = mechanism.evaluate_condition(
        model, rows, "cpu", 4, donor_prefixes=prefixes, patch_layer=1, component="attention"
    )
    np.testing.assert_array_equal(result["answer"], 7)
    np.testing.assert_array_equal(result["stop"], 1)
    torch.testing.assert_close(calls[0][0][:, :2], torch.tensor(prefixes), rtol=0, atol=0)
    assert not calls[0][0][:, 2:].any()
    assert calls[1][1]["donor_trace"] is calls[2][1]["donor_trace"]
    assert calls[1][1]["patch_layer"] == calls[2][1]["patch_layer"] == 1
    assert torch.equal(calls[2][0][:, -1], torch.tensor([7, 7]))
    np.testing.assert_array_equal(rows, before)


@pytest.mark.parametrize("split", ["test_composite", "ood_composite"])
def test_run_preserves_full_selected_split_masks_and_prerequisite_denominators(split):
    world = build_world(
        87,
        hops=2,
        entities=12,
        relations=4,
        degree=3,
        phi=0.4,
        id_fraction=0.7,
        id_test_fraction=0.3,
        evaluation_size=24,
    )
    donors = mechanism.select_donors(world, 883, split)
    model = small_model(1, 2)
    summary, arrays = mechanism.evaluate_run(model, world, donors, "cpu", batch_size=8)
    n = len(world[split])
    assert summary["split"] == split
    assert summary["executed_depth"] == 2
    for name, description in summary["conditions"].items():
        family = description["family"]
        expected = np.ones(n, dtype=bool) if family is None else donors[family + "_valid"]
        np.testing.assert_array_equal(arrays[name + "_valid"], expected)
        for key in ("answer", "stop"):
            assert arrays[name + "_" + key].shape == (n,)
            np.testing.assert_array_equal(arrays[name + "_" + key][~expected], -1)
        metrics = summary["scores"]["all_rows"]["conditions"][name]
        assert metrics["n"] == expected.sum() and metrics["total_n"] == n
    for family in ("original", "different", "same"):
        expected = n if family == "original" else int(donors[family + "_valid"].sum())
        assert summary["atomic_preconditions"][family]["n"] == expected
    missing = ~donors["different_valid"]
    missing_baseline = summary["scores"]["different_donor_missing"]["conditions"]["baseline"]
    assert missing_baseline["n"] == missing.sum()


def test_zero_donors_and_empty_split_are_valid_reported_outcomes():
    world = build_world(
        91,
        hops=2,
        entities=1,
        relations=1,
        degree=1,
        phi=0,
        id_fraction=1,
        id_test_fraction=1,
    )
    for split in ("test_composite", "ood_composite"):
        donors = mechanism.select_donors(world, 10, split)
        assert not donors["different_valid"].any()
        summary, arrays = mechanism.evaluate_run(small_model(1, 1), world, donors, "cpu")
        metrics = summary["scores"]["all_rows"]["conditions"]["e000_different_both"]
        assert metrics["n"] == 0 and metrics["target_complete_accuracy"] is None
        assert len(arrays["baseline_answer"]) == len(world[split])


def test_invalid_interventions_are_rejected():
    model = small_model()
    tokens = torch.tensor([[2, 20, 21, 5]])
    for layer in (-1, 4, True, 1.2):
        with pytest.raises(ValueError, match="patch_layer"):
            mechanism.traced_forward(model, tokens, patch_layer=layer, identity=True)
    with pytest.raises(ValueError, match="component"):
        mechanism.traced_forward(model, tokens, component="typo")
    with pytest.raises(ValueError, match="exactly one"):
        mechanism.traced_forward(model, tokens, patch_layer=1)
    with pytest.raises(ValueError, match="eval"):
        mechanism.traced_forward(model.train(), tokens)


@pytest.fixture
def frozen_toy_run(tmp_path):
    """Untrained synthetic checkpoint for source/reload/CLI engineering tests."""
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/analyze_grok_loop_mechanism.py"
    module_spec = importlib.util.spec_from_file_location("loop_mechanism_cli_for_test", script)
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    run = tmp_path / "toy"
    snapshot = run / "source"
    spec = {
        "entities": 12,
        "relations": 4,
        "width": 16,
        "layers": 1,
        "heads": 2,
        "repeats": 2,
        "dropout": 0.1,
        "init_scheme": "scaled_effective",
        "steps": 0,
        "weight_nodes": [0],
    }
    files = {}
    for name in (
        "bios_model",
        "grok_depth",
        "grok_depth_data",
        "grok_multihop_data",
        "grok_loop_model",
    ):
        relative = Path("src/llm_memory_editability") / (name + ".py")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
        files[str(root / relative)] = cli.digest(destination)
    config = snapshot / "configs/toy.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"base": spec, "runs": {"toy": {}}}))
    files[str(root / "configs/toy.json")] = cli.digest(config)
    (run / "metadata.json").write_text(json.dumps({"spec": spec, "files": files}))
    world = build_world(
        87,
        hops=2,
        entities=12,
        relations=4,
        degree=3,
        phi=0.4,
        id_fraction=0.7,
        id_test_fraction=0.3,
        evaluation_size=4,
    )
    np.savez_compressed(run / "world.npz", **{k: v for k, v in world.items() if k != "metadata"})
    (run / "world-metadata.json").write_text(json.dumps(world["metadata"]))
    (run / "data-audit.json").write_text(json.dumps(audit_world(world)))
    model = LoopGPT(
        ModelConfig(vocab_size=18, width=16, layers=1, heads=2, context=8),
        repeats=2,
    ).eval()
    torch.save({"model": model.state_dict(), "spec": spec, "step": 0}, run / "weights-0000000.pt")
    return cli, run, model


def test_frozen_loader_rejects_changed_sources_and_checkpoint_identity(frozen_toy_run):
    cli, run, original = frozen_toy_run
    world, spec, provenance, package = cli.load_source_world(run)
    assert provenance["dataset_sha256"] == world["metadata"]["dataset_sha256"]
    restored, checkpoint_provenance = cli.load_frozen_model(run, 0, spec, package, "cpu")
    tokens = torch.tensor([[2, 14, 15, 5]])
    torch.testing.assert_close(restored(tokens), original(tokens), rtol=0, atol=0)
    assert checkpoint_provenance["checkpoint_step"] == 0
    with pytest.raises(ValueError, match="registered"):
        cli.load_frozen_model(run, 1, spec, package, "cpu")
    path = run / "weights-0000000.pt"
    saved = torch.load(path, weights_only=False)
    saved["step"] = 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="identity"):
        cli.load_frozen_model(run, 0, spec, package, "cpu")
    source = run / "source/src/llm_memory_editability/grok_loop_model.py"
    source.write_text(source.read_text() + "\n# tampered source\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        cli.load_source_world(run)


def test_cli_synthetic_checkpoint_writes_complete_id_and_ood_artifacts(
    frozen_toy_run, monkeypatch, tmp_path
):
    cli, run, _ = frozen_toy_run
    out = tmp_path / "analysis"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_grok_loop_mechanism.py",
            str(run),
            "--device",
            "cpu",
            "--out",
            str(out),
            "--split",
            "test_composite",
            "--split",
            "ood_composite",
            "--batch-size",
            "8",
        ],
    )
    cli.main()
    for split in ("test_composite", "ood_composite"):
        directory = out / "toy" / f"step-0000000-{split}"
        assert json.loads((directory / "status.json").read_text())["state"] == "complete"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["split"] == split
        assert summary["engineering_checks"]["identity_all_layers_components"]
        assert (directory / "donors.npz").is_file()
        assert (directory / "predictions.npz").is_file()
        assert (directory / "analysis-source/scripts/analyze_grok_loop_mechanism.py").is_file()
