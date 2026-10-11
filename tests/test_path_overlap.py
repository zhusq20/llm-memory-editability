"""Contracts of the zero-training path-overlap (Z1) code."""

import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import ReproductionGPT, build_graph
from llm_memory_editability.path_overlap import (
    CODE_FILES,
    Executor,
    World,
    bootstrap_mean,
    contract_parameters,
    load_contract,
    load_model,
    majority_verdict,
    ported_node,
    predict,
    random_units,
    replication,
    sample_facts,
    sha256,
    window_edges,
    z1_node,
)

PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT / "configs/zero-training-path-overlap-v1.json"
RUN = "loop4x2-phi7.2-wd0.1"
STEP = 1500000


def tiny_model(vocab, unique, repeats, seed=0, width=32):
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=vocab,
        n_embd=width,
        n_layer=unique,
        n_head=4,
        n_positions=8,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = ReproductionGPT(cfg, repeats)
    return model.eval().requires_grad_(False)


def load_runner():
    path = PROJECT / "scripts/run_zero_training_path_overlap.py"
    spec = importlib.util.spec_from_file_location("run_zero_training_path_overlap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def params():
    _, p = load_contract(CONTRACT)
    return p


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A 40-entity graph and a small Loop 2x2 trained on its atoms and II chains."""
    data = tmp_path_factory.mktemp("data")
    graph = build_graph(920, entities=40, relations=24, degree=4)  # seed with nonempty OO
    np.savez(data / "world.npz", **graph)
    atoms, chains, kind = graph["atoms"], graph["chains"], graph["kind"]
    panels = {
        "atomic_id": atoms[~graph["ood"]][:60],
        "atomic_ood": atoms[graph["ood"]],
        "train_ii": chains[graph["train_order"][:60]],
        "test_ii": chains[graph["train_order"][-60:]],
        "test_io": chains[kind == 1][:60],
        "test_oi": chains[kind == 2][:60],
        "test_oo": chains[kind == 3][:60],
    }
    np.savez(data / "panels.npz", **panels)
    metadata = {
        "entity_offset": 3,
        "relation_offset": 43,
        "entities": 40,
        "relations": 24,
        "atomic_id": int((~graph["ood"]).sum()),
        "vocab_size": 3 + 40 + 24 + 2,
    }
    (data / "complete.json").write_text(json.dumps(metadata))
    world = World(data)
    phi = 2.0
    model = tiny_model(metadata["vocab_size"], 2, 2, seed=1, width=48)
    model.train().requires_grad_(True)
    support = world.train_order[: round(phi * metadata["atomic_id"])]
    x_atom = torch.as_tensor(world.atomic_tokens(np.arange(len(atoms))))
    y_atom = torch.as_tensor(world.atoms[:, 2] + world.eo)
    x_chain = torch.as_tensor(world.chain_tokens(support))
    y_chain = torch.as_tensor(world.chains[support, 3] + world.eo)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(400):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x_atom)[:, 1], y_atom) + F.cross_entropy(
            model(x_chain)[:, 2], y_chain
        )
        loss.backward()
        optimizer.step()
    model.eval().requires_grad_(False)
    return {
        "data": data,
        "world": world,
        "model": model,
        "phi": phi,
        "metadata": metadata,
        "graph": graph,
    }


def small(params):
    p = copy.deepcopy(params)
    p["z1"].update(
        n_id=6,
        n_ood=3,
        min_chains=1,
        chains_per_fact=4,
        same_answer=4,
        random_atoms=8,
        k_grid=[1, 2, 4, 8, 16, 32, 64],
        same_bridge=2,
        bootstrap={"iterations": 200, "seed": 1, "interval": [0.025, 0.975]},
        patch={"id": 40, "ood": 20, "seed": 3},
    )
    return p


def checkpoint_root(tiny, root):
    """A grokking-reproduction-v1 layout holding the tiny model as RUN at STEP."""
    (root / "development" / RUN).mkdir(parents=True)
    shutil.copytree(tiny["data"], root / "data")
    tiny["model"].config.save_pretrained(root / "reference/gpt2")
    spec = {"name": RUN, "unique_layers": 2, "repeats": 2, "phi": tiny["phi"]}
    path = root / "development" / RUN / f"checkpoint-{STEP:07d}.pt"
    torch.save({"model": tiny["model"].state_dict(), "spec": spec, "step": STEP}, path)
    ex = Executor(tiny["model"])
    world = tiny["world"]
    for name, rows in world.panels.items():
        position = 1 if rows.shape[1] == 3 else 2
        np.savez(
            path.parent / f"predictions-{STEP:07d}-{name}.npz",
            answer=predict(ex, world.row_tokens(rows), position),
            rows=rows,
        )
    return spec, path


def test_contract_parameters_are_read_from_the_frozen_text(params):
    assert params["z1"]["k_grid"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert params["z1"]["fact_seed"] == 20261011 and params["z1"]["retention"] == 0.9
    assert params["z1"]["empty_control_passes"] is True
    assert (
        params["replication"]["from_step"] == 16000 and params["replication"]["early_stop"] == 0.5
    )
    assert params["z1"]["loop_node_count"] == 14 and params["z1"]["standard_node_count"] == 22
    assert params["code_files"] == CODE_FILES
    assert params["outputs"] == ("z1.json", "ported.json", "hashes.json")
    assert "z2" not in params


def test_contract_guard_rejects_other_contracts():
    contract = json.loads(CONTRACT.read_text())
    with pytest.raises(ValueError, match="Expected the"):
        contract_parameters({**contract, "batch": "zero-training-path-format-v1"})
    with pytest.raises(ValueError, match="Z2 must be recorded as withdrawn"):
        contract_parameters({**contract, "z2": {}})
    with pytest.raises(ValueError, match="Z2 must be recorded as withdrawn"):
        contract_parameters({**contract, "withdrawn": {}})


def test_runner_paths_follow_the_contract(params):
    runner = load_runner()
    assert runner.CONTRACT == CONTRACT
    assert params["output_dir"] == f"{runner.OUT.as_posix()}/{{run}}/{{step:07d}}/"
    for name in CODE_FILES:
        assert (PROJECT / name).exists(), name


@pytest.mark.parametrize("unique,repeats", [(4, 1), (2, 2)])
def test_executor_without_interventions_matches_native_forward(unique, repeats):
    model = tiny_model(60, unique, repeats)
    ex = Executor(model)
    tokens = torch.randint(0, 60, (7, 3))
    native = model(tokens)
    ours = ex.readout(ex.run(ex.embed(tokens), range(ex.executions)))
    assert torch.allclose(native, ours, atol=1e-5)


def test_zero_gate_equals_zeroing_c_proj_rows():
    model = tiny_model(60, 2, 2, seed=4)
    ex = Executor(model)
    tokens = torch.randint(0, 60, (5, 3))
    units = np.array([[3, 130, 200, -1]] * 5)
    gated = ex.logits(
        tokens, 2, gates=ex.gates_for(torch.as_tensor(units), 5), gate_scope=ex.scope("all")
    )
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for u in (3, 130, 200):
            ablated.transformer.h[u // ex.neurons].mlp.c_proj.weight[u % ex.neurons] = 0.0
    assert torch.allclose(gated, ablated(tokens)[:, 2], atol=1e-5)


def test_round2_scope_leaves_round1_states_unchanged():
    model = tiny_model(60, 2, 2, seed=5)
    ex = Executor(model)
    tokens = torch.randint(0, 60, (5, 3))
    gates = ex.gates_for(torch.arange(0, 200, 2).repeat(5, 1), 5)
    _, base = ex.run(ex.embed(tokens), range(4), capture=True)
    _, late = ex.run(
        ex.embed(tokens), range(4), gates=gates, gate_scope=ex.scope("round2"), capture=True
    )
    assert all(torch.equal(a, b) for a, b in zip(base[:2], late[:2], strict=True))
    assert not torch.allclose(base[3], late[3])
    assert ex.scope("round1") == {0, 1} and ex.scope("round2") == {2, 3}


def test_attention_edge_masks_follow_the_author_window():
    assert window_edges(8, 0, 3, (2, 1)) == {0: [(2, 1)], 1: [(2, 1)]}
    assert window_edges(8, 7, 3, (2, 1)) == {6: [(2, 1)], 7: [(2, 1)]}
    assert window_edges(8, 4, 1, (1, 0)) == {4: [(1, 0)]}
    model = tiny_model(60, 2, 2, seed=6)
    ex = Executor(model)
    tokens = torch.randint(0, 60, (5, 3))
    x0 = ex.embed(tokens)
    assert torch.equal(ex.run(x0, range(4)), ex.run(x0, range(4), edges={}))
    masked = ex.run(x0, range(4), edges={e: [(1, 0)] for e in range(4)})
    native = ex.run(x0, range(4))
    assert torch.allclose(masked[:, 0], native[:, 0]) and not torch.allclose(
        masked[:, 1], native[:, 1]
    )


def test_random_control_matches_block_counts_and_excludes_selected_units():
    units = np.array([1, 5, 300, 301, 302])
    control = random_units(units, 256, 4, 20261013, 17)
    assert len(control) == len(units) and not set(control) & set(units)
    assert np.array_equal(
        np.bincount(control // 256, minlength=4), np.bincount(units // 256, minlength=4)
    )
    assert np.array_equal(control, random_units(units, 256, 4, 20261013, 17))


def test_selection_never_reads_two_hop_outcomes(tiny, params):
    p = small(params)
    world, ex = tiny["world"], Executor(tiny["model"])
    sample = sample_facts(world, tiny["phi"], p)
    assert sample["sha256"] == sample_facts(world, tiny["phi"], p)["sha256"]
    atom_answer = predict(ex, world.atomic_tokens(np.arange(len(world.atoms))), 1)
    first = z1_node(ex, world, sample, atom_answer, p, ["all", "round1", "round2"])
    shuffled = dict(sample)
    rng = np.random.default_rng(0)
    for key in ("chains", "trained", "same_bridge"):
        shuffled[key] = [rng.choice(len(world.chains), len(c), replace=False) for c in sample[key]]
    second = z1_node(ex, world, shuffled, atom_answer, p, ["all", "round1", "round2"])
    fields = ("k_star", "units", "retention", "selective", "selective_secondary", "control_units")
    for a, b in zip(first["facts"], second["facts"], strict=True):
        assert all(a[f] == b[f] for f in fields)
    assert any(r["k_star"] for r in first["facts"]), "the tiny model should localize some facts"
    for r in first["facts"]:
        for key, value in r.items():
            if key.startswith("untrained_") and key[-6:] != "margin" and isinstance(value, float):
                assert 0.0 <= value <= 1.0
    block = first["kinds"]["id"]["primary/untrained"]
    assert set(block) >= {"facts_with_eligible", "all", "round1", "round2"}


def test_ported_diagnostics_run(tiny, params):
    p = small(params)
    world, ex = tiny["world"], Executor(tiny["model"])
    ported = ported_node(ex, world, tiny["phi"], p)
    assert len(ported["logit_lens"]["test_ii"]) == ex.executions
    assert len(ported["bridge_patching"]["id"]["accuracy_by_layer"]) == ex.executions
    assert np.asarray(ported["layerwise_cosine"]["id"]["entity"]).shape == (ex.executions + 1,) * 2


def test_bootstrap_interval():
    spec = {"iterations": 2000, "seed": 7, "interval": [0.025, 0.975]}
    interval = bootstrap_mean([0.3] * 10, spec)
    assert interval["low"] == interval["high"] == pytest.approx(0.3)
    spread = bootstrap_mean(np.r_[np.zeros(50), np.ones(50)], spec)
    assert spread["low"] < 0.5 < spread["high"]
    assert bootstrap_mean([], spec) is None


def test_majority_verdict_on_constructed_inputs():
    assert majority_verdict(5, 0, 6, 14) == "inconclusive"
    assert majority_verdict(8, 1, 10, 14) == "supports"
    assert majority_verdict(1, 8, 10, 14) == "against"
    assert majority_verdict(5, 5, 10, 14) == "partial"


def test_checkpoint_loading_and_replication_check(tiny, params, tmp_path):
    root = tmp_path / "root"
    spec, path = checkpoint_root(tiny, root)
    world = tiny["world"]
    model, loaded_spec, _ = load_model(root, spec["name"], STEP, tiny["metadata"], "cpu")
    ex = Executor(model)
    _, failures = replication(ex, world, root, spec["name"], STEP, params)
    assert failures == [] and loaded_spec == spec
    stored = dict(np.load(path.parent / f"predictions-{STEP:07d}-test_ii.npz"))
    stored["answer"] = stored["answer"] + 1
    np.savez(path.parent / f"predictions-{STEP:07d}-test_ii.npz", **stored)
    _, failures = replication(ex, world, root, spec["name"], STEP, params)
    assert any(f.startswith("test_ii") for f in failures)


def tiny_contract(root, path):
    """The frozen contract with sample sizes cut to the tiny graph and its data hashes."""
    contract = json.loads(CONTRACT.read_text())
    z1 = contract["z1"]
    sel = z1["selection_on_single_hop_only"]

    def cut(text, old, new):
        assert old in text, old
        return text.replace(old, new)

    z1["fact_sample"] = cut(z1["fact_sample"], "256 ID facts and 256 OOD", "6 ID facts and 3 OOD")
    z1["fact_sample"] = cut(z1["fact_sample"], ">= 3 eligible-kind", ">= 1 eligible-kind")
    z1["fact_sample"] = cut(z1["fact_sample"], "Up to 16 such chains", "Up to 4 such chains")
    sel["k_grid"] = [1, 2, 4, 8, 16, 32, 64]
    controls = sel["selectivity_controls"]
    controls["same_answer"] = cut(controls["same_answer"], "up to 16", "up to 4")
    controls["random"] = cut(controls["random"], "64 random", "8 random")
    two = z1["two_hop_measure"]
    two["delta"] = cut(two["delta"], "bootstrap 10000 draws", "bootstrap 200 draws")
    ported = z1["ported_diagnostics_descriptive"]
    ported["bridge_patching"] = cut(ported["bridge_patching"], "3000 ID and 3000", "40 ID and 20")
    contract["source"]["world_sha256"] = sha256(root / "data/world.npz")
    contract["source"]["panels_sha256"] = sha256(root / "data/panels.npz")
    path.write_text(json.dumps(contract))
    return path


def test_run_writes_exactly_the_contract_outputs(tiny, params, tmp_path):
    runner = load_runner()
    root = tmp_path / "root"
    checkpoint_root(tiny, root)
    contract = tiny_contract(root, tmp_path / "contract.json")
    out = tmp_path / "out"
    args = argparse.Namespace(
        contract=contract, root=root, out=out, run=RUN, steps=[STEP], gpu=None
    )
    runner.run(args)
    node = out / RUN / f"{STEP:07d}"
    assert sorted(p.name for p in node.iterdir()) == sorted(params["outputs"])
    hashes = json.loads((node / "hashes.json").read_text())
    assert hashes["complete"] is True and hashes["replication"]["failures"] == []
    assert set(hashes["outputs_sha256"]) == {"z1.json", "ported.json"}
    assert set(hashes["code_sha256"]) == set(CODE_FILES)
    assert all(v is not None for v in hashes["code_sha256"].values())
    z1 = json.loads((node / "z1.json").read_text())
    assert z1["run"] == RUN and z1["scopes"] == ["all", "round1", "round2"]
    runner.run(args)  # a complete node is skipped, not recomputed
    assert (
        json.loads((node / "hashes.json").read_text())["completed_utc"] == hashes["completed_utc"]
    )


def test_run_keeps_the_replication_failure_and_stops(tiny, tmp_path):
    runner = load_runner()
    root = tmp_path / "root"
    _, path = checkpoint_root(tiny, root)
    stored = dict(np.load(path.parent / f"predictions-{STEP:07d}-test_ii.npz"))
    stored["answer"] = stored["answer"] + 1
    np.savez(path.parent / f"predictions-{STEP:07d}-test_ii.npz", **stored)
    contract = tiny_contract(root, tmp_path / "contract.json")
    out = tmp_path / "out"
    args = argparse.Namespace(
        contract=contract, root=root, out=out, run=RUN, steps=[STEP], gpu=None
    )
    with pytest.raises(SystemExit, match="Replication check failed"):
        runner.run(args)
    node = out / RUN / f"{STEP:07d}"
    assert [p.name for p in node.iterdir()] == [runner.FAILURE]
    failure = json.loads((node / runner.FAILURE).read_text())
    assert any(f.startswith("test_ii") for f in failure["failures"])
