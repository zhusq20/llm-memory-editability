"""Graph truth, two-write necessity, continual retention and exact replay."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability import ttcd_double_new_eval as dn
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def material(tmp_path):
    entities, relations = 32, 3
    atoms = np.array(
        [[h, r, (h + 8 + r) % entities] for h in range(entities) for r in range(relations)]
    )
    first, second, chains = [], [], []
    for i, (h, r, b) in enumerate(atoms):
        for r2 in range(relations):
            j = int(b * relations + r2)
            first.append(i)
            second.append(j)
            chains.append([h, r, r2, atoms[j, -1]])
    world = SimpleNamespace(
        atoms=atoms,
        chains=np.asarray(chains),
        first=np.asarray(first),
        second=np.asarray(second),
        entities=entities,
        metadata={
            "entity_offset": 3,
            "relation_offset": entities + 3,
            "end_marker": 2,
            "pad_token": 0,
            "entities": entities,
            "relations": relations,
            "vocab_size": entities + relations + 3,
        },
    )
    world.support = lambda _: np.zeros(len(world.chains), dtype=bool)
    bank = [
        {
            "case_id": f"evaluation-{h}",
            "split": "evaluation",
            "group": "id" if h < 4 else "ood",
            "atom": h * relations,
            "new_tail": h + 17,
            "D": np.flatnonzero(world.second == h * relations).tolist(),
        }
        for h in range(8)
    ]
    (tmp_path / "training-split.json").write_text(
        json.dumps(
            {
                "case_bank": bank,
                "train_edit_atoms": [
                    i for i in range(len(atoms)) if i not in {f["atom"] for f in bank}
                ],
            }
        )
    )
    config = {
        "data_dir": str(tmp_path),
        "phi": 2.0,
        "continual": {
            "train_length": 2,
            "eval_length": 8,
            "rotations": 8,
            "affected": 2,
            "controls": 2,
            "order_seed": 1729,
        },
        "writer": {
            "teacher_window": 32,
            "student_window": 16,
            "gap": 17,
            "steps": 2,
            "learning_rate": 0.001,
        },
    }
    settings = {"pair_seed": 20261015, "pairs_per_type": 1, "rotations": 8, "primary_t": 8}
    torch.manual_seed(19)
    cfg = GPT2Config(
        vocab_size=world.metadata["vocab_size"],
        n_positions=40,
        n_embd=8,
        n_head=2,
        n_layer=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    reader = TTCDReader(
        SupervisedAlignmentGPT(cfg, repeats=2, discoloop=True), native_causal_attention=True
    ).eval()
    return config, settings, world, reader


def test_graph_matching_covers_all_reserved_keys_balanced_and_changes_both_edges(material):
    config, settings, world, _ = material
    before = copy.deepcopy(config)
    panel = dn.build_panel(config, settings, world)
    assert panel == dn.build_panel(config, settings, world)
    assert config == before
    assert sorted(i for p in panel["pairs"] for i in (p["first"], p["second"])) == list(range(8))
    assert sorted(p["type"] for p in panel["pairs"]) == ["II", "IO", "OI", "OO"]
    for pair in panel["pairs"]:
        a, b = [panel["facts"][pair[k]] for k in ("first", "second")]
        assert a["new_tail"] == world.atoms[b["atom"], 0]
        assert a["new_tail"] != a["old_tail"]
        assert b["new_tail"] != b["old_tail"]
        assert len({pair["row"][-1], pair["first_only_target"], pair["target"]}) == 3
        assert world.second[pair["old_chain_index"]] not in panel["edited_atoms"]


def test_graph_selection_rejects_training_overlap_or_insufficient_matching(material):
    config, settings, world, _ = material
    split_path = dn.Path(config["data_dir"]) / "training-split.json"
    split = json.loads(split_path.read_text())
    split["train_edit_atoms"].append(split["case_bank"][0]["atom"])
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="excluded"):
        dn.build_panel(config, settings, world)
    split["train_edit_atoms"].pop()
    split_path.write_text(json.dumps(split))
    world.support = lambda _: np.ones(len(world.chains), dtype=bool)
    # Parent D construction also refuses a panel with no held-out queries.
    with pytest.raises(ValueError):
        dn.build_panel(config, settings, world)


def test_state_matrix_has_both_omissions_and_isolated_pair_orders(material):
    config, settings, world, _ = material
    panel = dn.build_panel(config, settings, world)
    plans = dn.state_plan(panel)
    assert len(plans) == 1 + 8 + 2 * 4 + 8 * 8
    for p, pair in enumerate(panel["pairs"]):
        forward, reverse = [plans[f"pair/p{p:03d}/d{d}"] for d in (0, 1)]
        assert forward["indices"] == [pair["first"], pair["second"]]
        assert reverse["indices"] == [pair["second"], pair["first"]]
    for spec in plans.values():
        if spec["kind"] == "main":
            assert len(spec["indices"]) == spec["t"]
            assert spec["write"] == spec["indices"][-1]


def test_partial_chain_truth_and_read_only_state(material):
    config, settings, world, reader = material
    panel = dn.build_panel(config, settings, world)
    pair = panel["pairs"][0]
    fast = reader.empty_fast()
    before = tensor_digest(fast)
    for indices, target in (
        ([], pair["row"][-1]),
        ([pair["first"]], pair["first_only_target"]),
        ([pair["second"]], pair["row"][-1]),
        ([pair["first"], pair["second"]], pair["target"]),
    ):
        readout = dn.read_state(reader, fast, world, panel, {"indices": indices}, "cpu")
        assert readout["pairs"][0]["current_target"] == target
        assert readout["pairs"][0]["eligible"] == (len(indices) == 2)
        assert readout["atom_current_targets"] == [
            f["new_tail"] if i in indices else f["old_tail"] for i, f in enumerate(panel["facts"])
        ]
    assert tensor_digest(fast) == before


def test_complete_evaluation_and_fresh_reader_audit(material, tmp_path):
    config, settings, world, reader = material
    panel = dn.build_panel(config, settings, world)
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    out = tmp_path / "evaluation"
    out.mkdir()

    def loader(*args):
        return reader, world

    def identity(*args):
        return {"immutable_fixture": True}

    base_digest = tensor_digest(reader.backbone.state_dict())
    result = dn.evaluate(
        config,
        settings,
        "single",
        out,
        panel_path,
        "cpu",
        reader_loader=loader,
        identity_factory=identity,
    )
    fresh = copy.deepcopy(reader)
    replay = dn.audit(
        config,
        settings,
        "single",
        out,
        panel_path,
        "cpu",
        reader_loader=lambda *args: (fresh, world),
        identity_factory=identity,
    )
    assert replay["passed"] and replay["states"] == 81
    assert result["primary"]["eligible_pair_readouts"] == 8 * 4
    assert result["primary"]["total_pair_readouts"] == 8 * 4
    assert result["curves"]["1"]["eligible_pair_readouts"] == 0
    assert result["curves"]["1"]["two_new_accuracy"] is None
    assert result["scientific_training_updates"] == 0
    assert tensor_digest(reader.backbone.state_dict()) == base_digest
    # The audit must detect altered exact tokens and summary truth, not just hashes.
    path = out / "states/main/r000/t008/record.json"
    saved = json.loads(path.read_text())
    saved["readout"]["predictions"]["chains"]["answer"][0] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(AssertionError, match="Replay value changed"):
        dn.audit(
            config,
            settings,
            "single",
            out,
            panel_path,
            "cpu",
            reader_loader=lambda *args: (fresh, world),
            identity_factory=identity,
        )


def test_retention_excludes_initial_failures_and_reports_zero_coverage():
    result = dn._conditional(
        [{"immediate": True, "correct": False}, {"immediate": False, "correct": True}],
        "immediate",
        "correct",
    )
    assert result == {"total": 2, "eligible": 1, "coverage": 0.5, "accuracy": 0.0}
    empty = dn._conditional([{"immediate": False, "correct": True}], "immediate", "correct")
    assert empty["eligible"] == 0 and empty["accuracy"] is None


def test_default_parent_loader_and_independent_cli_processes(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from test_ttcd_verified_base import fixture

    from llm_memory_editability import ttcd_verified_base as verified

    root = Path(__file__).resolve().parents[1]
    config = fixture(tmp_path)
    config["cases"]["evaluation_cases"] = 4
    config["continual"].update(eval_length=8, rotations=8)
    config["data_dir"] = str(tmp_path / "expanded-data")
    config["model_config"] = str(tmp_path / "expanded-data/gpt2-config.json")
    verified.prepare_data(config)
    settings = {
        "batch": "double-new-engineering-only",
        "parent_batch": config["batch"],
        "pair_seed": 20261015,
        "pairs_per_type": 1,
        "rotations": 8,
        "primary_t": 8,
        "runtime": {"cpu_threads": 1},
    }
    # These four tiny CPU fixture updates are engineering, not scientific training.
    verified.train(config, "common", tmp_path / "common", "cpu")
    verified.audit_training(config, "common", tmp_path / "common", "cpu")
    assert verified.common_gate(config, tmp_path / "common", "cpu")["passed"]
    verified.train(config, "single", tmp_path / "single", "cpu")
    verified.audit_training(config, "single", tmp_path / "single", "cpu")
    base_path, settings_path = tmp_path / "base.json", tmp_path / "settings.json"
    base_path.write_text(json.dumps(config))
    settings_path.write_text(json.dumps(settings))
    panel_path, out = tmp_path / "panel.json", tmp_path / "supplement"
    for stage in ("prepare", "evaluate", "audit"):
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/run_ttcd_double_new_eval.py"),
                stage,
                "--base-config",
                str(base_path),
                "--settings",
                str(settings_path),
                "--model",
                "single",
                "--panel",
                str(panel_path),
                "--out",
                str(out),
                "--device",
                "cpu",
                "--tracking",
                "disabled",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    result = json.loads((out / "result-audit.json").read_text())
    assert result["passed"] and result["write_states_exact"] and result["states"] == 81
    assert result["identity"]["parent"]["checkpoint_sha256"] == verified.sha256(
        config["models"]["single"]["checkpoint"]
    )
