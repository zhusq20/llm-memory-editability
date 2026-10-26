"""Original Wang pairing and finish-before-start scheduling contracts."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_joint_update_wang as launch  # noqa: E402
import run_joint_update_wang as runner  # noqa: E402

from llm_memory_editability.joint_update_gate import JointUpdateGateGPT  # noqa: E402
from llm_memory_editability.paper_update_gate import PaperUpdateGateGPT  # noqa: E402


@pytest.fixture
def recipes():
    c = json.loads((ROOT / "configs/joint-update-gate-wang-v1.json").read_text())
    reference = ROOT / c["paired_recipe"]["config"]
    assert runner.digest(reference) == c["paired_recipe"]["sha256"]
    return c, json.loads(reference.read_text())


def test_original_wang_recipe_is_exactly_paired(recipes):
    config, reference = recipes
    runner.validate_config(config, reference)
    assert config["arms"]["joint-token"]["unique_layers"] == 4
    assert config["arms"]["joint-token"]["repeats"] == 2
    assert config["inference_repeats"] == [1, 2]
    assert config["training"]["steps"] == 1500000


@pytest.mark.parametrize("field", ["training", "data", "input_origin", "model", "arms"])
def test_unpaired_recipe_changes_are_rejected(recipes, field):
    config, reference = recipes
    config[field] = {}
    with pytest.raises(ValueError):
        runner.validate_config(config, reference)


def test_wang_adapter_preserves_weights_rng_and_final_readout(recipes, monkeypatch):
    config, reference = copy.deepcopy(recipes)
    for c in (config, reference):
        c["model"].update(
            n_embd=12, n_head=3, n_positions=16, n_ctx=16, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
        )
    metadata = {"vocab_size": 23, "pad_token": 0, "end_marker": 2}
    monkeypatch.setattr(runner.original, "load", lambda _: ({}, {"test_oo": []}, metadata))
    old, old_spec, _, old_hash = runner.original.initialize(
        reference, "token-update", metadata, torch.device("cpu")
    )
    rng = torch.get_rng_state()
    before = {k: getattr(runner.native, k) for k in ("initialize", "load", "accounting")}
    with runner.installed(config):
        new, spec, arm, new_hash = runner.native.initialize(
            config, "joint-token", metadata, torch.device("cpu")
        )
        assert isinstance(new, JointUpdateGateGPT)
        assert spec == {**old_spec, "name": "joint-token"}
        assert arm["gate_scope"] == runner.SCOPE
        assert old_hash == new_hash
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
        new.eval()
        logits, _, h = new.forward_with_states(torch.tensor([[3, 8, 9, 4]]))
        assert torch.equal(logits, new.transformer.ln_f(h) @ new.transformer.wte.weight.T)
    assert runner.original.update.PaperUpdateGateGPT is PaperUpdateGateGPT
    assert all(getattr(runner.native, k) is v for k, v in before.items())


@pytest.fixture
def predecessor(tmp_path):
    write = launch.common.write
    write(tmp_path / "freeze.json", {"identity": "original"})
    dep = {
        "arm": "joint-token",
        "budget": 1500000,
        "required_stages": ["train", "grid"],
        "freeze_sha256": launch.common.sha(tmp_path / "freeze.json"),
    }
    complete = {
        "state": "complete",
        "step": 1500000,
        "budget": 1500000,
        "arm": "joint-token",
        "model_sha256": "m",
        "config_sha256": "c",
        "world_sha256": "w",
    }
    for stage in ("train", "grid"):
        write(tmp_path / "execution/joint-token" / stage / "exit.json", {"code": 0})
    write(tmp_path / "runs/joint-token/complete.json", complete)
    write(tmp_path / "runs/joint-token/grid.json", complete)
    write(tmp_path / "controller-exit.json", {"code": 0})
    return tmp_path, dep


def test_releases_only_after_train_and_grid_success(predecessor):
    folder, dep = predecessor
    assert launch.predecessor_ready(folder, dep)
    (folder / "execution/joint-token/grid/exit.json").unlink()
    assert not launch.predecessor_ready(folder, dep)


def test_failed_predecessor_never_releases_new_training(predecessor):
    folder, dep = predecessor
    launch.common.write(folder / "controller-exit.json", {"code": 1})
    with pytest.raises(RuntimeError, match="failed"):
        launch.predecessor_ready(folder, dep)


def test_unrelated_grid_cannot_satisfy_dependency(predecessor):
    folder, dep = predecessor
    grid = launch.common.read(folder / "runs/joint-token/grid.json")
    grid["model_sha256"] = "other"
    launch.common.write(folder / "runs/joint-token/grid.json", grid)
    with pytest.raises(ValueError, match="identity mismatch"):
        launch.predecessor_ready(folder, dep)
