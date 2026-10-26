"""Scientific resume and scheduling contracts for the two missing baselines."""

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_ungated_loop_continuation")
launcher = importlib.import_module("launch_ungated_loop_continuation")


def config():
    return json.loads((ROOT / launcher.RELATIVE).read_text())


@pytest.mark.parametrize("name", runner.ARMS)
def test_native_model_and_continued_adam_match_uninterrupted(name):
    c = config()
    runner.validate_config(c)
    c["model"].update(
        n_embd=12, n_head=3, n_positions=16, n_ctx=16, attn_pdrop=0, embd_pdrop=0, resid_pdrop=0
    )
    metadata = {"vocab_size": 23, "pad_token": 0, "end_marker": 1}
    native = runner.native
    model, spec, arm, _ = native.initialize(c, name, metadata, torch.device("cpu"))
    assert not any("gate_" in n for n in model.state_dict())
    assert arm["discoloop"] == name.startswith("disco")
    optimizer = native.optimizer_for(model, torch.tensor(1e-4), 0.1)

    def update(m, opt, step):
        for i, p in enumerate(m.parameters()):
            p.grad = torch.full_like(p, (i + step + 1) / 100)
        opt.step()

    for step in range(3):
        update(model, optimizer, step)
    resumed = copy.deepcopy(model)
    with runner.installed():
        restored, _, gate_lr = runner.continuation.restore_optimizer(
            resumed, copy.deepcopy(optimizer.state_dict()), spec, 1.0, torch.device("cpu")
        )
    assert gate_lr is None
    assert runner.continuation.optimizer_digest(
        model, optimizer
    ) == runner.continuation.optimizer_digest(resumed, restored)
    for step in range(3, 6):
        update(model, optimizer, step)
        update(resumed, restored, step)
        assert native.model_digest(model) == native.model_digest(resumed)
        assert runner.continuation.optimizer_digest(
            model, optimizer
        ) == runner.continuation.optimizer_digest(resumed, restored)


@pytest.mark.parametrize("mutation", ["budget", "gate", "warmup", "nodes"])
def test_reject_changed_scientific_scope(mutation):
    c = config()
    if mutation == "budget":
        c["training"]["steps"] = 3000000
    elif mutation == "gate":
        c["arms"]["disco-l1-r4"]["update_mode"] = "token"
    elif mutation == "warmup":
        c["continuation"]["new_warmup"] = True
    else:
        c["evaluation_nodes"] = [300000, 1500000]
    with pytest.raises(ValueError):
        runner.validate_config(c)


def test_predecessor_requires_successful_grid_and_bound_identity(tmp_path):
    d = {"root": str(tmp_path), "arm": "parent", "config_sha256": "expected"}
    assert not launcher.predecessor_ready(d)
    for stage in ("train", "grid"):
        path = tmp_path / "execution/parent" / stage / "exit.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"code":0}')
        if stage == "train":
            assert not launcher.predecessor_ready(d)
    run = tmp_path / "runs/parent"
    run.mkdir(parents=True)
    complete = dict(
        state="complete",
        step=1500000,
        budget=1500000,
        config_sha256="expected",
        arm="parent",
        world_sha256="world",
        model_sha256="model",
    )
    for name in ("complete", "grid"):
        (run / f"{name}.json").write_text(json.dumps(complete))
    assert launcher.predecessor_ready(d)
    (tmp_path / "execution/parent/grid/exit.json").write_text('{"code":1}')
    with pytest.raises(RuntimeError):
        launcher.predecessor_ready(d)
    (tmp_path / "execution/parent/grid/exit.json").write_text('{"code":0}')
    complete["model_sha256"] = "changed"
    (run / "grid.json").write_text(json.dumps(complete))
    with pytest.raises(ValueError):
        launcher.predecessor_ready(d)


def test_gpu_reservation_blocks_even_empty_gpu():
    rows = {1: dict(uuid="uuid", used=0, utilization=0, free=50000)}
    assert launcher.idle(1, "uuid", rows, {})
    assert not launcher.idle(1, "uuid", rows, {1: ["existing job"]})
    assert not launcher.idle(1, "other", rows, {})


def test_actual_parent_full_state_and_hashes():
    c = config()
    if not Path("/inputs/parents").exists():
        pytest.skip("requires frozen parent input mount")
    world, _, metadata = runner.native.load(c["input_root"])
    for name in runner.ARMS:
        with runner.installed():
            values = runner.continuation.prepare(c, name, world, metadata, torch.device("cpu"))
        model, _, arm, opt, _, _, stream, counts, source, complete, _, initial = values
        assert source["step"] == 300000
        assert arm == source["arm"]
        assert initial == complete["initial_model_sha256"]
        assert runner.native.model_digest(model) == source["model_sha256"]
        assert all(float(s["step"]) == 300000 for s in opt.state.values())
        assert np.array_equal(counts, source["counts"])
        other = runner.native.MergedBatchStream(stream.size, c["training"]["stream_seed"])
        other.load_state_dict(source["stream"])
        assert np.array_equal(stream.batch(512)[0], other.batch(512)[0])
