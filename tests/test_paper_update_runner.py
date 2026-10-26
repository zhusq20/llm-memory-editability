"""Integration contracts for the unchanged objective, scorer, and gated adapter."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grokking_reproduction import MergedBatchStream, digest, model_digest
from llm_memory_editability.loop_hop_data import load, make_table, panel_names, prepare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_paper_update_gate")
objective = importlib.import_module("run_supervised_alignment").losses


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return {
        "batch": "paper-update-runner-test",
        "data": {
            "world_seed": 11,
            "entities_per_graph": 4,
            "relations": 2,
            "train_compositions_per_hop": 6,
            "evaluation_templates": 1,
            "training_panel_size": 3,
        },
        "model": {
            "n_embd": 8,
            "n_head": 2,
            "n_positions": 8,
            "n_ctx": 8,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
        },
        "training": {
            "initialization": 13,
            "stream_seed": 17,
            "weight_decay": 0.1,
            "lr": 0.001,
            "warmup": 2,
            "batch_size": 8,
            "steps": 300000,
        },
        "dropout_seed": 19,
        "smoke_steps": 32,
        "arms": {
            f"{mode}-update": {
                "architecture": "discoloop",
                "unique_layers": 1,
                "repeats": 4,
                "update_mode": mode,
                "initial_update": 0.99,
            }
            for mode in ("fixed", "token")
        },
    }


@pytest.fixture
def task(tmp_path):
    config = tiny_config()
    config["input_root"] = str(tmp_path / "input")
    prepare(config, config["input_root"])
    world, panels, metadata = load(config["input_root"])
    config["input_origin"] = {key: metadata[key] for key in ("world_sha256", "panels_sha256")}
    return config, world, panels, metadata


def snapshot_adapter():
    return {
        **{
            key: getattr(runner.native, key)
            for key in ("initialize", "accounting", "optimizer_for", "preflight")
        },
        "cloud": runner.depth.cloud,
    }


def test_import_has_no_global_adapter_side_effects():
    before = snapshot_adapter()
    spec = importlib.util.spec_from_file_location(
        "paper_update_import_check", ROOT / "scripts/run_paper_update_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert snapshot_adapter() == before


@pytest.mark.parametrize("raise_inside", (False, True))
def test_scoped_adapter_restores_every_replaced_function_and_nested_context(raise_inside):
    before = snapshot_adapter()

    def exercise():
        with runner.installed():
            assert runner.native.initialize is runner.initialize
            assert runner.native.accounting is runner.accounting
            assert runner.native.optimizer_for is runner.paper.optimizer_for
            assert runner.native.preflight is runner.preflight
            assert runner.depth.cloud is runner.paper.cloud
            installed = snapshot_adapter()
            with runner.installed():
                assert snapshot_adapter() == installed
            assert snapshot_adapter() == installed
            if raise_inside:
                raise RuntimeError("controlled failure")

    if raise_inside:
        with pytest.raises(RuntimeError, match="controlled failure"):
            exercise()
    else:
        exercise()
    assert snapshot_adapter() == before


def test_initializer_preserves_pair_state_rng_checkpoint_metadata_and_optimizer_groups(task):
    config, _, _, metadata = task
    observed = []
    for name, structure in config["arms"].items():
        model, spec, arm, initial = runner.initialize(config, name, metadata, torch.device("cpu"))
        observed.append((model, initial, torch.get_rng_state().clone()))
        assert spec == {**config["training"], **structure, "name": name}
        assert arm["kind"] == "none" and arm["coefficient"] == 0
        assert arm["gate_variant"] == "paper"
        assert arm["update_mode"] == structure["update_mode"]
        assert arm["initial_update"] == 0.99
        assert arm["first_round_kv"] is False and arm["discoloop"] is True
        assert initial == model_digest(model)
        optimizer = runner.optimizer_for(model, spec["lr"], spec["weight_decay"])
        gates = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
        seen = [id(p) for group in optimizer.param_groups for p in group["params"]]
        assert len(seen) == len(set(seen)) == len(list(model.parameters()))
        assert all(
            group["weight_decay"] == 0
            for group in optimizer.param_groups
            if any(id(p) in gates for p in group["params"])
        )
        assert optimizer.defaults["capturable"] and optimizer.defaults["fused"]
    fixed, learned = observed
    assert fixed[1] == learned[1]
    assert torch.equal(fixed[2], learned[2])
    assert fixed[0].state_dict().keys() == learned[0].state_dict().keys()
    for name, value in fixed[0].state_dict().items():
        assert torch.equal(value, learned[0].state_dict()[name]), name


@pytest.mark.parametrize("name", ("fixed-update", "token-update"))
def test_native_loss_updates_and_fresh_instance_reload_preserve_full_scoring(task, tmp_path, name):
    config, world, panels, metadata = task
    device = torch.device("cpu")
    model, spec, arm, initial = runner.initialize(config, name, metadata, device)
    table = make_table(world, metadata, device)
    stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    configured = runner.optimizer_for(model, spec["lr"], spec["weight_decay"])
    # CPU engineering uses the same groups/betas/eps. CUDA capture and fused
    # execution are separate preflight contracts, not certified by this test.
    optimizer = torch.optim.AdamW(
        [
            {"params": group["params"], "weight_decay": group["weight_decay"]}
            for group in configured.param_groups
        ],
        lr=spec["lr"],
        betas=configured.defaults["betas"],
        eps=configured.defaults["eps"],
        fused=False,
        capturable=False,
    )
    for _ in range(3):
        indices = torch.as_tensor(stream.batch(spec["batch_size"])[0])
        optimizer.zero_grad(set_to_none=True)
        total, main, auxiliary, info = objective(model, table, indices, arm)
        assert torch.equal(total, main) and auxiliary.item() == 0
        assert all(value.item() == 0 for value in info.values())
        total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    trained = model_digest(model)
    assert trained != initial
    checkpoint = tmp_path / f"{name}.pt"
    torch.save({"model": model.state_dict(), "spec": spec, "arm": arm}, checkpoint)
    rng = torch.get_rng_state().clone()
    endpoint, _ = runner.native.measure_panels(model, world, panels, metadata, device, 32)
    metrics, predictions = runner.depth.sweep(model, world, panels, metadata, device, 32)
    assert model.repeats == 4 and model.training
    assert model_digest(model) == trained and torch.equal(rng, torch.get_rng_state())
    assert set(metrics) == set(predictions) == {"1", "2", "3", "4"}
    for repeat in metrics:
        assert set(metrics[repeat]) == set(panel_names())
        for panel in panel_names():
            assert metrics[repeat][panel]["n"] == len(panels[panel])
            assert "accuracy" in metrics[repeat][panel]  # answer AND generated EOS
            assert "answer_accuracy" in metrics[repeat][panel]
            np.testing.assert_array_equal(predictions[repeat][panel]["rows"], panels[panel])
    for panel in panel_names():
        assert all(metrics["4"][panel][key] == value for key, value in endpoint[panel].items())
    assert "answer_accuracy_given_common_premises" in metrics["4"]["ood_3"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored, restored_spec, restored_arm, _ = runner.initialize(config, name, metadata, device)
    assert saved["spec"] == restored_spec and saved["arm"] == restored_arm
    restored.load_state_dict(saved["model"], strict=True)
    replay_metrics, replay = runner.depth.sweep(restored, world, panels, metadata, device, 32)
    assert model_digest(restored) == trained
    assert replay_metrics == metrics
    for repeat in predictions:
        for panel in panel_names():
            for key in ("rows", "answer", "stop", "nll"):
                np.testing.assert_array_equal(
                    replay[repeat][panel][key], predictions[repeat][panel][key]
                )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def engineering_files(task, tmp_path):
    config, _, _, _ = task
    path = tmp_path / "config.json"
    write(path, config)
    base = tmp_path / "checks"
    for name in config["arms"]:
        write(
            base / "preflight" / f"{name}.json",
            {"passed": True, "engineering_only": True, "config_sha256": digest(path)},
        )
        folder = base / "smoke" / name
        folder.mkdir(parents=True)
        checkpoint = folder / "checkpoint-0000032.pt"
        checkpoint.write_bytes(name.encode())
        complete = {
            "state": "complete",
            "arm": name,
            "step": 32,
            "budget": 32,
            "engineering_only": True,
            "config_sha256": digest(path),
            "world_sha256": config["input_origin"]["world_sha256"],
            "model_sha256": "model-" + name,
            "final_checkpoint_sha256": digest(checkpoint),
        }
        write(folder / "complete.json", complete)
        write(
            folder / "audit.json",
            {
                "passed": True,
                "arm": name,
                "step": 32,
                "exposure_match": True,
                "panels": {
                    panel: {"passed": True, "answer_and_eos_ids_exact": True}
                    for panel in panel_names()
                },
            },
        )
        write(
            folder / "grid.json",
            {
                **complete,
                "training_repeats": 4,
                "unique_layers": 1,
                "panels_sha256": config["input_origin"]["panels_sha256"],
                "metrics": {
                    str(repeat): {panel: {} for panel in panel_names()} for repeat in range(1, 5)
                },
            },
        )
        write(
            folder / "grid-audit.json",
            {
                "passed": True,
                "model_sha256": complete["model_sha256"],
                "grid_sha256": digest(folder / "grid.json"),
                "findings": {
                    f"r{repeat}/{panel}": True for repeat in range(1, 5) for panel in panel_names()
                },
            },
        )
    return config, path, base


def test_engineering_receipt_binds_both_arm_checks_and_grid(engineering_files):
    config, path, base = engineering_files
    receipt = runner.engineering_receipt(config, path, base)
    assert receipt["passed"] is True
    assert receipt["config_sha256"] == digest(path)
    assert set(receipt["files"]) == {
        *(f"preflight/{name}.json" for name in config["arms"]),
        *(
            f"smoke/{name}/{filename}"
            for name in config["arms"]
            for filename in (
                "complete.json",
                "audit.json",
                "grid.json",
                "grid-audit.json",
                "checkpoint-0000032.pt",
            )
        ),
    }
    assert all(digest(base / key) == value for key, value in receipt["files"].items())


@pytest.mark.parametrize(
    ("relative", "key", "value"),
    [
        ("preflight/fixed-update.json", "config_sha256", "changed"),
        ("smoke/fixed-update/complete.json", "engineering_only", False),
        ("smoke/fixed-update/complete.json", "step", 20),
        ("smoke/token-update/audit.json", "arm", "fixed-update"),
        ("smoke/token-update/audit.json", "step", 20),
        ("smoke/token-update/grid.json", "config_sha256", "changed"),
        ("smoke/token-update/grid-audit.json", "model_sha256", "other-model"),
        ("smoke/token-update/grid-audit.json", "grid_sha256", "other-grid"),
    ],
)
def test_engineering_receipt_rejects_stale_or_swapped_checks(
    engineering_files, relative, key, value
):
    config, path, base = engineering_files
    destination = base / relative
    payload = json.loads(destination.read_text())
    payload[key] = value
    write(destination, payload)
    with pytest.raises(ValueError):
        runner.engineering_receipt(config, path, base)


def test_engineering_receipt_rejects_changed_smoke_checkpoint(engineering_files):
    config, path, base = engineering_files
    checkpoint = base / "smoke/fixed-update/checkpoint-0000032.pt"
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError):
        runner.engineering_receipt(config, path, base)
