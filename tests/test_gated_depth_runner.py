import importlib.util
from pathlib import Path

import numpy as np
import torch
from test_gated_depth_data import config as data_config

PATH = Path(__file__).resolve().parents[1] / "scripts/run_gated_depth.py"
SPEC = importlib.util.spec_from_file_location("run_gated_depth", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return {
        **data_config(),
        "arms": {
            "N-Fixed": {"gate_mode": "fixed", "feedback_beta": 0},
            "N-Gate": {"gate_mode": "token", "feedback_beta": 0},
        },
        "model": {
            "n_layer": 1,
            "n_embd": 16,
            "n_head": 2,
            "n_positions": 22,
            "resid_pdrop": 0,
            "embd_pdrop": 0,
            "attn_pdrop": 0,
        },
        "gate": {"initial_alpha": 0.9},
        "training": {"weight_decay": 0.1, "learning_rate": 0.001},
    }


def test_optimizer_gate_no_decay_and_initialization_pair():
    cfg, device = config(), torch.device("cpu")
    metadata = {"vocab_size": 24}
    a, b = [runner.initialize(cfg, metadata, arm, 11, device) for arm in cfg["arms"]]
    assert runner.backbone_hash(a) == runner.backbone_hash(b)
    optimizer = runner.optimizer_for(b, cfg, device)
    gate_ids = {id(p) for n, p in b.named_parameters() if n.startswith("gate_")}
    assert all(
        g["weight_decay"] == 0
        for g in optimizer.param_groups
        if any(id(p) in gate_ids for p in g["params"])
    )


def test_checkpoint_restores_optimizer_rng_and_independent_predictions(tmp_path):
    from llm_memory_editability.gated_depth_data import BalancedStream, generate, pack

    cfg, device = config(), torch.device("cpu")
    pools, _, metadata, _ = generate(cfg)
    model = runner.initialize(cfg, metadata, "N-Gate", 11, device)
    optimizer = runner.optimizer_for(model, cfg, device)
    stream = BalancedStream(pools, metadata, 2)
    packed, _ = stream.batch(4)
    x, positions, y = runner.tensors(packed, device)
    torch.nn.functional.cross_entropy(model(x, positions), y).backward()
    optimizer.step()
    runner.save_checkpoint(
        tmp_path / "best.pt", model, optimizer, stream, 1, cfg, "N-Gate", 11, 0.0
    )
    restored, payload = runner.restore(cfg, metadata, tmp_path, "best.pt", device)
    assert payload["optimizer"]["state"]
    assert payload["data_rng"] == stream.rng.bit_generator.state
    assert torch.equal(model(x, positions), restored(x, positions))
    rows = pools["eval_A_short"][:4]
    parts = pack(rows, metadata)
    first = runner.sweep(model, parts, device, [1, 5, 10, 20, 32], 2)
    second = runner.sweep(restored, parts, device, [1, 5, 10, 20, 32], 2)
    assert np.array_equal(first["prediction"], second["prediction"])
    assert np.array_equal(first["loss"], second["loss"])


def test_validation_macro_weights_hops_equally():
    rows = [{"hop_count": k} for k in [1, 1, 1, 2]]
    result = runner.summary(np.array([1, 1, 1, 0]), np.ones(4), rows)
    assert result["accuracy"] == 0.75
    assert result["macro_accuracy"] == 0.5


def test_complete_small_cpu_training_and_independent_audit(tmp_path, monkeypatch):
    cfg = config()
    cfg["arms"] = {"N-Gate": cfg["arms"]["N-Gate"]}
    cfg["smoke"] = {"steps": 2, "seed": 11}
    cfg["training"].update(
        batch_size=4,
        warmup_steps=1,
        clip=1.0,
        log_every=1,
        validation_every=1,
        checkpoint_every=2,
        precision="float32",
    )
    cfg["evaluation"] = {"batch_size": 4}

    class Tracking:
        def __init__(self):
            self.summary = {}

        def log(self, *args, **kwargs):
            pass

        def finish(self, **kwargs):
            pass

    monkeypatch.setattr(runner, "tracking", lambda *args: Tracking())
    runner.prepare(cfg, tmp_path / "data")
    runner.train(cfg, tmp_path, "smoke", "N-Gate", 11, torch.device("cpu"))
    runner.audit(cfg, tmp_path, "smoke", "N-Gate", 11, torch.device("cpu"))
    import json

    out = tmp_path / "smoke/N-Gate-seed11"
    receipt = json.loads((out / "audit.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    assert receipt["passed"]
    assert complete["steps"] == 2
    assert complete["initial_stability"]["probe_hops"] == [1, 2, 3, 4, 5]
    assert sum(complete["hop_exposures"].values()) == 8


def test_statistics_transitions_common_facts_and_bootstrap():
    import analyze_gated_depth as analysis

    correct = np.array([[True, False, True], [False, True, True], [False, False, False]])
    result = analysis.transitions(correct, [5, 10, 20])
    np.testing.assert_equal(result["retain"], [1, 0, 1])
    np.testing.assert_equal(result["recover"], [0, 0.5, 0.5])
    rows = [
        {"relation_ids": [0, 1], "intermediate_entities": [0, 1, 2]},
        {"relation_ids": [1, 0], "intermediate_entities": [1, 2, 3]},
    ]
    # Atom table is relation-major, deliberately different from head*R+relation.
    table = [(h, r) for r in range(2) for h in range(4)]
    index = {key: i for i, key in enumerate(table)}
    atoms = np.ones(8, dtype=bool)
    atoms[2] = False  # (head=2, relation=0), used only by the second path.
    assert analysis.common_known_mask(rows, atoms, index).tolist() == [True, False]
    # Reordering facts and their corresponding scores cannot change the subset.
    order = np.array([7, 2, 1, 4, 6, 0, 3, 5])
    index = {table[old]: new for new, old in enumerate(order)}
    assert analysis.common_known_mask(rows, atoms[order], index).tolist() == [True, False]
    assert analysis.template_bootstrap(np.ones(2), rows, 100, 1) == [1.0, 1.0]


def test_aggregate_and_real_pdf_render_with_synthetic_fixture(tmp_path):
    """Exercise the post-training branch using clearly synthetic fixture predictions."""
    import json

    import analyze_gated_depth as analysis

    from llm_memory_editability.gated_depth_data import load, prepare

    cfg = config()
    cfg["arms"] = {
        f"{prefix}-{mode}": {} for prefix in ("N", "Q") for mode in ("Full", "Fixed", "Gate")
    }
    cfg["seeds"] = [11, 22, 33]
    cfg["evaluation"] = {
        "loops": [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32],
        "robust_loops": [6, 8, 10, 12, 16, 20],
        "bootstrap_replicates": 10,
        "bootstrap_seed": 1,
    }
    prepare(cfg, tmp_path / "data")
    pools, _, _ = load(tmp_path / "data")
    for seed in cfg["seeds"]:
        for arm in cfg["arms"]:
            run = tmp_path / "train" / f"{arm}-seed{seed}"
            run.mkdir(parents=True)
            for name in ("audit", "evaluation", "cloud-audit"):
                (run / f"{name}.json").write_text(
                    json.dumps({"passed": True, "synthetic_fixture": True})
                )
            complete = {
                "passed": True,
                "synthetic_fixture": True,
                "steps": 2,
                "best_step": 1,
                "initial_backbone_sha256": str(seed),
                "stream_sha256": str(seed),
                "parameters": 1,
                "input_tokens": 1,
                "training_matrix_flops_estimate": 1,
                "wall_seconds": 1,
                "peak_cuda_bytes": 0,
            }
            (run / "complete.json").write_text(json.dumps(complete))
            for checkpoint in ("best", "last"):
                dest = run / f"evaluation-{checkpoint}"
                dest.mkdir()
                for name in (
                    "facts_train",
                    "eval_A_short",
                    "eval_A_long",
                    "eval_B_short",
                    "eval_B_long",
                ):
                    labels = np.array([row["answer_entity"] + 3 for row in pools[name]])
                    prediction = np.repeat(labels[:, None], 13, axis=1)
                    prediction[::2, :5] = -1
                    np.savez_compressed(dest / f"{name}.npz", prediction=prediction, labels=labels)
                runner.write_csv(
                    dest / "eval_A_short-dynamics.csv",
                    [{"loop": r, "alpha_mean": 0.9, "update_ratio": 0.1} for r in range(1, 33)],
                )
    receipt = analysis.analyze(cfg, tmp_path)
    assert receipt["passed"] and receipt["trajectories"] == 18
    for name in (
        "accuracy_vs_loops",
        "accuracy_heatmap",
        "retention_recovery",
        "gate_and_state_dynamics",
    ):
        assert (tmp_path / "analysis" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    import csv

    with (tmp_path / "analysis/paired_differences.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(float(row["difference"]) == 0 for row in rows)
    assert {row["hop"] for row in rows} >= {"8", "10", "all"}
