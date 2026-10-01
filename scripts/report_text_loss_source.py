"""Independent scoring, paired continuation audits, and factorial summaries."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from llm_memory_editability.grok_depth import EpochStream, write_json
from llm_memory_editability.text_loss_source import ARMS, SPLITS, loss_sources, measure, weights_for
from llm_memory_editability.text_pretrain import construct, sha


def full(pred, split):
    return (pred[split + "_answer"] == pred[split + "_target"]) & (
        pred[split + "_stops"] == [5, 1]
    ).all(-1)


def eligibility(pred, world):
    known = {
        (h, r): ok for (h, r, _), ok in zip(world["atomic"], full(pred, "atomic"), strict=True)
    }
    return np.array([known[h, r1] and known[b, r2] for h, r1, b, r2, t in world["target_test"]])


def dump_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(config, device):
    cfg = json.loads(Path(config).read_text())
    out = Path(cfg["lock"]).parent / "report"
    out.mkdir(exist_ok=True)
    lock = json.loads(Path(cfg["lock"]).read_text())
    for group in ("sources", "parents", "inputs"):
        for file, expected in lock[group].items():
            assert sha(file) == expected, file
    assert sha(config) == lock["config_sha256"] and sha(cfg["plan"]) == lock["plan_sha256"]
    rows, audits, cells = [], [], {}
    for name, run in cfg["runs"].items():
        root = Path(cfg["output_root"]) / name
        parent = Path(run["parent"])
        world = dict(np.load(parent / "world.npz"))
        original = torch.load(parent / "latest.pt", map_location="cpu", weights_only=False)
        endpoint = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
        spec = original["spec"]
        done = json.loads((root / "complete.json").read_text())
        history = json.loads((root / "learning.json").read_text())
        assert [r["step"] for r in history] == cfg["nodes"]
        assert done["checkpoint_sha256"] == sha(root / "latest.pt") and done["final"] == history[-1]
        initial = torch.load(root / "weights-000000.pt", map_location="cpu", weights_only=False)[
            "model"
        ]
        assert all(torch.equal(v, initial[k]) for k, v in original["model"].items())
        initial_pred = np.load(root / "predictions-000000.npz")
        parent_pred = np.load(parent / "predictions-032000.npz")
        for k in parent_pred.files:
            np.testing.assert_array_equal(initial_pred[k], parent_pred[k])
        bounds = np.cumsum(
            [0, len(world["atomic"]), len(world["background_train"]), len(world["target_train"])]
        )
        streams = [EpochStream(bounds[i + 1] - bounds[i], 0) for i in range(3)]
        for stream, saved in zip(streams, original["streams"], strict=True):
            stream.load_state_dict(saved)
        trace = hashlib.sha256()
        for _ in range(cfg["steps"]):
            idx = np.concatenate(
                [
                    s.take(n) + lo
                    for s, n, lo in zip(streams, spec["batch_parts"], bounds[:-1], strict=True)
                ]
            )
            trace.update(idx.tobytes())
        assert trace.hexdigest() == done["sample_trace_sha256"]
        for stream, saved in zip(streams, endpoint["streams"], strict=True):
            replay = stream.state_dict()
            assert replay["rng"] == saved["rng"]
            np.testing.assert_array_equal(replay["remaining"], saved["remaining"])
        checked = 0
        for node in history:
            p = np.load(root / f"predictions-{node['step']:06d}.npz")
            for split in SPLITS:
                np.testing.assert_array_equal(p[split + "_target"], world[split][:, -1])
                score = node["metrics"][split]
                assert score["n"] == len(world[split])
                assert score["accuracy"] == float(full(p, split).mean())
                assert score["answer_accuracy"] == float(
                    (p[split + "_answer"] == world[split][:, -1]).mean()
                )
                assert abs(score["answer_probability"] - p[split + "_probability"].mean()) < 1e-12
                checked += len(world[split])
            # Runner eligibility uses entity correctness; the common subset requires full answers.
            known = {
                (h, r): ok
                for (h, r, _), ok in zip(
                    world["atomic"], p["atomic_answer"] == world["atomic"][:, -1], strict=True
                )
            }
            elig = np.array(
                [known[h, r1] and known[b, r2] for h, r1, b, r2, t in world["target_test"]]
            )
            assert node["metrics"]["both_atomics"]["n"] == int(elig.sum())
            if elig.any():
                assert node["metrics"]["both_atomics"]["accuracy"] == float(
                    full(p, "target_test")[elig].mean()
                )
            rows.append(
                dict(
                    run=name,
                    world=spec["world"],
                    initialization=spec["initialization"],
                    arm=run["arm"],
                    step=node["step"],
                    target_accuracy=node["metrics"]["target_test"]["accuracy"],
                    target_probability=node["metrics"]["target_test"]["answer_probability"],
                    atomic_accuracy=node["metrics"]["atomic"]["accuracy"],
                    background_test_accuracy=node["metrics"]["background_test"]["accuracy"],
                    background_train_accuracy=node["metrics"]["background_train"]["accuracy"],
                    coverage=node["metrics"]["both_atomics"]["coverage"],
                    **{k + "_nll": v for k, v in node["metrics"]["training_text_nll"].items()},
                )
            )
        model = construct(spec, device)
        model.load_state_dict(endpoint["model"])
        table_file = np.load(parent / "training-table.npz")
        table = tuple(table_file[k] for k in ("tokens", "positions", "mask", "labels"))
        source = loss_sources(table[3], bounds)
        for a in ARMS:
            w = weights_for(source, a)
            assert np.all(w[table[3] == -100] == 0)
        actual, actual_p = measure(model, world, table, source, device)
        p = dict(np.load(root / f"predictions-{cfg['steps']:06d}.npz"))
        maxerror = 0.0
        for key, value in actual_p.items():
            if key.endswith("probability"):
                error = float(np.max(np.abs(value - p[key])))
                assert error < 1e-5
                maxerror = max(error, maxerror)
            else:
                np.testing.assert_array_equal(value, p[key])
        for k, v in actual["training_text_nll"].items():
            assert abs(v - history[-1]["metrics"]["training_text_nll"][k]) < 1e-6
        cell = cells.setdefault((spec["world"], spec["initialization"]), {})
        cell[run["arm"]] = dict(
            pred=p,
            world=world,
            original=initial_pred,
            trace=trace.hexdigest(),
            cpu_rng=endpoint["cpu_rng"],
            cuda_rng=endpoint["cuda_rng"],
            done=done,
        )
        audits.append(
            dict(
                run=name,
                raw_prediction_rows=checked,
                reload_probability_max_error=maxerror,
                initial_parameters_match=True,
                initial_predictions_match=True,
                sampler_replay_matches=True,
            )
        )
        print(name + " audited", flush=True)
    contrasts = []
    for (world, seed), cell in cells.items():
        assert set(cell) == set(ARMS)
        assert len({v["trace"] for v in cell.values()}) == 1
        assert all(
            torch.equal(v["cuda_rng"], cell["AB"]["cuda_rng"])
            and torch.equal(v["cpu_rng"], cell["AB"]["cpu_rng"])
            for v in cell.values()
        )
        common = np.logical_and.reduce([eligibility(v["pred"], v["world"]) for v in cell.values()])
        acc = {a: float(full(v["pred"], "target_test").mean()) for a, v in cell.items()}
        start = float(full(cell["AB"]["original"], "target_test").mean())
        selected = {
            a: float(full(v["pred"], "target_test")[common].mean()) if common.any() else None
            for a, v in cell.items()
        }
        contrasts.append(
            dict(
                world=world,
                initialization=seed,
                source=start,
                **acc,
                A_effect=0.5 * ((acc["AB"] - acc["B"]) + (acc["A"] - acc["N"])),
                B_effect=0.5 * ((acc["AB"] - acc["A"]) + (acc["B"] - acc["N"])),
                A_given_B=acc["AB"] - acc["B"],
                B_given_A=acc["AB"] - acc["A"],
                interaction=acc["AB"] - acc["A"] - acc["B"] + acc["N"],
                common_coverage=float(common.mean()),
                common_n=int(common.sum()),
                **{"common_" + k: v for k, v in selected.items()},
            )
        )
    worlds = sorted({r["world"] for r in contrasts})
    keys = [k for k in contrasts[0] if k not in ("world", "initialization", "common_n")]
    by_world = [
        dict(
            world=w,
            **{k: float(np.mean([r[k] for r in contrasts if r["world"] == w])) for k in keys},
        )
        for w in worlds
    ]
    means = {k: float(np.mean([r[k] for r in by_world])) for k in keys}
    endpoint_mean = {
        a: {
            k: float(np.mean([r[k] for r in rows if r["arm"] == a and r["step"] == cfg["steps"]]))
            for k in [
                "target_accuracy",
                "target_probability",
                "atomic_accuracy",
                "background_test_accuracy",
                "background_train_accuracy",
                "coverage",
                "atomic_nll",
                "background_nll",
                "neutral_nll",
            ]
        }
        for a in ARMS
    }
    summary = dict(
        config=config,
        source_worlds=worlds,
        runs=len(cfg["runs"]),
        steps=cfg["steps"],
        paired=contrasts,
        world_effects=by_world,
        means=means,
        endpoints=endpoint_mean,
        audit=dict(
            status="passed",
            runs=len(audits),
            raw_prediction_rows=sum(x["raw_prediction_rows"] for x in audits),
            runs_detail=audits,
        ),
        budget={
            k: sum(v["done"][k] for c in cells.values() for v in c.values())
            for k in ["visible_supervised_tokens", "active_supervised_tokens", "estimated_flops"]
        },
        training_seconds=sum(
            v["done"]["final"]["seconds"] for c in cells.values() for v in c.values()
        ),
        sources={__file__: sha(__file__)},
    )
    write_json(out / "summary.json", summary)
    dump_csv(out / "paired-effects.csv", contrasts)
    dump_csv(out / "learning.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    colors = {"AB": "#2563eb", "A": "#0f9d76", "B": "#e88919", "N": "#667085"}
    for a in ARMS:
        nodes = cfg["nodes"]
        points = [
            [r["target_accuracy"] * 100 for r in rows if r["arm"] == a and r["step"] == step]
            for step in nodes
        ]
        axes[0].plot(nodes, np.mean(points, axis=1), label=a, color=colors[a])
        axes[0].fill_between(
            nodes, np.min(points, axis=1), np.max(points, axis=1), color=colors[a], alpha=0.08
        )
    axes[0].set(
        title="Target held-out composition",
        xlabel="Continuation steps",
        ylabel="Full answer accuracy (%)",
        ylim=(0, 103),
    )
    axes[0].legend()
    for r in by_world:
        axes[1].plot(range(4), [100 * r[a] for a in ARMS], marker="o", label=str(r["world"]))
    axes[1].set(
        title="Paired endpoints by world",
        xticks=range(4),
        xticklabels=list(ARMS),
        ylabel="Accuracy (%)",
        ylim=(0, 103),
    )
    axes[1].legend(title="World")
    x = np.arange(4)
    axes[2].bar(
        x - 0.18,
        [100 * endpoint_mean[a]["atomic_accuracy"] for a in ARMS],
        width=0.36,
        label="Atomic",
    )
    axes[2].bar(
        x + 0.18,
        [100 * endpoint_mean[a]["background_test_accuracy"] for a in ARMS],
        width=0.36,
        label="Background held-out",
    )
    axes[2].set(
        title="Prerequisites and retention",
        xticks=x,
        xticklabels=list(ARMS),
        ylabel="Accuracy (%)",
        ylim=(0, 103),
    )
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(dict(means=means, endpoints=endpoint_mean)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main(args.config, args.device)
