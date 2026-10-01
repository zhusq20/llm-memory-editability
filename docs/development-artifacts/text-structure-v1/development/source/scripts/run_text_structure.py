"""Prepare, freeze, execute, audit and report matched composition-support runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import EpochStream, make_optimizer, utc, write_json
from llm_memory_editability.text_pretrain import TextGraphStep, construct, sha
from llm_memory_editability.text_structure import (
    SPLITS,
    TESTS,
    build_world,
    data_audit,
    evaluate_all,
    role_counts,
    run,
    score_predictions,
    training_table,
    verify_lock,
)

SOURCES = [
    "scripts/run_text_structure.py",
    "tests/test_text_structure.py",
    *[
        f"src/llm_memory_editability/{s}.py"
        for s in ("text_structure", "text_pretrain", "bios_model", "grok_depth", "grok_loop_model")
    ],
]


def preflight(config):
    cfg = json.loads(Path(config).read_text())
    output = Path(cfg["source_lock"]).parent
    records = {}
    for overrides in cfg["runs"].values():
        spec = dict(cfg["base"], **overrides)
        if spec["world"] in records:
            continue
        world = build_world(spec)
        records[spec["world"]] = data_audit(world)
        np.savez_compressed(output / f"world-{spec['world']}.npz", **world)
    write_json(output / "preflight.json", dict(created_utc=utc(), worlds=records))
    return records


def freeze(config):
    cfg = json.loads(Path(config).read_text())
    target = Path(cfg["source_lock"])
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    preflight(config)
    paths = SOURCES + [cfg["plan"], config]
    for p in paths:
        dst = target.parent / "source" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    write_json(
        target,
        dict(
            created_utc=utc(),
            config_sha256=sha(config),
            sources={p: sha(p) for p in paths},
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            python_version=sys.version,
            cuda_version=torch.version.cuda,
        ),
    )


def execute(config, gpus, workers):
    cfg = json.loads(Path(config).read_text())
    verify_lock(config, cfg["source_lock"])
    logs = Path(cfg["source_lock"]).parent / "logs"
    logs.mkdir(exist_ok=True)
    write_json(
        logs / "execution-start.json",
        dict(
            started_utc=utc(), pid=os.getpid(), gpus=gpus, workers=workers, runs=list(cfg["runs"])
        ),
    )

    def worker(i, names):
        for name in names:
            cmd = [
                sys.executable,
                "scripts/run_text_structure.py",
                "run",
                "--config",
                config,
                "--run",
                name,
            ]
            with (logs / (name + ".log")).open("a") as log:
                subprocess.run(
                    cmd,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[i % len(gpus)])),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(json.dumps(dict(run=name, status="complete", utc=utc())), flush=True)

    names = list(cfg["runs"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i, names[i::workers]) for i in range(workers)]
        for future in futures:
            future.result()


def calibrate(config, device):
    """Check new packed objective against eager steps without scientific data."""
    cfg = json.loads(Path(config).read_text())
    spec = dict(cfg["base"], width=16, heads=2, dropout=0.0)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    world = build_world(spec)
    cpu, _, _ = training_table(world, "broad")
    table = tuple(torch.as_tensor(t, device=device) for t in cpu)
    a, b = construct(spec, device), construct(spec, device)
    oa = make_optimizer(a, torch.tensor(spec["lr"], device=device), spec["weight_decay"])
    ob = make_optimizer(b, torch.tensor(spec["lr"], device=device), spec["weight_decay"])
    before = {k: v.clone() for k, v in a.state_dict().items()}
    captured = TextGraphStep(a, oa, table, 16)
    assert all(torch.equal(v, a.state_dict()[k]) for k, v in before.items())
    eager = TextGraphStep.__new__(TextGraphStep)
    eager.model, eager.optimizer, eager.table, eager.clip = b, ob, table, 1.0
    eager.index = torch.arange(16, device=device)
    for step in range(3):
        idx = torch.arange(step * 16, (step + 1) * 16, device=device)
        captured(idx)
        eager.index.copy_(idx)
        eager.eager()
    errors = [float((v - b.state_dict()[k]).abs().max()) for k, v in a.state_dict().items()]
    assert max(errors) < 1e-6
    result = dict(
        status="passed",
        utc=utc(),
        capture_restores_weights=True,
        steps=3,
        max_parameter_error=max(errors),
        device=torch.cuda.get_device_name(device),
    )
    write_json(Path(cfg["source_lock"]).parent / "gpu-calibration.json", result)
    return result


def audit(config, device):
    cfg = json.loads(Path(config).read_text())
    verify_lock(config, cfg["source_lock"])
    torch.set_num_threads(2)
    endpoints, trajectories, results, paired = [], [], [], {}
    for name, overrides in cfg["runs"].items():
        root = Path(cfg["output_root"]) / name
        spec = json.loads((root / "spec.json").read_text())
        expected = dict(
            cfg["base"], **overrides, source_lock=cfg["source_lock"], config_path=config
        )
        assert spec == expected
        complete = json.loads((root / "complete.json").read_text())
        assert complete["source_lock_sha256"] == sha(cfg["source_lock"])
        assert complete["world_sha256"] == sha(root / "world.npz")
        world = dict(np.load(root / "world.npz"))
        regenerated = build_world(spec)
        for k, v in regenerated.items():
            np.testing.assert_array_equal(world[k], v)
        data_audit(world)
        cpu, bounds, trains = training_table(world, spec["arm"])
        np.testing.assert_array_equal(world["train_composite"], trains)
        stored = dict(np.load(root / "training-table.npz"))
        for k, v in zip(("tokens", "positions", "mask", "labels"), cpu, strict=True):
            np.testing.assert_array_equal(stored[k], v)
        # Decode actual train tokens to verify the renderer's labels/relations.
        for tokens, row in zip(stored["tokens"][640:], trains, strict=True):
            h, r1, _b, r2, t = map(int, row)
            assert tokens[:10].tolist() == [2, h, 3, r1, 3, r2, 4, t, 5, 1]
        initial = json.loads((root / "initialization-hashes.json").read_text())
        pair = (spec["world"], spec["initialization"])
        paired.setdefault(pair, {})[spec["arm"]] = dict(initial=initial, trains=trains)
        saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
        counts = []
        for i in range(2):
            stream = EpochStream(bounds[i + 1] - bounds[i], spec["world"] + 100 + i)
            exposure = np.zeros(stream.size, dtype=np.int64)
            for _ in range(spec["steps"]):
                np.add.at(exposure, stream.take(spec["batch_parts"][i]), 1)
            state = stream.state_dict()
            assert state["rng"] == saved["streams"][i]["rng"]
            np.testing.assert_array_equal(state["remaining"], saved["streams"][i]["remaining"])
            counts.append(
                dict(
                    rows=stream.size,
                    total=int(exposure.sum()),
                    minimum=int(exposure.min()),
                    maximum=int(exposure.max()),
                )
            )
            if spec["arm"] != "positive":
                assert exposure.min() == exposure.max()
        history = json.loads((root / "learning.json").read_text())
        assert [d["step"] for d in history] == spec["nodes"]
        assert complete["final"] == history[-1] and saved["step"] == spec["steps"]
        checks = 0
        for node in history:
            pred = dict(np.load(root / f"predictions-{node['step']:06d}.npz"))
            atom_knowledge = {
                (int(h), int(r)): bool(ok)
                for (h, r, _t), ok in zip(
                    world["atomic"], pred["atomic_answer"] == world["atomic"][:, -1], strict=True
                )
            }
            for split in SPLITS:
                rows = trains if split == "train_composite" else world[split]
                p = {k: pred[split + "_" + k] for k in ("answer", "target", "stops", "probability")}
                np.testing.assert_array_equal(p["target"], rows[:, -1])
                assert score_predictions(p) == node["metrics"][split]
                checks += len(rows)
                if split in TESTS:
                    eligible = np.array(
                        [
                            atom_knowledge[int(h), int(r1)] and atom_knowledge[int(b), int(r2)]
                            for h, r1, b, r2, _t in rows
                        ]
                    )
                    np.testing.assert_array_equal(eligible, pred[split + "_eligible"])
                    conditional = node["metrics"][split + "_conditional"]
                    assert conditional["n"] == int(eligible.sum()) and conditional[
                        "coverage"
                    ] == float(eligible.mean())
                    if eligible.any():
                        assert all(
                            conditional[k] == v
                            for k, v in score_predictions(
                                {k: v[eligible] for k, v in p.items()}
                            ).items()
                        )
                    prefix = split + "_two_calls_"
                    tp = {k: pred[prefix + k] for k in ("answer", "target", "stops", "probability")}
                    np.testing.assert_array_equal(tp["target"], rows[:, -1])
                    np.testing.assert_array_equal(pred[prefix + "first_target"], rows[:, 2])
                    fp = pred[prefix + "first_answer"]
                    fs = pred[prefix + "first_stops"]
                    first_format = (fs[:, 0] == 5) & (fs[:, 1] == 1)
                    second_full = (
                        (tp["answer"] == rows[:, -1])
                        & (tp["stops"][:, 0] == 5)
                        & (tp["stops"][:, 1] == 1)
                    )
                    ts = node["metrics"][split + "_two_calls"]
                    assert ts["accuracy"] == float((first_format & second_full).mean())
                    assert ts["path_accuracy"] == float(
                        (first_format & (fp == rows[:, 2]) & second_full).mean()
                    )
                    assert ts["first_accuracy"] == float((first_format & (fp == rows[:, 2])).mean())
                    checks += 2 * len(rows)
                row = dict(
                    run=name,
                    world=spec["world"],
                    initialization=spec["initialization"],
                    arm=spec["arm"],
                    step=node["step"],
                    split=split,
                    **node["metrics"][split],
                )
                trajectories.append(row)
                if node["step"] == spec["steps"]:
                    endpoints.append(row)
        model = construct(spec, device).eval()
        model.load_state_dict(saved["model"])
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        metrics, pred = evaluate_all(model, world, trains, device)
        stored_predictions = dict(np.load(root / f"predictions-{spec['steps']:06d}.npz"))
        assert metrics == history[-1]["metrics"]
        for key, values in pred.items():
            np.testing.assert_array_equal(values, stored_predictions[key])
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        results.append(
            dict(
                run=name,
                status="passed",
                scored_rows=checks,
                endpoint_arrays=len(pred),
                endpoint_reload_exact=True,
                evaluation_parameters_unchanged=True,
                exposures=counts,
            )
        )
        print(json.dumps(dict(run=name, audit="passed")), flush=True)
    for arms in paired.values():
        if "restricted" in arms and "broad" in arms:
            assert arms["restricted"]["initial"] == arms["broad"]["initial"]
            assert role_counts(arms["restricted"]["trains"]) == role_counts(arms["broad"]["trains"])
            for split in TESTS:
                # Worlds and regenerations already match both arms exactly.
                assert split in world
    output = Path(cfg["source_lock"]).parent
    write_json(output / "endpoint-scores.json", endpoints)
    write_json(output / "trajectory-scores.json", trajectories)
    result = dict(
        status="passed",
        utc=utc(),
        runs=results,
        total_scored_rows=sum(r["scored_rows"] for r in results),
        paired_initializations=len(paired),
    )
    write_json(output / "audit.json", result)
    return result


def report(config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = json.loads(Path(config).read_text())
    output = Path(cfg["source_lock"]).parent
    assert json.loads((output / "audit.json").read_text())["status"] == "passed"
    rows = json.loads((output / "endpoint-scores.json").read_text())
    trajectory = json.loads((output / "trajectory-scores.json").read_text())
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    worlds = sorted({r["world"] for r in rows})
    scores, pairs = {}, []
    for split in ("atomic", "train_composite", *TESTS):
        scores[split] = {
            a: float(
                np.mean([r["accuracy"] for r in rows if r["split"] == split and r["arm"] == a])
            )
            for a in arms
        }
        for w in worlds:
            selected = {
                a: [
                    r["accuracy"]
                    for r in rows
                    if r["split"] == split and r["world"] == w and r["arm"] == a
                ]
                for a in arms
            }
            if selected.get("restricted") and selected.get("broad"):
                pairs.append(
                    dict(
                        world=w,
                        split=split,
                        restricted=float(np.mean(selected["restricted"])),
                        broad=float(np.mean(selected["broad"])),
                        effect_pp=100
                        * (
                            float(np.mean(selected["broad"]))
                            - float(np.mean(selected["restricted"]))
                        ),
                    )
                )
    details, budget = (
        [],
        dict(steps=0, supervised_tokens=0, estimated_flops=0, training_seconds=0.0),
    )
    for name, overrides in cfg["runs"].items():
        root = Path(cfg["output_root"]) / name
        result = json.loads((root / "complete.json").read_text())
        spec = json.loads((root / "spec.json").read_text())
        m = result["final"]["metrics"]
        details.append(
            dict(
                run=name,
                world=spec["world"],
                initialization=spec["initialization"],
                arm=overrides["arm"],
                metrics=m,
            )
        )
        for k in ("steps", "supervised_tokens", "estimated_flops", "training_seconds"):
            budget[k] += result["final"][{"steps": "step", "training_seconds": "seconds"}.get(k, k)]
    summary = dict(
        status="complete",
        utc=utc(),
        phase=cfg["phase"],
        scores=scores,
        world_effects=pairs,
        details=details,
        budget=budget,
        source_lock_sha256=sha(cfg["source_lock"]),
        audit_sha256=sha(output / "audit.json"),
    )
    write_json(output / "summary.json", summary)
    with (output / "endpoints.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    colors = dict(restricted="#5866b0", broad="#c15646", positive="#3b9573")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, split in zip(
        axes.flat,
        ("atomic", "id_test", "first_only", "second_only", "strict_test", "train_composite"),
        strict=True,
    ):
        for arm in arms:
            nodes = sorted({r["step"] for r in trajectory})
            values = [
                100
                * np.mean(
                    [
                        r["accuracy"]
                        for r in trajectory
                        if r["arm"] == arm and r["split"] == split and r["step"] == n
                    ]
                )
                for n in nodes
            ]
            ax.plot(nodes, values, label=arm, color=colors[arm], marker="o", markersize=3)
        ax.set_title(split)
        ax.set_ylim(0, 102)
        ax.set_xlabel("training steps")
        ax.set_ylabel("full answer accuracy (%)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    write_json(
        output / "completion-manifest.json",
        dict(
            status="complete",
            utc=utc(),
            expected_runs=len(cfg["runs"]),
            completed_runs=len(details),
            expected_nodes=sum(len(dict(cfg["base"], **o)["nodes"]) for o in cfg["runs"].values()),
            source_lock_sha256=sha(cfg["source_lock"]),
            artifacts={
                str(p): sha(p)
                for p in (
                    output / "summary.json",
                    output / "audit.json",
                    output / "endpoints.csv",
                    output / "comparison.png",
                )
            },
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["preflight", "freeze", "calibrate", "run", "execute", "audit", "report"]
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="2")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg["source_lock"]).parent
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        freeze(args.config)
    elif args.command == "preflight":
        print(json.dumps(preflight(args.config)), flush=True)
    elif args.command == "calibrate":
        print(json.dumps(calibrate(args.config, args.device)), flush=True)
    elif args.command == "run":
        names = cfg["runs"] if args.run is None else {args.run: cfg["runs"][args.run]}
        for name, overrides in names.items():
            spec = dict(
                cfg["base"], **overrides, source_lock=cfg["source_lock"], config_path=args.config
            )
            run(spec, Path(cfg["output_root"]) / name, args.device)
    elif args.command == "execute":
        execute(args.config, [int(g) for g in args.gpus.split(",")], args.workers)
    elif args.command == "audit":
        print(json.dumps(audit(args.config, args.device)), flush=True)
    elif args.command == "report":
        print(json.dumps(report(args.config)["scores"]), flush=True)


if __name__ == "__main__":
    main()
