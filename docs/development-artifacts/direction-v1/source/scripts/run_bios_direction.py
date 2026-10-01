"""Run separately frozen direction tuning and complete editing trajectories."""

import argparse
import copy
import itertools
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import (
    ARMS,
    ROOT,
    candidate_configs,
    digest,
    edit_batch,
    load_parent,
    minimal_task,
)

OUTPUT = ROOT / "results/bios-direction-v1"
ARTIFACT = ROOT / "docs/development-artifacts/direction-v1"
CONFIG = ROOT / "configs/bios-direction-v1.json"


def freeze():
    config = json.loads(CONFIG.read_text())
    files = [
        CONFIG,
        ROOT / "src/llm_memory_editability/bios_direction.py",
        Path(__file__),
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "src/llm_memory_editability/bios_data.py",
    ]
    files += [ROOT / f"results/bios-path-minimal-v1/seed-{seed}/parent.pt" for seed in (0, 1)]
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        config=config,
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
        python=sys.version,
        executable=sys.executable,
        torch=torch.__version__,
        platform=platform.platform(),
        arms={arm: candidate_configs(arm) for arm in ARMS},
    )
    if (ARTIFACT / "lock.json").exists():
        assert json.loads((ARTIFACT / "lock.json").read_text())["files"] == manifest["files"], (
            "Frozen sources changed"
        )
        return
    write_json(ARTIFACT / "lock.json", manifest)
    (ARTIFACT / "preregistration.md").write_text(
        (ROOT / "docs/experimental-protocol.md").read_text()
    )
    for p in files:
        if p.suffix in (".py", ".json"):
            target = ARTIFACT / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    write_json(
        ARTIFACT / "tasks.json",
        [
            minimal_task(s, p, k, "cpu").manifest()
            for s, p, k in itertools.product(
                (0, 1), (0, 1, 2), ("selective", "coherent", "independent")
            )
        ],
    )
    print(json.dumps({"event": "frozen", "hash": digest(ARTIFACT / "lock.json")}), flush=True)


def verify():
    lock = json.loads((ARTIFACT / "lock.json").read_text())
    for path, sha in lock["files"].items():
        assert digest(ROOT / path) == sha, f"Frozen source changed: {path}"
    return lock


def tune(seed, device, arms):
    model = load_parent(ROOT / f"results/bios-path-minimal-v1/seed-{seed}/parent.pt", device)
    for arm in arms:
        tasks = []
        configs = []
        for kind, cfg in itertools.product(
            ("selective", "coherent", "independent"), candidate_configs(arm)
        ):
            tasks.append(minimal_task(seed, 0, kind, device))
            configs.append(cfg)
        records = edit_batch(
            model, 0, tasks, arm, configs, 256, OUTPUT / "tune" / f"seed-{seed}" / arm
        )
        print(
            json.dumps({"event": "tune", "seed": seed, "arm": arm, "records": len(records)}),
            flush=True,
        )


def select():
    # Shared soft penalty per object: objective held fixed across the two solvers.
    summaries = {}
    for arm in ARMS:
        rows = []
        for seed in (0, 1):
            records = json.loads(
                (OUTPUT / "tune" / f"seed-{seed}" / arm / "metrics.json").read_text()
            )
            for record in records:
                metrics = record["timeline"][-1]["sets"]
                rows.append(
                    dict(
                        config=record["config"],
                        v_broken=metrics["V"]["broken"],
                        e_nll=metrics["E"]["nll"],
                    )
                )
        aggregated = []
        for cfg in candidate_configs(arm):
            matching = [x for x in rows if x["config"] == cfg]
            assert len(matching) == 6
            aggregated.append(
                dict(
                    config=cfg,
                    v_broken=sum(x["v_broken"] for x in matching),
                    e_nll=sum(x["e_nll"] for x in matching) / 6,
                )
            )
        summaries[arm] = aggregated
    choices = {}
    for obj in ("func", "repr"):
        soft = [f"{obj}-soft-{solver}" for solver in ("adam", "gn")]
        levels = sorted({x["config"]["level"] for x in summaries[soft[0]]})
        candidates = []
        for level in levels:
            best = [
                min(
                    (x for x in summaries[arm] if x["config"]["level"] == level),
                    key=lambda x: (x["v_broken"], x["e_nll"]),
                )
                for arm in soft
            ]
            candidates.append(
                (sum(x["v_broken"] for x in best), sum(x["e_nll"] for x in best), best)
            )
        best = min(candidates, key=lambda x: x[:2])[2]
        for arm, row in zip(soft, best, strict=True):
            choices[arm] = row["config"]
        for solver in ("adam", "gn"):
            arm = f"{obj}-hard-{solver}"
            choices[arm] = min(summaries[arm], key=lambda x: (x["v_broken"], x["e_nll"]))["config"]
    write_json(ARTIFACT / "tuning-summary.json", summaries)
    write_json(
        ARTIFACT / "selected.json",
        dict(
            created=datetime.now(timezone.utc).isoformat(),
            choices=choices,
            rule=(
                "Minimize summed V broken, then mean E NLL; soft penalty shared across solvers; "
                "final accepted-budget sample; no U"
            ),
        ),
    )
    print(json.dumps(choices), flush=True)


def main_runs(seed, device, arms):
    selected = json.loads((ARTIFACT / "selected.json").read_text())["choices"]
    model = load_parent(ROOT / f"results/bios-path-minimal-v1/seed-{seed}/parent.pt", device)
    for arm in arms:
        tasks = [
            minimal_task(seed, p, k, device)
            for p, k in itertools.product((0, 1, 2), ("selective", "coherent", "independent"))
        ]
        records = edit_batch(
            model,
            0,
            tasks,
            arm,
            [copy.deepcopy(selected[arm]) for _ in tasks],
            1024,
            OUTPUT / "main" / f"seed-{seed}" / arm,
        )
        print(
            json.dumps({"event": "main", "seed": seed, "arm": arm, "records": len(records)}),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "tune", "select", "main"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
        return
    verify()
    if args.command == "select":
        select()
        return
    print(
        json.dumps(
            {
                "event": "start",
                "command": args.command,
                "seed": args.seed,
                "gpu": torch.cuda.get_device_name(args.device),
                "visible": os.getenv("CUDA_VISIBLE_DEVICES"),
            }
        ),
        flush=True,
    )
    (tune if args.command == "tune" else main_runs)(args.seed, torch.device(args.device), args.arms)


if __name__ == "__main__":
    main()
