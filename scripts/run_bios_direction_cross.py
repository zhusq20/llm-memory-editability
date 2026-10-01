"""Frozen eight-layer transfer phase, with independent E/V tuning."""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm_memory_editability import bios_direction as engine
from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import (
    ROOT,
    candidate_configs,
    digest,
    edit_batch,
    load_parent,
)
from llm_memory_editability.bios_direction_cross import (
    CROSS_ARMS,
    bounded_representation_geometry,
    cross_tasks,
)

OUT = ROOT / "results/bios-direction-cross-v1"
ART = ROOT / "docs/development-artifacts/direction-cross-v1"
CONFIG = ROOT / "configs/bios-direction-cross-v1.json"


def parent(world, seed):
    return (
        ROOT
        / "results/bios-cross-scale-dev-v1/width-256"
        / f"world-{world}-seed-{seed}-neither/model-15360.pt"
    )


def freeze():
    files = [
        CONFIG,
        Path(__file__),
        ROOT / "src/llm_memory_editability/bios_direction.py",
        ROOT / "src/llm_memory_editability/bios_direction_cross.py",
        ROOT / "src/llm_memory_editability/bios_cross.py",
        ROOT / "src/llm_memory_editability/bios_cross_train.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "src/llm_memory_editability/bios_data.py",
    ]
    files += [parent(w, s) for w, s in itertools.product((0, 1), (0, 1))]
    lock = dict(
        created=datetime.now(timezone.utc).isoformat(),
        config=json.loads(CONFIG.read_text()),
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
        torch=torch.__version__,
    )
    if (ART / "lock.json").exists():
        assert json.loads((ART / "lock.json").read_text())["files"] == lock["files"]
        return
    write_json(ART / "lock.json", lock)
    write_json(
        ART / "tasks.json",
        [
            t.manifest()
            for w, s in itertools.product((0, 1), (0, 1))
            for t in cross_tasks(w, s, "cpu")
        ],
    )
    (ART / "preregistration.md").write_text((ROOT / "docs/experimental-protocol.md").read_text())
    for p in files:
        if p.suffix in (".py", ".json"):
            target = ART / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())


def verify():
    lock = json.loads((ART / "lock.json").read_text())
    for p, h in lock["files"].items():
        assert digest(ROOT / p) == h, p


def run(command, world, seed, device, arms):
    engine.representation_geometry = bounded_representation_geometry
    model = load_parent(parent(world, seed), device)
    tasks = cross_tasks(world, seed, device)
    selected = (
        json.loads((ART / "selected.json").read_text())["choices"] if command == "main" else None
    )
    for arm in arms:
        if command == "tune":
            base = tasks[:2]
            configs = candidate_configs(arm)
            batch = [task for task in base for cfg in configs]
            cfgs = configs * len(base)
            steps = 256
        else:
            batch = tasks
            cfgs = [selected[arm] for _ in batch]
            steps = 1024
        records = edit_batch(
            model, 4, batch, arm, cfgs, steps, OUT / command / f"world-{world}-seed-{seed}" / arm
        )
        print(
            json.dumps(dict(event=command, world=world, seed=seed, arm=arm, runs=len(records))),
            flush=True,
        )


def select():
    summaries = {}
    for arm in CROSS_ARMS:
        records = []
        for seed in (0, 1):
            records += json.loads(
                (OUT / "tune" / f"world-0-seed-{seed}" / arm / "metrics.json").read_text()
            )
        rows = []
        for cfg in candidate_configs(arm):
            group = [r for r in records if r["config"] == cfg]
            assert len(group) == 4
            rows.append(
                dict(
                    config=cfg,
                    v_broken=sum(r["timeline"][-1]["sets"]["V"]["broken"] for r in group),
                    e_nll=sum(r["timeline"][-1]["sets"]["E"]["nll"] for r in group) / 4,
                )
            )
        summaries[arm] = rows
    choices = {}
    best = []
    for level in (0.0, 1.0, 10.0, 100.0):
        pair = [
            min(
                (r for r in summaries[a] if r["config"]["level"] == level),
                key=lambda r: (r["v_broken"], r["e_nll"]),
            )
            for a in ("func-soft-adam", "func-soft-gn")
        ]
        best.append((sum(r["v_broken"] for r in pair), sum(r["e_nll"] for r in pair), pair))
    pair = min(best, key=lambda x: x[:2])[2]
    for a, r in zip(("func-soft-adam", "func-soft-gn"), pair, strict=True):
        choices[a] = r["config"]
    for arm in ("repr-hard-adam", "func-hard-gn"):
        choices[arm] = min(summaries[arm], key=lambda r: (r["v_broken"], r["e_nll"]))["config"]
    write_json(ART / "tuning-summary.json", summaries)
    write_json(
        ART / "selected.json",
        dict(
            created=datetime.now(timezone.utc).isoformat(),
            choices=choices,
            rule=(
                "World 0, seeds 0/1, chain 0, first group, both update types; "
                "minimum V broken then E NLL; shared soft objective; no U/D selection"
            ),
        ),
    )
    print(json.dumps(choices), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "tune", "select", "main"))
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--arms", nargs="+", default=list(CROSS_ARMS))
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
    run(args.command, args.world, args.seed, torch.device(args.device), args.arms)


if __name__ == "__main__":
    main()
