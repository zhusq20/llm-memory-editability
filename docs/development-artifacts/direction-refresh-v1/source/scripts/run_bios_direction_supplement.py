"""Numerical sensitivity and serial timing; no new independent-world selection."""

import argparse
import copy
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm_memory_editability import bios_direction as engine
from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest, load_parent, minimal_task
from llm_memory_editability.bios_direction_cross import CROSS_ARMS

ART = ROOT / "docs/development-artifacts/direction-supplement-v1"
OUT = ROOT / "results/bios-direction-supplement-v1"
CHOICES = ROOT / "docs/development-artifacts/direction-v1/selected.json"


def freeze():
    files = [Path(__file__), Path(engine.__file__), CHOICES]
    payload = dict(
        created=datetime.now(timezone.utc).isoformat(),
        rank_sensitivity=dict(
            worlds=list(range(5000, 5008)),
            seeds=[0, 1],
            train_arms=["baseline", "up-0.25"],
            methods=["func-hard-gn", "func-soft-gn"],
            rank=32,
            edit_steps=1024,
            trajectories=384,
            changed="Only sketch rank 8 -> 32; no retuning or selection on these results.",
        ),
        timing=dict(
            parents="Original miniature seeds 0,1; all nine tasks in each independent batch",
            methods=list(CROSS_ARMS),
            repetitions=2,
            order="Forward then reverse method order, same device, synchronized setup and loop",
            trajectories=144,
            caveat="Shared GPU. Engineering replays, not independent statistical replicates.",
        ),
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
    )
    if (ART / "lock.json").exists():
        assert json.loads((ART / "lock.json").read_text())["files"] == payload["files"]
        return
    write_json(ART / "lock.json", payload)
    (ART / "preregistration.md").write_text((ROOT / "docs/experimental-protocol.md").read_text())
    for p in files:
        target = ART / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())


def rank_runs(worlds, device):
    selected = json.loads(CHOICES.read_text())["choices"]
    for world, seed, train_arm in itertools.product(worlds, (0, 1), ("baseline", "up-0.25")):
        parent = (
            ROOT / "results/bios-direction-confirm-v1" / f"world-{world}-seed-{seed}" / train_arm
        )
        saved = torch.load(parent / "model-1024.pt", map_location="cpu", weights_only=False)
        model = load_parent(parent / "model-1024.pt", device)
        tasks = [
            minimal_task(seed, p, k, device, saved["labels"])
            for p, k in itertools.product((0, 1, 2), ("coherent", "independent"))
        ]
        for task in tasks:
            task.metadata.update(world=world, train_arm=train_arm)
        for arm in ("func-hard-gn", "func-soft-gn"):
            cfg = dict(selected[arm], rank=32)
            engine.edit_batch(
                model,
                0,
                tasks,
                arm,
                [copy.deepcopy(cfg) for _ in tasks],
                1024,
                OUT / "rank32" / f"world-{world}-seed-{seed}" / train_arm / arm,
            )
        print(json.dumps(dict(world=world, seed=seed, train_arm=train_arm)), flush=True)


def timing(device):
    selected = json.loads(CHOICES.read_text())["choices"]
    original_setup = engine.batch_setup
    timings = []

    def timed_setup(*args, **kwargs):
        torch.cuda.synchronize(device)
        start = time.monotonic()
        result = original_setup(*args, **kwargs)
        torch.cuda.synchronize(device)
        timings.append(time.monotonic() - start)
        return result

    engine.batch_setup = timed_setup
    for repeat, seed in itertools.product((0, 1), (0, 1)):
        model = load_parent(ROOT / f"results/bios-path-minimal-v1/seed-{seed}/parent.pt", device)
        tasks = [
            minimal_task(seed, p, k, device)
            for p, k in itertools.product((0, 1, 2), ("selective", "coherent", "independent"))
        ]
        arms = CROSS_ARMS if repeat == 0 else tuple(reversed(CROSS_ARMS))
        for arm in arms:
            out = OUT / "timing" / f"repeat-{repeat}-seed-{seed}" / arm
            if (out / "timing.json").exists():
                continue
            assert not (out / "complete.json").exists(), "Incomplete timing replay: keep separate"
            timings.clear()
            engine.edit_batch(
                model,
                0,
                tasks,
                arm,
                [copy.deepcopy(selected[arm]) for _ in tasks],
                1024,
                out,
            )
            assert len(timings) == 1
            cost = json.loads((out / "cost.json").read_text())
            write_json(
                out / "timing.json",
                dict(
                    setup_seconds=timings[0],
                    loop_seconds=cost["elapsed_seconds_including_evaluation"],
                    total_seconds=timings[0] + cost["elapsed_seconds_including_evaluation"],
                    device=torch.cuda.get_device_name(device),
                    repeat=repeat,
                    seed=seed,
                    arm=arm,
                    includes="Frozen prefix/reference, covariance/eigh, full updates and evaluation",
                    excludes="Model loading and final disk serialization",
                ),
            )
            print(json.dumps(dict(repeat=repeat, seed=seed, arm=arm)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "rank", "timing"))
    parser.add_argument("--worlds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
        return
    lock = json.loads((ART / "lock.json").read_text())
    for name, sha in lock["files"].items():
        assert digest(ROOT / name) == sha, name
    if args.command == "rank":
        assert set(args.worlds) <= set(lock["rank_sensitivity"]["worlds"])
        rank_runs(args.worlds, torch.device(args.device))
    else:
        timing(torch.device(args.device))


if __name__ == "__main__":
    main()
