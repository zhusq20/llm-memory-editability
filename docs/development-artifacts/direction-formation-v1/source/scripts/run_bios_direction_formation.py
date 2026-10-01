"""Frozen phase for new-world learning and intervention predictions."""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest
from llm_memory_editability.bios_direction_formation import (
    TRAIN_ARMS,
    edit_parent,
    train,
    world_labels,
)

ART = ROOT / "docs/development-artifacts/direction-formation-v1"
OUT = ROOT / "results/bios-direction-formation-v1"


def freeze():
    files = [
        Path(__file__),
        ROOT / "src/llm_memory_editability/bios_direction_formation.py",
        ROOT / "src/llm_memory_editability/bios_direction.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "docs/development-artifacts/direction-v1/selected.json",
    ]
    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        worlds=[4000, 4001, 4002, 4003],
        seeds=[0, 1],
        arms=TRAIN_ARMS,
        steps=1024,
        lr=0.001,
        weight_decay=0.1,
        intervention_window=[0, 256],
        checkpoints=[0, 8, 32, 128, 256, 512, 1024],
        edit_steps=1024,
        design=(
            "56 learning trajectories with prospective geometry; four editors at final checkpoint. "
            "Intermediate weights retained for targeted follow-up; no full checkpoint-editor sweep."
        ),
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
    )
    if (ART / "lock.json").exists():
        assert json.loads((ART / "lock.json").read_text())["files"] == manifest["files"]
        return
    write_json(ART / "lock.json", manifest)
    write_json(ART / "worlds.json", {str(w): world_labels(w).tolist() for w in manifest["worlds"]})
    (ART / "preregistration.md").write_text((ROOT / "docs/experimental-protocol.md").read_text())
    for p in files:
        target = ART / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "train", "edit"))
    parser.add_argument("--worlds", nargs="+", type=int, default=[4000, 4001, 4002, 4003])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--arms", nargs="+", default=list(TRAIN_ARMS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
        return
    manifest = json.loads((ART / "lock.json").read_text())
    for p, h in manifest["files"].items():
        assert digest(ROOT / p) == h, p
    for world, seed, arm in itertools.product(args.worlds, args.seeds, args.arms):
        out = OUT / f"world-{world}-seed-{seed}" / arm
        (train if args.command == "train" else edit_parent)(
            world, seed, arm, torch.device(args.device), out
        )
        print(json.dumps(dict(event=args.command, world=world, seed=seed, arm=arm)), flush=True)


if __name__ == "__main__":
    main()
