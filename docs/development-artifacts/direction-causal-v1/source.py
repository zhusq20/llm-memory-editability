"""Frozen early/late/permanent freezing for the development-selected component."""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest
from llm_memory_editability.bios_direction_formation import edit_parent, train

ART = ROOT / "docs/development-artifacts/direction-causal-v1"
OUT = ROOT / "results/bios-direction-causal-v1"
PRED = ROOT / "docs/development-artifacts/direction-prediction-v1/predictor.json"


def freeze():
    selected = json.loads(PRED.read_text())["selected_training_arm"].split("-")[0]
    files = [
        Path(__file__),
        PRED,
        ROOT / "src/llm_memory_editability/bios_direction_formation.py",
        ROOT / "src/llm_memory_editability/bios_direction.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "docs/development-artifacts/direction-v1/selected.json",
    ]
    payload = dict(
        created=datetime.now(timezone.utc).isoformat(),
        worlds=[4000, 4001, 4002, 4003],
        seeds=[0, 1],
        arms=[selected + "-" + name for name in ("freezeearly", "freezelate", "freezeall")],
        methods=["func-soft-adam", "func-hard-gn"],
        training_trajectories=24,
        editing_trajectories=288,
        hypothesis=(
            "Early freezing and restoration should differ from late freezing "
            "if early organization affects independent editability."
        ),
        limitation="Development only; global impairment cannot establish selective necessity.",
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
    )
    if (ART / "lock.json").exists():
        assert json.loads((ART / "lock.json").read_text())["files"] == payload["files"]
        return
    write_json(ART / "lock.json", payload)
    (ART / "source.py").write_bytes(Path(__file__).read_bytes())


def run(worlds, device):
    lock = json.loads((ART / "lock.json").read_text())
    for p, h in lock["files"].items():
        assert digest(ROOT / p) == h, p
    for world, seed, arm in itertools.product(worlds, lock["seeds"], lock["arms"]):
        out = OUT / f"world-{world}-seed-{seed}" / arm
        train(world, seed, arm, device, out)
        edit_parent(world, seed, arm, device, out, edit_arms=lock["methods"])
        print(json.dumps(dict(world=world, seed=seed, arm=arm)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--worlds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
    else:
        run(args.worlds, torch.device(args.device))
