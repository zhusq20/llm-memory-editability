"""Predeclared per-trajectory refresh audit after rank-sensitivity diagnostics."""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import run_bios_direction_supplement as supplement
import torch

from llm_memory_editability import bios_direction as original
from llm_memory_editability import bios_direction_refresh as corrected
from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest, load_parent, minimal_task

ART = ROOT / "docs/development-artifacts/direction-refresh-v1"
OUT = ROOT / "results/bios-direction-refresh-v1"


def freeze():
    files = [
        Path(__file__),
        Path(original.__file__),
        Path(corrected.__file__),
        Path(supplement.__file__),
        supplement.CHOICES,
    ]
    if (ART / "lock.json").exists():
        raise RuntimeError("Do not replace the frozen refresh audit")
    write_json(
        ART / "lock.json",
        dict(
            created=datetime.now(timezone.utc).isoformat(),
            rank=32,
            worlds=list(range(5000, 5008)),
            trajectories=384,
            change="Per-trajectory refresh and frozen idle counters; other settings fixed.",
            interpretation="Numerical audit; no new independent worlds or retuning.",
            files={str(p.relative_to(ROOT)): digest(p) for p in files},
        ),
    )
    for p in files:
        target = ART / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    (ART / "preregistration.md").write_text((ROOT / "docs/experimental-protocol.md").read_text())


def verify_no_rejection_equivalence(device):
    parent = ROOT / "results/bios-direction-confirm-v1/world-5000-seed-0/baseline"
    saved = torch.load(parent / "model-1024.pt", map_location="cpu", weights_only=False)
    model = load_parent(parent / "model-1024.pt", device)
    tasks = [
        minimal_task(0, p, k, device, saved["labels"])
        for p in (0, 1, 2)
        for k in ("coherent", "independent")
    ]
    for t in tasks:
        t.metadata.update(world=5000, train_arm="baseline")
    selected = json.loads(supplement.CHOICES.read_text())["choices"]
    checks = []
    for method in ("func-hard-gn", "func-soft-gn"):
        out = OUT / "equivalence" / method
        corrected.edit_batch(
            model, 0, tasks, method, [copy.deepcopy(selected[method]) for _ in tasks], 1024, out
        )
        old = parent / "edits" / method
        a = torch.load(old / "state.pt", map_location="cpu", weights_only=False)
        b = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        assert torch.equal(a["weights"], b["weights"])
        np.testing.assert_array_equal(
            np.load(old / "steps.npz")["values"], np.load(out / "steps.npz")["values"]
        )
        checks.append(dict(method=method, weights_and_all_step_values_identical=True))
    # All original primary trajectories accepted every attempt; corrected branch is dormant.
    rows = 0
    for p in (ROOT / "results/bios-direction-confirm-v1").glob("**/steps.npz"):
        values = np.load(p)["values"]
        assert values.shape[0] == 1024 and np.all(values[:, :, 6] == 1)
        rows += values.shape[1]
    assert rows == 768
    write_json(
        ART / "equivalence.json", dict(checks=checks, primary_trajectories_without_rejection=rows)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("freeze", "run", "equivalence"))
    p.add_argument("--worlds", nargs="+", type=int)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
    else:
        lock = json.loads((ART / "lock.json").read_text())
        for name, sha in lock["files"].items():
            assert digest(ROOT / name) == sha, name
        if args.command == "equivalence":
            verify_no_rejection_equivalence(torch.device(args.device))
        else:
            assert set(args.worlds) <= set(lock["worlds"])
            supplement.OUT = OUT
            supplement.engine.edit_batch = corrected.edit_batch
            supplement.rank_runs(args.worlds, torch.device(args.device))
