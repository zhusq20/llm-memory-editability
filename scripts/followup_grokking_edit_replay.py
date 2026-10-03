"""A separately frozen role-balanced replay repair; original branches remain intact."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from run_grokking_dynamics import ARTIFACTS, RESULTS, config_path, run_name
from run_grokking_dynamics_mechanism import load_model

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.grokking_dynamics_mechanism import (
    editor,
    select_cases,
    serialize_case,
    taught_atoms,
)
from llm_memory_editability.latent_scaling import build_world
from llm_memory_editability.storage_composition import file_hash


def balanced(case, world):
    case = {**case, "tasks": dict(case["tasks"])}
    atoms = taught_atoms(world)
    successor_keys = {tuple(row[:2]) for row in case["tasks"]["successor_atomic"]}
    excluded = {case["atomic_index"]} | {
        i for i, row in enumerate(atoms) if tuple(row[:2]) in successor_keys
    }
    replay, keep = [], []
    for first in [True, False]:
        order = sorted(
            (
                i
                for i, row in enumerate(atoms)
                if (13 <= row[1] <= 16) == first and i not in excluded
            ),
            key=lambda i: tuple(atoms[i]),
        )
        replay.extend(order[:16])
        keep.extend(order[16:32])
    assert len(replay) == len(keep) == 32
    unused = [i for i in range(len(atoms)) if i not in {case["atomic_index"], *replay, *keep}]
    case.update(replay_indices=replay, keep_indices=keep, unused_indices=unused)
    case["tasks"].update(R_atomic=atoms[replay], Kdev_atomic=atoms[keep], U_atomic=atoms[unused])
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["calibration", "development"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(args.device))
    pilot = args.phase == "calibration"
    train_config = json.loads(config_path(args.phase).read_text())
    specs = [s for s in train_config["specs"] if s["weight_decay"] == (0.01 if pilot else 0.1)]
    world = build_world(specs[0])
    cases = [
        balanced(c, world)
        for c in select_cases(world, n_per_group_role=1 if pilot else 2, calibration=True)
        if c["role"] == "second"
    ]
    folder = ARTIFACTS / "balanced-editor" / args.phase
    sources = [
        str(Path(__file__).relative_to(Path.cwd())),
        "src/llm_memory_editability/grokking_dynamics_mechanism.py",
        "scripts/run_grokking_dynamics_mechanism.py",
        "tests/test_grokking_edit_replay.py",
    ]
    rates = (
        [0.0001, 0.0003, 0.001, 0.003]
        if pilot
        else [
            json.loads((ARTIFACTS / "balanced-editor/calibration/decision.json").read_text())["lr"]
        ]
    )
    if rates == [None]:
        raise ValueError("Balanced editor calibration failed")
    config = {
        "phase": args.phase,
        "source": {p: file_hash(p) for p in sources},
        "specs": specs,
        "cases": [serialize_case(c) for c in cases],
        "learning_rates": rates,
        "nodes": [16000] if pilot else [8000, 512000],
        "scope": "Posthoc premise repair: same familiar second-hop targets, "
        "same parents/200 steps/CE+KL; R and Kdev each contain 16 first/16 second facts. "
        "Main followup uses wd=.1, both architectures and initializations. "
        "Selection reads pilot E and Kdev only; D/U never select LR.",
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        write_json(path, config)
        for p in sources:
            target = folder / "source" / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    records = []
    for spec in specs:
        for node in config["nodes"]:
            model, parent = load_model(spec, args.phase, node, args.device)
            out = RESULTS / "balanced-editor" / args.phase / f"{run_name(spec)}-t{node:06d}"
            out.mkdir(parents=True, exist_ok=True)
            for i, case in enumerate(cases):
                for lr in rates:
                    for arm in ["edit", "review"]:
                        name = f"case{i:02d}-lr{lr:g}-{arm}"
                        result_path = out / f"{name}.json"
                        if result_path.exists():
                            record = json.loads(result_path.read_text())
                        else:
                            record, raw, weight = editor(model, case, args.device, arm, lr)
                            np.savez_compressed(out / f"{name}.npz", **raw)
                            torch.save({"weight": weight, "parent": parent}, out / f"{name}.pt")
                            record.update(
                                name=run_name(spec),
                                checkpoint=node,
                                architecture="standard2" if spec["layers"] == 2 else "loop2",
                            )
                            write_json(result_path, record)
                        records.append(record)
    if pilot:
        candidates = []
        for lr in rates:
            selected = [r for r in records if r["lr"] == lr]
            passed = all(
                r["history"][-1]["metrics"]["E_new" if r["arm"] == "edit" else "E_old"]["accuracy"]
                == 1
                and r["history"][-1]["metrics"]["Kdev_atomic"]["accuracy"] >= 0.95
                for r in selected
            )
            candidates.append({"lr": lr, "passed": passed})
        selected = next((r for r in candidates if r["passed"]), None)
        write_json(
            folder / "decision.json",
            {
                "passed": selected is not None,
                "lr": selected["lr"] if selected else None,
                "candidates": candidates,
            },
        )
    summary = {
        "branches": len(records),
        "updates": len(records) * 200,
        "all_exact_tensor_reloads_passed": all(r["exact_weight_reload_passed"] for r in records),
        "records": records,
    }
    write_json(folder / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    main()
