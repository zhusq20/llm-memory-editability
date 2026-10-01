#!/usr/bin/env python3
"""Independent receipts, lineage, node, exposure and optimizer audit of the full matrix."""

from __future__ import annotations

import itertools
from datetime import datetime

import torch

from llm_memory_editability.hebbian_learning import (
    ARTIFACTS,
    CONFIG_PATH,
    DATA,
    RESULTS,
    config,
    now,
    read_json,
    sha256,
    write_json,
)
from llm_memory_editability.hebbian_model import aggregate_hash, tensor_hash


def main():
    cfg = config()
    episodes = read_json(DATA / "episodes.json")
    pools = read_json(DATA / "pools.json")
    b_lock = read_json(ARTIFACTS / "B-lock.json")
    c_lock = read_json(ARTIFACTS / "C-lock.json")
    blr = b_lock["learning_rate"]
    paths = [
        RESULTS / f"B/dev-lr{lr:g}-e{e}"
        for lr in cfg["adapt"]["learning_rates"]
        for e in range(len(episodes["B_dev"]))
    ]
    paths += [RESULTS / f"B/eval-lr{blr:g}-e{e}" for e in range(len(episodes["B_eval"]))]
    paths += [RESULTS / f"C/dev-C0-lr{lr:g}" for lr in cfg["form"]["learning_rates"]]
    paths += [RESULTS / f"C/dev-C1-lambda{lam:g}" for lam in cfg["form"]["lambdas"]]
    paths += [RESULTS / "C/dev-C2-engineering"]
    for c, lam in [("C0", 0), ("C1", 0.01), ("C1", 0.1)]:
        paths += [
            RESULTS / f"C/adapt-dev-{c}-lambda{lam:g}-e{e}" for e in range(len(episodes["C_dev"]))
        ]
    paths += [
        RESULTS / f"C/form-{c}-s{s}" for c, s in itertools.product(["C0", "C1", "C2"], [0, 1, 2])
    ]
    paths += [
        RESULTS / f"C/adapt-{c}-s{s}-e{e}"
        for c, s, e in itertools.product(
            ["C0", "C1", "C2"], [0, 1, 2], range(len(episodes["C_eval"]))
        )
    ]
    missing = [str(p.relative_to(RESULTS)) for p in paths if not (p / "complete.json").exists()]
    if missing:
        write_json(
            ARTIFACTS / "completion-audit.json", {"time": now(), "pass": False, "missing": missing}
        )
        raise RuntimeError(f"Missing {len(missing)} completed runs")
    first = paths[0]
    base = read_json(first / "complete.json")["final_hashes"].copy()
    initial = torch.load(first / "checkpoint-0000.pt", map_location="cpu", weights_only=False)
    base["model.layers.14.mlp.down_proj.weight"] = tensor_hash(
        initial["parameters"]["down_proj.weight"]
    )
    base_hash = aggregate_hash(base)
    checks = []
    all_by_id = {r["case_id"]: r for rows in pools.values() for r in rows}
    for path in paths:
        receipt = read_json(path / "complete.json")
        meta = receipt["meta"]
        form = meta["phase"] == "form"
        scope = [
            "model.layers.14.mlp." + name
            for name in (["up_proj.weight", "gate_proj.weight"] if form else ["down_proj.weight"])
        ]
        parent_path = meta["parent_path"]
        parent = (
            read_json(__import__("pathlib").Path(parent_path).parent / "complete.json")
            if parent_path
            else None
        )
        parent_hashes = parent["final_hashes"] if parent else base
        nodes = [n for n in cfg["form" if form else "adapt"]["nodes"] if n <= meta["steps"]]
        expected_exposure = meta["steps"] * 8 // len(meta["case_ids"])
        state = torch.load(
            path / f"checkpoint-{meta['steps']:04d}.pt", map_location="cpu", weights_only=False
        )
        optimizer_steps = [int(s["step"]) for s in state["optimizer"]["state"].values()]
        source_checks = [
            any(
                (snapshot / name).exists() and sha256(snapshot / name) == digest
                for snapshot in ARTIFACTS.glob("execution-source*")
            )
            for name, digest in meta["source_hashes"].items()
        ]
        evaluation_ok = True
        for node in nodes:
            rows = read_json(path / f"evaluation-{node:04d}.json")["rows"]
            expected = (
                len(pools["V_form"]) if form and meta["split"] == "dev" else len(meta["case_ids"])
            )
            evaluation_ok &= len(rows) == 3 * expected
            evaluation_ok &= len({(r["case_id"], r["view_id"]) for r in rows}) == len(rows)
            evaluation_ok &= all(
                r["parent_hash"] == meta["parent_hash"] and r["step"] == node for r in rows
            )
        parent_ok = meta["parent_hash"] == (parent["final_model_hash"] if parent else base_hash)
        frozen_ok = all(
            receipt["final_hashes"][name] == digest
            for name, digest in parent_hashes.items()
            if name not in scope
        )
        c2_ok = True
        if form and meta["condition"] == "C2":
            for batch in read_json(path / "batch-manifest.json"):
                groups = batch["groups"]
                c2_ok &= sorted(i for group in groups for i in group) == list(range(24))
                c2_ok &= all(g[0] // 3 != g[1] // 3 and g[0] // 3 != g[2] // 3 for g in groups)
                c2_ok &= all(case in all_by_id for case in batch["case_ids"])
        prospective_ok = True
        if meta["condition"] == "B" and meta["split"] == "eval":
            prediction = read_json(path / "prospective-predictions.json")
            timestamp = datetime.fromisoformat(prediction["saved_at"]).timestamp()
            prospective_ok = (
                datetime.fromisoformat(b_lock["time"]).timestamp()
                < timestamp
                < (path / "evaluation-0000.json").stat().st_mtime
            )
        if form and meta["split"] == "eval":
            prospective_ok &= (
                datetime.fromisoformat(c_lock["time"]).timestamp()
                < (path / "contract.json").stat().st_mtime
            )
        check = {
            "run": str(path.relative_to(RESULTS)),
            "parent_lineage": parent_ok,
            "scope": sorted(meta["trainable_names"]) == sorted(scope),
            "all_frozen_parameters_unchanged": frozen_ok,
            "evaluations": evaluation_ok,
            "exposures": set(receipt["exposures"].values()) == {expected_exposure},
            "optimizer_steps": optimizer_steps == [meta["steps"]] * len(scope),
            "cursor": state["data_cursor"] == meta["steps"],
            "c2_grouping": c2_ok,
            "prospective_freezing": prospective_ok,
            "exact_source_snapshot": all(source_checks),
            "config_hash": meta["config_sha256"] == sha256(CONFIG_PATH),
        }
        check["pass"] = all(v for k, v in check.items() if k != "run")
        checks.append(check)
    lock = read_json(ARTIFACTS / "data-lock.json")
    data_ok = all(sha256(DATA / name) == digest for name, digest in lock["files"].items())
    result = {
        "time": now(),
        "expected_training_runs": len(paths),
        "checks": checks,
        "data_hashes": data_ok,
        "A": read_json(ARTIFACTS / "A/audit.json")["pass"],
        "pass": all(r["pass"] for r in checks) and data_ok,
    }
    write_json(ARTIFACTS / "completion-audit.json", result)
    print("Full matrix audit:", result["pass"], len(paths), "runs", flush=True)
    if not result["pass"]:
        raise RuntimeError("Completion audit failed")


if __name__ == "__main__":
    main()
