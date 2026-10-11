"""Execute and analyze zero-training-path-overlap-v1 (Z1 path-overlap interventions).

The contract is configs/zero-training-path-overlap-v1.json; the bridge-format
check (Z2) is withdrawn there and is not run. ``run`` evaluates the existing
grokking-reproduction-v1 checkpoints of one run node by node in float32. It
first replicates the stored predictions of that node; if the check fails it
keeps replication-failure.json and stops. A completed node holds exactly the
contract's per-node outputs: z1.json, ported.json and hashes.json, the last
also recording the replication check. ``analyze`` reads only the node outputs
and applies the frozen interpretation mechanically. ``plan`` lists the 8 x 14
node tasks and their state.

Usage on the experiment host (inside the fixed LM image):
  python scripts/run_zero_training_path_overlap.py run --run loop4x2-phi7.2-wd0.1 --gpu 0
  python scripts/run_zero_training_path_overlap.py analyze
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.path_overlap import (
    CODE_FILES,
    Executor,
    World,
    load_contract,
    load_model,
    majority_verdict,
    ported_node,
    predict,
    replication,
    sample_facts,
    sha256,
    z1_loop_flags,
    z1_node,
    z1_standard_flags,
)

PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT / "configs/zero-training-path-overlap-v1.json"
ROOT = Path("results/grokking-reproduction-v1")
OUT = Path("results/zero-training-path-overlap-v1")
FAILURE = "replication-failure.json"


def node_dir(out, run, step):
    return Path(out) / run / f"{step:07d}"


def complete(node):
    path = node / "hashes.json"
    return path.exists() and json.loads(path.read_text()).get("complete") is True


def run(args):
    contract, p = load_contract(args.contract)
    assert p["code_files"] == CODE_FILES, "Code files differ from the contract"
    assert p["outputs"][-1] == "hashes.json", "hashes.json must be the last per-node output"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    root = Path(args.root)
    data = root / "data"
    assert sha256(data / "world.npz") == contract["source"]["world_sha256"], "world.npz differs"
    assert sha256(data / "panels.npz") == contract["source"]["panels_sha256"], "panels.npz differs"
    world = World(data)
    assert args.run in p["runs"], f"{args.run} is not a contract run"
    steps = args.steps or p["nodes"]
    assert set(steps) <= set(p["nodes"]), "Only contract nodes may be evaluated"
    code = {
        name: (sha256(PROJECT / name) if (PROJECT / name).exists() else None) for name in CODE_FILES
    }
    for step in steps:
        node = node_dir(args.out, args.run, step)
        if complete(node):
            print(
                json.dumps({"run": args.run, "step": step, "state": "skipped, complete"}),
                flush=True,
            )
            continue
        node.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timing, started = {}, time.perf_counter()

        def mark(stage, since, timing=timing):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing[stage] = time.perf_counter() - since
            return time.perf_counter()

        model, spec, checkpoint = load_model(root, args.run, step, world.metadata, device)
        ex = Executor(model)
        clock = mark("load", started)
        rep, failures = replication(ex, world, root, args.run, step, p)
        clock = mark("replication", clock)
        if failures:
            write_json(
                node / FAILURE,
                {
                    "run": args.run,
                    "step": step,
                    "panels": rep,
                    "failures": failures,
                    "checkpoint": str(checkpoint),
                    "created_utc": utc(),
                },
            )
            raise SystemExit(f"Replication check failed at {args.run}@{step}: {failures}")
        atom_answer = predict(ex, world.atomic_tokens(np.arange(len(world.atoms))), 1)
        sample = sample_facts(world, spec["phi"], p)
        scopes = p["z1"]["scopes"]["loop" if spec["repeats"] > 1 else "standard"]
        z1 = z1_node(ex, world, sample, atom_answer, p, scopes)
        z1.update(
            {
                "run": args.run,
                "step": step,
                "phi": spec["phi"],
                "fact_sample_sha256": sample["sha256"],
                "atomic_all_answer_accuracy": float(
                    np.mean(atom_answer == world.atoms[:, 2] + world.eo)
                ),
            }
        )
        write_json(node / "z1.json", z1)
        clock = mark("z1", clock)
        write_json(
            node / "ported.json",
            {"run": args.run, "step": step, **ported_node(ex, world, spec["phi"], p)},
        )
        clock = mark("ported", clock)
        timing["total"] = time.perf_counter() - started
        write_json(
            node / "hashes.json",
            {
                "complete": True,
                "run": args.run,
                "step": step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "world_sha256": sha256(data / "world.npz"),
                "panels_sha256": sha256(data / "panels.npz"),
                "contract_sha256": sha256(args.contract),
                "code_sha256": code,
                "replication": {"panels": rep, "failures": failures},
                "outputs_sha256": {name: sha256(node / name) for name in p["outputs"][:-1]},
                "seconds": timing,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "tf32": False,
                "pid": os.getpid(),
                "completed_utc": utc(),
            },
        )
        print(
            json.dumps({"run": args.run, "step": step, "seconds": round(timing["total"], 1)}),
            flush=True,
        )
        del model, ex
        if device.type == "cuda":
            torch.cuda.empty_cache()


# Analysis ---------------------------------------------------------------------


def learning_curves(root, runs):
    curves = {}
    for run_name in runs:
        history = json.loads((Path(root) / "development" / run_name / "learning.json").read_text())
        curves[run_name] = {
            r["step"]: {k: v["accuracy"] for k, v in r["metrics"].items()} for r in history
        }
    return curves


def primary_nodes(curves, p):
    q = p["z1"]
    io_min, ood_min = q["loop_nodes"]
    loop = [
        (r, s)
        for r in p["runs"]
        if r.startswith("loop")
        for s in p["nodes"]
        if curves[r][s]["test_io"] >= io_min and curves[r][s]["atomic_ood"] >= ood_min
    ]
    standard = [
        (r, s)
        for r in p["runs"]
        if r.startswith("standard")
        for s in p["nodes"]
        if curves[r][s]["test_ii"] >= q["standard_nodes"]
    ]
    assert len(loop) == q["loop_node_count"], f"Loop primary nodes {len(loop)} != contract"
    assert len(standard) == q["standard_node_count"], (
        f"Standard primary nodes {len(standard)} != contract"
    )
    return loop, standard


def read(out, run_name, step, name):
    node = node_dir(out, run_name, step)
    if not complete(node):
        return None
    return json.loads((node / name).read_text())


def z1_verdict(out, nodes, kind, flags, p):
    rows, missing = [], []
    for run_name, step in nodes:
        z1 = read(out, run_name, step, "z1.json")
        if z1 is None:
            missing.append(f"{run_name}@{step}")
            continue
        block = z1["kinds"][kind]["primary/untrained"]
        rows.append(
            {
                "node": f"{run_name}@{step}",
                "facts": block["facts_with_eligible"],
                "powered": block["facts_with_eligible"] >= p["z1"]["power"],
                **flags(block, p),
            }
        )
    if missing:
        return {"verdict": "incomplete", "missing": missing, "nodes": rows}
    powered = [r for r in rows if r["powered"]]
    support = sum(r["support"] for r in powered)
    against = sum(r["against"] for r in powered)
    return {
        "verdict": majority_verdict(support, against, len(powered), len(rows)),
        "powered_nodes": len(powered),
        "primary_nodes": len(rows),
        "support_nodes": support,
        "against_nodes": against,
        "nodes": rows,
    }


def analyze(args):
    contract, p = load_contract(args.contract)
    curves = learning_curves(args.root, p["runs"])
    loop_nodes, standard_nodes = primary_nodes(curves, p)
    loop = z1_verdict(args.out, loop_nodes, "ood", z1_loop_flags, p)
    loop_id = z1_verdict(args.out, loop_nodes, "id", z1_loop_flags, p)
    standard = z1_verdict(args.out, standard_nodes, "id", z1_standard_flags, p)
    standard["verdict"] = {"supports": "separate_copy", "against": "shared"}.get(
        standard["verdict"], standard["verdict"]
    )
    available = sum(complete(node_dir(args.out, r, s)) for r in p["runs"] for s in p["nodes"])
    summary = {
        "batch": contract["batch"],
        "contract_sha256": sha256(args.contract),
        "created_utc": utc(),
        "nodes_complete": available,
        "nodes_expected": len(p["runs"]) * len(p["nodes"]),
        "z1": {"loop_ood": loop, "loop_id_secondary": loop_id, "standard_id": standard},
        "note": "verdicts computed mechanically from the frozen thresholds; nodes are repeated "
        "measurements of one trajectory per run, not independent replications",
    }
    write_json(Path(args.out) / "summary.json", summary)
    print(
        json.dumps(
            {
                "nodes_complete": available,
                "nodes_expected": summary["nodes_expected"],
                "z1_loop": loop["verdict"],
                "z1_standard": standard["verdict"],
            }
        )
    )


def plan(args):
    _, p = load_contract(args.contract)
    seconds = []
    rows = []
    for run_name in p["runs"]:
        for step in p["nodes"]:
            node = node_dir(args.out, run_name, step)
            state = (
                "complete"
                if complete(node)
                else ("replication-failed" if (node / FAILURE).exists() else "pending")
            )
            if state == "complete":
                seconds.append(json.loads((node / "hashes.json").read_text())["seconds"]["total"])
            rows.append((run_name, step, state))
    for r in rows:
        print(*r, sep="\t")
    pending = sum(r[2] != "complete" for r in rows)
    if seconds:
        print(
            f"# {pending} pending; mean {np.mean(seconds):.0f}s per completed node; "
            f"remaining ~{pending * np.mean(seconds) / 3600:.2f} GPU hours"
        )
    else:
        print(f"# {pending} pending; no completed node to estimate time")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "analyze", "plan"):
        s = sub.add_parser(name)
        s.add_argument("--contract", type=Path, default=CONTRACT)
        s.add_argument("--root", type=Path, default=ROOT)
        s.add_argument("--out", type=Path, default=OUT)
        if name == "run":
            s.add_argument("--run", required=True)
            s.add_argument("--steps", type=int, nargs="*")
            s.add_argument("--gpu", type=int)
    args = parser.parse_args()
    {"run": run, "analyze": analyze, "plan": plan}[args.command](args)


if __name__ == "__main__":
    main()
