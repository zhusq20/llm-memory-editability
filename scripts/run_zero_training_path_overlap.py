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

Both selection designs of the contract are computed at every node: the primary
design and the pre-specified fallback (iterative attribution, z1.fallback_selection).
Each verdict group reports the verdict of the design chosen by the contract's
verdict-source rule; the other design is reported too.

``select`` runs the selection stage only, as a feasibility check before any
two-hop outcome exists (roadmap §5.0.1). It uses the same selection code as
``run``, adds eligible-query counts from unablated two-hop answers, and never
ablates a two-hop query. It writes selection-stage/{run}/{step}/selection.json
and leaves the contract's per-node directories untouched. ``select-summary``
counts powered primary nodes per verdict group and design, using the contract's
power rule, and names the design the verdict will come from.

Usage on the experiment host (inside the fixed LM image):
  python scripts/run_zero_training_path_overlap.py select --run loop4x2-phi7.2-wd0.1 --gpu 0
  python scripts/run_zero_training_path_overlap.py select-summary
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
    DESIGNS,
    Executor,
    World,
    eligible_counts,
    load_contract,
    load_model,
    majority_verdict,
    ported_node,
    predict,
    replication,
    sample_facts,
    selection_records,
    sha256,
    summarize_selection,
    verdict_source,
    z1_loop_flags,
    z1_node,
    z1_selection,
    z1_standard_flags,
)

PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT / "configs/zero-training-path-overlap-v1.json"
ROOT = Path("results/grokking-reproduction-v1")
OUT = Path("results/zero-training-path-overlap-v1")
FAILURE = "replication-failure.json"
SELECTION = "selection-stage"


def node_dir(out, run, step):
    return Path(out) / run / f"{step:07d}"


def complete(node, name="hashes.json"):
    path = node / name
    return path.exists() and json.loads(path.read_text()).get("complete") is True


def prepare(args):
    """Contract, device, verified data and the requested nodes of one run."""
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
    return contract, p, device, root, data, world, steps, code


def checked_model(root, run, step, world, device, node, p):
    """Load one checkpoint and pass the replication check, or keep the failure and stop."""
    model, spec, checkpoint = load_model(root, run, step, world.metadata, device)
    ex = Executor(model)
    rep, failures = replication(ex, world, root, run, step, p)
    if failures:
        write_json(
            node / FAILURE,
            {
                "run": run,
                "step": step,
                "panels": rep,
                "failures": failures,
                "checkpoint": str(checkpoint),
                "created_utc": utc(),
            },
        )
        raise SystemExit(f"Replication check failed at {run}@{step}: {failures}")
    return model, ex, spec, checkpoint, rep


def run(args):
    contract, p, device, root, data, world, steps, code = prepare(args)
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

        model, ex, spec, checkpoint, rep = checked_model(
            root, args.run, step, world, device, node, p
        )
        clock = mark("load_and_replication", started)
        atom_answer = predict(ex, world.atomic_tokens(np.arange(len(world.atoms))), 1)
        sample = sample_facts(world, spec["phi"], p)
        scopes = p["z1"]["scopes"]["loop" if spec["repeats"] > 1 else "standard"]
        z1 = {
            "run": args.run,
            "step": step,
            "phi": spec["phi"],
            "fact_sample_sha256": sample["sha256"],
            "atomic_all_answer_accuracy": float(
                np.mean(atom_answer == world.atoms[:, 2] + world.eo)
            ),
            # Both designs at every node; the verdict source is fixed by the contract.
            **{
                design: z1_node(ex, world, sample, atom_answer, p, scopes, design)
                for design in DESIGNS
            },
        }
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
                "replication": {"panels": rep, "failures": []},
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


def select(args):
    """Selection stage only: S_f, selectivity and eligible-query counts, no two-hop ablation."""
    contract, p, device, root, data, world, steps, code = prepare(args)
    for step in steps:
        node = node_dir(Path(args.out) / SELECTION, args.run, step)
        if complete(node, "selection.json"):
            print(
                json.dumps({"run": args.run, "step": step, "state": "skipped, complete"}),
                flush=True,
            )
            continue
        node.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model, ex, spec, checkpoint, rep = checked_model(
            root, args.run, step, world, device, node, p
        )
        atom_answer = predict(ex, world.atomic_tokens(np.arange(len(world.atoms))), 1)
        sample = sample_facts(world, spec["phi"], p)
        atom_correct = atom_answer == world.atoms[:, 2] + world.eo
        eligible = eligible_counts(ex, world, sample, atom_correct)
        designs = {}
        for design in DESIGNS:
            records = selection_records(
                sample, z1_selection(ex, world, sample, atom_answer, p, design)
            )
            for rec, count in zip(records, eligible, strict=True):
                rec["untrained_eligible"] = int(count)
            designs[design] = {
                "kinds": summarize_selection(records, eligible, p["z1"]),
                "facts": records,
            }
        write_json(
            node / "selection.json",
            {
                "complete": True,
                "stage": "selection only (feasibility check before any two-hop outcome)",
                "two_hop": "unablated answers of the sampled untrained chains, used only to count "
                "eligible queries; no two-hop query was evaluated under any ablation",
                "run": args.run,
                "step": step,
                "phi": spec["phi"],
                "fact_sample_sha256": sample["sha256"],
                "power_rule": f">= {p['z1']['power']} selective facts with >= 1 eligible query",
                "designs": designs,
                "replication": {"panels": rep, "failures": []},
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "contract_sha256": sha256(args.contract),
                "code_sha256": code,
                "seconds": time.perf_counter() - started,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "torch": torch.__version__,
                "completed_utc": utc(),
            },
        )
        print(json.dumps({"run": args.run, "step": step, "state": "selection done"}), flush=True)
        del model, ex
        if device.type == "cuda":
            torch.cuda.empty_cache()


def verdict_groups(loop_nodes, standard_nodes):
    return {
        "loop_ood": (loop_nodes, "ood"),
        "loop_id_secondary": (loop_nodes, "id"),
        "standard_id": (standard_nodes, "id"),
    }


def selection_feasibility(selection_out, loop_nodes, standard_nodes, p):
    """Powered primary nodes per verdict group and design, read from selection.json files.

    A design leaves a group 'inconclusive' when fewer than half of its primary
    nodes are powered (majority_verdict); enough_powered is the complement. The
    verdict source follows the contract rule (verdict_source).
    """
    out = {}
    for label, (nodes, kind) in verdict_groups(loop_nodes, standard_nodes).items():
        rows = {design: [] for design in DESIGNS}
        missing = []
        for run_name, step in nodes:
            node = node_dir(selection_out, run_name, step)
            if not complete(node, "selection.json"):
                missing.append(f"{run_name}@{step}")
                continue
            record = json.loads((node / "selection.json").read_text())
            for design in DESIGNS:
                summary = record["designs"][design]["kinds"][kind]
                rows[design].append(
                    {
                        "node": f"{run_name}@{step}",
                        "selective_with_eligible": summary["selective_with_eligible"],
                        "not_localized_rate": summary["not_localized_rate"],
                        "powered": summary["powered"],
                    }
                )
        group = {
            "state": "incomplete" if missing else "complete",
            "missing": missing,
            "primary_nodes": len(nodes),
        }
        for design in DESIGNS:
            powered = sum(r["powered"] for r in rows[design])
            group[design] = {
                "powered_nodes": powered,
                "enough_powered": None if missing else powered >= len(nodes) / 2,
                "nodes": rows[design],
            }
        if missing:
            group["verdict_source"] = None
            group["expected_inconclusive"] = None
        else:
            source = verdict_source(group["primary_design"]["powered_nodes"], len(nodes))
            group["verdict_source"] = source
            group["expected_inconclusive"] = not group[source]["enough_powered"]
        out[label] = group
    return out


def select_summary(args):
    contract, p = load_contract(args.contract)
    curves = learning_curves(args.root, p["runs"])
    loop_nodes, standard_nodes = primary_nodes(curves, p)
    selection_out = Path(args.out) / SELECTION
    groups = selection_feasibility(selection_out, loop_nodes, standard_nodes, p)
    summary = {
        "batch": contract["batch"],
        "contract_sha256": sha256(args.contract),
        "created_utc": utc(),
        "stage": "selection only; no two-hop query was evaluated under any ablation",
        "groups": groups,
    }
    write_json(selection_out / "selection-summary.json", summary)
    print(
        json.dumps(
            {
                label: {
                    "state": g["state"],
                    "primary_nodes": g["primary_nodes"],
                    "powered": {d: g[d]["powered_nodes"] for d in DESIGNS},
                    "verdict_source": g["verdict_source"],
                    "expected_inconclusive": g["expected_inconclusive"],
                }
                for label, g in groups.items()
            }
        )
    )


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


def z1_verdict(out, nodes, kind, flags, p, design):
    rows, missing = [], []
    for run_name, step in nodes:
        z1 = read(out, run_name, step, "z1.json")
        if z1 is None:
            missing.append(f"{run_name}@{step}")
            continue
        block = z1[design]["kinds"][kind]["primary/untrained"]
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


STANDARD_LABELS = {"supports": "separate_copy", "against": "shared"}


def group_verdict(out, nodes, kind, flags, p, labels=None):
    """Verdicts of both designs and the reported verdict chosen by the contract rule.

    The source is decided from the primary design's powered-node count, which
    rests on selection-stage information only (selective facts and unablated
    eligible queries), never on an ablated two-hop outcome.
    """
    labels = labels or {}
    by_design = {}
    for design in DESIGNS:
        result = z1_verdict(out, nodes, kind, flags, p, design)
        result["verdict"] = labels.get(result["verdict"], result["verdict"])
        by_design[design] = result
    primary = by_design["primary_design"]
    if (
        primary["verdict"] == "incomplete"
        or by_design["fallback_design"]["verdict"] == "incomplete"
    ):
        return {"verdict": "incomplete", "verdict_source": None, **by_design}
    source = verdict_source(primary["powered_nodes"], primary["primary_nodes"])
    return {"verdict": by_design[source]["verdict"], "verdict_source": source, **by_design}


def analyze(args):
    contract, p = load_contract(args.contract)
    curves = learning_curves(args.root, p["runs"])
    loop_nodes, standard_nodes = primary_nodes(curves, p)
    loop = group_verdict(args.out, loop_nodes, "ood", z1_loop_flags, p)
    loop_id = group_verdict(args.out, loop_nodes, "id", z1_loop_flags, p)
    standard = group_verdict(
        args.out, standard_nodes, "id", z1_standard_flags, p, labels=STANDARD_LABELS
    )
    available = sum(complete(node_dir(args.out, r, s)) for r in p["runs"] for s in p["nodes"])
    summary = {
        "batch": contract["batch"],
        "contract_sha256": sha256(args.contract),
        "created_utc": utc(),
        "nodes_complete": available,
        "nodes_expected": len(p["runs"]) * len(p["nodes"]),
        "z1": {"loop_ood": loop, "loop_id_secondary": loop_id, "standard_id": standard},
        "note": "verdicts computed mechanically from the frozen thresholds; each group's reported "
        "verdict comes from the design named in verdict_source (contract "
        "z1.fallback_selection); nodes are repeated measurements of one trajectory per run, "
        "not independent replications",
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
    commands = {
        "run": run,
        "select": select,
        "select-summary": select_summary,
        "analyze": analyze,
        "plan": plan,
    }
    for name in commands:
        s = sub.add_parser(name)
        s.add_argument("--contract", type=Path, default=CONTRACT)
        s.add_argument("--root", type=Path, default=ROOT)
        s.add_argument("--out", type=Path, default=OUT)
        if name in ("run", "select"):
            s.add_argument("--run", required=True)
            s.add_argument("--steps", type=int, nargs="*")
            s.add_argument("--gpu", type=int)
    args = parser.parse_args()
    commands[args.command](args)


if __name__ == "__main__":
    main()
