"""Sequential writes protected by address neighbours (neighbor-protection-development-v1).

Identical to the no-bound arms of ce-cross-sample-ablation-v1 (same native300k model, panel,
streams, 20 fresh-Adam CE steps per fact, KL factor, clip, no element bound, full readout after
every write) except for which old examples the KL protects at each step.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import run_ce_cross_sample as parent  # noqa: E402
import torch  # noqa: E402

from llm_memory_editability.ce_cross_sample import (  # noqa: E402
    facts_by_id,
    read_state,
    settings_for,
)
from llm_memory_editability.ce_cross_sample_ablation import write_fact  # noqa: E402
from llm_memory_editability.ce_cross_sample_ablation_summary import (  # noqa: E402
    summarize_streams,
)
from llm_memory_editability.ce_scope_editing import arrays, restore  # noqa: E402
from llm_memory_editability.grok_depth import write_json  # noqa: E402
from llm_memory_editability.neighbor_protection import (  # noqa: E402
    eligibility,
    neighbor_pools,
    neighbor_schedule,
)


def excluded_rows(panel):
    rows = {tuple(row) for pool in panel["old_eval_pools"].values() for row in pool}
    rows.update(tuple(q["old_fact"]) for q in panel["mixed_queries"])
    return rows


@torch.no_grad()
def teacher_batch(teacher, rows, metadata, device):
    tokens, positions, _ = arrays(rows, metadata, device)
    return {
        "tokens": tokens,
        "positions": positions,
        "log_probability": teacher(tokens, positions).log_softmax(-1),
    }


def protection(teacher, world, fact, index, original, count, seed, excluded, masks):
    """20 steps of [atoms, chains] batches: `count` neighbours plus original random rows."""
    pools = neighbor_pools(world, fact, *masks)
    rng = np.random.RandomState(seed + index)
    schedule = neighbor_schedule(pools, len(original), count, rng)
    device = next(teacher.parameters()).device
    steps, used = [], {"atoms": set(), "chains": set()}
    for step, picks in enumerate(schedule):
        batches = []
        for slot, kind in enumerate(("atoms", "chains")):
            rows = getattr(world, kind)[picks[kind]]
            if any(tuple(row) in excluded for row in rows.tolist()):
                raise AssertionError("An evaluation row entered the protection set")
            used[kind].update(picks[kind])
            batch = teacher_batch(teacher, rows, world.metadata, device)
            fill = 16 - count
            if fill:
                source = original[step][slot]
                if source["kind"] != kind:
                    raise ValueError("Original teacher order changed")
                for key in ("tokens", "positions", "log_probability"):
                    batch[key] = torch.cat((batch[key], source[key][:fill].to(device)))
            batches.append(batch)
        steps.append(batches)
    sizes = {k: {s: int(len(v)) for s, v in sides.items()} for k, sides in pools.items()}
    return steps, {"pool_sizes": sizes, "distinct_used": {k: len(v) for k, v in used.items()}}


def run(config, arm, out, device):
    item = config["arms"][arm]
    architecture, count = item["architecture"], item["neighbor_per_kind_per_step"]
    base_plan = json.loads((ROOT / config["base_plan"]).read_text())
    reader, world, panel, base, anchor, original, before, _ = parent.initialize(
        base_plan, architecture, device
    )
    teacher = copy.deepcopy(reader).eval()
    settings = settings_for(base_plan, architecture)
    facts = facts_by_id(panel)
    order = list(facts)
    excluded = excluded_rows(panel)
    support = world.support(config["protection"].get("chain_support_phi", 7.2))
    masks = eligibility(world, excluded, support)
    seed = config["protection"]["seed"]
    (out / "records").mkdir(parents=True)
    (out / "states").mkdir()
    run = None
    if config.get("tracking_online", True):
        import wandb

        run = wandb.init(
            entity=config["tracking"]["entity"],
            project=config["tracking"]["project"],
            group=config["batch"],
            name=f"{config['batch']}/{arm}",
            config={**config, "arm": arm},
            dir=str(out),
            settings=wandb.Settings(disable_git=True),
        )
        write_json(out / "wandb.json", {"id": run.id, "url": run.url})
    records, updates, schedules = [], 0, {}
    baseline = {
        "key": "baseline",
        "stream": "baseline",
        "step": 0,
        "written": [],
        "readout": read_state(reader, panel),
    }
    write_json(out / "records/baseline.json", baseline)
    records.append(baseline)
    start = time.time()
    for stream_name, stream in panel["streams"].items():
        restore(reader, base)
        for step, fid in enumerate(stream, 1):
            key = f"{stream_name}-{step:03d}"
            teachers, info_p = protection(
                teacher,
                world,
                facts[fid],
                order.index(fid),
                original,
                count,
                seed,
                excluded,
                masks,
            )
            schedules[fid] = info_p
            info, _ = write_fact(
                reader,
                facts[fid],
                world.metadata,
                settings,
                teachers,
                anchor,
                optimizer_mode="fresh_per_fact",
                constraint_enabled=False,
            )
            updates += info["steps"]
            readout = read_state(reader, panel)
            record = {
                "key": key,
                "stream": stream_name,
                "step": step,
                "written": stream[:step],
                "write_info": info,
                "protection": info_p,
                "readout": readout,
            }
            if step == 64:
                record["two_step"] = parent.two_step(reader, panel, readout["atoms"])
                values = {n: reader.backbone.get_parameter(n).detach().cpu() for n in anchor}
                torch.save({"values": values, "written": stream}, out / "states" / f"{key}.pt")
            write_json(out / "records" / f"{key}.json", record)
            records.append(record)
            written = {tuple(facts[f]) for f in stream[:step]}
            atoms = readout["atoms"]
            metrics = {
                "updates": updates,
                "new_ce_final": info["history"][-1]["new_ce_before"],
                "old_kl_final": float(np.mean(info["history"][-1]["old_KL_before"])),
                "new_atoms_correct": sum(
                    c
                    for row, c in zip(atoms["rows"], atoms["final_correct"], strict=True)
                    if tuple(row) in written
                ),
                "double_new_correct": sum(readout["double_new"]["final_correct"]),
                **{
                    f"old_{p}": float(np.mean(v["final_correct"]))
                    for p, v in readout["old"].items()
                },
            }
            print(key, json.dumps(metrics), round(time.time() - start), flush=True)
            if run:
                run.log(metrics, step=updates)
    restore(reader, base)
    result = summarize_streams(records, panel, base_plan["evaluation"])
    result.update(
        arm=arm,
        architecture=architecture,
        neighbor_per_kind_per_step=count,
        scientific_optimizer_updates=updates,
        learning_rate=settings["learning_rate"],
        constraint_enabled=False,
        protection_schedules=schedules,
        elapsed_seconds=time.time() - start,
    )
    write_json(out / "result.json", result)
    if run:
        run.summary.update(
            {
                "double_new_endpoint": result["primary_endpoint"]["double_new"]["accuracy"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        run.finish()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/neighbor-protection-development-v1.json")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    config["tracking_online"] = not args.offline
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(config, args.arm, args.out / args.arm, args.device)


if __name__ == "__main__":
    main()
