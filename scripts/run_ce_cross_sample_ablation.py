"""Execute and independently reload frozen single-factor continuous-write ablations."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import run_ce_cross_sample as parent  # noqa: E402
import torch  # noqa: E402

from llm_memory_editability.ce_cross_sample import (  # noqa: E402
    atomic_write,
    compare_readouts,
    facts_by_id,
    greedy_rows,
    read_state,
    settings_for,
)
from llm_memory_editability.ce_cross_sample_ablation import (  # noqa: E402
    arm_settings,
    gate_summary,
    make_optimizer,
    optimizer_counts,
    optimizer_snapshot,
    tree_digest,
    write_fact,
)
from llm_memory_editability.ce_cross_sample_ablation_summary import summarize_streams  # noqa: E402
from llm_memory_editability.ce_scope_editing import arrays, load_edit, restore  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.ttcd_editing import tensor_digest  # noqa: E402

sha, load_json = parent.sha, parent.load_json


def values_for(reader, anchor):
    return {n: reader.backbone.get_parameter(n).detach().cpu().clone() for n in anchor}


def frozen_versions(reader, anchor):
    return {n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor}


def base_configuration(plan):
    path = ROOT / plan["base_plan"]
    if sha(path) != plan["base_plan_sha256"]:
        raise ValueError("The original frozen design changed")
    return load_json(path)


def prepare(plan, base_plan, architecture, out, model, log):
    reader, world, panel, base, anchor, before_teachers, before, inputs = model
    teachers = before_teachers
    dev = next(p for p in panel["pairs"] if p["split"] == "development")
    with torch.no_grad():
        tokens, positions, _ = arrays(
            world.atoms[:8], world.metadata, next(reader.parameters()).device
        )
        if not torch.equal(reader(tokens, positions), reader.backbone(tokens, positions)):
            raise AssertionError("The native wrapper changed foundation logits")
    versions = frozen_versions(reader, anchor)
    settings = settings_for(base_plan, architecture)
    old = atomic_write(reader, dev["A"]["fact"], world.metadata, settings, teachers, anchor)
    expected = tensor_digest(reader.backbone.state_dict())
    if frozen_versions(reader, anchor) != versions:
        raise AssertionError("Original writer changed frozen parameters")
    restore(reader, base)
    versions = frozen_versions(reader, anchor)
    new, _ = write_fact(reader, dev["A"]["fact"], world.metadata, settings, teachers, anchor)
    compare_readouts(new["history"], old["history"], tolerance=0)
    if (
        tensor_digest(reader.backbone.state_dict()) != expected
        or frozen_versions(reader, anchor) != versions
    ):
        raise AssertionError("Baseline writer differs from the complete original20-step update")
    log(
        40,
        {"operation": "exact_original_writer_compatibility", "engineering_optimizer_updates": 40},
    )
    restore(reader, base)
    settings = settings_for(base_plan, architecture, 2)
    info, optimizer = write_fact(
        reader,
        dev["A"]["fact"],
        world.metadata,
        settings,
        teachers[:2],
        anchor,
        optimizer_mode="carry_per_stream",
    )
    checkpoint = {
        "values": values_for(reader, anchor),
        "optimizer_state": optimizer_snapshot(optimizer),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
    }
    torch.save(checkpoint, out / "engineering-continuation.pt")
    info_b, optimizer = write_fact(
        reader,
        dev["B"]["fact"],
        world.metadata,
        settings,
        teachers[:2],
        anchor,
        optimizer=optimizer,
        optimizer_mode="carry_per_stream",
    )
    expected_model = tensor_digest(reader.backbone.state_dict())
    expected_optimizer = tree_digest(optimizer.state_dict())
    restore(reader, base)
    checkpoint = torch.load(
        out / "engineering-continuation.pt", map_location="cpu", weights_only=True
    )
    load_edit(reader, checkpoint["values"], "mlp")
    optimizer = make_optimizer(reader, settings)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if tree_digest(optimizer.state_dict()) != info["optimizer_state_digest"]:
        raise AssertionError("Saved Adam state does not load exactly")
    torch.set_rng_state(checkpoint["rng_state"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    replay, optimizer = write_fact(
        reader,
        dev["B"]["fact"],
        world.metadata,
        settings,
        teachers[:2],
        anchor,
        optimizer=optimizer,
        optimizer_mode="carry_per_stream",
    )
    compare_readouts(replay["history"], info_b["history"], tolerance=0)
    if (
        tensor_digest(reader.backbone.state_dict()) != expected_model
        or tree_digest(optimizer.state_dict()) != expected_optimizer
        or set(optimizer_counts(optimizer)) != {4}
    ):
        raise AssertionError("Carried Adam continuation differs after independent disk reload")
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Engineering checks failed to restore theta0")
    log(46, {"operation": "adam_continuation_exact", "engineering_optimizer_updates": 46})
    return {
        "passed": True,
        "scientific_optimizer_updates": 0,
        "development_optimizer_updates": 0,
        "engineering_optimizer_updates": 46,
        "exact_native_logits": True,
        "original20_step_history_and_full_model_exact": True,
        "carried_adam_disk_continuation_exact": True,
        "restored_theta0": True,
        "foundation_digest": before,
        "input_hashes": inputs,
    }


def save_state(out, key, model, written, info, optimizer, endpoint=False):
    reader, _, _, _, anchor, _, before, _ = model
    values = values_for(reader, anchor)
    payload = {
        "values": values,
        "foundation_digest": before,
        "key": key,
        "written": written,
        "write_info": info,
    }
    if endpoint and info["optimizer_mode"] == "carry_per_stream":
        payload["optimizer_state"] = optimizer_snapshot(optimizer)
        payload["optimizer_state_digest"] = tree_digest(payload["optimizer_state"])
        if payload["optimizer_state_digest"] != info["optimizer_state_digest"]:
            raise AssertionError("Adam snapshot differs from live optimizer")
    state = out / "states" / f"{key}.pt"
    torch.save(payload, state)
    return {
        "key": key,
        "written": list(written),
        "parameter_digest": tensor_digest(values),
        "model_digest": tensor_digest(reader.backbone.state_dict()),
        "state": str(state.relative_to(out)),
        "state_sha256": sha(state),
        "write_info": info,
        "reused": None,
        "pair_id": None,
        "order": None,
    }


def checked_write(model, fact, settings, mode, bound, optimizer, log, updates, category, key):
    reader, world, _, _, anchor, teachers, _, _ = model
    versions = frozen_versions(reader, anchor)

    def progress(step, entry):
        if step == 1 or step % 5 == 0:
            log(
                updates + step,
                {
                    "operation": "write",
                    "key": key,
                    "step_in_fact": step,
                    f"{category}_optimizer_updates": updates + step,
                    "new_ce_before": entry["new_ce_before"],
                    "old_KL_mean": float(np.mean(entry["old_KL_before"])),
                    "gradient_norm_before_clip": entry["gradient_norm_before_clip"],
                    "new_target_tokens": updates + step,
                    "old_protection_examples": (updates + step) * 32,
                },
            )

    info, optimizer = write_fact(
        reader,
        fact,
        world.metadata,
        settings,
        teachers,
        anchor,
        optimizer=optimizer if mode == "carry_per_stream" else None,
        optimizer_mode=mode,
        constraint_enabled=bound,
        progress=progress,
    )
    if frozen_versions(reader, anchor) != versions:
        raise AssertionError("A write changed frozen native parameters")
    log(
        updates + info["steps"],
        {
            "operation": "parameter_update",
            "key": key,
            **info["geometry"],
            "adam_counter_before": info["optimizer_counts_before"][0],
            "adam_counter_after": info["optimizer_counts_after"][0],
        },
    )
    return info, optimizer


def gate(plan, base_plan, arm, out, model, log):
    reader, _, panel, base, _, _, before, inputs = model
    dev = [p for p in panel["pairs"] if p["split"] == "development"]
    if len(dev) != 8:
        raise ValueError("Exactly the original8 development pairs are required")
    ids = [p[s]["fact_id"] for s in ("A", "B") for p in dev]
    facts = {p[s]["fact_id"]: p[s]["fact"] for p in dev for s in ("A", "B")}
    streams = {"forward": ids, "reverse": ids[::-1]}
    (out / "records").mkdir()
    (out / "states").mkdir()
    candidates, updates, selected = [], 0, None
    for candidate, lr in enumerate(plan["arms"][arm]["learning_rate_candidates"], 1):
        settings, mode, bound = arm_settings(plan, base_plan, arm, lr)
        records = []
        for stream_name, stream in streams.items():
            restore(reader, base)
            optimizer = None
            for step, fid in enumerate(stream, 1):
                key = f"candidate-{candidate}-{stream_name}-{step:03d}"
                info, optimizer = checked_write(
                    model,
                    facts[fid],
                    settings,
                    mode,
                    bound,
                    optimizer,
                    log,
                    updates,
                    "development",
                    key,
                )
                updates += info["steps"]
                record = save_state(
                    out, key, model, stream[:step], info, optimizer, endpoint=step == 16
                )
                record.update(
                    candidate=candidate,
                    learning_rate=lr,
                    stream=stream_name,
                    step=step,
                    current_fact=facts[fid],
                    readout=greedy_rows(reader, [facts[fid]], panel["metadata"]),
                )
                write_json(out / "records" / f"{key}.json", record)
                records.append(record)
                log(
                    updates,
                    {
                        "operation": "current_fact_readout",
                        "key": key,
                        "current_correct": bool(record["readout"]["final_correct"][0]),
                        "development_optimizer_updates": updates,
                    },
                )
        summary = {
            "candidate": candidate,
            "learning_rate": lr,
            **gate_summary(records, plan["gate"]),
        }
        candidates.append(summary)
        if summary["eligible"]:
            selected = lr
            break
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Development gate failed to restore theta0")
    eligible = selected is not None
    return {
        "passed": True,
        "eligible": eligible,
        "gate_valid": eligible,
        "selected_learning_rate": selected,
        "candidates": candidates,
        "development_optimizer_updates": updates,
        "scientific_optimizer_updates": 0,
        "engineering_optimizer_updates": 0,
        "input_hashes": inputs,
        "foundation_digest": before,
        "record_hashes": {p.name: sha(p) for p in sorted((out / "records").glob("*.json"))},
    }


def science(plan, base_plan, arm, out, root, model, log):
    gate_result = load_json(root / "gate/result.json")
    audit_result = load_json(root / "audit-gate/result.json")
    if (
        not gate_result["eligible"]
        or not audit_result["passed"]
        or audit_result["gate_result_sha256"] != sha(root / "gate/result.json")
    ):
        raise ValueError("Formal streams require a valid independently reloaded development gate")
    settings, mode, bound = arm_settings(
        plan, base_plan, arm, gate_result["selected_learning_rate"]
    )
    reader, _, panel, base, anchor, _, before, inputs = model
    (out / "records").mkdir()
    (out / "states").mkdir()
    records, updates = [], 0
    baseline = {
        "key": "baseline",
        "stream": "baseline",
        "step": 0,
        "written": [],
        "state": None,
        "state_sha256": None,
        "reused": None,
        "write_info": None,
        "pair_id": None,
        "order": None,
        "parameter_digest": tensor_digest(values_for(reader, anchor)),
        "model_digest": before,
        "readout": read_state(reader, panel),
    }
    write_json(out / "records/baseline.json", baseline)
    records.append(baseline)
    for stream_name, stream in panel["streams"].items():
        restore(reader, base)
        optimizer = None
        for step, fid in enumerate(stream, 1):
            key = f"{stream_name}-{step:03d}"
            info, optimizer = checked_write(
                model,
                facts_by_id(panel)[fid],
                settings,
                mode,
                bound,
                optimizer,
                log,
                updates,
                "scientific",
                key,
            )
            updates += info["steps"]
            expected_counter = (step - 1) * 20 if mode == "carry_per_stream" else 0
            if set(info["optimizer_counts_before"]) != {expected_counter}:
                raise AssertionError("Adam lifecycle differs from the assigned single factor")
            record = save_state(
                out, key, model, stream[:step], info, optimizer, endpoint=step == 64
            )
            record.update(stream=stream_name, step=step, readout=read_state(reader, panel))
            if step == 64:
                record["two_step"] = parent.two_step(reader, panel, record["readout"]["atoms"])
            write_json(out / "records" / f"{key}.json", record)
            records.append(record)
            log(
                updates,
                {
                    "operation": "readout",
                    "key": key,
                    "completed_states": len(records),
                    "scientific_optimizer_updates": updates,
                    "double_new_target_hits": sum(record["readout"]["double_new"]["final_correct"]),
                },
            )
    if updates != 2560 or len(records) != 129:
        raise AssertionError("Formal continuous-only budget changed")
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Formal streams failed to restore theta0")
    result = summarize_streams(records, panel, plan["evaluation"])
    result.update(
        passed=True,
        gate_valid=True,
        selected_learning_rate=settings["learning_rate"],
        scientific_optimizer_updates=updates,
        development_optimizer_updates=0,
        engineering_optimizer_updates=0,
        new_composition_supervision=0,
        new_foundation_updates=0,
        new_writer_outer_updates=0,
        saved_new_states=128,
        foundation_digest=before,
        input_hashes=inputs,
        gate_result_sha256=sha(root / "gate/result.json"),
        record_hashes={p.name: sha(p) for p in sorted((out / "records").glob("*.json"))},
    )
    return result


def audit(plan, base_plan, arm, out, root, model, log, *, development):
    reader, _, panel, base, anchor, _, before, inputs = model
    stage = "gate" if development else "science"
    source = root / stage
    expected_result = load_json(source / "result.json")
    records, endpoint_optimizers = [], 0
    paths = sorted((source / "records").glob("*.json"))
    if set(expected_result["record_hashes"]) != {p.name for p in paths}:
        raise ValueError("Saved record coverage changed")
    for i, path in enumerate(paths, 1):
        if sha(path) != expected_result["record_hashes"][path.name]:
            raise ValueError("Saved record bytes changed")
        expected = load_json(path)
        restore(reader, base)
        if expected["state"]:
            state = source / expected["state"]
            if sha(state) != expected["state_sha256"]:
                raise ValueError("Saved MLP state bytes changed")
            payload = torch.load(state, map_location="cpu", weights_only=True)
            if (
                payload["foundation_digest"] != before
                or payload["key"] != expected["key"]
                or payload["written"] != expected["written"]
            ):
                raise ValueError("State foundation, prefix or identity changed")
            load_edit(reader, payload["values"], "mlp")
            info = expected["write_info"]
            lr = (
                expected["learning_rate"]
                if development
                else expected_result["selected_learning_rate"]
            )
            settings, mode, bound = arm_settings(plan, base_plan, arm, lr)
            compare_readouts(info["settings"], json.loads(json.dumps(settings)))
            if info["optimizer_mode"] != mode or info["constraint_enabled"] != bound:
                raise AssertionError("Recorded intervention settings differ")
            expected_before = (expected["step"] - 1) * 20 if mode == "carry_per_stream" else 0
            if set(info["optimizer_counts_before"]) != {expected_before} or set(
                info["optimizer_counts_after"]
            ) != {expected_before + 20}:
                raise AssertionError("Saved Adam prefix counters differ")
            if "optimizer_state" in payload:
                optimizer = make_optimizer(reader, settings)
                optimizer.load_state_dict(payload["optimizer_state"])
                if (
                    tree_digest(optimizer.state_dict()) != info["optimizer_state_digest"]
                    or payload["optimizer_state_digest"] != info["optimizer_state_digest"]
                    or set(optimizer_counts(optimizer)) != {expected_before + 20}
                ):
                    raise AssertionError("Independent endpoint Adam reload differs")
                endpoint_optimizers += 1
        if (
            tensor_digest(values_for(reader, anchor)) != expected["parameter_digest"]
            or tensor_digest(reader.backbone.state_dict()) != expected["model_digest"]
        ):
            raise AssertionError("Independent full model or MLP values differ")
        actual = (
            greedy_rows(reader, [expected["current_fact"]], panel["metadata"])
            if development
            else read_state(reader, panel)
        )
        compare_readouts(actual, expected["readout"])
        if "two_step" in expected:
            compare_readouts(parent.two_step(reader, panel, actual["atoms"]), expected["two_step"])
        records.append({**expected, "readout": actual})
        log(
            i,
            {
                "operation": "independent_readout",
                "replayed_states": i,
                "scientific_optimizer_updates": 0,
                "development_optimizer_updates": 0,
            },
        )
    if development:
        if len(records) != 32 * len(expected_result["candidates"]):
            raise AssertionError("Gate audit omitted a candidate or state")
        for candidate in expected_result["candidates"]:
            summary = gate_summary(
                [r for r in records if r["candidate"] == candidate["candidate"]], plan["gate"]
            )
            for key, value in summary.items():
                compare_readouts(value, candidate[key])
    else:
        if len(records) != 129:
            raise AssertionError("Formal audit omitted a state")
        summary = summarize_streams(records, panel, plan["evaluation"])
        for key, value in summary.items():
            compare_readouts(value, expected_result[key])
        write_json(out / "replayed-summary.json", summary)
    expected_endpoints = 2 * len(expected_result["candidates"]) if development else 2
    if (
        plan["arms"][arm]["optimizer_mode"] == "carry_per_stream"
        and endpoint_optimizers != expected_endpoints
    ):
        raise AssertionError("Persistent Adam endpoint audit coverage is incomplete")
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Independent audit did not restore theta0")
    return {
        "passed": True,
        "gate_valid": expected_result["gate_valid"],
        "scientific_optimizer_updates": 0,
        "development_optimizer_updates": 0,
        "engineering_optimizer_updates": 0,
        "replayed_states": len(records),
        "reloaded_endpoint_optimizers": endpoint_optimizers,
        "discrete_readouts_exact": True,
        "float_tolerance": 1e-5,
        "summary_recomputed": True,
        "foundation_digest": before,
        "input_hashes": inputs,
        f"{stage}_result_sha256": sha(source / "result.json"),
    }


def cloud(plan, arm, root):
    import wandb

    architecture = plan["arms"][arm]["architecture"]
    targets = [("prepare", root.parents[1] / f"{architecture}-prepare/output/prepare")]
    gate_result = load_json(root / "gate/result.json")
    targets.extend(
        (stage, root / stage)
        for stage in (
            ("gate", "audit-gate", "science", "audit")
            if gate_result["eligible"]
            else ("gate", "audit-gate")
        )
    )
    api, receipts = wandb.Api(timeout=60), []
    for stage, directory in targets:
        tracking, path = load_json(directory / "wandb.json"), directory / "result.json"
        expected = load_json(path)
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if run.state != "finished" or run.summary.get("result_sha256") != sha(path):
            raise AssertionError("W&B endpoint state or canonical JSON identity differs")
        for key in (
            "scientific_optimizer_updates",
            "development_optimizer_updates",
            "engineering_optimizer_updates",
        ):
            if run.summary.get(key) != expected[key]:
                raise AssertionError("W&B optimizer count differs")
        downloaded = run.file("result.json").download(
            root=str(root / "cloud" / stage), replace=False
        )
        if sha(downloaded.name) != sha(path):
            raise AssertionError("Downloaded complete W&B result differs")
        receipts.append(
            {
                "stage": stage,
                **tracking,
                "result_path": str(path),
                "result_sha256": sha(path),
                "state": run.state,
            }
        )
    return {
        "passed": True,
        "gate_valid": gate_result["eligible"],
        "canonical_full_json_exact": True,
        "runs": receipts,
        "scientific_optimizer_updates": 0,
        "development_optimizer_updates": 0,
        "engineering_optimizer_updates": 0,
    }


def main():
    parent.identity.verify_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "gate", "audit-gate", "science", "audit", "cloud")
    )
    parser.add_argument("--architecture", choices=("loop", "discoloop"))
    parser.add_argument("--arm")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan, out = load_json(args.plan), args.root / args.stage
    base_plan = base_configuration(plan)
    architecture = (
        args.architecture if args.stage == "prepare" else plan["arms"][args.arm]["architecture"]
    )
    if architecture is None or (args.stage == "prepare" and args.arm is not None):
        raise ValueError("Prepare requires architecture; other stages require an arm")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "failure.json").exists():
        raise FileExistsError("Never overwrite or automatically repeat a completed/failed stage")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026101612)
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run, started = None, time.monotonic()
        try:
            if args.stage == "cloud":
                result = cloud(plan, args.arm, args.root)
            else:
                import wandb

                run = wandb.init(
                    entity="zhusq20",
                    project="llm-memory-editability",
                    name=f"{plan['batch']}/{args.arm or architecture}/{args.stage}",
                    group=plan["batch"],
                    job_type=args.stage,
                    config=plan,
                    dir=str(out),
                    mode="online",
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / "wandb.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)
                event = 0

                def log(step, entry):
                    nonlocal event
                    event += 1
                    record = {"utc": utc(), "step": step, **entry}
                    write_json(out / "progress.json", record)
                    print(json.dumps({"stage": args.stage, **record}), flush=True)
                    run.log({"event": event, **entry})

                model = parent.initialize(base_plan, architecture, args.device)
                if args.stage == "prepare":
                    result = prepare(plan, base_plan, architecture, out, model, log)
                elif args.stage == "gate":
                    result = gate(plan, base_plan, args.arm, out, model, log)
                elif args.stage == "science":
                    result = science(plan, base_plan, args.arm, out, args.root, model, log)
                else:
                    result = audit(
                        plan,
                        base_plan,
                        args.arm,
                        out,
                        args.root,
                        model,
                        log,
                        development=args.stage == "audit-gate",
                    )
                parent.verify_inputs(base_plan, architecture)
                result["compute"] = dict(model[0].compute)
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            result.update(
                stage=args.stage,
                arm=args.arm,
                architecture=architecture,
                wall_seconds=time.monotonic() - started,
                source_sha256=sha(ROOT / "source-manifest.json"),
            )
            write_json(out / "result.json", result)
            if run:
                run.summary["result_sha256"] = sha(out / "result.json")
                for key in (
                    "scientific_optimizer_updates",
                    "development_optimizer_updates",
                    "engineering_optimizer_updates",
                ):
                    run.summary[key] = result[key]
                run.save(str(out / "result.json"), base_path=str(out), policy="now")
                run.finish()
            print(json.dumps({"complete": True, "stage": args.stage}), flush=True)
        except BaseException as error:
            write_json(
                out / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()}
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
