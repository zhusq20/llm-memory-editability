"""Run the frozen double-new continuous CE protocol on one audited native foundation."""

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
import run_alphaedit_discoloop as identity  # noqa: E402
import torch  # noqa: E402
from transformers import GPT2Config  # noqa: E402

from llm_memory_editability.ce_cross_sample import (  # noqa: E402
    atomic_write,
    compare_readouts,
    facts_by_id,
    formal_pairs,
    greedy_rows,
    read_state,
    settings_for,
    validate_panel,
)
from llm_memory_editability.ce_cross_sample_summary import summarize  # noqa: E402
from llm_memory_editability.ce_scope_editing import (  # noqa: E402
    arrays,
    freeze,
    load_edit,
    parameter_names,
    restore,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import model_digest  # noqa: E402
from llm_memory_editability.path_overlap import World  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest  # noqa: E402

sha = identity.digest


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_inputs(plan, architecture):
    arm = plan["architectures"][architecture]
    expected = {
        arm["checkpoint"]: arm["checkpoint_sha256"],
        arm["foundation_gate"]: arm["foundation_gate_sha256"],
        arm["old_teacher_file"]: arm["old_teacher_sha256"],
        plan["model_config"]: plan["model_config_sha256"],
        str(Path(plan["data_dir"]) / "world.npz"): plan["world_sha256"],
        str(Path(plan["data_dir"]) / "complete.json"): plan["metadata_sha256"],
        str(ROOT / plan["panel"]): plan["panel_sha256"],
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"Bound input changed: {path}")
    return expected


def initialize(plan, architecture, device):
    inputs = verify_inputs(plan, architecture)
    arm = plan["architectures"][architecture]
    panel = load_json(ROOT / plan["panel"])
    world = World(plan["data_dir"])
    validate_panel(panel, world)
    gate = load_json(arm["foundation_gate"])
    saved = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
    if (
        not gate["passed"]
        or gate["checkpoint_sha256"] != arm["checkpoint_sha256"]
        or saved["step"] != 300000
        or saved["architecture"]["discoloop"] != (architecture == "discoloop")
        or saved["architecture"]["first_round_kv"]
        or saved["spec"]["unique_layers"] != arm["physical_layers"]
        or saved["spec"]["repeats"] != arm["repeats"]
    ):
        raise ValueError("Use the exact audited native300k checkpoint and architecture")
    config = GPT2Config.from_json_file(plan["model_config"])
    config.n_layer, config.vocab_size, config.use_cache = (
        arm["physical_layers"],
        world.metadata["vocab_size"],
        False,
    )
    config.embd_pdrop = config.resid_pdrop = config.attn_pdrop = 0.0
    model = SupervisedAlignmentGPT(config, arm["repeats"], **saved["architecture"])
    model.load_state_dict(saved["model"], strict=True)
    if model_digest(model) != gate["model_sha256"]:
        raise ValueError("Model tensors differ from the independent foundation audit")
    del saved
    reader = TTCDReader(model.to(device), native_causal_attention=True).eval()
    freeze(reader)
    base = {n: v.detach().clone() for n, v in reader.backbone.state_dict().items()}
    names = parameter_names(reader.backbone, "mlp")
    if sum(base[n].numel() for n in names) != arm["editable_parameters"]:
        raise ValueError("Unexpected unique editable MLP parameter count")
    anchor = {n: base[n] for n in names}
    base_digest = tensor_digest(base)
    teachers = torch.load(arm["old_teacher_file"], map_location=device, weights_only=True)
    if len(teachers) != 20:
        raise ValueError("Exactly the original20 protective teacher batches are required")
    formal_rows = {tuple(row) for rows in panel["old_eval_pools"].values() for row in rows}
    for batches in teachers:
        if len(batches) != 2 or {b["kind"] for b in batches} != {"atoms", "chains"}:
            raise ValueError("Each teacher step needs old atoms and old trained chains")
        for batch in batches:
            if batch["tokens"].shape[0] != 16 or any(
                tuple(row) in formal_rows for row in batch["rows"]
            ):
                raise ValueError("Protection exposure or old formal separation changed")
    for pool in panel["old_eval_pools"].values():
        rows = np.asarray(pool)
        if rows.shape[1] == 3:
            indices = world.lookup[rows[:, 0], rows[:, 1]]
            if (indices < 0).any() or not np.array_equal(world.atoms[indices], rows):
                raise ValueError("An old atomic evaluation truth changed")
        else:
            world.row_facts(rows)
    return reader, world, panel, base, anchor, teachers, base_digest, inputs


def reference_payload(panel, architecture, pair, base_digest, anchor):
    ref = next(
        r for r in panel["reused_A_only_states"][architecture] if r["pair_id"] == pair["pair_id"]
    )
    if sha(ref["record"]) != ref["record_sha256"] or sha(ref["state"]) != ref["state_sha256"]:
        raise ValueError("An audited A-only record or saved state changed")
    payload = torch.load(ref["state"], map_location="cpu", weights_only=True)
    if (
        payload["scope"] != "mlp"
        or payload["foundation_digest"] != base_digest
        or payload["fact"]["novel_fact"] != pair["A"]["fact"]
        or set(payload["values"]) != set(anchor)
    ):
        raise ValueError("Reused A-only state belongs to another foundation, fact, or scope")
    for name, value in payload["values"].items():
        origin = anchor[name].cpu()
        if value.shape != origin.shape or value.dtype != origin.dtype:
            raise ValueError("A-only state shape or precision changed")
        if not torch.all((value >= origin - 0.0005) & (value <= origin + 0.0005)):
            raise ValueError("A-only state violates the original global parameter region")
    return payload, ref


def prepare(plan, architecture, out, model, log):
    reader, world, panel, base, anchor, teachers, before, inputs = model
    with torch.no_grad():
        tokens, positions, _ = arrays(
            world.atoms[:8], world.metadata, next(reader.parameters()).device
        )
        if not torch.equal(reader(tokens, positions), reader.backbone(tokens, positions)):
            raise AssertionError("Native wrapper changed exact foundation logits")
    for i, pair in enumerate(formal_pairs(panel), 1):
        payload, _ = reference_payload(panel, architecture, pair, before, anchor)
        info = payload["write_info"]
        settings = settings_for(plan, architecture)
        for name in ("steps", "learning_rate", "kl_factor", "norm_constraint", "gradient_clip"):
            if info["settings"][name] != settings[name]:
                raise ValueError("Inherited A-only write settings differ")
        del payload
        log(0, {"operation": "verify_reused_A_only", "verified_states": i})
    dev = next(p for p in panel["pairs"] if p["split"] == "development")
    frozen_before = {
        n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
    }
    for side in ("A", "B"):
        atomic_write(
            reader,
            dev[side]["fact"],
            world.metadata,
            settings_for(plan, architecture, 2),
            teachers[:2],
            anchor,
        )
    if frozen_before != {
        n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
    }:
        raise AssertionError("Engineering writes changed a frozen parameter")
    values = {n: reader.backbone.get_parameter(n).detach().cpu().clone() for n in anchor}
    path = out / "engineering-state.pt"
    torch.save({"values": values}, path)
    expected = tensor_digest(reader.backbone.state_dict())
    restore(reader, base)
    load_edit(reader, torch.load(path, map_location="cpu", weights_only=True)["values"], "mlp")
    if tensor_digest(reader.backbone.state_dict()) != expected:
        raise AssertionError("Engineering state does not reload exactly")
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Engineering stage failed to restore the foundation")
    return {
        "passed": True,
        "scientific_optimizer_updates": 0,
        "engineering_optimizer_updates": 4,
        "reused_A_states_verified": 32,
        "exact_native_logits": True,
        "global_anchor_and_frozen_parameters_checked": True,
        "reload_and_restore_exact": True,
        "input_hashes": inputs,
        "foundation_digest": before,
    }


def two_step(reader, panel, atoms):
    eo = panel["metadata"]["entity_offset"]
    result = []
    for i, pair in enumerate(formal_pairs(panel)):
        predicted = atoms["answer"][2 * i] - eo
        item = {
            "pair_id": pair["pair_id"],
            "predicted_bridge": predicted,
            "bridge_correct": predicted == pair["bridge"],
            "strict_correct": False,
        }
        if 0 <= predicted < panel["metadata"]["entities"]:
            readout = greedy_rows(
                reader, [[predicted, pair["B"]["fact"][1], pair["answer"]]], panel["metadata"]
            )
            item["second"] = readout
            item["strict_correct"] = bool(
                atoms["final_correct"][2 * i] and readout["final_correct"][0]
            )
        result.append(item)
    return result


def evaluate(plan, architecture, out, model, log):
    reader, world, panel, base, anchor, teachers, before, inputs = model
    records, updates = [], 0
    (out / "records").mkdir()
    (out / "states").mkdir()
    settings = settings_for(plan, architecture)
    frozen_versions = {
        n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
    }

    def record(
        key,
        stream,
        step,
        written,
        *,
        pair_id=None,
        order=None,
        state_path=None,
        reused=None,
        write_info=None,
    ):
        values = {n: reader.backbone.get_parameter(n).detach().cpu().clone() for n in anchor}
        if state_path is not None:
            torch.save(
                {
                    "values": values,
                    "foundation_digest": before,
                    "key": key,
                    "written": written,
                    "write_info": write_info,
                },
                state_path,
            )
        item = {
            "key": key,
            "stream": stream,
            "step": step,
            "written": list(written),
            "pair_id": pair_id,
            "order": order,
            "parameter_digest": tensor_digest(values),
            "model_digest": tensor_digest(reader.backbone.state_dict()),
            "state": str(state_path.relative_to(out)) if state_path else None,
            "state_sha256": sha(state_path) if state_path else None,
            "reused": reused,
            "write_info": write_info,
        }
        item["readout"] = read_state(reader, panel)
        if stream in panel["streams"] and step == 64:
            item["two_step"] = two_step(reader, panel, item["readout"]["atoms"])
        write_json(out / "records" / f"{key}.json", item)
        records.append(item)
        log(
            updates,
            {
                "operation": "readout",
                "key": key,
                "completed_states": len(records),
                "scientific_optimizer_updates": updates,
                "double_new_target_hits": sum(item["readout"]["double_new"]["final_correct"]),
            },
        )
        return item

    def write(fact_id, key):
        nonlocal updates

        def progress(step, item):
            write_json(
                out / "write-progress.json",
                {
                    "key": key,
                    "fact_id": fact_id,
                    "step_in_fact": step,
                    "scientific_optimizer_updates": updates + step,
                    **item,
                },
            )
            if step == 1 or step % 5 == 0:
                log(
                    updates + step,
                    {
                        "operation": "write",
                        "key": key,
                        "fact_id": fact_id,
                        "step_in_fact": step,
                        "scientific_optimizer_updates": updates + step,
                        "new_ce_before": item["new_ce_before"],
                        "old_KL_mean": float(np.mean(item["old_KL_before"])),
                        "gradient_norm_before_clip": item["gradient_norm_before_clip"],
                        "new_target_tokens": updates + step,
                        "old_protection_examples": (updates + step) * 32,
                    },
                )

        info = atomic_write(
            reader,
            facts_by_id(panel)[fact_id],
            world.metadata,
            settings,
            teachers,
            anchor,
            progress,
        )
        updates += info["steps"]
        if frozen_versions != {
            n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
        }:
            raise AssertionError("A scientific write changed a frozen parameter")
        return info

    record("baseline", "baseline", 0, [])
    for stream_name, stream in panel["streams"].items():
        restore(reader, base)
        frozen_versions = {
            n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
        }
        for step, fact_id in enumerate(stream, 1):
            key = f"{stream_name}-{step:03d}"
            info = write(fact_id, key)
            record(
                key,
                stream_name,
                step,
                stream[:step],
                state_path=out / "states" / f"{key}.pt",
                write_info=info,
            )
    for pair in formal_pairs(panel):
        a, b, pair_id = pair["A"]["fact_id"], pair["B"]["fact_id"], pair["pair_id"]
        restore(reader, base)
        payload, ref = reference_payload(panel, architecture, pair, before, anchor)
        load_edit(reader, payload["values"], "mlp")
        frozen_versions = {
            n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
        }
        record(pair_id + "-A_only", "isolated", 1, [a], pair_id=pair_id, order="A_only", reused=ref)
        info = write(b, pair_id + "-A_then_B")
        record(
            pair_id + "-A_then_B",
            "isolated",
            2,
            [a, b],
            pair_id=pair_id,
            order="A_then_B",
            state_path=out / "states" / (pair_id + "-A_then_B.pt"),
            write_info=info,
        )
        restore(reader, base)
        frozen_versions = {
            n: p._version for n, p in reader.backbone.named_parameters() if n not in anchor
        }
        info = write(b, pair_id + "-B_only")
        record(
            pair_id + "-B_only",
            "isolated",
            1,
            [b],
            pair_id=pair_id,
            order="B_only",
            state_path=out / "states" / (pair_id + "-B_only.pt"),
            write_info=info,
        )
        info = write(a, pair_id + "-B_then_A")
        record(
            pair_id + "-B_then_A",
            "isolated",
            2,
            [b, a],
            pair_id=pair_id,
            order="B_then_A",
            state_path=out / "states" / (pair_id + "-B_then_A.pt"),
            write_info=info,
        )
    if updates != 4480 or len(records) != 257:
        raise AssertionError("The complete frozen scientific matrix or budget differs")
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Foundation did not restore after the complete matrix")
    result = summarize(records, panel, plan["evaluation"])
    result.update(
        passed=True,
        architecture=architecture,
        scientific_optimizer_updates=updates,
        engineering_optimizer_updates=0,
        new_composition_supervision=0,
        new_foundation_updates=0,
        new_writer_outer_updates=0,
        foundation_digest=before,
        input_hashes=inputs,
        saved_new_states=224,
        reused_A_only_states=32,
        record_hashes={p.name: sha(p) for p in sorted((out / "records").glob("*.json"))},
    )
    return result


def audit(plan, architecture, out, root, model, log):
    reader, _, panel, base, anchor, _, before, inputs = model
    science = root / "evaluate"
    expected_result = load_json(science / "result.json")
    records = []
    for i, path in enumerate(sorted((science / "records").glob("*.json")), 1):
        if sha(path) != expected_result["record_hashes"][path.name]:
            raise ValueError("Saved scientific record changed")
        expected = load_json(path)
        restore(reader, base)
        if expected["state"]:
            state = science / expected["state"]
            if sha(state) != expected["state_sha256"]:
                raise ValueError("Saved MLP state changed")
            payload = torch.load(state, map_location="cpu", weights_only=True)
            if payload["foundation_digest"] != before or payload["key"] != expected["key"]:
                raise ValueError("Saved state belongs to another foundation or node")
            load_edit(reader, payload["values"], "mlp")
        elif expected["reused"]:
            pair = next(p for p in formal_pairs(panel) if p["pair_id"] == expected["pair_id"])
            payload, _ = reference_payload(panel, architecture, pair, before, anchor)
            load_edit(reader, payload["values"], "mlp")
        values = {n: reader.backbone.get_parameter(n).detach().cpu() for n in anchor}
        if tensor_digest(values) != expected["parameter_digest"]:
            raise AssertionError("Independent parameter values differ")
        if tensor_digest(reader.backbone.state_dict()) != expected["model_digest"]:
            raise AssertionError("Independent full model digest differs")
        actual = read_state(reader, panel)
        compare_readouts(actual, expected["readout"])
        if "two_step" in expected:
            compare_readouts(two_step(reader, panel, actual["atoms"]), expected["two_step"])
        records.append({**expected, "readout": actual})
        log(
            i,
            {
                "operation": "independent_readout",
                "replayed_states": i,
                "scientific_optimizer_updates": 0,
            },
        )
    if len(records) != 257:
        raise AssertionError("Independent replay did not cover the entire matrix")
    summary = summarize(records, panel, plan["evaluation"])
    for key, value in summary.items():
        compare_readouts(value, expected_result[key])
    write_json(out / "replayed-summary.json", summary)
    restore(reader, base)
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Independent audit failed to restore the foundation")
    return {
        "passed": True,
        "scientific_optimizer_updates": 0,
        "replayed_states": len(records),
        "new_states_replayed": 224,
        "reused_states_replayed": 32,
        "discrete_readouts_exact": True,
        "float_tolerance": 1e-5,
        "summary_recomputed": True,
        "foundation_digest": before,
        "input_hashes": inputs,
        "evaluate_result_sha256": sha(science / "result.json"),
    }


def cloud(root):
    import wandb

    api, receipts = wandb.Api(timeout=60), []
    for stage in ("prepare", "evaluate", "audit"):
        tracking, path = load_json(root / stage / "wandb.json"), root / stage / "result.json"
        expected = load_json(path)
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if (
            run.state != "finished"
            or run.summary.get("result_sha256") != sha(path)
            or run.summary.get("scientific_optimizer_updates")
            != expected["scientific_optimizer_updates"]
        ):
            raise AssertionError("W&B state, update count, or result identity differs")
        actual = run.file("result.json").download(root=str(root / "cloud" / stage), replace=False)
        if sha(actual.name) != sha(path):
            raise AssertionError("Complete W&B JSON differs from local evidence")
        receipts.append({"stage": stage, **tracking, "result_sha256": sha(path)})
    return {
        "passed": True,
        "canonical_full_json_exact": True,
        "runs": receipts,
        "scientific_optimizer_updates": 0,
    }


def main():
    identity.verify_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "evaluate", "audit", "cloud"))
    parser.add_argument(
        "--architecture", choices=("transformer", "loop", "discoloop"), required=True
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan, out = load_json(args.plan), args.root / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "failure.json").exists():
        raise FileExistsError("Never overwrite or automatically repeat a prior stage")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026101612)
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run, started = None, time.monotonic()
        try:
            if args.stage == "cloud":
                result = cloud(args.root)
            else:
                import wandb

                run = wandb.init(
                    entity="zhusq20",
                    project="llm-memory-editability",
                    name=f"{plan['batch']}/{args.architecture}/{args.stage}",
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

                def log(step, record):
                    nonlocal event
                    event += 1
                    write_json(out / "progress.json", {"utc": utc(), "step": step, **record})
                    print(json.dumps({"stage": args.stage, "step": step, **record}), flush=True)
                    run.log({"event": event, **record})

                model = initialize(plan, args.architecture, args.device)
                if args.stage == "prepare":
                    result = prepare(plan, args.architecture, out, model, log)
                elif args.stage == "evaluate":
                    result = evaluate(plan, args.architecture, out, model, log)
                else:
                    result = audit(plan, args.architecture, out, args.root, model, log)
                verify_inputs(plan, args.architecture)
                result["compute"] = dict(model[0].compute)
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            result.update(
                stage=args.stage,
                architecture=args.architecture,
                wall_seconds=time.monotonic() - started,
                source_sha256=sha(ROOT / "source-manifest.json"),
            )
            write_json(out / "result.json", result)
            if run:
                run.summary["result_sha256"] = sha(out / "result.json")
                run.summary["scientific_optimizer_updates"] = result["scientific_optimizer_updates"]
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
