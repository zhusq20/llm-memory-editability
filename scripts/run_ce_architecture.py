"""Entire-MLP CE on two audited native foundations, paired to saved DiscoLoop edits."""

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

import run_alphaedit_discoloop as alpha  # noqa: E402
import run_ce_scope_discoloop as ce  # noqa: E402
import torch  # noqa: E402
from transformers import GPT2Config  # noqa: E402

from llm_memory_editability.ce_architecture import (  # noqa: E402
    inspect_endpoints,
    old_retention_rates,
    summarize_endpoints,
)
from llm_memory_editability.ce_scope_editing import (  # noqa: E402
    arrays,
    atomic_ce_edit,
    choose_recipe,
    load_edit,
    parameter_names,
    restore,
    teacher_batches,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import model_digest  # noqa: E402
from llm_memory_editability.path_overlap import World  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest  # noqa: E402

readouts = alpha.readouts
readouts.inspect_rows = inspect_endpoints
snapshot, verify_base, load_protection = ce.snapshot, ce.verify_base, ce.load_protection


def arm_plan(common, arm):
    if arm not in ("loop", "transformer") or common["scopes"] != ["mlp"]:
        raise ValueError("Only the two authorized entire-MLP controls may run")
    plan = {**common, **common["architectures"][arm], "architecture_name": arm}
    plan["bound_inputs"] = dict(common["bound_inputs"])
    plan["bound_inputs"][plan["checkpoint"]] = plan["checkpoint_sha256"]
    gate = plan["foundation_gate"]
    plan["bound_inputs"][gate] = common["architecture_input_hashes"][arm][gate]
    return plan


def initialize(plan, device):
    for path, expected in plan["bound_inputs"].items():
        if alpha.digest(path) != expected:
            raise ValueError(f"Immutable input changed: {path}")
    if alpha.digest(ROOT / plan["panel"]) != plan["panel_sha256"]:
        raise ValueError("Novel panel changed")
    panel = json.loads((ROOT / plan["panel"]).read_text())
    world = World(plan["data_dir"])
    gate = json.loads(Path(plan["foundation_gate"]).read_text())
    if not gate["passed"] or gate["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Independent foundation audit must bind the exact checkpoint")
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
    if (
        saved["step"] != 300000
        or saved["architecture"]["discoloop"]
        or saved["architecture"]["first_round_kv"]
        or saved["spec"]["unique_layers"] != plan["expected_layers"]
        or saved["spec"]["repeats"] != plan["expected_repeats"]
    ):
        raise ValueError("Expected audited native300k plain architecture without TTCD")
    hf = GPT2Config.from_json_file(plan["model_config"])
    hf.n_layer, hf.vocab_size, hf.use_cache = (
        plan["expected_layers"],
        world.metadata["vocab_size"],
        False,
    )
    hf.embd_pdrop = hf.resid_pdrop = hf.attn_pdrop = 0.0
    model = SupervisedAlignmentGPT(hf, plan["expected_repeats"], **saved["architecture"])
    model.load_state_dict(saved["model"], strict=True)
    if model_digest(model) != gate["model_sha256"]:
        raise AssertionError("Native foundation model digest differs")
    if sum(p.numel() for p in model.parameters()) != plan["expected_parameters"]:
        raise AssertionError("Native parameter count differs")
    reader = TTCDReader(model.to(device), native_causal_attention=True).eval()
    if len({tuple(f["novel_fact"][:2]) for f in panel["facts"]}) != 40:
        raise ValueError("All development/formal addition keys must be distinct")
    for fact in panel["facts"]:
        readouts.validate_fact(fact, world)
    with torch.no_grad():
        for rows in (
            world.atoms[panel["U_atoms"][:8]],
            world.chains[panel["U_chains"][:8]],
        ):
            tokens, positions, _ = arrays(rows, world.metadata, device)
            if not torch.equal(
                reader(tokens, positions, fast=reader.empty_fast()),
                model(tokens, positions),
            ):
                raise AssertionError("Zero-increment wrapper changed native logits")
    return reader, world, panel


def prepare(plan, out, reader, world, panel, log):
    protected = torch.load(plan["alpha_protection"], map_location="cpu", weights_only=True)
    split = protected["split"]
    if split != alpha.old_split(world, panel, plan["protection"]):
        raise AssertionError("Protection/development/formal split differs from DiscoLoop")
    write_json(out / "old-sample-split.json", split)
    teachers = teacher_batches(reader, world, split, plan["ce"])
    torch.save(teachers, out / "old-teachers.pt")
    original = torch.load(plan["discoloop_prepare_teachers"], map_location="cpu", weights_only=True)
    for left, right in zip(teachers, original, strict=True):
        for a, b in zip(left, right, strict=True):
            for key in ("kind", "indices", "rows", "target_slots"):
                if a[key] != b[key]:
                    raise AssertionError("Old protective exposure differs from DiscoLoop")
            for key in ("tokens", "positions"):
                if not torch.equal(a[key], b[key]):
                    raise AssertionError("Old protective inputs differ from DiscoLoop")
    base, before = snapshot(reader), tensor_digest(reader.backbone.state_dict())
    first = next(f["novel_fact"] for f in panel["facts"] if f["split"] == "development")
    settings = {**plan["ce"], "steps": 2, "learning_rate": plan["learning_rates"][0]}
    names = parameter_names(reader.backbone, "mlp")
    values, info = atomic_ce_edit(reader, first, world.metadata, "mlp", settings, teachers[:2])
    for name, value in reader.backbone.named_parameters():
        if name not in names and not torch.equal(value, base[name]):
            raise AssertionError("Full-size preflight changed a frozen parameter")
    if not any(not torch.equal(reader.backbone.get_parameter(n), base[n]) for n in names):
        raise AssertionError("Full-size preflight did not perform an actual edit")
    saved = tensor_digest(reader.backbone.state_dict())
    restore(reader, base)
    load_edit(reader, values, "mlp")
    if tensor_digest(reader.backbone.state_dict()) != saved:
        raise AssertionError("Full-size saved parameters do not reload exactly")
    restore(reader, base)
    verify_base(reader, before)
    log(2, {"operation": "full_size_mlp_preflight", "restore_exact": True})
    return {
        "passed": True,
        "engineering_optimizer_updates": 2,
        "scientific_optimizer_updates": 0,
        "new_composition_supervision": 0,
        "trainable_parameters": info["trainable_parameters"],
        "native_model_parameters": plan["expected_parameters"],
        "native_metrics": plan["native_metrics"],
        "old_teacher_sha256": alpha.digest(out / "old-teachers.pt"),
        "old_split_sha256": alpha.digest(out / "old-sample-split.json"),
        "old_inputs_exact_to_discoloop": True,
        "own_model_teacher": True,
        "frozen_unchanged": True,
        "reload_exact": True,
        "restore_exact": True,
    }


def develop(plan, out, root, scope, reader, world, panel, log):
    split, teachers = load_protection(root, reader)
    base, before = snapshot(reader), tensor_digest(reader.backbone.state_dict())
    baseline = alpha.old_pools(reader, None, split["development"], world)
    facts = [f for f in panel["facts"] if f["split"] == "development"]
    candidates, updates, reused_updates = [], 0, 0
    for learning_rate in plan["learning_rates"]:
        recovered = plan.get("development_recovery", {}).get(scope, {}).get(str(learning_rate))
        if recovered:
            path = Path(recovered["path"])
            if alpha.digest(path) != recovered["sha256"]:
                raise AssertionError("Completed development candidate changed")
            saved = json.loads(path.read_text())
            if [r["case_id"] for r in saved["facts"]] != [f["case_id"] for f in facts]:
                raise AssertionError("Recovered development panel differs")
            expected_settings = {**plan["ce"], "learning_rate": learning_rate}
            if any(
                r["write"]["settings"] != expected_settings
                or r["write"]["steps"] != plan["ce"]["steps"]
                for r in saved["facts"]
            ):
                raise AssertionError("Recovered candidate used different optimization")
            candidate = saved["selection"]
            if candidate["learning_rate"] != learning_rate:
                raise AssertionError("Recovered learning rate differs")
            candidates.append(candidate)
            reused_updates += sum(r["write"]["steps"] for r in saved["facts"])
            write_json(out / f"candidate-{learning_rate}.json", saved)
            log(
                updates,
                {
                    "scope": scope,
                    "operation": "reuse_complete_candidate",
                    "reused_optimizer_updates": reused_updates,
                    **candidate,
                },
            )
            continue
        records, correct = [], 0
        totals = {name: {"retained": 0, "originally_correct": 0} for name in alpha.POOLS}
        for fact in facts:
            restore(reader, base)
            settings = {**plan["ce"], "learning_rate": learning_rate}
            _, info = atomic_ce_edit(
                reader, fact["novel_fact"], world.metadata, scope, settings, teachers
            )
            atom = readouts.greedy_rows(reader, None, [fact["novel_fact"]], world.metadata)
            after = alpha.old_pools(reader, None, split["development"], world)
            details = {name: alpha.retention(baseline[name], after[name]) for name in alpha.POOLS}
            for name in alpha.POOLS:
                for key in ("retained", "originally_correct"):
                    totals[name][key] += details[name][key]
            correct += sum(atom["final_correct"])
            updates += info["steps"]
            records.append(
                {
                    "case_id": fact["case_id"],
                    "atomic": atom,
                    "write": info,
                    "old_retention": details,
                }
            )
            write_json(
                out / f"development-fact-{learning_rate}-{fact['case_id']}.json",
                records[-1],
            )
            log(
                updates,
                {
                    "scope": scope,
                    "learning_rate": learning_rate,
                    "development_cases": len(records),
                    "atomic_correct": correct,
                },
            )
        rates, minimum = old_retention_rates(totals)
        candidate = {
            "learning_rate": learning_rate,
            "atomic_accuracy": correct / len(facts),
            "minimum_old_retention": minimum,
            "retention_by_pool": rates,
        }
        candidates.append(candidate)
        write_json(
            out / f"candidate-{learning_rate}.json",
            {"selection": candidate, "facts": records},
        )
        log(updates, {"scope": scope, "operation": "candidate_complete", **candidate})
    restore(reader, base)
    verify_base(reader, before)
    return {
        "passed": True,
        "scope": scope,
        "candidates": candidates,
        "selected": choose_recipe(candidates, plan["minimum_development_retention"]),
        "scientific_optimizer_updates": updates,
        "reused_optimizer_updates": reused_updates,
        "scientific_unique_optimizer_updates": updates + reused_updates,
        "new_composition_supervision": 0,
        "new_composition_queries": 0,
        "independent_foundation_restored": True,
    }


def evaluate(plan, out, root, scope, reader, world, panel, log, *, audit=False):
    development = json.loads((root / f"development-{scope}/result.json").read_text())
    split, teachers = load_protection(root, reader)
    base, before = snapshot(reader), tensor_digest(reader.backbone.state_dict())
    settings = {**plan["ce"], "learning_rate": development["selected"]["learning_rate"]}
    facts = [f for f in panel["facts"] if f["split"] == "evaluation"]
    baseline_old = alpha.old_pools(reader, None, split["formal"], world)
    records, updates, state_hashes = [], 0, {}
    evaluation = root / f"evaluation-{scope}"
    if not audit:
        (out / "states").mkdir(exist_ok=False)
    for fact in facts:
        restore(reader, base)
        case = fact["case_id"]
        baseline_fast = reader.empty_fast()
        raw_before = readouts.evaluate_state(reader, baseline_fast, fact, world, panel)
        raw_before["extended_old"] = baseline_old
        inherited = json.loads((Path(plan["alpha_evaluation"]) / f"{case}.json").read_text())
        if inherited["fact"] != fact:
            raise AssertionError("Architecture comparison changed the fixed fact panel")
        for name, pool in baseline_old.items():
            if pool["rows"] != inherited["before"]["extended_old"][name]["rows"]:
                raise AssertionError("Architecture comparison changed a formal old pool")
        state_path = evaluation / "states" / f"{case}.pt"
        if audit:
            expected_record = json.loads((evaluation / f"{case}.json").read_text())
            if alpha.digest(state_path) != expected_record["state_sha256"]:
                raise AssertionError("Saved formal edit changed")
            saved = torch.load(state_path, map_location="cpu", weights_only=True)
            if (
                saved["fact"] != fact
                or saved["scope"] != scope
                or saved["foundation_digest"] != before
            ):
                raise AssertionError("Saved edit belongs to another foundation/fact/scope")
            values, info = saved["values"], saved["write_info"]
            if info["settings"] != settings:
                raise AssertionError("Saved edit did not use the frozen developed recipe")
            load_edit(reader, values, scope)
        else:
            values, info = atomic_ce_edit(
                reader, fact["novel_fact"], world.metadata, scope, settings, teachers
            )
            for name, parameter in reader.backbone.named_parameters():
                if name not in values and not torch.equal(parameter, base[name]):
                    raise AssertionError("Formal CE changed a frozen parameter")
            torch.save(
                {
                    "values": values,
                    "fact": fact,
                    "scope": scope,
                    "write_info": info,
                    "foundation_digest": before,
                },
                state_path,
            )
            updates += info["steps"]
        edited_digest = tensor_digest(reader.backbone.state_dict())
        after = readouts.evaluate_state(reader, reader.empty_fast(), fact, world, panel)
        after["extended_old"] = alpha.old_pools(reader, None, split["formal"], world)
        if tensor_digest(reader.backbone.state_dict()) != edited_digest:
            raise AssertionError("Evaluation changed edited parameters")
        record = alpha.normalize(
            {
                "fact": fact,
                "scope": scope,
                "write_info": info,
                "before": raw_before,
                "after": after,
                "parameter_digest": tensor_digest(values),
                "edited_model_digest": edited_digest,
                "state_sha256": alpha.digest(state_path),
            }
        )
        if audit and record != expected_record:
            raise AssertionError("Fresh-container saved-state readout differs")
        write_json(out / f"{case}.json", record)
        state_hashes[case] = record["state_sha256"]
        records.append(record)
        log(
            updates if not audit else len(records),
            {
                "scope": scope,
                "completed_facts": len(records),
                "scientific_optimizer_updates": updates,
                "atomic_correct": sum(after["novel_atom"]["final_correct"]),
                "first_hop_correct": sum(after["first_hop"]["final_correct"]),
                "second_hop_correct": sum(after["second_hop"]["final_correct"]),
            },
        )
    restore(reader, base)
    verify_base(reader, before)
    result = summarize_endpoints(records, plan["statistics"])
    result["extended_retention"] = {
        name: alpha.retention_summary(records, name) for name in alpha.POOLS
    }
    result.update(
        passed=True,
        scope=scope,
        selected_learning_rate=settings["learning_rate"],
        development_retention_threshold_met=development["selected"]["retention_threshold_met"],
        scientific_optimizer_updates=updates,
        new_composition_supervision=0,
        base_pretraining_updates=0,
        independent_foundation_restored=True,
        independent_saved_state_replay=audit,
        state_sha256=state_hashes,
        trainable_parameters=records[0]["write_info"]["trainable_parameters"],
        exposures_per_fact={
            k: records[0]["write_info"][k]
            for k in (
                "new_atomic_exposures",
                "new_supervised_tokens",
                "old_atomic_exposures",
                "old_composite_exposures",
                "old_distillation_slots",
            )
        },
    )
    return result


def cloud(root):
    import wandb

    api, receipts = wandb.Api(timeout=60), []
    for stage in ("prepare", "development-mlp", "evaluation-mlp", "audit-mlp"):
        tracking = json.loads((root / stage / "wandb.json").read_text())
        expected = root / stage / "result.json"
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if run.state != "finished" or run.summary.get("result_sha256") != alpha.digest(expected):
            raise AssertionError("Cloud run state or result identity differs")
        actual = run.file("result.json").download(root=str(root / "cloud" / stage), replace=False)
        if alpha.digest(actual.name) != alpha.digest(expected):
            raise AssertionError("Downloaded full cloud JSON differs")
        if (
            run.summary.get("scientific_optimizer_updates")
            != json.loads(expected.read_text())["scientific_optimizer_updates"]
        ):
            raise AssertionError("Cloud optimizer update count differs")
        receipts.append({"stage": stage, **tracking, "result_sha256": alpha.digest(expected)})
    return {
        "passed": True,
        "canonical_full_json_exact": True,
        "runs": receipts,
        "scientific_optimizer_updates": 0,
    }


def main():
    alpha.verify_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "development", "evaluation", "audit", "cloud"))
    parser.add_argument("--architecture", choices=("loop", "transformer"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = arm_plan(json.loads(args.plan.read_text()), args.architecture)
    stage = args.stage + ("-mlp" if args.stage in ("development", "evaluation", "audit") else "")
    out = args.root / stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "failure.json").exists():
        raise FileExistsError("Preserve previous stage evidence; no automatic retry")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan["seed"])
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(plan["resources"]["torch_allocator_fraction"])
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
                    name=f"{plan['batch']}/{args.architecture}/{stage}",
                    group=plan["batch"],
                    job_type=args.stage,
                    config=plan,
                    dir=str(out),
                    mode="online",
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / "wandb.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)
                reader, world, panel = initialize(plan, args.device)
                before = tensor_digest(reader.backbone.state_dict())

                def log(step, record):
                    write_json(out / "progress.json", {"utc": utc(), "step": step, **record})
                    print(json.dumps({"stage": stage, "step": step, **record}), flush=True)
                    run.log(record, step=step)

                if args.stage == "prepare":
                    result = prepare(plan, out, reader, world, panel, log)
                elif args.stage == "development":
                    result = develop(plan, out, args.root, "mlp", reader, world, panel, log)
                else:
                    result = evaluate(
                        plan,
                        out,
                        args.root,
                        "mlp",
                        reader,
                        world,
                        panel,
                        log,
                        audit=args.stage == "audit",
                    )
                verify_base(reader, before)
                for path, expected in plan["bound_inputs"].items():
                    if alpha.digest(path) != expected:
                        raise AssertionError("Immutable input changed")
                result.update(
                    foundation_digest=before,
                    input_hashes=plan["bound_inputs"],
                    compute=dict(reader.compute),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    native_metrics=plan["native_metrics"],
                    architecture=args.architecture,
                )
            result.update(
                stage=stage,
                wall_seconds=time.monotonic() - started,
                source_sha256=alpha.digest(ROOT / "source-manifest.json"),
            )
            write_json(out / "result.json", result)
            if run:
                run.summary["result_sha256"] = alpha.digest(out / "result.json")
                run.summary["scientific_optimizer_updates"] = result["scientific_optimizer_updates"]
                run.save(str(out / "result.json"), base_path=str(out), policy="now")
                run.finish()
            print(json.dumps({"complete": True, "stage": stage}), flush=True)
        except BaseException as error:
            write_json(
                out / "failure.json",
                {"error": repr(error), "traceback": traceback.format_exc()},
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
