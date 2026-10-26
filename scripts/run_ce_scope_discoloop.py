"""CE parameter-scope controls paired to the completed AlphaEdit novel-fact panel."""

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
import run_alphaedit_discoloop as alpha  # noqa: E402
import torch  # noqa: E402

from llm_memory_editability.alphaedit_summary import bootstrap, rate, summarize  # noqa: E402
from llm_memory_editability.ce_scope_editing import (  # noqa: E402
    SCOPES,
    arrays,
    atomic_ce_edit,
    choose_recipe,
    load_edit,
    parameter_names,
    restore,
    teacher_batches,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.ttcd_editing import tensor_digest  # noqa: E402

readouts = alpha.readouts


def snapshot(reader):
    return {k: v.detach().clone() for k, v in reader.backbone.state_dict().items()}


def verify_base(reader, expected):
    if tensor_digest(reader.backbone.state_dict()) != expected:
        raise AssertionError("Independent edit did not restore the exact foundation")


def prepare(plan, out, reader, world, panel, log):
    protected = torch.load(plan["alpha_protection"], map_location="cpu", weights_only=True)
    split = protected["split"]
    if split != alpha.old_split(world, panel, plan["protection"]):
        raise AssertionError("Old protection/development/formal pools differ from AlphaEdit")
    write_json(out / "old-sample-split.json", split)
    teachers = teacher_batches(reader, world, split, plan["ce"])
    torch.save(teachers, out / "old-teachers.pt")
    base, before = snapshot(reader), tensor_digest(reader.backbone.state_dict())
    first = next(f["novel_fact"] for f in panel["facts"] if f["split"] == "development")
    smoke = []
    for scope in SCOPES:
        settings = {
            **plan["ce"],
            "steps": 2,
            "learning_rate": plan["learning_rates"][0],
        }
        names = parameter_names(reader.backbone, scope)
        values, info = atomic_ce_edit(reader, first, world.metadata, scope, settings, teachers[:2])
        for name, value in reader.backbone.named_parameters():
            if name not in names and not torch.equal(value, base[name]):
                raise AssertionError("CUDA smoke changed a frozen parameter")
        if not any(not torch.equal(reader.backbone.get_parameter(n), base[n]) for n in names):
            raise AssertionError("CUDA smoke produced no actual parameter edit")
        x, p, _ = arrays([first], world.metadata, next(reader.parameters()).device)
        with torch.no_grad():
            if not torch.equal(reader(x, p), reader(x, p, fast=reader.empty_fast())):
                raise AssertionError("Materialized edit and evaluation wrapper differ")
        saved_digest = tensor_digest(reader.backbone.state_dict())
        restore(reader, base)
        load_edit(reader, values, scope)
        if tensor_digest(reader.backbone.state_dict()) != saved_digest:
            raise AssertionError("CUDA smoke saved parameters do not reload exactly")
        restore(reader, base)
        verify_base(reader, before)
        smoke.append(
            {
                "scope": scope,
                "trainable_parameters": info["trainable_parameters"],
                "steps": info["steps"],
                "max_change": info["maximum_absolute_parameter_change"],
                "frozen_unchanged": True,
                "reload_exact": True,
                "restore_exact": True,
            }
        )
        log(2 * len(smoke), {"operation": "cuda_scope_smoke", "scope": scope})
    return {
        "passed": True,
        "smoke": smoke,
        "engineering_optimizer_updates": 6,
        "scientific_optimizer_updates": 0,
        "new_composition_supervision": 0,
        "old_teacher_sha256": alpha.digest(out / "old-teachers.pt"),
        "old_split_sha256": alpha.digest(out / "old-sample-split.json"),
    }


def load_protection(root, reader):
    prepared = json.loads((root / "prepare/result.json").read_text())
    if alpha.digest(root / "prepare/old-teachers.pt") != prepared["old_teacher_sha256"]:
        raise AssertionError("Fixed protective teacher changed")
    if alpha.digest(root / "prepare/old-sample-split.json") != prepared["old_split_sha256"]:
        raise AssertionError("Old sample split changed")
    teachers = torch.load(root / "prepare/old-teachers.pt", map_location="cpu", weights_only=True)
    device = next(reader.parameters()).device
    for step in teachers:
        for batch in step:
            for key in ("tokens", "positions", "log_probability"):
                batch[key] = batch[key].to(device)
    return json.loads((root / "prepare/old-sample-split.json").read_text()), teachers


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
                out / f"development-fact-{learning_rate}-{fact['case_id']}.json", records[-1]
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
        rates = {name: v["retained"] / v["originally_correct"] for name, v in totals.items()}
        candidate = {
            "learning_rate": learning_rate,
            "atomic_accuracy": correct / len(facts),
            "minimum_old_retention": min(rates.values()),
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
        if alpha.normalize(raw_before) != inherited["before"]:
            raise AssertionError("Foundation readout differs from the paired AlphaEdit record")
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
    result = summarize(records, plan["statistics"])
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


def paired_comparison(plan, root):
    methods = {"alphaedit": Path(plan["alpha_evaluation"])}
    methods.update({scope: root / f"evaluation-{scope}" for scope in SCOPES})
    results, records = {}, {}
    for method, folder in methods.items():
        results[method] = json.loads((folder / "result.json").read_text())
        records[method] = [
            json.loads(p.read_text()) for p in sorted(folder.glob("evaluation-novel-*.json"))
        ]
        if len(records[method]) != 32:
            raise AssertionError("Comparison must include all32 facts for every method")
    cases = [[r["fact"] for r in records[m]] for m in methods]
    if any(value != cases[0] for value in cases[1:]):
        raise AssertionError("Comparison changed the novel panel")
    paired = {}
    for i, left in enumerate(methods):
        for right in list(methods)[i + 1 :]:
            changes = {}
            for kind in ("novel_atom", "first_hop", "second_hop"):
                differences = [
                    np.mean(a["after"][kind]["final_correct"])
                    - np.mean(b["after"][kind]["final_correct"])
                    for a, b in zip(records[left], records[right], strict=True)
                ]
                changes[kind] = bootstrap(differences, plan["statistics"])
            paired[f"{left}_minus_{right}"] = changes
    compact = {}
    for method, result in results.items():
        old_atoms, old_chains = [], []
        for record in records[method]:
            for name in alpha.POOLS:
                b = record["before"]["extended_old"][name]["final_correct"]
                a = record["after"]["extended_old"][name]["final_correct"]
                dest = old_atoms if name.startswith("atomic") else old_chains
                dest.extend(x for x, keep in zip(a, b, strict=True) if keep)
        compact[method] = {
            "new_atomic": result["conditions"]["after"]["novel_atom"],
            "new_then_old": result["conditions"]["after"]["first_hop"],
            "old_then_new": result["conditions"]["after"]["second_hop"],
            "old_atoms_retained": rate(old_atoms),
            "old_compositions_retained": rate(old_chains),
            "conditional": result["conditional"],
            "scientific_optimizer_updates": result.get("scientific_optimizer_updates"),
            "compute": result.get("compute"),
            "wall_seconds": result["wall_seconds"],
        }
    return {
        "passed": True,
        "facts": 32,
        "methods": compact,
        "paired_fact_bootstrap": paired,
        "ce_fixed_steps_per_fact": plan["ce"]["steps"],
        "ce_budget_matched": "updates, new target tokens, old batches and development grid",
        "alphaedit_budget_matched": False,
        "new_composition_supervision": 0,
        "scientific_optimizer_updates": 0,
    }


def cloud(plan, root):
    import wandb

    api, receipts = wandb.Api(timeout=60), []
    for stage in [
        "prepare",
        *[f"{kind}-{scope}" for kind in ("development", "evaluation", "audit") for scope in SCOPES],
    ]:
        tracking = json.loads((root / stage / "wandb.json").read_text())
        expected = root / stage / "result.json"
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if run.state != "finished" or run.summary.get("result_sha256") != alpha.digest(expected):
            raise AssertionError("Cloud endpoint identity/state differs")
        actual = run.file("result.json").download(root=str(root / "cloud" / stage), replace=False)
        if alpha.digest(actual.name) != alpha.digest(expected):
            raise AssertionError("Cloud canonical JSON differs")
        if (
            run.summary.get("scientific_optimizer_updates")
            != json.loads(expected.read_text())["scientific_optimizer_updates"]
        ):
            raise AssertionError("Cloud update accounting differs")
        receipts.append({"stage": stage, **tracking, "result_sha256": alpha.digest(expected)})
    comparison = paired_comparison(plan, root)
    write_json(root / "cloud/comparison.json", comparison)
    return {
        "passed": True,
        "canonical_full_json_exact": True,
        "runs": receipts,
        "comparison_sha256": alpha.digest(root / "cloud/comparison.json"),
        "scientific_optimizer_updates": 0,
    }


def main():
    alpha.verify_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "development", "evaluation", "audit", "cloud"))
    parser.add_argument("--scope", choices=SCOPES)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.stage in ("development", "evaluation", "audit")) != (args.scope is not None):
        raise ValueError("Exactly the three scientific stages require a parameter scope")
    plan = json.loads(args.plan.read_text())
    stage_name = args.stage + (f"-{args.scope}" if args.scope else "")
    out = args.root / stage_name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "failure.json").exists():
        raise FileExistsError("Preserve all previous outputs; no automatic stage retry")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan["seed"])
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run, started = None, time.monotonic()
        try:
            if args.stage == "cloud":
                result = cloud(plan, args.root)
            else:
                import wandb

                run = wandb.init(
                    entity="zhusq20",
                    project="llm-memory-editability",
                    name=f"{plan['batch']}/{stage_name}",
                    group=plan["batch"],
                    job_type=args.stage,
                    config={**plan, "scope": args.scope},
                    dir=str(out),
                    mode="online",
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / "wandb.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)
                reader, world, panel = alpha.initialize(plan, args.device)
                before = tensor_digest(reader.backbone.state_dict())

                def log(step, record):
                    write_json(out / "progress.json", {"utc": utc(), "step": step, **record})
                    print(
                        json.dumps({"stage": stage_name, "step": step, **record}),
                        flush=True,
                    )
                    run.log(record, step=step)

                if args.stage == "prepare":
                    result = prepare(plan, out, reader, world, panel, log)
                elif args.stage == "development":
                    result = develop(plan, out, args.root, args.scope, reader, world, panel, log)
                else:
                    result = evaluate(
                        plan,
                        out,
                        args.root,
                        args.scope,
                        reader,
                        world,
                        panel,
                        log,
                        audit=args.stage == "audit",
                    )
                verify_base(reader, before)
                for path, expected in plan["bound_inputs"].items():
                    if alpha.digest(path) != expected:
                        raise AssertionError("An immutable bound input changed")
                result.update(
                    foundation_digest=before,
                    input_hashes=plan["bound_inputs"],
                    compute=dict(reader.compute),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                )
            result.update(
                stage=stage_name,
                wall_seconds=time.monotonic() - started,
                source_sha256=alpha.digest(ROOT / "source-manifest.json"),
            )
            write_json(out / "result.json", result)
            if run:
                run.summary["result_sha256"] = alpha.digest(out / "result.json")
                run.summary["scientific_optimizer_updates"] = result["scientific_optimizer_updates"]
                run.save(str(out / "result.json"), base_path=str(out), policy="now")
                run.finish()
            print(json.dumps({"complete": True, "stage": stage_name}), flush=True)
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
