"""Isolated new atomic facts edited by AlphaEdit on the native300k DiscoLoop."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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
import run_ttcd_novel_fact_transfer as readouts  # noqa: E402
import torch  # noqa: E402
from check_alphaedit_edit_rounds import inspect_rows  # noqa: E402
from transformers import GPT2Config  # noqa: E402

from llm_memory_editability.alphaedit_loop import (  # noqa: E402
    collect_old_covariances,
    fit_target,
    null_basis,
    select_l2,
    write_target,
)
from llm_memory_editability.alphaedit_summary import summarize  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import encode_rows, model_digest  # noqa: E402
from llm_memory_editability.path_overlap import World  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest  # noqa: E402

POOLS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")
# Observation only: use full native-shape RMS reconstruction on CUDA.
readouts.inspect_rows = inspect_rows
readouts.summarize = summarize


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalize(value):
    return json.loads(json.dumps(value))


def verify_source():
    path = ROOT / "source-manifest.json"
    if digest(path) != os.environ["LM_SOURCE_MANIFEST_SHA256"]:
        raise ValueError("Frozen source manifest differs")
    manifest = json.loads(path.read_text())
    for name, expected in manifest["files"].items():
        candidate = ROOT / name
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Frozen source escaped its root")
        if digest(candidate) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def initialize(plan, device):
    for path, expected in plan["bound_inputs"].items():
        if digest(path) != expected:
            raise ValueError(f"Immutable input changed: {path}")
    if digest(ROOT / plan["panel"]) != plan["panel_sha256"]:
        raise ValueError("Novel fact panel identity differs")
    panel = json.loads((ROOT / plan["panel"]).read_text())
    world = World(plan["data_dir"])
    gate = json.loads(Path(plan["foundation_gate"]).read_text())
    if not gate["passed"] or gate["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Foundation audit must bind the checkpoint")
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
    if saved["step"] != 300000 or not saved["architecture"]["discoloop"]:
        raise ValueError("Use native300k DiscoLoop without record/writer adaptation")
    hf = GPT2Config.from_json_file(plan["model_config"])
    hf.n_layer = saved["spec"]["unique_layers"]
    hf.vocab_size = world.metadata["vocab_size"]
    hf.use_cache = False
    hf.embd_pdrop = hf.resid_pdrop = hf.attn_pdrop = 0.0
    model = SupervisedAlignmentGPT(hf, saved["spec"]["repeats"], **saved["architecture"])
    model.load_state_dict(saved["model"], strict=True)
    if model_digest(model) != gate["model_sha256"]:
        raise ValueError("Foundation model digest differs")
    reader = TTCDReader(model.to(device), native_causal_attention=True).eval()
    for fact in panel["facts"]:
        readouts.validate_fact(fact, world)
    if len({tuple(f["novel_fact"][:2]) for f in panel["facts"]}) != len(panel["facts"]):
        raise ValueError("Development and evaluation addition keys must be distinct")
    with torch.no_grad():
        arrays = encode_rows(world.atoms[panel["U_atoms"][:8]], world.metadata)
        tokens, positions, _ = (torch.as_tensor(a, device=device) for a in arrays)
        if not torch.equal(
            reader(tokens, positions, fast=reader.empty_fast()), model(tokens, positions)
        ):
            raise AssertionError("Zero down increments changed native logits")
    return reader, world, panel


def old_split(world, panel, settings):
    """Sample once from graph IDs; formal old pools cannot set projection/recipe."""
    rng = np.random.RandomState(settings["seed"])
    excluded = {tuple(row) for row in world.atoms[panel["U_atoms"]]}
    excluded.update(tuple(row) for row in world.chains[panel["U_chains"]])
    development, formal = {}, {}
    for name in POOLS:
        available = [row.tolist() for row in world.panels[name] if tuple(row) not in excluded]
        order = rng.permutation(len(available))
        n, d = settings["formal_per_pool"], settings["development_per_pool"]
        formal[name] = [available[i] for i in order[:n]]
        development[name] = [available[i] for i in order[n : n + d]]
        excluded.update(tuple(row) for row in formal[name] + development[name])
    atom_candidates = [i for i, row in enumerate(world.atoms) if tuple(row) not in excluded]
    chain_candidates = [
        i for i in np.flatnonzero(world.support(7.2)) if tuple(world.chains[i]) not in excluded
    ]
    atoms = rng.choice(atom_candidates, settings["atoms"], replace=False).tolist()
    chains = rng.choice(chain_candidates, settings["chains"], replace=False).tolist()
    return {
        "protection_atoms": atoms,
        "protection_chains": chains,
        "development": development,
        "formal": formal,
        "inherited_U_atoms": list(panel["U_atoms"]),
        "inherited_U_chains": list(panel["U_chains"]),
    }


def kl_controls(fact, world, split, count=2):
    excluded = {tuple(row) for rows in split["formal"].values() for row in rows}
    excluded.update(tuple(row) for row in world.atoms[split["inherited_U_atoms"]])
    h = fact[0]
    rows = world.atoms[world.atoms[:, 0] == h]
    return [row.tolist() for row in rows if tuple(row) not in excluded][:count]


def old_pools(reader, fast, rows, world):
    return {
        name: readouts.greedy_rows(reader, fast, np.asarray(pool), world.metadata)
        for name, pool in rows.items()
    }


def retention(before, after):
    kept = [a for a, b in zip(after["final_correct"], before["final_correct"], strict=True) if b]
    return {
        "retained": sum(kept),
        "originally_correct": len(kept),
        "retention": sum(kept) / len(kept) if kept else None,
    }


def develop(plan, out, reader, world, panel, log):
    split = old_split(world, panel, plan["protection"])
    write_json(out / "old-sample-split.json", split)
    log(0, {"operation": "old_covariance", "base_parameter_updates": 0})
    covariances, counts = collect_old_covariances(
        reader,
        (world.atoms[split["protection_atoms"]], world.chains[split["protection_chains"]]),
        world.metadata,
        plan["evaluation_batch_size"],
    )
    bases, geometry = {}, {}
    for layer, covariance in covariances.items():
        basis, detail = null_basis(covariance, plan["protection"]["nullspace_threshold"])
        bases[layer] = basis.cpu()
        detail["selected_eigenvalues"] = detail["selected_eigenvalues"].tolist()
        geometry[layer] = detail
        log(
            0,
            {
                "operation": "null_space",
                "layer": layer,
                "null_dimension": detail["null_dimension"],
                "protected_dimension": detail["protected_dimension"],
            },
        )
    torch.save(
        {"bases": bases, "geometry": geometry, "counts": counts, "split": split},
        out / "protection.pt",
    )
    write_json(out / "geometry.json", {"layers": geometry, "counts": counts})
    del covariances
    device = next(reader.parameters()).device
    bases = {k: v.to(device=device) for k, v in bases.items()}
    baseline = old_pools(reader, None, split["development"], world)
    facts = [f for f in panel["facts"] if f["split"] == "development"]
    targets, latent_updates = {}, 0
    for fact in facts:
        controls = kl_controls(fact["novel_fact"], world, split)
        target, info = fit_target(
            reader, fact["novel_fact"], controls, world.metadata, plan["target"]
        )
        targets[fact["case_id"]] = (target, info)
        latent_updates += info["latent_updates"]
        log(
            latent_updates,
            {
                "operation": "fit_target",
                "case_id": fact["case_id"],
                "target_probability": info["history"][-1]["target_probability"],
            },
        )
    candidates, down_updates = [], 0
    for l2 in plan["l2_candidates"]:
        correct, records = 0, []
        sums = {name: {"retained": 0, "originally_correct": 0} for name in POOLS}
        for fact in facts:
            target, target_info = targets[fact["case_id"]]
            state, info = write_target(
                reader, fact["novel_fact"], world.metadata, target, bases, l2
            )
            atom = readouts.greedy_rows(reader, state, [fact["novel_fact"]], world.metadata)
            after = old_pools(reader, state, split["development"], world)
            detail = {name: retention(baseline[name], after[name]) for name in POOLS}
            for name in POOLS:
                for key in ("retained", "originally_correct"):
                    sums[name][key] += detail[name][key]
            correct += sum(atom["final_correct"])
            records.append(
                {
                    "case_id": fact["case_id"],
                    "atomic": atom,
                    "target_fit": target_info,
                    "write": info,
                    "old_retention": detail,
                }
            )
            down_updates += info["down_matrix_updates"]
        rates = {
            name: v["retained"] / v["originally_correct"] if v["originally_correct"] else None
            for name, v in sums.items()
        }
        record = {
            "l2": l2,
            "atomic_accuracy": correct / len(facts),
            "minimum_old_retention": min(v for v in rates.values() if v is not None),
            "retention_by_pool": rates,
        }
        candidates.append(record)
        write_json(out / f"candidate-l2-{l2}.json", {"selection": record, "facts": records})
        log(latent_updates + down_updates, {"operation": "development_candidate", **record})
    selected = select_l2(candidates, plan["minimum_development_retention"])
    return {
        "passed": True,
        "selected": selected,
        "candidates": candidates,
        "geometry": geometry,
        "base_parameter_updates": 0,
        "latent_updates": latent_updates,
        "down_matrix_updates": down_updates,
        "new_composition_queries": 0,
        "new_composition_supervision": 0,
    }


def retention_summary(records, name):
    before = [v for r in records for v in r["before"]["extended_old"][name]["final_correct"]]
    after = [v for r in records for v in r["after"]["extended_old"][name]["final_correct"]]
    return {
        "before": readouts.rate(before),
        "after": readouts.rate(after),
        "initially_correct_retained": readouts.rate(
            a for a, b in zip(after, before, strict=True) if b
        ),
        "initially_correct_lost": sum(b and not a for a, b in zip(after, before, strict=True)),
        "unique_queries": len(records[0]["before"]["extended_old"][name]["rows"]),
        "independent_edit_states": len(records),
    }


def recover_complete_evaluation(plan, out, reader, world, panel, log):
    """Summarize all32 immutable saved records without fitting or writing again."""
    records = []
    expected = plan["recovery"]["complete_records"]
    for fact in panel["facts"]:
        if fact["split"] != "evaluation":
            continue
        case = fact["case_id"]
        path, state_path = out / f"{case}.json", out / "states" / f"{case}.pt"
        if digest(path) != expected[case] or digest(state_path) != plan["recovery"]["states"][case]:
            raise AssertionError("Completed original evaluation evidence changed")
        record = json.loads(path.read_text())
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if record["fact"] != fact or saved["fact"] != fact:
            raise AssertionError("Completed evidence belongs to another fact")
        if tensor_digest(saved["fast"]) != record["fast_digest"]:
            raise AssertionError("Completed raw record and saved update differ")
        if saved["target_info"] != record["target_fit"]:
            raise AssertionError("Completed target-fit accounting differs")
        records.append(record)
    if len(records) != 32:
        raise AssertionError("The original formal panel must be complete")
    result = summarize(records, plan["statistics"])
    result["extended_retention"] = {name: retention_summary(records, name) for name in POOLS}
    development = json.loads((out.parent / "development/result.json").read_text())
    if (
        digest(out.parent / "development/result.json")
        != plan["recovery"]["development_result_sha256"]
    ):
        raise AssertionError("Original development selection changed")
    result.update(
        passed=True,
        selected_l2=development["selected"]["l2"],
        base_parameter_updates=0,
        latent_updates=0,
        down_matrix_updates=0,
        scientific_latent_updates_including_prior_attempts=sum(
            r["target_fit"]["latent_updates"] for r in records
        ),
        unique_formal_down_matrix_updates=128,
        new_composition_supervision=0,
        recovered_complete_records=len(records),
        original_record_sha256=expected,
        operation="recover summary from complete original predictions; no new science writes",
        original_formal_postflight=(
            "Original process stopped before final model digest; "
            "fresh recovery and audit check theta and bound inputs"
        ),
    )
    log(
        0,
        {
            "completed_facts": len(records),
            "base_parameter_updates": 0,
            "latent_updates": 0,
            "down_matrix_updates": 0,
            "atomic_accuracy": result["conditions"]["after"]["novel_atom"]["accuracy"],
            "first_hop_accuracy": result["conditions"]["after"]["first_hop"]["accuracy"],
            "second_hop_accuracy": result["conditions"]["after"]["second_hop"]["accuracy"],
        },
    )
    return result


def evaluate(plan, out, root, reader, world, panel, log, *, audit=False):
    if "recovery" in plan:
        if (
            digest(root / "development/result.json")
            != plan["recovery"]["development_result_sha256"]
        ):
            raise ValueError("Reused development recipe differs")
        if digest(root / "development/protection.pt") != plan["recovery"]["protection_sha256"]:
            raise ValueError("Reused projection differs")
    development = json.loads((root / "development/result.json").read_text())
    protected = torch.load(
        root / "development/protection.pt", map_location="cpu", weights_only=True
    )
    split = protected["split"]
    device = next(reader.parameters()).device
    bases = {k: v.to(device=device) for k, v in protected["bases"].items()}
    l2 = development["selected"]["l2"]
    facts = [f for f in panel["facts"] if f["split"] == "evaluation"]
    if not audit:
        (out / "states").mkdir(exist_ok=False)
    baseline = reader.empty_fast()
    baseline_old = old_pools(reader, baseline, split["formal"], world)
    records, latent_updates, down_updates = [], 0, 0
    inherited_latent, inherited_down, recovered_states = 0, 0, 0
    for fact in facts:
        path = root / "evaluation/states" / f"{fact['case_id']}.pt"
        prior = plan.get("recovery", {}).get("states", {}).get(fact["case_id"])
        if audit:
            saved = torch.load(path, map_location=device, weights_only=True)
            if saved["fact"] != fact:
                raise AssertionError("Saved edit belongs to another fact")
            target, target_info = saved["target"], saved["target_info"]
        elif prior:
            prior_path = (
                Path(plan["recovery"]["prior_evaluation"]) / "states" / f"{fact['case_id']}.pt"
            )
            if digest(prior_path) != prior:
                raise AssertionError("Prior saved state identity differs")
            inherited = torch.load(prior_path, map_location=device, weights_only=True)
            if inherited["fact"] != fact:
                raise AssertionError("Prior edit belongs to another fact")
            target, target_info = inherited["target"], inherited["target_info"]
            inherited_latent += target_info["latent_updates"]
            inherited_down += len(inherited["fast"])
            recovered_states += 1
        else:
            controls = kl_controls(fact["novel_fact"], world, split)
            target, target_info = fit_target(
                reader, fact["novel_fact"], controls, world.metadata, plan["target"]
            )
            latent_updates += target_info["latent_updates"]
        rewritten, info = write_target(
            reader, fact["novel_fact"], world.metadata, target, bases, l2
        )
        down_updates += info["down_matrix_updates"]
        if audit:
            if any(not torch.equal(saved["fast"][k], v) for k, v in rewritten.items()):
                raise AssertionError("Independent AlphaEdit solve differs from saved state")
            state = saved["fast"]
        else:
            state = rewritten
            if prior:
                if any(not torch.equal(inherited["fast"][k], v) for k, v in rewritten.items()):
                    raise AssertionError("Recovered edit changed during observer repair")
                state = inherited["fast"]
            torch.save(
                {
                    "fast": {k: v.cpu() for k, v in state.items()},
                    "fact": fact,
                    "target": target.cpu(),
                    "target_info": target_info,
                },
                path,
            )
        before = readouts.evaluate_state(reader, baseline, fact, world, panel)
        before["extended_old"] = baseline_old
        after = readouts.evaluate_state(reader, state, fact, world, panel)
        after["extended_old"] = old_pools(reader, state, split["formal"], world)
        record = normalize(
            {
                "fact": fact,
                "target_fit": target_info,
                "write_info": info,
                "before": before,
                "after": after,
                "fast_digest": tensor_digest(state),
            }
        )
        if audit and record != json.loads(
            (root / "evaluation" / f"{fact['case_id']}.json").read_text()
        ):
            raise AssertionError("Independent saved-state readout differs")
        write_json(out / f"{fact['case_id']}.json", record)
        records.append(record)
        log(
            latent_updates + down_updates,
            {
                "completed_facts": len(records),
                "atomic_correct": sum(after["novel_atom"]["final_correct"]),
                "first_hop_correct": sum(after["first_hop"]["final_correct"]),
                "second_hop_correct": sum(after["second_hop"]["final_correct"]),
                "latent_updates": latent_updates,
                "down_matrix_updates": down_updates,
            },
        )
    result = readouts.summarize(records, plan["statistics"])
    result["extended_retention"] = {name: retention_summary(records, name) for name in POOLS}
    result.update(
        passed=True,
        selected_l2=l2,
        base_parameter_updates=0,
        latent_updates=latent_updates,
        down_matrix_updates=down_updates,
        independent_solve_and_saved_state_replay=audit,
        audit_latent_optimizer_updates=0 if audit else None,
        new_composition_supervision=0,
        inherited_latent_updates=inherited_latent,
        inherited_down_matrix_updates=inherited_down,
        recovered_prior_states=recovered_states,
        total_unique_fact_latent_updates=sum(r["target_fit"]["latent_updates"] for r in records),
        unique_formal_down_matrix_updates=4 * len(facts),
        recovery_down_recomputations=inherited_down if not audit else down_updates,
    )
    return result


def cloud(root):
    import wandb

    api, receipts = wandb.Api(timeout=60), []
    for stage in ("development", "evaluation", "audit"):
        tracking = json.loads((root / stage / "wandb.json").read_text())
        expected = root / stage / "result.json"
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if run.state != "finished" or run.summary.get("result_sha256") != digest(expected):
            raise AssertionError("Cloud endpoint identity/state mismatch")
        download = root / "cloud" / stage
        download.mkdir(parents=True, exist_ok=True)
        actual = run.file("result.json").download(root=str(download), replace=False)
        if digest(actual.name) != digest(expected):
            raise AssertionError("Cloud canonical full result differs")
        for key in ("latent_updates", "down_matrix_updates", "base_parameter_updates"):
            if run.summary.get(key) != json.loads(expected.read_text())[key]:
                raise AssertionError(f"Cloud update accounting differs: {key}")
        receipts.append({"stage": stage, **tracking, "result_sha256": digest(expected)})
    return {"passed": True, "canonical_full_json_exact": True, "runs": receipts}


def main():
    verify_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "evaluation", "audit", "cloud"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    parser.add_argument("--recover-complete-evaluation", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = args.root / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() or (out / "failure.json").exists():
        raise FileExistsError("Preserve prior stage outputs; no automatic retry")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan["seed"])
    if args.tracking == "online":
        os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run, started = None, time.monotonic()
        try:
            if args.stage == "cloud":
                result = cloud(args.root)
            else:
                if args.tracking == "online":
                    import wandb

                    run = wandb.init(
                        entity="zhusq20",
                        project="llm-memory-editability",
                        name=f"{plan['batch']}/{args.stage}",
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
                    print(json.dumps({"stage": args.stage, "step": step, **record}), flush=True)
                    if run:
                        run.log(record, step=step)

                if args.recover_complete_evaluation:
                    if args.stage != "evaluation":
                        raise ValueError("Complete-record recovery applies only to evaluation")
                    result = recover_complete_evaluation(plan, out, reader, world, panel, log)
                else:
                    result = (
                        develop(plan, out, reader, world, panel, log)
                        if args.stage == "development"
                        else evaluate(
                            plan,
                            out,
                            args.root,
                            reader,
                            world,
                            panel,
                            log,
                            audit=args.stage == "audit",
                        )
                    )
                if before != tensor_digest(reader.backbone.state_dict()):
                    raise AssertionError("Frozen backbone changed")
                for path, expected in plan["bound_inputs"].items():
                    if digest(path) != expected:
                        raise AssertionError("Immutable scientific input changed")
                result.update(
                    backbone_unchanged=True,
                    model_digest=before,
                    input_hashes=plan["bound_inputs"],
                    compute=dict(reader.compute),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device)
                    if str(args.device).startswith("cuda")
                    else None,
                )
            result.update(
                stage=args.stage,
                wall_seconds=time.monotonic() - started,
                source_sha256=digest(ROOT / "source-manifest.json"),
            )
            write_json(out / "result.json", result)
            if run:
                run.summary["result_sha256"] = digest(out / "result.json")
                for key in ("base_parameter_updates", "latent_updates", "down_matrix_updates"):
                    run.summary[key] = result[key]
                run.save(str(out / "result.json"), base_path=str(out), policy="now")
                run.finish()
            print(json.dumps({"complete": True, "stage": args.stage}), flush=True)
        except BaseException:
            write_json(out / "failure.json", {"utc": utc(), "traceback": traceback.format_exc()})
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
