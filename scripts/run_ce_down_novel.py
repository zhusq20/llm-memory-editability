"""Direct atomic CE addition and projected CE on an audited native DiscoLoop."""

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
import torch  # noqa: E402
from transformers import GPT2Config  # noqa: E402

import run_ttcd_novel_fact_transfer as readouts  # noqa: E402
from llm_memory_editability.ce_down_projection import (  # noqa: E402
    atomic_ce_write,
    basis_from_covariance,
    choose_learning_rate,
    collect_covariances,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import encode_rows, model_digest  # noqa: E402
from llm_memory_editability.path_overlap import World  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest  # noqa: E402

METHODS = ("atomic_ce", "projected_ce")


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
    return manifest


def initialize(plan, device):
    for path, expected in plan["bound_inputs"].items():
        if digest(path) != expected:
            raise ValueError(f"Immutable input changed: {path}")
    panel = json.loads(Path(plan["panel"]).read_text())
    world = World(plan["data_dir"])
    gate = json.loads(Path(plan["foundation_gate"]).read_text())
    if not gate["passed"] or gate["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Native foundation audit must bind the checkpoint")
    saved = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
    if saved["step"] != 300000 or not saved["architecture"]["discoloop"]:
        raise ValueError("Use the native300k DiscoLoop, without record/writer adaptation")
    hf = GPT2Config.from_json_file(plan["model_config"])
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
        raise ValueError("Development and test addition keys must be distinct")
    with torch.no_grad():
        arrays = encode_rows(world.atoms[panel["U_atoms"][:8]], world.metadata)
        tokens, positions, _ = (torch.as_tensor(a, device=device) for a in arrays)
        if not torch.equal(reader(tokens, positions, fast=reader.empty_fast()), model(tokens, positions)):
            raise AssertionError("Zero down increments changed native logits")
    return reader, world, panel


def old_split(world, panel, settings):
    rng = np.random.RandomState(settings["seed"])
    reserved_atoms, reserved_chains = set(panel["U_atoms"]), set(panel["U_chains"])
    candidates = np.setdiff1d(np.arange(len(world.atoms)), list(reserved_atoms))
    chosen = rng.choice(candidates, settings["atoms"] + settings["development_controls"], replace=False)
    chain_candidates = np.setdiff1d(np.flatnonzero(world.support(7.2)), list(reserved_chains))
    chains = rng.choice(chain_candidates, settings["chains"] + settings["development_controls"], replace=False)
    return {
        "protection_atoms": chosen[: settings["atoms"]].tolist(),
        "protection_chains": chains[: settings["chains"]].tolist(),
        "development_atoms": chosen[settings["atoms"] :].tolist(),
        "development_chains": chains[settings["chains"] :].tolist(),
        "formal_atoms": list(panel["U_atoms"]),
        "formal_chains": list(panel["U_chains"]),
    }


def retention(before, after):
    kept = [a for a, b in zip(after["final_correct"], before["final_correct"], strict=True) if b]
    return sum(kept), len(kept)


def develop(plan, out, reader, world, panel, log):
    split = old_split(world, panel, plan["protection"])
    write_json(out / "old-sample-split.json", split)
    covariances, counts = collect_covariances(
        reader,
        (world.atoms[split["protection_atoms"]], world.chains[split["protection_chains"]]),
        world.metadata,
        plan["evaluation_batch_size"],
    )
    bases, geometry = {}, {}
    for layer, covariance in covariances.items():
        bases[layer], geometry[layer] = basis_from_covariance(covariance, plan["protection"]["energy"])
    torch.save({"bases": bases, "geometry": geometry, "counts": counts, "split": split}, out / "protection.pt")
    write_json(out / "geometry.json", {"layers": geometry, "counts": counts})
    del covariances
    bases = {k: v.to(device=next(reader.parameters()).device, dtype=torch.float32) for k, v in bases.items()}
    controls = (world.atoms[split["development_atoms"]], world.chains[split["development_chains"]])
    baseline = [readouts.greedy_rows(reader, None, rows, world.metadata) for rows in controls]
    facts = [f for f in panel["facts"] if f["split"] == "development"]
    results, selected, writes = {}, {}, 0
    for method in METHODS:
        results[method] = []
        for rate in plan["learning_rates"]:
            correct, kept, denominator, per_fact = 0, 0, 0, []
            for fact in facts:
                state, info = atomic_ce_write(reader, fact["novel_fact"], world.metadata, rate,
                                              bases=bases if method == "projected_ce" else None,
                                              steps=plan["writer_steps"], stop_loss=plan["stop_loss"],
                                              max_relative_update=plan["max_relative_update"])
                atom = readouts.greedy_rows(reader, state, [fact["novel_fact"]], world.metadata)
                correct += sum(atom["final_correct"])
                for before, rows in zip(baseline, controls, strict=True):
                    after = readouts.greedy_rows(reader, state, rows, world.metadata)
                    k, n = retention(before, after)
                    kept, denominator = kept + k, denominator + n
                per_fact.append({"case_id": fact["case_id"], "atomic": atom, "write": info})
                writes += info["steps"]
            record = {"learning_rate": rate, "atomic": correct / len(facts),
                      "old_retention": kept / denominator if denominator else 0.0}
            results[method].append(record)
            write_json(out / f"development-{method}-{rate}.json", {"selection": record, "facts": per_fact})
            log(writes, {"method": method, "fast_updates": writes, **record})
        selected[method] = choose_learning_rate(results[method], plan["minimum_development_retention"])
    return {"passed": True, "selected": selected, "grid": results, "geometry": geometry,
            "new_composition_queries": 0, "base_parameter_updates": 0, "fast_updates": writes}


def evaluate(plan, out, root, reader, world, panel, log, *, audit=False):
    development = json.loads((root / "development/result.json").read_text())
    protected = torch.load(root / "development/protection.pt", map_location="cpu", weights_only=True)
    device = next(reader.parameters()).device
    bases = {k: v.to(device=device, dtype=torch.float32) for k, v in protected["bases"].items()}
    facts = [f for f in panel["facts"] if f["split"] == "evaluation"]
    result, all_records, writes = {}, {}, 0
    baseline = reader.empty_fast()
    for method in METHODS:
        directory = out / method
        directory.mkdir(exist_ok=False)
        if not audit:
            (directory / "states").mkdir()
        rate = development["selected"][method]["learning_rate"]
        records = []
        for fact in facts:
            state, info = atomic_ce_write(reader, fact["novel_fact"], world.metadata, rate,
                                          bases=bases if method == "projected_ce" else None,
                                          steps=plan["writer_steps"], stop_loss=plan["stop_loss"],
                                          max_relative_update=plan["max_relative_update"])
            state_path = root / "evaluation" / method / "states" / f"{fact['case_id']}.pt"
            if audit:
                saved = torch.load(state_path, map_location=device, weights_only=True)
                if saved["fact"] != fact:
                    raise AssertionError("Saved state belongs to a different addition")
                if any(not torch.equal(saved["fast"][k], v) for k, v in state.items()):
                    raise AssertionError("Independent CE rewrite differs from saved down update")
                state = saved["fast"]
            else:
                torch.save({"fast": {k: v.cpu() for k, v in state.items()}, "fact": fact}, state_path)
            record = normalize({"fact": fact, "write_info": info,
                                "before": readouts.evaluate_state(reader, baseline, fact, world, panel),
                                "after": readouts.evaluate_state(reader, state, fact, world, panel),
                                "fast_digest": tensor_digest(state)})
            if audit and record != json.loads((root / "evaluation" / method / f"{fact['case_id']}.json").read_text()):
                raise AssertionError("Independent saved-state readout differs")
            write_json(directory / f"{fact['case_id']}.json", record)
            records.append(record)
            writes += info["steps"]
            log(writes, {"method": method, "completed": len(records), "fast_updates": writes})
        result[method] = readouts.summarize(records, plan["statistics"])
        all_records[method] = records
    paired = {}
    for kind in ("novel_atom", "first_hop", "second_hop"):
        differences = [float(np.mean(b["after"][kind]["final_correct"]) - np.mean(a["after"][kind]["final_correct"]))
                       for a, b in zip(all_records["atomic_ce"], all_records["projected_ce"], strict=True)]
        paired[kind] = readouts.bootstrap(differences, plan["statistics"])
    return {"passed": True, "methods": result, "projected_minus_plain": paired,
            "base_parameter_updates": 0, "fast_updates": writes, "maximum_writer_steps_per_fact": plan["writer_steps"],
            "independent_rewrite_and_saved_state_replay": audit, "new_composition_supervision": 0}


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
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.root.mkdir(parents=True, exist_ok=True)
    out = args.root / args.stage
    out.mkdir(exist_ok=True)
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

                    run = wandb.init(entity="zhusq20", project="llm-memory-editability",
                                     name=f"{plan['batch']}/{args.stage}", group=plan["batch"],
                                     job_type=args.stage, config=plan, dir=str(out), mode="online",
                                     settings=wandb.Settings(disable_git=True))
                    write_json(out / "wandb.json", {"id": run.id, "url": run.url})
                    print(json.dumps({"wandb": run.url}), flush=True)
                reader, world, panel = initialize(plan, args.device)
                before = tensor_digest(reader.backbone.state_dict())

                def log(step, record):
                    write_json(out / "progress.json", {"utc": utc(), "step": step, **record})
                    print(json.dumps({"stage": args.stage, "step": step, **record}), flush=True)
                    if run:
                        run.log(record, step=step)

                if args.stage == "development":
                    result = develop(plan, out, reader, world, panel, log)
                else:
                    result = evaluate(plan, out, args.root, reader, world, panel, log, audit=args.stage == "audit")
                if before != tensor_digest(reader.backbone.state_dict()):
                    raise AssertionError("Frozen backbone changed")
                for path, expected in plan["bound_inputs"].items():
                    if digest(path) != expected:
                        raise AssertionError("Immutable scientific input changed")
                result.update(backbone_unchanged=True, model_digest=before, input_hashes=plan["bound_inputs"],
                              compute=dict(reader.compute),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device)
                              if str(args.device).startswith("cuda") else None)
            result.update(stage=args.stage, wall_seconds=time.monotonic() - started, source_sha256=digest(ROOT / "source-manifest.json"))
            write_json(out / "result.json", result)
            if run:
                run.summary["result_sha256"] = digest(out / "result.json")
                run.summary["base_parameter_updates"] = 0
                run.summary["fast_updates"] = result["fast_updates"]
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
