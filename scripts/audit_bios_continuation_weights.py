"""Read-only checksum and CPU state audit of every newly produced P2 checkpoint."""

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.bios_cross_continue import (
    CHECKPOINTS,
    continuation_sources,
    tree_hash,
    validate_optimizer,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_organization_train import state_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, sources):
    data = Path(path).read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def load_checkpoint(path, records, root):
    before = path.stat()
    contents = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Checkpoint changed while reading: {path}")
    records[str(path.relative_to(root))] = {
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
        "mtime_ns": after.st_mtime_ns,
    }
    return torch.load(io.BytesIO(contents), map_location="cpu", weights_only=False)


def check_model(state, shapes):
    if set(state) != set(shapes):
        raise ValueError("Checkpoint parameter names differ from the frozen architecture")
    for name, tensor in state.items():
        if tuple(tensor.shape) != shapes[name] or tensor.dtype != torch.float32:
            raise ValueError(f"Parameter shape or precision mismatch: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Nonfinite model parameter: {name}")
    return state_hash(state)


def check_optimizer(optimizer, state, names, step, lr):
    validate_optimizer(optimizer, step, lr)
    groups = optimizer["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(names))):
        raise ValueError("AdamW parameter order/count differs from the declared update scope")
    if tuple(groups[0]["betas"]) != (0.9, 0.999) or groups[0]["eps"] != 1e-8:
        raise ValueError("AdamW hyperparameters changed")
    for index, name in enumerate(names):
        moments = optimizer["state"][index]
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = moments[key]
            if tensor.shape != state[name].shape or tensor.dtype != torch.float32:
                raise ValueError(f"AdamW moment shape/dtype mismatch: {name}/{key}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Nonfinite AdamW moment: {name}/{key}")
        if (moments["exp_avg_sq"] < 0).any().item():
            raise ValueError("AdamW second moments must be nonnegative")
    return tree_hash(optimizer)


def check_rng(saved):
    for key in ("torch_rng", "cuda_rng"):
        value = saved[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
            raise ValueError(f"Missing or invalid {key}")
        if value.ndim != 1 or not value.numel():
            raise ValueError(f"Invalid {key} shape")


def audit_run(job, root, frozen_sources):
    run = Path(job["output"])
    records, sources = {}, {}
    config = read_json(run / "config.json", sources)
    config_sha = sources[str(run / "config.json")]
    if config["sources"] != frozen_sources:
        raise ValueError("Worker scientific sources changed")
    if any(config[k] != job[k] for k in ("world", "seed", "condition")):
        raise ValueError("Worker identity differs from launch contract")
    if config["study"]["width"] != job["width"] or config["study"]["lr"] != job["lr"]:
        raise ValueError("Worker continuation branch/width differs")
    done = read_json(run / "complete.json", sources)
    learned = read_json(run / "learning-complete.json", sources)
    timeline = read_json(run / "learning.json", sources)
    if done["status"] != "complete" or done["learning_steps"] != 30720 or done["edit_cases"] != 4:
        raise ValueError("Worker completion marker differs from fixed P2 budget")
    if [r["step"] for r in timeline] != list(CHECKPOINTS):
        raise ValueError("Learning checkpoint grid changed")
    with torch.device("meta"):
        architecture = CausalLM(ModelConfig(**config["model"]))
    shapes = {name: tuple(value.shape) for name, value in architecture.state_dict().items()}
    parameter_names = [name for name, _ in architecture.named_parameters()]
    edit_names = [
        name
        for name in parameter_names
        if any(name.startswith(f"blocks.{layer}.mlp.") for layer in (3, 4, 5))
    ]
    expected_files = {f"model-{step}.pt" for step in CHECKPOINTS} | {"resume.pt"}
    cases = [
        f"{chain}-{kind}-mlp"
        for chain in ("company", "project")
        for kind in ("coherent", "exception")
    ]
    expected_files |= {
        f"edits/{case}/{name}" for case in cases for name in ("model-final.pt", "resume.pt")
    }
    if {str(p.relative_to(run)) for p in run.rglob("*.pt")} != expected_files:
        raise ValueError("Worker checkpoint inventory has missing or unexpected files")
    learning_hashes = {}
    for step in CHECKPOINTS:
        checkpoint = load_checkpoint(run / f"model-{step}.pt", records, root)
        if checkpoint["step"] != step or checkpoint["config"] != config["model"]:
            raise ValueError("Learning snapshot step/config identity mismatch")
        learning_hashes[str(step)] = check_model(checkpoint["model"], shapes)
        if step == 15360 and learning_hashes[str(step)] != config["parent"]["model_sha256"]:
            raise ValueError("Continuation origin differs from locked parent model state")
        if step == 30720:
            baseline = checkpoint["model"]
    if learned["status"] != "complete" or learned["model_sha256"] != learning_hashes["30720"]:
        raise ValueError("Final learning model differs from completion-state hash")
    saved = load_checkpoint(run / "resume.pt", records, root)
    if saved["step"] != 30720 or saved["config_sha256"] != config_sha:
        raise ValueError("Learning resume step/config mismatch")
    if check_model(saved["model"], shapes) != learning_hashes["30720"]:
        raise ValueError("Learning resume and final model disagree")
    if saved["timeline"] != timeline or learned["final"] != timeline[-1]:
        raise ValueError("Learning resume timeline/completion mismatch")
    learning_optimizer = check_optimizer(
        saved["optimizer"], saved["model"], parameter_names, 30720, job["lr"]
    )
    check_rng(saved)
    prediction_path = run / "predictions-30720.npz"
    sources[str(prediction_path)] = digest(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as arrays:
        for name, stored in (("counts", "exposure"), ("slots", "slots"), ("weighted", "weighted")):
            np.testing.assert_array_equal(saved[name], arrays[stored])
    del saved
    edit_checks = []
    for case in cases:
        folder = run / "edits" / case
        chain, kind, scope = case.split("-")
        complete = read_json(folder / "complete.json", sources)
        trajectory = read_json(folder / "trajectory.json", sources)
        if (complete["status"], complete["chain"], complete["kind"], complete["scope"]) != (
            "complete",
            chain,
            kind,
            scope,
        ) or complete["final"] != trajectory[-1]:
            raise ValueError("Edit completion identity or final metrics changed")
        if [r["step"] for r in trajectory] != [0, 32, 128, 512]:
            raise ValueError("Edit checkpoint grid changed")
        final = load_checkpoint(folder / "model-final.pt", records, root)
        if final["config"] != config["model"]:
            raise ValueError("Edit final architecture mismatch")
        final_sha = check_model(final["model"], shapes)
        saved = load_checkpoint(folder / "resume.pt", records, root)
        if saved["step"] != 512 or saved["config_sha256"] != config_sha or saved["case"] != case:
            raise ValueError("Edit resume step/config/case mismatch")
        if saved["timeline"] != trajectory or check_model(saved["model"], shapes) != final_sha:
            raise ValueError("Edit resume does not match final model or trajectory")
        optimizer_sha = check_optimizer(saved["optimizer"], saved["model"], edit_names, 512, 3e-5)
        check_rng(saved)
        unchanged = [name for name in parameter_names if name not in edit_names]
        if any(not torch.equal(baseline[name], final["model"][name]) for name in unchanged):
            raise ValueError("Edit changed parameters outside MLP layers 3–5")
        edit_checks.append(
            {
                "case": case,
                "model_state_sha256": final_sha,
                "optimizer_state_sha256": optimizer_sha,
                "step": 512,
                "updated_parameter_tensors": len(edit_names),
                "frozen_parameter_tensors_equal": len(unchanged),
            }
        )
        del final, saved
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f"Small source changed during audit: {path}")
    for relative, record in records.items():
        stat = (root / relative).stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"Weight source changed after reading: {relative}")
    return {
        "id": job["id"],
        "learning_state_sha256": learning_hashes,
        "learning_optimizer_sha256": learning_optimizer,
        "edits": edit_checks,
        "files": records,
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, output = args.root.resolve(), args.output.resolve()
    launch_path = root / "launch-contract.json"
    launch = json.loads(launch_path.read_text())
    launch_sha = digest(launch_path)
    if launch["sources"] != continuation_sources() or len(launch["jobs"]) != 36:
        raise ValueError("Frozen P2 launch sources or matrix mismatch")
    if len({job["id"] for job in launch["jobs"]}) != 36:
        raise ValueError("P2 launch jobs are duplicated")
    output.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(audit_run, job, root, launch["sources"]): job for job in launch["jobs"]
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"job": futures[future]["id"], "error": repr(exc)})
            print(json.dumps({"verified_runs": len(results), "errors": errors}), flush=True)
    if digest(launch_path) != launch_sha:
        errors.append({"error": "Launch contract changed during audit"})
    results.sort(key=lambda result: result["id"])
    checksums = {
        relative: record for result in results for relative, record in result["files"].items()
    }
    complete = not errors and len(results) == 36 and len(checksums) == 468
    audit = {
        "complete": complete,
        "learning_runs": len(results),
        "learning_snapshots": 4 * len(results),
        "learning_resumes": len(results),
        "edit_cases": sum(len(result["edits"]) for result in results),
        "checkpoint_files_verified": len(checksums),
        "checkpoint_bytes_read_and_hashed": sum(record["bytes"] for record in checksums.values()),
        "launch_sha256": launch_sha,
        "source_sha256": digest(__file__),
        "frozen_training_sources": launch["sources"],
        "errors": errors,
        "storage": "Original durable shared filesystem; no duplicate tar required",
        "method": "Each serialized file read once for SHA256 and CPU load; all model "
        "and optimizer tensors validated; intermediate snapshots retain source/step "
        "identity but are not rerun; final/resume state equality and frozen edit scope "
        "verified. Small sources rehashed at end; weights guarded by size/mtime.",
        "model_predictions_rerun": False,
        "new_worlds_read": False,
    }
    for name, value in [
        ("weight-checksums.json", checksums),
        ("weight-run-checks.json", results),
        ("weight-audit.json", audit),
    ]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit), flush=True)
    if not complete:
        raise RuntimeError("P2 checkpoint audit incomplete; see errors")


if __name__ == "__main__":
    main()
