#!/usr/bin/env python3
"""Run the authorized depth/hop matrix using the existing full-GPT runner.

The imported runner is frozen with this file. Native-horizon training is
unchanged; endpoint sweeps use independent eager forwards, never feedback of
bridge entities, and never change the saved training horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_hop as native  # noqa: E402

from llm_memory_editability.experiment_tracking import learning_metrics  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402
from llm_memory_editability.loop_hop_data import load, panel_names  # noqa: E402


def sweep(model, world, panels, metadata, device, batch_size):
    original_repeats, original_mode = model.repeats, model.training
    before = model_digest(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    metrics, predictions = {}, {}
    try:
        for repeats in range(1, 5):
            model.repeats = repeats
            metrics[str(repeats)], predictions[str(repeats)] = native.measure_panels(
                model, world, panels, metadata, device, batch_size
            )
    finally:
        model.repeats = original_repeats
        model.train(original_mode)
    if model_digest(model) != before or not torch.equal(cpu_rng, torch.get_rng_state()):
        raise AssertionError("Endpoint evaluation changed weights or CPU RNG")
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise AssertionError("Endpoint evaluation consumed training CUDA RNG")
    # Common-premise subset across inference budgets, in addition to all rows.
    atomic = []
    for r in predictions:
        atomic.append(
            np.concatenate(
                [
                    predictions[r][p]["answer"] == panels[p][:, -1] + metadata["entity_offset"]
                    for p in ("atomic_id", "atomic_ood")
                ]
            )
        )
    common = np.logical_and.reduce(atomic)
    for r in metrics:
        for p in panel_names()[2:]:
            conditional = native.premise_metrics(
                panels[p], predictions[r][p], common, world, metadata
            )
            metrics[r][p].update(
                common_premise_n=conditional["premise_correct_n"],
                common_premise_coverage=conditional["premise_coverage"],
                answer_accuracy_given_common_premises=conditional["answer_accuracy_given_premises"],
            )
    return metrics, predictions


def endpoint(config, name, base, config_path, smoke=False, audit=False):
    out = base / ("smoke" if smoke else "runs") / name
    complete = json.loads((out / "complete.json").read_text())
    saved_path = out / f"checkpoint-{complete['step']:07d}.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    world, panels, metadata = load(config["input_root"])
    model, spec, arm, initial = native.initialize(config, name, metadata, torch.device("cuda:0"))
    if (
        saved["config_sha256"] != digest(config_path)
        or saved["world_sha256"] != metadata["world_sha256"]
        or saved["spec"] != {**spec, "steps": complete["budget"]}
        or saved["arm"] != arm
        or saved["step"] != complete["step"]
        or digest(saved_path) != complete["final_checkpoint_sha256"]
    ):
        raise ValueError("Endpoint checkpoint identity mismatch")
    model.load_state_dict(saved["model"], strict=True)
    if model_digest(model) != complete["model_sha256"]:
        raise ValueError("Endpoint parameter hash mismatch")
    metrics, predictions = sweep(
        model, world, panels, metadata, torch.device("cuda:0"), config["evaluation_batch_size"]
    )
    destination = out / ("grid-audit.json" if audit else "grid.json")
    if destination.exists():
        raise FileExistsError(destination)
    if audit:
        original = json.loads((out / "grid.json").read_text())
        findings = {}
        for r in metrics:
            for p in panel_names():
                with np.load(out / "grid" / f"r{r}" / f"{p}.npz") as previous:
                    ids = all(
                        np.array_equal(previous[k], predictions[r][p][k])
                        for k in ("rows", "answer", "stop")
                    )
                    nll = np.allclose(
                        previous["nll"], predictions[r][p]["nll"], atol=1e-6, rtol=1e-6
                    )
                previous_metrics = original["metrics"][r][p]
                scalars = all(
                    previous_metrics[k] is None
                    if v is None
                    else previous_metrics[k] is not None
                    and np.isclose(previous_metrics[k], v, atol=1e-6, rtol=1e-6)
                    for k, v in metrics[r][p].items()
                )
                findings[f"r{r}/{p}"] = bool(ids and nll and scalars)
        result = {
            "passed": all(findings.values()),
            "findings": findings,
            "model_sha256": model_digest(model),
            "grid_sha256": digest(out / "grid.json"),
            "fresh_process": os.getpid(),
            "utc": utc(),
        }
        write_json(destination, result)
        if not result["passed"]:
            raise AssertionError("Independent 1-4 loop endpoint replay failed")
    else:
        for r in predictions:
            folder = out / "grid" / f"r{r}"
            folder.mkdir(parents=True, exist_ok=False)
            for p, values in predictions[r].items():
                np.savez_compressed(folder / f"{p}.npz", **values)
        write_json(
            destination,
            {
                "arm": name,
                "training_repeats": spec["repeats"],
                "unique_layers": spec["unique_layers"],
                "step": saved["step"],
                "engineering_only": smoke,
                "world_sha256": metadata["world_sha256"],
                "panels_sha256": metadata["panels_sha256"],
                "config_sha256": digest(config_path),
                "model_sha256": model_digest(model),
                "initial_model_sha256": initial,
                "metrics": metrics,
                "utc": utc(),
            },
        )


def tracked_train(config, name, base, path):
    import wandb

    original_write = native.write_json
    run = None
    previous = None

    def publish(destination, value):
        nonlocal run, previous
        original_write(destination, value)
        if Path(destination).name == "run.json":
            run = wandb.init(
                entity="zhusq20",
                project="llm-memory-editability",
                group=config["batch"],
                name=name,
                id=f"lh{digest(path)[:6]}{list(config['arms']).index(name):02d}",
                config={"experiment": config, "run": value},
                dir=str(Path(destination).parent),
                mode="online",
                save_code=False,
                settings=wandb.Settings(disable_git=True),
            )
            original_write(
                Path(destination).parent / "tracking.json", {"id": run.id, "url": run.url}
            )
        elif Path(destination).name == "learning.json" and run is not None:
            record = value[-1]
            run.log(learning_metrics(record, previous), step=record["step"])
            previous = record
        elif Path(destination).name == "status.json" and run is not None:
            run.summary["training_state"] = value["state"]
            run.summary["scientific_final_step"] = value["step"]

    native.write_json = publish
    try:
        native.train(config, name, base / "runs" / name, False, path, base)
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    finally:
        native.write_json = original_write


def cloud(config, name, base):
    import wandb

    out = base / "runs" / name
    tracking = json.loads((out / "tracking.json").read_text())
    audit = json.loads((out / "grid-audit.json").read_text())
    if not audit["passed"]:
        raise ValueError("Cannot certify a failed independent replay")
    grid = json.loads((out / "grid.json").read_text())
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        id=tracking["id"],
        resume="must",
        dir=str(out),
        mode="online",
        save_code=False,
        settings=wandb.Settings(disable_git=True),
    )
    run.summary.update(
        endpoint_grid=grid,
        independent_grid_audit=audit,
        independently_reloaded=True,
        endpoint_grid_sha256=digest(out / "grid.json"),
    )
    run.finish()
    remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{tracking['id']}")
    passed = (
        remote.summary.get("endpoint_grid") == grid
        and remote.summary.get("independent_grid_audit") == audit
    )
    write_json(out / "cloud-audit.json", {"passed": passed, "url": tracking["url"], "utc": utc()})
    if not passed:
        raise AssertionError("Cloud endpoint JSON does not match local audited result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "smoke",
            "smoke-audit",
            "smoke-grid",
            "smoke-grid-audit",
            "train",
            "audit",
            "grid",
            "grid-audit",
            "cloud",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "cloud":
        cloud(config, args.arm, args.base)
        return
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    smoke = args.stage.startswith("smoke")
    out = args.base / ("smoke" if smoke else "runs") / args.arm
    try:
        if args.stage == "preflight":
            native.preflight(config, args.arm, args.base)
        elif args.stage == "smoke":
            native.train(config, args.arm, out, True, args.config, args.base)
        elif args.stage in ("audit", "smoke-audit"):
            native.audit(config, args.arm, out, args.config)
        elif "grid" in args.stage:
            endpoint(config, args.arm, args.base, args.config, smoke, args.stage.endswith("audit"))
        elif args.stage == "train":
            tracked_train(config, args.arm, args.base, args.config)
    except BaseException as error:
        import traceback

        write_json(
            args.base / "failures" / f"{args.arm}-{args.stage}.json",
            {
                "utc": utc(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
