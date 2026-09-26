"""Execute the frozen A/B/C study without modifying historical experiments."""

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_data import N_QUERIES, array_hash, load_world, write_json


def read_json(path, kind=dict):
    path = Path(path)
    try:

        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON constant: {value}")

        value = json.loads(path.read_text(), parse_constant=reject_constant)
        if not isinstance(value, kind):
            raise ValueError(f"Expected {kind.__name__}")
        return value
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Unreadable or malformed artifact {path}: {error}") from error


def require_fields(actual, expected, label):
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f"{label}: mismatched or missing {key}")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(directory, names):
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty artifact: {path}")


def validate_study_config(config):
    """Reject scientific knobs that the current executable does not implement."""
    from llm_memory_editability.bios_edit import EDIT_CHECKPOINTS
    from llm_memory_editability.bios_organization_train import checkpoint_steps

    require_fields(
        config,
        {
            "protocol": "v2.4",
            "dataset": "bioS-Work-Organization-v1",
            "conditions": {"A": "linked", "B": "entity", "C": "shuffled"},
            "documents_per_epoch": 2048,
            "facts_per_document": 7,
            "document_batch_size": 16,
            "derived_batch_size": 28,
            "atomic_presentations_per_epoch": 14336,
            "default_fact_repetitions_per_epoch": 32,
            "person_fact_repetitions_per_epoch": 1,
            "optimizer": "AdamW",
            "weight_decay": 0.1,
            "warmup_epochs": 16,
            "gradient_clip": 1.0,
            "derived_training": "all old-world derived queries; identical isolated stream "
            "in every condition",
            "loss": "0.8 * mean document answer/EOS CE + 0.2 * mean isolated derived answer/EOS CE",
            "learning_rate_schedule": "constant within each complete epoch; "
            "lr * min((epoch + 1) / 16, 1)",
        },
        "Study configuration",
    )
    require_fields(
        config.get("model", {}),
        {
            "layers": 8,
            "width": 768,
            "heads": 12,
            "mlp_width": 3072,
            "context": 128,
        },
        "Study model (not configurable through runner)",
    )
    for field in ("world_seeds", "initialization_seeds"):
        values = config.get(field)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"Study {field} must be nonempty and unique")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"Study {field} must contain nonnegative integer seeds")
    if set(config.get("conditions", {})) != {"A", "B", "C"}:
        raise ValueError("Study conditions must be A, B and C")
    if type(config.get("organization_seed")) is not int or config["organization_seed"] < 0:
        raise ValueError("Invalid organization seed")
    steps = config.get("world_steps")
    if type(steps) is not int or steps <= 0 or steps % 896:
        raise ValueError("Training budget must contain complete seven-epoch cycles")
    require_fields(
        config,
        {"epochs": steps / 128, "learning_checkpoints": checkpoint_steps(steps)},
        "Study budget",
    )
    editing = config.get("editing", {})
    require_fields(
        editing,
        {
            "k": 1,
            "types": ["coherent", "exception"],
            "selection": "prespecified unmatched balanced targets",
        },
        "Study edits",
    )
    supports, scopes = editing.get("support_indices", []), editing.get("scopes", [])
    if (
        not supports
        or len(set(supports)) != len(supports)
        or any(type(s) is not int or s < 0 for s in supports)
    ):
        raise ValueError("Edit supports must be unique nonnegative integers")
    if not scopes or len(set(scopes)) != len(scopes) or not set(scopes) <= {"mlp", "all"}:
        raise ValueError("Unsupported edit scopes")
    window = editing.get("window_zero_based", [])
    if (
        len(window) != 3
        or type(window[0]) is not int
        or window[0] < 0
        or window != list(range(window[0], window[0] + 3))
        or window[-1] >= 8
    ):
        raise ValueError("Edit window must identify three consecutive available layers")
    edit_steps = editing.get("max_steps")
    if type(edit_steps) is not int or edit_steps < 1:
        raise ValueError("Invalid edit budget")
    require_fields(
        editing,
        {
            "checkpoints": sorted(
                {step for step in EDIT_CHECKPOINTS if step <= edit_steps} | {edit_steps}
            )
        },
        "Study edit checkpoints",
    )
    for value in (config.get("learning_rate"), editing.get("learning_rate")):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Learning rates must be positive and finite")
    retention = editing.get("retention_kl_weight")
    if not isinstance(retention, (int, float)) or not math.isfinite(retention) or retention < 0:
        raise ValueError("Retention weight must be finite and nonnegative")
    require_fields(
        config.get("organization_measurement", {}),
        {
            "steps": [steps],
            "all_layers": True,
            "full_answer_behavior": True,
        },
        "Study measurement (endpoint full behavior only)",
    )
    runs = len(config["world_seeds"]) * len(config["initialization_seeds"]) * 3
    require_fields(
        config.get("expected", {}),
        {
            "learning_runs": runs,
            "editing_cases": runs * len(supports) * len(scopes) * 2,
            "endpoint_measurements": runs,
        },
        "Study expected counts",
    )


def selected_jobs(config, worlds=None, seeds=None, conditions=None):
    selections = []
    for requested, allowed, label in (
        (worlds, config["world_seeds"], "worlds"),
        (seeds, config["initialization_seeds"], "seeds"),
        (conditions, config["conditions"], "conditions"),
    ):
        values = list(allowed) if requested is None else requested
        if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
            raise ValueError(f"Requested {label} must be a unique subset of the frozen study")
        selections.append(values)
    worlds, seeds, conditions = selections
    return [
        (world, seed, condition) for seed in seeds for world in worlds for condition in conditions
    ]


def validate_run_config(run, study, identity, world_path):
    from llm_memory_editability.bios_organization import make_documents
    from llm_memory_editability.bios_organization_train import (
        core_sources,
        derived_schedule,
        schedule_hash,
    )

    world_id, seed, condition = identity
    world = load_world(world_path)
    if world.seed != world_id:
        raise ValueError("World path metadata does not match scheduled world")
    config = read_json(run / "config.json")
    steps = study["world_steps"]
    model = {key: study["model"][key] for key in ("width", "layers", "heads", "context")}
    model["vocab_size"] = world.vocab_size
    d = model["width"]
    parameters = (
        (world.vocab_size + model["context"]) * d + model["layers"] * (12 * d * d + 13 * d) + 2 * d
    )
    documents = make_documents(world, condition, study["organization_seed"])
    derived = derived_schedule(world.seed, steps)
    expected = {
        "protocol": study["protocol"],
        "dataset": "bios-organization-v1",
        "phase": "development",
        "world_seed": world_id,
        "seed": seed,
        "condition": condition,
        "organization_seed": study["organization_seed"],
        "steps": steps,
        "model": model,
        "parameters": parameters,
        "lr": study["learning_rate"],
        "optimizer": "AdamW",
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 16,
        "warmup_steps": 2048,
        "steps_per_epoch": 128,
        "epochs": steps / 128,
        "documents_per_step": 16,
        "facts_per_document": 7,
        "derived_queries_per_step": 28,
        "document_weight": 0.8,
        "derived_weight": 0.2,
        "cross_document_attention": False,
        "document_shape": {"sequence": 42, "supervised_positions": 14},
        "derived_shape": {"sequence": 6, "supervised_positions": 2},
        "checkpoints": study["learning_checkpoints"],
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "lengths_sha256": array_hash(world.lengths),
        "documents_sha256": array_hash(documents),
        "document_schedule_sha256": schedule_hash(documents, world.seed, steps // 128),
        "derived_schedule_sha256": array_hash(derived),
        "planned_atomic_presentations": steps * 112,
        "planned_derived_presentations": steps * 28,
        "planned_total_presentations": steps * 140,
        "planned_document_presentations": steps * 16,
        "planned_input_tokens_including_padding": steps * 840,
        "planned_supervised_tokens": steps * 280,
        "core_source_sha256": core_sources(),
    }
    require_fields(config, expected, str(run / "config.json"))
    for field, path in (("world", world_path), ("output", run)):
        if not config.get(field) or Path(config[field]).resolve() != path.resolve():
            raise ValueError(f"Run configuration {field} path mismatch")
    with np.load(run / "schedule.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["documents"], documents) or not np.array_equal(
            saved["derived"], derived
        ):
            raise ValueError("Saved schedule differs from frozen study")
    return config


def checkpoint_facts(path, config, step):
    import torch

    from llm_memory_editability.bios_organization_train import state_hash

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require_fields(checkpoint, {"step": step, "config": config["model"]}, str(path))
    if sum(value.numel() for value in checkpoint["model"].values()) != config["parameters"]:
        raise ValueError(f"Checkpoint parameter count mismatch: {path}")
    return {"sha256": file_hash(path), "state_sha256": state_hash(checkpoint["model"])}


def validate_learning(run, study, identity, world_path):
    config = validate_run_config(run, study, identity, world_path)
    steps = study["world_steps"]
    marker = read_json(run / "complete.json")
    require_fields(marker, {"status": "complete", "step": steps}, "Learning completion")
    trajectory = read_json(run / "learning.json", list)
    if [point.get("step") for point in trajectory] != study["learning_checkpoints"]:
        raise ValueError("Learning trajectory has a missing or unexpected checkpoint")
    if marker.get("final") != trajectory[-1]:
        raise ValueError("Learning completion final differs from recorded trajectory")
    require_fields(
        trajectory[-1],
        {
            "step": steps,
            "presentations": steps * 140,
            "base_presentations": steps * 112,
            "derived_presentations": steps * 28,
            "supervised_tokens": steps * 280,
            "input_tokens_including_padding": steps * 840,
        },
        "Learning endpoint budget",
    )
    for step in study["learning_checkpoints"]:
        require_files(run, [f"model-{step}.pt", f"predictions-{step}.npz"])
    for field in ("exposure_sha256", "slot_exposure_sha256"):
        if not marker.get(field) or marker[field] != trajectory[-1].get(field):
            raise ValueError(f"Learning completion {field} mismatch")
    initial = checkpoint_facts(run / "model-0.pt", config, 0)
    final = checkpoint_facts(run / f"model-{steps}.pt", config, steps)
    if initial["state_sha256"] != config.get("model_initial_sha256"):
        raise ValueError("Initial checkpoint hash mismatch")
    if final["state_sha256"] != marker.get("model_final_sha256"):
        raise ValueError("Final checkpoint hash mismatch")
    return final


def validate_editing(run, study, world_path, allow_partial=False):
    directory = run / "edits"
    edit = study["editing"]
    marker_path = directory / "complete.json"
    marker = read_json(marker_path) if marker_path.exists() else None
    if marker is None and not allow_partial:
        raise ValueError(f"Missing editing completion: {marker_path}")
    paths = [path for path in directory.glob("support-*") if path.is_dir()]
    if marker is None and not paths and not (directory / "run.json").exists():
        return
    metadata = read_json(directory / "run.json")
    if marker is not None:
        require_fields(marker, {"status": "complete"}, "Editing completion")
    require_fields(
        metadata,
        {
            "supports": edit["support_indices"],
            "scopes": edit["scopes"],
            "steps": edit["max_steps"],
            "window": edit["window_zero_based"][0],
            "lr": edit["learning_rate"],
            "retention": edit["retention_kl_weight"],
            "branches": False,
            "world_step": study["world_steps"],
        },
        "Editing run metadata",
    )
    for field, path in (
        ("world", world_path),
        ("output", directory),
        ("checkpoint", run / f"model-{study['world_steps']}.pt"),
    ):
        if not metadata.get(field) or Path(metadata[field]).resolve() != path.resolve():
            raise ValueError(f"Editing {field} path mismatch")
    expected = {
        (support, kind, scope)
        for support in edit["support_indices"]
        for kind in edit["types"]
        for scope in edit["scopes"]
    }
    cases = (
        marker.get("cases", [])
        if marker is not None
        else [
            read_json(path / "complete.json") for path in paths if (path / "complete.json").exists()
        ]
    )
    keys = [(case.get("support"), case.get("kind"), case.get("scope")) for case in cases]
    if (
        len(keys) != len(set(keys))
        or not set(keys) <= expected
        or (marker is not None and set(keys) != expected)
    ):
        raise ValueError("Editing completion has missing, duplicated or unexpected cases")
    names = {f"support-{support}-{kind}-{scope}" for support, kind, scope in expected}
    actual_names = {path.name for path in paths}
    if not actual_names <= names or (marker is not None and actual_names != names):
        raise ValueError("Editing case directories differ from the frozen case set")
    with np.load(run / f"predictions-{study['world_steps']}.npz", allow_pickle=False) as old:
        old_arrays = {key: old[key] for key in ("prediction", "ended", "correct")}
    for summary in cases:
        support, kind, scope = summary["support"], summary["kind"], summary["scope"]
        case = directory / f"support-{support}-{kind}-{scope}"
        result = read_json(case / "complete.json")
        if result != summary:
            raise ValueError(f"Editing aggregate disagrees with case marker: {case}")
        expected_fields = {
            "support": support,
            "kind": kind,
            "scope": scope,
            "branch": None,
            "branch_parameters": 0,
            "steps": edit["max_steps"],
            "lr": edit["learning_rate"],
            "retention_kl_weight": edit["retention_kl_weight"],
            "window_zero_based": edit["window_zero_based"],
            "selection": "prespecified_unmatched",
        }
        require_fields(result, {**expected_fields, "status": "complete"}, str(case))
        require_fields(read_json(case / "config.json"), expected_fields, str(case / "config.json"))
        points = read_json(case / "trajectory.json", list)
        if [point.get("step") for point in points] != edit["checkpoints"] or result.get(
            "final"
        ) != points[-1]:
            raise ValueError(f"Editing trajectory/completion budget mismatch: {case}")
        require_files(case, ["sets.npz", *[f"predictions-{s}.npz" for s in edit["checkpoints"]]])
        with np.load(case / "sets.npz", allow_pickle=False) as sets:
            if (
                len(sets["E"]) != 93
                or len(sets["D"]) != 96
                or sets["target"].shape != (N_QUERIES,)
                or not np.array_equal(sets["old_correct"], old_arrays["correct"])
            ):
                raise ValueError(f"Editing target or baseline mismatch: {case}")
        with np.load(case / "predictions-0.npz", allow_pickle=False) as initial:
            if any(
                not np.array_equal(initial[key], old_arrays[key]) for key in ("prediction", "ended")
            ):
                raise ValueError(f"Editing step-zero predictions differ from endpoint: {case}")


def validate_measurement(run, study, checkpoint_sha256):
    directory = run / "organization-final"
    marker = read_json(directory / "measurements.json")
    require_fields(
        marker,
        {"step": study["world_steps"], "behavior": True, "checkpoint_sha256": checkpoint_sha256},
        "Measurement completion",
    )
    checkpoint = run / f"model-{study['world_steps']}.pt"
    if not marker.get("checkpoint") or Path(marker["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Measurement checkpoint path mismatch")
    layers = set(range(study["model"]["layers"]))
    geometry = marker.get("geometry", [])
    if len(geometry) != len(layers) or {row.get("layer") for row in geometry} != layers:
        raise ValueError("Measurement geometry does not cover all layers")
    probes = marker.get("probes", [])
    kinds = {"donor", "same_answer_donor", "wrong_donor", "random_equal_norm", "ablation"}
    recipients = {row.get("recipient") for row in probes}
    keys = [(row.get("recipient"), row.get("layer"), row.get("intervention")) for row in probes]
    expected = {
        (recipient, layer, kind) for recipient in recipients for layer in layers for kind in kinds
    }
    if len(recipients) != 8 or len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Measurement probes are missing, duplicated or unexpected")
    groups = {
        "same_company_nonexception",
        "old_exception",
        "independent_attribute",
        "unrelated_company",
    }
    if any(set(row.get("generation", {})) != groups for row in probes):
        raise ValueError("Measurement lacks full-answer behavior")
    require_files(
        directory, ["actual-city-activations.npz", "input-activation-output-position-sample.npz"]
    )


def active_processes(run):
    """Detect existing legacy workers, which predate the orchestration lock."""
    modules = {
        "llm_memory_editability.bios_organization_train",
        "llm_memory_editability.bios_edit",
        "llm_memory_editability.bios_measure",
    }
    found = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            arguments = process.joinpath("cmdline").read_bytes().decode().split("\0")
            if not modules.intersection(arguments) or "--output" not in arguments:
                continue
            output = Path(arguments[arguments.index("--output") + 1])
            if not output.is_absolute():
                output = process.joinpath("cwd").resolve() / output
            if output.resolve() in {
                run.resolve(),
                (run / "edits").resolve(),
                (run / "organization-final").resolve(),
            }:
                found.append(int(process.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU indices or stable UUIDs")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--root", default="results/bios-organization-dev-v1")
    parser.add_argument("--world-root", default="data/bios-organization-v1")
    parser.add_argument("--config", default="configs/bios-organization-development-v1.json")
    parser.add_argument(
        "--phase", choices=["all", "learning", "editing", "measurement"], default="all"
    )
    parser.add_argument("--worlds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--conditions", nargs="+", choices=["A", "B", "C"])
    args = parser.parse_args()
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be positive")
    config = read_json(args.config)
    validate_study_config(config)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "study-config.json"
    with (root / ".study.lock").open("a") as study_lock:
        fcntl.flock(study_lock.fileno(), fcntl.LOCK_EX)
        if frozen.exists():
            if read_json(frozen) != config:
                raise ValueError("Study configuration changed; use a new output root")
        else:
            write_json(frozen, config)
    jobs = queue.Queue()
    for identity in selected_jobs(config, args.worlds, args.seeds, args.conditions):
        jobs.put(identity)
    planned = list(jobs.queue)
    records, lock, failed = [], threading.Lock(), threading.Event()
    phases = ["learning", "editing", "measurement"] if args.phase == "all" else [args.phase]
    invocation = now().replace(":", "-") + f"-pid-{os.getpid()}"
    ledger_path = root / f"execution-{args.phase}-{invocation}.json"

    def save():
        write_json(
            ledger_path,
            {
                "started_utc": invocation,
                "planned": planned,
                "arguments": vars(args),
                "records": records,
            },
        )

    save()

    def run_job(gpu, identity, run):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4"}
        world, seed, condition = identity
        name = run.name
        world_path = Path(args.world_root) / f"world-{world}"
        checkpoint = run / f"model-{config['world_steps']}.pt"
        endpoint = None

        def verify(phase):
            nonlocal endpoint
            if endpoint is None:
                endpoint = validate_learning(run, config, identity, world_path)
            if phase == "editing":
                validate_editing(run, config, world_path)
            elif phase == "measurement":
                validate_measurement(run, config, endpoint["sha256"])

        for phase in phases:
            if failed.is_set():
                return
            common = ["--world", str(world_path), "--device", "cuda"]
            if phase == "learning":
                complete = run / "complete.json"
                command = [
                    sys.executable,
                    "-m",
                    "llm_memory_editability.bios_organization_train",
                    *common,
                    "--output",
                    str(run),
                    "--condition",
                    condition,
                    "--seed",
                    str(seed),
                    "--organization-seed",
                    str(config["organization_seed"]),
                    "--steps",
                    str(config["world_steps"]),
                    "--lr",
                    str(config["learning_rate"]),
                ]
                if not complete.exists() and (run / "resume.pt").exists():
                    command.append("--resume")
            elif phase == "editing":
                complete = run / "edits" / "complete.json"
                command = [
                    sys.executable,
                    "-m",
                    "llm_memory_editability.bios_edit",
                    *common,
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(run / "edits"),
                    "--supports",
                    *map(str, config["editing"]["support_indices"]),
                    "--scopes",
                    *config["editing"]["scopes"],
                    "--steps",
                    str(config["editing"]["max_steps"]),
                    "--lr",
                    str(config["editing"]["learning_rate"]),
                    "--retention",
                    str(config["editing"]["retention_kl_weight"]),
                    "--window",
                    str(config["editing"]["window_zero_based"][0]),
                ]
            else:
                complete = run / "organization-final" / "measurements.json"
                command = [
                    sys.executable,
                    "-m",
                    "llm_memory_editability.bios_measure",
                    *common,
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(run / "organization-final"),
                    "--behavior",
                ]
            if complete.exists():
                verify(phase)
                with lock:
                    records.append(
                        {
                            "run": name,
                            "phase": phase,
                            "status": "already_complete",
                            "completion_sha256": file_hash(complete),
                            "validation": "frozen contract and artifacts checked",
                        }
                    )
                    save()
                continue
            if phase == "learning":
                if (run / "config.json").exists():
                    validate_run_config(run, config, identity, world_path)
            else:
                verify("learning")
                if phase == "editing":
                    # The editor internally skips completed individual cases. Audit
                    # partial results too, before allowing that implicit reuse.
                    validate_editing(run, config, world_path, allow_partial=True)
            # Validation can take seconds while reading endpoint weights. Recheck
            # legacy workers immediately before dispatch, including between phases.
            active = active_processes(run)
            if active:
                with lock:
                    records.append(
                        {
                            "run": name,
                            "phase": "coordination",
                            "status": "already_running",
                            "pids": active,
                        }
                    )
                    save()
                return
            record = {
                "run": name,
                "phase": phase,
                "gpu": gpu,
                "start_utc": now(),
                "command": command,
                "status": "running",
            }
            with lock:
                records.append(record)
                save()
                print(json.dumps(record), flush=True)
            with (run / f"{phase}.log").open("a") as stream:
                stream.write(f"\nInvocation {invocation}\n")
                stream.flush()
                result = subprocess.run(
                    command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            with lock:
                record.update(end_utc=now(), returncode=result.returncode)
            if result.returncode:
                with lock:
                    record["status"] = "failed"
                    failed.set()
                    save()
                return
            # A successful process exit alone does not certify a complete study phase.
            verify(phase)
            with lock:
                record.update(
                    status="complete",
                    completion_sha256=file_hash(complete),
                    validation="frozen contract and artifacts checked",
                )
                save()
                print(json.dumps(record), flush=True)

    def worker(gpu):
        while not failed.is_set():
            try:
                identity = jobs.get_nowait()
            except queue.Empty:
                return
            world, seed, condition = identity
            run = root / f"world-{world}-seed-{seed}-{condition}"
            run.mkdir(parents=True, exist_ok=True)
            try:
                with (run / ".runner.lock").open("a") as run_lock:
                    try:
                        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        busy = True
                    else:
                        busy = bool(active_processes(run))
                        with (run / ".train.lock").open("a") as train_lock:
                            try:
                                fcntl.flock(train_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                busy = True
                            else:
                                fcntl.flock(train_lock.fileno(), fcntl.LOCK_UN)
                    if busy:
                        with lock:
                            records.append(
                                {
                                    "run": run.name,
                                    "phase": "coordination",
                                    "status": "already_running",
                                }
                            )
                            save()
                        continue
                    run_job(gpu, identity, run)
            except Exception as error:
                with lock:
                    for record in records:
                        if record.get("run") == run.name and record.get("status") == "running":
                            record.update(status="failed", reason=str(error), end_utc=now())
                    records.append(
                        {
                            "run": run.name,
                            "phase": "validation_or_execution",
                            "status": "failed",
                            "reason": str(error),
                            "end_utc": now(),
                        }
                    )
                    failed.set()
                    save()
                    print(json.dumps(records[-1]), flush=True)
                return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(args.gpus) * args.workers_per_gpu
    ) as pool:
        futures = [
            pool.submit(worker, gpu) for gpu in args.gpus for _ in range(args.workers_per_gpu)
        ]
        for future in futures:
            future.result()
    if failed.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
