"""Freeze and supervise the two PERK Figure 3 method reconstruction arms.

Launch requires both CUDA development runs and independent token replay. Each
scientific arm trains once, then evaluates its validation-best checkpoint in two
fresh containers. Failed stages are recorded and never automatically retried.
The resource sidecar records status; native trainers own the W&B learning curves.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import netrc
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path("/ossfs/workspace/research-recovery/perk-paper-reproduction-v1")
DOCKER = "/workspaces/lm-memory-tools/docker/docker"
CREDENTIALS = Path("/ossfs/workspace/.lm-memory-credentials/wandb.netrc")
ARMS = ("perk", "fticr")
STAGES = ("train", "evaluation", "replay")
DATA_REL = "data/prepared-author-qa2-8k-v1/manifest.json"
NETWORK = "host"
RESERVED_CPUS = tuple(range(16, 64))
DEVELOPMENT_SOURCE_NAMES = {"perk": "source-v2", "fticr": "source-v3"}
ARM_SOURCE_FILES = {
    "perk": (
        "src/llm_memory_editability/perk_reproduction.py",
        "scripts/run_perk_reproduction.py",
    ),
    "fticr": (
        "src/llm_memory_editability/perk_fticr.py",
        "src/llm_memory_editability/perk_data.py",
        "scripts/run_perk_fticr.py",
    ),
}
METHOD_SOURCE_FILES = tuple(path for arm in ARMS for path in ARM_SOURCE_FILES[arm])


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source(base):
    return Path(base) / "frozen/source"


def config_path(base, arm):
    return Path(base) / "frozen/configs" / f"{arm}.json"


def docker(arguments, **kwargs):
    return subprocess.run([DOCKER, "--context", "lm-memory", *arguments], **kwargs)


def file_inventory(root):
    return sorted(
        path
        for path in Path(root).rglob("*")
        if path.is_file()
        and not {".git", "__pycache__", ".cache"}.intersection(path.parts)
        and not path.name.endswith((".pyc", ".tmp", ".lock"))
    )


def default_configs(base):
    base = Path(base)
    manifest = read(base / DATA_REL)
    common = {
        "reproduction_kind": "method_reconstruction",
        "seed": 42,
        "model_name_or_path": str(base / "reference/gpt2"),
        "data_manifest": str(base / DATA_REL),
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "patience": 10,
        "wandb_project": "llm-memory-editability",
        "wandb_entity": "zhusq20",
    }
    return {
        "perk": {
            **common,
            "data_dir": manifest["data_dir"],
            "dataset_name": manifest["dataset_name"],
            "dataset_subset": manifest["dataset_subset"],
            "lora_rank": 256,
            "lora_alpha": 16,
            "n_inner_iter": 4,
            "unroll_start": 3,
            "inner_lr": 5e-5,
            "val_check_interval": 0.5,
            "num_train_epochs": 2,
            "packing": False,
            "attn_implementation": "eager",
            "bf16": True,
            "wandb_run_id": "perk-paper-reproduction-v1-perk-train",
            "wandb_name": "perk-paper-reproduction-v1-perk-train",
        },
        "fticr": {
            **common,
            "max_seq_length": 9216,
            "max_position_embeddings": 10240,
            "validation_interval": 4500,
            "gradient_accumulation_steps": 1,
            "preprocessing_num_workers": 4,
            "device": "cuda",
            "wandb_run_id": "perk-paper-reproduction-v1-fticr-train",
            "wandb_name": "perk-paper-reproduction-v1-fticr-train",
        },
    }


def validate_science_configs(configs):
    if set(configs) != set(ARMS):
        raise ValueError("The scientific matrix must contain exactly PERK and FT-ICR")
    for arm, config in configs.items():
        if config.get("reproduction_kind") != "method_reconstruction":
            raise ValueError("Incomplete original recipe must be labelled method reconstruction")
        if any(key.startswith("smoke_") or key.startswith("limit_") for key in config):
            raise ValueError("Development limits cannot enter the scientific matrix")
        if any(word in key.lower() for key in config for word in ("api_key", "password", "secret")):
            raise ValueError("Credentials cannot enter frozen configuration")
        if config["learning_rate"] != 1e-5 or config["weight_decay"] != 0.01:
            raise ValueError("The published standard optimizer settings must be retained")
        if arm == "perk" and (
            config["lora_rank"] != 256
            or config["n_inner_iter"] != 4
            or config["num_train_epochs"] != 2
        ):
            raise ValueError("The published PERK model/adaptation budget must be retained")
        if arm == "fticr" and (
            config["max_seq_length"] != 9216 or config["max_position_embeddings"] != 10240
        ):
            raise ValueError("Frozen APE/sequence lengths changed")


def freeze(base, image, *, supplied_configs=None):
    base = Path(base)
    lock_file = base / "frozen/freeze.json"
    if lock_file.exists() or source(base).exists():
        raise RuntimeError("Existing freeze is immutable")
    if not image or not image.startswith("sha256:") or len(image) != 71:
        raise ValueError("Freeze requires the full immutable Docker image SHA256")
    inspected = json.loads(
        docker(["image", "inspect", image], capture_output=True, text=True, check=True).stdout
    )[0]
    if inspected["Id"] != image:
        raise ValueError("Docker image identity disagrees with requested freeze")
    model_manifest = read(base / "reference/gpt2/source-manifest.json")
    if model_manifest["revision"] != "607a30d783dfa663caf39e06633721c8d4cfcd7e":
        raise ValueError("Unexpected pretrained GPT-2 revision")
    for name, entry in model_manifest["files"].items():
        if sha(base / "reference/gpt2" / name) != entry["sha256"]:
            raise ValueError(f"Prepared GPT-2 file changed: {name}")
    manifest = read(base / DATA_REL)
    if any(manifest["exact_sample_overlap"].values()):
        raise ValueError("Author train/validation/test isolation audit did not pass")
    configs = supplied_configs or default_configs(base)
    validate_science_configs(configs)
    expected_rows = {"train": 9000, "validation": 999, "test": 999}
    for split, expected in expected_rows.items():
        for arm in ARMS:
            row = manifest["outputs"][split][arm]
            if sha(row["path"]) != row["sha256"]:
                raise ValueError(f"Prepared author pool changed: {split}/{arm}")
            if (
                sum(bool(line.strip()) for line in Path(row["path"]).read_text().splitlines())
                != expected
            ):
                raise ValueError("The complete author-derived matrix must be retained")
    names = [
        "scripts/execute_perk_reproduction.py",
        "scripts/run_perk_reproduction.py",
        "scripts/run_perk_fticr.py",
        "scripts/prepare_perk_reproduction.py",
        "tests/test_perk_reproduction.py",
        "tests/test_perk_fticr.py",
        "tests/test_perk_data.py",
        "tests/test_perk_execution.py",
        "configs/experiment-tracking-defaults.json",
        "docs/development-artifacts/perk-paper-reproduction-v1/protocol.md",
        "docs/development-artifacts/perk-paper-reproduction-v1/data-notes.md",
        "docs/development-artifacts/perk-paper-reproduction-v1/method-notes.md",
        "docs/development-artifacts/perk-paper-reproduction-v1/requirements-runtime.txt",
    ]
    files = list((PROJECT / "src/llm_memory_editability").rglob("*.py"))
    files.extend(PROJECT / name for name in names)
    for path in files:
        target = source(base) / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for arm, config in configs.items():
        write(config_path(base, arm), config)
    inputs = file_inventory(base / "reference/gpt2")
    inputs += file_inventory(base / "reference/perk")
    inputs += file_inventory(base / "reference/perk-execution-repaired")
    inputs += file_inventory(base / "reference/open-instruct")
    inputs += [base / DATA_REL]
    inputs += [
        Path(entry["path"]) for split in manifest["outputs"].values() for entry in split.values()
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        input_hashes = dict(pool.map(lambda p: (str(p), sha(p)), sorted(set(inputs))))
    author_commit = subprocess.check_output(
        ["git", "-C", str(base / "reference/perk"), "rev-parse", "HEAD"], text=True
    ).strip()
    if author_commit != "9b1a918368b935d633691839cb360318ff2045f2":
        raise ValueError("Unexpected original PERK source commit")
    write(
        lock_file,
        {
            "batch": "perk-paper-reproduction-v1",
            "reproduction_kind": "method_reconstruction",
            "image_sha256": image,
            "created_utc": utc(),
            "author_perk_commit": author_commit,
            "source_files": {
                str(p.relative_to(source(base))): sha(p) for p in file_inventory(source(base))
            },
            "configs": {arm: sha(config_path(base, arm)) for arm in ARMS},
            "inputs": input_hashes,
            "rows": expected_rows,
            "resources": {
                "gpus": {"perk": 0, "fticr": 1},
                "cpus_per_container": 32,
                "memory_gib": 96,
                "shm_gib": 16,
                "network": NETWORK,
                "reserved_cpus_for_z1": list(RESERVED_CPUS),
            },
            "preflight_receipt": str(base / "development/cuda-validation.json"),
            "development_sources": {
                arm: str(base / "development" / name)
                for arm, name in DEVELOPMENT_SOURCE_NAMES.items()
            },
        },
    )
    return read(lock_file)


def verify_freeze(base, *, inputs=True):
    base = Path(base)
    frozen = read(base / "frozen/freeze.json")
    for name, expected in frozen["source_files"].items():
        if sha(source(base) / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    configs = {}
    for arm, expected in frozen["configs"].items():
        if sha(config_path(base, arm)) != expected:
            raise ValueError(f"Frozen configuration changed: {arm}")
        configs[arm] = read(config_path(base, arm))
    validate_science_configs(configs)
    if inputs:
        for name, expected in frozen["inputs"].items():
            if sha(name) != expected:
                raise ValueError(f"Frozen input changed: {name}")
    return frozen


def prediction_files(root, arm):
    root = Path(root)
    if arm == "perk":
        return (
            root / "independent-evaluation/test-predictions.jsonl",
            root / "independent-evaluation/test-summary.json",
        )
    return root / "test-generations.jsonl", root / "test-metrics.json"


def compare_replay(evaluation, replay, arm, expected_rows):
    a, summary_a = prediction_files(evaluation, arm)
    b, summary_b = prediction_files(replay, arm)
    if a.read_bytes() != b.read_bytes():
        raise ValueError(f"Independent {arm} generated text/token replay differs")
    rows = [json.loads(line) for line in a.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows or len({row["guid"] for row in rows}) != expected_rows:
        raise ValueError("Independent test coverage differs from the frozen pool")
    if any(not isinstance(row.get("generated_token_ids"), list) for row in rows):
        raise ValueError("Each replay record must retain actual generated tokens")
    x, y = read(summary_a), read(summary_b)
    if arm == "perk":
        if (
            any(
                x[key] != y[key]
                for key in (
                    "checkpoint_sha256",
                    "parameter_sha256",
                    "metrics",
                    "mean_outer_loss",
                    "persistent_parameters_unchanged",
                    "evaluation_exposure",
                )
            )
            or x["persistent_parameters_unchanged"] is not True
        ):
            raise ValueError("PERK independent checkpoint/adaptation audit differs")
        if not math.isfinite(x["mean_outer_loss"]):
            raise ValueError("PERK independent test loss is nonfinite")
        metrics = x["metrics"]
    else:
        stable_x = {key: value for key, value in x.items() if key != "elapsed_seconds"}
        stable_y = {key: value for key, value in y.items() if key != "elapsed_seconds"}
        if stable_x != stable_y:
            raise ValueError("FT-ICR independent metrics differ")
        metrics = stable_x
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Nonfinite test metrics")
    return {
        "passed": True,
        "rows": len(rows),
        "predictions_sha256": sha(a),
        "generated_tokens": sum(len(row["generated_token_ids"]) for row in rows),
        "exact_generated_text_and_token_ids": True,
        "metrics": metrics,
    }


def audit_preflight(base):
    base = Path(base)
    frozen = verify_freeze(base)
    method = read(base / "development/perk/train/execution-manifest.json")
    if (
        method["adapter_sha256"]
        != frozen["source_files"]["src/llm_memory_editability/perk_reproduction.py"]
    ):
        raise ValueError("CUDA short-run actually executed a different PERK adapter")
    results, artifacts = {}, {}
    for arm in ARMS:
        development_source = Path(frozen["development_sources"][arm])
        method_hashes = {
            relative: frozen["source_files"][relative] for relative in ARM_SOURCE_FILES[arm]
        }
        for relative, expected in method_hashes.items():
            if sha(development_source / relative) != expected:
                raise ValueError(
                    f"CUDA short-run method differs from frozen science source: {arm}/{relative}"
                )
            artifacts[str(development_source / relative)] = expected
        root = base / "development" / arm
        train_result = read(root / "train/completion.json")
        steps = train_result.get("global_step", train_result.get("optimizer_steps"))
        if steps != 2:
            raise ValueError(f"{arm} CUDA development must execute exactly two optimizer steps")
        progress = read(root / "train/training-progress.json")
        if arm == "perk" and not all(
            math.isfinite(progress[key]) for key in ("inner_loss", "outer_loss")
        ):
            raise ValueError("PERK CUDA training loss is nonfinite")
        validation_rows = (
            train_result.get("validation_exposure", {}).get("problems")
            if arm == "perk"
            else progress.get("validation_samples_exposed")
        )
        if validation_rows != 2:
            raise ValueError(f"{arm} CUDA development must validate exactly two samples")
        for stage in STAGES:
            exit_receipt = read(root / f"{stage}-exit.json")
            if exit_receipt.get("exit_code") != 0 or exit_receipt.get("device") != "cuda":
                raise ValueError(f"Missing successful independent CUDA stage: {arm}/{stage}")
            if exit_receipt.get("image_sha256") != frozen["image_sha256"]:
                raise ValueError("Development image differs from the frozen scientific image")
            if exit_receipt.get("source_root") != str(development_source):
                raise ValueError(f"CUDA stage source root differs: {arm}/{stage}")
            executed_hashes = exit_receipt.get("source_files", {})
            if any(executed_hashes.get(path) != value for path, value in method_hashes.items()):
                raise ValueError(f"CUDA stage source hashes differ: {arm}/{stage}")
            for relative, expected in executed_hashes.items():
                if (
                    relative not in METHOD_SOURCE_FILES
                    or sha(development_source / relative) != expected
                ):
                    raise ValueError(
                        f"CUDA stage source snapshot differs: {arm}/{stage}/{relative}"
                    )
        results[arm] = compare_replay(root / "evaluation", root / "replay", arm, 2)
        for path in (
            root / "train/completion.json",
            root / "train/training-progress.json",
            *(root / f"{stage}-exit.json" for stage in STAGES),
            *prediction_files(root / "evaluation", arm),
            *prediction_files(root / "replay", arm),
        ):
            artifacts[str(path)] = sha(path)
    receipt = {
        "passed": True,
        "freeze_sha256": sha(base / "frozen/freeze.json"),
        "image_sha256": frozen["image_sha256"],
        "arms": results,
        "artifacts": artifacts,
        "created_utc": utc(),
    }
    write(base / "development/cuda-validation.json", receipt)
    return receipt


def parse_cpuset(text):
    result = []
    for part in text.split(",") if text else []:
        limits = part.split("-")
        result.extend(range(int(limits[0]), int(limits[-1]) + 1))
    return result


def resource_check(base):
    frozen = verify_freeze(base, inputs=False)
    values = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    gpu_rows = {int(row.split(",")[0]): row.split(",") for row in values}
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    ).splitlines()
    requested = {0, 1}
    if any(int(gpu_rows[gpu][2]) >= 1000 for gpu in requested) or any(
        line.split(",")[0].strip() in {gpu_rows[gpu][1].strip() for gpu in requested}
        for line in apps
    ):
        raise RuntimeError("GPU 0/1 has an existing compute process or >= 1000 MiB allocated")
    ids = docker(["ps", "-q"], text=True, capture_output=True, check=True).stdout.splitlines()
    containers = (
        json.loads(docker(["inspect", *ids], text=True, capture_output=True, check=True).stdout)
        if ids
        else []
    )
    occupied = set(RESERVED_CPUS)
    for item in containers:
        host = item["HostConfig"]
        requests = host.get("DeviceRequests", []) or []
        if any(
            req.get("Count") == -1 or requested.intersection(map(int, req.get("DeviceIDs", [])))
            for req in requests
        ):
            raise RuntimeError(f"Running container already owns GPU 0/1: {item['Name']}")
        occupied.update(parse_cpuset(host.get("CpusetCpus", "")))
    free = sorted(set(os.sched_getaffinity(0)) - occupied)
    if len(free) < 64:
        raise RuntimeError("Need 64 unoccupied CPU cores for the two independent containers")
    memory = available_memory_kib()
    if memory < 192 * 1024 * 1024:
        raise RuntimeError("Need 192 GiB currently available host memory")
    image = json.loads(
        docker(
            ["image", "inspect", frozen["image_sha256"]], text=True, capture_output=True, check=True
        ).stdout
    )[0]["Id"]
    if image != frozen["image_sha256"]:
        raise ValueError("Frozen image identity changed")
    return {
        "cpus": {"perk": free[:32], "fticr": free[32:64]},
        "gpus": {"perk": 0, "fticr": 1},
        "image": image,
        "reserved_cpus_for_z1": list(RESERVED_CPUS),
        "created_utc": utc(),
    }


def available_memory_kib():
    return next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def credential_environment():
    environment = os.environ.copy()
    if not environment.get("WANDB_API_KEY"):
        credentials = netrc.netrc(str(CREDENTIALS)).authenticators("api.wandb.ai")
        if not credentials:
            raise RuntimeError("Authorized local W&B credentials are unavailable")
        environment["WANDB_API_KEY"] = credentials[2]
    return environment


def container_command(base, arm, stage, cpus):
    if (
        arm not in ARMS
        or stage not in STAGES
        or len(cpus) != 32
        or len(set(cpus)) != 32
        or set(cpus).intersection(RESERVED_CPUS)
    ):
        raise ValueError("Requested task is outside the frozen two-arm matrix")
    base = Path(base)
    frozen = read(base / "frozen/freeze.json")
    name = f"lm-perk-v1-{arm}-{stage}"
    command = [
        DOCKER,
        "--context",
        "lm-memory",
        "run",
        "--name",
        name,
        "--gpus",
        f"device={0 if arm == 'perk' else 1}",
        "--cpus",
        "32",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        "96g",
        "--memory-swap",
        "96g",
        "--shm-size",
        "16g",
        "--tmpfs",
        "/tmp:rw,size=16g",
        "--network",
        NETWORK,
        "--mount",
        f"type=bind,source={base},target={base}",
        "--mount",
        f"type=bind,source={base / 'frozen'},target={base / 'frozen'},readonly",
        "--mount",
        f"type=bind,source={base / 'reference'},target={base / 'reference'},readonly",
        "--mount",
        f"type=bind,source={base / 'data'},target={base / 'data'},readonly",
        "--mount",
        f"type=bind,source={CREDENTIALS},target=/root/.netrc,readonly",
        "-e",
        f"WANDB_RUN_ID={read(config_path(base, arm))['wandb_run_id']}",
        "-e",
        "WANDB_MODE=online",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "OMP_NUM_THREADS=32",
        "-e",
        "TOKENIZERS_PARALLELISM=false",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        f"PYTHONPATH={source(base) / 'src'}",
        frozen["image_sha256"],
        "python",
        "-u",
        str(source(base) / "scripts/execute_perk_reproduction.py"),
        "worker",
        "--base",
        str(base),
        "--arm",
        arm,
        "--stage",
        stage,
    ]
    return name, command


def checkpoint_inventory(path):
    path = Path(path)
    paths = [path] if path.is_file() else file_inventory(path)
    if not paths:
        raise ValueError("Validation-best checkpoint is empty")
    return {str(p): sha(p) for p in paths}


def worker(base, arm, stage):
    base = Path(base)
    verify_freeze(base)
    root = base / "runs" / arm
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / stage
        if output.exists():
            raise RuntimeError("Stage output already exists; no retry or overwrite")
        config = read(config_path(base, arm))
        sys.path.insert(0, str(source(base) / "src"))
        if stage == "train":
            if arm == "perk":
                from llm_memory_editability.perk_reproduction import resolve_config, train

                result = train(resolve_config(config, output), base / "reference/perk")
            else:
                from llm_memory_editability.perk_fticr import train

                result = train(config, source_dir=base / "reference/open-instruct", output=output)
            best = result.get("best_checkpoint")
            if not best:
                raise ValueError(
                    "Scientific test requires a checkpoint selected by validation loss"
                )
            write(
                root / "checkpoint-selection.json",
                {
                    "checkpoint": best,
                    "selected_by": "validation_loss_only",
                    "files": checkpoint_inventory(best),
                    "created_utc": utc(),
                },
            )
            write(
                root / "native-training-endpoint.json",
                {
                    "training_complete": True,
                    "independent_audit": "pending",
                    "cloud_verification": "pending",
                    "run_id": config["wandb_run_id"],
                    "url": f"https://wandb.ai/zhusq20/llm-memory-editability/runs/{config['wandb_run_id']}",
                    "completion_sha256": sha(output / "completion.json"),
                    "created_utc": utc(),
                },
            )
        else:
            selection = read(root / "checkpoint-selection.json")
            for name, expected in selection["files"].items():
                if sha(name) != expected:
                    raise ValueError("Validation-best checkpoint changed before independent load")
            if arm == "perk":
                from llm_memory_editability.perk_reproduction import (
                    evaluate_checkpoint,
                    resolve_config,
                )

                result = evaluate_checkpoint(
                    resolve_config(config, output), base / "reference/perk", selection["checkpoint"]
                )
            else:
                from llm_memory_editability.perk_fticr import evaluate

                config["checkpoint"] = selection["checkpoint"]
                result = evaluate(config, author_perk_source=base / "reference/perk", output=output)
            write(output / "stage-completion.json", {"result": result, "created_utc": utc()})


def start_container(base, arm, stage, cpus, environment):
    root = Path(base) / "runs" / arm
    root.mkdir(parents=True, exist_ok=True)
    name, command = container_command(base, arm, stage, cpus)
    # Runtime artifacts retain the key's name, never its value or full inspection.
    write(root / f"{stage}-command.json", command)
    with (root / f"{stage}.log").open("x") as handle:
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, env=environment
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        inspection = docker(
            ["inspect", "--format", "{{json .State}}", name], capture_output=True, text=True
        )
        if inspection.returncode == 0:
            state = json.loads(inspection.stdout)
            if state.get("Running"):
                write(
                    root / f"{stage}-runtime.json",
                    {
                        "container": name,
                        "host_pid": state["Pid"],
                        "client_pid": process.pid,
                        "physical_gpu": 0 if arm == "perk" else 1,
                        "cpus": cpus,
                        "memory_gib": 96,
                        "shm_gib": 16,
                        "network": NETWORK,
                        "image_sha256": read(Path(base) / "frozen/freeze.json")["image_sha256"],
                        "credential_transport": "readonly_netrc_mount",
                        "created_utc": utc(),
                    },
                )
                break
        if process.poll() is not None:
            break
        time.sleep(0.1)
    return {
        "process": process,
        "arm": arm,
        "stage": stage,
        "container": name,
        "cpus": cpus,
        "started_utc": utc(),
        "started_unix": time.time(),
        "started_monotonic": time.monotonic(),
    }


def audit_science(base, arm):
    base = Path(base)
    verify_freeze(base)
    root = base / "runs" / arm
    selection = read(root / "checkpoint-selection.json")
    for name, expected in selection["files"].items():
        if sha(name) != expected:
            raise ValueError("Checkpoint changed during replay")
    result = compare_replay(root / "evaluation", root / "replay", arm, 999)
    result.update(
        checkpoint_files=selection["files"],
        created_utc=utc(),
        freeze_sha256=sha(base / "frozen/freeze.json"),
    )
    write(root / "audit.json", result)
    return result


def controller(base):
    base = Path(base)
    verify_freeze(base)
    allocation = read(base / "launch-resources.json")
    environment = os.environ.copy()
    environment.pop("WANDB_API_KEY", None)
    with (base / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, trained, completed, failed = {}, [], [], []
        for arm in ARMS:
            try:
                active[arm] = start_container(
                    base, arm, "train", allocation["cpus"][arm], environment
                )
            except Exception as error:
                failure = {"arm": arm, "stage": "train", "error": str(error)}
                failed.append(failure)
                write(base / "runs" / arm / "failure.json", failure)
        while active:
            write(
                base / "controller-state.json",
                {
                    "state": "running",
                    "active": [
                        {k: v for k, v in item.items() if k not in ("process", "started_monotonic")}
                        for item in active.values()
                    ],
                    "completed": completed,
                    "training_completed": trained,
                    "scientific_training": "complete" if len(trained) == 2 else "in_progress",
                    "failed": failed,
                    "updated_utc": utc(),
                },
            )
            for arm, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                inspect = docker(
                    ["inspect", "--format", "{{json .State}}", item["container"]],
                    capture_output=True,
                    text=True,
                )
                state = json.loads(inspect.stdout) if inspect.returncode == 0 else {}
                receipt = {"exit_code": code, "container_state": state, "created_utc": utc()}
                write(base / "runs" / arm / f"{item['stage']}-exit.json", receipt)
                if code != 0 or state.get("ExitCode") != 0 or state.get("Running") is True:
                    failed.append({"arm": arm, "stage": item["stage"], "exit_code": code})
                    del active[arm]
                    continue
                if item["stage"] == "train":
                    trained.append(arm)
                try:
                    index = STAGES.index(item["stage"])
                    if index < 2:
                        # A different, freshly created process/container performs every load.
                        active[arm] = start_container(
                            base, arm, STAGES[index + 1], item["cpus"], environment
                        )
                    else:
                        audit_science(base, arm)
                        completed.append(arm)
                        del active[arm]
                except Exception as error:
                    write(
                        base / "runs" / arm / "failure.json",
                        {
                            "stage": item["stage"],
                            "error": str(error),
                            "created_utc": utc(),
                        },
                    )
                    failed.append({"arm": arm, "stage": item["stage"], "error": str(error)})
                    del active[arm]
            if active:
                time.sleep(15)
        write(
            base / "controller-state.json",
            {
                "state": "complete" if not failed else "finished_with_failures",
                "active": [],
                "completed": completed,
                "training_completed": trained,
                "failed": failed,
                "updated_utc": utc(),
                "scientific_training": "complete" if len(trained) == 2 else "partial",
                "independent_audit": "passed" if len(completed) == 2 else "partial",
                "native_wandb_endpoint": "separate_per_arm",
                "cloud_verification": "pending",
            },
        )


def monitor(base):
    """Local process/resource sidecar; it never duplicates native W&B loss logging."""
    base = Path(base)
    with (base / "resources.jsonl").open("x") as handle:
        while True:
            state = (
                read(base / "controller-state.json")
                if (base / "controller-state.json").exists()
                else {}
            )
            resource = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            rows = [
                line
                for line in resource.stdout.splitlines()
                if line.split(",")[0].strip() in ("0", "1")
            ]
            value = {
                "created_utc": utc(),
                "controller_state": state.get("state"),
                "gpu_rows": rows,
                "nvidia_smi_exit": resource.returncode,
                "active": state.get("active", []),
                "timeout_monitor": {
                    "mode": "advisory",
                    "threshold_hours": 72,
                    "overdue_stages": [
                        {
                            "arm": item["arm"],
                            "stage": item["stage"],
                            "elapsed_seconds": time.time() - item["started_unix"],
                        }
                        for item in state.get("active", [])
                        if time.time() - item["started_unix"] > 72 * 3600
                    ],
                },
            }
            registration = base / "controller-process.json"
            if registration.exists():
                pid = read(registration)["controller_pid"]
                try:
                    os.kill(pid, 0)
                    value["controller_process_alive"] = True
                except ProcessLookupError:
                    value["controller_process_alive"] = False
                    value["advisory"] = "controller_exited_inspect_preserved_containers"
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            write(base / "resource-status.json", value)
            if state.get("state") in ("complete", "finished_with_failures"):
                return
            time.sleep(15)


def launch(base):
    base = Path(base)
    frozen = verify_freeze(base)
    receipt = read(base / "development/cuda-validation.json")
    if receipt.get("passed") is not True or set(receipt.get("arms", {})) != set(ARMS):
        raise ValueError("Both CUDA development and independent replay prerequisites must pass")
    if receipt["freeze_sha256"] != sha(base / "frozen/freeze.json"):
        raise ValueError("CUDA prerequisite belongs to a different freeze")
    if not receipt.get("artifacts"):
        raise ValueError("CUDA prerequisite must identify the actual development artifacts")
    for name, expected in receipt["artifacts"].items():
        if sha(name) != expected:
            raise ValueError(f"CUDA prerequisite artifact changed: {name}")
    for arm in ARMS:
        compare_replay(
            base / "development" / arm / "evaluation", base / "development" / arm / "replay", arm, 2
        )
    allocation = resource_check(base)
    marker = base / "controller-process.json"
    with (base / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker.exists() or (base / "controller-state.json").exists():
            raise RuntimeError("This batch was already submitted; inspect its actual state")
        write(base / "launch-resources.json", allocation)
        write(base / "controller-state.json", {"state": "starting", "created_utc": utc()})
        script = source(base) / "scripts/execute_perk_reproduction.py"
        children = {}
        for action in ("monitor", "controller"):
            with (base / f"{action}.log").open("x") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script), action, "--base", str(base)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children[f"{action}_pid"] = process.pid
        write(
            marker,
            {
                **children,
                "created_utc": utc(),
                "image_sha256": frozen["image_sha256"],
                "wandb_project": "https://wandb.ai/zhusq20/llm-memory-editability",
            },
        )
    return read(marker)


def verify_wandb(base):
    """Read the native cloud runs independently; absence is preserved as pending."""
    import wandb

    base = Path(base)
    environment = credential_environment()
    api = wandb.Api(api_key=environment["WANDB_API_KEY"])
    result = {}
    for arm in ARMS:
        config = read(config_path(base, arm))
        try:
            run = api.run(f"zhusq20/llm-memory-editability/{config['wandb_run_id']}")
            local = read(base / "runs" / arm / "train/completion.json")
            expected = local.get("global_step", local.get("optimizer_steps"))
            cloud_step = run.summary.get("trainer/global_step", run.summary.get("_step"))
            result[arm] = {
                "state": run.state,
                "url": run.url,
                "cloud_final_step": cloud_step,
                "local_optimizer_steps": expected,
                "finished": run.state == "finished",
                "exact_step_verified": cloud_step == expected or cloud_step == expected - 1,
            }
        except Exception as error:
            result[arm] = {
                "state": "unverified",
                "finished": False,
                "exact_step_verified": False,
                "error_type": type(error).__name__,
            }
    value = {
        "passed": all(row["finished"] and row["exact_step_verified"] for row in result.values()),
        "runs": result,
        "created_utc": utc(),
    }
    write(base / "wandb-cloud-verification.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "freeze",
            "verify",
            "resource-check",
            "audit-preflight",
            "launch",
            "controller",
            "monitor",
            "worker",
            "verify-wandb",
        ),
    )
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--image-sha")
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(
            args.base, args.image_sha, supplied_configs=read(args.configs) if args.configs else None
        )
    elif args.action == "verify":
        result = verify_freeze(args.base)
    elif args.action == "worker":
        if not args.arm or not args.stage:
            parser.error("worker requires --arm and --stage")
        result = worker(args.base, args.arm, args.stage)
    else:
        name = args.action.replace("-", "_")
        result = globals()[name](args.base)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
