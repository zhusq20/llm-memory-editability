"""Frozen Docker execution, artifact tracking and independent replay for Z1 v3.

This wrapper never changes the inherited scientific matrix or thresholds. The
select stage finishes all 112 nodes before its fixed power rule permits science.
Every science run is then replayed in a fresh container, with exact JSON equality
and checkpoint/data/code hash checks, before a completion receipt is written.

Commands: freeze, verify, resource-check, launch, controller, worker, tracker.
No command retries a failed task. A submitted controller cannot be submitted again.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import netrc
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path("/ossfs/workspace/research-recovery/zero-training-path-overlap-v3")
PREVIOUS_BASE = Path("/ossfs/workspace/research-recovery/zero-training-path-overlap-v2")
DOCKER = "/workspaces/lm-memory-tools/docker/docker"
IMAGE = "sha256:0087c5381ef5286d30b022825ab773f044fa15c161e5999ff5b47fbab681b18a"
CONTRACT_NAME = "configs/zero-training-path-overlap-v3.json"
PREVIOUS_CONTRACT = "configs/zero-training-path-overlap-v2.json"
PANELS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")
MAJOR_GROUPS = ("loop_ood", "standard_id")
STAGES = ("select", "science", "audit")
CREDENTIALS = Path("/ossfs/workspace/.lm-memory-credentials/wandb.netrc")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(base):
    return Path(base) / "frozen/source"


def input_root(base):
    return Path(base) / "inputs/grokking-reproduction-v1"


def contract(base):
    return read(source(base) / CONTRACT_NAME)


def receipt_checkpoint_map(receipt):
    rows = receipt["checkpoints"]
    result = {(row["run"], int(row["step"])): row["sha256"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate checkpoint identities in input receipt")
    return result


def receipt_file_map(base, receipt):
    root = input_root(base)
    result = {}
    for row in receipt["files"]:
        path = Path(row["path"])
        path = path if path.is_absolute() else Path(base) / path
        name = str(path.relative_to(root))
        if name in result:
            raise ValueError(f"Duplicate metadata identity in input receipt: {name}")
        result[name] = row
    return result


def input_names(root, cfg):
    names = ["data/world.npz", "data/panels.npz", "data/complete.json"]
    names.extend(
        str(path.relative_to(root))
        for path in (root / "reference/gpt2").rglob("*")
        if path.is_file()
    )
    if not any(name.startswith("reference/gpt2/") for name in names):
        raise ValueError("Missing local GPT-2 reference configuration")
    for run in cfg["source"]["runs"]:
        names.append(f"development/{run}/learning.json")
        for node in cfg["source"]["nodes"]:
            names.append(f"development/{run}/checkpoint-{node:07d}.pt")
            names.extend(
                f"development/{run}/predictions-{node:07d}-{panel}.npz" for panel in PANELS
            )
    return sorted(set(names))


def freeze(base):
    """Copy immutable source and verify each weight against the prepared receipt."""
    base = Path(base)
    destination = source(base)
    frozen = base / "frozen/freeze.json"
    if frozen.exists() or destination.exists():
        raise RuntimeError("Existing frozen source is immutable; inspect instead of replacing it")
    cfg = read(PROJECT / CONTRACT_NAME)
    root = input_root(base)
    receipt = read(base / "input-receipt.json")
    expected = receipt_checkpoint_map(receipt)
    metadata = receipt_file_map(base, receipt)
    matrix = {(r, s) for r in cfg["source"]["runs"] for s in cfg["source"]["nodes"]}
    if set(expected) != matrix or len(matrix) != 112:
        raise ValueError("Input receipt must cover exactly the inherited 8 x 14 checkpoints")
    files = list((PROJECT / "src/llm_memory_editability").rglob("*.py"))
    files.extend(
        PROJECT / name
        for name in (
            "scripts/execute_path_overlap.py",
            "scripts/run_zero_training_path_overlap.py",
            "scripts/read_path_hop_secondary.py",
            "scripts/validate_path_overlap_cuda.py",
            "configs/zero-training-path-overlap-v2.json",
            "configs/zero-training-path-overlap-v3.json",
            "configs/zero-training-path-overlap-v1.json",
            "configs/experiment-tracking-defaults.json",
            "tests/test_path_overlap_execution.py",
            "tests/test_path_overlap.py",
        )
    )
    for path in files:
        target = destination / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    names = sorted(set(input_names(root, cfg)) | set(metadata))

    def inspect(name):
        path = root / name
        actual = sha(path)
        if path.name.startswith("checkpoint-"):
            run = path.parent.name
            node = int(path.stem.split("-")[-1])
            if actual != expected[run, node]:
                raise ValueError(f"Prepared checkpoint hash mismatch: {run}@{node}")
        elif name not in metadata or actual != metadata[name]["sha256"]:
            raise ValueError(f"Prepared metadata hash mismatch or missing receipt: {name}")
        return name, {
            "sha256": actual,
            "bytes": path.stat().st_size,
            "resolved_path": str(path.resolve()),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        inputs = dict(pool.map(inspect, names))
    for name, field in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        if inputs[f"data/{name}"]["sha256"] != cfg["source"][field]:
            raise ValueError(f"Contract data hash mismatch: {name}")
    write(
        frozen,
        {
            "schema_version": 1,
            "batch": cfg["batch"],
            "source_files": {
                str(path.relative_to(destination)): sha(path)
                for path in destination.rglob("*")
                if path.is_file()
            },
            "contract": {"path": CONTRACT_NAME, "sha256": sha(destination / CONTRACT_NAME)},
            "inputs": {
                "root": str(root),
                "files": inputs,
                "receipt_sha256": sha(base / "input-receipt.json"),
            },
            "image_sha256": IMAGE,
            "validation_files": {
                str(path.relative_to(base)): sha(path)
                for path in (base / "validation").glob("*")
                if path.is_file() and path.suffix in {".json", ".npy"}
            },
            "optimizer_updates": 0,
            "created_utc": utc(),
        },
    )
    print(
        json.dumps({"frozen": str(frozen), "inputs": len(inputs), "checkpoints": len(matrix)}),
        flush=True,
    )


def verify_freeze(base, run=None, *, all_inputs=False):
    base = Path(base)
    frozen = read(base / "frozen/freeze.json")
    if frozen["image_sha256"] != IMAGE or frozen["optimizer_updates"] != 0:
        raise ValueError("Unexpected execution image or training updates")
    if sha(base / "input-receipt.json") != frozen["inputs"]["receipt_sha256"]:
        raise ValueError("Prepared input receipt changed")
    for name, expected in frozen["source_files"].items():
        if sha(source(base) / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in frozen.get("validation_files", {}).items():
        if sha(base / name) != expected:
            raise ValueError(f"Frozen CUDA validation changed: {name}")
    root = input_root(base)
    if str(root) != frozen["inputs"]["root"]:
        raise ValueError("Input root changed")
    receipt = receipt_checkpoint_map(read(base / "input-receipt.json"))
    for name, expected in frozen["inputs"]["files"].items():
        relevant = (
            all_inputs
            or run is not None
            and (not name.startswith("development/") or name.startswith(f"development/{run}/"))
        )
        if relevant and not Path(name).name.startswith("checkpoint-"):
            path = root / name
            if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
                raise ValueError(f"Frozen input changed: {name}")
        if Path(name).name.startswith("checkpoint-"):
            path = Path(name)
            identity = (path.parent.name, int(path.stem.split("-")[-1]))
            if expected["sha256"] != receipt[identity]:
                raise ValueError(f"Freeze disagrees with prepared checkpoint receipt: {identity}")
            if all_inputs and sha(root / name) != receipt[identity]:
                raise ValueError(f"Checkpoint changed: {identity}")
    if (base / "selection-adoption.json").exists():
        verify_adoption(base)
    return frozen


def verify_checkpoint(base, run, step):
    """Called immediately before EVERY model load, including replay loads."""
    expected = receipt_checkpoint_map(read(Path(base) / "input-receipt.json"))[run, int(step)]
    path = input_root(base) / "development" / run / f"checkpoint-{step:07d}.pt"
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Checkpoint differs from prior source receipt: {run}@{step}")
    return actual


def scientific_source_signature(text):
    """Ignore documentation and version labels, while retaining every computation."""
    tree = ast.parse(text.replace("zero-training-path-overlap-v2", "zero-training-path-overlap-v3"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def version_normalized(value):
    return json.loads(
        json.dumps(value).replace("zero-training-path-overlap-v2", "zero-training-path-overlap-v3")
    )


def replication_parameters(cfg):
    text = cfg["execution"]["replication_check"]
    patterns = {
        "min_agreement": r">= (0\.\d+) on every panel",
        "from_step": r"every node with step >= (\d+)",
        "final_test_ii": r"and >= (0\.\d+) on test_ii at 1\.5M",
        "early_stop": r"the node stops only below (0\.\d+)",
    }
    return {name: float(re.search(pattern, text).group(1)) for name, pattern in patterns.items()}


def validate_adoption_sources(base, previous):
    """Only a batch-label change and the authorized terminal threshold may differ."""
    old = read(previous / "frozen/freeze.json")
    new = read(base / "frozen/freeze.json")
    old_cfg = read(source(previous) / PREVIOUS_CONTRACT)
    new_cfg = contract(base)
    if (
        old_cfg["batch"] != "zero-training-path-overlap-v2"
        or new_cfg["batch"] != "zero-training-path-overlap-v3"
    ):
        raise ValueError("Selection adoption requires the explicit v2 to v3 amendment")
    for key in ("source", "z1", "s1_s2_readout", "outputs", "not_permitted", "withdrawn"):
        if version_normalized(old_cfg[key]) != version_normalized(new_cfg[key]):
            raise ValueError(f"Scientific contract changed beyond the replication amendment: {key}")
    old_execution = {
        key: value for key, value in old_cfg["execution"].items() if key != "replication_check"
    }
    new_execution = {
        key: value for key, value in new_cfg["execution"].items() if key != "replication_check"
    }
    if version_normalized(old_execution) != version_normalized(new_execution):
        raise ValueError("Scientific execution changed beyond the replication amendment")
    old_replication, new_replication = (
        replication_parameters(old_cfg),
        replication_parameters(new_cfg),
    )
    if (
        old_replication.pop("final_test_ii") != 0.99
        or new_replication.pop("final_test_ii") != 0.95
        or old_replication != new_replication
    ):
        raise ValueError("Only the final test_ii agreement 0.99 to 0.95 is authorized for adoption")
    old_inputs = {name: row["sha256"] for name, row in old["inputs"]["files"].items()}
    new_inputs = {name: row["sha256"] for name, row in new["inputs"]["files"].items()}
    if old_inputs != new_inputs:
        raise ValueError("Adoption data, metadata or weights differ from v2")
    core = (
        "src/llm_memory_editability/path_overlap.py",
        "scripts/run_zero_training_path_overlap.py",
        "scripts/read_path_hop_secondary.py",
        "src/llm_memory_editability/grokking_reproduction.py",
        "src/llm_memory_editability/grok_depth.py",
    )
    code = {}
    for name in core:
        old_path, new_path = source(previous) / name, source(base) / name
        if sha(old_path) != old["source_files"][name] or sha(new_path) != new["source_files"][name]:
            raise ValueError(f"Frozen scientific source changed: {name}")
        if (
            name
            not in {
                "src/llm_memory_editability/path_overlap.py",
                "scripts/run_zero_training_path_overlap.py",
            }
            and old["source_files"][name] != new["source_files"][name]
        ):
            raise ValueError(f"An unamended scientific dependency changed: {name}")
        if scientific_source_signature(old_path.read_text()) != scientific_source_signature(
            new_path.read_text()
        ):
            raise ValueError(f"Scientific computation changed beyond batch/version labels: {name}")
        code[name] = {
            "v2_sha256": old["source_files"][name],
            "v3_sha256": new["source_files"][name],
        }
    return {
        "scientific_source": code,
        "input_files": len(old_inputs),
        "replication_amendment": {
            "final_test_ii": {"v2": 0.99, "v3": 0.95},
            "other_parameters": old_replication,
        },
    }


def verify_adoption(base, run=None):
    adoption = read(base / "selection-adoption.json")
    previous = Path(adoption["source_base"])
    if sha(base / "frozen/freeze.json") != adoption["destination_freeze_sha256"]:
        raise ValueError("Adoption destination freeze changed")
    if sha(previous / "frozen/freeze.json") != adoption["source_freeze_sha256"]:
        raise ValueError("Adopted v2 freeze changed")
    for node in adoption["nodes"]:
        if run is not None and node["run"] != run:
            continue
        if (
            sha(Path(node["source_path"])) != node["sha256"]
            or sha(base / node["destination"]) != node["sha256"]
        ):
            raise ValueError(f"Adopted selection evidence changed: {node['run']}@{node['step']}")
    attachment = base / "execution-attachment.json"
    if attachment.exists() and read(attachment)["selection_adoption_sha256"] != sha(
        base / "selection-adoption.json"
    ):
        raise ValueError("Submitted selection adoption receipt changed")
    return adoption


def adopt_v2(base, previous):
    """Copy previous selection evidence byte for byte; keep its original provenance."""
    base, previous = Path(base), Path(previous)
    verify_freeze(base, all_inputs=True)
    if (base / "selection-adoption.json").exists() or (base / "selection-stage").exists():
        raise RuntimeError("Selection adoption already exists and is immutable")
    proof = validate_adoption_sources(base, previous)
    old_freeze = read(previous / "frozen/freeze.json")
    cfg = contract(base)
    nodes, missing = [], []
    # Validate every source record before creating any destination evidence.
    for run in cfg["source"]["runs"]:
        for step in cfg["source"]["nodes"]:
            path = previous / "selection-stage" / run / f"{step:07d}/selection.json"
            if not path.exists():
                missing.append({"run": run, "step": step})
                continue
            value = read(path)
            if (
                value.get("complete") is not True
                or value["run"] != run
                or value["step"] != step
                or value["replication"]["failures"]
            ):
                raise ValueError(f"Incomplete or failed source selection: {run}@{step}")
            if value["contract_sha256"] != old_freeze["contract"]["sha256"]:
                raise ValueError(f"Source selection has another contract: {run}@{step}")
            for name, expected in value["code_sha256"].items():
                if expected != old_freeze["source_files"][name]:
                    raise ValueError(f"Source selection has another code hash: {run}@{step}/{name}")
            expected_checkpoint = old_freeze["inputs"]["files"][
                f"development/{run}/checkpoint-{step:07d}.pt"
            ]["sha256"]
            if value["checkpoint_sha256"] != expected_checkpoint:
                raise ValueError(f"Source selection has another checkpoint: {run}@{step}")
            nodes.append(
                {
                    "run": run,
                    "step": step,
                    "source_batch": "zero-training-path-overlap-v2",
                    "source_path": str(path),
                    "source_contract_sha256": value["contract_sha256"],
                    "sha256": sha(path),
                    "destination": f"selection-stage/{run}/{step:07d}/selection.json",
                }
            )
    if len(nodes) != 111 or missing != [{"run": "standard8-phi3.6-wd0.1", "step": 1500000}]:
        raise ValueError(
            "Authorized adoption is exactly 111 v2 nodes and the one missing terminal node"
        )
    for node in nodes:
        target = base / node["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(node["source_path"], target)
        target.chmod(0o444)
    write(
        base / "selection-adoption.json",
        {
            "schema_version": 1,
            "source_base": str(previous),
            "source_batch": "zero-training-path-overlap-v2",
            "source_freeze_sha256": sha(previous / "frozen/freeze.json"),
            "destination_batch": cfg["batch"],
            "destination_freeze_sha256": sha(base / "frozen/freeze.json"),
            "proof": proof,
            "nodes": nodes,
            "missing_nodes": missing,
            "v3_selection_recomputed": False,
            "optimizer_updates": 0,
            "created_utc": utc(),
        },
    )
    for node in nodes:
        record_progress(
            base,
            node["run"],
            "select",
            node["step"],
            read(base / node["destination"]),
            cfg["source"]["nodes"],
            provenance=node,
        )
    print(
        json.dumps(
            {
                "adopted_v2_selection_nodes": len(nodes),
                "missing_nodes": missing,
                "science_or_audit_started": False,
            }
        ),
        flush=True,
    )


def adopted_selection_nodes(base, run):
    if not (base / "selection-adoption.json").exists():
        return set()
    return {node["step"] for node in verify_adoption(base, run)["nodes"] if node["run"] == run}


def register_adopted_tracking(base):
    if not (base / "selection-adoption.json").exists():
        return
    adoption = verify_adoption(base)
    cfg = contract(base)
    for run in cfg["source"]["runs"]:
        out = run_dir(base, run)
        # No active CUDA process exists for copied evidence. An empty runtime
        # keeps the existing sidecar from attaching to a historical worker PID.
        write(out / "container-runtime.json", {})
        tracking_metadata(base, run, None, cfg["source"]["nodes"])
        metadata = read(out / "run.json")
        metadata.pop("gpu", None)
        metadata.pop("pid", None)
        metadata["selection_evidence"] = {
            "origin": "adopted_v2",
            "nodes": len([node for node in adoption["nodes"] if node["run"] == run]),
            "receipt_sha256": sha(base / "selection-adoption.json"),
        }
        metadata["spec"]["selection_adoption_sha256"] = sha(base / "selection-adoption.json")
        write(out / "run.json", metadata)


def load_runner(base):
    sys.path.insert(0, str(source(base) / "src"))
    spec = importlib.util.spec_from_file_location(
        "frozen_z1_science", source(base) / "scripts/run_zero_training_path_overlap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dir(base, run):
    return Path(base) / "runs" / run


def analysis_index(stage, step, nodes):
    return STAGES.index(stage) * len(nodes) + nodes.index(step) + 1


def tracking_metadata(base, run, physical_gpu, nodes):
    out = run_dir(base, run)
    existing = read(out / "run.json") if (out / "run.json").exists() else {}
    runtime = (
        read(out / "container-runtime.json") if (out / "container-runtime.json").exists() else {}
    )
    adoption = (
        {"selection_adoption_sha256": sha(Path(base) / "selection-adoption.json")}
        if (Path(base) / "selection-adoption.json").exists()
        else {}
    )
    write(
        out / "run.json",
        {
            **existing,
            "batch": contract(base)["batch"],
            "name": run,
            "spec": {
                "parent_run": run,
                "optimizer_updates": 0,
                "contract_sha256": sha(source(base) / CONTRACT_NAME),
                **adoption,
            },
            "phase": "zero_training_analysis",
            "job_type": "evaluation",
            "learning_step_unit": (
                "analysis_index (14 selection + 14 science + 14 independent replay nodes)"
            ),
            "status_step_unit": "analysis_index",
            "evaluation_nodes": nodes,
            "world_sha256": contract(base)["source"]["world_sha256"],
            "execution_lock_sha256": sha(Path(base) / "frozen/freeze.json"),
            "gpu": physical_gpu,
            "pid": runtime.get("host_pid", os.getpid()),
            "dtype": "float32",
            "optimizer_updates": 0,
            "updated_utc": utc(),
        },
    )
    if not (out / "learning.json").exists():
        write(out / "learning.json", [])


def record_progress(base, run, stage, node, value, nodes, *, peak_memory=None, provenance=None):
    out = run_dir(base, run)
    path = out / "learning.json"
    history = read(path) if path.exists() else []
    index = analysis_index(stage, node, nodes)
    if any(row["step"] == index for row in history):
        raise RuntimeError(f"Refusing duplicate analysis index: {run}/{stage}/{node}")
    seconds = value.get("seconds", 0)
    seconds = seconds.get("total", 0) if isinstance(seconds, dict) else seconds
    metrics = {"node": {"source_training_step": node, "seconds": 0 if provenance else seconds}}
    if provenance:
        metrics["node"]["adopted_source_execution_seconds"] = seconds
    if stage == "select":
        metrics["selection"] = {
            design: {kind: dict(record) for kind, record in block["kinds"].items()}
            for design, block in value["designs"].items()
        }
    elif stage == "science":
        z1 = read(Path(base) / run / f"{node:07d}/z1.json")
        metrics["science"] = {"atomic_all_answer_accuracy": z1["atomic_all_answer_accuracy"]}
        for design in ("primary_design", "fallback_design"):
            metrics["science"][design] = z1[design]["kinds"]
    else:
        metrics["audit"] = {"exact_outputs": 2, "checkpoint_data_code_hashes_verified": 1}
    if peak_memory is not None:
        metrics["node"]["peak_memory_bytes"] = peak_memory
    runtime = (
        read(out / "container-runtime.json") if (out / "container-runtime.json").exists() else {}
    )
    elapsed = (
        0
        if provenance
        else (
            time.time() - read(Path(base) / "launch-time.json")["unix_time"]
            if (Path(base) / "launch-time.json").exists()
            else seconds
        )
    )
    history.append(
        {
            "step": index,
            "condition": f"{stage}_adopted_v2" if provenance else stage,
            "stage": stage,
            "source_training_step": node,
            "optimizer_updates": 0,
            "wall_seconds": elapsed,
            "metrics": metrics,
            "physical_gpu": runtime.get("physical_gpu"),
            "host_pid": runtime.get("host_pid"),
            "created_utc": utc(),
            **({"selection_evidence_origin": provenance} if provenance else {}),
        }
    )
    write(path, sorted(history, key=lambda row: row["step"]))
    write(
        out / "status.json",
        {
            "state": f"{stage}_running",
            "step": index,
            "source_training_step": node,
            "optimizer_updates": 0,
            "updated_utc": utc(),
        },
    )


def audit_node(base, run, step):
    """Scientific output equality is exact; volatile timing/device metadata is excluded."""
    base = Path(base)
    original = base / run / f"{step:07d}"
    replay = base / "audit-replay" / run / f"{step:07d}"
    expected_files = {"z1.json", "ported.json", "hashes.json"}
    for folder in (original, replay):
        if {path.name for path in folder.iterdir()} != expected_files:
            raise ValueError(
                f"Per-node directory does not have exactly the inherited three files: {folder}"
            )
    a, b = read(original / "hashes.json"), read(replay / "hashes.json")
    for block, folder in ((a, original), (b, replay)):
        if block.get("complete") is not True or block["replication"]["failures"]:
            raise ValueError("Incomplete or replication-failed node")
        for name, expected in block["outputs_sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError(f"Output hash mismatch: {folder / name}")
    for name in ("z1.json", "ported.json"):
        if (original / name).read_bytes() != (replay / name).read_bytes():
            raise ValueError(f"Exact independent replay mismatch: {run}@{step}/{name}")
    for key in (
        "run",
        "step",
        "checkpoint_sha256",
        "world_sha256",
        "panels_sha256",
        "contract_sha256",
        "code_sha256",
        "tf32",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Replay provenance differs: {key}")
    frozen = read(base / "frozen/freeze.json")
    root = input_root(base)
    for key, name in (("world_sha256", "data/world.npz"), ("panels_sha256", "data/panels.npz")):
        if a[key] != frozen["inputs"]["files"][name]["sha256"] or sha(root / name) != a[key]:
            raise ValueError(f"Data provenance differs from freeze: {key}")
    if a["contract_sha256"] != frozen["contract"]["sha256"]:
        raise ValueError("Contract provenance differs from freeze")
    for name, expected in a["code_sha256"].items():
        if expected != frozen["source_files"][name] or sha(source(base) / name) != expected:
            raise ValueError(f"Code provenance differs from freeze: {name}")
    if a["checkpoint_sha256"] != verify_checkpoint(base, run, step):
        raise ValueError("Checkpoint provenance differs from prepared input receipt")
    return {
        "step": step,
        "passed": True,
        "outputs_exact": ["z1.json", "ported.json"],
        "hashes_verified": True,
    }


def worker(args):
    base = args.base
    verify_freeze(base, args.run)
    cfg = contract(base)
    nodes = cfg["source"]["nodes"]
    if (
        args.run not in cfg["source"]["runs"]
        or args.steps is not None
        and not set(args.steps) <= set(nodes)
    ):
        raise ValueError("Worker requested a task outside the frozen scientific matrix")
    selected = args.steps or nodes
    if len(selected) != len(set(selected)) or selected != sorted(selected, key=nodes.index):
        raise ValueError("Requested nodes must be unique and in frozen order")
    out = run_dir(base, args.run)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get("LM_Z1_CONTAINER"):
            deadline = time.monotonic() + 30
            while True:
                runtime = (
                    read(out / "container-runtime.json")
                    if (out / "container-runtime.json").exists()
                    else {}
                )
                if (
                    runtime.get("stage") == args.stage
                    and runtime.get("physical_gpu") == args.physical_gpu
                ):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Controller did not register the physical GPU and host PID")
                time.sleep(0.1)
        target_root = (
            base / "selection-stage"
            if args.stage == "select"
            else base / "audit-replay"
            if args.stage == "audit"
            else base
        )
        if any((target_root / args.run / f"{node:07d}").exists() for node in selected):
            if args.stage != "select":
                raise RuntimeError(
                    "Existing stage output is immutable; no automatic replay or retry"
                )
            adopted = adopted_selection_nodes(base, args.run)
            existing = {
                node for node in selected if (target_root / args.run / f"{node:07d}").exists()
            }
            if not existing <= adopted:
                raise RuntimeError(
                    "Existing stage output is immutable; no automatic replay or retry"
                )
            selected = [node for node in selected if node not in adopted]
            if not selected:
                write(
                    out / "status.json",
                    {
                        "state": "select_complete",
                        "adopted_only": True,
                        "optimizer_updates": 0,
                        "updated_utc": utc(),
                    },
                )
                return
        if args.stage != "select":
            if not selection_decision(read(base / "selection-stage/selection-summary.json")):
                raise RuntimeError("Frozen selection prerequisite was not met")
        runtime = (
            read(out / "container-runtime.json")
            if (out / "container-runtime.json").exists()
            else {}
        )
        process_identity = {
            "pid": os.getpid(),
            "host_pid": runtime.get("host_pid", os.getpid()),
            "container": runtime.get("container"),
            "stage": args.stage,
            "started_utc": utc(),
        }
        if args.stage == "audit":
            previous = read(out / "science-process.json")
            if previous["host_pid"] == process_identity["host_pid"]:
                raise RuntimeError("Independent replay must use a new process")
            if (
                os.environ.get("LM_Z1_CONTAINER")
                and previous["container"] == process_identity["container"]
            ):
                raise RuntimeError("Independent replay must use a new container")
        write(out / f"{args.stage}-process.json", process_identity)
        tracking_metadata(base, args.run, args.physical_gpu, nodes)
        runner = load_runner(base)
        original_checked = runner.checked_model
        original_writer = runner.write_json
        audit_records = []

        def checked(*positional, **keywords):
            verify_checkpoint(base, args.run, positional[2])
            if args.gpu is not None:
                runner.torch.cuda.reset_peak_memory_stats(args.gpu)
            return original_checked(*positional, **keywords)

        def recorded(path, value):
            original_writer(path, value)
            path = Path(path)
            target = "selection.json" if args.stage == "select" else "hashes.json"
            if path.name != target or value.get("complete") is not True:
                return
            node = int(value["step"])
            if args.stage == "audit":
                audit_records.append(audit_node(base, args.run, node))
            peak = value.get("peak_memory_bytes")
            if args.gpu is not None and peak is None:
                runner.torch.cuda.synchronize(args.gpu)
                peak = int(runner.torch.cuda.max_memory_allocated(args.gpu))
            record_progress(base, args.run, args.stage, node, value, nodes, peak_memory=peak)

        runner.checked_model = checked
        runner.write_json = recorded
        arguments = argparse.Namespace(
            contract=source(base) / CONTRACT_NAME,
            root=input_root(base),
            out=base / "audit-replay" if args.stage == "audit" else base,
            run=args.run,
            gpu=args.gpu,
            steps=selected,
        )
        try:
            runner.select(arguments) if args.stage == "select" else runner.run(arguments)
            if args.stage == "audit":
                if len(audit_records) != len(selected):
                    raise RuntimeError(
                        "Replay skipped nodes instead of independently recomputing them"
                    )
                audit = {
                    "passed": True,
                    "fresh_process": True,
                    "fresh_container": bool(os.environ.get("LM_Z1_CONTAINER")),
                    "science_process": previous,
                    "audit_process": process_identity,
                    "nodes": audit_records,
                    "optimizer_updates": 0,
                    "completed_utc": utc(),
                }
                write(out / ("audit.json" if selected == nodes else "partial-audit.json"), audit)
                if selected == nodes:
                    write(
                        out / "complete.json",
                        {
                            "state": "complete",
                            "audit_sha256": sha(out / "audit.json"),
                            "analysis_index": len(nodes) * len(STAGES),
                            "optimizer_updates": 0,
                            "completed_utc": utc(),
                        },
                    )
            write(
                out / "status.json",
                {
                    "state": "complete"
                    if args.stage == "audit" and selected == nodes
                    else f"{args.stage}_complete",
                    "optimizer_updates": 0,
                    "updated_utc": utc(),
                },
            )
        except BaseException as error:
            write(
                out / "failure.json",
                {
                    "run": args.run,
                    "stage": args.stage,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "optimizer_updates": 0,
                    "created_utc": utc(),
                },
            )
            write(
                out / "status.json",
                {
                    "state": "failed",
                    "stage": args.stage,
                    "optimizer_updates": 0,
                    "updated_utc": utc(),
                },
            )
            raise


def selection_decision(summary):
    if any(summary["groups"][name]["state"] != "complete" for name in MAJOR_GROUPS):
        raise ValueError("Selection feasibility summary is incomplete")
    values = [summary["groups"][name]["expected_inconclusive"] for name in MAJOR_GROUPS]
    if any(value not in (True, False) for value in values):
        raise ValueError("Selection feasibility power rule was not evaluated")
    return not all(values)


def docker(arguments, **options):
    return subprocess.run([DOCKER, "--context", "lm-memory", *arguments], **options)


def cpu_allocations(affinity, count, occupied=()):
    available = [cpu for cpu in sorted(affinity) if cpu not in set(occupied)]
    # The first sixteen free cores remain available to host processes and PERK.
    if len(available) < 16 + count * 8:
        raise RuntimeError("Insufficient unoccupied CPU cores for isolated Z1 workers")
    return [available[16 + 8 * index : 24 + 8 * index] for index in range(count)]


def parse_cpuset(text):
    result = []
    for part in text.split(",") if text else []:
        bounds = part.split("-")
        result.extend(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def resource_check(base, gpus, reserved_cpus=None):
    if len(gpus) != len(set(gpus)) or not 1 <= len(gpus) <= 6 or not set(gpus) <= set(range(2, 8)):
        raise ValueError("Z1 may use up to six distinct GPUs from 2..7; GPUs 0/1 are reserved")
    memory = [
        int(row)
        for row in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        ).splitlines()
    ]
    processes = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if any(memory[gpu] >= 1000 for gpu in gpus) or any(
        line.split(",")[0].strip() in {uuids[gpu] for gpu in gpus} for line in processes
    ):
        raise RuntimeError(
            "A requested GPU has an existing compute process or >= 1000 MiB allocated"
        )
    ids = docker(["ps", "-q"], capture_output=True, text=True, check=True).stdout.splitlines()
    containers = (
        json.loads(docker(["inspect", *ids], capture_output=True, text=True, check=True).stdout)
        if ids
        else []
    )
    occupied = []
    for container in containers:
        host = container["HostConfig"]
        assigned = {
            str(gpu)
            for request in host.get("DeviceRequests", []) or []
            for gpu in request.get("DeviceIDs", []) or []
        }
        if assigned & set(map(str, gpus)) or any(
            request.get("Count") == -1 for request in host.get("DeviceRequests", []) or []
        ):
            raise RuntimeError(f"Running container owns a requested GPU: {container['Name']}")
        occupied.extend(parse_cpuset(host.get("CpusetCpus", "")))
    if reserved_cpus is not None and set(reserved_cpus) & set(occupied):
        raise RuntimeError("An existing container owns a requested CPU core")
    image = docker(["image", "inspect", IMAGE], capture_output=True, text=True, check=True)
    if json.loads(image.stdout)[0]["Id"] != IMAGE:
        raise RuntimeError("Fixed LM image digest changed")
    cpus = cpu_allocations(os.sched_getaffinity(0), len(gpus), occupied)
    value = {
        "gpus": gpus,
        "gpu_memory_used_mib": memory,
        "gpu_compute_processes": processes,
        "running_containers": [
            {"name": c["Name"], "cpuset": c["HostConfig"].get("CpusetCpus")} for c in containers
        ],
        "cpus": cpus,
        "memory_gib_per_container": 48,
        "image": IMAGE,
        "created_utc": utc(),
    }
    write(Path(base) / f"resource-check-{time.time_ns()}.json", value)
    return value


def mount_paths(base):
    base = Path(base)
    frozen = read(base / "frozen/freeze.json")
    external = set()
    resolved_root = input_root(base).resolve()
    if resolved_root != input_root(base) and not resolved_root.is_relative_to(base):
        external.add(resolved_root)
    for name, row in frozen["inputs"]["files"].items():
        original = input_root(base) / name
        resolved = Path(row["resolved_path"])
        if resolved != original and not resolved.is_relative_to(base):
            external.add(resolved.parent)
    return sorted(external)


def container_command(base, run, stage, gpu, cpus):
    base = Path(base)
    name = f"lm-z1-v3-{run}-{stage}"
    command = [
        DOCKER,
        "--context",
        "lm-memory",
        "run",
        "--name",
        name,
        "--gpus",
        f"device={gpu}",
        "--cpus",
        "8",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        "48g",
        "--memory-swap",
        "48g",
        "--shm-size",
        "8g",
        "--network",
        "none",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--mount",
        f"type=bind,source={base},target={base}",
        "--mount",
        f"type=bind,source={base / 'frozen'},target={base / 'frozen'},readonly",
        "--mount",
        f"type=bind,source={base / 'inputs'},target={base / 'inputs'},readonly",
    ]
    for path in mount_paths(base):
        command.extend(["--mount", f"type=bind,source={path},target={path},readonly"])
    command.extend(
        [
            "-e",
            f"PYTHONPATH={source(base) / 'src'}",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "LM_Z1_CONTAINER=1",
            IMAGE,
            "python",
            "-u",
            str(source(base) / "scripts/execute_path_overlap.py"),
            "worker",
            "--base",
            str(base),
            "--run",
            run,
            "--stage",
            stage,
            "--gpu",
            "0",
            "--physical-gpu",
            str(gpu),
        ]
    )
    return name, command


def start_container(base, run, stage, gpu, cpus):
    out = run_dir(base, run)
    out.mkdir(parents=True, exist_ok=True)
    name, command = container_command(base, run, stage, gpu, cpus)
    with (out / f"{stage}.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    started = time.monotonic()
    while time.monotonic() - started < 30:
        inspect = docker(
            ["inspect", "--format", "{{json .State}}", name], capture_output=True, text=True
        )
        if inspect.returncode == 0:
            state = json.loads(inspect.stdout)
            if state["Running"]:
                runtime = {
                    "host_pid": state["Pid"],
                    "physical_gpu": gpu,
                    "container": name,
                    "stage": stage,
                    "cpu_affinity": cpus,
                    "memory_gib": 48,
                    "image": IMAGE,
                    "network": "none",
                    "created_utc": utc(),
                }
                write(out / "container-runtime.json", runtime)
                write(out / f"{stage}-runtime.json", runtime)
                tracking_metadata(base, run, gpu, contract(base)["source"]["nodes"])
                break
        if process.poll() is not None:
            break
        time.sleep(0.1)
    return {
        "process": process,
        "run": run,
        "stage": stage,
        "gpu": gpu,
        "cpus": cpus,
        "container": name,
    }


def persist_exit(base, item, code):
    inspection = docker(
        ["inspect", "--format", "{{json .State}}", item["container"]],
        capture_output=True,
        text=True,
    )
    value = {
        "client_exit_code": code,
        "container_state": json.loads(inspection.stdout) if inspection.returncode == 0 else None,
        "created_utc": utc(),
    }
    write(run_dir(base, item["run"]) / f"{item['stage']}-exit.json", value)
    return (
        code != 0 or value["container_state"] is None or value["container_state"]["ExitCode"] != 0
    )


def controller_state(base, state, active, completed, failed, **extra):
    write(
        Path(base) / "controller-state.json",
        {
            "state": state,
            "active": [
                {key: item[key] for key in ("run", "stage", "gpu", "cpus", "container")}
                for item in active.values()
            ],
            "completed": completed,
            "failed": failed,
            "optimizer_updates": 0,
            "updated_utc": utc(),
            **extra,
        },
    )


def monitor_stage(base, runs, slots, stage, *, audit_after=False):
    pending = list(runs)
    active, completed, failed = {}, [], []
    if stage == "select":
        nodes = contract(base)["source"]["nodes"]
        complete_adopted = [
            run for run in pending if set(nodes) <= adopted_selection_nodes(base, run)
        ]
        completed.extend(complete_adopted)
        pending = [run for run in pending if run not in complete_adopted]
    while pending or active:
        for slot, (gpu, cpus) in enumerate(slots):
            if slot not in active and pending:
                run = pending.pop(0)
                # Recheck GPU ownership immediately before assigning a previously free slot.
                try:
                    resource_check(base, [gpu], reserved_cpus=cpus)
                    active[slot] = start_container(base, run, stage, gpu, cpus)
                except Exception as error:
                    failure = {
                        "run": run,
                        "stage": stage,
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "created_utc": utc(),
                    }
                    failed.append(failure)
                    write(run_dir(base, run) / "failure.json", failure)
        controller_state(base, f"{stage}_running", active, completed, failed, pending=pending)
        for slot, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            if persist_exit(base, item, code):
                failure = {
                    "run": item["run"],
                    "stage": item["stage"],
                    "exit_code": code,
                    "created_utc": utc(),
                }
                failed.append(failure)
                if not (run_dir(base, item["run"]) / "failure.json").exists():
                    write(run_dir(base, item["run"]) / "failure.json", failure)
                del active[slot]
            elif audit_after and item["stage"] == "science":
                try:
                    resource_check(base, [item["gpu"]], reserved_cpus=item["cpus"])
                    active[slot] = start_container(
                        base, item["run"], "audit", item["gpu"], item["cpus"]
                    )
                except Exception as error:
                    failure = {
                        "run": item["run"],
                        "stage": "audit_start",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "created_utc": utc(),
                    }
                    failed.append(failure)
                    write(run_dir(base, item["run"]) / "failure.json", failure)
                    del active[slot]
            else:
                completed.append(item["run"])
                del active[slot]
        if pending or active:
            time.sleep(3)
    return completed, failed


def summary_command(base, action):
    command = [
        sys.executable,
        "-u",
        str(source(base) / "scripts/run_zero_training_path_overlap.py"),
        action,
        "--contract",
        str(source(base) / CONTRACT_NAME),
        "--root",
        str(input_root(base)),
        "--out",
        str(base),
    ]
    subprocess.run(command, env=host_environment(base), check=True)


def controller(args):
    base = args.base
    verify_freeze(base, all_inputs=True)
    with (base / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = contract(base)
        completed, failed = [], []
        try:
            resource = read(base / "launch-resource-check.json")
            slots = list(zip(resource["gpus"], resource["cpus"], strict=True))
            selected, failed = monitor_stage(base, cfg["source"]["runs"], slots, "select")
            if failed:
                for run in selected:
                    write(
                        run_dir(base, run) / "failure.json",
                        {
                            "stage": "selection_batch_stopped",
                            "reason": "Another selection run failed; no science was started",
                            "created_utc": utc(),
                        },
                    )
                controller_state(base, "finished_with_failures", {}, [], failed, selected=selected)
                return
            summary_command(base, "select-summary")
            selection = read(base / "selection-stage/selection-summary.json")
            if not selection_decision(selection):
                for run in selected:
                    write(
                        run_dir(base, run) / "status.json",
                        {
                            "state": "development_prerequisite_not_met",
                            "optimizer_updates": 0,
                            "two_hop_interventions_executed": False,
                            "updated_utc": utc(),
                        },
                    )
                controller_state(
                    base,
                    "development_prerequisite_not_met",
                    {},
                    [],
                    [],
                    selected=selected,
                    selection_summary_sha256=sha(base / "selection-stage/selection-summary.json"),
                )
                return
            completed, failed = monitor_stage(
                base, cfg["source"]["runs"], slots, "science", audit_after=True
            )
            if not failed:
                summary_command(base, "analyze")
                write(
                    base / "completion-manifest.json",
                    {
                        "passed": True,
                        "runs": completed,
                        "nodes": 112,
                        "optimizer_updates": 0,
                        "independent_replay": {
                            "nodes": 112,
                            "exact_outputs_per_node": ["z1.json", "ported.json"],
                        },
                        "summary_sha256": sha(base / "summary.json"),
                        "wandb_cloud_verification": "Separate receipt required",
                        "completed_utc": utc(),
                    },
                )
            controller_state(
                base,
                "complete" if not failed else "finished_with_failures",
                {},
                completed,
                failed,
                selected=selected,
            )
        except BaseException as error:
            failed.append(
                {
                    "stage": "controller",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "created_utc": utc(),
                }
            )
            write(base / "controller-failure.json", failed[-1])
            # All registered trajectories receive terminal failure markers so the sidecar drains.
            for out in (base / "runs").glob("*"):
                if not (out / "complete.json").exists() and not (out / "failure.json").exists():
                    write(out / "failure.json", failed[-1])
            controller_state(base, "finished_with_failures", {}, completed, failed)
            raise


def host_environment(base):
    env = dict(os.environ)
    env.pop("WANDB_API_KEY", None)
    env["PYTHONPATH"] = str(source(base) / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["LD_LIBRARY_PATH"] = "/lib64" + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def launch(args):
    base = args.base
    verify_freeze(base, all_inputs=True)
    with (base / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base / "controller-process.json").exists() or (base / "tracking-process.json").exists():
            raise RuntimeError("Batch already submitted; inspect state and Docker processes")
        validation = read(base / "validation/cuda-validation.json")
        audit = read(base / "validation/cuda-audit.json")
        if (
            validation.get("passed") is not True
            or audit.get("passed") is not True
            or audit.get("fresh_process") is not True
        ):
            raise RuntimeError(
                "Original-scale CUDA validation and fresh-process audit must pass first"
            )
        resource = resource_check(base, args.gpus)
        write(base / "launch-resource-check.json", resource)
        auth = netrc.netrc(str(CREDENTIALS)).authenticators("api.wandb.ai")
        if not auth or not auth[2]:
            raise RuntimeError("W&B credential file has no api.wandb.ai password")
        write(base / "launch-time.json", {"unix_time": time.time(), "created_utc": utc()})
        write(
            base / "execution-attachment.json",
            {
                "freeze_sha256": sha(base / "frozen/freeze.json"),
                "selection_adoption_sha256": sha(base / "selection-adoption.json")
                if (base / "selection-adoption.json").exists()
                else None,
                "created_utc": utc(),
            },
        )
        register_adopted_tracking(base)
        controller_state(base, "launching", {}, [], [])
        env = host_environment(base)
        tracker_env = {**env, "WANDB_API_KEY": auth[2]}
        tracking = [
            sys.executable,
            "-u",
            str(source(base) / "scripts/execute_path_overlap.py"),
            "tracker",
            "--base",
            str(base),
        ]
        with (base / "tracking.log").open("x") as log:
            tracker = subprocess.Popen(
                tracking,
                env=tracker_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write(
            base / "tracking-process.json",
            {"pid": tracker.pid, "command": tracking, "mode": "online", "created_utc": utc()},
        )
        deadline = time.monotonic() + 45
        while not (base / "tracking-ready.json").exists():
            if tracker.poll() is not None or time.monotonic() >= deadline:
                failure = {
                    "stage": "tracker_startup",
                    "message": "Tracking did not initialize before controller startup",
                    "created_utc": utc(),
                }
                write(base / "controller-failure.json", failure)
                controller_state(base, "finished_with_failures", {}, [], [failure])
                raise RuntimeError(failure["message"])
            time.sleep(0.1)
        command = [
            sys.executable,
            "-u",
            str(source(base) / "scripts/execute_path_overlap.py"),
            "controller",
            "--base",
            str(base),
        ]
        with (base / "controller.log").open("x") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(
            base / "controller-process.json",
            {"pid": process.pid, "command": command, "frozen": True, "created_utc": utc()},
        )
        print(
            json.dumps(
                {
                    "controller_pid": process.pid,
                    "tracker_pid": tracker.pid,
                    "wandb_project": "https://wandb.ai/zhusq20/llm-memory-editability",
                    "base": str(base),
                }
            ),
            flush=True,
        )


def tracker(base):
    sys.path.insert(0, str(source(base) / "src"))
    from llm_memory_editability import interface_tracking

    class SelectionTracker(interface_tracking.AuditedRunTracker):
        def poll(self):
            finished = super().poll()
            if finished:
                return True
            status = read(self.path / "status.json") if (self.path / "status.json").exists() else {}
            if status.get("state") == "development_prerequisite_not_met":
                self.run.summary["independently_reloaded"] = False
                self.run.summary["two_hop_interventions_executed"] = False
                self.run.summary["scientific_final_step"] = self.cursor
                self.run.finish(exit_code=0)
                self.state["last_logged_step"] = self.cursor
                interface_tracking.tracking.write_json(self.state_path, self.state)
                return True
            return False

    settings = read(source(base) / "configs/experiment-tracking-defaults.json")
    sdk = __import__("wandb")
    folder = base / "tracking-wandb"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "tracker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(base / "tracking-ready.json", {"sdk_version": sdk.__version__, "created_utc": utc()})
        interface_tracking.drain_runs(
            sdk, base, base / "runs", settings, tracker_factory=SelectionTracker
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "freeze",
            "verify",
            "adopt-v2",
            "resource-check",
            "launch",
            "controller",
            "worker",
            "tracker",
        ),
    )
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--previous-base", type=Path, default=PREVIOUS_BASE)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(2, 8)))
    parser.add_argument("--run")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.action == "adopt-v2":
        adopt_v2(args.base, args.previous_base)
    elif args.action == "worker":
        if args.run is None or args.stage is None:
            parser.error("worker requires --run and --stage")
        worker(args)
    elif args.action in ("launch", "controller"):
        globals()[args.action](args)
    elif args.action == "verify":
        verify_freeze(args.base, all_inputs=True)
    elif args.action == "resource-check":
        print(json.dumps(resource_check(args.base, args.gpus)), flush=True)
    else:
        globals()[args.action](args.base)


if __name__ == "__main__":
    main()
