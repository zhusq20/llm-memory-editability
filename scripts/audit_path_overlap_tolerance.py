"""Audit-only recovery for inherited Z1 v3 science with a cosine tolerance.

Only floating values under ported.json /layerwise_cosine use abs=1e-6,
rel=1e-5. Z1 bytes, all other ported values, source/data/weight hashes remain
exact. This script imports the unchanged frozen scientific worker and repeats
only its independent audit stage. Selection and scientific output are inherited.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

ORIGIN = Path("/ossfs/workspace/research-recovery/zero-training-path-overlap-v3")
SCIENCE = ORIGIN / "execution-attempts/source-base-mount-v1"
ATTEMPT_NAME = "audit-tolerance-v1"
SCRIPT_NAME = "audit_path_overlap_tolerance.py"
ABS_TOL, REL_TOL = 1e-6, 1e-5


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
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_contract():
    return {
        "z1": "byte_exact",
        "ported": "exact except finite float values under /layerwise_cosine",
        "cosine_abs_tolerance": ABS_TOL,
        "cosine_relative_tolerance": REL_TOL,
        "rule": "abs(new-old) <= abs_tolerance + relative_tolerance * abs(old)",
        "source_data_checkpoint_hashes": "exact",
    }


def compare_ported(original, replay):
    stats = {
        "cosine_floats": 0,
        "different_cosine_floats": 0,
        "max_cosine_absolute_error": 0.0,
        "max_cosine_relative_error": 0.0,
    }

    def compare(a, b, path=()):
        if type(a) is not type(b):
            raise ValueError(f"Replay JSON type changed: /{'/'.join(map(str, path))}")
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError("Replay JSON keys changed")
            for key in a:
                compare(a[key], b[key], (*path, key))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError("Replay JSON array shape changed")
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                compare(x, y, (*path, index))
        elif isinstance(a, float) and path and path[0] == "layerwise_cosine":
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite cosine cannot pass tolerance audit")
            error = abs(b - a)
            relative = error / max(abs(a), 1e-300)
            if error > ABS_TOL + REL_TOL * abs(a):
                raise ValueError(
                    f"Cosine replay exceeds authorized tolerance at /{'/'.join(map(str, path))}"
                )
            stats["cosine_floats"] += 1
            stats["different_cosine_floats"] += int(a != b)
            stats["max_cosine_absolute_error"] = max(stats["max_cosine_absolute_error"], error)
            stats["max_cosine_relative_error"] = max(stats["max_cosine_relative_error"], relative)
        elif a != b:
            raise ValueError(f"Exact ported replay mismatch at /{'/'.join(map(str, path))}")

    compare(original, replay)
    return stats


def verify_node(folder, run, step, origin, legacy):
    if {path.name for path in folder.iterdir()} != {"z1.json", "ported.json", "hashes.json"}:
        raise ValueError("Node directory must retain exactly the inherited three files")
    metadata = read(folder / "hashes.json")
    if (
        metadata.get("complete") is not True
        or metadata["run"] != run
        or metadata["step"] != step
        or metadata["replication"]["failures"]
    ):
        raise ValueError("Incomplete or baseline-replication-failed source node")
    if set(metadata["outputs_sha256"]) != {"z1.json", "ported.json"}:
        raise ValueError("Incomplete output hash coverage")
    for name, expected in metadata["outputs_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Node output hash differs: {name}")
    frozen = read(origin / "frozen/freeze.json")
    if metadata["contract_sha256"] != frozen["contract"]["sha256"]:
        raise ValueError("Node contract hash differs from the unchanged scientific freeze")
    expected_code = {
        name: frozen["source_files"][name]
        for name in (
            "src/llm_memory_editability/path_overlap.py",
            "scripts/run_zero_training_path_overlap.py",
            "scripts/read_path_hop_secondary.py",
        )
    }
    if metadata["code_sha256"] != expected_code:
        raise ValueError("Node scientific source hash differs")
    for name, expected in expected_code.items():
        if sha(origin / "frozen/source" / name) != expected:
            raise ValueError("Scientific source bytes differ from freeze")
    for key, name in (
        ("world_sha256", "data/world.npz"),
        ("panels_sha256", "data/panels.npz"),
        ("checkpoint_sha256", f"development/{run}/checkpoint-{step:07d}.pt"),
    ):
        if metadata[key] != frozen["inputs"]["files"][name]["sha256"]:
            raise ValueError(f"Node data or checkpoint provenance differs: {key}")
        if (
            key != "checkpoint_sha256"
            and sha(Path(frozen["inputs"]["root"]) / name) != metadata[key]
        ):
            raise ValueError(f"Data bytes differ from freeze: {key}")
    if metadata["tf32"] is not False:
        raise ValueError("Replay numeric mode changed")
    return metadata


def audit_node(base, run, step, origin, legacy):
    original, replay = base / run / f"{step:07d}", base / "audit-replay" / run / f"{step:07d}"
    a, b = (
        verify_node(original, run, step, origin, legacy),
        verify_node(replay, run, step, origin, legacy),
    )
    for key in (
        "checkpoint_sha256",
        "world_sha256",
        "panels_sha256",
        "contract_sha256",
        "code_sha256",
        "tf32",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Independent replay source provenance differs: {key}")
    if a["checkpoint_sha256"] != legacy.verify_checkpoint(base, run, step):
        raise ValueError("Checkpoint bytes differ from the original prepared receipt")
    if (original / "z1.json").read_bytes() != (replay / "z1.json").read_bytes():
        raise ValueError(f"Exact Z1 replay mismatch: {run}@{step}")
    stats = compare_ported(read(original / "ported.json"), read(replay / "ported.json"))
    return {
        "step": step,
        "passed": True,
        "outputs_exact": ["z1.json"],
        "ported_byte_exact": (original / "ported.json").read_bytes()
        == (replay / "ported.json").read_bytes(),
        "ported_comparison": comparison_contract()["ported"],
        "hashes_verified": True,
        **stats,
    }


def load_mount_helper(science):
    path = science / "orchestration/resume_path_overlap_mount.py"
    if sha(path) != read(science / "recovery-receipt.json")["helper_sha256"]:
        raise ValueError("Inherited mount helper changed")
    spec = importlib.util.spec_from_file_location("unchanged_mount_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ATTEMPT_NAME, module.HELPER_NAME = ATTEMPT_NAME, SCRIPT_NAME
    return module


def build_legacy(args):
    mounting = load_mount_helper(args.science_base)
    legacy = mounting.install_wrappers(
        mounting.load_controller(args.origin_base), args.origin_base, args.attempt_base
    )
    previous_command, previous_write = legacy.container_command, legacy.write
    audited_nodes = {}

    def command(*args):
        name, argv = previous_command(*args)
        name = name.replace("lm-z1-v3-mountfix-", "lm-z1-v3-tolerance-")
        argv[argv.index("--name") + 1] = name
        argv.extend(["--science-base", str(args_namespace.science_base)])
        return name, argv

    args_namespace = args

    def recorded(path, value):
        path = Path(path)
        if path.name == "learning.json" and path.parent.parent == args.attempt_base / "runs":
            amended = []
            for row in value:
                if row.get("stage") == "audit":
                    stats = audited_nodes[(path.parent.name, row["source_training_step"])]
                    row = {
                        **row,
                        "metrics": {
                            **row["metrics"],
                            "audit": {
                                "exact_outputs": 1,
                                "ported_cosine_tolerance_verified": 1,
                                "checkpoint_data_code_hashes_verified": 1,
                                "different_cosine_floats": stats["different_cosine_floats"],
                                "max_cosine_absolute_error": stats["max_cosine_absolute_error"],
                                "max_cosine_relative_error": stats["max_cosine_relative_error"],
                            },
                        },
                    }
                amended.append(row)
            value = amended
        if (
            path.name in {"audit.json", "partial-audit.json"}
            and path.parent.parent == args.attempt_base / "runs"
        ):
            nodes = value["nodes"]
            value = {
                **value,
                "comparison_contract": comparison_contract(),
                "different_cosine_floats": sum(node["different_cosine_floats"] for node in nodes),
                "max_cosine_absolute_error": max(
                    (node["max_cosine_absolute_error"] for node in nodes), default=0
                ),
                "max_cosine_relative_error": max(
                    (node["max_cosine_relative_error"] for node in nodes), default=0
                ),
                "ported_byte_exact": all(node["ported_byte_exact"] for node in nodes),
                "inherited_science_receipt_sha256": sha(
                    args.attempt_base / "audit-recovery-receipt.json"
                ),
            }
        if path.name == "run.json" and path.parent.parent == args.attempt_base / "runs":
            value = {
                **value,
                "job_type": "verification",
                "phase": "independent_audit_of_inherited_science",
                "inherited_science_nodes": 14,
                "inherited_selection_nodes": 14,
                "spec": {**value["spec"], "audit_comparison": comparison_contract()},
            }
        previous_write(path, value)

    legacy.container_command, legacy.write = command, recorded

    def compared(base, run, step):
        result = audit_node(base, run, step, args.origin_base, legacy)
        audited_nodes[(run, step)] = result
        return result

    legacy.audit_node = compared
    return legacy, mounting


def process_alive(metadata):
    if not metadata.exists():
        return False
    value = read(metadata)
    pid = value.get("pid")
    proc = Path(f"/proc/{pid}")
    try:
        state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        if state == "Z":
            return False
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        return not value.get("command") or any(
            str(part) in command for part in value["command"] if "orchestration/" in str(part)
        )
    except FileNotFoundError:
        return False


def prepare(args):
    base, science = args.attempt_base, args.science_base
    state = read(science / "controller-state.json")
    if (
        state.get("state") not in {"complete", "finished_with_failures"}
        or state.get("active")
        or process_alive(science / "controller-process.json")
        or process_alive(science / "tracking-process.json")
    ):
        raise RuntimeError(
            "Wait for the preceding controller and tracker to exit before audit-only recovery"
        )
    if base.exists():
        raise RuntimeError("Audit-tolerance attempt already exists")
    legacy, _ = build_legacy(args)
    # Validate the original scientific freeze and all inherited artifacts first.
    legacy.verify_freeze(args.origin_base, all_inputs=True)
    cfg = legacy.contract(args.origin_base)
    files = []
    for run in cfg["source"]["runs"]:
        history = read(science / "runs" / run / "learning.json")
        inherited = [row for row in history if row["step"] <= 28]
        if [row["step"] for row in inherited] != list(range(1, 29)) or any(
            row["optimizer_updates"] != 0 for row in inherited
        ):
            raise ValueError(
                "The preceding attempt must have completed 14 selection "
                "and 14 science nodes per run"
            )
        for step in cfg["source"]["nodes"]:
            folder = science / run / f"{step:07d}"
            verify_node(folder, run, step, args.origin_base, legacy)
            for name in ("z1.json", "ported.json", "hashes.json"):
                path = folder / name
                files.append(
                    {
                        "source": str(path),
                        "destination": str(path.relative_to(science)),
                        "sha256": sha(path),
                    }
                )
            path = science / f"selection-stage/{run}/{step:07d}/selection.json"
            if read(path).get("complete") is not True:
                raise ValueError("Inherited selection is incomplete")
            files.append(
                {
                    "source": str(path),
                    "destination": str(path.relative_to(science)),
                    "sha256": sha(path),
                }
            )
    if len(files) != 448:
        raise ValueError("Audit-only recovery must inherit all 112 scientific and selection nodes")
    archive = science / "scheduling-intervention" / ATTEMPT_NAME
    if archive.exists():
        raise RuntimeError("Preceding failure snapshot already exists")
    evidence = []
    roots = [
        path
        for path in science.glob("*")
        if path.is_file()
        and (
            path.suffix == ".log"
            or path.name.startswith(("controller-", "tracking-", "launch-", "execution-"))
            or path.name == "floating-point-audit-differences.json"
        )
    ]
    paths = (
        roots
        + [
            path
            for path in (science / "runs").rglob("*")
            if path.is_file() and path.suffix in {".json", ".log"}
        ]
        + list((science / "tracking-wandb").glob("*/state.json"))
    )
    for path in sorted(set(paths)):
        target = archive / "original-attempt" / path.relative_to(science)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        evidence.append({"path": str(path), "snapshot": str(target), "sha256": sha(path)})
    for folder in ("frozen", "inputs", "orchestration"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    for name, target in (
        ("frozen/freeze.json", args.origin_base / "frozen/freeze.json"),
        ("frozen/source", args.origin_base / "frozen/source"),
        ("inputs/grokking-reproduction-v1", args.origin_base / "inputs/grokking-reproduction-v1"),
        ("input-receipt.json", args.origin_base / "input-receipt.json"),
        ("validation", args.origin_base / "validation"),
    ):
        (base / name).symlink_to(target)
    shutil.copy2(science / "selection-adoption.json", base / "selection-adoption.json")
    for item in files:
        target = base / item["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["source"], target)
        target.chmod(0o444)
    shutil.copy2(
        science / "selection-stage/selection-summary.json",
        base / "selection-stage/selection-summary.json",
    )
    for run in cfg["source"]["runs"]:
        out = base / "runs" / run
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(science / "runs" / run / "science-process.json", out / "science-process.json")
        old_history = read(science / "runs" / run / "learning.json")
        history = []
        for row in old_history:
            if row["step"] > 28:
                continue
            stage = "select" if row["step"] <= 14 else "science"
            metrics = row["metrics"]
            node = {
                **metrics["node"],
                "seconds": 0,
                "inherited_source_execution_seconds": metrics["node"]["seconds"],
            }
            history.append(
                {
                    **row,
                    "condition": f"{stage}_inherited_from_science_attempt",
                    "stage": stage,
                    "wall_seconds": 0,
                    "physical_gpu": None,
                    "host_pid": None,
                    "metrics": {**metrics, "node": node},
                    "inherited_evidence": {
                        "source_learning": str(science / "runs" / run / "learning.json"),
                        "source_learning_sha256": sha(science / "runs" / run / "learning.json"),
                        "source_step": row["step"],
                    },
                }
            )
        write(out / "learning.json", history)
    script = base / "orchestration" / SCRIPT_NAME
    shutil.copy2(Path(__file__), script)
    script.chmod(0o444)
    receipt = {
        "scientific_batch": cfg["batch"],
        "execution_attempt": ATTEMPT_NAME,
        "origin_base": str(args.origin_base),
        "science_base": str(science),
        "attempt_base": str(base),
        "scientific_freeze_sha256": sha(args.origin_base / "frozen/freeze.json"),
        "script_sha256": sha(script),
        "source_mount_recovery_receipt_sha256": sha(science / "recovery-receipt.json"),
        "selection_adoption_sha256": sha(base / "selection-adoption.json"),
        "inherited_files": files,
        "failure_evidence": evidence,
        "new_optimizer_updates": 0,
        "new_scientific_interventions": 0,
        "planned_independent_replay_nodes": 112,
        "comparison_contract": comparison_contract(),
        "runtime_config_amendment": (
            "Only container_runtime may change across process restarts; "
            "scientific spec/config cannot change"
        ),
        "created_utc": legacy.utc(),
    }
    write(base / "audit-recovery-receipt.json", receipt)
    # The inherited launch function hashes this established receipt pathname.
    (base / "recovery-receipt.json").symlink_to("audit-recovery-receipt.json")
    write(archive / "schedule-amendment.json", receipt)
    print(
        json.dumps(
            {
                "attempt_base": str(base),
                "inherited_science_nodes": 112,
                "inherited_selection_nodes": 112,
                "planned_replay_nodes": 112,
            }
        ),
        flush=True,
    )


def verify(args):
    receipt = read(args.attempt_base / "audit-recovery-receipt.json")
    if (
        any(
            receipt[key] != str(value)
            for key, value in (
                ("origin_base", args.origin_base),
                ("science_base", args.science_base),
                ("attempt_base", args.attempt_base),
            )
        )
        or receipt["comparison_contract"] != comparison_contract()
    ):
        raise ValueError("Audit recovery path or authorized comparison changed")
    for key, path in (
        ("script_sha256", args.attempt_base / "orchestration" / SCRIPT_NAME),
        ("scientific_freeze_sha256", args.origin_base / "frozen/freeze.json"),
        ("source_mount_recovery_receipt_sha256", args.science_base / "recovery-receipt.json"),
        ("selection_adoption_sha256", args.attempt_base / "selection-adoption.json"),
    ):
        if sha(path) != receipt[key]:
            raise ValueError(f"Audit recovery lineage changed: {key}")
    for item in receipt["inherited_files"]:
        if (
            sha(item["source"]) != item["sha256"]
            or sha(args.attempt_base / item["destination"]) != item["sha256"]
        ):
            raise ValueError("Inherited scientific or selection evidence changed")
    for item in receipt["failure_evidence"]:
        if sha(item["snapshot"]) != item["sha256"]:
            raise ValueError("Preceding failure snapshot changed")
    return build_legacy(args)


class RuntimeConfigProxy:
    def __init__(self, config):
        self.config = config

    def update(self, values, *args, **kwargs):
        if set(values) == {"container_runtime"}:
            kwargs["allow_val_change"] = True
        elif kwargs.get("allow_val_change"):
            raise ValueError("Scientific config cannot be authorized to change")
        return self.config.update(values, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.config, name)


class RunProxy:
    def __init__(self, run):
        self.run, self.config = run, RuntimeConfigProxy(run.config)

    def __getattr__(self, name):
        return getattr(self.run, name)


class SDKProxy:
    def __init__(self, sdk):
        self.sdk = sdk

    def init(self, *args, **kwargs):
        return RunProxy(self.sdk.init(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self.sdk, name)


def tracker(args, legacy):
    import sys

    sys.path.insert(0, str(legacy.source(args.attempt_base) / "src"))
    import wandb

    from llm_memory_editability import interface_tracking

    base = args.attempt_base
    settings = read(legacy.source(base) / "configs/experiment-tracking-defaults.json")
    folder = base / "tracking-wandb"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "tracker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(
            base / "tracking-ready.json",
            {
                "sdk_version": wandb.__version__,
                "runtime_config_amendment": "container_runtime only",
                "created_utc": legacy.utc(),
            },
        )
        interface_tracking.drain_runs(SDKProxy(wandb), base, base / "runs", settings)


def controller(args, legacy):
    base = args.attempt_base
    legacy.verify_freeze(base, all_inputs=True)
    with (base / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed, failed = [], []
        try:
            resource = read(base / "launch-resource-check.json")
            slots = list(zip(resource["gpus"], resource["cpus"], strict=True))
            completed, failed = legacy.monitor_stage(
                base, legacy.contract(base)["source"]["runs"], slots, "audit"
            )
            if not failed:
                legacy.summary_command(base, "analyze")
                audits = [read(base / "runs" / run / "audit.json") for run in completed]
                write(
                    base / "completion-manifest.json",
                    {
                        "passed": True,
                        "scientific_batch": legacy.contract(base)["batch"],
                        "execution_attempt": ATTEMPT_NAME,
                        "runs": completed,
                        "inherited_science_nodes": 112,
                        "inherited_selection_nodes": 112,
                        "independently_replayed_nodes": 112,
                        "new_optimizer_updates": 0,
                        "new_scientific_interventions": 0,
                        "comparison_contract": comparison_contract(),
                        "different_cosine_floats": sum(
                            audit["different_cosine_floats"] for audit in audits
                        ),
                        "max_cosine_absolute_error": max(
                            audit["max_cosine_absolute_error"] for audit in audits
                        ),
                        "max_cosine_relative_error": max(
                            audit["max_cosine_relative_error"] for audit in audits
                        ),
                        "ported_byte_exact": all(audit["ported_byte_exact"] for audit in audits),
                        "summary_sha256": sha(base / "summary.json"),
                        "audit_recovery_receipt_sha256": sha(base / "audit-recovery-receipt.json"),
                        "wandb_cloud_verification": "Separate receipt required",
                        "completed_utc": legacy.utc(),
                    },
                )
            legacy.controller_state(
                base, "complete" if not failed else "finished_with_failures", {}, completed, failed
            )
        except BaseException as error:
            failure = {
                "stage": "audit_tolerance_controller",
                "error_type": type(error).__name__,
                "message": str(error),
                "created_utc": legacy.utc(),
            }
            write(base / "controller-failure.json", failure)
            for out in (base / "runs").glob("*"):
                if not (out / "complete.json").exists() and not (out / "failure.json").exists():
                    write(out / "failure.json", failure)
            legacy.controller_state(
                base, "finished_with_failures", {}, completed, [*failed, failure]
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "verify",
            "resource-check",
            "launch",
            "controller",
            "worker",
            "tracker",
        ),
    )
    parser.add_argument("--origin-base", type=Path, default=ORIGIN)
    parser.add_argument("--science-base", type=Path, default=SCIENCE)
    parser.add_argument("--attempt-base", "--base", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(2, 8)))
    parser.add_argument("--run")
    parser.add_argument("--stage", choices=("audit",))
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    args.attempt_base = args.attempt_base or args.origin_base / "execution-attempts" / ATTEMPT_NAME
    args.base = args.attempt_base
    if args.action == "prepare":
        prepare(args)
        return
    legacy, mounting = verify(args)
    if args.action == "launch":
        mounting.launch(args, legacy)
    elif args.action == "verify":
        legacy.verify_freeze(args.base, all_inputs=True)
    elif args.action == "resource-check":
        print(json.dumps(legacy.resource_check(args.base, args.gpus)), flush=True)
    elif args.action == "controller":
        controller(args, legacy)
    elif args.action == "worker":
        if args.stage != "audit":
            raise ValueError("Audit-only recovery cannot run selection or scientific stages")
        legacy.worker(args)
    else:
        tracker(args, legacy)


if __name__ == "__main__":
    main()
