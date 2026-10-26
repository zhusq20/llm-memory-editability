#!/usr/bin/env python3
"""Guarded, resumable scheduling of the authorized Knowledge use comparison.

Default validates a frozen plan without Docker operations. --execute polls the
eight-arm gate, or a source-bound explicit user amendment, before new CUDA stages.
Scientific failures are kept
and never automatically retried; resumption only adopts the original container.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_resource_snapshot as resources  # noqa: E402

DOCKER = ["docker", "--context", "lm-memory"]
IMAGE = "sha256:8eb44ee1611fe88f5a7113eb271cae9a0894fe0e95c988360b994f79e4017eba"
RELEASED_GPU4 = "824bf0b9a897ee2eb770329589ef690d3286b21fbd090af2a7b65ecbcfc44ba3"
FACTORIAL_ARMS = (
    "pd_lora",
    "sft_lora",
    "pd_full",
    "sft_full",
    "pd_down",
    "sft_down",
    "pd_down_lora",
    "sft_down_lora",
)
FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},"state":{{json .State}},'
    '"labels":{{json .Config.Labels}},"requests":{{json .HostConfig.DeviceRequests}},'
    '"restart":{{json .HostConfig.RestartPolicy}}}'
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def call(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=120, **kwargs)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def reserved_devices(requests, rows):
    by_uuid = {row["uuid"]: index for index, row in rows.items()}
    result = set()
    for request in requests or []:
        if not any("gpu" in caps for caps in request.get("Capabilities", [])):
            continue
        devices = request.get("DeviceIDs") or ["all"]
        for device in devices:
            if device == "all":
                result.update(rows)
            elif device.isdigit():
                if int(device) in rows:
                    result.add(int(device))
            elif device in by_uuid:
                result.add(by_uuid[device])
            else:
                result.update(rows)
    return result


def admissible(snapshot, lane, runtime, *, owned_uuids=(), own_container=None):
    """One exclusive stage per GPU; only the exact released GPU4 reservation is waived."""
    snapshot = dict(snapshot)
    release = []
    for record in snapshot["containers"]:
        if (
            record["id"] == RELEASED_GPU4
            and resources.released_created(record, [RELEASED_GPU4])
            and reserved_devices(record["requests"], snapshot["rows"]) == {4}
        ):
            release.append(RELEASED_GPU4)
    if own_container:
        # A created stage we already own reserves this lane while its source is copied.
        snapshot["containers"] = [c for c in snapshot["containers"] if c["id"] != own_container]
    rows, blocked = resources.knowledge_view(snapshot, reserved_devices, release)
    row = rows.get(lane["gpu"])
    reasons = list(blocked.get(lane["gpu"], []))
    if row is None or row["uuid"] != lane["gpu_uuid"]:
        reasons.append("GPU identity differs")
    elif (
        row["used"] > runtime.get("maximum_idle_gpu_used_mib", 512)
        or row["free"] < runtime.get("minimum_free_gpu_mib", 85000)
        or row["utilization"] > 5
    ):
        reasons.append("GPU is not independently idle with the frozen memory reserve")
    owned = set(owned_uuids)
    if len(owned) > 4 or (lane["gpu_uuid"] not in owned and len(owned) >= 4):
        reasons.append("Four user GPUs are already occupied")
    return reasons


def source_archive(path, expected_sha256):
    """Repack an already frozen repository tar with root-owned 0444/0555 entries."""
    if sha(path) != expected_sha256:
        raise ValueError("Frozen source tar changed")
    files = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe source archive member")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Only regular source files are accepted")
            parts = name.parts[1:] if name.parts[0] == "project" else name.parts
            relative = str(PurePosixPath(*parts))
            if not parts or relative in files or relative == ".launch-source.json":
                raise ValueError("Repeated or reserved source archive path")
            files[relative] = archive.extractfile(member).read()
    if not files:
        raise ValueError("Empty source archive")
    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    payload_files = {**files, ".launch-source.json": json.dumps(manifest, sort_keys=True).encode()}
    directories = {"project"}
    for name in payload_files:
        directories.update(str(p) for p in PurePosixPath("project", name).parents if str(p) != ".")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(directories):
            entry = tarfile.TarInfo(name)
            entry.type, entry.mode, entry.uid, entry.gid = tarfile.DIRTYPE, 0o555, 0, 0
            archive.addfile(entry)
        for name, content in sorted(payload_files.items()):
            entry = tarfile.TarInfo(f"project/{name}")
            entry.size, entry.mode, entry.uid, entry.gid = len(content), 0o444, 0, 0
            archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue(), manifest, files


def credentials(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("Private credential file mode 0600/0400 is required")
    value = path.read_bytes().strip()
    if not value:
        raise ValueError("Empty tracking credential")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            entry = tarfile.TarInfo(name)
            entry.type, entry.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(entry)
        entry = tarfile.TarInfo("run/secrets/wandb")
        entry.size, entry.mode, entry.uid, entry.gid = len(value), 0o400, os.getuid(), os.getgid()
        archive.addfile(entry, io.BytesIO(value))
    return buffer.getvalue()


def pipeline(config):
    jobs = {}
    parallel_preparation = config.get("parallel_preparation", False)

    def add(name, dependencies, **spec):
        jobs[name] = {"dependencies": list(dependencies), **spec}
        return name

    def prepare(cohort, dependencies, *, prepare_early=False):
        prep = add(
            f"{cohort}-prepare",
            [] if prepare_early else dependencies,
            kind="gpu",
            cohort=cohort,
            action="prepare",
        )
        check = add(
            f"{cohort}-preflight",
            [prep, *dependencies] if prepare_early else [prep],
            kind="gpu",
            cohort=cohort,
            action="preflight",
        )
        return add(f"{cohort}-baseline", [check], kind="gpu", cohort=cohort, action="baseline")

    def trajectories(cohort, objective, stage, dependencies, nodes):
        ends = []
        for arm in ("A", "B", "C"):
            name = f"{cohort}-{objective}-{arm}-{stage}"
            spec = dict(kind="gpu", cohort=cohort, objective=objective, arm=arm, stage=stage)
            trained = add(f"{name}-train", dependencies, action="train", nodes=nodes, **spec)
            previous = trained
            for node in nodes:
                # Serialize node publication within one W&B run to avoid summary races.
                previous = add(
                    f"{name}-evaluate-{node}", [previous], action="evaluate", nodes=[node], **spec
                )
                ends.append(previous)
        return ends

    development = prepare("development", [])
    dev_ends = trajectories(
        "development", "pd", "addition", [development], config["development_token_nodes"]
    )
    chosen = add("select-budget", dev_ends, kind="cpu", action="select")
    formal = prepare("formal", [chosen], prepare_early=parallel_preparation)
    pd = trajectories("formal", "pd", "addition", [formal], ["formal"])
    sft = trajectories("formal", "sft", "addition", pd, ["formal"])
    rewritten = []
    for objective in ("pd", "sft"):
        rewritten.extend(trajectories("formal", objective, "rewrite", sft, ["rewrite"]))
    calibrated = add(
        "natural-calibrate",
        [] if parallel_preparation else rewritten,
        kind="gpu",
        cohort="natural",
        action="calibrate",
    )
    reviewed = add("natural-quality-review", [calibrated], kind="cpu", action="quality-review")
    natural = prepare("natural", [reviewed, *rewritten] if parallel_preparation else [reviewed])
    pd = trajectories("natural", "pd", "addition", [natural], ["natural"])
    sft = trajectories("natural", "sft", "addition", pd, ["natural"])
    add("report", sft, kind="cpu", action="report")
    return jobs


def resolved_nodes(spec, selection):
    result = []
    for node in spec.get("nodes", []):
        if isinstance(node, int):
            result.append(node)
        elif node == "rewrite":
            result.append(selection["rewrite_selected_tokens"])
        else:
            result.append(selection["formal_selected_tokens"])
    return result


def runner_command(plan, spec, selection):
    common = ["--config", f"/project/{plan['scientific_config']}", "--batch", "/batch"]
    cohort, action = spec["cohort"], spec["action"]
    if action == "preflight":
        return [
            "python",
            "-u",
            "scripts/check_knowledge_use_coverage_cuda.py",
            *common,
            "--cohort",
            cohort,
        ]
    natural = cohort == "natural"
    script = "run_knowledge_use_coverage_natural.py" if natural else "run_knowledge_use_coverage.py"
    argv = ["python", "-u", f"scripts/{script}", *common, "--action", action]
    argv += ["--factorial-batch", "/factorial"] if natural else ["--cohort", cohort]
    if action in ("train", "evaluate"):
        argv += ["--objective", spec["objective"], "--arm", spec["arm"]]
        if not natural:
            argv += ["--stage", spec["stage"]]
        argv += ["--nodes", *map(str, resolved_nodes(spec, selection))]
        if action == "train" and spec["stage"] == "rewrite":
            name = f"{cohort}-{spec['objective']}-{spec['arm']}-addition"
            argv += [
                "--parent",
                f"/batch/runs/{name}/node-{selection['formal_selected_tokens']}/resume.pt",
            ]
    return argv


BOOT = """import hashlib,json,os,stat,sys
from pathlib import Path
root=Path('/project')
manifest=json.loads((root/'.launch-source.json').read_text())
for name, expected in manifest.items():
    path=root/name
    info=path.stat()
    if info.st_uid!=0 or info.st_gid!=0 or stat.S_IMODE(info.st_mode)!=0o444:
        raise RuntimeError('Source ownership or mode differs: '+name)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Source content differs: '+name)
    for parent in path.parents:
        if parent==Path('/'):
            break
        info=parent.stat()
        if info.st_uid!=0 or info.st_gid!=0 or stat.S_IMODE(info.st_mode)!=0o555:
            raise RuntimeError('Source directory is writable: '+str(parent))
os.umask(0o022)
os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip()
os.execvp(sys.argv[1],sys.argv[1:])
"""


def container_command(plan, name, spec, lane, selection):
    runtime = plan["runtime"]
    label = f"{plan.get('container_prefix', 'kuc-v1')}-{name}"
    argv = [
        *DOCKER,
        "create",
        "--name",
        label,
        "--restart",
        "no",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpuset-cpus",
        lane["cpus"],
        "--cpus",
        "10",
        "--memory",
        "128g",
        "--memory-swap",
        "128g",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,size=16g",
        "--network",
        "host",
        "--workdir",
        "/project",
        "--gpus",
        f"device={lane['gpu_uuid']}",
    ]
    for key, value in {
        "lm-memory.batch": plan["batch"],
        "lm-memory.job": name,
        "lm-memory.source": plan["source_tar_sha256"],
    }.items():
        argv += ["--label", f"{key}={value}"]
    environment = {
        "PYTHONPATH": "/deps:/project/src:/project/scripts",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "10",
        "MKL_NUM_THREADS": "10",
        "HF_HOME": "/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/cache",
        "WANDB_CACHE_DIR": "/tmp/wandb-cache",
        "WANDB_CONSOLE": "off",
        "WANDB_SILENT": "true",
        "WANDB_DISABLE_CODE": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "MASTER_PORT": str(runtime.get("master_port_base", 29620) + lane["gpu"]),
    }
    for key, value in environment.items():
        argv += ["--env", f"{key}={value}"]
    for source, target, readonly in (
        (plan["output"], "/batch", False),
        (runtime["hf_cache"], "/hf", True),
        (runtime["deps"], "/deps", True),
        (plan["factorial_batch"], "/factorial", True),
        (runtime["upstream"], "/factorial/upstream", True),
    ):
        argv += [
            "--mount",
            f"type=bind,source={source},target={target}" + (",readonly" if readonly else ""),
        ]
    return label, [
        *argv,
        runtime["image"],
        "python",
        "-c",
        BOOT,
        *runner_command(plan, spec, selection),
    ]


def factorial_gate(plan):
    """Read each JSON and its digest from the same bytes; defer changing snapshots."""
    observed = {}

    def observed_read(path, default=None):
        path = Path(path)
        if path not in observed:
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                raw = None
            observed[path] = raw
        raw = observed[path]
        return json.loads(raw) if raw is not None else default

    def observed_sha(path):
        path = Path(path)
        observed_read(path)
        return hashlib.sha256(observed[path]).hexdigest() if observed[path] is not None else None

    result = _factorial_gate(plan, observed_read, observed_sha)
    for path, before in observed.items():
        try:
            after = path.read_bytes()
        except FileNotFoundError:
            after = None
        if before != after:
            return {
                "passed": False,
                "state": "waiting_factorial",
                "reasons": ["factorial receipts changed during read; retry a consistent snapshot"],
            }
    return result


def _factorial_gate(plan, read, sha):
    """Require local endpoints and the independently verified complete cloud analysis."""
    mode = plan.get("start_mode", "after_complete_factorial")
    if mode == "user_authorized_full_before_factorial":
        return _authorized_full_start_gate(plan, read, sha)
    if mode != "after_complete_factorial":
        return {"passed": False, "state": "needs_attention", "reasons": ["Unknown start mode"]}
    summary = read(plan["factorial_analysis"], {})
    collector = read(plan["factorial_collector_receipt"], {})
    if collector.get("state") == "needs_attention":
        return {
            "passed": False,
            "state": "needs_attention",
            "reasons": ["factorial collector needs attention"],
        }
    if not (
        collector.get("state") == "complete"
        and collector.get("local_and_cloud_verified") is True
        and collector.get("training_processes_modified") is False
        and set(collector.get("arms", {})) == set(FACTORIAL_ARMS)
        and set(collector.get("container_exits", {})) == set(FACTORIAL_ARMS)
        and all(
            collector["container_exits"][arm] == {"status": "exited", "exit_code": 0}
            and collector["arms"][arm].get("exposure") == 1489200
            and all(
                collector["arms"][arm].get(key) is True
                for key in ("train_complete", "audit_passed", "evaluation_complete")
            )
            for arm in FACTORIAL_ARMS
        )
    ):
        return {
            "passed": False,
            "state": "waiting_factorial",
            "reasons": ["eight original containers have not all finished"],
        }
    reasons = []
    if not (
        summary.get("complete") is True
        and summary.get("status") == "complete"
        and set(summary.get("complete_arms", [])) == set(FACTORIAL_ARMS)
        and summary.get("cloud", {}).get("complete") is True
    ):
        return {
            "passed": False,
            "state": "waiting_factorial",
            "reasons": ["eight-arm analysis incomplete"],
        }
    snapshot = Path(
        plan.get(
            "factorial_cloud_snapshot",
            Path(plan["factorial_analysis"]).with_name("cloud-snapshot.json"),
        )
    )
    if not snapshot.is_file() or summary["cloud"].get("snapshot_sha256") != sha(snapshot):
        return {
            "passed": False,
            "state": "needs_attention",
            "reasons": ["cloud snapshot identity differs from unified analysis"],
        }
    batch = Path(plan["factorial_batch"])
    weights = {}
    for arm in FACTORIAL_ARMS:
        directory = batch / "runs" / arm
        complete = read(directory / "complete.json", {})
        audit = read(directory / "audit/audit.json", {})
        evaluation = read(directory / "paper-eval/evaluate.json", {})
        weight = complete.get("weights_sha256")
        weights[arm] = weight
        if (
            complete.get("exposure") != 1489200
            or complete.get("smoke", True)
            or not audit.get("passed")
            or not evaluation.get("complete")
            or not weight
            or evaluation.get("source", {}).get("weights_sha256") != weight
            or summary.get("arms", {}).get(arm, {}).get("source", {}).get("weights_sha256")
            != weight
            or not summary["cloud"].get("arms", {}).get(arm, {}).get("terminal_verified")
        ):
            reasons.append(f"Local/cloud endpoint identity incomplete: {arm}")
    if reasons:
        return {"passed": False, "state": "needs_attention", "reasons": reasons}
    receipt = read(plan["scope_decision_receipt"], {})
    if not receipt:
        return {
            "passed": False,
            "state": "waiting_scope_decision",
            "reasons": ["scope receipt missing"],
        }
    if (
        receipt.get("decision") != "approved"
        or receipt.get("scope") != plan["scope"]
        or receipt.get("scientific_config_sha256") != plan["scientific_config_sha256"]
        or receipt.get("factorial_endpoint_weights") != weights
        or receipt.get("analysis_sha256") != sha(plan["factorial_analysis"])
        or receipt.get("collector_sha256") != sha(plan["factorial_collector_receipt"])
        or receipt.get("cloud_snapshot_sha256") != sha(snapshot)
        or not receipt.get("decision_rationale")
    ):
        return {"passed": False, "state": "needs_attention", "reasons": ["scope receipt differs"]}
    return {
        "passed": True,
        "state": "ready",
        "factorial_endpoint_weights": weights,
        "analysis_sha256": sha(plan["factorial_analysis"]),
        "collector_sha256": sha(plan["factorial_collector_receipt"]),
        "cloud_snapshot_sha256": sha(snapshot),
        "scope_receipt_sha256": sha(plan["scope_decision_receipt"]),
        "utc": utc(),
    }


def _authorized_full_start_gate(plan, read, sha):
    """Apply the user's dated full-scope amendment without inventing eight endpoints."""
    path = plan["scope_decision_receipt"]
    receipt = read(path, {})
    if not receipt:
        return {
            "passed": False,
            "state": "waiting_scope_decision",
            "reasons": ["Explicit full-scope start amendment missing"],
        }
    expected = {
        "decision": "approved",
        "authorization_source": "user_message",
        "authorization_text": "立即开始，固定全参数",
        "supersedes_complete_factorial_prerequisite": True,
        "factorial_complete_at_decision": False,
        "batch": plan["batch"],
        "scope": "full",
        "gpu_lanes": [4, 6, 8],
        "scientific_config_sha256": plan["scientific_config_sha256"],
        "source_tar_sha256": plan["source_tar_sha256"],
    }
    if (
        plan["scope"] != "full"
        or [lane["gpu"] for lane in plan["lanes"]] != [4, 6, 8]
        or any(receipt.get(key) != value for key, value in expected.items())
        or not receipt.get("decision_rationale")
        or not receipt.get("utc")
    ):
        return {
            "passed": False,
            "state": "needs_attention",
            "reasons": ["Explicit start amendment differs from the frozen full-scope plan"],
        }
    evidence = receipt.get("evidence", {})
    if set(evidence) != {"analysis", "collector", "cloud"}:
        return {"passed": False, "state": "needs_attention", "reasons": ["Start evidence missing"]}
    for item in evidence.values():
        if not item.get("path") or not item.get("sha256") or sha(item["path"]) != item["sha256"]:
            return {
                "passed": False,
                "state": "needs_attention",
                "reasons": ["Preserved start-decision evidence changed"],
            }
    summary = read(evidence["analysis"]["path"], {})
    collector = read(evidence["collector"]["path"], {})
    expected_arms = set(FACTORIAL_ARMS) - {"pd_lora"}
    if (
        summary.get("complete") is not False
        or set(summary.get("complete_arms", [])) != expected_arms
        or collector.get("state") != "waiting_for_endpoints"
        or collector.get("training_processes_modified") is not False
        or collector.get("failed_arms") != []
        or set(collector.get("arms", {})) != set(FACTORIAL_ARMS)
        or collector["arms"]["pd_lora"].get("train_complete") is not False
        or not all(
            all(
                collector["arms"][arm].get(key) is True
                for key in ("train_complete", "audit_passed", "evaluation_complete")
            )
            for arm in expected_arms
        )
    ):
        return {
            "passed": False,
            "state": "needs_attention",
            "reasons": ["Start evidence does not match the reviewed seven-of-eight state"],
        }
    return {
        "passed": True,
        "state": "ready",
        "mode": "user_authorized_full_before_factorial",
        "factorial_complete_at_decision": False,
        "scope": "full",
        "scope_receipt_sha256": sha(path),
        "utc": utc(),
    }


def validate(plan):
    if plan.get("materialize_command"):
        raise ValueError("Materialize controlled data before starting this controller")
    if plan["batch"] != "knowledge-use-coverage-v1" or plan["runtime"]["image"] != IMAGE:
        raise ValueError("Unexpected batch or non-frozen LM image")
    output, factorial = Path(plan["output"]).resolve(), Path(plan["factorial_batch"]).resolve()
    if (
        not Path(plan["output"]).is_absolute()
        or output == factorial
        or output.is_relative_to(factorial)
        or factorial.is_relative_to(output)
    ):
        raise ValueError("A separate absolute new batch directory is required")
    for key in ("hf_cache", "deps", "upstream"):
        protected = Path(plan["runtime"][key]).resolve()
        if (
            protected == output
            or protected.is_relative_to(output)
            or output.is_relative_to(protected)
        ):
            raise ValueError("Writable output overlaps a read-only data or dependency mount")
    lanes = plan["lanes"]
    # 2026-10-26: user moved this batch to CUDA2/5, then added CUDA6.
    # Keep historical plans valid without permitting mixed expanded GPU pools.
    selected_gpus = {x["gpu"] for x in lanes}
    if (
        not lanes
        or len(lanes) > 4
        or not (selected_gpus <= {3, 4, 6, 8} or selected_gpus in ({2, 5}, {2, 5, 6}))
    ):
        raise ValueError("At most four authorized historical lanes or CUDA2/5[/6] are allowed")
    if len({x["gpu_uuid"] for x in lanes}) != len(lanes) or len({x["gpu"] for x in lanes}) != len(
        lanes
    ):
        raise ValueError("GPU lanes must be distinct")
    used_cpus = set()
    for lane in lanes:
        selected = set()
        for section in lane["cpus"].split(","):
            limits = [int(x) for x in section.split("-")]
            if len(limits) not in (1, 2) or min(limits) < 0 or limits[-1] < limits[0]:
                raise ValueError("Invalid CPU lane")
            selected.update(range(limits[0], limits[-1] + 1))
        if selected & used_cpus:
            raise ValueError("Each parallel GPU lane requires distinct CPU reservations")
        used_cpus.update(selected)
    if set(plan.get("released_created_ids", [RELEASED_GPU4])) - {RELEASED_GPU4}:
        raise ValueError("Only the exact retired GPU4 reservation may be excluded")
    payload, manifest, files = source_archive(plan["source_tar"], plan["source_tar_sha256"])
    config_bytes = files[plan["scientific_config"]]
    if hashlib.sha256(config_bytes).hexdigest() != plan["scientific_config_sha256"]:
        raise ValueError("Frozen scientific configuration identity differs")
    config = json.loads(config_bytes)
    if config["scope"] != plan["scope"] or config["batch"] != plan["batch"]:
        raise ValueError("Scope must be fixed by configuration and the later decision receipt")
    if config["scope"] not in {"down", "full"}:
        raise ValueError(
            "This frozen runner supports only independently taught dense down/full scopes"
        )
    development, formal = config["development_token_nodes"], config["formal_token_nodes"]
    if (
        development != [200192, 400384]
        or formal != [800768, 1601536]
        or config.get("rewrite_token_fraction") != 0.25
    ):
        raise ValueError("Exactly two fixed, token-aligned development/formal candidates required")
    if config.get("natural_token_nodes") != formal:
        raise ValueError("Natural candidates must share the frozen formal token candidates")
    for name in (
        "run_knowledge_use_coverage.py",
        "run_knowledge_use_coverage_natural.py",
        "check_knowledge_use_coverage_cuda.py",
        "analyze_knowledge_use_coverage.py",
    ):
        if f"scripts/{name}" not in manifest:
            raise ValueError(f"Frozen runner missing: {name}")
    return config, payload, manifest


def verified_evaluation(batch, directory, cohort, stage, config, analysis, *, context=False):
    """Require raw coverage and the same evaluation contract used by the final report."""
    if cohort == "natural":
        first, last = config["natural_validation_cases"]
        result = analysis._load_natural(directory, last - first)
    else:
        expected = analysis._expected(batch, cohort, stage)
        result = analysis._load_controlled(directory, expected, not context)
    if result["status"] != "complete":
        raise ValueError(f"Evaluation contract incomplete: {directory}: {result['problems']}")
    if cohort == "natural":
        expected_cases = {f"author-csv-row-{index:04d}" for index in range(first, last)}
        for category in ("natural_closed", "natural_oracle_context"):
            actual_cases = {r["case_id"] for r in result["rows"] if r["category"] == category}
            if actual_cases != expected_cases:
                raise ValueError("Natural application case identities differ from frozen split")
        source = result["summary"].get("source", {})
        if source.get("model_revision") != config["model_revision"] or source.get(
            "application_case_indices"
        ) != [first, last]:
            raise ValueError("Natural evaluation source contract differs")
    else:
        if result["summary"].get("with_context") is not context:
            raise ValueError("Evaluation context mode differs")
        expected_by_key = {analysis.row_key(row): row for row in expected}
        counted = {}
        for row in result["rows"]:
            key = analysis.row_key(row)
            original = expected_by_key.get(key)
            if row.get("mode") == "self_generated_chain":
                original = expected_by_key.get((key[0], "direct"))
            if original is None or any(
                row.get(field) != original.get(field) for field in ("question", "gold", "category")
            ):
                raise ValueError("Evaluated question, answer or category differs from frozen input")
            group = analysis._group(row)
            count = counted.setdefault(group, {"n": 0, "correct": 0, "unparsed": 0})
            count["n"] += 1
            count["correct"] += row["correct"]
            count["unparsed"] += row.get("parse_status") == "unparsed"
        groups = result["summary"].get("groups", {})
        if groups.keys() != counted.keys():
            raise ValueError("Evaluation summary group coverage differs from raw predictions")
        for group, count in counted.items():
            reported = groups[group]
            if (
                any(reported.get(key) != value for key, value in count.items())
                or abs(reported.get("accuracy", -1) - count["correct"] / count["n"]) > 1e-10
            ):
                raise ValueError("Evaluation summary counts differ from raw predictions")
    return result


def verify_outputs(batch, spec, selection, config, analysis):
    """A zero process exit is insufficient without scientific completion markers."""
    cohort, action = spec["cohort"], spec["action"]
    if action == "prepare":
        out = batch / "data" / cohort
        done = read(out / "prepare-complete.json", {})
        required = {f"prepared-{arm}.jsonl" for arm in "ABC"} | {"prepared-replay.jsonl"}
        if cohort != "natural":
            required.add("prepared-rewrite.jsonl")
        if not required <= done.get("sha256", {}).keys():
            raise ValueError("Preparation has no content manifest")
        for name, expected in done["sha256"].items():
            if Path(name).name != name or sha(out / name) != expected:
                raise ValueError("Prepared data hash differs")
    elif action == "calibrate":
        out = batch / "data/natural-development"
        done = read(out / "quality-summary.json", {})
        if (
            done.get("complete") is not True
            or done.get("request_count") != 96
            or done.get("status") != "awaiting_manual_semantic_review"
        ):
            raise ValueError("Natural development calibration is incomplete")
        for name, key in (
            ("raw-questions.jsonl", "raw_questions_sha256"),
            ("parsed-questions.jsonl", "parsed_questions_sha256"),
            ("question-requests.jsonl", "question_requests_sha256"),
        ):
            if sha(out / name) != done.get(key):
                raise ValueError("Natural calibration evidence identity differs")
    elif action == "preflight":
        done = read(batch / "engineering" / cohort / "preflight.json", {})
        if (
            done.get("status") != "passed"
            or done.get("passed") is not True
            or done.get("config") != analysis.resolved_run_config(config, cohort)
            or done.get("scientific_updates") != 0
        ):
            raise ValueError("Actual CUDA preflight has not passed")
        for objective in ("pd", "sft"):
            out = batch / "engineering" / cohort / objective
            exercise, reload = read(out / "exercise.json", {}), read(out / "reload.json", {})
            if (
                exercise.get("passed") is not True
                or not exercise.get("restoration_exact")
                or not exercise.get("teacher_unchanged")
                or exercise.get("weights_sha256") != sha(out / "resume.pt")
                or not all(
                    reload.get(key) is True
                    for key in (
                        "passed",
                        "full_trainable_weights_exact",
                        "logits_and_greedy_exact",
                        "new_process",
                    )
                )
            ):
                raise ValueError("Independent engineering exercise/reload evidence incomplete")
    elif action == "baseline":
        directory = batch / "baseline" / cohort
        if cohort == "natural":
            verified_evaluation(batch, directory, cohort, "addition", config, analysis)
        else:
            for side, stage, context in (
                ("closed", "addition", False),
                ("context", "addition", True),
                ("rewrite-context", "rewrite", True),
            ):
                verified_evaluation(
                    batch, directory / side, cohort, stage, config, analysis, context=context
                )
    else:
        name = f"{cohort}-{spec['objective']}-{spec['arm']}-{spec['stage']}"
        directory = batch / "runs" / name
        nodes = resolved_nodes(spec, selection)
        done = read(directory / "complete.json", {})
        trajectory_nodes = config["development_token_nodes"] if cohort == "development" else nodes
        contract = analysis.validate_run_contract(
            directory,
            config,
            cohort,
            spec["objective"],
            spec["arm"],
            spec["stage"],
            trajectory_nodes,
            selection.get("formal_selected_tokens"),
        )
        if contract["status"] != "complete":
            raise ValueError(f"Scientific trajectory contract differs: {contract['problems']}")
        for node in nodes:
            cost = analysis.node_cost(directory, node)
            if (
                not cost["exact_node_observed"]
                or not cost["stage_complete"]
                or cost["training_failure"]
                or not isinstance(cost["step"], int)
                or cost["step"] * config["main_tokens_per_update"] != node
                or cost["replay_tokens"] != cost["step"] * config["replay_tokens_per_update"]
            ):
                raise ValueError("Exact main/replay node budget is incomplete or differs")
        if action == "train":
            if (
                done.get("nodes") != nodes
                or done.get("smoke", True)
                or done.get("supervised_tokens") != max(nodes)
                or not all(
                    (directory / f"node-{n}" / file).is_file()
                    for n in nodes
                    for file in ("resume.pt", "reload-reference.pt")
                )
            ):
                raise ValueError("Scientific training endpoint differs")
        else:
            tracking = read(batch / "tracking-wandb" / name / "state.json", {})
            for node in nodes:
                current = directory / f"node-{node}"
                audit = read(current / "audit.json", {})
                evaluation = read(current / "evaluation/summary.json", {})
                cloud = read(current / "cloud-verification.json", {})
                if (
                    not audit.get("passed")
                    or not audit.get("weights_sha256")
                    or audit.get("weights_sha256") != sha(current / "resume.pt")
                    or not evaluation.get("complete")
                    or not cloud.get("passed")
                    or cloud.get("state") != "finished"
                    or not cloud.get("config_equal")
                    or not cloud.get("node_summary_equal")
                    or not tracking.get("run_id")
                    or cloud.get("run_id") != tracking["run_id"]
                    or cloud.get("url") != tracking.get("url")
                ):
                    raise ValueError("Independent reload, evaluation or cloud identity incomplete")
                verified_evaluation(
                    batch, current / "evaluation", cohort, spec["stage"], config, analysis
                )
                if cohort == "natural" and (
                    evaluation.get("source", {}).get("weights_sha256") != audit["weights_sha256"]
                    or evaluation.get("source", {}).get("arm")
                    != f"{spec['objective']}_{spec['arm']}"
                    or evaluation.get("source", {}).get("phase") != "after"
                ):
                    raise ValueError("Natural evaluation endpoint identity differs")
    return True


class Controller:
    def __init__(self, plan, config, payload, manifest):
        self.plan, self.config, self.payload, self.manifest = plan, config, payload, manifest
        self.batch = Path(plan["output"])
        self.jobs = pipeline(config)
        self.leases = {}
        self.selection = read(self.batch / "budget-selection.json", {})
        self.lock_directory = Path(
            plan.get("lock_directory", f"/tmp/lm-memory-gpu-locks-{os.getuid()}")
        )
        self.state = read(
            self.batch / "controller-state.json",
            {
                "batch": plan["batch"],
                "plan_sha256": fingerprint(plan),
                "state": "waiting_factorial",
                "jobs": {name: {"state": "pending"} for name in self.jobs},
                "created_utc": utc(),
            },
        )
        if (
            self.state["plan_sha256"] != fingerprint(plan)
            or self.state["jobs"].keys() != self.jobs.keys()
        ):
            raise ValueError("Existing controller plan differs; preserve original attempts")
        choice = self.state["jobs"]["select-budget"]
        if choice["state"] == "complete" and choice.get("selection_sha256") != sha(
            self.batch / "budget-selection.json"
        ):
            raise ValueError("Previously selected development budget changed")

    def save(self):
        self.state["updated_utc"] = utc()
        write(self.batch / "controller-state.json", self.state)

    def lease(self, lane):
        gpu = lane["gpu"]
        if gpu in self.leases:
            return
        self.lock_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle = (self.lock_directory / f"gpu-{gpu}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            handle.close()
            raise
        self.leases[gpu] = handle

    def inspect(self, job):
        name = job.get("container_id") or job["container"]
        actual = json.loads(call([*DOCKER, "inspect", "--format", FORMAT, name]).stdout)
        if job.get("container_id") and actual["id"] != job["container_id"]:
            raise ValueError("Container identity changed; no replacement allowed")
        labels = actual.get("labels") or {}
        if (
            labels.get("lm-memory.batch") != self.plan["batch"]
            or labels.get("lm-memory.job") != job["job_id"]
            or labels.get("lm-memory.source") != self.plan["source_tar_sha256"]
            or actual["restart"]["Name"] != "no"
        ):
            raise ValueError("Container provenance or restart policy differs")
        return actual

    def fail(self, name, error):
        self.state["jobs"][name].update(state="needs_attention", error=str(error), failed_utc=utc())
        self.state.update(state="needs_attention", failed_job=name)
        self.save()

    def poll(self):
        for name, job in self.state["jobs"].items():
            if job["state"] == "cpu_running":
                self.fail(name, "Interrupted CPU stage requires review; no automatic retry")
                continue
            if job["state"] not in ("running", "starting", "created", "creating"):
                continue
            try:
                actual = self.inspect(job)
                state = actual["state"]
                if state["Status"] == "running" and job["state"] in ("running", "starting"):
                    self.lease(job["lane"])
                    job.update(state="running", container_id=actual["id"])
                elif state["Status"] == "exited":
                    folder = self.batch / "execution" / name
                    with (folder / "stdout.log").open("w") as log:
                        subprocess.run(
                            [*DOCKER, "logs", actual["id"]],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=60,
                            check=True,
                        )
                    write(folder / "exit.json", {"code": state["ExitCode"], "utc": utc()})
                    if state["ExitCode"] or state.get("OOMKilled"):
                        raise RuntimeError("Original stage failed; no automatic scientific retry")
                    verify_outputs(
                        self.batch,
                        self.jobs[name],
                        self.selection,
                        self.config,
                        self.load_analysis(),
                    )
                    job.update(state="complete", completed_utc=utc(), container_id=actual["id"])
                    handle = self.leases.pop(job["lane"]["gpu"], None)
                    if handle:
                        handle.close()
                elif state["Status"] == "created" and state["Pid"] == 0:
                    self.lease(job["lane"])
                    job.update(state="created", container_id=actual["id"])
                else:
                    raise ValueError("Ambiguous original container state; no restart")
            except Exception as error:
                self.fail(name, error)
        self.save()

    def refresh_gate(self):
        commands = self.plan.get("gate_refresh_commands", [])
        interval = self.plan.get("gate_refresh_seconds", 300)
        if not commands or time.time() - self.state.get("last_gate_refresh_unix", 0) < interval:
            return
        if not self.plan.get("gate_refresh_cpu_only"):
            raise ValueError("Gate refresh commands must be explicitly marked CPU-only")
        self.state["last_gate_refresh_unix"] = time.time()
        number = self.state.get("gate_refresh_attempts", 0) + 1
        self.state["gate_refresh_attempts"] = number
        self.save()
        for index, command in enumerate(commands):
            folder = self.batch / "gate-refresh" / f"{number:05d}" / str(index)
            write(folder / "command.json", {"argv": command, "utc": utc()})
            with (folder / "stdout.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            write(folder / "exit.json", {"code": result.returncode, "utc": utc()})
            if result.returncode:
                raise resources.ResourceUnavailable(
                    "Read-only gate refresh failed; admission blocked"
                )

    def verify_host_source(self):
        source = Path(self.plan.get("source_dir", self.batch / "source"))
        for name, expected in self.manifest.items():
            if sha(source / name) != expected:
                raise ValueError("Frozen host source differs from container source")
        return source

    def load_analysis(self):
        source = self.verify_host_source()
        sys.path[:0] = [str(source / "src"), str(source / "scripts")]
        path = source / "scripts/analyze_knowledge_use_coverage.py"
        module_spec = importlib.util.spec_from_file_location("frozen_coverage_analysis", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def cpu(self, name):
        spec, job = self.jobs[name], self.state["jobs"][name]
        try:
            source = self.verify_host_source()
            action = spec["action"]
            if (
                action == "quality-review"
                and not (self.batch / "data/natural-development/calibration-approved.json").exists()
            ):
                self.state["state"] = "waiting_quality_review"
                self.save()
                return
            job.update(state="cpu_running", started_utc=utc())
            self.save()
            if action == "select":
                module = self.load_analysis()
                selected = module.select_development_node(self.batch, self.config)
                if selected.get("status") != "complete":
                    raise ValueError("Development selection is incomplete or invalid")
                index = self.config["development_token_nodes"].index(
                    selected["development_selected_tokens"]
                )
                formal = self.config["formal_token_nodes"][index]
                if selected["formal_selected_tokens"] != formal or selected[
                    "rewrite_selected_tokens"
                ] != int(formal * self.config["rewrite_token_fraction"]):
                    raise ValueError("Development-to-formal token mapping differs")
                if (self.batch / "budget-selection.json").exists() and self.selection != selected:
                    raise ValueError("Existing selected budget differs")
                write(self.batch / "budget-selection.json", selected)
                self.selection = selected
                job["selection_sha256"] = sha(self.batch / "budget-selection.json")
            elif action == "quality-review":
                sys.path[:0] = [str(source / "src"), str(source / "scripts")]
                path = source / "scripts/run_knowledge_use_coverage_natural.py"
                module_spec = importlib.util.spec_from_file_location(
                    "frozen_natural_calibration", path
                )
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                module.approved_calibration(module.natural_config(self.config), self.batch)
                job["receipt_sha256"] = sha(
                    self.batch / "data/natural-development/calibration-approved.json"
                )
            else:
                command = [
                    sys.executable,
                    str(source / "scripts/analyze_knowledge_use_coverage.py"),
                    "--batch",
                    str(self.batch),
                    "--config",
                    str(source / self.plan["scientific_config"]),
                    "--output",
                    str(self.batch),
                ]
                folder = self.batch / "execution" / name
                write(folder / "command.json", {"argv": command, "utc": utc()})
                environment = dict(os.environ, PYTHONPATH=f"{source / 'src'}:{source / 'scripts'}")
                with (folder / "stdout.log").open("w") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=1800
                    )
                write(folder / "exit.json", {"code": result.returncode, "utc": utc()})
                if result.returncode:
                    raise RuntimeError("CPU stage failed; preserved without retry")
                if action == "report" and not read(self.batch / "summary.json", {}).get("complete"):
                    raise ValueError("Final analysis is still incomplete")
            job.update(state="complete", completed_utc=utc())
            self.save()
        except Exception as error:
            self.fail(name, error)

    def own_uuids(self, snap):
        own = resources.owned_gpu_uuids(snap)
        for record in snap["containers"]:
            if record["context"] != "lm-memory":
                continue
            devices = reserved_devices(record["requests"], snap["rows"])
            if resources.released_created(record, [RELEASED_GPU4]) and devices == {4}:
                continue
            own.update(snap["rows"][gpu]["uuid"] for gpu in devices)
        own.update(
            job["lane"]["gpu_uuid"]
            for job in self.state["jobs"].values()
            if job.get("lane") and job["state"] in ("creating", "created", "starting", "running")
        )
        return own

    def start(self, name, lane):
        spec, job = self.jobs[name], self.state["jobs"][name]
        folder = self.batch / "execution" / name
        self.lease(lane)
        if job["state"] == "pending":
            if folder.exists():
                raise ValueError(
                    "Untracked existing stage directory; preserve without duplicate submission"
                )
            if spec["cohort"] != "natural" and spec["action"] == "prepare":
                if not (self.batch / "data" / spec["cohort"] / "manifest.json").exists():
                    raise ValueError("Controlled data must be materialized before GPU preparation")
            label, command = container_command(self.plan, name, spec, lane, self.selection)
            if call([*DOCKER, "ps", "-aq", "--filter", f"name=^/{label}$"]).stdout.strip():
                raise FileExistsError(
                    "Original container name already exists without tracked identity"
                )
            write(folder / "command.json", {"argv": command, "utc": utc()})
            job.update(
                state="creating",
                job_id=name,
                lane=lane,
                container=label,
                source_sha256=self.plan["source_tar_sha256"],
                started_utc=utc(),
            )
            self.save()
            job.update(container_id=call(command).stdout.strip(), state="created")
            self.save()
        actual = self.inspect(job)
        if actual["state"]["Status"] != "created" or actual["state"]["Pid"] != 0:
            raise ValueError("Only the unchanged created container may be started")
        for key, payload in (
            ("source_copied", self.payload),
            ("credentials_copied", credentials(self.plan["wandb_token_file"])),
        ):
            if not job.get(key):
                # Docker's archive mode adopts Config.User on this daemon. Source
                # must remain root-owned; only the private secret adopts run UID.
                copy_command = [*DOCKER, "cp"]
                if key == "credentials_copied":
                    copy_command.append("--archive")
                subprocess.run(
                    [*copy_command, "-", f"{actual['id']}:/"],
                    input=payload,
                    capture_output=True,
                    timeout=120,
                    check=True,
                )
                job[key] = True
                self.save()
        # External work can appear while code is copied; check again before start.
        snap = resources.snapshot()
        reasons = admissible(
            snap,
            lane,
            self.plan["runtime"],
            owned_uuids=self.own_uuids(snap),
            own_container=actual["id"],
        )
        if reasons:
            job["waiting_reasons"] = reasons
            self.save()
            return
        job.update(state="starting", waiting_reasons=[])
        self.save()
        call([*DOCKER, "start", actual["id"]])
        job.update(state="running", cuda_stage_started_utc=utc())
        self.save()

    def tick(self):
        self.poll()
        if self.state["state"] == "needs_attention":
            return "needs_attention"
        try:
            if not self.state.get("gate", {}).get("passed"):
                self.refresh_gate()
            self.state["gate"] = factorial_gate(self.plan)
        except resources.ResourceUnavailable as error:
            self.state["gate"] = {
                "passed": False,
                "state": "waiting_factorial",
                "reasons": [str(error)],
            }
        except Exception as error:
            self.state["gate"] = {
                "passed": False,
                "state": "needs_attention",
                "reasons": [str(error)],
            }
        if not self.state["gate"]["passed"]:
            self.state["state"] = self.state["gate"]["state"]
            self.save()
            return self.state["state"]
        self.state["state"] = "running"
        ready = [
            name
            for name, spec in self.jobs.items()
            if self.state["jobs"][name]["state"] in ("pending", "created")
            and all(self.state["jobs"][key]["state"] == "complete" for key in spec["dependencies"])
        ]
        for name in ready:
            if self.jobs[name]["kind"] == "cpu":
                self.cpu(name)
                return self.state["state"]
            job = self.state["jobs"][name]
            lanes = [job["lane"]] if job.get("lane") else self.plan["lanes"]
            for lane in lanes:
                busy = any(
                    other != name
                    and value.get("lane", {}).get("gpu") == lane["gpu"]
                    and value["state"] in ("creating", "created", "starting", "running")
                    for other, value in self.state["jobs"].items()
                )
                if busy:
                    continue
                try:
                    with locked(self.lock_directory / "admission.lock"):
                        self.lease(lane)
                        snap = resources.snapshot()
                        write(self.batch / "latest-resources.json", snap)
                        reasons = admissible(
                            snap,
                            lane,
                            self.plan["runtime"],
                            owned_uuids=self.own_uuids(snap),
                            own_container=job.get("container_id"),
                        )
                        if reasons:
                            job["waiting_reasons"] = reasons
                            if job["state"] == "pending":
                                self.leases.pop(lane["gpu"]).close()
                            continue
                        self.start(name, lane)
                    break
                except (BlockingIOError, resources.ResourceUnavailable) as error:
                    job["waiting_reasons"] = [str(error)]
                    if job["state"] == "pending":
                        handle = self.leases.pop(lane["gpu"], None)
                        if handle:
                            handle.close()
                except Exception as error:
                    self.fail(name, error)
                    return "needs_attention"
        if all(job["state"] == "complete" for job in self.state["jobs"].values()):
            self.state.update(state="complete", completed_utc=utc())
        self.save()
        return self.state["state"]

    def close(self):
        for handle in self.leases.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    config, payload, manifest = validate(plan)
    jobs = pipeline(config)
    if not args.execute:
        print(
            json.dumps({"mode": "validate_only", "stages": jobs, "cuda_started": False}, indent=2)
        )
        return 0
    if socket.gethostname() != plan["host"]:
        raise ValueError("Frozen execution host differs")
    with locked(Path(plan["output"]) / "controller.lock"):
        controller = Controller(plan, config, payload, manifest)
        write(
            Path(plan["output"]) / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "utc": utc(),
                "plan_sha256": fingerprint(plan),
                "source_tar_sha256": plan["source_tar_sha256"],
            },
        )
        try:
            while True:
                status = controller.tick()
                if args.once or status in ("complete", "needs_attention"):
                    return int(status == "needs_attention")
                time.sleep(min(30, plan["runtime"].get("poll_seconds", 30)))
        finally:
            controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
