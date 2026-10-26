#!/usr/bin/env python3
"""Package without execution; explicitly launch on a separately chosen host.

This module uses only the standard library. No import, plan, package or verify
operation initializes CUDA, creates a container or contacts W&B.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "norm-handoff-multihop-v1"
CONFIG = f"configs/{BATCH}.json"
GUIDE = f"docs/development-artifacts/{BATCH}/handoff.md"
CORE = (
    "scripts/run_loop_hop.py",
    "scripts/run_loop_hop_depth.py",
    "scripts/run_supervised_alignment.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "src/llm_memory_editability/loop_hop_data.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/bios_model.py",
    "src/llm_memory_editability/experiment_tracking.py",
)
EXTRA = (
    CONFIG,
    GUIDE,
    "pyproject.toml",
    "scripts/run_norm_handoff.py",
    "scripts/launch_norm_handoff.py",
    "scripts/run_dual_gate.py",
    "scripts/report_dual_gate.py",
    "scripts/report_norm_handoff.py",
    "scripts/verify_norm_handoff_baseline.py",
    "configs/loop-hop-unified-v1.json",
    "scripts/diagnose_norm_handoff.py",
    "tests/test_norm_handoff.py",
    "tests/test_norm_handoff_launcher.py",
)
ENGINEERING = ("preflight", "smoke", "smoke-audit", "smoke-grid", "smoke-grid-audit")
SCIENCE = ("train", "audit", "grid", "grid-audit", "cloud")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate(config):
    expected = {
        name: {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "handoff": handoff,
            "residual_scale": scale,
        }
        for name, handoff, scale in (
            ("R", "fixed", 0.25),
            ("N", "normalized", 1),
            ("A", "additive", 1),
            ("C", "complementary", 1),
        )
    }
    if config["batch"] != BATCH or config["arms"] != expected:
        raise ValueError("Only R/N/A/C train; fixed F must be reused")
    if config["training"]["steps"] != 300000 or config["smoke_steps"] != 32:
        raise ValueError("Fixed scientific/engineering budgets changed")
    if config["budget"]["new_scientific_updates"] != 1200000:
        raise ValueError("Four scientific runs only")
    if config["runtime"]["tracking"] != "online":
        raise ValueError("Online tracking is required for target-machine scientific training")
    if config["handoff"] != {
        "eps": 1e-6,
        "gate_layer_norm_eps": 1e-5,
        "temperature": 1,
        "channel_coefficient": 1,
        "gate_initial_logit": 0,
        "gate_weight_decay": 0,
        "normalization": "N forms native rounded h+q before FP32 RMS reduction; "
        "A/C mix FP32 h and q then cast once; FP64 retained in numerical contracts.",
    }:
        raise ValueError("Handoff contract changed")


def verify_baseline(config, baseline_root, source_root):
    baseline_root, source_root = Path(baseline_root), Path(source_root)
    baseline = config["baseline"]
    run = baseline_root / "runs" / baseline["arm"]
    frozen = read(baseline_root / "freeze.json")
    for name, expected in baseline["files"].items():
        if sha(run / name) != expected:
            raise ValueError(f"Fixed baseline file changed: {name}")
    for name in ("audit.json", "grid-audit.json"):
        if not read(run / name)["passed"]:
            raise ValueError("Both independent baseline audits must pass")
    prior_path = "source/configs/loop-hop-unified-v1.json"
    if sha(baseline_root / prior_path) != frozen["files"][prior_path]:
        raise ValueError("Baseline recipe no longer matches its freeze")
    prior = read(baseline_root / prior_path)
    for key in (
        "data",
        "model",
        "training",
        "dropout_seed",
        "input_origin",
        "evaluation_nodes",
        "evaluation_batch_size",
    ):
        if prior[key] != config[key]:
            raise ValueError(f"Scientific recipe differs from F: {key}")
    for relative in CORE:
        expected = frozen["files"][f"source/{relative}"]
        if sha(baseline_root / "source" / relative) != expected:
            raise ValueError(f"Baseline frozen core changed: {relative}")
        if sha(source_root / relative) != expected:
            raise ValueError(f"Native source differs from F: {relative}")
    if frozen["image"] != config["runtime"]["image"]:
        raise ValueError("Use the same fixed image; do not silently replace it")
    complete = read(run / "complete.json")
    if (
        complete["step"] != 300000
        or complete["initial_model_sha256"] != baseline["initial_model_sha256"]
    ):
        raise ValueError("Baseline budget or initialization changed")
    return {"passed": True, "new_baseline_updates": 0, "utc": utc()}


def package(destination, input_dir, baseline_root):
    destination = Path(destination).resolve()
    config = read(ROOT / CONFIG)
    validate(config)
    receipt = verify_baseline(config, baseline_root, ROOT)
    input_dir, baseline_root = Path(input_dir), Path(baseline_root)
    for filename, key in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        if sha(input_dir / filename) != config["input_origin"][key]:
            raise ValueError(f"Data identity changed: {filename}")
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "source"
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for rel in dict.fromkeys((*CORE, *EXTRA)):
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (destination / "input").mkdir()
    for name in ("world.npz", "panels.npz", "complete.json", "audit.json"):
        shutil.copy2(input_dir / name, destination / "input" / name)
    paths = ["freeze.json", "source/configs/loop-hop-unified-v1.json"]
    paths += [f"source/{r}" for r in CORE]
    paths += [f"runs/{config['baseline']['arm']}/{n}" for n in config["baseline"]["files"]]
    for relative in paths:
        target = destination / "baseline" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_root / relative, target)
    shutil.copy2(
        baseline_root / "source/configs/loop-hop-unified-v1.json",
        source / "configs/loop-hop-unified-v1.json",
    )
    baseline_run = baseline_root / "runs" / config["baseline"]["arm"]
    shutil.copytree(
        baseline_run / "grid",
        destination / "baseline" / "runs" / config["baseline"]["arm"] / "grid",
    )
    shutil.copy2(ROOT / GUIDE, destination / "START_HERE.md")
    write(destination / "baseline-reuse.json", receipt)
    manifest = {
        "batch": BATCH,
        "utc": utc(),
        "image": config["runtime"]["image"],
        "credentials_included": False,
        "image_included": False,
        "model_execution_performed": False,
        "files": {
            str(p.relative_to(destination)): sha(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    write(destination / "bundle.json", manifest)
    verify_bundle(destination)
    return destination


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / "bundle.json")
    for relative, expected in manifest["files"].items():
        path = bundle / relative
        if not path.resolve().is_relative_to(bundle) or sha(path) != expected:
            raise ValueError(f"Bundle file changed or escaped root: {relative}")
    config = read(bundle / "source" / CONFIG)
    validate(config)
    verify_baseline(config, bundle / "baseline", bundle / "source")
    for filename, key in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        if sha(bundle / "input" / filename) != config["input_origin"][key]:
            raise ValueError("Bundle data changed")
    return config


def checked(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=60).stdout


def gpu_available(index, contexts):
    rows = (
        checked(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        .strip()
        .split(",")
    )
    uuid, used, free, utilization = [s.strip() for s in rows]
    if int(used) > 512 or int(free) < 18000 or int(utilization) > 5:
        raise RuntimeError(f"Selected GPU {index} is busy or has insufficient memory")
    processes = checked(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if any(line.split(",")[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError(f"Selected GPU {index} has an active compute process")
    for context in contexts:
        identifiers = checked(["docker", "--context", context, "ps", "-q"]).split()
        if not identifiers:
            continue
        for container in json.loads(
            checked(["docker", "--context", context, "inspect", *identifiers])
        ):
            for request in container["HostConfig"].get("DeviceRequests") or []:
                devices = request.get("DeviceIDs") or []
                # An all-GPU reservation is treated conservatively as occupied.
                if request.get("Count") == -1 or str(index) in devices or uuid in devices:
                    raise RuntimeError(f"GPU {index} reserved by {container['Name']} in {context}")
    return uuid


def local_engine(context):
    endpoint = json.loads(checked(["docker", "context", "inspect", context]))[0]["Endpoints"][
        "docker"
    ]["Host"]
    if not endpoint.startswith("unix://"):
        raise ValueError("Run on the target host itself with a local Unix Docker context")


def prior_lane_released(root, gpu):
    """A temporarily idle GPU is still reserved by the previous batch's queue."""
    if root is None:
        return True
    path = root / "handoff" / f"lane-{gpu}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing previous queue receipt: {path}")
    lane = read(path)
    if lane["state"] != "finished":
        return False
    if not lane.get("outcomes") or any(code != 0 for code in lane["outcomes"].values()):
        raise RuntimeError(f"Previous GPU {gpu} queue needs attention; no automatic takeover")
    for job in lane["outcomes"]:
        for name in ("result-audit.json", "result-cloud-audit.json"):
            if not read(root / "jobs" / job / name)["passed"]:
                raise RuntimeError(f"Previous job {job} has not passed {name}")
    return True


def effective_runtime(args, config):
    """Keep the original bundle immutable; bind a local environment amendment."""
    amendment_path = getattr(args, "runtime_amendment", None)
    if amendment_path is None:
        return config
    amendment = read(amendment_path)
    if (
        amendment["hostname"] != platform.node()
        or amendment["bundle_sha256"] != sha(args.bundle / "bundle.json")
        or amendment["original_image"] != config["runtime"]["image"]
        or amendment["gpus"] != args.gpus
        or amendment["launcher_sha256"] != sha(Path(__file__))
        or amendment["prior_batch"] != str(args.wait_for_prior_batch.resolve())
    ):
        raise ValueError("Local runtime amendment does not match this launch")
    actual = amendment["actual_image"]
    if not actual.startswith("sha256:") or len(actual) != 71:
        raise ValueError("Replacement image must be pinned by full SHA256")
    config = json.loads(json.dumps(config))
    config["runtime"]["image"] = actual
    return config


class Controller:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.bundle = args.bundle.resolve()
        self.output = args.output.resolve()
        self.docker = ["docker", "--context", args.docker_context]
        self.contexts = list(dict.fromkeys([args.docker_context, *args.reservation_context]))
        identity = hashlib.sha256(str(self.output).encode()).hexdigest()[:10]
        self.prefix = f"nh-{identity}"

    def stage(self, arm, stage, gpu=None):
        folder = self.output / "execution" / arm / stage
        receipt = folder / "exit.json"
        if receipt.exists() and read(receipt)["code"] == 0:
            return
        # A claimed, failed or interrupted stage is never automatically relaunched.
        if folder.exists():
            raise FileExistsError(folder)
        verify_bundle(self.bundle)
        if gpu is not None:
            if getattr(self.args, "wait_for_prior_batch", None) is not None:
                wait_path = self.output / "waiting" / f"{arm}-{stage}.json"
                while True:
                    reason = "previous training/audit queue is still active"
                    if prior_lane_released(self.args.wait_for_prior_batch, gpu):
                        try:
                            gpu_available(gpu, self.contexts)
                            break
                        except RuntimeError as error:
                            reason = str(error)
                    write(
                        wait_path,
                        {"state": "waiting", "gpu": gpu, "reason": reason, "utc": utc()},
                    )
                    time.sleep(30)
                write(wait_path, {"state": "released", "gpu": gpu, "utc": utc()})
            uuid = gpu_available(gpu, self.contexts)
        folder.mkdir(parents=True, exist_ok=False)
        label = f"{self.prefix}-{arm.lower()}-{stage}"
        tracked = stage in ("train", "cloud", "report")
        command = [
            *self.docker,
            "run",
            "--name",
            label,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpus",
            str(self.args.cpus_per_run),
            "--memory",
            f"{self.args.memory_gib}g",
            "--memory-swap",
            f"{self.args.memory_gib}g",
            "--shm-size",
            "2g",
            "--tmpfs",
            "/tmp:rw,size=4g",
            "--network",
            "host" if tracked else "none",
            "--workdir",
            "/frozen",
        ]
        for value in (
            "PYTHONPATH=/frozen/src:/frozen/scripts",
            "PYTHONDONTWRITEBYTECODE=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "XDG_CACHE_HOME=/tmp/cache",
            "RUFF_CACHE_DIR=/tmp/ruff-cache",
            "WANDB_CACHE_DIR=/tmp/wandb-cache",
            "WANDB_CONSOLE=off",
            "WANDB_SILENT=true",
            "WANDB_DISABLE_CODE=true",
            "OMP_NUM_THREADS=2",
            "MKL_NUM_THREADS=2",
            f"WANDB_DOCKER={self.config['runtime']['image']}",
        ):
            command += ["--env", value]
        mounts = [
            (self.bundle / "source", "/frozen", True),
            (self.bundle / "input", "/inputs", True),
            (self.bundle / "baseline" / "runs" / self.config["baseline"]["arm"], "/baseline", True),
            (self.output, "/output", False),
        ]
        if tracked:
            mounts.append((self.args.wandb_key_file.resolve(), "/run/secrets/wandb", True))
        for host, guest, readonly in mounts:
            if "," in str(host):
                raise ValueError("Docker bind paths must not contain commas")
            command += [
                "--mount",
                f"type=bind,source={host},target={guest}" + (",readonly" if readonly else ""),
            ]
        if gpu is not None:
            command += ["--gpus", f"device={uuid}"]
        if stage == "contracts":
            code = (
                "import json,subprocess;from pathlib import Path;"
                "from llm_memory_editability.loop_hop_data import load,audit_data;"
                f"c=json.loads(Path('{CONFIG}').read_text());"
                "w,p,m=load('/inputs');audit_data(w,p,m);"
                "assert all(m[k]==v for k,v in c['data'].items());"
                "subprocess.run(['python','-m','pytest','-q','tests/test_norm_handoff.py',"
                "'tests/test_norm_handoff_launcher.py','--junitxml=/output/contracts.xml'],check=True)"
            )
            entry = ["python", "-c", code]
        elif stage == "baseline-replay":
            entry = [
                "python",
                "scripts/verify_norm_handoff_baseline.py",
                "--base",
                "/output",
                "--baseline",
                "/baseline",
                "--config",
                CONFIG,
            ]
        elif stage == "diagnostic":
            entry = [
                "python",
                "scripts/diagnose_norm_handoff.py",
                "--base",
                "/output",
                "--baseline",
                "/baseline",
                "--config",
                CONFIG,
            ]
        elif stage == "report":
            entry = [
                "python",
                "scripts/report_norm_handoff.py",
                "--base",
                "/output",
                "--baseline",
                "/baseline",
                "--config",
                CONFIG,
            ]
        else:
            entry = [
                "python",
                "-u",
                "scripts/run_norm_handoff.py",
                stage,
                "--config",
                CONFIG,
                "--base",
                "/output",
                "--arm",
                arm,
            ]
        bootstrap = "import os,sys;os.umask(0o022);"
        if tracked:
            bootstrap += (
                "from pathlib import Path;"
                "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
            )
        bootstrap += "os.execvp(sys.argv[1],sys.argv[1:])"
        command += [self.config["runtime"]["image"], "python", "-c", bootstrap, *entry]
        write(folder / "command.json", {"argv": command, "utc": utc()})
        if arm in self.config["arms"]:
            write(
                self.output / "runtime" / f"{arm}-{stage}.json",
                {
                    "gpu": gpu,
                    "gpu_uuid": uuid if gpu is not None else None,
                    "container": label,
                    "cpu_limit": self.args.cpus_per_run,
                    "memory_gib": self.args.memory_gib,
                    "exclusive_gpu": True,
                    "utc": utc(),
                },
            )
        with (folder / "stdout.log").open("x") as log:
            started = time.perf_counter()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            write(folder / "process.json", {"pid": process.pid, "utc": utc()})
            while process.poll() is None:
                if gpu is not None and arm in self.config["arms"]:
                    try:
                        used, utilization = (
                            checked(
                                [
                                    "nvidia-smi",
                                    f"--id={gpu}",
                                    "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits",
                                ]
                            )
                            .strip()
                            .split(",")
                        )
                        write(
                            self.output / "resources" / f"{arm}.json",
                            {
                                "gpu": gpu,
                                "used_mib": int(used),
                                "utilization": int(utilization),
                                "utc": utc(),
                            },
                        )
                    except (subprocess.SubprocessError, ValueError) as error:
                        write(
                            folder / "resource-sampling-error.json",
                            {"error": repr(error), "utc": utc()},
                        )
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    write(folder / "heartbeat.json", {"pid": process.pid, "utc": utc()})
            # Reap the CLI; the stopped container and its logs remain available.
            process.wait()
        write(
            receipt,
            {
                "code": process.returncode,
                "container": label,
                "stage_wall_seconds": time.perf_counter() - started,
                "utc": utc(),
            },
        )
        if process.returncode:
            raise RuntimeError(f"{arm}/{stage} failed; preserve {folder}, no automatic retry")

    def wave(self, stages):
        # One worker per selected device; a device never hosts two batch arms.
        slots = min(len(self.args.gpus), len(self.config["arms"]))
        arms = list(self.config["arms"])

        def worker(slot):
            for name in arms[slot::slots]:
                for stage in stages:
                    self.stage(name, stage, None if stage == "cloud" else self.args.gpus[slot])

        with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
            futures = [pool.submit(worker, slot) for slot in range(slots)]
            for future in futures:
                future.result()

    def run(self):
        self.stage("all", "contracts")
        self.stage("all", "baseline-replay", self.args.gpus[0])
        if not read(self.output / "baseline-target-replay.json")["passed"]:
            raise ValueError("Reused F must replay on the target GPU before any new training")
        self.wave(ENGINEERING)
        for name in self.config["arms"]:
            for relative in (
                f"preflight/{name}.json",
                f"smoke/{name}/audit.json",
                f"smoke/{name}/grid-audit.json",
            ):
                if not read(self.output / relative)["passed"]:
                    raise ValueError(f"Engineering evidence failed: {relative}")
        write(
            self.output / "engineering-complete.json",
            {
                "passed": True,
                "arms": list(self.config["arms"]),
                "config_sha256": sha(self.bundle / "source" / CONFIG),
                "utc": utc(),
            },
        )
        self.wave(SCIENCE)
        self.stage("all", "diagnostic")
        self.stage("all", "report")
        if not read(self.output / "report-cloud-audit.json")["passed"]:
            raise ValueError("Report cloud verification failed")
        write(
            self.output / "completion-manifest.json",
            {
                "passed": True,
                "new_scientific_updates": 1200000,
                "fixed_baseline_new_updates": 0,
                "utc": utc(),
            },
        )


def execute(args):
    config = verify_bundle(args.bundle)
    config = effective_runtime(args, config)
    if (
        not args.gpus
        or len(args.gpus) > 4
        or len(set(args.gpus)) != len(args.gpus)
        or min(args.gpus) < 0
    ):
        raise ValueError("Select 1-4 distinct nonnegative physical GPUs explicitly")
    if args.cpus_per_run < config["runtime"]["cpu_threads"] or args.memory_gib < 16:
        raise ValueError("At least 2 CPU cores and 16 GiB memory per run required")
    if not args.wandb_key_file.is_file() or not args.wandb_key_file.read_text().strip():
        raise ValueError("An external nonempty W&B key file is required")
    for context in dict.fromkeys([args.docker_context, *args.reservation_context]):
        local_engine(context)
    image = json.loads(
        checked(
            [
                "docker",
                "--context",
                args.docker_context,
                "image",
                "inspect",
                config["runtime"]["image"],
            ]
        )
    )[0]["Id"]
    if image != config["runtime"]["image"]:
        raise ValueError("Import the original fixed image before launch")
    args.output = args.output.resolve()
    bundle = args.bundle.resolve()
    if args.output.is_relative_to(bundle) or bundle.is_relative_to(args.output):
        raise ValueError("Keep read-only bundle and mutable output in separate directories")
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError("Resume requires an existing output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = {
            "bundle_sha256": sha(bundle / "bundle.json"),
            "gpus": args.gpus,
            "docker_context": args.docker_context,
            "reservation_contexts": args.reservation_context,
            "cpus_per_run": args.cpus_per_run,
            "memory_gib": args.memory_gib,
            "image": image,
            "hostname": platform.node(),
            "launcher_sha256": sha(Path(__file__)),
            "runtime_amendment_sha256": (
                sha(args.runtime_amendment) if args.runtime_amendment else None
            ),
            "wait_for_prior_batch": (
                str(args.wait_for_prior_batch.resolve()) if args.wait_for_prior_batch else None
            ),
            "gpu_uuids": {
                str(index): checked(
                    [
                        "nvidia-smi",
                        f"--id={index}",
                        "--query-gpu=uuid",
                        "--format=csv,noheader",
                    ]
                ).strip()
                for index in args.gpus
            },
        }
        receipt = args.output / "launch-request.json"
        if args.resume and read(receipt) != request:
            raise ValueError("Resume must preserve the original launch request")
        write(receipt, request)
        write(args.output / "controller-process.json", {"pid": os.getpid(), "utc": utc()})
        write(args.output / "controller-state.json", {"state": "running", "utc": utc()})
        controller = Controller(args, config)
        try:
            controller.run()
        except BaseException as error:
            write(
                args.output / "controller-state.json",
                {
                    "state": "needs_attention",
                    "error": repr(error),
                    "utc": utc(),
                },
            )
            raise
        write(args.output / "controller-state.json", {"state": "complete", "utc": utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan")
    pack = sub.add_parser("package", help="Copy code/data/F; never execute a model")
    pack.add_argument("--destination", type=Path, required=True)
    pack.add_argument("--input-dir", type=Path)
    pack.add_argument("--baseline-root", type=Path)
    pack.add_argument("--tar", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    run = sub.add_parser("run", help="Target-host execution requires --execute")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpus", type=int, nargs="+", required=True)
    run.add_argument("--docker-context", required=True)
    run.add_argument("--reservation-context", action="append", default=[])
    run.add_argument("--wandb-key-file", type=Path, required=True)
    run.add_argument("--cpus-per-run", type=int, default=2)
    run.add_argument("--memory-gib", type=int, default=16)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--runtime-amendment", type=Path)
    run.add_argument("--wait-for-prior-batch", type=Path)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "plan":
        config = read(ROOT / CONFIG)
        validate(config)
        print(json.dumps({"arms": config["arms"], "budget": config["budget"], "executed": False}))
    elif args.action == "package":
        config = read(ROOT / CONFIG)
        target = package(
            args.destination,
            args.input_dir or Path(config["input_origin"]["root"]),
            args.baseline_root or Path(config["baseline"]["root"]),
        )
        if args.tar:
            archive = target.with_suffix(".tar")
            with archive.open("xb") as raw, tarfile.open(fileobj=raw, mode="w") as stream:
                stream.add(target, arcname=target.name)
            archive.with_suffix(".tar.sha256").write_text(f"{sha(archive)}  {archive.name}\n")
        print(json.dumps({"bundle": str(target), "verified": True, "model_execution": False}))
    elif args.action == "verify":
        verify_bundle(args.bundle)
        print(json.dumps({"verified": True, "model_execution": False}))
    elif not args.execute:
        print(json.dumps({"executed": False, "reason": "Add --execute only on the target host"}))
    else:
        execute(args)


if __name__ == "__main__":
    main()
