"""Small, paired follow-ups using frozen editing and gradient implementations.

Historical source files are left intact. A process-local, scoped adapter supplies
only a new supervised weight vector to the established P1 training loop.
"""

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from . import bios_context_gradient as gradient
from . import bios_mechanism_edit as editing
from .bios_cross import make_cross_world
from .bios_data import write_json
from .bios_model import CausalLM, ModelConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/bios-followup-v1.json"
ROOT_MASSES = {"root125": 0.125, "root250": 0.25, "root500": 0.5}
ORIGINAL_MAKE_ARM = editing.make_arm
ORIGINAL_SOURCE_HASHES = editing.source_hashes


def mixture_spec(world, chain, arm, pair=None):
    if "-root" not in arm:
        return ORIGINAL_MAKE_ARM(world, chain, arm, pair)
    kind, tag = arm.split("-", 1)
    if kind not in ("coherent", "exception") or tag not in ROOT_MASSES:
        raise ValueError("Unknown mixture arm")
    spec = ORIGINAL_MAKE_ARM(world, chain, kind, pair)
    mass = ROOT_MASSES[tag]
    mask = spec["S_root_mask"]
    spec["S_weights"] = np.where(
        mask, mass * len(mask) / mask.sum(), (1 - mass) * len(mask) / (~mask).sum()
    )
    return spec


def production_hashes():
    return {**ORIGINAL_SOURCE_HASHES(), Path(__file__).name: editing.file_hash(__file__)}


@contextmanager
def mixture_adapter():
    old_make, old_hash = editing.make_arm, editing.source_hashes
    editing.make_arm, editing.source_hashes = mixture_spec, production_hashes
    try:
        yield
    finally:
        editing.make_arm, editing.source_hashes = old_make, old_hash


def uniform_forward(self, x):
    """First-layer uniform causal attention, with the full value/projection path."""
    width = x.shape[-1]
    value = self.qkv(x)[..., 2 * width :]
    mask = gradient.allowed_mask(x.shape[1], False, x.device).to(x.dtype)
    mask = mask / mask.sum(-1, keepdim=True)
    return self.proj(mask @ value)


@contextmanager
def uniform_first_layer(model):
    attention = model.blocks[0].attention
    original = attention.forward
    attention.forward = MethodType(uniform_forward, attention)
    try:
        yield
    finally:
        attention.forward = original


def load_config():
    config = json.loads((ROOT / CONFIG).read_text())
    if config["root_mass"] != {"uniform": 1 / 31, **ROOT_MASSES}:
        raise ValueError("The declared mixture weights differ from the implementation")
    return config


def frozen_paths(config):
    paths = [
        ROOT / CONFIG,
        Path(__file__),
        ROOT / "scripts/run_bios_followup.py",
        ROOT / "tests/test_bios_followup.py",
        ROOT / "src/llm_memory_editability/bios_context_gradient.py",
        ROOT / "src/llm_memory_editability/bios_mechanism_edit.py",
        ROOT / "scripts/run_bios_mechanism_edit.py",
        ROOT / config["gradient_config"],
        ROOT / "docs/development-artifacts/context-gradient-v1/preregistration-lock.json",
    ]
    paths += [ROOT / "src/llm_memory_editability" / name for name in editing.parent_source_hashes()]
    for world, seed, condition in itertools.product(
        config["worlds"], config["seeds"], config["conditions"]
    ):
        name = f"world-{world}-seed-{seed}-{condition}"
        paths += [ROOT / config["parent_root"] / name / "config.json"]
        paths += [ROOT / config["balanced_root"] / name / "contract.json"]
    return sorted(set(paths))


def freeze(config):
    artifacts = ROOT / config["artifacts"]
    artifacts.mkdir(parents=True, exist_ok=True)
    lock = artifacts / "preregistration-lock.json"
    if lock.exists():
        raise FileExistsError("Existing prospective lock cannot be replaced")
    protocol = (ROOT / "docs/experimental-protocol.md").read_text().split("## 17. ", 1)[1]
    (artifacts / "preregistration.md").write_text("## 17. " + protocol)
    sources = {}
    for path in frozen_paths(config):
        relative = path.relative_to(ROOT)
        sources[str(relative)] = editing.file_hash(path)
        if relative.parts[0] not in ("results", "data"):
            destination = artifacts / "frozen-files" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    write_json(
        lock,
        {
            "created_at": gradient.stamp(),
            "config": config,
            "sources": sources,
            "protocol_sha256": editing.file_hash(artifacts / "preregistration.md"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )
    return {"lock": str(lock), "sha256": editing.file_hash(lock)}


def verify_lock(config):
    artifacts = ROOT / config["artifacts"]
    path = artifacts / "preregistration-lock.json"
    lock = json.loads(path.read_text())
    if lock["config"] != config or lock["torch"] != torch.__version__:
        raise ValueError("Frozen configuration/runtime mismatch")
    for source, digest in lock["sources"].items():
        if editing.file_hash(ROOT / source) != digest:
            raise ValueError("Frozen source mismatch: " + source)
    if editing.file_hash(artifacts / "preregistration.md") != lock["protocol_sha256"]:
        raise ValueError("Frozen protocol mismatch")
    return editing.file_hash(path)


def run_edit(config, world, seed, condition, device):
    verify_lock(config)
    if (world, seed, condition) not in itertools.product(
        config["worlds"], config["seeds"], config["conditions"]
    ):
        raise ValueError("Edit worker is outside the frozen matrix")
    name = f"world-{world}-seed-{seed}-{condition}"
    destination = ROOT / config["output"] / "editing" / name
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with mixture_adapter():
            editing.run_model(
                ROOT / config["parent_root"] / name, destination, config["edit"], device
            )


def verify_receipt(path):
    receipt = json.loads((path / "receipt.json").read_text())
    for name, digest in receipt["files"].items():
        if editing.file_hash(path / name) != digest:
            raise ValueError("Gradient receipt mismatch: " + str(path / name))


def run_gradient(config, world_seed, seed, device):
    lock_hash = verify_lock(config)
    if world_seed not in config["worlds"] or seed not in config["seeds"]:
        raise ValueError("Gradient worker is outside the frozen matrix")
    original = json.loads((ROOT / config["gradient_config"]).read_text())
    gradient.verify_lock(ROOT, original)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world = make_cross_world(world_seed, ROOT / original["source_world_root"])
    name = f"world-{world_seed}-seed-{seed}"
    source = ROOT / config["gradient_parent"] / name
    destination = ROOT / config["output"] / "gradient" / name
    destination.mkdir(parents=True, exist_ok=True)
    seal = json.loads((source / "complete.json").read_text())
    consumed = {"complete.json": editing.file_hash(source / "complete.json")}
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "complete.json").exists():
            raise FileExistsError("Completed gradient worker cannot be overwritten")
        for step in config["gradient"]["states"]:
            checkpoint = source / f"state-{step}.pt"
            digest = editing.file_hash(checkpoint)
            if digest != seal["files"][checkpoint.name]:
                raise ValueError("Archived common state hash mismatch")
            consumed[checkpoint.name] = digest
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["step"] != step:
                raise ValueError("Common state step mismatch")
            model = CausalLM(ModelConfig(**saved["config"])).to(device)
            model.load_state_dict(saved["model"])
            sham = config["gradient"]["sham"]
            if (world_seed, seed, step) == (sham["world"], sham["seed"], sham["step"]):
                path = destination / "sham"
                if not (path / "receipt.json").exists():
                    gradient.measure_arm(
                        model, world, sham["condition"], False, original, device, path
                    )
                verify_receipt(path)
                before = source / f"step-{step}-open-{sham['condition']}"
                verify_receipt(before)
                errors = {}
                for kind in ("answer", "eos"):
                    with np.load(before / f"{kind}.npz") as a, np.load(path / f"{kind}.npz") as b:
                        for parameter in a.files:
                            reference = np.linalg.norm(a[parameter].astype(np.float64))
                            difference = np.linalg.norm(
                                b[parameter].astype(np.float64) - a[parameter]
                            )
                            error = (
                                float(difference / reference) if reference else float(difference)
                            )
                            errors[f"{kind}/{parameter}"] = error
                if max(errors.values()) > config["gradient"]["sham_relative_tolerance"]:
                    raise ValueError("Historical-gradient sham failed")
                write_json(destination / "sham-check.json", errors)
            for condition in config["conditions"]:
                path = destination / f"step-{step}-uniform-{condition}"
                if not (path / "receipt.json").exists():
                    with uniform_first_layer(model):
                        gradient.measure_arm(model, world, condition, False, original, device, path)
                verify_receipt(path)
                print(json.dumps({"event": "gradient_arm_complete", "path": str(path)}), flush=True)
            del model, saved
        verify_lock(config)
        write_json(
            destination / "complete.json",
            {
                "complete": True,
                "created_at": gradient.stamp(),
                "lock_sha256": lock_hash,
                "parent": str(source),
                "consumed": consumed,
                "world": world_seed,
                "seed": seed,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "new_uniform_arms": 9,
                "files": {
                    str(p.relative_to(destination)): editing.file_hash(p)
                    for p in sorted(destination.rglob("*"))
                    if p.is_file() and p.suffix in ("npz", ".npz", ".json")
                },
            },
        )


def jobs(config):
    result = []
    for world, seed in itertools.product(config["worlds"], config["seeds"]):
        name = f"world-{world}-seed-{seed}"
        result.append(("gradient/" + name, ["run-gradient", "--world", world, "--seed", seed]))
        for condition in config["conditions"]:
            result.append(
                (
                    "editing/" + name + "-" + condition,
                    ["run-edit", "--world", world, "--seed", seed, "--condition", condition],
                )
            )
    return result


def dispatch(config, gpus, slots):
    if not gpus or len(set(gpus)) != len(gpus) or slots < 1 or 0 in gpus:
        raise ValueError(
            "Unique healthy GPUs and positive concurrency required; GPU0 has ECC errors"
        )
    lock_hash = verify_lock(config)
    destination = ROOT / config["output"]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending, complete, failed, active = [], [], [], []
        for name, arguments in jobs(config):
            if (destination / name / "complete.json").exists():
                complete.append(name)
            else:
                pending.append((name, arguments))
        while pending or active:
            for item in active[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                good = code == 0 and (destination / item["name"] / "complete.json").exists()
                (complete if good else failed).append(item["name"])
                active.remove(item)
            if not failed:
                for gpu in gpus:
                    while pending and sum(i["gpu"] == gpu for i in active) < slots:
                        name, arguments = pending.pop(0)
                        out = destination / name
                        out.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, "-m", "llm_memory_editability.bios_followup"]
                        command += [str(a) for a in arguments]
                        log = (out / "run.log").open("a")
                        env = {
                            **os.environ,
                            "PYTHONPATH": str(ROOT / "src"),
                            "CUDA_VISIBLE_DEVICES": str(gpu),
                            "OMP_NUM_THREADS": "2",
                            "OPENBLAS_NUM_THREADS": "2",
                        }
                        process = subprocess.Popen(
                            command, cwd=ROOT, env=env, stdout=log, stderr=log
                        )
                        write_json(
                            out / "launch.json",
                            {
                                "pid": process.pid,
                                "gpu": gpu,
                                "command": command,
                                "started": gradient.stamp(),
                            },
                        )
                        active.append({"name": name, "gpu": gpu, "process": process, "log": log})
            write_json(
                destination / "status.json",
                {
                    "state": "failed" if failed else "running" if active or pending else "complete",
                    "lock_sha256": lock_hash,
                    "complete": complete,
                    "failed": failed,
                    "pending": len(pending),
                    "active": [
                        {"name": i["name"], "gpu": i["gpu"], "pid": i["process"].pid}
                        for i in active
                    ],
                    "updated": gradient.stamp(),
                },
            )
            if failed and not active:
                raise RuntimeError("Worker failure; unstarted jobs retained")
            if pending or active:
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "dispatch", "run-edit", "run-gradient"))
    parser.add_argument("--world", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--gpus", type=int, nargs="+")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_config()
    if args.mode == "freeze":
        print(json.dumps(freeze(config)))
    elif args.mode == "dispatch":
        dispatch(config, args.gpus, args.slots_per_gpu)
    elif args.mode == "run-edit":
        run_edit(config, args.world, args.seed, args.condition, torch.device(args.device))
    else:
        run_gradient(config, args.world, args.seed, torch.device(args.device))


if __name__ == "__main__":
    main()
