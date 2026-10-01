"""Prospective v2.13: forward-preserving path interventions and local transfer.

Only a single MLP output weight is differentiated/updated in production.
Downstream parameters are frozen; the custom attention paths alter its input VJP.
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from .bios_cross import edit_pair, make_cross_world
from .bios_cross_train import tensor_queries
from .bios_data import EOS, array_hash, write_json
from .bios_model import CausalLM, ModelConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bios-path-transfer-v1.json"
ARMS = ("full", "no_cross", "no_mlp", "fixed_qk")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def inner(a, b):
    return (a.double() * b.double()).sum().item()


def norm(a):
    return a.double().norm().item()


def relative(a, b):
    return norm(a - b) / max(norm(b), 1e-30)


def cosine(a, b):
    return inner(a, b) / max(norm(a) * norm(b), 1e-30)


def attention_surrogate(module, x, mode):
    """Correct input Jacobian for frozen QK, or only the diagonal position blocks.

    no_cross makes T copies of the input: copy s only differentiates x_s,
    then computes output s with the full original causal context. Thus Q_s,
    K_s and V_s derivatives survive; all x_t -> output_s, t != s vanish.
    """
    batch, length, width = x.shape
    heads, dim = module.heads, width // module.heads
    if mode == "no_cross":
        expanded = x[:, None].expand(batch, length, length, width)
        diagonal = torch.eye(length, device=x.device, dtype=x.dtype)[None, :, :, None]
        routed = expanded.detach() + diagonal * (expanded - expanded.detach())
        q, k, v = module.qkv(routed).reshape(batch, length, length, 3, heads, dim).unbind(3)
        index = torch.arange(length, device=x.device)
        q = q[:, index, index]  # B, query, H, D
        scores = torch.einsum("bshd,bsthd->bhst", q, k) / dim**0.5
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        a = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        output = torch.einsum("bhst,bsthd->bshd", a, v).reshape(batch, length, width)
    else:
        q, k, v = module.qkv(x).reshape(batch, length, 3, heads, dim).unbind(2)
        scores = torch.einsum("bshd,bthd->bhst", q, k) / dim**0.5
        mask = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        a = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        if mode == "fixed_qk":
            a = a.detach()
        elif mode != "manual_full":
            raise ValueError(mode)
        output = torch.einsum("bhst,bthd->bshd", a, v).reshape(batch, length, width)
    return module.proj(output)


@contextmanager
def path_mode(model, layer, arm):
    if arm not in (*ARMS, "manual_full"):
        raise ValueError(arm)
    saved = []
    try:
        for block in model.blocks[layer + 1 :]:
            if arm == "no_mlp":
                module = block.mlp
                original = module.forward

                def stopped(self, x, original=original):
                    with torch.no_grad():
                        return original(x)

                module.forward = MethodType(stopped, module)
                saved.append((module, original))
            elif arm != "full":
                module = block.attention
                original = module.forward

                def altered(self, x, original=original, arm=arm):
                    with torch.no_grad():
                        reference = original(x)
                    surrogate = attention_surrogate(self, x, arm)
                    return reference + (surrogate - surrogate.detach())

                module.forward = MethodType(altered, module)
                saved.append((module, original))
        yield
    finally:
        for module, original in saved:
            module.forward = original


def cases(world):
    result = []
    for chain in range(2):
        pair = edit_pair(world, chain)
        for group in pair["groups"]:
            members = np.flatnonzero(world.memberships[chain] == group)
            eligible = members[
                np.isin(world.derived_ids[chain, members], world.heldout_ids)
                & np.isin(world.derived_ids[chain, members], pair["conflict_D"])
            ]
            if not len(eligible):
                raise ValueError("No prespecified heldout conflict person")
            person = int(eligible[0])
            others = members[
                (members != person)
                & ~world.exceptions[chain, members]
                & np.isin(world.derived_ids[chain, members], world.heldout_ids)
            ]
            if not len(others):
                raise ValueError("No prespecified second heldout person")
            other = int(others[0])
            distant = int(np.flatnonzero(~np.isin(world.memberships[chain], pair["groups"]))[0])
            root = int(world.root_ids[chain, group])
            actual = int(world.actual_ids[chain, person])
            default = int(pair["coherent"][root])
            alternative = int(pair["exception"][actual])
            if default == alternative:
                raise ValueError("Selected person is not a new conflict")
            for kind in ("root", "coherent", "conflict"):
                source = root if kind == "root" else actual
                target = alternative if kind == "conflict" else default
                associated = actual if kind == "root" else root
                ids = [
                    source,
                    int(world.derived_ids[chain, person]),
                    int(world.derived_ids[chain, other]),
                    associated,
                    int(world.actual_ids[1 - chain, person]),
                    int(world.actual_ids[chain, distant]),
                ]
                targets = [target, default, default, default, *world.answers[ids[-2:]].tolist()]
                result.append(
                    {
                        "name": f"chain-{chain}-group-{int(group)}-{kind}",
                        "chain": chain,
                        "group": int(group),
                        "kind": kind,
                        "person": person,
                        "other_person": other,
                        "ids": ids,
                        "targets": targets,
                        "old_targets": world.answers[ids].tolist(),
                        "roles": [
                            "source",
                            "derived_same",
                            "derived_other",
                            "associated",
                            "retention_local",
                            "retention_distant",
                        ],
                    }
                )
    return result


def example_data(world, case, device):
    # Use the original interface, including teacher-forced answer for EOS loss.
    targets = world.answers.copy()
    targets[case["ids"]] = case["targets"]
    all_data = tensor_queries(world, device, targets)
    ids = torch.tensor(case["ids"], device=device)
    return {key: value[ids] for key, value in all_data.items()}


def subset(data, ids):
    return {key: value[ids] for key, value in data.items()}


def losses(logits, labels):
    return (
        F.cross_entropy(
            logits.double().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        )
        .reshape(-1, 2)
        .mean(-1)
    )


def measure(model, data, layer, arm):
    projection = model.blocks[layer].mlp.down
    captured = {}

    def hook(module, inputs, output):
        captured["z"], captured["h"] = inputs[0], output

    handle = projection.register_forward_hook(hook)
    try:
        with path_mode(model, layer, arm):
            logits = model(data["tokens"], data["positions"])
            loss = losses(logits, data["labels"])
            actual, delta = torch.autograd.grad(loss.sum(), (projection.weight, captured["h"]))
        z = captured["z"].detach()
        delta = delta.detach()
        per_position = torch.einsum("btd,btk->btdk", delta, z)
        mask = torch.zeros(z.shape[:2], device=z.device, dtype=torch.bool)
        mask.scatter_(1, data["positions"], True)
        g = per_position.sum(1)
        supervised = (per_position * mask[:, :, None, None]).sum(1)
        unsupervised = (per_position * ~mask[:, :, None, None]).sum(1)
        return {
            "z": z,
            "delta": delta,
            "g": g,
            "supervised": supervised,
            "unsupervised": unsupervised,
            "mask": mask,
            "loss": loss.detach(),
            "logits": logits.detach(),
            "reconstruction_error": relative(g.sum(0), actual),
            "decomposition_error": relative(supervised + unsupervised, g),
        }
    finally:
        handle.remove()


@torch.no_grad()
def assess(model, data):
    logits = model(data["tokens"], data["positions"])
    loss = losses(logits, data["labels"])
    first = model(data["prompts"], (data["lengths"] - 1)[:, None])[:, 0].argmax(-1)
    continuation = data["tokens"].clone()
    rows = torch.arange(len(first), device=first.device)
    continuation[rows, data["lengths"]] = first
    ended = model(continuation, data["lengths"][:, None])[:, 0].argmax(-1).eq(EOS)
    return loss, first, ended


def frozen_files(config):
    paths = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/run_bios_path_transfer.py",
        ROOT / "tests/test_bios_path_transfer.py",
    ]
    for name in ("bios_model.py", "bios_cross.py", "bios_cross_train.py", "bios_data.py"):
        paths.append(ROOT / "src/llm_memory_editability" / name)
    for world in config["worlds"]:
        directory = ROOT / config["source_world_root"] / f"world-{world}"
        paths += [
            directory / "world.npz",
            directory / "metadata.json",
            directory / "evaluation.npz",
        ]
    for world, seed in itertools.product(config["worlds"], config["seeds"]):
        directory = ROOT / config["parent_root"] / f"world-{world}-seed-{seed}-neither"
        paths += [directory / "config.json", directory / "learning-complete.json"]
        paths += [directory / f"model-{step}.pt" for step in config["states"]]
    return paths


def freeze(config):
    directory = ROOT / config["artifacts"]
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / "preregistration-lock.json"
    if lock_path.exists():
        raise FileExistsError("Cannot replace a prospective lock")
    protocol = (
        "## 18. " + (ROOT / "docs/experimental-protocol.md").read_text().split("## 18. ", 1)[1]
    )
    (directory / "preregistration.md").write_text(protocol)
    manifests = {}
    for world in config["worlds"]:
        generated = make_cross_world(world, ROOT / config["source_world_root"])
        selected = cases(generated)
        if len(selected) != config["cases_per_state"]:
            raise ValueError("Case count mismatch")
        manifests[str(world)] = {
            "truth_sha256": array_hash(generated.answers),
            "prompts_sha256": array_hash(generated.prompts),
            "cases": selected,
        }
    write_json(directory / "cases.json", manifests)
    sources = {}
    for path in frozen_files(config):
        relative_path = path.relative_to(ROOT)
        sources[str(relative_path)] = sha(path)
        if relative_path.parts[0] in ("src", "scripts", "tests", "configs"):
            destination = directory / "frozen-files" / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    lock = {
        "created_at": stamp(),
        "config": config,
        "sources": sources,
        "protocol_sha256": sha(directory / "preregistration.md"),
        "cases_sha256": sha(directory / "cases.json"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    write_json(lock_path, lock)
    return {"path": str(lock_path), "sha256": sha(lock_path)}


def verify_lock(config):
    directory = ROOT / config["artifacts"]
    path = directory / "preregistration-lock.json"
    lock = json.loads(path.read_text())
    if lock["config"] != config or lock["torch"] != torch.__version__:
        raise ValueError("Frozen config/runtime mismatch")
    for name, digest in lock["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen input changed: " + name)
    for name, key in (("preregistration.md", "protocol_sha256"), ("cases.json", "cases_sha256")):
        if sha(directory / name) != lock[key]:
            raise ValueError("Prospective manifest changed")
    return sha(path)


def save_factors(path, baseline, sources, data):
    arrays = {
        "positions": data["positions"].cpu().numpy(),
        "labels": data["labels"].cpu().numpy(),
        "tokens": data["tokens"].cpu().numpy(),
    }
    for prefix, result in [("evaluation", baseline), *sources.items()]:
        for key in ("z", "delta", "loss"):
            arrays[prefix + "_" + key] = result[key].cpu().numpy()
    np.savez_compressed(path, **arrays)


def run_case(model, world, case, config, destination):
    layer = config["target_layer"]
    w = model.blocks[layer].mlp.down.weight
    parent = w.detach().clone()
    frozen_before = {
        name: tensor_sha(value)
        for name, value in model.state_dict().items()
        if name != config["target_parameter"]
    }
    data = example_data(world, case, w.device)
    baseline = measure(model, data, layer, "full")
    before, old_prediction, old_ended = assess(model, data)
    sources, diagnostics, updates = {}, [], []
    with torch.no_grad():
        if not torch.equal(before, baseline["loss"]):
            raise ValueError("Evaluation/gradient forward mismatch")
    for arm in config["arms"]:
        source = measure(model, subset(data, slice(0, 1)), layer, arm)
        sources[arm] = source
        error = (source["logits"] - sources["full"]["logits"]).abs().max().item()
        if error > config["forward_tolerance"]:
            raise ValueError(f"Forward preservation failed: {error}")
        if (
            max(source["reconstruction_error"], baseline["reconstruction_error"])
            > config["gradient_tolerance"]
        ):
            raise ValueError("Outer-product reconstruction failed")
        direction = source["g"][0]
        gfull = sources["full"]["g"][0]
        if arm == "no_cross" and norm(source["unsupervised"]) > 1e-9:
            raise ValueError("Cross-position cut did not eliminate unsupervised-position gradient")
        for j, role in enumerate(case["roles"]):
            row = {
                "arm": arm,
                "role": role,
                "probe": j,
                "forward_error": error,
                "reconstruction_error": source["reconstruction_error"],
                "gradient_norm": norm(direction),
                "gradient_cosine_full": cosine(direction, gfull),
                "supervised_norm": norm(source["supervised"]),
                "unsupervised_norm": norm(source["unsupervised"]),
                "kernel": inner(baseline["g"][j], direction),
                "source_loss": source["loss"].item(),
                "probe_loss": before[j].item(),
            }
            for left, right in itertools.product(("supervised", "unsupervised"), repeat=2):
                row[f"kernel_{left}_{right}"] = inner(baseline[left][j], source[right][0])
            diagnostics.append(row)
        for scale, fraction in itertools.product(config["scales"], config["step_fractions"]):
            denominator = norm(gfull if scale == "shared_lr" else direction)
            eta = fraction * norm(parent) / denominator if denominator else 0.0
            with torch.no_grad():
                w.copy_(parent - eta * direction)
                displacement = w.detach() - parent
                weight_hash = tensor_sha(w)
            after, predicted_token, ended = assess(model, data)
            predicted = (baseline["g"].double() * displacement.double()).sum((-1, -2))
            for j, role in enumerate(case["roles"]):
                updates.append(
                    {
                        "arm": arm,
                        "scale": scale,
                        "fraction": fraction,
                        "eta": eta,
                        "update_norm": norm(displacement),
                        "weight_sha256": weight_hash,
                        "role": role,
                        "probe": j,
                        "id": case["ids"][j],
                        "target": case["targets"][j],
                        "old_target": case["old_targets"][j],
                        "before": before[j].item(),
                        "after": after[j].item(),
                        "observed_change": (after[j] - before[j]).item(),
                        "predicted_change": predicted[j].item(),
                        "ideal_predicted_change": -eta * inner(baseline["g"][j], direction),
                        "baseline_prediction": int(old_prediction[j]),
                        "baseline_ended": bool(old_ended[j]),
                        "prediction": int(predicted_token[j]),
                        "ended": bool(ended[j]),
                    }
                )
            with torch.no_grad():
                w.copy_(parent)
    if not torch.equal(w, parent):
        raise ValueError("Parent weight was not restored")
    frozen_after = {
        name: tensor_sha(value)
        for name, value in model.state_dict().items()
        if name != config["target_parameter"]
    }
    if frozen_before != frozen_after:
        raise ValueError("A frozen parameter changed")
    destination.mkdir(parents=True, exist_ok=True)
    save_factors(destination / "factors.npz", baseline, sources, data)
    write_json(
        destination / "measurements.json",
        {"case": case, "diagnostics": diagnostics, "updates": updates},
    )
    write_json(
        destination / "receipt.json",
        {
            "finished_at": stamp(),
            "files": {
                name: sha(destination / name) for name in ("factors.npz", "measurements.json")
            },
            "baseline_reconstruction_error": baseline["reconstruction_error"],
            "frozen_parameters_unchanged": True,
            "parent_restored": True,
        },
    )


def verify_receipt(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    for name, digest in receipt["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Receipt mismatch: " + str(directory / name))


def run(config, world_id, seed, device):
    lock_hash = verify_lock(config)
    if world_id not in config["worlds"] or seed not in config["seeds"]:
        raise ValueError("Worker outside the frozen matrix")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world = make_cross_world(world_id, ROOT / config["source_world_root"])
    name = f"world-{world_id}-seed-{seed}-neither"
    source = ROOT / config["parent_root"] / name
    out = ROOT / config["output"] / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        write_json(
            out / "runtime.json",
            {
                "started_at": stamp(),
                "lock_sha256": lock_hash,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if "cuda" in str(device) else None,
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__,
            },
        )
        for step in config["states"]:
            checkpoint = torch.load(
                source / f"model-{step}.pt", map_location="cpu", weights_only=False
            )
            if checkpoint["step"] != step:
                raise ValueError("Parent step mismatch")
            model = CausalLM(ModelConfig(**checkpoint["config"])).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            for pname, parameter in model.named_parameters():
                parameter.requires_grad_(pname == config["target_parameter"])
            selected = cases(world)
            state_out = out / f"step-{step}"
            state_out.mkdir(exist_ok=True)
            # Original-forward repeated gradient, not a new scientific sample.
            sham_data = subset(example_data(world, selected[0], device), slice(0, 1))
            first = measure(model, sham_data, config["target_layer"], "full")
            second = measure(model, sham_data, config["target_layer"], "full")
            sham_error = relative(first["g"], second["g"])
            if sham_error > config["gradient_tolerance"]:
                raise ValueError("Sham gradient failed")
            write_json(state_out / "sham.json", {"relative_gradient_error": sham_error})
            for case in selected:
                destination = state_out / case["name"]
                if (destination / "receipt.json").exists():
                    verify_receipt(destination)
                else:
                    run_case(model, world, case, config, destination)
                print(
                    json.dumps(
                        {
                            "event": "case",
                            "world": world_id,
                            "seed": seed,
                            "step": step,
                            "case": case["name"],
                        }
                    ),
                    flush=True,
                )
            del model, checkpoint
        files = {
            str(path.relative_to(out)): sha(path)
            for path in out.rglob("*")
            if path.is_file() and path.name not in (".lock", "complete.json")
        }
        write_json(
            out / "complete.json",
            {
                "finished_at": stamp(),
                "seconds": time.monotonic() - started,
                "lock_sha256": lock_hash,
                "cases": len(config["states"]) * len(selected),
                "files": files,
            },
        )


def dispatch(config, gpus):
    verify_lock(config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    jobs = list(itertools.product(config["worlds"], config["seeds"]))
    active, finished = [], []
    with (output / ".dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while jobs or active:
            for gpu in gpus:
                if jobs and not any(item["gpu"] == gpu for item in active):
                    world, seed = jobs.pop(0)
                    path = output / f"world-{world}-seed-{seed}-neither"
                    if (path / "complete.json").exists():
                        finished.append({"world": world, "seed": seed, "reused": True})
                        continue
                    log = (output / f"world-{world}-seed-{seed}.log").open("a")
                    env = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONPATH": str(ROOT / "src"),
                        "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2",
                    }
                    command = [
                        sys.executable,
                        str(ROOT / "scripts/run_bios_path_transfer.py"),
                        "worker",
                        "--world",
                        str(world),
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda:0",
                    ]
                    proc = subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    active.append(
                        {"gpu": gpu, "world": world, "seed": seed, "process": proc, "log": log}
                    )
            for item in active[:]:
                code = item["process"].poll()
                if code is not None:
                    item["log"].close()
                    record = {k: item[k] for k in ("gpu", "world", "seed")}
                    finished.append({**record, "returncode": code})
                    active.remove(item)
            write_json(
                output / "dispatch.json",
                {
                    "updated_at": stamp(),
                    "pending": jobs,
                    "running": [{k: x[k] for k in ("gpu", "world", "seed")} for x in active],
                    "finished": finished,
                },
            )
            if active:
                time.sleep(2)
    if any(item.get("returncode", 0) for item in finished):
        raise RuntimeError("One or more workers failed; inspect preserved logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "worker", "dispatch"))
    parser.add_argument("--world", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", type=int, nargs="+", default=[5, 8])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.command == "freeze":
        print(json.dumps(freeze(config)))
    elif args.command == "worker":
        run(config, args.world, args.seed, torch.device(args.device))
    else:
        dispatch(config, args.gpus)


if __name__ == "__main__":
    main()
