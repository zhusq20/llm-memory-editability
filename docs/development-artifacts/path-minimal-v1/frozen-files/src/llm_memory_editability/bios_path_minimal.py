"""Two-layer follow-up: common-error removal versus competing new answers."""

import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_data import write_json
from .bios_model import CausalLM, ModelConfig
from .bios_path_transfer import ROOT, inner, norm, path_mode, sha, stamp, tensor_sha

CONFIG = ROOT / "configs/bios-path-minimal-v1.json"


def toy_data(config, device):
    people = config["people"]
    person = torch.arange(people, device=device).repeat_interleave(3)
    relation = torch.arange(3, device=device).repeat(people)
    value_start = 3 + people + 3
    tokens = torch.stack(
        (torch.ones_like(person), 3 + person, 3 + people + relation, torch.full_like(person, 2)), -1
    )
    labels = value_start + (person + (relation == 2)) % 3
    return tokens, labels, value_start


def probe_data(config, person, kind, device):
    tokens, old, values = toy_data(config, device)
    ids = torch.arange(person * 3, person * 3 + 3, device=device)
    a, b, c = values + (person + 1) % 3, values + (person + 2) % 3, values + person % 3
    labels = old[ids].clone()
    labels[0], labels[1] = a if kind == "coherent" else b, a
    return tokens[ids], labels, (a, b, c)


def gradients(model, tokens, labels, arm):
    projection = model.blocks[0].mlp.down
    captured = {}

    def hook(module, inputs, output):
        captured["z"], captured["h"] = inputs[0], output

    handle = projection.register_forward_hook(hook)
    try:
        with path_mode(model, 0, arm):
            logits = model(tokens)[:, -1]
            loss = F.cross_entropy(logits.double(), labels, reduction="none")
            actual, delta = torch.autograd.grad(loss.sum(), (projection.weight, captured["h"]))
        z, delta = captured["z"].detach(), delta.detach()
        g = torch.einsum("btd,btk->btdk", delta, z).sum(1)
        if norm(g.sum(0) - actual) / max(norm(actual), 1e-30) > 5e-5:
            raise ValueError("Minimal-model gradient reconstruction failed")
        return {"logits": logits.detach(), "loss": loss.detach(), "g": g, "z": z, "delta": delta}
    finally:
        handle.remove()


@torch.no_grad()
def evaluate(model, tokens, labels, values):
    logits = model(tokens)[:, -1].double()
    p = logits.softmax(-1)
    a, b, c = values
    return {
        "loss": F.cross_entropy(logits, labels, reduction="none").cpu().tolist(),
        "prediction": logits.argmax(-1).cpu().tolist(),
        "probabilities_abc": p[:, [a, b, c]].cpu().tolist(),
        "source_target_probability": p[0, labels[0]].item(),
        "derived_margin_ab": (logits[1, a] - logits[1, b]).item(),
        "isotropic_same_probability_reference": (p[1].square().sum() - p[1, a] - p[1, b]).item(),
    }


def freeze(config):
    directory = ROOT / config["artifacts"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "preregistration-lock.json"
    if path.exists():
        raise FileExistsError("Cannot replace minimal-model prospective lock")
    section = (
        "### 18.7 " + (ROOT / "docs/experimental-protocol.md").read_text().split("### 18.7 ", 1)[1]
    )
    (directory / "preregistration.md").write_text(section)
    files = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/run_bios_path_minimal.py",
        ROOT / "scripts/analyze_bios_path_margin.py",
        ROOT / "tests/test_bios_path_minimal.py",
        ROOT / "src/llm_memory_editability/bios_path_transfer.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "docs/development-artifacts/path-transfer-v1/audit.json",
    ]
    sources = {}
    for file in files:
        relative = file.relative_to(ROOT)
        sources[str(relative)] = sha(file)
        destination = directory / "frozen-files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file.read_bytes())
    write_json(
        path,
        {
            "created_at": stamp(),
            "config": config,
            "sources": sources,
            "protocol_sha256": sha(directory / "preregistration.md"),
            "torch": torch.__version__,
        },
    )
    return {"lock": str(path), "sha256": sha(path)}


def verify(config):
    directory = ROOT / config["artifacts"]
    path = directory / "preregistration-lock.json"
    lock = json.loads(path.read_text())
    if lock["config"] != config or lock["torch"] != torch.__version__:
        raise ValueError("Minimal config/runtime changed")
    for name, digest in lock["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Minimal frozen source changed: " + name)
    if sha(directory / "preregistration.md") != lock["protocol_sha256"]:
        raise ValueError("Minimal protocol changed")
    return sha(path)


def learn(config, seed, device, out):
    torch.manual_seed(seed)
    tokens, labels, values = toy_data(config, device)
    model = CausalLM(ModelConfig(values + 3, config["width"], 2, config["heads"], context=8)).to(
        device
    )
    if (out / "parent.pt").exists():
        saved = torch.load(out / "parent.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        return model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train_lr"], weight_decay=config["train_weight_decay"]
    )
    trajectory = []
    for step in range(config["train_steps"] + 1):
        logits = model(tokens)[:, -1]
        loss = F.cross_entropy(logits, labels)
        if step in (0, 128, 512, config["train_steps"]):
            trajectory.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "accuracy": logits.argmax(-1).eq(labels).float().mean().item(),
                }
            )
        if step == config["train_steps"]:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": model.config_dict(),
            "step": config["train_steps"],
        },
        out / "parent.pt",
    )
    write_json(out / "learning.json", trajectory)
    return model


def trajectory(model, parent, config, person, kind, arm, device, directory):
    model.load_state_dict(parent)
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == config["target_parameter"])
    tokens, labels, values = probe_data(config, person, kind, device)
    w = model.blocks[0].mlp.down.weight
    timeline, steps, factors = [], [], []
    for step in range(config["edit_steps"] + 1):
        measure = step % config["measure_every"] == 0 or step == config["edit_steps"]
        full = gradients(
            model, tokens if measure else tokens[:1], labels if measure else labels[:1], "full"
        )
        source = full if arm == "full" else gradients(model, tokens[:1], labels[:1], arm)
        if not torch.equal(full["logits"][:1], source["logits"][:1]):
            # Different batch sizes allow only normal FP32 GEMM rounding.
            if (full["logits"][:1] - source["logits"][:1]).abs().max().item() > 3e-5:
                raise ValueError("Minimal forward mismatch")
        raw, original = source["g"][0], full["g"][0]
        direction = raw * (norm(original) / norm(raw)) if norm(raw) else raw
        if measure:
            row = {
                "step": step,
                **evaluate(model, tokens, labels, values),
                "source_gradient_norm": norm(original),
                "intervened_gradient_norm": norm(raw),
                "kernel_source": inner(full["g"][0], direction),
                "kernel_derived": inner(full["g"][1], direction),
                "kernel_retention": inner(full["g"][2], direction),
                "kernel_derived_full": inner(full["g"][1], original),
            }
            timeline.append(row)
            factors.append(
                {
                    key: value.detach().cpu().numpy()
                    for key, value in {
                        "z": full["z"],
                        "delta": full["delta"],
                        "source_z": source["z"][:1],
                        "source_delta": source["delta"][:1],
                    }.items()
                }
            )
        if step == config["edit_steps"]:
            break
        before_weight = w.detach().clone() if measure else None
        with torch.no_grad():
            w.add_(direction, alpha=-config["edit_lr"])
        if measure:
            after = evaluate(model, tokens, labels, values)
            actual_delta = w.detach() - before_weight
            row["one_step_observed_derived"] = after["loss"][1] - row["loss"][1]
            row["one_step_predicted_derived"] = inner(full["g"][1], actual_delta)
            row["one_step_margin_change"] = after["derived_margin_ab"] - row["derived_margin_ab"]
        steps.append(
            {
                "step": step,
                "source_loss": full["loss"][0].item(),
                "direction_norm": norm(direction),
                "source_gradient_norm": norm(original),
            }
        )
    for name, value in model.state_dict().items():
        if name != config["target_parameter"] and not torch.equal(value, parent[name]):
            raise ValueError("Minimal frozen parameter changed")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "trajectory.json",
        {
            "person": person,
            "kind": kind,
            "arm": arm,
            "values_abc": values,
            "labels": labels.cpu().tolist(),
            "timeline": timeline,
            "steps": steps,
        },
    )
    np.savez_compressed(
        directory / "factors.npz",
        **{key: np.stack([factor[key] for factor in factors]) for key in factors[0]},
    )
    torch.save(w.detach().cpu(), directory / "edited-weight.pt")
    write_json(
        directory / "complete.json",
        {
            "finished_at": stamp(),
            "steps": len(steps),
            "weight_sha256": tensor_sha(w),
            "frozen_parameters_unchanged": True,
            "files": {
                name: sha(directory / name)
                for name in ("trajectory.json", "factors.npz", "edited-weight.pt")
            },
        },
    )


def run(config, seed, device):
    lock_hash = verify(config)
    if seed not in config["seeds"]:
        raise ValueError("Unknown seed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT / config["output"] / f"seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            out / "runtime.json",
            {
                "started_at": stamp(),
                "lock_sha256": lock_hash,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            },
        )
        model = learn(config, seed, device, out)
        parent = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for person, kind, arm in itertools.product(
            config["people_to_edit"], config["kinds"], config["arms"]
        ):
            directory = out / f"person-{person}-{kind}-{arm}"
            if not (directory / "complete.json").exists():
                trajectory(model, parent, config, person, kind, arm, device, directory)
            print(
                json.dumps(
                    {
                        "event": "trajectory",
                        "seed": seed,
                        "person": person,
                        "kind": kind,
                        "arm": arm,
                    }
                ),
                flush=True,
            )
        write_json(
            out / "complete.json",
            {
                "finished_at": stamp(),
                "lock_sha256": lock_hash,
                "files": {
                    str(path.relative_to(out)): sha(path)
                    for path in out.rglob("*")
                    if path.is_file() and path != out / "complete.json" and path.name != ".lock"
                },
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "worker", "dispatch"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", nargs="+", type=int, default=[5, 8])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.command == "freeze":
        print(json.dumps(freeze(config)))
    elif args.command == "worker":
        run(config, args.seed, torch.device(args.device))
    else:
        verify(config)
        output = ROOT / config["output"]
        output.mkdir(parents=True, exist_ok=True)
        workers = []
        for index, seed in enumerate(config["seeds"]):
            log = (output / f"seed-{seed}.log").open("a")
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(args.gpus[index % len(args.gpus)]),
                "PYTHONPATH": str(ROOT / "src"),
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
            }
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_bios_path_minimal.py"),
                    "worker",
                    "--seed",
                    str(seed),
                ],
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            workers.append((proc, log))
        codes = []
        for proc, log in workers:
            codes.append(proc.wait())
            log.close()
        write_json(output / "dispatch.json", {"finished_at": stamp(), "returncodes": codes})
        if any(codes):
            raise RuntimeError("Minimal worker failed; see retained logs")


if __name__ == "__main__":
    main()
