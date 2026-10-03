"""Equal-token causal-prefix intervention, with no bridge supervision."""

from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import diagnose_realworld_loop as common
import numpy as np
import torch

ROOT = common.PROJECT / "results/realworld-loop-prefix-v1"
ART = common.PROJECT / "docs/development-artifacts/realworld-loop-prefix-v1"
DATA = common.PROJECT / "data/realworld-loop-prefix-v1/prepared.json"
SEED = 914001
NODES = (0, 500, 2000, 8000)


def parent_path(arch):
    path = common.ROOT / "development" / f"{arch}-canonical"
    assert json.loads((path / "audit.json").read_text())["passed"]
    return path / "latest.pt"


def questions(atom_question, second_relation):
    return {
        "compatible": atom_question + "\nAnswer:\nNext relation: " + second_relation + ".",
        "displaced": atom_question + "\nNext relation: " + second_relation + ".\nAnswer:",
    }


def prepare():
    assert not DATA.exists()
    original, _, source_manifest = common.load_data("canonical")
    tokenizer = common.GPT2TokenizerFast.from_pretrained(
        common.BASE["tokenizer"], local_files_only=True
    )
    lookup = {r["id"]: r for r in original["atoms"]}
    payload = {
        "atoms": original["atoms"],
        "panels": original["panels"],
        "compatible": {},
        "displaced": {},
    }
    checked = 0
    for split in ["train_compositions", "evaluation_compositions"]:
        for condition in ["compatible", "displaced"]:
            payload[condition][split] = []
        for row in original[split]:
            atom = lookup[row["atom_ids"][0]]
            texts = questions(atom["question"], row["edges"][1][1])
            encoded = {
                k: common.encode_example(tokenizer, q, row["answer"]) for k, q in texts.items()
            }
            a, b = encoded["compatible"], encoded["displaced"]
            assert a["target"] == b["target"] == row["encoded"]["target"]
            assert len(a["prefix"]) == len(b["prefix"])
            assert collections.Counter(a["prefix"]) == collections.Counter(b["prefix"])
            prefix = atom["encoded"]["prefix"]
            assert a["prefix"][: len(prefix)] == prefix
            assert b["prefix"][: len(prefix)] != prefix
            for condition in texts:
                payload[condition][split].append(
                    {
                        **copy.deepcopy(row),
                        "question": texts[condition],
                        "encoded": encoded[condition],
                    }
                )
            checked += 1
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    manifest = {
        "created_utc": common.utc(),
        "prepared_sha256": common.sha256(DATA),
        "source_data_sha256": source_manifest["prepared_sha256"],
        "checked_pairs": checked,
        "parent_sha256": {a: common.sha256(parent_path(a)) for a in ["standard8", "loop4x2"]},
        "nodes": NODES,
        "seed": SEED,
        "source_sha256": {
            str(p): common.sha256(p)
            for p in [Path(__file__), ART / "design.md", Path(common.__file__)]
        },
    }
    common.write_json(ART / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def load_data(condition):
    manifest = json.loads((ART / "manifest.json").read_text())
    assert common.sha256(DATA) == manifest["prepared_sha256"]
    payload = json.loads(DATA.read_text())
    return {**payload[condition], "atoms": payload["atoms"], "panels": payload["panels"]}, manifest


def worker(arch, condition, gpu):
    device = common.configure(gpu)
    out = ROOT / "development" / f"{arch}-{condition}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "run.json").exists()
    data, manifest = load_data(condition)
    assert common.sha256(parent_path(arch)) == manifest["parent_sha256"][arch]
    spec = common.spec_for(arch)
    spec.update(sampling_seed=SEED, dropout_seed=SEED, steps=NODES[-1])
    tokenizer = common.GPT2TokenizerFast.from_pretrained(
        common.BASE["tokenizer"], local_files_only=True
    )
    model = common.construct(common.BASE["model"], spec, device)
    source = torch.load(parent_path(arch), map_location="cpu", weights_only=False)
    assert source["step"] == 8000
    model.load_state_dict(source["model"])
    optimizer = common.optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    optimizer.load_state_dict(source["optimizer"])
    del source
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    counts = {k: np.zeros(len(data[k]), dtype=np.int64) for k in ["atoms", "train_compositions"]}
    common.write_json(
        out / "run.json",
        {
            "spec": {**spec, "condition": condition, "parent_total_step": 308000},
            "pid": os.getpid(),
            "gpu": gpu,
            "gpu_name": torch.cuda.get_device_name(gpu),
            "model": common.BASE["model"],
            "world_sha256": manifest["prepared_sha256"],
            "initial_model_sha256": common.model_digest(model),
            "tracking_group": "realworld-loop-diagnosis-v1",
            "evaluation_nodes": NODES,
            "job_type": "causal-prefix-continuation",
        },
    )
    common.write_json(out / "learning.json", [])
    history, seconds, examples, tokens, flops, loss = [], 0.0, 0, 0, 0, None
    began = time.perf_counter()
    for step in range(NODES[-1] + 1):
        if step:
            records = []
            for key in ["atoms", "train_compositions"]:
                ids = rng.integers(len(data[key]), size=256)
                np.add.at(counts[key], ids, 1)
                records.extend(data[key][i] for i in ids)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            values = common.update(model, optimizer, records, spec, device, tokenizer.eos_token_id)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - tick
            loss = values["loss"]
            examples += len(records)
            tokens += values["effective_input_tokens"]
            flops += values["estimated_matmul_training_flops"]
        if step % 100 == 0:
            common.write_json(
                out / "status.json",
                {
                    "state": "training",
                    "step": step,
                    "loss": loss,
                    "pid": os.getpid(),
                    "training_seconds": seconds,
                    "updated_utc": common.utc(),
                },
            )
        if step in NODES:
            metrics, pred = common.evaluate_selected(model, data, [], tokenizer, device, False)
            history.append(
                {
                    "step": step,
                    "metrics": metrics,
                    "loss": loss,
                    "training_seconds": seconds,
                    "examples": examples,
                    "effective_input_tokens": tokens,
                    "estimated_matmul_training_flops": flops,
                    "wall_seconds": time.perf_counter() - began,
                }
            )
            common.write_json(out / "learning.json", history)
            common.write_json(out / f"predictions-{step:07d}.json", pred)
            print(
                json.dumps(
                    {
                        "step": step,
                        "ii": metrics["test_ii"]["alias_em"],
                        "oo": metrics["test_oo"]["alias_em"],
                    }
                ),
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "rng": rng.bit_generator.state,
                    "counts": counts,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                out / "latest.pt",
            )
    metrics, predictions = common.evaluate_selected(model, data, [], tokenizer, device, True)
    common.write_json(out / "endpoint.json", metrics)
    common.write_json(out / "endpoint-predictions.json", predictions)
    common.write_json(
        out / "status.json",
        {"state": "trained_pending_audit", "step": NODES[-1], "pid": os.getpid()},
    )


def audit(arch, condition, gpu):
    device = common.configure(gpu)
    out = ROOT / "development" / f"{arch}-{condition}"
    data, _ = load_data(condition)
    tokenizer = common.GPT2TokenizerFast.from_pretrained(
        common.BASE["tokenizer"], local_files_only=True
    )
    model = common.construct(common.BASE["model"], common.spec_for(arch), device)
    model.load_state_dict(
        torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["model"]
    )
    metrics, pred = common.evaluate_selected(model, data, [], tokenizer, device, True)
    assert metrics == json.loads((out / "endpoint.json").read_text())
    assert pred == json.loads((out / "endpoint-predictions.json").read_text())
    result = {
        "passed": True,
        "pid": os.getpid(),
        "utc": common.utc(),
        "full_predictions_recomputed": sum(map(len, pred.values())),
    }
    common.write_json(out / "audit.json", result)
    common.write_json(out / "complete.json", result)


def controller():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cases = [(a, c) for a in ["loop4x2", "standard8"] for c in ["compatible", "displaced"]]
    active, done, failed = {}, [], []
    for gpu, (arch, condition) in zip([3, 4, 5, 7], cases, strict=True):
        out = ROOT / "development" / f"{arch}-{condition}"
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "worker.log").open("a")
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "worker",
            "--arch",
            arch,
            "--condition",
            condition,
            "--gpu",
            str(gpu),
        ]
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        active[gpu] = (process, log, arch, condition, "worker")
    while active:
        for gpu, (process, log, arch, condition, mode) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            if code:
                failed.append({"arch": arch, "condition": condition, "mode": mode, "code": code})
            elif mode == "worker":
                out = ROOT / "development" / f"{arch}-{condition}"
                log = (out / "audit.log").open("a")
                cmd = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "audit",
                    "--arch",
                    arch,
                    "--condition",
                    condition,
                    "--gpu",
                    str(gpu),
                ]
                child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (child, log, arch, condition, "audit")
            else:
                done.append(f"{arch}-{condition}")
        common.write_json(
            ROOT / "controller-state.json",
            {
                "state": "running"
                if active
                else ("finished_with_failures" if failed else "complete"),
                "completed": done,
                "failed": failed,
                "active": {
                    str(g): {"pid": p.pid, "arch": a, "condition": c, "mode": m}
                    for g, (p, _, a, c, m) in active.items()
                },
                "updated_utc": common.utc(),
            },
        )
        if active:
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "audit", "controller"])
    parser.add_argument("--arch", choices=["standard8", "loop4x2"])
    parser.add_argument("--condition", choices=["compatible", "displaced"])
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.command in ["prepare", "controller"]:
        {"prepare": prepare, "controller": controller}[args.command]()
    else:
        try:
            {"worker": worker, "audit": audit}[args.command](args.arch, args.condition, args.gpu)
        except Exception:
            common.write_json(
                ROOT / "development" / f"{args.arch}-{args.condition}" / "failure.json",
                {"error": traceback.format_exc(), "mode": args.command},
            )
            raise
