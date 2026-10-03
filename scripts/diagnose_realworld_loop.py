"""Paired continuation experiments using the frozen real-world trainer."""

from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
BASE = json.loads((PROJECT / "configs/realworld-composition-confirmation-v1.json").read_text())
sys.path.insert(0, str(Path(BASE["source_snapshot"]) / "src"))
from transformers import GPT2TokenizerFast  # noqa: E402

from llm_memory_editability.grok_depth import utc  # noqa: E402
from llm_memory_editability.realworld_composition import (  # noqa: E402
    aggregate,
    construct,
    evaluate,
    model_digest,
    optimizer_for,
    update,
)
from llm_memory_editability.realworld_composition_data import (  # noqa: E402
    QUESTION_TEMPLATES,
    encode_example,
    sha256,
    write_json,
)

ART = PROJECT / "docs/development-artifacts/realworld-loop-diagnosis-v1"
ROOT = PROJECT / "results/realworld-loop-diagnosis-v1"
DATA = PROJECT / "data/realworld-loop-diagnosis-v1/prepared.json"
SEED = 913001
NODES = (0, 500, 2000, 8000)


def subject_of(atom):
    before, after = QUESTION_TEMPLATES[atom["edge"][1]].split("{subject}")
    question = atom["question"]
    assert question.startswith(before) and question.endswith(after)
    return question[len(before) : len(question) - len(after) if after else None]


def canonical_question(subject, relations):
    return f"Starting entity: {subject}. Relation sequence: {' -> '.join(relations)}."


def augmentation_pairs(atoms, training, heldout):
    used = {a for row in training for a in row["atom_ids"]}
    outgoing = collections.defaultdict(list)
    for aid in sorted(used):
        outgoing[atoms[aid]["edge"][0]].append(aid)
    original = {tuple(r["atom_ids"]) for r in training}
    forbidden = {tuple(r["atom_ids"]) for r in heldout}
    excluded = original | forbidden
    forbidden_answers = {(r["edges"][0][0], r["edges"][1][2]) for r in heldout}
    operations = {(r["edges"][0][1], r["edges"][1][1]) for r in training}
    pairs = []
    for first in sorted(used):
        e1 = atoms[first]["edge"]
        for second in outgoing[e1[2]]:
            e2 = atoms[second]["edge"]
            if (first, second) in excluded:
                continue
            if e1[0] == e2[2] or (e1[0], e2[2]) in forbidden_answers:
                continue
            if (e1[1], e2[1]) not in operations:
                continue
            pairs.append((first, second))
    return pairs


def compact(row):
    keys = [
        "id",
        "question",
        "answer",
        "aliases",
        "unambiguous_answers",
        "encoded",
        "atom_ids",
        "edges",
        "edge",
        "role",
        "required_role",
        "canonical_answer_globally_unambiguous",
    ]
    return {k: row[k] for k in keys if k in row}


def prepare():
    assert not DATA.exists(), "Prepared data is immutable; use a new batch for changes"
    raw = json.loads(Path(BASE["data_file"]).read_text())
    assert sha256(BASE["data_file"]) == BASE["data_sha256"]
    tokenizer = GPT2TokenizerFast.from_pretrained(BASE["tokenizer"], local_files_only=True)
    atoms = {r["id"]: r for r in raw["atoms"]}
    native = {
        k: [compact(r) for r in raw[k]]
        for k in ["atoms", "train_compositions", "evaluation_compositions"]
    }
    canonical = copy.deepcopy(native)
    for kind, records in canonical.items():
        for row in records:
            if kind == "atoms":
                head, relations = subject_of(row), [row["edge"][1]]
            else:
                head = subject_of(atoms[row["atom_ids"][0]])
                relations = [e[1] for e in row["edges"]]
            row["question"] = canonical_question(head, relations)
            previous = row["encoded"]["target"]
            row["encoded"] = encode_example(tokenizer, row["question"], row["answer"])
            assert row["encoded"]["target"] == previous
    pairs = augmentation_pairs(atoms, raw["train_compositions"], raw["evaluation_compositions"])
    added = []
    for first, second in pairs:
        a, b = atoms[first], atoms[second]
        question = canonical_question(subject_of(a), [a["edge"][1], b["edge"][1]])
        row = {
            **compact(b),
            "id": "new-" + first + "-" + second,
            "question": question,
            "atom_ids": [first, second],
            "edges": [a["edge"], b["edge"]],
            "role": "II",
            "required_role": "II",
        }
        row.pop("edge", None)
        row["encoded"] = encode_example(tokenizer, question, b["answer"])
        added.append(row)
    assert added
    payload = {"native": native, "canonical": canonical, "added": added, "panels": raw["panels"]}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    counts = collections.Counter(i for r in raw["train_compositions"] for i in r["atom_ids"])
    source_files = [Path(__file__), ART / "design.md"]
    manifest = {
        "created_utc": utc(),
        "source_data_sha256": BASE["data_sha256"],
        "prepared_sha256": sha256(DATA),
        "atomic": len(atoms),
        "original_compositions": len(native["train_compositions"]),
        "added_compositions": len(added),
        "id_atoms": len(counts),
        "id_atoms_used_once": sum(v == 1 for v in counts.values()),
        "nodes": NODES,
        "seed": SEED,
        "source_sha256": {str(p): sha256(p) for p in source_files},
        "base_checkpoints": {
            arch: sha256(checkpoint_path(arch)) for arch in ["standard8", "loop4x2"]
        },
    }
    write_json(ART / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def checkpoint_path(arch):
    return Path(BASE["results_root"]) / "confirmation" / f"confirmation-{arch}-init1/latest.pt"


def spec_for(arch):
    spec = copy.deepcopy(next(s for s in BASE["runs"] if s["name"] == f"confirmation-{arch}-init1"))
    spec.update(steps=NODES[-1], microbatch_size=512)
    return spec


def evaluate_selected(model, data, added, tokenizer, device, full):
    atoms = {r["id"]: r for r in data["atoms"]}
    panels = data["panels"]
    testids = {i for k, ids in panels.items() if k.startswith("test_") for i in ids}
    test = [r for r in data["evaluation_compositions"] if full or r["id"] in testids]
    needed = {i for r in test for i in r["atom_ids"]} | set(panels["atomic"])
    trainids = set(panels["train_composition"])
    train = [r for r in data["train_compositions"] if r["id"] in trainids]
    if full:
        train = sorted(
            data["train_compositions"], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
        )[:512]
    groups = {
        "atomic": [atoms[i] for i in sorted(needed)],
        "train_composition": train,
        "test_all": test,
    }
    if added:
        groups["new_composition"] = added[:256] if full else added[:64]
    metrics, predictions = {}, {}
    for key, records in groups.items():
        metrics[key], predictions[key] = evaluate(model, records, tokenizer, device)
    for role in ["II", "IO", "OI", "OO"]:
        metrics["test_" + role.lower()] = aggregate(
            [r for r in predictions["test_all"] if r["role"] == role]
        )
    lookup = {r["id"]: r for r in predictions["atomic"]}
    correct = {r["id"] for r in test if all(lookup[a]["alias_em"] for a in r["atom_ids"])}
    metrics["prerequisites_correct"] = aggregate(
        [r for r in predictions["test_all"] if r["id"] in correct]
    )
    return metrics, predictions


def load_data(condition):
    manifest = json.loads((ART / "manifest.json").read_text())
    assert sha256(DATA) == manifest["prepared_sha256"]
    payload = json.loads(DATA.read_text())
    data = payload["native" if condition == "native" else "canonical"]
    data["panels"] = payload["panels"]
    added = payload["added"] if condition != "native" else []
    return data, added, manifest


def configure(gpu):
    assert os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] == "/lib64"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    return f"cuda:{gpu}"


def engineering(gpu, reload_only=False):
    device = configure(gpu)
    data, _, _ = load_data("canonical")
    tokenizer = GPT2TokenizerFast.from_pretrained(BASE["tokenizer"], local_files_only=True)
    spec = spec_for("loop4x2")
    model = construct(BASE["model"], spec, device)
    out = ROOT / "engineering"
    out.mkdir(parents=True, exist_ok=True)
    if reload_only:
        model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
        expected = json.loads((out / "forward.json").read_text())
        metrics, predictions = evaluate(model, data["atoms"][:8], tokenizer, device)
        assert expected["predictions"] == predictions and expected["metrics"] == metrics
        assert expected["pid"] != os.getpid()
        write_json(out / "audit.json", {"passed": True, "pid": os.getpid(), "utc": utc()})
        return
    source = torch.load(checkpoint_path("loop4x2"), map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"])
    optimizer = optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    optimizer.load_state_dict(source["optimizer"])
    del source
    records = data["atoms"][:256] + data["train_compositions"][:256]
    timing, losses = [], []
    for _ in range(8):
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = update(model, optimizer, records, spec, device, tokenizer.eos_token_id)
        torch.cuda.synchronize()
        timing.append(time.perf_counter() - began)
        losses.append(result["loss"])
    metrics, predictions = evaluate(model, data["atoms"][:8], tokenizer, device)
    torch.save(model.state_dict(), out / "model.pt")
    write_json(
        out / "forward.json",
        {
            "pid": os.getpid(),
            "metrics": metrics,
            "predictions": predictions,
            "seconds": timing,
            "losses": losses,
        },
    )
    print(json.dumps({"seconds": timing, "losses": losses}), flush=True)


def worker(arch, condition, gpu):
    device = configure(gpu)
    out = ROOT / "development" / f"{arch}-{condition}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "run.json").exists(), "Do not overwrite a trajectory"
    data, added, manifest = load_data(condition)
    assert sha256(checkpoint_path(arch)) == manifest["base_checkpoints"][arch]
    spec = spec_for(arch)
    tokenizer = GPT2TokenizerFast.from_pretrained(BASE["tokenizer"], local_files_only=True)
    model = construct(BASE["model"], spec, device)
    source = torch.load(checkpoint_path(arch), map_location="cpu", weights_only=False)
    assert source["step"] == 300000
    model.load_state_dict(source["model"])
    optimizer = optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    optimizer.load_state_dict(source["optimizer"])
    del source
    torch.manual_seed(SEED)
    rngs = [np.random.default_rng(SEED + i) for i in range(3)]
    counts = {
        k: np.zeros(len(v), dtype=np.int64)
        for k, v in [
            ("atoms", data["atoms"]),
            ("original", data["train_compositions"]),
            ("added", added),
        ]
    }
    metadata = {
        "spec": {
            **spec,
            "condition": condition,
            "parent_step": 300000,
            "parent_sha256": manifest["base_checkpoints"][arch],
        },
        "model": BASE["model"],
        "pid": os.getpid(),
        "gpu": gpu,
        "gpu_name": torch.cuda.get_device_name(gpu),
        "world_sha256": manifest["prepared_sha256"],
        "initial_model_sha256": model_digest(model),
        "evaluation_nodes": NODES,
        "tracking_group": ROOT.name,
        "phase": "development",
        "job_type": "causal-continuation",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    write_json(out / "run.json", metadata)
    write_json(out / "learning.json", [])
    began = time.perf_counter()
    history, seconds, examples, tokens, flops, loss = [], 0.0, 0, 0, 0, None
    for step in range(NODES[-1] + 1):
        if step:
            ai = rngs[0].integers(len(data["atoms"]), size=256)
            ci = rngs[1].integers(len(data["train_compositions"]), size=128)
            pool = added if condition == "augmented" else data["train_compositions"]
            ei = rngs[2].integers(len(pool), size=128)
            records = (
                [data["atoms"][i] for i in ai]
                + [data["train_compositions"][i] for i in ci]
                + [pool[i] for i in ei]
            )
            np.add.at(counts["atoms"], ai, 1)
            np.add.at(counts["original"], ci, 1)
            np.add.at(counts["added" if condition == "augmented" else "original"], ei, 1)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            values = update(model, optimizer, records, spec, device, tokenizer.eos_token_id)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - tick
            loss = values["loss"]
            examples += len(records)
            tokens += values["effective_input_tokens"]
            flops += values["estimated_matmul_training_flops"]
        if step % 100 == 0:
            write_json(
                out / "status.json",
                {
                    "state": "training",
                    "step": step,
                    "loss": loss,
                    "pid": os.getpid(),
                    "training_seconds": seconds,
                    "updated_utc": utc(),
                },
            )
        if step in NODES:
            metrics, predictions = evaluate_selected(model, data, added, tokenizer, device, False)
            record = {
                "step": step,
                "metrics": metrics,
                "loss": loss,
                "training_seconds": seconds,
                "examples": examples,
                "effective_input_tokens": tokens,
                "estimated_matmul_training_flops": flops,
                "wall_seconds": time.perf_counter() - began,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            history.append(record)
            write_json(out / "learning.json", history)
            write_json(out / f"predictions-{step:07d}.json", predictions)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss,
                        "test_ii": metrics["test_ii"]["alias_em"],
                        "test_oo": metrics["test_oo"]["alias_em"],
                    }
                ),
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "rngs": [r.bit_generator.state for r in rngs],
                    "counts": counts,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                out / "latest.pt",
            )
    metrics, predictions = evaluate_selected(model, data, added, tokenizer, device, True)
    write_json(out / "endpoint.json", metrics)
    write_json(out / "endpoint-predictions.json", predictions)
    write_json(
        out / "status.json",
        {
            "state": "trained_pending_audit",
            "step": NODES[-1],
            "pid": os.getpid(),
            "model_sha256": model_digest(model),
            "updated_utc": utc(),
        },
    )


def audit(arch, condition, gpu):
    device = configure(gpu)
    out = ROOT / "development" / f"{arch}-{condition}"
    data, added, _ = load_data(condition)
    tokenizer = GPT2TokenizerFast.from_pretrained(BASE["tokenizer"], local_files_only=True)
    model = construct(BASE["model"], spec_for(arch), device)
    model.load_state_dict(
        torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["model"]
    )
    metrics, predictions = evaluate_selected(model, data, added, tokenizer, device, True)
    assert predictions == json.loads((out / "endpoint-predictions.json").read_text())
    assert metrics == json.loads((out / "endpoint.json").read_text())
    result = {
        "passed": True,
        "pid": os.getpid(),
        "completed_utc": utc(),
        "full_predictions_recomputed": sum(len(v) for v in predictions.values()),
    }
    write_json(out / "audit.json", result)
    write_json(out / "complete.json", result)


def controller():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    todo = [(a, c) for c in ["canonical", "native", "augmented"] for a in ["standard8", "loop4x2"]]
    active, done, failed = {}, [], []
    while todo or active:
        for gpu in [3, 4, 5, 6, 7]:
            if gpu in active or not todo:
                continue
            arch, condition = todo.pop(0)
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
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (proc, log, arch, condition, "worker")
        for gpu, (proc, log, arch, condition, mode) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            if code:
                failed.append(
                    {"arch": arch, "condition": condition, "mode": mode, "exit_code": code}
                )
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
        state = (
            "running" if todo or active else ("finished_with_failures" if failed else "complete")
        )
        write_json(
            ROOT / "controller-state.json",
            {
                "state": state,
                "pid": os.getpid(),
                "queued": todo,
                "completed": done,
                "failed": failed,
                "active": {
                    str(g): {"pid": p.pid, "arch": a, "condition": c, "mode": m}
                    for g, (p, _, a, c, m) in active.items()
                },
                "updated_utc": utc(),
            },
        )
        if todo or active:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "worker", "audit", "controller", "engineering", "engineering-audit"],
    )
    parser.add_argument("--arch", choices=["standard8", "loop4x2"])
    parser.add_argument("--condition", choices=["native", "canonical", "augmented"])
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "controller":
        controller()
    elif args.command.startswith("engineering"):
        engineering(args.gpu, args.command == "engineering-audit")
    else:
        try:
            {"worker": worker, "audit": audit}[args.command](args.arch, args.condition, args.gpu)
        except Exception:
            path = ROOT / "development" / f"{args.arch}-{args.condition}" / "failure.json"
            write_json(path, {"error": traceback.format_exc(), "mode": args.command, "utc": utc()})
            raise


if __name__ == "__main__":
    main()
