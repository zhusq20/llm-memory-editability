"""Same real graph, paired natural-name versus single-token-entity training."""

from __future__ import annotations

import argparse
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

import diagnose_realworld_loop as common
import numpy as np
import torch

from llm_memory_editability.realworld_composition import shared_prefix_digest

ROOT = common.PROJECT / "results/realworld-loop-entity-v1"
ART = common.PROJECT / "docs/development-artifacts/realworld-loop-entity-v1"
DATA = common.PROJECT / "data/realworld-loop-entity-v1/prepared.json"
TOKENIZER = DATA.parent / "tokenizer"
SEED = 915001
NODES = (0, 2000, 8000, 32000)


class NodeCodec:
    """Decode pre-encoded data without rebuilding a 70k-added-token trie."""

    def __init__(self):
        self.base = common.GPT2TokenizerFast.from_pretrained(
            common.BASE["tokenizer"], local_files_only=True
        )
        self.eos_token_id = self.base.eos_token_id
        self.offset = len(self.base)

    def decode(self, ids, skip_special_tokens=True):
        parts, ordinary = [], []

        def flush():
            if ordinary:
                parts.append(
                    self.base.decode(
                        ordinary,
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=False,
                    )
                )
                ordinary.clear()

        for token in ids:
            if token < self.offset:
                ordinary.append(token)
            else:
                flush()
                parts.append(f"<node{token - self.offset}>")
        flush()
        return self.base.clean_up_tokenization("".join(parts))


def prepare():
    assert not DATA.exists()
    natural, _, _ = common.load_data("canonical")
    tokenizer = common.GPT2TokenizerFast.from_pretrained(
        common.BASE["tokenizer"], local_files_only=True
    )
    entities = sorted({x for r in natural["atoms"] for x in [r["edge"][0], r["edge"][2]]})
    names = {entity: f"<node{i}>" for i, entity in enumerate(entities)}
    first_token = len(tokenizer)
    assert tokenizer.add_tokens(list(names.values())) == len(entities)
    token_ids = {e: first_token + i for i, e in enumerate(entities)}
    tokenizer.save_pretrained(TOKENIZER)
    symbolic = copy.deepcopy(natural)
    checked = 0
    for split in ["atoms", "train_compositions", "evaluation_compositions"]:
        for source, row in zip(natural[split], symbolic[split], strict=True):
            edges = [row["edge"]] if split == "atoms" else row["edges"]
            head, tail = edges[0][0], edges[-1][2]
            row["question"] = common.canonical_question(names[head], [e[1] for e in edges])
            row["answer"] = names[tail]
            row["aliases"] = []
            row["unambiguous_answers"] = [names[tail]]
            row["canonical_answer_globally_unambiguous"] = True
            # Added tokens split off the leading space; supervise the entity, not
            # an extra whitespace token, while keeping the common QA prefix.
            prefix = tokenizer.encode(
                "Question: " + row["question"] + "\nAnswer:", add_special_tokens=False
            )
            target = [token_ids[tail], tokenizer.eos_token_id]
            row["encoded"] = {"prefix": prefix, "target": target, "input": prefix + [target[0]]}
            assert tokenizer.convert_ids_to_tokens(token_ids[head]) == names[head]
            assert prefix.count(token_ids[head]) == 1
            assert row["id"] == source["id"]
            assert row.get("atom_ids") == source.get("atom_ids")
            assert row.get("role") == source.get("role")
            assert tokenizer.decode(target[:-1]).strip() == row["answer"]
            checked += 1
    payload = {"natural": natural, "entity": symbolic}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    model = {
        **common.BASE["model"],
        "vocab_size": len(tokenizer),
        "hidden_size": 256,
        "attention_heads": 4,
    }
    common.write_json(
        ART / "manifest.json",
        {
            "prepared_sha256": common.sha256(DATA),
            "entities": len(entities),
            "vocab_size": len(tokenizer),
            "checked_records": checked,
            "entity_map_sha256": hashlib.sha256(
                json.dumps(token_ids, sort_keys=True).encode()
            ).hexdigest(),
            "model": model,
            "nodes": NODES,
            "seed": SEED,
            "created_utc": common.utc(),
            "source_sha256": {
                str(p): common.sha256(p) for p in [Path(__file__), ART / "design.md"]
            },
            "tokenizer_sha256": {
                str(p): common.sha256(p) for p in sorted(TOKENIZER.glob("*")) if p.is_file()
            },
        },
    )
    common.write_json(DATA.parent / "entity-map.json", token_ids)
    print(json.dumps({"entities": len(entities), "model": model, "checked": checked}), flush=True)


def load(condition):
    manifest = json.loads((ART / "manifest.json").read_text())
    assert common.sha256(DATA) == manifest["prepared_sha256"]
    assert all(common.sha256(p) == h for p, h in manifest["tokenizer_sha256"].items())
    return json.loads(DATA.read_text())[condition], manifest


def spec_for(arch):
    spec = common.spec_for(arch)
    spec.update(
        initialization=SEED,
        sampling_seed=SEED,
        dropout_seed=SEED,
        learning_rate=1e-4,
        weight_decay=0.1,
        steps=NODES[-1],
        microbatch_size=512,
    )
    return spec


def engineering(gpu):
    device = common.configure(gpu)
    results = {}
    for condition in ["natural", "entity"]:
        data, manifest = load(condition)
        spec = spec_for("loop4x2")
        model = common.construct(manifest["model"], spec, device)
        optimizer = common.optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
        tokenizer = NodeCodec()
        records = data["atoms"][:256] + data["train_compositions"][:256]
        times, losses = [], []
        for _ in range(8):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = common.update(model, optimizer, records, spec, device, tokenizer.eos_token_id)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            losses.append(value["loss"])
        results[condition] = {"seconds": times, "losses": losses}
        del model, optimizer, data
        torch.cuda.empty_cache()
    common.write_json(ROOT / "engineering.json", results)
    print(json.dumps(results), flush=True)


def worker(arch, condition, gpu):
    device = common.configure(gpu)
    out = ROOT / "development" / f"{arch}-{condition}"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "run.json").exists()
    data, manifest = load(condition)
    spec = spec_for(arch)
    tokenizer = NodeCodec()
    model = common.construct(manifest["model"], spec, device)
    optimizer = common.optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    counts = {k: np.zeros(len(data[k]), dtype=np.int64) for k in ["atoms", "train_compositions"]}
    common.write_json(
        out / "run.json",
        {
            "spec": {**spec, "condition": condition},
            "model": manifest["model"],
            "pid": os.getpid(),
            "gpu": gpu,
            "gpu_name": torch.cuda.get_device_name(gpu),
            "world_sha256": manifest["prepared_sha256"],
            "initial_model_sha256": common.model_digest(model),
            "common_prefix_sha256": shared_prefix_digest(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "tracking_group": "realworld-loop-diagnosis-v1",
            "evaluation_nodes": NODES,
            "job_type": "entity-representation",
        },
    )
    common.write_json(out / "learning.json", [])
    history = []
    seconds = 0.0
    examples = tokens = flops = 0
    loss = None
    began = time.perf_counter()
    for step in range(NODES[-1] + 1):
        if step:
            records = []
            for key in ["atoms", "train_compositions"]:
                ids = rng.integers(len(data[key]), size=256)
                np.add.at(counts[key], ids, 1)
                records.extend(data[key][i] for i in ids)
            for group in optimizer.param_groups:
                group["lr"] = spec["learning_rate"] * min((step - 1) / 2000, 1.0)
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
                        "atomic": metrics["atomic"]["alias_em"],
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
    metrics, pred = common.evaluate_selected(model, data, [], tokenizer, device, True)
    common.write_json(out / "endpoint.json", metrics)
    common.write_json(out / "endpoint-predictions.json", pred)
    common.write_json(
        out / "status.json",
        {"state": "trained_pending_audit", "step": NODES[-1], "pid": os.getpid()},
    )


def audit(arch, condition, gpu):
    device = common.configure(gpu)
    out = ROOT / "development" / f"{arch}-{condition}"
    data, manifest = load(condition)
    tokenizer = NodeCodec()
    model = common.construct(manifest["model"], spec_for(arch), device)
    model.load_state_dict(
        torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["model"]
    )
    metrics, pred = common.evaluate_selected(model, data, [], tokenizer, device, True)
    assert metrics == json.loads((out / "endpoint.json").read_text())
    assert pred == json.loads((out / "endpoint-predictions.json").read_text())
    record = {"passed": True, "pid": os.getpid(), "utc": common.utc()}
    common.write_json(out / "audit.json", record)
    common.write_json(out / "complete.json", record)


def controller():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    todo = [(a, c) for c in ["entity", "natural"] for a in ["standard8", "loop4x2"]]
    active = {}
    done = []
    failed = []
    while todo or active:
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        free = [
            int(r.split(",")[0])
            for r in memory.splitlines()
            if int(r.split(",")[0]) in [3, 4, 5, 6, 7] and int(r.split(",")[1]) < 100
        ]
        for gpu in free:
            if not todo or gpu in active:
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
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (child, log, arch, condition, "worker")
        for gpu, (p, log, arch, condition, mode) in list(active.items()):
            code = p.poll()
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
                if todo or active
                else ("finished_with_failures" if failed else "complete"),
                "queued": todo,
                "completed": done,
                "failed": failed,
                "active": {
                    str(g): {"pid": p.pid, "arch": a, "condition": c, "mode": m}
                    for g, (p, _, a, c, m) in active.items()
                },
                "updated_utc": common.utc(),
            },
        )
        if todo or active:
            time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "engineering", "worker", "audit", "controller"]
    )
    parser.add_argument("--arch", choices=["standard8", "loop4x2"])
    parser.add_argument("--condition", choices=["natural", "entity"])
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.command in ["prepare", "controller"]:
        {"prepare": prepare, "controller": controller}[args.command]()
    elif args.command == "engineering":
        engineering(args.gpu)
    else:
        try:
            {"worker": worker, "audit": audit}[args.command](args.arch, args.condition, args.gpu)
        except Exception:
            common.write_json(
                ROOT / "development" / f"{args.arch}-{args.condition}" / "failure.json",
                {"mode": args.command, "error": traceback.format_exc()},
            )
            raise
