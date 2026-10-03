"""Full-Qwen, nested compositional support with matched atomic supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import small_lm_composition as base

FIRST = ("mentor", "advisor")
SECOND = ("live", "birth")
STRICT_CITIES = (
    "Athens Prague Havana Tunis Perth Zurich Bristol Munich "
    "Tallinn Riga Quito Dakar Nairobi Accra Calgary Warsaw"
).split()
ROOT = Path("docs/development-artifacts/natural-composition-confirmation-v1")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    base.write(path, value)


def build_world(seed):
    rng = np.random.default_rng(seed)
    first_names = (
        "Rhea Mara Evan Luca Nora Iris Owen Hugo Cora Theo Nina Leon Vera Finn Dana Arlo"
    ).split()
    names = [
        f"{a} {b}"
        for a in first_names
        for b in "Hale Vale Mercer Rowan Keane Flint Arden Voss North Lark".split()
    ]
    names = rng.permutation(names).tolist()
    atoms, chains = [], []
    for pool, offset, cities in (("familiar", 0, base.CITIES), ("strict", 32, STRICT_CITIES)):
        heads, bridges = names[offset : offset + 16], names[offset + 16 : offset + 32]
        mappings = [rng.permutation(16).tolist()]
        mappings.append(np.roll(mappings[0], int(rng.choice(np.arange(2, 16, 2)))).tolist())
        city_maps = [rng.permutation(cities).tolist()]
        city_maps.append(np.roll(city_maps[0], int(rng.integers(1, 16))).tolist())
        ids = {}
        for j, head in enumerate(heads):
            for r, relation in enumerate(FIRST):
                atom = dict(
                    id=len(atoms),
                    pool=pool,
                    role="first",
                    subject=head,
                    relation=relation,
                    answer=bridges[mappings[r][j]],
                )
                atoms.append(atom)
                ids["first", j, r] = atom["id"]
        for j, bridge in enumerate(bridges):
            for s, relation in enumerate(SECOND):
                atom = dict(
                    id=len(atoms),
                    pool=pool,
                    role="second",
                    subject=bridge,
                    relation=relation,
                    answer=city_maps[s][j],
                )
                atoms.append(atom)
                ids["second", j, s] = atom["id"]
        for j, head in enumerate(heads):
            for r, first in enumerate(FIRST):
                for s, second in enumerate(SECOND):
                    bridge_index = mappings[r][j]
                    chains.append(
                        dict(
                            id=len(chains),
                            pool=pool,
                            head_index=j,
                            head=head,
                            bridge=bridges[bridge_index],
                            first=first,
                            second=second,
                            answer=city_maps[s][bridge_index],
                            atom_ids=[ids["first", j, r], ids["second", bridge_index, s]],
                            split="strict"
                            if pool == "strict"
                            else ("support" if s == (r ^ (j % 2)) else "heldout"),
                        )
                    )
    return dict(
        seed=seed,
        atoms=atoms,
        chains=chains,
        high_support=[c["id"] for c in chains if c["split"] == "support"],
        low_support=[c["id"] for c in chains if c["split"] == "support" and c["head_index"] < 4],
    )


def question(row, variant=0, atomic=False):
    if atomic:
        subject, relation = row["subject"], row["relation"]
        if row["role"] == "first":
            templates = (
                "Who is {subject}'s {relation}?",
                "Who is the {relation} of {subject}?",
                "Name the {relation} of {subject}.",
                "Give the name of {subject}'s {relation}.",
                "Give the name of the person serving as {subject}'s {relation}.",
            )
        elif relation == "live":
            templates = (
                "Which city does {subject} live in?",
                "What city is {subject}'s home?",
                "Where does {subject} live?",
                "Name the city where {subject} lives.",
                "In which city is {subject} living?",
            )
        else:
            templates = (
                "Which city was {subject} born in?",
                "What is {subject}'s birth city?",
                "Where was {subject} born?",
                "Name the city where {subject} was born.",
                "In which city did {subject}'s birth take place?",
            )
        return templates[variant].format(subject=subject, relation=relation)
    subject = row["head"]
    relation = row["first"]
    if row["second"] == "live":
        templates = (
            "Which city does {subject}'s {relation} live in?",
            "What city is home to {subject}'s {relation}?",
            "Where does the {relation} of {subject} live?",
            "Name the city where {subject}'s {relation} lives.",
            "In which city is the person serving as {subject}'s {relation} living?",
        )
    else:
        templates = (
            "Which city was {subject}'s {relation} born in?",
            "What is the birth city of {subject}'s {relation}?",
            "Where was the {relation} of {subject} born?",
            "Name the city where {subject}'s {relation} was born.",
            "In which city did the birth of {subject}'s {relation} take place?",
        )
    return templates[variant].format(subject=subject, relation=relation)


def prompt(row, variant=0, atomic=False):
    return "Fictional registry.\nQuestion: " + question(row, variant, atomic) + "\nAnswer:"


def record(tokenizer, row, variant, atomic):
    # Joint text tokenization, exactly as in the calibrated full-Qwen v2 recipe.
    ids = tokenizer.encode(
        prompt(row, variant, atomic) + " " + row["answer"] + ".", add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    return dict(
        ids=ids,
        labels=[-100, *ids[1:]],
        id=row["id"],
        kind="atomic" if atomic else "composition",
        variant=variant,
    )


def streams(world, cfg):
    rng = np.random.default_rng(cfg["training_seed"])
    atomic_order, comp_order = [], {a: [] for a in cfg["arms"]}
    batches = {a: [] for a in cfg["arms"]}
    for _ in range(cfg["steps"]):
        if len(atomic_order) < cfg["atomic_per_step"]:
            atomic_order.extend(rng.permutation(len(world["atoms"])).tolist())
        atomic = [
            [int(atomic_order.pop(0)), int(rng.integers(4))] for _ in range(cfg["atomic_per_step"])
        ]
        variants = rng.integers(4, size=cfg["compositions_per_step"]).tolist()
        for arm in cfg["arms"]:
            support = world[arm + "_support"]
            if len(comp_order[arm]) < cfg["compositions_per_step"]:
                comp_order[arm].extend(rng.permutation(support).tolist())
            comp = [[int(comp_order[arm].pop(0)), int(v)] for v in variants]
            batches[arm].append(dict(atomic=atomic, composition=comp))
    return batches


def records(tokenizer, world):
    return {
        kind: {
            (r["id"], v): record(tokenizer, r, v, kind == "atomic")
            for r in world["atoms" if kind == "atomic" else "chains"]
            for v in range(4)
        }
        for kind in ("atomic", "composition")
    }


def docs_for(rec, batch):
    return [rec[kind][tuple(spec)] for kind in ("atomic", "composition") for spec in batch[kind]]


def audit(world, batches, rec):
    assert (
        len(world["atoms"]) == 128
        and len(world["high_support"]) == 32
        and len(world["low_support"]) == 8
    )
    assert set(world["low_support"]) < set(world["high_support"])
    familiar_atoms = {a["id"] for a in world["atoms"] if a["pool"] == "familiar"}
    covered = {i for j in world["high_support"] for i in world["chains"][j]["atom_ids"]}
    assert covered == familiar_atoms
    strict_atoms = {a["id"] for a in world["atoms"] if a["pool"] == "strict"}
    assert not covered & strict_atoms
    support = [world["chains"][i] for i in world["high_support"]]
    heldout = [c for c in world["chains"] if c["split"] == "heldout"]
    strict = [c for c in world["chains"] if c["split"] == "strict"]
    assert len(heldout) == 32 and len(strict) == 64
    # No alternate trained query gives the same head/answer endpoint as a held-out chain.
    assert not {(c["head"], c["answer"]) for c in support} & {
        (c["head"], c["answer"]) for c in heldout
    }

    def people(pool):
        return {a["subject"] for a in world["atoms"] if a["pool"] == pool}

    assert not people("familiar") & people("strict")
    assert not {c["answer"] for c in support} & {c["answer"] for c in strict}
    budgets, exposure = {}, {}
    for arm, stream in batches.items():
        budget, atomic_counts, comp_counts = Counter(), Counter(), Counter()
        for batch in stream:
            docs = docs_for(rec, batch)
            width = max(len(d["ids"]) for d in docs)
            budget.update(
                input_tokens=sum(len(d["ids"]) for d in docs),
                supervised_tokens=sum(len(d["ids"]) - 1 for d in docs),
                padded_tokens=len(docs) * width,
                attention_cells=len(docs) * width**2,
            )
            atomic_counts.update(i for i, _ in batch["atomic"])
            comp_counts.update(i for i, _ in batch["composition"])
            assert set(comp_counts) <= set(world[arm + "_support"])
        budgets[arm] = dict(budget)
        exposure[arm] = dict(
            atomic=sorted(atomic_counts.items()), composition=sorted(comp_counts.items())
        )
    if "low" in batches and "high" in batches:
        assert [b["atomic"] for b in batches["low"]] == [b["atomic"] for b in batches["high"]]
        assert exposure["low"]["atomic"] == exposure["high"]["atomic"]
    return dict(
        status="passed",
        budgets=budgets,
        exposures=exposure,
        familiar_fact_role_coverage_high=len(covered),
        familiar_fact_role_coverage_low=len(
            {i for j in world["low_support"] for i in world["chains"][j]["atom_ids"]}
        ),
        heldout_queries=len(heldout),
        strict_queries=len(strict),
        target_endpoint_leakage=0,
        exact_common_atomic_sequence=True,
        exact_atomic_stream_loss_weight=True,
        composition_slots_equal=True,
    )


def freeze(config_path):
    cfg = base.read(config_path)
    art = Path(cfg["artifact_root"])
    assert not (art / "lock.json").exists(), "Preserve frozen batch."
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    sources = [
        str(config_path),
        "src/llm_memory_editability/natural_composition_confirmation_v2.py",
        "src/llm_memory_editability/small_lm_composition.py",
        "scripts/run_natural_composition_confirmation_v2.py",
        "tests/test_natural_composition_confirmation_v2.py",
        "docs/development-artifacts/natural-composition-confirmation-v1/amendment-v2.md",
    ]
    source_hashes = {}
    for source in sources:
        dest = art / "source" / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        source_hashes[source] = sha(source)
    data_files = {}
    for seed in cfg["world_seeds"]:
        world = build_world(seed)
        batch = streams(world, cfg)
        report = audit(world, batch, records(tokenizer, world))
        for filename, obj in (
            ("world.json", world),
            ("streams.json", batch),
            ("data-audit.json", report),
        ):
            path = art / f"w{seed}" / filename
            write(path, obj)
            data_files[str(path)] = sha(path)
    model_files = {
        str(p): sha(p)
        for p in Path(cfg["model_path"]).iterdir()
        if p.suffix in (".json", ".safetensors") or p.name in ("LICENSE", "README.md")
    }
    write(
        art / "environment.json",
        dict(
            python=platform.python_version(),
            executable=str(Path(__import__("sys").executable)),
            torch=torch.__version__,
            transformers=transformers.__version__,
            gpu_inventory=__import__("subprocess").check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"],
                text=True,
            ),
        ),
    )
    write(
        art / "lock.json",
        dict(
            status="locked_before_training",
            utc=base.now(),
            sources=source_hashes,
            data_files=data_files,
            model_files=model_files,
            config=cfg,
        ),
    )
    print(
        json.dumps(dict(status="frozen", worlds=cfg["world_seeds"], arms=cfg["arms"])), flush=True
    )


def check_lock(cfg):
    lock = base.read(Path(cfg["artifact_root"]) / "lock.json")
    for group in ("sources", "data_files", "model_files"):
        for path, digest in lock[group].items():
            assert sha(path) == digest, path


def evaluation_rows(world, extras=False):
    rows = []
    for view in (0, 4) if extras else (0,):
        for atom in world["atoms"]:
            rows.append(dict(atom, mode="atomic", view=view, prompt=prompt(atom, view, True)))
        for chain in world["chains"]:
            rows.append(dict(chain, mode="direct", view=view, prompt=prompt(chain, view)))
    return rows


@torch.inference_mode()
def generate(model, tokenizer, rows, cfg):
    model.eval()
    tokenizer.padding_side = "left"
    output = []
    for start in range(0, len(rows), cfg["eval_batch"]):
        chunk = rows[start : start + cfg["eval_batch"]]
        inputs = tokenizer(
            [r["prompt"] for r in chunk],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(cfg["device"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=cfg["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[:, inputs.input_ids.shape[1] :]
        for row, tok in zip(chunk, tokens.cpu().tolist(), strict=True):
            if tokenizer.eos_token_id in tok:
                tok = tok[: tok.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(tok, skip_special_tokens=True)
            output.append(dict(row, tokens=tok, text=text, correct=base.grade(text, row["answer"])))
    return output


def evaluate(model, tokenizer, world, cfg, extras=False):
    direct = generate(model, tokenizer, evaluation_rows(world, extras), cfg)
    if not extras:
        return direct
    atoms = {a["id"]: a for a in world["atoms"]}
    lookup = {(r["id"], r["view"]): r for r in direct if r["mode"] == "atomic"}
    second_calls = []
    for row in direct:
        if row["mode"] != "direct":
            continue
        first = lookup[row["atom_ids"][0], row["view"]]
        generated_bridge = first["text"].strip().split("\n")[0].strip().rstrip(".").strip()
        second = dict(atoms[row["atom_ids"][1]], subject=generated_bridge)
        second_calls.append(
            dict(
                row,
                mode="autonomous",
                prompt=prompt(second, row["view"], True),
                generated_bridge=generated_bridge,
                first_tokens=first["tokens"],
                first_correct=first["correct"],
            )
        )
    return direct + generate(model, tokenizer, second_calls, cfg)


def metrics(rows, world, arm):
    result = {}
    for view in sorted({r["view"] for r in rows}):
        atomic = {r["id"]: r for r in rows if r["view"] == view and r["mode"] == "atomic"}
        for pool in ("familiar", "strict"):
            for role in ("first", "second"):
                selected = [r for r in atomic.values() if r["pool"] == pool and r["role"] == role]
                result[f"v{view}_{pool}_{role}"] = dict(
                    correct=sum(r["correct"] for r in selected), n=len(selected)
                )
        for split in ("heldout", "strict", "training"):
            ids = (
                set(world[arm + "_support"])
                if split == "training"
                else {c["id"] for c in world["chains"] if c["split"] == split}
            )
            for mode in ("direct", "autonomous"):
                selected = [
                    r for r in rows if r["view"] == view and r["mode"] == mode and r["id"] in ids
                ]
                if selected:
                    result[f"v{view}_{split}_{mode}"] = dict(
                        correct=sum(r["correct"] for r in selected), n=len(selected)
                    )
                    mastered = [
                        r for r in selected if all(atomic[i]["correct"] for i in r["atom_ids"])
                    ]
                    result[f"v{view}_{split}_{mode}_both_atomic"] = dict(
                        correct=sum(r["correct"] for r in mastered),
                        n=len(mastered),
                        full_n=len(selected),
                    )
    return result


def stream_loss(model, inputs, atomic_count, atomic_weight):
    logits = model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False
    ).logits[:, :-1]
    labels = inputs["labels"][:, 1:]
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(labels)
    atomic = losses[:atomic_count].sum() / (labels[:atomic_count] != -100).sum()
    comp = losses[atomic_count:].sum() / (labels[atomic_count:] != -100).sum()
    return atomic_weight * atomic + (1 - atomic_weight) * comp


def run(config_path, seed, arm):
    cfg = base.read(config_path)
    check_lock(cfg)
    assert seed in cfg["world_seeds"] and arm in cfg["arms"]
    art = Path(cfg["artifact_root"]) / f"w{seed}"
    out = Path(cfg["output_root"]) / f"w{seed}-{arm}"
    assert not (out / "status.json").exists(), "Preserve existing execution."
    world, batches = base.read(art / "world.json"), base.read(art / "streams.json")[arm]
    write(out / "status.json", dict(status="running", utc=base.now()))
    started = time.perf_counter()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["training_seed"])
    torch.cuda.set_device(cfg["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def load(path):
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        ).to(cfg["device"])

    model = load(cfg["model_path"])
    assert model.config.num_hidden_layers == 28 and model.config.hidden_size == 1024
    assert len({id(layer.mlp) for layer in model.model.layers}) == 28
    model.requires_grad_(True)
    samples = base.parameter_samples(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    rec = records(tokenizer, world)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], fused=True
    )
    curve, budget, train_seconds = [], Counter(), 0.0
    for step in range(cfg["steps"] + 1):
        if step in cfg["nodes"]:
            rows = evaluate(model, tokenizer, world, cfg, extras=step == cfg["steps"])
            write(out / f"predictions-{step:05d}.json", rows)
            point = dict(step=step, metrics=metrics(rows, world, arm))
            curve.append(point)
            write(out / "curve.json", curve)
            print(json.dumps(dict(seed=seed, arm=arm, **point)), flush=True)
        if step == cfg["steps"]:
            break
        decay = 0.5 * (
            1
            + math.cos(
                math.pi * max(0, step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"])
            )
        )
        lr = (
            cfg["minimum_learning_rate"]
            + (cfg["learning_rate"] - cfg["minimum_learning_rate"]) * decay
        ) * min(1.0, (step + 1) / cfg["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        docs = docs_for(rec, batches[step])
        width = max(len(d["ids"]) for d in docs)
        budget.update(
            input_tokens=sum(len(d["ids"]) for d in docs),
            supervised_tokens=sum(len(d["ids"]) - 1 for d in docs),
            padded_tokens=len(docs) * width,
            attention_cells=len(docs) * width**2,
        )
        inputs = base.batch_tensors(docs, tokenizer.pad_token_id, cfg["device"])
        model.train()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = stream_loss(model, inputs, cfg["atomic_per_step"], cfg["atomic_loss_weight"])
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            assert all(
                p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
        optimizer.step()
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - tick
        if (step + 1) % 64 == 0:
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        arm=arm,
                        step=step + 1,
                        loss=float(loss.detach()),
                        train_seconds=train_seconds,
                    )
                ),
                flush=True,
            )
    changed = {
        name: not torch.equal(before, base.parameter_samples(model)[name])
        for name, before in samples.items()
    }
    assert all(changed.values())
    model.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    architecture = model.config.to_dict()
    final_rows = rows
    del optimizer, model, samples, inputs, loss
    torch.cuda.empty_cache()
    again = evaluate(load(out / "model"), tokenizer, world, cfg, extras=True)
    assert [r["tokens"] for r in final_rows] == [r["tokens"] for r in again]
    assert [r["prompt"] for r in final_rows] == [r["prompt"] for r in again]
    write(
        out / "reload-audit.json",
        dict(
            status="passed",
            predictions=len(again),
            exact_generated_tokens=True,
            exact_autonomous_prompts=True,
        ),
    )
    assert dict(budget) == base.read(art / "data-audit.json")["budgets"][arm]
    d, layers = architecture["hidden_size"], architecture["num_hidden_layers"]
    query_width = architecture["num_attention_heads"] * architecture["head_dim"]
    kv = architecture["num_key_value_heads"] * architecture["head_dim"]
    per_token = (
        layers * (2 * d * query_width + 2 * d * kv + 3 * d * architecture["intermediate_size"])
        + d * architecture["vocab_size"]
    )
    flops = (
        6 * per_token * budget["padded_tokens"]
        + 12 * layers * query_width * budget["attention_cells"]
    )
    summary = dict(
        status="complete",
        phase=cfg["phase"],
        utc=base.now(),
        world_seed=seed,
        arm=arm,
        metrics=metrics(final_rows, world, arm),
        updates=cfg["steps"],
        budget=dict(budget),
        train_seconds=train_seconds,
        process_seconds=time.perf_counter() - started,
        estimated_training_matrix_flops=flops,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        parameter_count=parameter_count,
        sampled_parameters_changed=changed,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        tf32=False,
        lock_sha256=sha(Path(cfg["artifact_root"]) / "lock.json"),
        checkpoint_files={str(p): sha(p) for p in (out / "model").glob("*.safetensors")},
    )
    write(out / "summary.json", summary)
    write(out / "status.json", dict(status="complete", utc=base.now()))
    print(json.dumps(summary), flush=True)


def report(config_path):
    cfg = base.read(config_path)
    check_lock(cfg)
    runs, checks = {}, 0
    for seed in cfg["world_seeds"]:
        world = base.read(Path(cfg["artifact_root"]) / f"w{seed}" / "world.json")
        for arm in cfg["arms"]:
            out = Path(cfg["output_root"]) / f"w{seed}-{arm}"
            summary = base.read(out / "summary.json")
            assert base.read(out / "reload-audit.json")["status"] == "passed"
            for point in base.read(out / "curve.json"):
                rows = base.read(out / f"predictions-{point['step']:05d}.json")
                assert all(base.grade(r["text"], r["answer"]) == r["correct"] for r in rows)
                assert point["metrics"] == metrics(rows, world, arm)
                checks += len(rows)
            assert summary["metrics"] == metrics(rows, world, arm)
            runs[f"w{seed}-{arm}"] = summary
    differences = {}
    if len(cfg["arms"]) == 2:
        for seed in cfg["world_seeds"]:
            diff = {}
            for key in runs[f"w{seed}-high"]["metrics"]:
                high, low = [runs[f"w{seed}-{a}"]["metrics"][key] for a in ("high", "low")]
                if key.endswith("both_atomic") or "training" in key:
                    continue
                diff[key] = high["correct"] / high["n"] - low["correct"] / low["n"]
            differences[str(seed)] = diff
    write(
        Path(cfg["artifact_root"]) / "summary.json",
        dict(
            status="complete",
            phase=cfg["phase"],
            independent_worlds=len(cfg["world_seeds"]),
            training_initializations=1,
            runs=runs,
            per_world_high_minus_low=differences,
            independent_node_prediction_checks=checks,
            total_updates=sum(r["updates"] for r in runs.values()),
            total_train_seconds=sum(r["train_seconds"] for r in runs.values()),
            total_process_seconds=sum(r["process_seconds"] for r in runs.values()),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "report"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--world", type=int)
    parser.add_argument("--arm", choices=("low", "high"))
    args = parser.parse_args()
    if args.action == "run":
        run(args.config, args.world, args.arm)
    elif args.action == "freeze":
        freeze(args.config)
    else:
        report(args.config)
