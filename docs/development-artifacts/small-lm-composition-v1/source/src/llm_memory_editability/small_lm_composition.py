"""Development comparison in an unchanged, fully trained pretrained Qwen LM.

Only training document grouping differs. All weights, native embeddings,
residuals, attention, norms and distinct MLP layers are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = (
    (
        "Who is {head}'s mentor?",
        "Who mentors {head}?",
        "Name the mentor of {head}.",
        "Who is the mentor of {head}?",
    ),
    (
        "Which city does {bridge} live in?",
        "What city is {bridge}'s home?",
        "Where does {bridge} live?",
        "Name the city where {bridge} lives.",
    ),
    (
        "Which city does {head}'s mentor live in?",
        "What city is home to {head}'s mentor?",
        "Where does the mentor of {head} live?",
        "Name the city where {head}'s mentor lives.",
    ),
)
CITIES = (
    "Paris Rome London Berlin Tokyo Oslo Lima Cairo Delhi Seoul Dublin Madrid "
    "Vienna Boston Sydney Lisbon"
).split()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def build_world(config):
    first = (
        "Rhea Mara Evan Luca Nora Iris Owen Hugo Cora Theo Nina Leon Vera Finn Dana Arlo"
    ).split()
    last = "Hale Vale Mercer Rowan Keane Flint Arden Voss North Lark".split()
    names = [f"{a} {b}" for a in first for b in last]
    n = config["target_triples"] + config["background_triples"]
    assert 2 * n <= len(names)
    rng = np.random.default_rng(config["world_seed"])
    names = rng.permutation(names).tolist()
    rows = []
    for split, count in (
        ("target", config["target_triples"]),
        ("background", config["background_triples"]),
    ):
        cities = rng.permutation([CITIES[i % len(CITIES)] for i in range(count)])
        for city in cities:
            i = len(rows)
            rows.append(
                dict(id=i, split=split, head=names[2 * i], bridge=names[2 * i + 1], city=str(city))
            )
    return rows


def prompt(row, hop, variant=0):
    return "Fictional registry.\nQuestion: " + QUESTIONS[hop][variant].format(**row) + "\nAnswer:"


def answer(row, hop):
    return row["bridge"] if hop == 0 else row["city"]


def encode_document(tokenizer, row, hop, variant):
    ids = tokenizer.encode(
        prompt(row, hop, variant) + " " + answer(row, hop) + ".", add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    # The initial token of each atomic document has no within-document predecessor.
    # It remains unscored when documents are joined, preserving every target token.
    return dict(ids=ids, labels=[-100, *ids[1:]], triple=row["id"], hop=hop, variant=variant)


def corpus(tokenizer, world):
    records = {}
    for row in world:
        hops = range(3) if row["split"] == "background" else range(2)
        for hop in hops:
            for variant in range(4):
                records[row["id"], hop, variant] = encode_document(tokenizer, row, hop, variant)
    return records


def stream(config, world):
    rng = np.random.default_rng(config["training_seed"])
    background = [r["id"] for r in world if r["split"] == "background"]
    result = []
    for _ in range(config["steps"]):
        pairs = [
            [int(i), int(v)]
            for i, v in zip(
                rng.choice(len(world), config["atomic_pairs_per_step"], replace=False),
                rng.integers(4, size=config["atomic_pairs_per_step"]),
                strict=True,
            )
        ]
        compositions = [
            [int(i), int(v)]
            for i, v in zip(
                rng.choice(background, config["background_compositions_per_step"], replace=False),
                rng.integers(4, size=config["background_compositions_per_step"]),
                strict=True,
            )
        ]
        result.append(dict(pairs=pairs, compositions=compositions))
    return result


def documents(records, batch, arm):
    docs = []
    for i, v in batch["pairs"]:
        first, second = records[i, 0, v], records[i, 1, v]
        if arm == "separate":
            docs.extend([first, second])
        elif arm == "linked":
            docs.append(
                dict(ids=first["ids"] + second["ids"], labels=first["labels"] + second["labels"])
            )
        else:
            raise ValueError(arm)
    docs.extend(records[i, 2, v] for i, v in batch["compositions"])
    return docs


def batch_tensors(docs, pad_id, device):
    width = max(len(d["ids"]) for d in docs)
    ids = torch.full((len(docs), width), pad_id, device=device, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, doc in enumerate(docs):
        n = len(doc["ids"])
        ids[i, :n] = torch.tensor(doc["ids"], device=device)
        labels[i, :n] = torch.tensor(doc["labels"], device=device)
        mask[i, :n] = 1
    return dict(input_ids=ids, attention_mask=mask, labels=labels)


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().rstrip(".").strip()).casefold()


def grade(text, gold):
    first_line = text.strip().split("\n")[0]
    return normalize(first_line) == normalize(gold)


def evaluation_rows(world):
    return [
        dict(**row, hop=hop, prompt=prompt(row, hop), answer=answer(row, hop))
        for row in world
        for hop in (range(3) if row["split"] == "target" else [2])
    ]


@torch.inference_mode()
def evaluate(model, tokenizer, world, config):
    model.eval()
    tokenizer.padding_side = "left"
    rows = evaluation_rows(world)
    results = []
    for start in range(0, len(rows), config["eval_batch"]):
        chunk = rows[start : start + config["eval_batch"]]
        enc = tokenizer(
            [r["prompt"] for r in chunk],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(config["device"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[:, enc.input_ids.shape[1] :]
        for row, tokens in zip(chunk, generated.cpu().tolist(), strict=True):
            # Drop batching padding after the first EOS; retain the actual generated EOS.
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[: tokens.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(
                dict(**row, tokens=tokens, text=text, correct=grade(text, row["answer"]))
            )
    return results


def score(rows):
    result = {}
    for split, hop, key in (
        ("target", 0, "first_hop"),
        ("target", 1, "second_hop"),
        ("target", 2, "two_hop"),
        ("background", 2, "background_two_hop"),
    ):
        selected = [r for r in rows if r["split"] == split and r["hop"] == hop]
        result[key] = sum(r["correct"] for r in selected) / len(selected)
    indexed = {(r["id"], r["hop"]): r for r in rows if r["split"] == "target"}
    mastered = [
        i for i, h in indexed if h == 0 and indexed[i, 0]["correct"] and indexed[i, 1]["correct"]
    ]
    result["both_atomic_coverage"] = len(mastered) / sum(h == 0 for _, h in indexed)
    result["two_hop_if_both_atomic"] = (
        sum(indexed[i, 2]["correct"] for i in mastered) / len(mastered) if mastered else None
    )
    return result


def prepare(config_path):
    config = read(config_path)
    art = Path(config["artifact_root"])
    if (art / "lock.json").exists():
        raise RuntimeError("Development lock already exists; do not overwrite it.")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    world = build_world(config)
    batches = stream(config, world)
    records = corpus(tokenizer, world)
    assert all(h != 2 or world[i]["split"] == "background" for i, h, _ in records)
    stats = {}
    for arm in config["arms"]:
        counts = Counter()
        token_counts = Counter()
        input_tokens = supervised_tokens = padded_tokens = attention_cells = 0
        for batch in batches:
            docs = documents(records, batch, arm)
            for d in docs:
                input_tokens += len(d["ids"])
                supervised_tokens += sum(t != -100 for t in d["labels"])
                counts.update(t for t in d["labels"] if t != -100)
                token_counts.update(d["ids"])
            width = max(len(d["ids"]) for d in docs)
            padded_tokens += len(docs) * width
            attention_cells += len(docs) * width**2
        stats[arm] = dict(
            input_tokens=input_tokens,
            supervised_tokens=supervised_tokens,
            padded_tokens=padded_tokens,
            attention_cells=attention_cells,
            target_token_counts=sorted(counts.items()),
            input_token_counts=sorted(token_counts.items()),
        )
    for key in ("input_tokens", "supervised_tokens", "target_token_counts", "input_token_counts"):
        assert stats["separate"][key] == stats["linked"][key], key
    write(art / "world.json", world)
    write(art / "stream.json", batches)
    write(art / "data-audit.json", dict(status="passed", stats=stats, target_composition_labels=0))
    sources = [
        str(config_path),
        __file__,
        "scripts/run_small_lm_composition.py",
        "tests/test_small_lm_composition.py",
        str(art / "development-plan.md"),
    ]
    frozen = {}
    for source in sources:
        path = Path(source).resolve()
        relative = path.relative_to(Path.cwd())
        dest = art / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        frozen[str(relative)] = sha(path)
    model_files = {
        str(p): sha(p)
        for p in Path(config["model_path"]).iterdir()
        if p.suffix in (".safetensors", ".json")
    }
    write(
        art / "lock.json",
        dict(
            status="locked_before_training",
            utc=now(),
            sources=frozen,
            model_files=model_files,
            world_sha256=sha(art / "world.json"),
            stream_sha256=sha(art / "stream.json"),
        ),
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "stats": {
                    a: {k: v for k, v in s.items() if not k.endswith("counts")}
                    for a, s in stats.items()
                },
            }
        ),
        flush=True,
    )


def parameter_samples(model):
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.27.self_attn.o_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
    ]
    params = dict(model.named_parameters())
    return {n: params[n].detach().flatten()[:4096].clone() for n in names}


def train(config_path, arm):
    config = read(config_path)
    assert arm in config["arms"]
    art = Path(config["artifact_root"])
    lock = read(art / "lock.json")
    for path, expected in lock["sources"].items():
        assert sha(path) == expected, path
    for path, expected in lock["model_files"].items():
        assert sha(path) == expected, path
    assert sha(art / "world.json") == lock["world_sha256"]
    assert sha(art / "stream.json") == lock["stream_sha256"]
    out = Path(config["output_root"]) / arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise RuntimeError("A run record already exists; preserve it and use a new output root.")
    write(out / "status.json", dict(status="running", arm=arm, utc=now()))
    begun = time.perf_counter()
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["training_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(config["device"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(config["device"])
    model.requires_grad_(True)
    assert all(p.requires_grad for p in model.parameters())
    assert model.config.num_hidden_layers == 28 and model.config.hidden_size == 1024
    assert len({id(layer.mlp) for layer in model.model.layers}) == 28
    samples = parameter_samples(model)
    world, batches = read(art / "world.json"), read(art / "stream.json")
    records = corpus(tokenizer, world)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True,
    )
    curve = []
    token_count = 0
    train_seconds = 0.0
    initial_gradients = {}
    for step in range(config["steps"] + 1):
        if step in config["nodes"]:
            rows = evaluate(model, tokenizer, world, config)
            write(out / f"predictions-{step:05d}.json", rows)
            record = dict(step=step, **score(rows))
            curve.append(record)
            write(out / "curve.json", curve)
            print(json.dumps(dict(arm=arm, **record)), flush=True)
        if step == config["steps"]:
            break
        rate = config["minimum_learning_rate"] + 0.5 * (
            config["learning_rate"] - config["minimum_learning_rate"]
        ) * (
            1
            + math.cos(
                math.pi
                * max(0, step - config["warmup_steps"])
                / (config["steps"] - config["warmup_steps"])
            )
        )
        rate *= min(1.0, (step + 1) / config["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        docs = documents(records, batches[step], arm)
        inputs = batch_tensors(docs, tokenizer.pad_token_id, config["device"])
        token_count += int(inputs["labels"].ne(-100).sum())
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**inputs, use_cache=False).loss
        assert torch.isfinite(loss), float(loss)
        loss.backward()
        if step == 0:
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                if name in samples:
                    initial_gradients[name] = float(p.grad.norm())
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - tick
        if (step + 1) % 32 == 0:
            print(
                json.dumps(
                    dict(
                        arm=arm,
                        step=step + 1,
                        loss=float(loss),
                        learning_rate=rate,
                        train_seconds=train_seconds,
                    )
                ),
                flush=True,
            )
    changed = {
        n: bool(not torch.equal(old, parameter_samples(model)[n])) for n, old in samples.items()
    }
    assert all(changed.values()), changed
    parameter_count = sum(p.numel() for p in model.parameters())
    model.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    final_rows = rows
    del optimizer, model, samples, inputs, loss
    torch.cuda.empty_cache()
    reloaded = AutoModelForCausalLM.from_pretrained(
        out / "model",
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(config["device"])
    reloaded_rows = evaluate(reloaded, tokenizer, world, config)
    assert [r["tokens"] for r in final_rows] == [r["tokens"] for r in reloaded_rows]
    write(
        out / "reload-audit.json",
        dict(status="passed", predictions=len(final_rows), exact_generated_tokens=True),
    )
    budget = read(art / "data-audit.json")["stats"][arm]
    assert token_count == budget["supervised_tokens"]
    summary = dict(
        status="complete",
        phase="development",
        arm=arm,
        utc=now(),
        metrics=score(final_rows),
        model_repo=config["model_repo"],
        model_revision=config["model_revision"],
        lock_sha256=sha(art / "lock.json"),
        parameter_count=parameter_count,
        trainable_parameters=parameter_count,
        sampled_parameters_changed=changed,
        initial_sampled_gradient_norms=initial_gradients,
        budget={k: v for k, v in budget.items() if not k.endswith("counts")},
        updates=config["steps"],
        train_seconds=train_seconds,
        process_seconds=time.perf_counter() - begun,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        train_weight_dtype="float32",
        forward_dtype="bfloat16 autocast",
        tf32=False,
        checkpoint_files={str(p): sha(p) for p in (out / "model").glob("*.safetensors")},
    )
    write(out / "summary.json", summary)
    write(out / "status.json", dict(status="complete", utc=now()))
    print(json.dumps(summary), flush=True)


def report(config_path):
    config = read(config_path)
    art = Path(config["artifact_root"])
    runs, predictions = {}, {}
    for arm in config["arms"]:
        out = Path(config["output_root"]) / arm
        runs[arm] = read(out / "summary.json")
        assert runs[arm]["status"] == "complete"
        assert read(out / "reload-audit.json")["status"] == "passed"
        rows = read(out / f"predictions-{config['steps']:05d}.json")
        assert score(rows) == runs[arm]["metrics"]
        predictions[arm] = {(r["id"], r["hop"]): r for r in rows if r["split"] == "target"}
    common = [
        i
        for i, h in predictions["separate"]
        if h == 0
        and all(predictions[a][i, hop]["correct"] for a in config["arms"] for hop in (0, 1))
    ]
    conditional = {
        a: sum(predictions[a][i, 2]["correct"] for i in common) / len(common) if common else None
        for a in config["arms"]
    }
    result = dict(
        status="complete",
        phase="development_only",
        utc=now(),
        runs=runs,
        common_atomic_n=len(common),
        common_atomic_coverage=len(common) / config["target_triples"],
        common_atomic_two_hop=conditional,
        two_hop_difference=runs["linked"]["metrics"]["two_hop"]
        - runs["separate"]["metrics"]["two_hop"],
        notes=[
            "One synthetic world and one training seed; no independent confirmation.",
            "Natural-language fictional facts; original pretrained Qwen architecture.",
            "Exact fact, input-token and supervised-token exposure match.",
            "Document grouping changes context, positions and attention work.",
            "Linked training co-presents the head and final city; not zero cooccurrence.",
            "This behavior comparison does not uniquely identify an MLP mechanism.",
        ],
    )
    write(art / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}), flush=True)
