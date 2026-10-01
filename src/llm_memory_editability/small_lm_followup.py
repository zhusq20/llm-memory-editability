"""Small full-Qwen development: matched rehearsal, pairing, and factual updates."""

from __future__ import annotations

import copy
import itertools
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import small_lm_composition as base

HELDOUT = (
    "Give the name of the person mentoring {head}.",
    "In which city is {bridge} living?",
    "In which city is the person mentoring {head} living?",
)


def pairing_stream(config, world):
    batches = base.stream(config, world)
    rng = np.random.default_rng(config["pairing_seed"])
    for batch in batches:
        pairs = batch["pairs"]

        def valid(order, pairs=pairs):
            return all(
                world[i]["city"] != world[pairs[k][0]]["city"]
                for (i, _), k in zip(pairs, order, strict=True)
            )

        candidates = (rng.permutation(len(pairs)).tolist() for _ in range(1000))
        order = next((p for p in candidates if valid(p)), None)
        if order is None:
            order = next(
                (list(p) for p in itertools.permutations(range(len(pairs))) if valid(p)), None
            )
        if order is None:
            raise ValueError("Batch cannot match distinct cities; revise before training.")
        batch["second_order"] = order
    return batches


def training_documents(records, batch, arm, copies=1):
    pairs = batch["pairs"]
    singles = [records[i, h, v] for i, v in pairs for h in (0, 1)]
    docs = []
    if arm == "separate":
        docs.extend(singles)
    else:
        order = range(len(pairs)) if arm == "linked" else batch["second_order"]
        if arm not in ("linked", "shuffled"):
            raise ValueError(arm)
        for (i, v), j in zip(pairs, order, strict=True):
            k, w = pairs[j]
            first, second = records[i, 0, v], records[k, 1, w]
            docs.append(
                dict(ids=first["ids"] + second["ids"], labels=first["labels"] + second["labels"])
            )
    docs.extend(singles * copies)
    docs.extend(records[i, 2, v] for i, v in batch["compositions"])
    return docs


def edit_design(config, world):
    rng = np.random.default_rng(config["edit_seed"])
    ids = []
    for city in base.CITIES:
        options = [r["id"] for r in world if r["split"] == "target" and r["city"] == city]
        ids.extend(int(i) for i in rng.choice(options, len(options) // 2, replace=False))
    cities = rng.permutation(base.CITIES).tolist()
    shift = int(rng.integers(1, len(cities)))
    mapping = dict(zip(cities, cities[shift:] + cities[:shift], strict=True))
    changed = copy.deepcopy(world)
    for row in changed:
        if row["id"] in ids:
            row["city"] = mapping[row["city"]]
    rng = np.random.default_rng(config["edit_training_seed"])
    background = [r["id"] for r in world if r["split"] == "background"]
    batches = []
    for _ in range(config["edit_steps"]):
        specs = []
        for pool, count, hop in (
            (ids, config["edited_second_per_step"], 1),
            (ids, config["replayed_first_per_step"], 0),
            (background, config["background_atomic_per_step"], None),
            (background, config["background_compositions_edit_per_step"], 2),
        ):
            for i in rng.choice(pool, count, replace=False):
                h = int(rng.integers(2)) if hop is None else hop
                specs.append([int(i), h, int(rng.integers(4))])
        batches.append(specs)
    return dict(edited_ids=sorted(ids), city_mapping=mapping, world=changed, stream=batches)


def view_prompt(row, hop, view):
    if view == 0:
        return base.prompt(row, hop)
    return "Fictional registry.\nQuestion: " + HELDOUT[hop].format(**row) + "\nAnswer:"


def evaluation_rows(world, extras=False):
    return [
        dict(
            **row,
            hop=hop,
            view=view,
            mode="direct",
            prompt=view_prompt(row, hop, view),
            answer=base.answer(row, hop),
        )
        for view in ((0, 1) if extras else (0,))
        for row in world
        if view == 0 or row["split"] == "target"
        for hop in range(3)
    ]


def autonomous_rows(world, direct):
    lookup = {r["id"]: r for r in world}
    result = []
    for first in direct:
        if first["split"] != "target" or first["hop"] != 0:
            continue
        bridge = first["text"].strip().split("\n")[0].strip().rstrip(".").strip()
        row = lookup[first["id"]]
        substituted = dict(row, bridge=bridge)
        result.append(
            dict(
                **row,
                hop=2,
                view=first["view"],
                mode="autonomous",
                prompt=view_prompt(substituted, 1, first["view"]),
                answer=row["city"],
                generated_bridge=bridge,
                first_tokens=first["tokens"],
                first_text=first["text"],
                first_correct=first["correct"],
            )
        )
    return result


@torch.inference_mode()
def generate(model, tokenizer, rows, config):
    model.eval()
    tokenizer.padding_side = "left"
    result = []
    for start in range(0, len(rows), config["eval_batch"]):
        chunk = rows[start : start + config["eval_batch"]]
        x = tokenizer(
            [r["prompt"] for r in chunk],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(config["device"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = model.generate(
                **x,
                do_sample=False,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[:, x.input_ids.shape[1] :]
        for row, tok in zip(chunk, tokens.cpu().tolist(), strict=True):
            if tokenizer.eos_token_id in tok:
                tok = tok[: tok.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(tok, skip_special_tokens=True)
            result.append(dict(row, tokens=tok, text=text, correct=base.grade(text, row["answer"])))
    return result


def evaluate(model, tokenizer, world, config, extras=False):
    direct = generate(model, tokenizer, evaluation_rows(world, extras), config)
    if not extras:
        return direct
    return direct + generate(model, tokenizer, autonomous_rows(world, direct), config)


def mean_correct(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else None


def metrics(rows, old_world, edit):
    primary = [r for r in rows if r["view"] == 0 and r["mode"] == "direct"]
    result = base.score(primary)
    result["background_atomic"] = mean_correct(
        [r for r in primary if r["split"] == "background" and r["hop"] < 2]
    )
    edited = set(edit["edited_ids"])
    old = {r["id"]: r for r in old_world}
    for view in (0, 1):
        subset = [r for r in rows if r["view"] == view]
        if not subset:
            continue
        for pool, keep in (("E", True), ("U", False)):
            target = [r for r in subset if r["split"] == "target" and (r["id"] in edited) == keep]
            for hop, name in ((0, "first"), (1, "second"), (2, "two")):
                cell = [r for r in target if r["hop"] == hop and r["mode"] == "direct"]
                result[f"v{view}_{pool}_{name}"] = mean_correct(cell)
                if hop == 2 and cell:
                    result[f"v{view}_{pool}_old_two"] = sum(
                        base.grade(r["text"], old[r["id"]]["city"]) for r in cell
                    ) / len(cell)
            result[f"v{view}_{pool}_autonomous"] = mean_correct(
                [r for r in target if r["mode"] == "autonomous"]
            )
        if view == 1:
            for hop, name in ((0, "first"), (1, "second"), (2, "two")):
                result[f"heldout_{name}"] = mean_correct(
                    [r for r in subset if r["mode"] == "direct" and r["hop"] == hop]
                )
    return result


def document_budget(docs):
    width = max(len(d["ids"]) for d in docs)
    return dict(
        input_tokens=sum(len(d["ids"]) for d in docs),
        supervised_tokens=sum(sum(t != -100 for t in d["labels"]) for d in docs),
        padded_tokens=len(docs) * width,
        attention_cells=len(docs) * width**2,
    )


def prepare(config_path):
    config = base.read(config_path)
    art = Path(config["artifact_root"])
    if (art / "lock.json").exists():
        raise RuntimeError("Preserve existing lock.")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    world = base.build_world(config)
    batches = pairing_stream(config, world)
    edit = edit_design(config, world)
    records = base.corpus(tokenizer, world)
    stats = {arm: Counter() for arm in config["arms"]}
    for batch in batches:
        signatures = []
        for arm in config["arms"]:
            docs = training_documents(records, batch, arm, config["standalone_rehearsal_copies"])
            stats[arm].update(document_budget(docs))
            signatures.append(
                (
                    Counter(t for d in docs for t in d["ids"]),
                    Counter(t for d in docs for t in d["labels"] if t != -100),
                )
            )
        assert all(s == signatures[0] for s in signatures)
    changed = base.corpus(tokenizer, edit["world"])
    edit_stats = {branch: Counter() for branch in ("update", "replay")}
    for specs in edit["stream"]:
        for branch, rec in (("update", changed), ("replay", records)):
            docs = [rec[tuple(s)] for s in specs]
            edit_stats[branch].update(document_budget(docs))
        assert [len(changed[tuple(s)]["ids"]) for s in specs] == [
            len(records[tuple(s)]["ids"]) for s in specs
        ]
    assert edit_stats["update"] == edit_stats["replay"]
    base.write(art / "world.json", world)
    base.write(art / "stream.json", batches)
    base.write(art / "edit-design.json", edit)
    base.write(
        art / "data-audit.json",
        dict(
            status="passed",
            training=stats,
            editing=edit_stats,
            paired_batch_checks=len(batches),
            equal_edit_length_checks=len(edit["stream"]),
            target_composition_labels=0,
            edit_target_composition_labels=0,
        ),
    )
    sources = [
        str(config_path),
        str(Path(__file__).relative_to(Path.cwd())) if Path(__file__).is_absolute() else __file__,
        "src/llm_memory_editability/small_lm_composition.py",
        "scripts/run_small_lm_followup.py",
        "tests/test_small_lm_followup.py",
        str(art / "development-plan.md"),
    ]
    hashes = {}
    for source in sources:
        relative = Path(source).resolve().relative_to(Path.cwd())
        dest = art / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, dest)
        hashes[str(relative)] = base.sha(relative)
    model_files = {
        str(p): base.sha(p)
        for p in Path(config["model_path"]).iterdir()
        if p.suffix in (".safetensors", ".json")
    }
    base.write(
        art / "lock.json",
        dict(
            status="locked_before_training",
            utc=base.now(),
            sources=hashes,
            model_files=model_files,
            data_files={
                str(art / f): base.sha(art / f)
                for f in ("world.json", "stream.json", "edit-design.json", "data-audit.json")
            },
        ),
    )
    print(json.dumps(dict(status="prepared", training=stats, editing=edit_stats)), flush=True)


def check_lock(config):
    art = Path(config["artifact_root"])
    lock = base.read(art / "lock.json")
    for group in ("sources", "model_files", "data_files"):
        for path, digest in lock[group].items():
            assert base.sha(path) == digest, path


def learning_rate(step, steps, lr, minimum, warmup):
    decay = 0.5 * (1 + math.cos(math.pi * max(0, step - warmup) / (steps - warmup)))
    return (minimum + (lr - minimum) * decay) * min(1.0, (step + 1) / warmup)


def run(config_path, arm, branch=None):
    config = base.read(config_path)
    check_lock(config)
    assert arm in config["arms"] and branch in (None, "update", "replay")
    art = Path(config["artifact_root"])
    root = Path(config["output_root"])
    out = root / ("training" if branch is None else "editing") / arm
    if branch:
        out = out / branch
    if (out / "status.json").exists():
        raise RuntimeError(f"Preserve existing run: {out}")
    old_world = base.read(art / "world.json")
    edit = base.read(art / "edit-design.json")
    # Both update and replay are scored against the SAME edited world.
    world = edit["world"] if branch else old_world
    source = Path(config["model_path"]) if branch is None else root / "training" / arm / "model"
    if branch:
        parent = base.read(root / "training" / arm / "summary.json")
        for path, digest in parent["checkpoint_files"].items():
            assert base.sha(path) == digest
    steps = config["edit_steps"] if branch else config["steps"]
    nodes = config["edit_nodes"] if branch else config["nodes"]
    seed = config["edit_training_seed"] if branch else config["training_seed"]
    prefix = "edit_" if branch else ""
    batches = edit["stream"] if branch else base.read(art / "stream.json")
    base.write(out / "status.json", dict(status="running", utc=base.now(), arm=arm, branch=branch))
    started = time.perf_counter()
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(config["device"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def load_model(path):
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
        ).to(config["device"])

    model = load_model(source)
    model.requires_grad_(True)
    assert model.config.num_hidden_layers == 28 and model.config.hidden_size == 1024
    assert len({id(layer.mlp) for layer in model.model.layers}) == 28
    samples = base.parameter_samples(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    records = base.corpus(tokenizer, edit["world"] if branch == "update" else old_world)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config[prefix + "learning_rate"],
        weight_decay=config["weight_decay"],
        fused=True,
    )
    curve, budget = [], Counter()
    train_seconds = 0.0
    for step in range(steps + 1):
        if step in nodes:
            rows = evaluate(model, tokenizer, world, config, extras=step == steps)
            base.write(out / f"predictions-{step:05d}.json", rows)
            record = dict(step=step, **metrics(rows, old_world, edit))
            curve.append(record)
            base.write(out / "curve.json", curve)
            print(json.dumps(dict(arm=arm, branch=branch, **record)), flush=True)
        if step == steps:
            break
        rate = learning_rate(
            step,
            steps,
            config[prefix + "learning_rate"],
            config[prefix + "minimum_learning_rate"],
            config[prefix + "warmup_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        docs = (
            [records[tuple(s)] for s in batches[step]]
            if branch
            else training_documents(
                records, batches[step], arm, config["standalone_rehearsal_copies"]
            )
        )
        budget.update(document_budget(docs))
        inputs = base.batch_tensors(docs, tokenizer.pad_token_id, config["device"])
        model.train()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**inputs, use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            assert all(
                p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - tick
        if (step + 1) % 64 == 0:
            print(
                json.dumps(
                    dict(
                        arm=arm,
                        branch=branch,
                        step=step + 1,
                        loss=float(loss.detach()),
                        train_seconds=train_seconds,
                    )
                ),
                flush=True,
            )
    changed = {
        n: not torch.equal(before, base.parameter_samples(model)[n])
        for n, before in samples.items()
    }
    assert all(changed.values())
    architecture = model.config.to_dict()
    model.save_pretrained(out / "model", safe_serialization=True)
    tokenizer.save_pretrained(out / "model")
    final_rows = rows
    del optimizer, model, samples, inputs, loss
    torch.cuda.empty_cache()
    reloaded = load_model(out / "model")
    again = evaluate(reloaded, tokenizer, world, config, extras=True)
    assert [r["tokens"] for r in final_rows] == [r["tokens"] for r in again]
    assert [r["prompt"] for r in final_rows] == [r["prompt"] for r in again]
    base.write(
        out / "reload-audit.json",
        dict(
            status="passed",
            predictions=len(again),
            exact_generated_tokens=True,
            exact_autonomous_prompts=True,
        ),
    )
    expected = base.read(art / "data-audit.json")
    assert budget == expected["editing" if branch else "training"][branch or arm]
    # Dense matrix estimate: 3x forward for training; all vocab logits are evaluated.
    d, layers, intermediate = 1024, 28, architecture["intermediate_size"]
    kv = architecture["num_key_value_heads"] * architecture["head_dim"]
    query_width = architecture["num_attention_heads"] * architecture["head_dim"]
    attention_projections = 2 * d * query_width + 2 * d * kv
    per_token = layers * (attention_projections + 3 * d * intermediate)
    per_token += d * architecture["vocab_size"]
    flops = (
        6 * per_token * budget["padded_tokens"]
        + 12
        * layers
        * (architecture["num_attention_heads"] * architecture["head_dim"])
        * budget["attention_cells"]
    )
    summary = dict(
        status="complete",
        phase="development_only",
        arm=arm,
        branch=branch,
        utc=base.now(),
        metrics=metrics(final_rows, old_world, edit),
        updates=steps,
        parameter_count=parameter_count,
        trainable_parameters=parameter_count,
        initial_all_gradients_finite=True,
        sampled_parameters_changed=changed,
        source_checkpoint=str(source),
        fresh_optimizer=True,
        budget=dict(budget),
        estimated_training_matrix_flops=flops,
        train_seconds=train_seconds,
        process_seconds=time.perf_counter() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        tf32=False,
        lock_sha256=base.sha(art / "lock.json"),
        checkpoint_files={str(p): base.sha(p) for p in (out / "model").glob("*.safetensors")},
    )
    base.write(out / "summary.json", summary)
    base.write(out / "status.json", dict(status="complete", utc=base.now()))
    print(json.dumps(summary), flush=True)


def report(config_path):
    config = base.read(config_path)
    check_lock(config)
    art, root = Path(config["artifact_root"]), Path(config["output_root"])
    world, edit = base.read(art / "world.json"), base.read(art / "edit-design.json")
    runs = {}
    checks = 0
    for arm in config["arms"]:
        for branch in (None, "update", "replay"):
            out = root / ("editing" if branch else "training") / arm
            if branch:
                out /= branch
            key = f"{arm}/{branch or 'training'}"
            summary = base.read(out / "summary.json")
            assert summary["status"] == "complete"
            assert base.read(out / "reload-audit.json")["status"] == "passed"
            for node in base.read(out / "curve.json"):
                rows = base.read(out / f"predictions-{node['step']:05d}.json")
                assert all(base.grade(r["text"], r["answer"]) == r["correct"] for r in rows)
                assert dict(step=node["step"], **metrics(rows, world, edit)) == node
                checks += len(rows)
            assert metrics(rows, world, edit) == summary["metrics"]
            runs[key] = summary
    result = dict(
        status="complete",
        phase="single_world_development",
        utc=base.now(),
        runs=runs,
        node_prediction_checks=checks,
        total_train_seconds=sum(r["train_seconds"] for r in runs.values()),
        total_process_seconds=sum(r["process_seconds"] for r in runs.values()),
        total_updates=sum(r["updates"] for r in runs.values()),
        total_input_tokens=sum(r["budget"]["input_tokens"] for r in runs.values()),
        total_supervised_tokens=sum(r["budget"]["supervised_tokens"] for r in runs.values()),
        estimated_training_matrix_flops=sum(
            r["estimated_training_matrix_flops"] for r in runs.values()
        ),
        lock_sha256=base.sha(art / "lock.json"),
    )
    base.write(art / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}), flush=True)
