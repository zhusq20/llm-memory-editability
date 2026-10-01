#!/usr/bin/env python3
"""Measure native biography continuation on frozen pretraining checkpoints, without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from llm_memory_editability.bios_original_data import date, source_templates
from llm_memory_editability.bios_original_model import GPT
from llm_memory_editability.bios_original_train import amp, setup

OUTPUT = Path("docs/development-artifacts/bios-original-trajectory-v1")
CONFIG = Path("configs/bios-original-development-v1.json")
SCRIPT = Path(__file__)
SOURCE_PATHS = (
    SCRIPT,
    Path("src/llm_memory_editability/bios_original_model.py"),
    Path("src/llm_memory_editability/bios_original_data.py"),
    Path("src/llm_memory_editability/bios_original_train.py"),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def native_case(person, attr, templates, ids, sentence_tokens, sentence_lengths, tokenizer):
    """Use the actual variant-0 date/birthplace order, including sentence-token boundaries."""
    index = 0 if attr == "date" else 1
    form = int(index != 0)
    template = templates[index][int(ids[index])]
    placeholder = "birthday" if attr == "date" else "birthcity"
    if attr == "date" and template.index("{name}") > template.index("{birthday}"):
        raise ValueError("date-before-name")
    fields = dict(
        name=person["name"] if attr == "date" else person["pronoun"],
        birthday=date(person),
        birthcity=person["birthcity"],
        university=person["university"],
        field=person["field"],
        company1name=person["company"],
        company1city=person["workcity"],
    )
    target = fields[placeholder]
    fields[placeholder] = "___ATTRIBUTE___"
    marked = " " + template.format(**fields)
    prefix, suffix = marked.split("___ATTRIBUTE___")
    trimmed = prefix.rstrip()
    target_surface = prefix[len(trimmed) :] + target
    prefix_ids = tokenizer.encode(trimmed).ids
    target_ids = tokenizer.encode(target_surface).ids
    full = sentence_tokens[index, form, : int(sentence_lengths[index, form])].tolist()
    n = len(prefix_ids) + len(target_ids)
    if prefix_ids + target_ids != full[:n] or n >= len(full):
        raise ValueError("attribute-token-boundary")
    initial = (
        [] if attr == "date" else sentence_tokens[0, 0, : int(sentence_lengths[0, 0])].tolist()
    )
    prompt_ids = [50256] + initial + prefix_ids
    if person["name"] not in tokenizer.decode(prompt_ids, skip_special_tokens=True):
        raise ValueError("name-not-in-prefix")
    if target in tokenizer.decode(prompt_ids, skip_special_tokens=True):
        raise ValueError("target-already-in-prefix")
    return dict(
        person_id=person["id"],
        split=person["split"],
        attribute=attr,
        variant=0,
        template_id=int(ids[index]),
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        target=target,
        target_surface=target_surface,
        suffix=suffix,
        boundary_token=full[n],
        prefix=tokenizer.decode(prompt_ids, skip_special_tokens=False),
        seen_surface_S=True,
        seen_surface_M=True,
        same_sentences_MP=True,
        MP_order_exposure="permuted; exact prefix occurrence not individually reconstructed",
        contains_true_date_prefix=attr == "birthcity",
    )


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "lock.json").exists():
        validate(output)
        return
    cfg = json.loads(CONFIG.read_text())
    tokenizer_path = Path(cfg["data_root"]) / "tokenizer/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    templates = source_templates()
    all_rows, coverage, files = [], [], [CONFIG, tokenizer_path]
    for world in cfg["world_seeds"]:
        root = Path(cfg["data_root"]) / f"world-{world}"
        people = json.loads((root / "people.json").read_text())
        ids = np.load(root / "template-ids.npy", mmap_mode="r")
        tokens = np.load(root / "sentence-tokens.npy", mmap_mode="r")
        lengths = np.load(root / "sentence-lengths.npy", mmap_mode="r")
        candidates, rejected = [], defaultdict(int)
        for person in people:
            if person["split"] != "dev":
                continue
            try:
                rows = [
                    native_case(
                        person,
                        attr,
                        templates,
                        ids[person["id"], 0],
                        tokens[person["id"], 0],
                        lengths[person["id"], 0],
                        tokenizer,
                    )
                    for attr in ("date", "birthcity")
                ]
            except ValueError as error:
                rejected[str(error)] += 1
            else:
                candidates.append(rows)
        selected = random.Random(world + 20260930).sample(candidates, 32)
        for pair in selected:
            all_rows.extend(dict(world=world, **row) for row in pair)
        coverage.append(
            dict(
                world=world,
                candidate_dev_people=sum(p["split"] == "dev" for p in people),
                eligible_people=len(candidates),
                selected_people=len(selected),
                rejected=dict(rejected),
                selection_seed=world + 20260930,
            )
        )
        files.extend(
            root / name
            for name in (
                "people.json",
                "template-ids.npy",
                "sentence-tokens.npy",
                "sentence-lengths.npy",
            )
        )
    save(output / "cases.json", all_rows)
    save(output / "coverage.json", coverage)
    generation_length = max(len(row["target_ids"]) for row in all_rows) + 2
    for path in SOURCE_PATHS:
        target = output / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    save(
        output / "lock.json",
        dict(
            phase="frozen-before-inference",
            cases_sha256=sha(output / "cases.json"),
            coverage_sha256=sha(output / "coverage.json"),
            config=cfg,
            people_per_world=32,
            attributes=["date", "birthcity"],
            nodes=cfg["pretrain"]["nodes"],
            expected_model_nodes=36,
            expected_predictions=2304,
            batch_size=4,
            max_new_tokens=generation_length,
            gpu_peak_limit_bytes=3 * 1024**3,
            precision="FP32 parameters, original BF16 autocast and original TF32 settings",
            measures=[
                "full attribute token NLL/probability",
                "greedy full attribute token sequence exact",
                "greedy full attribute sequence plus next native boundary token exact",
            ],
            scope="Seen native biography completion, not QA or unseen-expression generalization",
            sources=[dict(path=str(p), sha256=sha(p)) for p in SOURCE_PATHS],
            inputs=[dict(path=str(p), sha256=sha(p)) for p in files],
            selection=(
                "Fixed-seed dev sampling; variant-0 name precedes date, native token boundaries "
                "verified; no model result used"
            ),
        ),
    )


def validate(output):
    lock = json.loads((output / "lock.json").read_text())
    if sha(output / "cases.json") != lock["cases_sha256"]:
        raise ValueError("Frozen cases changed")
    for rec in lock["sources"]:
        if sha(rec["path"]) != rec["sha256"]:
            raise ValueError(f"Frozen implementation changed: {rec['path']}")
    return lock


@torch.inference_mode()
def measure(model, rows, tokenizer, device, max_new_tokens):
    values = []
    # Only attribute tokens have labels; there is deliberately no fabricated EOS target.
    seqs = [r["prompt_ids"] + r["target_ids"] for r in rows]
    x = torch.full((len(rows), max(map(len, seqs)) - 1), 50256, device=device, dtype=torch.long)
    indices, target_ids, owners = [], [], []
    for i, (row, seq) in enumerate(zip(rows, seqs, strict=True)):
        x[i, : len(seq) - 1] = torch.tensor(seq[:-1], device=device)
        for j, token in enumerate(row["target_ids"]):
            indices.append((i, len(row["prompt_ids"]) - 1 + j))
            target_ids.append(token)
            owners.append(i)
    with amp(device):
        hidden = model.hidden(x)
        idx = torch.tensor(indices, device=device)
        logits = model.logits(hidden[idx[:, 0], idx[:, 1]]).float()
        losses = torch.nn.functional.cross_entropy(
            logits,
            torch.tensor(target_ids, device=device),
            reduction="none",
        )
    sums = (
        torch.zeros(len(rows), device=device)
        .scatter_add_(
            0,
            torch.tensor(owners, device=device),
            losses,
        )
        .tolist()
    )
    seqs = [list(r["prompt_ids"]) for r in rows]
    generated = [[] for _ in rows]
    for _ in range(max_new_tokens):
        x = torch.full((len(rows), max(map(len, seqs))), 50256, device=device, dtype=torch.long)
        for i, seq in enumerate(seqs):
            x[i, : len(seq)] = torch.tensor(seq, device=device)
        with amp(device):
            hidden = model.hidden(x)
            last = hidden[
                torch.arange(len(rows), device=device),
                torch.tensor([len(s) - 1 for s in seqs], device=device),
            ]
            pred = model.logits(last).argmax(-1).tolist()
        for i, token in enumerate(pred):
            seqs[i].append(token)
            generated[i].append(token)
    for row, nll, gen in zip(rows, sums, generated, strict=True):
        n = len(row["target_ids"])
        exact = gen[:n] == row["target_ids"]
        values.append(
            dict(
                **row,
                full_attribute_nll=nll,
                mean_attribute_nll=nll / n,
                full_attribute_probability=math.exp(-nll),
                attribute_exact=exact,
                attribute_and_boundary_exact=exact and gen[n] == row["boundary_token"],
                generated_ids=gen,
                generated=tokenizer.decode(gen, skip_special_tokens=False),
            )
        )
    return values


def run(output, device):
    lock = validate(output)
    cfg = lock["config"]
    setup(20260930, device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(lock["gpu_peak_limit_bytes"] / total_memory, device)
    tokenizer = Tokenizer.from_file(str(Path(cfg["data_root"]) / "tokenizer/tokenizer.json"))
    cases = json.loads((output / "cases.json").read_text())
    model = GPT(cfg["model"]).to(device).eval()
    for world in cfg["world_seeds"]:
        rows = [r for r in cases if r["world"] == world]
        for init in cfg["initialization_seeds"]:
            for condition in cfg["pretrain"]["conditions"]:
                for node in lock["nodes"]:
                    key = f"world-{world}-init-{init}-{condition}-pass-{node:03d}"
                    destination = output / "nodes" / f"{key}.json"
                    if destination.exists():
                        continue
                    destination.parent.mkdir(exist_ok=True)
                    checkpoint = Path(cfg["result_root"]) / (
                        f"world-{world}/init-{init}/{condition}/pretrain/pass-{node:03d}.pt"
                    )
                    started = time.monotonic()
                    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model.load_state_dict(state["model"], strict=True)
                    del state
                    torch.cuda.reset_peak_memory_stats(device)
                    results = []
                    for start in range(0, len(rows), lock["batch_size"]):
                        results.extend(
                            measure(
                                model,
                                rows[start : start + lock["batch_size"]],
                                tokenizer,
                                device,
                                lock["max_new_tokens"],
                            )
                        )
                    peak = torch.cuda.max_memory_allocated(device)
                    if peak > lock["gpu_peak_limit_bytes"]:
                        raise RuntimeError("GPU peak exceeded 3 GiB")
                    save(
                        destination,
                        dict(
                            world=world,
                            initialization=init,
                            condition=condition,
                            passes=node,
                            checkpoint=str(checkpoint),
                            checkpoint_sha256=sha(checkpoint),
                            lock_sha256=sha(output / "lock.json"),
                            gpu_peak_bytes=peak,
                            seconds=time.monotonic() - started,
                            predictions=results,
                        ),
                    )
                    print(
                        json.dumps(
                            dict(
                                node=key,
                                n=len(results),
                                peak=peak,
                                seconds=time.monotonic() - started,
                            )
                        ),
                        flush=True,
                    )
    report(output)


def report(output):
    lock = validate(output)
    nodes = [json.loads(p.read_text()) for p in sorted((output / "nodes").glob("*.json"))]
    if len(nodes) != lock["expected_model_nodes"]:
        raise ValueError("Incomplete checkpoint matrix")
    rows = []
    for node in nodes:
        if len(node["predictions"]) != 64 or node["lock_sha256"] != sha(output / "lock.json"):
            raise ValueError("Invalid node predictions or provenance")
        for attr in lock["attributes"]:
            data = [r for r in node["predictions"] if r["attribute"] == attr]
            rows.append(
                dict(
                    world=node["world"],
                    initialization=node["initialization"],
                    condition=node["condition"],
                    passes=node["passes"],
                    attribute=attr,
                    n=len(data),
                    accuracy=sum(r["attribute_exact"] for r in data) / len(data),
                    boundary_accuracy=sum(r["attribute_and_boundary_exact"] for r in data)
                    / len(data),
                    mean_full_nll=sum(r["full_attribute_nll"] for r in data) / len(data),
                    mean_full_probability=sum(r["full_attribute_probability"] for r in data)
                    / len(data),
                )
            )
    with (output / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(
        output / "summary.json",
        dict(
            completed_nodes=len(nodes),
            predictions=sum(len(n["predictions"]) for n in nodes),
            peak_gpu_bytes=max(n["gpu_peak_bytes"] for n in nodes),
            summed_node_seconds=sum(n["seconds"] for n in nodes),
            rows=rows,
            scope="Seen BIO continuation; birthplace conditioned on true preceding date",
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, attr in enumerate(lock["attributes"]):
        for condition in ("S", "M", "MP"):
            data = [r for r in rows if r["attribute"] == attr and r["condition"] == condition]
            for j, metric in enumerate(("accuracy", "mean_full_probability")):
                means = [
                    sum(r[metric] for r in data if r["passes"] == n) / 2 for n in lock["nodes"]
                ]
                axes[i, j].plot(lock["nodes"], means, marker="o", label=condition)
                axes[i, j].set(title=f"{attr}: {metric}", ylim=(-0.03, 1.03))
                axes[i, j].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Biography exposures per person")
    fig.suptitle("Seen native biography continuation (two development worlds)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"native-attribute-curves.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:5")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.output)
    elif args.phase == "run":
        run(args.output, args.device)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
