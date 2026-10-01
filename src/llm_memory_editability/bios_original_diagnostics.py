"""Frozen endpoint diagnostics for bioS; no training and no ground-truth auto inputs."""

from __future__ import annotations

import calendar
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .bios_original_data import ATTRS, TASKS, answer, date, digest, load_json, question, write_json
from .bios_original_train import (
    amp,
    encode_row,
    generate,
    load_model,
    normalize,
    sequence_scores,
    setup,
)

DEFAULT_ROOT = "docs/development-artifacts/bios-original-diagnostics-v1"
TASK_PAIR = ("parity", "comparison")


def select_people(people, world, count=24):
    """Sample whole development partner pairs without model-dependent filtering."""
    if count % 2:
        raise ValueError("The development count must consist of whole pairs")
    pairs = [
        (p["id"], p["partner"]) for p in people if p["split"] == "dev" and p["id"] < p["partner"]
    ]
    pairs = random.Random(world + 930001).sample(pairs, count // 2)
    return sorted(i for pair in pairs for i in pair)


def parse_date(text):
    """Strict full-string date parser; malformed generations remain failures."""
    text = normalize(text)
    names = {name.casefold(): i for i, name in enumerate(calendar.month_name) if i}
    match = re.fullmatch(r"([a-z]+) ([1-9]|1\d|2[0-8]), (\d{4})", text)
    if match is None or match[1] not in names:
        return None
    return (int(match[3]), names[match[1]], int(match[2]))


def operation_prompt(task, first, second=None):
    """Exact rule-training prompt; only supplied strings enter this function."""
    if task == "parity":
        parsed = parse_date(first)
        month = calendar.month_name[parsed[1]] if parsed else first
        return f"Question: Is {month} an even-numbered month?\nAnswer:"
    if task == "comparison":
        return f"Question: Which date is earlier, {first} or {second}?\nAnswer:"
    raise ValueError(task)


def decode_operation(task, generated, first, second=None, names=None):
    """Map a model-selected supplied date to its name without consulting truth."""
    if task == "parity":
        return normalize(generated)
    chosen = normalize(generated)
    values = [normalize(first), normalize(second)]
    if values[0] == values[1] or chosen not in values:
        return None
    return names[values.index(chosen)]


def symbolic_operation(task, first, second=None, names=None):
    """A separately labelled host-computation ceiling, never a model result."""
    a = parse_date(first)
    if a is None:
        return None
    if task == "parity":
        return "yes" if a[1] % 2 == 0 else "no"
    b = parse_date(second)
    if b is None or a == b:
        return None
    return names[0 if a < b else 1]


def audit_batches(people, tasks, cot, batch_size):
    """Select one original physical batch per split, independent of predictions."""
    eligible = [p for p in people if p["split"] in ("dev", "test")]
    rows_per_person = len(tasks) * 6 * (2 if cot else 1)
    starts = set()
    for split in ("dev", "test"):
        person_index = next(i for i, p in enumerate(eligible) if p["split"] == split)
        starts.add((person_index * rows_per_person // batch_size) * batch_size)
    return sorted(starts)


def read_selected_batches(path, starts, batch_size):
    batches = {s: [] for s in starts}
    with Path(path).open() as stream:
        for i, line in enumerate(stream):
            for s in starts:
                if s <= i < s + batch_size:
                    batches[s].append(dict(load_json_line=json.loads(line), source_row=i))
            if i >= max(starts) + batch_size - 1:
                break
    if any(len(rows) != batch_size for rows in batches.values()):
        raise ValueError("Incomplete historical physical batch")
    return [dict(start=s, rows=rows) for s, rows in batches.items()]


def prepare(config_path, root=DEFAULT_ROOT, count=24):
    root = Path(root)
    if (root / "plan.json").exists():
        raise FileExistsError("Diagnostic inputs already frozen")
    cfg = load_json(config_path)
    source_files = [
        "src/llm_memory_editability/bios_original_diagnostics.py",
        "scripts/run_bios_original_diagnostics.py",
        "src/llm_memory_editability/bios_original_data.py",
        "src/llm_memory_editability/bios_original_model.py",
        "src/llm_memory_editability/bios_original_train.py",
    ]
    # Loading current implementations is valid only if original implementations match.
    frozen = Path(cfg["result_root"]) / "execution-source"
    for file in source_files[2:]:
        if digest(file) != digest(frozen / file):
            raise ValueError(f"Original source changed: {file}")
    plan = dict(
        version=1,
        created_unix=time.time(),
        config_path=str(config_path),
        config_sha256=digest(config_path),
        config=cfg,
        count=count,
        views=[0, 1],
        diagnostic_batch_size=16,
        audit_batch_size=cfg["evaluation"]["batch_size"],
        max_new_tokens=cfg["evaluation"]["max_new_tokens"],
        source_sha256={f: digest(f) for f in source_files},
        tokenizer_sha256=digest(Path(cfg["data_root"]) / "tokenizer/tokenizer.json"),
        worlds={},
        endpoints=[],
    )
    for world in cfg["world_seeds"]:
        data_path = Path(cfg["data_root"]) / f"world-{world}/people.json"
        people = load_json(data_path)
        plan["worlds"][str(world)] = dict(
            person_ids=select_people(people, world, count),
            people_path=str(data_path),
            sha256=digest(data_path),
        )
        for init in cfg["initialization_seeds"]:
            for condition in cfg["task"]["conditions"]:
                stages = ["adapt", "task"] if condition in ("S", "M", "MP") else ["task"]
                for stage in stages:
                    path = (
                        Path(cfg["result_root"]) / f"world-{world}/init-{init}/{condition}/{stage}"
                    )
                    complete = load_json(path / "complete.json")
                    evaluation = load_json(path / "evaluation/evaluation-complete.json")
                    if complete["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
                        raise ValueError("Checkpoint/evaluation hash disagreement")
                    endpoint_id = f"world-{world}-{condition}-{stage}"
                    tasks = ATTRS if stage == "adapt" else TASKS
                    cot = stage == "task" and condition == "MP-CoT"
                    selected = read_selected_batches(
                        path / "evaluation/predictions.jsonl",
                        audit_batches(people, tasks, cot, plan["audit_batch_size"]),
                        plan["audit_batch_size"],
                    )
                    sample = root / "audit-inputs" / f"{endpoint_id}.json"
                    write_json(sample, selected)
                    plan["endpoints"].append(
                        dict(
                            id=endpoint_id,
                            world=world,
                            init=init,
                            condition=condition,
                            stage=stage,
                            checkpoint=str(path / "final.pt"),
                            checkpoint_sha256=complete["checkpoint_sha256"],
                            audit_sample=str(sample),
                            audit_sample_sha256=digest(sample),
                        )
                    )
    write_json(root / "plan.json", plan)
    return plan


def verify_plan(plan):
    for path, sha in plan["source_sha256"].items():
        if digest(path) != sha:
            raise ValueError(f"Frozen diagnostic source changed: {path}")
    for data in plan["worlds"].values():
        if digest(data["people_path"]) != data["sha256"]:
            raise ValueError("People data changed")


def run_audit(model, tok, endpoint, plan, output, device):
    sample = Path(endpoint["audit_sample"])
    if digest(sample) != endpoint["audit_sample_sha256"]:
        raise ValueError("Audit inputs changed")
    report = []
    for batch in load_json(sample):
        rows = [r["load_json_line"] for r in batch["rows"]]
        encoded = [encode_row(tok, r["prompt"], r["answer"]) for r in rows]
        with torch.no_grad(), amp(device):
            avg, full = sequence_scores(model, encoded, device)
        generated, ended = generate(
            model, tok, [r["prompt"] for r in rows], device, plan["max_new_tokens"]
        )
        for original, text, eos, mean_lp, full_lp in zip(
            batch["rows"], generated, ended, avg.tolist(), full.tolist(), strict=True
        ):
            old = original["load_json_line"]
            extracted = text.rsplit("Answer:", 1)[-1].strip() if old["cot"] else text
            report.append(
                dict(
                    source_row=original["source_row"],
                    physical_batch_start=batch["start"],
                    original=old,
                    reloaded_generated=text,
                    reloaded_eos=eos,
                    generated_identical=text == old["generated"],
                    eos_identical=eos == old["eos"],
                    correctness_identical=(normalize(extracted) == normalize(old["answer"]))
                    == old["correct"],
                    mean_logp_difference=mean_lp - old["answer_mean_logp"],
                    full_logp_difference=full_lp - old["answer_full_logp"],
                )
            )
    write_json(output / "audit-rows.json", report)
    summary = dict(
        rows=len(report),
        generation_mismatches=sum(not r["generated_identical"] for r in report),
        eos_mismatches=sum(not r["eos_identical"] for r in report),
        correctness_mismatches=sum(not r["correctness_identical"] for r in report),
        max_abs_full_logp_difference=max(abs(r["full_logp_difference"]) for r in report),
        exact_original_batch=True,
        precision="FP32 parameters, BF16 autocast, TF32 enabled",
    )
    summary["status"] = (
        "passed"
        if all(
            summary[k] == 0
            for k in ("generation_mismatches", "eos_mismatches", "correctness_mismatches")
        )
        and summary["max_abs_full_logp_difference"] <= 1e-4
        else "discrepancy"
    )
    write_json(output / "audit-summary.json", summary)
    return summary


def generate_rows(model, tok, rows, plan, device):
    for start in range(0, len(rows), plan["diagnostic_batch_size"]):
        batch = rows[start : start + plan["diagnostic_batch_size"]]
        outputs, ended = generate(
            model, tok, [r["prompt"] for r in batch], device, plan["max_new_tokens"]
        )
        for row, text, eos in zip(batch, outputs, ended, strict=True):
            row.update(generated=text, eos=eos)
        print(
            json.dumps(
                dict(generation_done=min(start + len(batch), len(rows)), generation_total=len(rows))
            ),
            flush=True,
        )


def run_diagnostics(model, tok, endpoint, plan, output, device):
    world = plan["worlds"][str(endpoint["world"])]
    people = load_json(world["people_path"])
    selected = [people[i] for i in world["person_ids"]]
    extraction = [
        dict(person_id=p["id"], view=v, prompt=question(p, "date", v), target=date(p))
        for p in selected
        for v in plan["views"]
    ]
    generate_rows(model, tok, extraction, plan, device)
    lookup = {}
    for row in extraction:
        row.update(
            parseable=parse_date(row["generated"]) is not None,
            correct=normalize(row["generated"]) == normalize(row["target"]),
        )
        lookup[row["person_id"], row["view"]] = row
    write_json(output / "extractions.json", extraction)
    rows = []
    for p in selected:
        other = people[p["partner"]]
        for task in TASK_PAIR:
            for view in plan["views"]:
                base = dict(
                    person_id=p["id"],
                    partner=other["id"],
                    view=view,
                    task=task,
                    target=answer(p, task, other),
                )
                for cot in (False, True):
                    rows.append(
                        dict(
                            base,
                            mode="cot" if cot else "direct",
                            prompt=question(p, task, view, other, cot),
                            cot_trained=endpoint["condition"] == "MP-CoT",
                        )
                    )
                for mode in ("oracle_rule", "autonomous_rule"):
                    a = date(p) if mode == "oracle_rule" else lookup[p["id"], view]["generated"]
                    b = (
                        date(other)
                        if mode == "oracle_rule"
                        else lookup[other["id"], view]["generated"]
                    )
                    parsed = parse_date(a) is not None and (
                        task != "comparison" or parse_date(b) is not None
                    )
                    valid = normalize(a) == normalize(date(p)) and (
                        task != "comparison" or normalize(b) == normalize(date(other))
                    )
                    symbolic = symbolic_operation(task, a, b, (p["name"], other["name"]))
                    rows.append(
                        dict(
                            base,
                            mode=mode,
                            prompt=operation_prompt(task, a, b),
                            supplied_dates=[a, b] if task == "comparison" else [a],
                            inputs_parseable=parsed,
                            inputs_correct=valid,
                            symbolic_prediction=symbolic,
                            symbolic_correct=symbolic is not None
                            and normalize(symbolic) == normalize(base["target"]),
                        )
                    )
    generate_rows(model, tok, rows, plan, device)
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        p, other = people[row["person_id"]], people[row["partner"]]
        if row["mode"] in ("direct", "cot"):
            pred = (
                row["generated"].rsplit("Answer:", 1)[-1].strip()
                if row["mode"] == "cot"
                else row["generated"]
            )
        else:
            dates = row["supplied_dates"]
            pred = decode_operation(
                row["task"],
                row["generated"],
                dates[0],
                dates[1] if len(dates) == 2 else None,
                (p["name"], other["name"]),
            )
        row["prediction"] = pred
        row["output_in_question_pair"] = (
            pred is not None and normalize(pred) in (normalize(p["name"]), normalize(other["name"]))
            if row["task"] == "comparison"
            else None
        )
        if row["task"] == "comparison" and row["mode"] in ("oracle_rule", "autonomous_rule"):
            row["output_is_supplied_date"] = normalize(row["generated"]) in {
                normalize(value) for value in row["supplied_dates"]
            }
        row["correct"] = pred is not None and normalize(pred) == normalize(row["target"])
        row["correct_with_eos"] = row["correct"] and row["eos"]
        key = f"{row['task']}/view-{row['view']}/{row['mode']}"
        total = totals[key]
        total["n"] += 1
        total["correct"] += row["correct"]
        total["correct_with_eos"] += row["correct_with_eos"]
        if row["task"] == "comparison":
            total["output_in_question_pair"] += row["output_in_question_pair"]
            if "output_is_supplied_date" in row:
                total["output_is_supplied_date"] += row["output_is_supplied_date"]
        if row["mode"] == "autonomous_rule":
            for condition in ("inputs_parseable", "inputs_correct"):
                total[condition + "_n"] += row[condition]
                total[condition + "_correct"] += row[condition] and row["correct"]
            total["symbolic_correct"] += row["symbolic_correct"]
    with (output / "diagnostics.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    summary = dict(
        rows=len(rows),
        extraction_rows=len(extraction),
        extraction_correct=sum(r["correct"] for r in extraction),
        extraction_parseable=sum(r["parseable"] for r in extraction),
        totals=dict(totals),
        people=len(selected),
        independent_partner_pairs=len(selected) // 2,
        split="development",
        no_training=True,
        limitations=[
            "Oracle and autonomous rules use externally supplied text and a changed prompt.",
            "Non-MP-CoT models did not train on the CoT instruction.",
            "Host date-to-name mapping and symbolic ceilings are not model reasoning.",
            "Twenty-four people are a fixed diagnostic subset, not population estimates.",
        ],
    )
    write_json(output / "diagnostic-summary.json", summary)
    return summary


def run_endpoint(root, endpoint_id, device="cuda:0", memory_fraction=0.045):
    root = Path(root)
    plan = load_json(root / "plan.json")
    verify_plan(plan)
    endpoint = next(e for e in plan["endpoints"] if e["id"] == endpoint_id)
    output = root / "endpoints" / endpoint_id
    if (output / "complete.json").exists():
        return load_json(output / "complete.json")
    output.mkdir(parents=True, exist_ok=True)
    if digest(endpoint["checkpoint"]) != endpoint["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    cfg = plan["config"]
    setup(endpoint["init"], device)
    if str(device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    tok_path = Path(cfg["data_root"]) / "tokenizer/tokenizer.json"
    if digest(tok_path) != plan["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    tok = Tokenizer.from_file(str(tok_path))
    start = time.monotonic()
    model = load_model(endpoint["checkpoint"], cfg, device, adapters=True).eval()
    model.requires_grad_(False)
    write_json(
        output / "running.json",
        dict(
            endpoint=endpoint_id,
            started_unix=time.time(),
            pid=os.getpid(),
            device=device,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            memory_fraction=memory_fraction,
            shared_gpu=True,
            torch_version=torch.__version__,
        ),
    )
    audit = run_audit(model, tok, endpoint, plan, output, device)
    diagnostics = (
        run_diagnostics(model, tok, endpoint, plan, output, device)
        if endpoint["stage"] == "task"
        else None
    )
    result = dict(
        endpoint=endpoint,
        seconds=time.monotonic() - start,
        audit=audit,
        diagnostics=diagnostics,
        shared_gpu=True,
        memory_fraction=memory_fraction,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
    )
    write_json(output / "complete.json", result)
    return result
