#!/usr/bin/env python3
"""Full-size CUDA engineering checks; no scientific training or optimizer archives."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from llm_memory_editability import knowledge_method_data as data
from llm_memory_editability import knowledge_method_factorial as F


def version_map(model, frozen_only=False):
    return {
        name: p._version
        for name, p in model.named_parameters()
        if not frozen_only or not p.requires_grad
    }


def input_batch(tok, config, arm, pressure):
    if not pressure:
        rows = []
        with Path("/prior/shared/train.jsonl").open() as handle:
            for line in handle:
                rows.append(json.loads(line))
                if len(rows) == 4:
                    break
        if len(rows) != 4:
            raise ValueError("Need four preserved real training rows")
        samples = [F.encode_sample(tok, row, arm, config["max_sequence_tokens"]) for row in rows]
        provenance = {"kind": "first_four_preserved_training_rows", "ids": [r["id"] for r in rows]}
    else:
        cases = data.read_source(Path(config["source_csv"]), config["n_cases"])
        longest = None
        count = 0
        for case in cases:
            for index, paragraph in enumerate(case["paragraphs"]):
                ids = tok.encode(paragraph, add_special_tokens=False)
                count += 1
                if longest is None or len(ids) > len(longest[3]):
                    longest = (case["case_id"], index, paragraph, ids)
        if count != config["expected_paragraphs"]:
            raise ValueError("Pressure source paragraph coverage mismatch")
        question_ids = tok.encode(
            "Which factual information is provided in this passage? " * 200,
            add_special_tokens=False,
        )[:512]
        question = tok.decode(question_ids, skip_special_tokens=False)
        if len(tok.encode(question, add_special_tokens=False)) != 512:
            raise ValueError("Pressure question must retokenize to exactly 512 tokens")
        paragraph_ids = longest[3]
        answer_ids = (paragraph_ids * (512 // len(paragraph_ids) + 1))[:512]
        student = tok.encode(
            data.render_chat(tok, data.chat_user(question)), add_special_tokens=False
        )
        teacher = tok.encode(
            data.render_chat(tok, data.context_user(longest[2], question)), add_special_tokens=False
        )
        sample = {"student": student, "teacher": teacher, "answer": answer_ids, "id": "pressure"}
        if max(len(student), len(teacher)) + len(answer_ids) > config["max_sequence_tokens"]:
            raise ValueError(
                "Longest paragraph plus 512Q/512A exceeds the configured sequence limit"
            )
        samples = [sample] * 4
        provenance = {
            "kind": "longest_actual_paragraph_plus_512_question_and_512_answer_tokens",
            "case_id": longest[0],
            "paragraph_index": longest[1],
            "paragraph_tokens": len(paragraph_ids),
            "paragraphs_scanned": count,
            "question_tokens": 512,
            "answer_tokens": 512,
            "batch_size": 4,
        }
    batch = F.shared.collate(samples, tok.pad_token_id)
    provenance["student_input_tokens"] = batch["student"]["attention_mask"].sum(1).tolist()
    provenance["teacher_input_tokens"] = batch["teacher"]["attention_mask"].sum(1).tolist()
    provenance["target_tokens"] = batch["student"]["answer_mask"].sum(1).tolist()
    return batch, provenance


def teacher_logits(model, batch):
    model.eval()
    t = batch["teacher"]
    with torch.no_grad(), F.autocast():
        logits = (
            model(input_ids=t["input_ids"], attention_mask=t["attention_mask"], use_cache=False)
            .logits[t["answer_mask"]]
            .cpu()
        )
    torch.cuda.synchronize()
    return logits


def exercise(config, engineering, arm, pressure=False):
    F.shared.seed_all(config["seed"])
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    began = time.time()
    model, tok = F.load_model(config)
    model = F.configure_arm(model, config, arm)
    teacher = F.load_teacher(model, config, arm)
    contract = F.parameter_contract(model, arm)
    frozen_versions = version_map(model, frozen_only=True)
    teacher_versions = version_map(teacher) if teacher is not None else {}
    batch, provenance = input_batch(tok, config, arm, pressure)
    reference_teacher = teacher_logits(teacher, batch) if teacher is not None else None
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    records = []
    model.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.batch_loss(model, batch, arm, config, teacher)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        missing = [
            name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
        ]
        if missing:
            raise ValueError(f"Missing trainable gradients: {missing}")
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        if version_map(model, frozen_only=True) != frozen_versions:
            raise AssertionError("Frozen student parameter version changed")
        if teacher is not None and version_map(teacher) != teacher_versions:
            raise AssertionError("Frozen teacher parameter version changed")
        records.append(
            {
                "update": step + 1,
                "loss": float(loss.detach()),
                "grad_norm": float(norm),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    teacher_equal = None
    if teacher is not None:
        actual_teacher = teacher_logits(teacher, batch)
        teacher_equal = torch.equal(reference_teacher, actual_teacher)
        if not teacher_equal:
            raise AssertionError("Independent frozen teacher logits changed after student updates")
    result = {
        "passed": True,
        "arm": arm,
        "pressure": pressure,
        "input": provenance,
        "parameters": contract,
        "frozen_student_parameter_count": len(frozen_versions),
        "frozen_student_versions_unchanged": True,
        "independent_teacher": teacher is not None,
        "independent_teacher_parameter_count": len(teacher_versions),
        "independent_teacher_versions_unchanged": teacher is not None,
        "independent_teacher_logits_exactly_unchanged": teacher_equal,
        "updates": records,
        "seconds": time.time() - began,
        "gpu_name": torch.cuda.get_device_name(0),
        "precision": "fp32_parameters_bf16_autocast",
    }
    if arm.startswith("pd_") and not pressure:
        folder = engineering / "saved" / arm
        folder.mkdir(parents=True, exist_ok=False)
        F.shared.atomic_torch_save(F.shared.state(model), folder / "weights.pt")
        model.eval()
        s = batch["student"]
        with torch.no_grad(), F.autocast():
            logits = model(
                input_ids=s["input_ids"], attention_mask=s["attention_mask"], use_cache=False
            ).logits
        F.shared.atomic_torch_save(
            {
                "input_ids": s["input_ids"].cpu(),
                "attention_mask": s["attention_mask"].cpu(),
                "answer_mask": s["answer_mask"].cpu(),
                "greedy": logits.argmax(-1).cpu(),
                "logits": logits[s["answer_mask"]][:, :1024].cpu(),
            },
            folder / "reference.pt",
        )
        result["saved_weights_sha256"] = F.shared.digest(folder / "weights.pt")
        result["saved_weights_bytes"] = (folder / "weights.pt").stat().st_size
        result["reload_state"] = "pending_new_process"
    return result


def reload_check(config, engineering, arm):
    model, _ = F.load_model(config)
    model = F.configure_arm(model, config, arm)
    folder = engineering / "saved" / arm
    F.shared.restore(
        model, torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    reference = torch.load(folder / "reference.pt", map_location="cpu", weights_only=True)
    with torch.no_grad(), F.autocast():
        logits = model(
            input_ids=reference["input_ids"].cuda(),
            attention_mask=reference["attention_mask"].cuda(),
            use_cache=False,
        ).logits
    selected = logits[reference["answer_mask"].cuda()][:, :1024].cpu()
    difference = (selected.float() - reference["logits"].float()).abs().max().item()
    greedy_equal = torch.equal(logits.argmax(-1).cpu(), reference["greedy"])
    training = json.loads((engineering / "results" / f"exercise-{arm}.json").read_text())
    hash_equal = training["saved_weights_sha256"] == F.shared.digest(folder / "weights.pt")
    return {
        "passed": difference == 0 and greedy_equal and hash_equal,
        "arm": arm,
        "max_abs_logit_difference": difference,
        "greedy_all_positions_equal": greedy_equal,
        "weights_hash_equal": hash_equal,
        "new_process_pid": os.getpid(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engineering", type=Path, default=Path("/batch/engineering"))
    parser.add_argument("--stage", choices=["all", "exercise", "reload", "pressure"], default="all")
    parser.add_argument("--arm", choices=F.ARMS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    engineering = args.engineering
    if args.stage == "all":
        tasks = [
            ("exercise", arm)
            for arm in (
                "pd_full",
                "pd_down",
                "pd_lora",
                "pd_down_lora",
                "sft_full",
                "sft_down",
                "sft_lora",
                "sft_down_lora",
            )
        ]
        tasks += [("reload", arm) for arm in ("pd_full", "pd_down", "pd_lora", "pd_down_lora")]
        tasks += [("pressure", "pd_full")]
        results = {}
        for stage, arm in tasks:
            name = f"{stage}-{arm}"
            F.write(
                engineering / "progress.json",
                {"state": "running", "task": name, "finished": results},
            )
            print(json.dumps({"event": "starting", "task": name}), flush=True)
            log_path = engineering / "logs" / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log:
                process = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        __file__,
                        "--config",
                        str(args.config),
                        "--engineering",
                        str(engineering),
                        "--stage",
                        stage,
                        "--arm",
                        arm,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result_path = engineering / "results" / f"{name}.json"
            result = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {
                    "passed": False,
                    "error": "Child exited without a result",
                    "exit_code": process.returncode,
                }
            )
            result["exit_code"] = process.returncode
            results[name] = result
            print(
                json.dumps({"event": "finished", "task": name, "passed": result["passed"]}),
                flush=True,
            )
        passed = all(value["passed"] and value["exit_code"] == 0 for value in results.values())
        F.write(
            engineering / "full-size-check.json",
            {
                "passed": passed,
                "scientific_training": False,
                "stage_results": results,
                "source_manifest": json.loads((engineering / "source-manifest.json").read_text()),
                "config_sha256": F.shared.digest(args.config),
                "fixed_image": os.environ.get("ENGINEERING_IMAGE"),
                "independent_reload_arms": [arm for stage, arm in tasks if stage == "reload"],
                "gpu_scope": "GPU3 only; shared project lease held by the host controller",
            },
        )
        F.write(
            engineering / "progress.json",
            {"state": "complete" if passed else "failed", "finished": results},
        )
        return 0 if passed else 1
    output = engineering / "results" / f"{args.stage}-{args.arm}.json"
    try:
        if args.stage == "reload":
            result = reload_check(config, engineering, args.arm)
        else:
            result = exercise(config, engineering, args.arm, pressure=args.stage == "pressure")
        F.write(output, result)
        return 0 if result["passed"] else 1
    except BaseException as error:
        F.write(
            output,
            {
                "passed": False,
                "arm": args.arm,
                "stage": args.stage,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
