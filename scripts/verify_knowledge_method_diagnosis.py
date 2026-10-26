#!/usr/bin/env python3
"""Full-size engineering updates and independent reloads, never scientific data.

The ``all`` stage uses a fresh child process for every training and reload stage.
Only the supplied engineering output directory is written. No optimizer state is
saved and these eight engineering updates count as zero scientific updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from llm_memory_editability import knowledge_method_diagnosis as K

ARMS = ("pd_lora", "sft_lora", "sft_full", "sft_down")


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def fixed_rows():
    facts = [
        ("What color is the engineering test cube?", "The engineering test cube is blue.", "Blue."),
        ("What color is the engineering test cone?", "The engineering test cone is red.", "Red."),
    ]
    return [
        {
            "id": f"engineering-{i}",
            "question": facts[i % 2][0],
            "context": facts[i % 2][1],
            "pd_answer": facts[i % 2][2],
            "sft_answer": facts[i % 2][2],
            "pd_truncated": False,
            "sft_truncated": False,
        }
        for i in range(8)
    ]


def source_hashes(config_path):
    from llm_memory_editability import knowledge_method_data

    return {
        "engineering_script": K.digest(__file__),
        "training_module": K.digest(K.__file__),
        "data_module": K.digest(knowledge_method_data.__file__),
        "scientific_config": K.digest(config_path),
    }


def adapter_flags(model):
    return [
        bool(module.disable_adapters)
        for module in model.modules()
        if hasattr(module, "disable_adapters") and isinstance(module.disable_adapters, bool)
    ]


def logits_on(model, inputs):
    with torch.no_grad(), K.autocast():
        return (
            model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
            .logits.detach()
            .cpu()
        )


def stress(model, tok, config, arm, optimizer):
    """Data-shaped long sequence, with the real training optimizer still resident."""
    from llm_memory_editability.knowledge_method_data import (
        chat_user,
        context_user,
        read_source,
        render_chat,
    )

    cases = read_source(Path(config["source_csv"]), config["n_cases"])
    question_ids = tok.encode(
        "Which engineering object has this property? " * 150, add_special_tokens=False
    )[:512]
    question = tok.decode(question_ids, skip_special_tokens=False)
    assert len(tok.encode(question, add_special_tokens=False)) == 512
    paragraphs = [
        (case["case_id"], index, paragraph)
        for case in cases
        for index, paragraph in enumerate(case["paragraphs"])
    ]
    case_id, paragraph_id, context = max(
        paragraphs,
        key=lambda item: len(
            tok.encode(render_chat(tok, context_user(item[2], question)), add_special_tokens=False)
        ),
    )
    student = tok.encode(render_chat(tok, chat_user(question)), add_special_tokens=False)
    teacher = tok.encode(
        render_chat(tok, context_user(context, question)), add_special_tokens=False
    )
    answer = tok.encode("Engineering answer text. " * 200, add_special_tokens=False)[:512]
    assert len(answer) == 512
    sample = {"id": "engineering-stress", "student": student, "teacher": teacher, "answer": answer}
    assert max(len(student), len(teacher)) + len(answer) <= config["max_sequence_tokens"]
    batch = K.collate([sample] * 4, tok.pad_token_id)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    model.train()
    loss = K.batch_loss(model, batch, arm, config)
    assert bool(torch.isfinite(loss)) and model.training
    loss.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
        if p.requires_grad
    )
    torch.cuda.synchronize()
    result = {
        "passed": True,
        "batch_size": 4,
        "question_tokens": 512,
        "answer_tokens": 512,
        "student_sequence_tokens": len(student) + len(answer),
        "teacher_sequence_tokens": len(teacher) + len(answer),
        "longest_context_case": case_id,
        "longest_context_paragraph": paragraph_id,
        "available_paragraphs": len(paragraphs),
        "source_sha256": K.digest(config["source_csv"]),
        "loss": loss.item(),
        "all_gradients_finite": True,
        "seconds": time.monotonic() - began,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "optimizer_states_resident": True,
        "updates": 0,
        "scientific_updates": 0,
    }
    optimizer.zero_grad(set_to_none=True)
    return result


def train(config, output, arm, hashes):
    began = time.monotonic()
    out = output / arm
    if (out / "train.json").exists():
        raise RuntimeError("Existing engineering result must not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    K.seed_all(config["seed"])
    model, tok = K.load_model(config)
    model = K.configure_arm(model, config, arm)
    samples = [
        K.encode_sample(tok, row, arm, config["max_sequence_tokens"]) for row in fixed_rows()
    ]
    probe = K.collate(samples[:2], tok.pad_token_id)
    frozen = {name: tensor_hash(p) for name, p in model.named_parameters() if not p.requires_grad}
    versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    trainable_before = {name: tensor_hash(p) for name, p in parameters.items()}
    K.write(out / "frozen-before.json", frozen)
    if arm == "pd_lora":
        with model.disable_adapter():
            model.eval()
            teacher_before = logits_on(model, probe["teacher"])
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-5, weight_decay=config["weight_decay"])
    torch.cuda.reset_peak_memory_stats()
    model.train()
    steps = []
    forward_modes = []

    def record_forward(module, _inputs):
        forward_modes.append(
            {
                "training": module.training,
                "grad_enabled": torch.is_grad_enabled(),
                "adapter_disabled": adapter_flags(module),
            }
        )

    hook = model.register_forward_pre_hook(record_forward)
    for step in range(2):
        batch = K.collate(samples[step * 4 : (step + 1) * 4], tok.pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        before_calls = len(forward_modes)
        loss = K.batch_loss(model, batch, arm, config)
        assert bool(torch.isfinite(loss)), "nonfinite loss"
        assert model.training, "student train mode was not restored"
        calls = forward_modes[before_calls:]
        if arm == "pd_lora":
            assert len(calls) == 2, calls
            teacher, student = calls
            assert not teacher["training"] and not teacher["grad_enabled"], teacher
            assert teacher["adapter_disabled"] and all(teacher["adapter_disabled"]), teacher
            assert student["training"] and student["grad_enabled"], student
            assert student["adapter_disabled"] and not any(student["adapter_disabled"]), student
        loss.backward()
        missing, nonfinite = [], []
        for name, parameter in parameters.items():
            if parameter.grad is None:
                missing.append(name)
            elif not bool(torch.isfinite(parameter.grad).all()):
                nonfinite.append(name)
        assert not missing, f"missing gradients: {missing}"
        assert not nonfinite, f"nonfinite gradients: {nonfinite}"
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters.values(), config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        steps.append(
            {
                "step": step + 1,
                "loss": loss.item(),
                "grad_norm": norm.item(),
                "learning_rate": 1e-5,
                "all_trainable_gradients_finite": True,
            }
        )
        K.write(out / "progress.json", {"arm": arm, "steps": steps})
    hook.remove()
    stress_result = None
    if arm in {"pd_lora", "sft_full"}:
        stress_result = stress(model, tok, config, arm, optimizer)
        K.write(out / "stress.json", stress_result)
    frozen_after = {
        name: tensor_hash(p) for name, p in model.named_parameters() if not p.requires_grad
    }
    assert frozen_after == frozen, "frozen parameter checksums changed"
    assert versions == {
        name: p._version for name, p in model.named_parameters() if not p.requires_grad
    }
    changed = [name for name, p in parameters.items() if tensor_hash(p) != trainable_before[name]]
    assert changed, "no trainable parameter changed"
    teacher_equal = None
    if arm == "pd_lora":
        with model.disable_adapter():
            model.eval()
            teacher_after = logits_on(model, probe["teacher"])
        teacher_equal = torch.equal(teacher_before, teacher_after)
        assert teacher_equal, "frozen teacher outputs changed"
    del optimizer
    model.zero_grad(set_to_none=True)
    K.atomic_torch_save(K.state(model), out / "weights.pt")
    model.eval()
    reference_logits = logits_on(model, probe["student"])
    K.atomic_torch_save(
        {
            "input_ids": probe["student"]["input_ids"].cpu(),
            "attention_mask": probe["student"]["attention_mask"].cpu(),
            "logits": reference_logits,
        },
        out / "reference.pt",
    )
    result = {
        "passed": True,
        "arm": arm,
        "pid": os.getpid(),
        "engineering_updates": 2,
        "scientific_updates": 0,
        "source_sha256": hashes,
        "config": config,
        "training_rows": fixed_rows(),
        "steps": steps,
        "trainable_parameters": sum(p.numel() for p in parameters.values()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "frozen_tensors": len(frozen),
        "all_frozen_checksums_unchanged": True,
        "all_frozen_tensor_versions_unchanged": True,
        "changed_trainable_tensors": len(changed),
        "teacher_logits_equal": teacher_equal,
        "forward_modes": forward_modes,
        "data_shape_stress": stress_result,
        "weights_sha256": K.digest(out / "weights.pt"),
        "reference_sha256": K.digest(out / "reference.pt"),
        "seconds": time.monotonic() - began,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    K.write(out / "train.json", result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"config", "training_rows", "forward_modes"}
            }
        ),
        flush=True,
    )


def reload(config, output, arm, hashes):
    began = time.monotonic()
    out = output / arm
    trained = json.loads((out / "train.json").read_text())
    assert trained["pid"] != os.getpid()
    assert trained["source_sha256"] == hashes
    assert trained["config"] == config
    assert trained["weights_sha256"] == K.digest(out / "weights.pt")
    assert trained["reference_sha256"] == K.digest(out / "reference.pt")
    K.seed_all(config["seed"])
    model, _ = K.load_model(config)
    model = K.configure_arm(model, config, arm)
    K.restore(model, torch.load(out / "weights.pt", weights_only=True, map_location="cpu"))
    model.eval()
    reference = torch.load(out / "reference.pt", weights_only=True, map_location="cpu")
    inputs = {key: reference[key].to("cuda") for key in ("input_ids", "attention_mask")}
    logits = logits_on(model, inputs)
    difference = (logits.float() - reference["logits"].float()).abs().max().item()
    passed = torch.equal(logits, reference["logits"])
    result = {
        "passed": passed,
        "arm": arm,
        "pid": os.getpid(),
        "training_pid": trained["pid"],
        "max_abs_logit_difference": difference,
        "all_logits_equal": passed,
        "logit_shape": list(logits.shape),
        "seconds": time.monotonic() - began,
        "source_sha256": hashes,
        "scientific_updates": 0,
    }
    K.write(out / "reload.json", result)
    assert passed, result
    print(json.dumps(result), flush=True)


def eval_smoke(config, output, hashes, checkpoint):
    from llm_memory_editability.knowledge_method_data import context_user, render_chat

    began = time.monotonic()
    config = {
        **config,
        "n_cases": 2,
        "eval_batch_size": 1,
        "eval_max_new_tokens": 16,
        "retention_max_new_tokens": 16,
    }
    shared = output / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    training = [
        {**row, "case_id": f"engineering-case-{index}"}
        for index, row in enumerate(fixed_rows()[:2])
    ]
    (shared / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in training))
    K.seed_all(config["seed"])
    model, tok = K.load_model(config)
    rows = [
        {
            "id": "engineering/closed",
            "group": "closed",
            "gold": "Blue",
            "prompt": render_chat(tok, training[0]["question"]),
        },
        {
            "id": "engineering/context",
            "group": "context",
            "gold": "Blue",
            "prompt": render_chat(
                tok, context_user(training[0]["context"], training[0]["question"])
            ),
        },
        {
            "id": "engineering/retention",
            "group": "retention",
            "gold": "France",
            "type": "country",
            "exclude": [],
            "prompt": render_chat(tok, "In what country is Paris? Answer with the country name."),
        },
    ]
    before = K.predict(model, tok, rows, config)
    before_fit = K.fit_diagnostic(model, tok, config, output, "sft_down")
    model = K.configure_arm(model, config, "sft_down")
    K.restore(model, torch.load(checkpoint, weights_only=True, map_location="cpu"))
    after = K.predict(model, tok, rows, config)
    after_fit = K.fit_diagnostic(model, tok, config, output, "sft_down")
    assert len(before) == len(after) == 3
    assert all(row["token_ids"] and "em" in row and "f1" in row for row in before + after)
    assert "strict" in before[2] and "strict" in after[2]
    assert before_fit["summary"]["n"] == after_fit["summary"]["n"] == 2
    result = {
        "passed": True,
        "scientific_updates": 0,
        "engineering_updates": 0,
        "interpretation": "Generation/scoring/fit interface smoke only; not scientific accuracy",
        "source_sha256": hashes,
        "checkpoint_sha256": K.digest(checkpoint),
        "before": before,
        "after": after,
        "teacher_target_fit": {"before": before_fit, "after": after_fit},
        "config": config,
        "seconds": time.monotonic() - began,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    K.write(output / "eval-smoke.json", result)
    print(
        json.dumps({"passed": True, "seconds": result["seconds"], "scientific_updates": 0}),
        flush=True,
    )


def generation_smoke(config, output, hashes):
    from llm_memory_editability.knowledge_method_data import (
        _HFGenerator,
        context_user,
        question_user,
        read_source,
        render_chat,
    )

    generator = _HFGenerator({**config, "max_new_tokens": 64})
    cases = read_source(Path(config["source_csv"]), config["n_cases"])
    tok = generator.tokenizer
    ids = tok.encode(
        "Which engineering object has this property? " * 150, add_special_tokens=False
    )[:512]
    question = tok.decode(ids, skip_special_tokens=False)
    assert len(tok.encode(question, add_special_tokens=False)) == 512
    paragraphs = [
        (case["case_id"], i, paragraph)
        for case in cases
        for i, paragraph in enumerate(case["paragraphs"])
    ]
    case_id, paragraph_id, context = max(
        paragraphs,
        key=lambda item: len(
            tok.encode(render_chat(tok, context_user(item[2], question)), add_special_tokens=False)
        ),
    )
    results = {}
    for name, user, maximum in (
        ("short", question_user("The engineering cube is blue and the cone is red."), 64),
        ("longest", context_user(context, question), 512),
    ):
        generator.max_new_tokens = maximum
        torch.cuda.reset_peak_memory_stats()
        began = time.monotonic()
        generations = generator([user] * 32, config["question_temperature"], config["seed"])
        torch.cuda.synchronize()
        assert len(generations) == 32
        assert all(row["generated_tokens"] > 0 for row in generations)
        results[name] = {
            "batch_size": 32,
            "max_new_tokens": maximum,
            "generations": generations,
            "seconds": time.monotonic() - began,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "prompt": user,
            "truncated": sum(row["truncated"] for row in generations),
        }
        K.write(output / f"generation-{name}.json", results[name])
    result = {
        "passed": True,
        "scientific_updates": 0,
        "engineering_updates": 0,
        "source_sha256": hashes,
        "source_csv_sha256": K.digest(config["source_csv"]),
        "longest_context_case": case_id,
        "longest_context_paragraph": paragraph_id,
        "results": {
            name: {k: v for k, v in value.items() if k not in {"generations", "prompt"}}
            for name, value in results.items()
        },
        "interpretation": "Batch32 interface/memory check only; generated data must not be reused",
    }
    K.write(output / "generation-smoke.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "train", "reload", "eval", "generation"), default="all"
    )
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(warmup_steps=0, learning_rate=1e-5, batch_size=4)
    hashes = source_hashes(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "generation":
        generation_smoke(config, args.output, hashes)
        return
    if args.stage == "eval":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for eval smoke")
        eval_smoke(config, args.output, hashes, args.checkpoint)
        return
    if args.stage != "all":
        if args.arm is None:
            parser.error("--arm is required for a child stage")
        globals()[args.stage](config, args.output, args.arm, hashes)
        return
    if (args.output / "complete.json").exists():
        raise RuntimeError("Completed engineering batch must not be rerun")
    K.write(
        args.output / "started.json",
        {
            "source_sha256": hashes,
            "config": config,
            "parent_pid": os.getpid(),
            "scientific_updates": 0,
            "engineering_updates_planned": 8,
        },
    )
    try:
        results = {}
        for arm in ARMS:
            for stage in ("train", "reload"):
                command = [
                    sys.executable,
                    "-u",
                    __file__,
                    "--config",
                    str(args.config),
                    "--output",
                    str(args.output),
                    "--stage",
                    stage,
                    "--arm",
                    arm,
                ]
                with (args.output / f"{arm}-{stage}.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                result = json.loads((args.output / arm / f"{stage}.json").read_text())
                assert result["passed"]
                results[f"{arm}/{stage}"] = result
                K.write(args.output / "progress.json", {"completed_stages": list(results)})
        K.write(
            args.output / "complete.json",
            {
                "passed": True,
                "scientific_updates": 0,
                "engineering_updates": 8,
                "source_sha256": hashes,
                "results": results,
            },
        )
    except BaseException:
        K.write(
            args.output / "failed.json",
            {"traceback": traceback.format_exc(), "scientific_updates": 0},
        )
        raise


if __name__ == "__main__":
    main()
