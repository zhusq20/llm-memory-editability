"""PD/SFT crossed with four parameter scopes, preserving the frozen v1 runner.

Data preparation and evaluation are independent of this module. Dense PD uses
a separately loaded, frozen base teacher; LoRA PD disables its adapters while
reading the frozen base. Common encoding, losses, persistence and fit diagnostics
come from knowledge_method_diagnosis rather than a second copy of that script.
"""

from __future__ import annotations

import json
import math
import random
import time

import torch
import torch.nn.functional as F

from . import knowledge_method_diagnosis as shared

SCOPES = ("full", "lora", "down", "down_lora")
ARMS = tuple(f"{objective}_{scope}" for objective in ("pd", "sft") for scope in SCOPES)
load_model = shared.load_model
autocast = shared.autocast
write = shared.write
wandb_run = shared.wandb_run


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown factorial arm: {arm}")
    return tuple(arm.split("_", 1))


def parameter_contract(model, arm):
    """Check the actual trainable tensors, not merely the requested adapter name."""
    _, scope = arm_spec(arm)
    parameters = dict(model.named_parameters())
    names = [name for name, p in parameters.items() if p.requires_grad]
    if not names:
        raise ValueError("Arm has no trainable parameters")
    if scope == "full" and len(names) != len(parameters):
        raise ValueError("Full arm must train every base parameter")
    if scope == "down" and set(names) != {name for name in parameters if ".mlp.down_proj." in name}:
        raise ValueError("Down arm must train exactly all down_proj parameters")
    if scope in ("lora", "down_lora"):
        if any(".lora_" not in name for name in names):
            raise ValueError("LoRA teacher requires every base parameter to remain frozen")
        if scope == "down_lora" and any(".mlp.down_proj.lora_" not in name for name in names):
            raise ValueError("Down-LoRA includes an adapter outside down_proj")
    return {
        "trainable_names": names,
        "trainable_parameters": sum(parameters[name].numel() for name in names),
        "total_parameters": sum(p.numel() for p in parameters.values()),
    }


def configure_arm(model, config, arm):
    objective, scope = arm_spec(arm)
    if arm not in config["arms"]:
        raise ValueError(f"Arm absent from configuration: {arm}")
    if any(".lora_" in name for name, _ in model.named_parameters()):
        raise ValueError("Configure an unadapted base, not an existing LoRA model")
    canonical = f"{objective}_lora" if scope in ("lora", "down_lora") else f"sft_{scope}"
    adapted_config = {**config, "arms": [canonical]}
    if scope == "down_lora":
        adapted_config["lora_targets"] = ["down_proj"]
    model = shared.configure_arm(model, adapted_config, canonical)
    parameter_contract(model, arm)
    return model


def encode_sample(tok, row, arm, max_length):
    objective, _ = arm_spec(arm)
    return shared.encode_sample(tok, row, f"{objective}_lora", max_length)


def _validate_independent_teacher(student, teacher):
    if teacher is None or teacher is student:
        raise ValueError("Dense PD requires an independent frozen base teacher")
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Independent teacher contains trainable parameters")
    student_storage = {
        (p.device, p.untyped_storage().data_ptr()) for p in student.parameters() if p.numel()
    }
    if any(
        (p.device, p.untyped_storage().data_ptr()) in student_storage
        for p in teacher.parameters()
        if p.numel()
    ):
        raise ValueError("Teacher and student share parameter storage")


def load_teacher(student, config, arm):
    """Reload theta_0, never snapshot an already trained or resumed student."""
    objective, scope = arm_spec(arm)
    if objective == "sft" or scope in ("lora", "down_lora"):
        return None
    teacher, _ = load_model(config)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.config.use_cache = False
    _validate_independent_teacher(student, teacher)
    return teacher


def batch_loss(student, batch, arm, config, teacher=None):
    objective, scope = arm_spec(arm)
    s = batch["student"]
    if objective == "pd":
        t = batch["teacher"]
        if scope in ("full", "down"):
            _validate_independent_teacher(student, teacher)
            teacher.eval()
            with torch.no_grad(), autocast():
                targets = (
                    teacher(
                        input_ids=t["input_ids"],
                        attention_mask=t["attention_mask"],
                        use_cache=False,
                    )
                    .logits[t["answer_mask"]]
                    .detach()
                )
        else:
            if teacher is not None:
                raise ValueError("LoRA PD uses its adapter-disabled frozen base teacher")
            parameter_contract(student, arm)
            was_training = student.training
            try:
                with student.disable_adapter(), torch.no_grad(), autocast():
                    student.eval()
                    targets = (
                        student(
                            input_ids=t["input_ids"],
                            attention_mask=t["attention_mask"],
                            use_cache=False,
                        )
                        .logits[t["answer_mask"]]
                        .detach()
                    )
            finally:
                student.train(was_training)
    elif teacher is not None:
        raise ValueError("SFT does not use a teacher forward pass")
    with autocast():
        logits = student(
            input_ids=s["input_ids"], attention_mask=s["attention_mask"], use_cache=False
        ).logits[s["answer_mask"]]
    if objective == "pd":
        return shared.forward_kl(logits, targets, s["answer_mask"].sum(1), config["kl_temperature"])
    return F.cross_entropy(logits.float(), s["labels"][s["answer_mask"]])


def fit_diagnostic(model, tok, config, batch, arm):
    objective, _ = arm_spec(arm)
    result = shared.fit_diagnostic(model, tok, config, batch, f"{objective}_lora")
    return {**result, "arm": arm}


def train(config, batch, arm, smoke_steps=None):
    """Train the supplied data unit; this function does not choose or pool cases."""
    objective, scope = arm_spec(arm)
    out = batch / "runs" / arm
    if (out / "complete.json").exists():
        raise RuntimeError("Completed training must not be restarted")
    if not (batch / "shared/prepare-complete.json").exists():
        raise RuntimeError("Preparation has not passed its coverage checks")
    shared.seed_all(config["seed"])
    model, tok = load_model(config)
    model = configure_arm(model, config, arm)
    teacher = load_teacher(model, config, arm)
    source = batch / "shared/train.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    if not rows or (smoke_steps is None and len(rows) != config["expected_training_rows"]):
        raise ValueError("Training coverage differs from the supplied data contract")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(rows) / config["batch_size"])
    total = steps_per_epoch * config["epochs"]
    if smoke_steps is not None:
        if smoke_steps < 1:
            raise ValueError("Smoke steps must be positive")
        total = min(total, smoke_steps)
    teacher_mode = (
        "none"
        if objective == "sft"
        else "independent_frozen_base"
        if teacher is not None
        else "adapter_disabled_frozen_base"
    )
    manifest = {
        "arm": arm,
        "objective": objective,
        "scope": scope,
        "rows": len(rows),
        "steps": total,
        **parameter_contract(model, arm),
        "data_sha256": shared.digest(source),
        "config": config,
        "teacher_mode": teacher_mode,
        "teacher_model_revision": config.get("model_revision") if objective == "pd" else None,
        "independent_teacher_parameters": 0
        if teacher is None
        else sum(p.numel() for p in teacher.parameters()),
    }
    write(out / "manifest.json", manifest)
    checkpoint = out / "resume.pt"
    start = exposure = answer_tokens = forward_tokens = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            saved["data_sha256"] != manifest["data_sha256"]
            or saved["config"] != config
            or saved["arm"] != arm
        ):
            raise RuntimeError("Resume data/config/arm mismatch")
        shared.restore(model, saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        shared.restore_rng(saved["rng"])
        start, exposure = saved["step"], saved["exposure"]
        answer_tokens, forward_tokens = saved["answer_tokens"], saved["forward_tokens"]
    run = None if smoke_steps is not None else wandb_run(config, batch, "train", arm)
    began = time.time()
    record = {
        "step": start,
        "exposure": exposure,
        "answer_tokens": answer_tokens,
        "forward_tokens": forward_tokens,
        "resumed_at_terminal_step": start == total,
    }
    order, current_epoch = [], None
    try:
        model.train()
        for step in range(start, total):
            epoch, index = divmod(step, steps_per_epoch)
            if epoch != current_epoch:
                order = list(range(len(rows)))
                random.Random(config["seed"] + epoch).shuffle(order)
                current_epoch = epoch
            chosen = order[index * config["batch_size"] : (index + 1) * config["batch_size"]]
            # Keep only this minibatch's token lists; large corpora need not be expanded
            # into hundreds of thousands of duplicated context/answer token arrays.
            encoded = [
                encode_sample(tok, rows[i], arm, config["max_sequence_tokens"]) for i in chosen
            ]
            batch_t = shared.collate(encoded, tok.pad_token_id)
            warmup = config["warmup_steps"]
            scale = (
                step / max(1, warmup) if step < warmup else (total - step) / max(1, total - warmup)
            )
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * scale
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch_t, arm, config, teacher)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, config["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            exposure += len(chosen)
            answer_tokens += int(batch_t["student"]["answer_mask"].sum())
            forward_tokens += int(batch_t["student"]["attention_mask"].sum())
            if objective == "pd":
                forward_tokens += int(batch_t["teacher"]["attention_mask"].sum())
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "grad_norm": float(norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "exposure": exposure,
                "answer_tokens": answer_tokens,
                "forward_tokens": forward_tokens,
                "seconds_this_process": time.time() - began,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            if (step + 1) % config["log_steps"] == 0 or step + 1 == total:
                write(out / "progress.json", record)
                with (out / "curve.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                if run:
                    run.log(record, step=step + 1)
            if (step + 1) % config["checkpoint_steps"] == 0 or step + 1 == total:
                shared.atomic_torch_save(
                    {
                        "weights": shared.state(model),
                        "optimizer": optimizer.state_dict(),
                        "rng": shared.rng_state(),
                        "step": step + 1,
                        "exposure": exposure,
                        "answer_tokens": answer_tokens,
                        "forward_tokens": forward_tokens,
                        "config": config,
                        "data_sha256": manifest["data_sha256"],
                        "arm": arm,
                    },
                    checkpoint,
                )
        shared.atomic_torch_save(shared.state(model), out / "weights.pt")
        model.eval()
        audit_samples = [
            encode_sample(tok, row, arm, config["max_sequence_tokens"]) for row in rows[:2]
        ]
        audit_inputs = shared.collate(audit_samples, tok.pad_token_id)["student"]
        with torch.no_grad(), autocast():
            logits = model(
                input_ids=audit_inputs["input_ids"],
                attention_mask=audit_inputs["attention_mask"],
                use_cache=False,
            ).logits
        shared.atomic_torch_save(
            {
                "input_ids": audit_inputs["input_ids"].cpu(),
                "attention_mask": audit_inputs["attention_mask"].cpu(),
                "greedy": logits.argmax(-1).cpu(),
                "logits": logits[:, -1, :1024].float().cpu(),
            },
            out / "reload-reference.pt",
        )
        write(
            out / "complete.json",
            {
                **record,
                "weights_sha256": shared.digest(out / "weights.pt"),
                "smoke": smoke_steps is not None,
                "teacher_mode": teacher_mode,
            },
        )
    finally:
        if run:
            run.finish()


def audit(config, batch, arm):
    out = batch / "runs" / arm
    model, _ = load_model(config)
    model = configure_arm(model, config, arm)
    shared.restore(model, torch.load(out / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    reference = torch.load(out / "reload-reference.pt", map_location="cpu", weights_only=True)
    device = next(model.parameters()).device
    with torch.no_grad(), autocast():
        logits = model(
            input_ids=reference["input_ids"].to(device),
            attention_mask=reference["attention_mask"].to(device),
            use_cache=False,
        ).logits
    maximum = (logits[:, -1, :1024].float().cpu() - reference["logits"]).abs().max().item()
    greedy_equal = torch.equal(logits.argmax(-1).cpu(), reference["greedy"])
    complete = json.loads((out / "complete.json").read_text())
    hash_equal = shared.digest(out / "weights.pt") == complete["weights_sha256"]
    passed = maximum == 0 and greedy_equal and hash_equal
    write(
        out / "audit/audit.json",
        {
            "passed": passed,
            "max_abs_logit_difference": maximum,
            "all_token_greedy_equal": greedy_equal,
            "weights_hash_equal": hash_equal,
            "scope": "fresh process reload; two fixed training sequences",
        },
    )
    if not passed:
        raise RuntimeError("Independent reload audit failed")
