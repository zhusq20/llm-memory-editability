"""Run an indexed-context reconstruction using the original Open-Instruct trainer."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

from .perk_data import sha256_file

OPEN_INSTRUCT_REVISION = "5595872b67513add2facdf866d3da7d594e07ba2"
OPEN_INSTRUCT_FINETUNE_SHA256 = "30cbced37c50aad7ed31e4f9df939fdeb7f40cf6bfd94811149c4d74321c582d"


def interpolate_absolute_positions(weight, new_length: int):
    """Karypis et al. NAACL 2024 §3.2, integer expansion; last row fills the tail."""
    import torch

    old_length = len(weight)
    if new_length < old_length or new_length % old_length:
        raise ValueError("Position extension must be an integral multiple of the native table")
    factor = new_length // old_length
    positions = torch.arange(new_length, device=weight.device)
    lower = positions // factor
    upper = (lower + 1).clamp(max=old_length - 1)
    remainder = (positions % factor).to(weight.dtype).unsqueeze(1) / factor
    return weight[lower] * (1 - remainder) + weight[upper] * remainder


def extend_gpt2_positions(model, new_length: int) -> None:
    import torch

    old_weight = model.transformer.wpe.weight
    new_weight = interpolate_absolute_positions(old_weight.detach(), new_length)
    model.transformer.wpe = torch.nn.Embedding.from_pretrained(
        new_weight, freeze=not old_weight.requires_grad
    )
    model.config.n_positions = new_length
    model.config.n_ctx = new_length
    # GPT2's eager implementation also has an explicit causal-mask buffer.
    mask = torch.ones((new_length, new_length), dtype=torch.bool, device=old_weight.device)
    mask.tril_()
    mask = mask[None, None]
    for block in model.transformer.h:
        block.attn.register_buffer("bias", mask, persistent=False)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_data(config: dict) -> dict:
    manifest = json.loads(Path(config["data_manifest"]).read_text())
    for split in ("train", "validation", "test"):
        entry = manifest["outputs"][split]["fticr"]
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Changed frozen baseline data: {split}")
    if any(manifest["exact_sample_overlap"].values()):
        raise ValueError("Split isolation check failed")
    maximum = max(
        manifest["audits"][split]["measurements"]["fticr_tokens"]["max"]
        for split in ("train", "validation", "test")
    )
    if config["max_seq_length"] < maximum:
        raise ValueError("Baseline sequence cap would truncate the published data or target")
    if config["max_position_embeddings"] < config["max_seq_length"] + 512:
        raise ValueError(
            "Position table must also preserve the paper's 512 generation-token budget"
        )
    return manifest


def _load_upstream(source_dir: str | Path):
    path = Path(source_dir) / "open_instruct" / "finetune.py"
    if sha256_file(path) != OPEN_INSTRUCT_FINETUNE_SHA256:
        raise ValueError("The original Open-Instruct trainer source has changed")
    spec = importlib.util.spec_from_file_location("perk_author_open_instruct", path)
    module = importlib.util.module_from_spec(spec)
    # Dataset.map's worker processes unpickle the official encoder by module
    # name. Register this dynamically loaded module before executing its source.
    sys.modules[spec.name] = module
    # The optional OLMo warning uses Accelerate's logger before its state exists.
    # Deleting only this warning permits the published GPT2 path to import.
    code = path.read_text().replace(
        'logger.warning("OLMo not installed. Ignore if using a different model.")',
        "pass  # optional OLMo is unused in the GPT2 reconstruction",
    )
    exec(compile(code, str(path), "exec"), module.__dict__)
    # The author's optional-import except branch leaves this symbol unbound.
    if not hasattr(module, "OLMoTokenizerFast"):
        module.OLMoTokenizerFast = type("UnusedOLMoTokenizer", (), {})
    return module


def validation_loss(
    module, model, tokenizer, rows: list[dict], max_seq_length: int, *, counters=None
) -> float:
    import torch

    was_training = model.training
    model.eval()
    losses = []
    start = time.monotonic()
    device = next(model.parameters()).device
    with torch.no_grad():
        for row in rows:
            batch = module.encode_with_prompt_completion_format(row, tokenizer, max_seq_length)
            batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
            if counters is not None:
                counters["validation_samples_exposed"] += len(batch["input_ids"])
                counters["validation_input_tokens"] += int(batch["attention_mask"].sum())
                counters["validation_supervised_tokens"] += int(
                    (batch["labels"][:, 1:] != -100).sum()
                )
            losses.append(float(model(**batch, use_cache=False).loss))
    model.train(was_training)
    result = sum(losses) / len(losses)
    if not math.isfinite(result):
        raise ValueError("Nonfinite frozen validation loss")
    if counters is not None:
        counters["validation_seconds"] += time.monotonic() - start
    return result


class ValidationEarlyStop(Exception):
    """Stop the original loop after a checkpoint has been safely saved."""


def train(config: dict, *, source_dir: str | Path, output: str | Path) -> dict:
    """Call the public Open-Instruct loop; adapters do not change its loss or optimizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = check_data(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    module = _load_upstream(source_dir)
    state = {
        "best_loss": math.inf,
        "best_checkpoint": None,
        "bad_checks": 0,
        "checks": [],
        "optimizer_steps": 0,
        "train_samples_exposed": 0,
        "train_input_tokens": 0,
        "train_supervised_tokens": 0,
        "validation_samples_exposed": 0,
        "validation_input_tokens": 0,
        "validation_supervised_tokens": 0,
        "validation_seconds": 0,
    }
    started = time.monotonic()
    original_optimizer = module.torch.optim.AdamW

    def write_progress():
        state["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "training-progress.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "training-progress.json")

    class CountingAdamW(original_optimizer):
        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            state["optimizer_steps"] += 1
            write_progress()
            return result

    module.torch.optim.AdamW = CountingAdamW
    original_model_loader = AutoModelForCausalLM.from_pretrained
    original_tokenizer_loader = AutoTokenizer.from_pretrained

    def model_loader(*args, **kwargs):
        # Replace the deprecated flag used by this original trainer. SDPA computes
        # the same causal attention and supports GPT2 without an external kernel.
        kwargs.pop("use_flash_attention_2", None)
        kwargs["attn_implementation"] = "sdpa"
        model = original_model_loader(*args, **kwargs)
        state["native_model_parameters"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        extend_gpt2_positions(model, config["max_position_embeddings"])
        state["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        state["trainable_parameters"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

        def count_exposure(inner_model, args, kwargs):
            if not inner_model.training or kwargs.get("labels") is None:
                return
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            state["train_samples_exposed"] += len(input_ids)
            mask = kwargs.get("attention_mask")
            state["train_input_tokens"] += (
                int(mask.sum()) if mask is not None else input_ids.numel()
            )
            state["train_supervised_tokens"] += int((kwargs["labels"][:, 1:] != -100).sum())

        model.register_forward_pre_hook(count_exposure, with_kwargs=True)
        return model

    def tokenizer_loader(*args, **kwargs):
        tokenizer = original_tokenizer_loader(*args, **kwargs)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = config["max_position_embeddings"]
        return tokenizer

    # Proxy classes keep these changes local to the imported official trainer.
    module.AutoModelForCausalLM = type(
        "GPT2Loader", (), {"from_pretrained": staticmethod(model_loader)}
    )
    module.AutoTokenizer = type(
        "GPT2TokenizerLoader", (), {"from_pretrained": staticmethod(tokenizer_loader)}
    )
    original_save = module.save_with_accelerate
    validation_rows = load_jsonl(manifest["outputs"]["validation"]["fticr"]["path"])
    validation_limit = config.get("smoke_validation_limit", config.get("validation_limit"))
    if validation_limit is not None:
        if config.get("smoke_max_train_steps") is None:
            raise ValueError(
                "Validation limits are permitted only in an explicit development smoke"
            )
        if validation_limit <= 0:
            raise ValueError("Validation limit must be positive")
        validation_rows = validation_rows[:validation_limit]
    state["validation_rows_per_check"] = len(validation_rows)
    state["validation_expected_rows"] = manifest["audits"]["validation"]["rows"]

    def save_and_validate(accelerator, model, tokenizer, output_dir, args):
        original_save(accelerator, model, tokenizer, output_dir, args)
        if not accelerator.is_main_process or Path(output_dir) == output:
            return
        tokenizer.save_pretrained(output_dir)
        loss = validation_loss(
            module,
            accelerator.unwrap_model(model),
            tokenizer,
            validation_rows,
            args.max_seq_length,
            counters=state,
        )
        name = Path(output_dir).name
        step = int(name.removeprefix("step_")) if name.startswith("step_") else None
        state["checks"].append({"checkpoint": str(output_dir), "step": step, "val_loss": loss})
        if loss < state["best_loss"]:
            state.update(best_loss=loss, best_checkpoint=str(output_dir), bad_checks=0)
        else:
            state["bad_checks"] += 1
        (output / "validation-state.json").write_text(json.dumps(state, indent=2) + "\n")
        write_progress()
        if args.with_tracking:
            accelerator.log({"val_loss": loss}, step=step)
        if state["bad_checks"] >= config.get("patience", 5):
            raise ValidationEarlyStop

    module.save_with_accelerate = save_and_validate
    original_trackers = module.Accelerator.init_trackers
    accelerator_box = {}
    run_configuration = config
    default_run_id = (
        "perk-paper-reproduction-v1-fticr-development-smoke"
        if config.get("smoke_max_train_steps") is not None
        else "perk-paper-reproduction-v1-fticr-train"
    )

    def init_trackers(accelerator, project_name, config=None, init_kwargs=None):
        accelerator_box["accelerator"] = accelerator
        tracker_kwargs = dict(init_kwargs or {})
        tracker_kwargs["wandb"] = dict(tracker_kwargs.get("wandb", {}))
        tracker_kwargs["wandb"].update(
            entity="zhusq20",
            name=run_configuration.get("wandb_name", default_run_id),
            id=run_configuration.get("wandb_run_id", default_run_id),
            resume="allow",
        )
        return original_trackers(
            accelerator, "llm-memory-editability", config=config, init_kwargs=tracker_kwargs
        )

    module.Accelerator.init_trackers = init_trackers
    arguments = [
        "open-instruct-author-finetune",
        "--model_name_or_path",
        config["model_name_or_path"],
        "--tokenizer_name",
        config.get("tokenizer_name_or_path", config["model_name_or_path"]),
        "--train_file",
        manifest["outputs"]["train"]["fticr"]["path"],
        "--output_dir",
        str(output),
        "--max_seq_length",
        str(config["max_seq_length"]),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(config.get("gradient_accumulation_steps", 1)),
        "--learning_rate",
        str(config.get("learning_rate", 1e-5)),
        "--weight_decay",
        str(config.get("weight_decay", 0.01)),
        "--num_train_epochs",
        "2",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.03",
        "--checkpointing_steps",
        str(config.get("validation_interval", 500)),
        "--logging_steps",
        "25",
        "--preprocessing_num_workers",
        str(config.get("preprocessing_num_workers", 4)),
        "--seed",
        str(config.get("seed", 42)),
        "--gradient_checkpointing",
        "--with_tracking",
        "--report_to",
        "wandb",
        "--wandb_entity",
        "zhusq20",
    ]
    if config.get("smoke_max_train_steps") is not None:
        arguments += ["--max_train_steps", str(config["smoke_max_train_steps"])]
    previous_argv = sys.argv
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "author-command.json").write_text(json.dumps(arguments, indent=2) + "\n")
    (output / "execution-adaptations.json").write_text(
        json.dumps(
            {
                "trainer_revision": OPEN_INSTRUCT_REVISION,
                "trainer_sha256": OPEN_INSTRUCT_FINETUNE_SHA256,
                "changes": [
                    "register the official encoder module for multiprocessing deserialization",
                    "suppress unused optional OLMo warning and provide an unused type guard",
                    "replace obsolete use_flash_attention_2 flag with GPT2 SDPA selection",
                    "apply Karypis integer APE interpolation and extend the causal mask",
                    "pad with GPT2 EOS without adding vocabulary entries",
                    "evaluate train-derived validation at checkpoint save hooks; preserve best",
                    "stop after the declared number of nonimproving validation checks",
                    "use the project's authorized W&B project/entity",
                ],
                "unchanged": (
                    "official prompt/completion encoder, loss, optimizer and training loop"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    try:
        sys.argv = arguments
        module.main()
        state["early_stopped"] = False
    except ValidationEarlyStop:
        state["early_stopped"] = True
    finally:
        sys.argv = previous_argv
        module.Accelerator.init_trackers = original_trackers
        module.torch.optim.AdamW = original_optimizer
        if "accelerator" in accelerator_box and state.get("early_stopped"):
            accelerator_box["accelerator"].end_training()
    if state["best_checkpoint"] is None:
        state["best_checkpoint"] = str(output)
    (output / "completion.json").write_text(json.dumps(state, indent=2) + "\n")
    write_progress()
    return state


def evaluate(config: dict, *, author_perk_source: str | Path, output: str | Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = check_data(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(manifest["outputs"]["test"]["fticr"]["path"])
    if config.get("smoke_eval_limit") is not None:
        if config.get("smoke_max_train_steps") is None:
            raise ValueError(
                "Evaluation limits are permitted only in an explicit development smoke"
            )
        if config["smoke_eval_limit"] <= 0:
            raise ValueError("Evaluation limit must be positive")
        rows = rows[: config["smoke_eval_limit"]]
    tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["checkpoint"], local_files_only=True, attn_implementation="sdpa"
    ).to(config.get("device", "cuda"))
    model.eval()
    sys.path.insert(0, str(author_perk_source))
    from perk.evaluate import evaluate as author_evaluate

    records = []
    test_input_tokens = 0
    generated_tokens = 0
    start = time.monotonic()
    predictions_path = output / "test-generations.jsonl"
    with predictions_path.open("x") as stream, torch.no_grad():
        for row in rows:
            inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
            test_input_tokens += int(inputs["attention_mask"].sum())
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            answer = tokenizer.decode(
                generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            record = {
                "guid": row["source_id"],
                "prompt": row["prompt"],
                "output": answer,
                "answer": row["completion"],
                "generated_token_ids": generated[0, inputs.input_ids.shape[1] :].tolist(),
            }
            records.append(record)
            generated_tokens += len(record["generated_token_ids"])
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    metrics, _ = author_evaluate(records)
    metrics["evaluated_rows"] = len(records)
    metrics["expected_rows"] = len(load_jsonl(manifest["outputs"]["test"]["fticr"]["path"]))
    metrics["test_input_tokens"] = test_input_tokens
    metrics["generated_tokens"] = generated_tokens
    metrics["elapsed_seconds"] = time.monotonic() - start
    (output / "test-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
