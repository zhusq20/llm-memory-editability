"""Run the published PERK implementation with recorded execution repairs.

This adapter deliberately imports the author's CausalLM, meta-training step,
learned inner rates, and vendored Higher. It is a reconstruction of the Figure 3
conditions, not a claim that unreported training settings have been recovered.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

AUTHOR_COMMIT = "9b1a918368b935d633691839cb360318ff2045f2"
AUTHOR_URL = "https://github.com/eric11eca/perk"
SOURCE_FILES = (
    "perk/model.py",
    "perk/dataset.py",
    "perk/modules/meta_module.py",
    "perk/modules/base_module.py",
    "perk/lslr_schedular.py",
    "perk/evaluate.py",
    "perk/inference.py",
    "perk/utils/data_utils.py",
    "higher/__init__.py",
    "higher/patch.py",
    "higher/optim.py",
    "higher/utils.py",
)
EXECUTION_REPAIRS = {
    "unused_ademamix_import": (
        "Remove the unavailable higher.ademamix import and its unused optimizer registration "
        "in a separate source copy; AdamW implementation is byte-for-byte unchanged."
    ),
    "missing_optimizer_import": "Alias perk.optimizer to the supplied perk.lslr_schedular.",
    "training_subset": "Prevent do_eval from replacing the training pool with its first row.",
    "evaluation_pool": "Use frozen files once each; disable random.choices resampling.",
    "missing_runner_wandb": "Construct the author module and Lightning trainer directly.",
    "silent_oom_skip": "Propagate training exceptions; no omitted or retried examples.",
    "test_epoch_typo": "Run author validation_step and evaluate directly for frozen test rows.",
    "gpt2_packing": "Use the author's unpacked branch; GPT-2 rejects Qwen varlen kwargs.",
    "second_order_attention": "Use eager attention for the required higher-order derivatives.",
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)


def source_manifest(source_dir: str | Path) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != AUTHOR_COMMIT:
        raise ValueError(f"Author source must be {AUTHOR_COMMIT}; found {commit}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("Tracked author source was changed; retain it and use this adapter")
    return {
        "url": AUTHOR_URL,
        "commit": commit,
        "files": {name: sha256(source / name) for name in SOURCE_FILES},
        "execution_repairs": EXECUTION_REPAIRS,
    }


def load_author(source_dir: str | Path):
    """Load only the frozen author implementation, including its modified Higher."""
    source = Path(source_dir).resolve()
    source_manifest(source)
    runtime_source = source.parent / "perk-execution-repaired"
    if not runtime_source.exists():
        shutil.copytree(
            source, runtime_source, ignore=shutil.ignore_patterns(".git", "__pycache__")
        )
        optimizer_file = runtime_source / "higher/optim.py"
        text = optimizer_file.read_text()
        for unused in (
            "from . import ademamix as _ademamix\n",
            "    _ademamix.AdEMAMix: DifferentiableAdEMAMix,\n",
        ):
            if text.count(unused) != 1:
                raise RuntimeError("The frozen unused optimizer import repair no longer matches")
            text = text.replace(unused, "")
        optimizer_file.write_text(text)
        write_json(
            runtime_source / "execution-repair-manifest.json",
            {
                "original_sha256": sha256(source / "higher/optim.py"),
                "repaired_sha256": sha256(optimizer_file),
                "removed_lines": [
                    "from . import ademamix as _ademamix",
                    "_ademamix.AdEMAMix: DifferentiableAdEMAMix,",
                ],
            },
        )
    expected = (
        (source / "higher/optim.py")
        .read_text()
        .replace("from . import ademamix as _ademamix\n", "")
        .replace("    _ademamix.AdEMAMix: DifferentiableAdEMAMix,\n", "")
    )
    if (runtime_source / "higher/optim.py").read_text() != expected:
        raise RuntimeError("Execution source copy does not match the declared two-line repair")
    for filename in SOURCE_FILES:
        if filename != "higher/optim.py" and sha256(runtime_source / filename) != sha256(
            source / filename
        ):
            raise RuntimeError(f"Undeclared execution source change: {filename}")
    for package in ("higher", "perk"):
        existing = sys.modules.get(package)
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(
            runtime_source
        ):
            raise RuntimeError(f"{package} was loaded from a different implementation")
    if str(runtime_source) not in sys.path:
        sys.path.insert(0, str(runtime_source))
    sys.modules["perk.optimizer"] = importlib.import_module("perk.lslr_schedular")
    return importlib.import_module("perk.modules.meta_module")


def resolve_config(values: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Explicit reconstructed fields are saved alongside their source provenance."""
    config = {
        "reproduction_kind": "method_reconstruction",
        "paper_target": "Figure 3, GPT-2 / BabiLong QA2, 8K train and test",
        "model_name_or_path": "gpt2",
        "dataset_name": "babilong",
        "dataset_subset": "qa2_8k",
        "do_train": True,
        "do_eval": False,
        "peft_lora": True,
        "lora_rank": 256,
        "lora_alpha": 16,
        "weighted_loss": True,
        "weight_normalize": False,
        "weight_non_linearity": "softplus",
        "attn_implementation": "eager",
        "packing": False,
        "n_inner_iter": 4,
        "dyna_lr": True,
        "inner_lr": 5e-5,
        "inner_funct": "truncated",
        "unroll_start": 3,
        "first_order": False,
        "inner_grad_accum": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "warmup_proportion": 0.03,
        "max_grad_norm": 1.0,
        "num_train_epochs": 2,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_workers": 4,
        "n_gpu": 1,
        "bf16": True,
        "use_liger_kernel": False,
        "max_eval_data": 0,
        "callback_monitor": "val_loss",
        "patience": 10,
        "val_check_interval": 0.5,
        "seed": 42,
        "wandb_entity": "zhusq20",
        "wandb_project": "llm-memory-editability",
        "wandb_name": "perk-figure3-gpt2-qa2-8k-reconstruction",
        "max_new_tokens": 512,
    }
    config.update(values)
    for key in config:
        if any(secret in key.lower() for secret in ("api_key", "password", "secret")):
            raise ValueError("Credentials must be supplied through the environment")
    if config["reproduction_kind"] not in ("method_reconstruction", "development_smoke"):
        raise ValueError("Unreported settings cannot be labelled exact paper replication")
    if config["train_batch_size"] != 1 or config["eval_batch_size"] != 1:
        raise ValueError("Author unroll collator supports one reasoning problem per batch")
    if config["packing"] or config["attn_implementation"] != "eager":
        raise ValueError("GPT-2 reconstruction needs independent chunks and eager attention")
    if config["n_gpu"] != 1 or config["gradient_accumulation_steps"] != 1:
        raise ValueError("This reconstruction preserves the author's one-problem optimizer step")
    if not config.get("data_dir"):
        raise ValueError("A frozen author-format data_dir is required")
    output = Path(output).resolve()
    config.update(
        output_dir=str(output),
        run_dir=str(output),
        gen_dir=str(output / "outputs"),
        eval_dir=str(output / "evals"),
    )
    return config


def dataset_hashes(config: dict[str, Any]) -> dict[str, Any]:
    folder = Path(config["data_dir"]) / config["dataset_name"] / config["dataset_subset"]
    result = {}
    for split in ("train", "validation", "test"):
        path = folder / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty {split} dataset")
        if any(not isinstance(row["context"], list) or not row["context"] for row in rows):
            raise ValueError("Author PERK expects a nonempty list of document subsequences")
        result[split] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
    return result


def runtime_class(source_dir: str | Path):
    author = load_author(source_dir)
    torch = importlib.import_module("torch")
    dataset_module = importlib.import_module("perk.dataset")
    OmegaConf = importlib.import_module("omegaconf").OmegaConf

    class FrozenDataset(dataset_module.MetaKnowledgeDataset):
        def __init__(self, args, tokenizer, data_path, data_type, is_training):
            # Only bypass the two public debug/sampling branches. All reading,
            # prompts, labels, support supervision, and collation remain original.
            copied = OmegaConf.create(dict(args))
            copied.do_eval = False
            copied.max_eval_data = 0
            super().__init__(copied, tokenizer, data_path, data_type, is_training)
            self.args = args

    class ReconstructedPERK(author.MetaLMModule):
        def load_dataset(self):
            for name, split, training in (
                ("train_data", "train", True),
                ("dev_data", "validation", False),
                ("test_data", "test", False),
            ):
                setattr(
                    self,
                    name,
                    FrozenDataset(
                        self.hparams, self.tokenizer, self.hparams.data_dir, split, training
                    ),
                )
            self.exposure = {
                "problems": 0,
                "inner_subsequence_presentations": 0,
                "inner_input_tokens": 0,
                "inner_supervised_tokens": 0,
                "inner_tensor_tokens_including_padding": 0,
                "outer_input_tokens": 0,
                "outer_supervised_tokens": 0,
                "outer_tensor_tokens_including_padding": 0,
            }
            self.validation_exposure = dict.fromkeys(self.exposure, 0)

        def count_exposure(self, counter, batch):
            counter["problems"] += 1
            count = self.hparams.n_inner_iter
            counter["inner_subsequence_presentations"] += len(batch["train_input_ids"]) * count
            for side, prefix, multiplier in (("inner", "train_", count), ("outer", "", 1)):
                counter[f"{side}_tensor_tokens_including_padding"] += (
                    batch[f"{prefix}input_ids"].numel() * multiplier
                )
                counter[f"{side}_input_tokens"] += (
                    int(batch[f"{prefix}attention_mask"].sum()) * multiplier
                )
                counter[f"{side}_supervised_tokens"] += (
                    int((batch[f"{prefix}labels"][:, 1:] != -100).sum()) * multiplier
                )

        def training_step(self, batch, batch_idx):
            # Author step, including differentiable AdamW, LSLR and truncated GU.
            # Exceptions must stop the run instead of silently dropping samples.
            output = self.step(batch)
            for metric in ("inner_loss", "outer_loss"):
                self.log(f"batch_{metric}", output[metric], on_step=True, prog_bar=True)
            self.global_trainin_step += 1
            self.count_exposure(self.exposure, batch)
            write_json(
                Path(self.hparams.run_dir) / "training-progress.json",
                {
                    "completed_outer_updates": self.global_trainin_step,
                    "exposure": self.exposure,
                    "inner_loss": float(output["inner_loss"]),
                    "outer_loss": float(output["outer_loss"]),
                    "timestamp_unix": time.time(),
                },
            )
            return output

        def validation_step(self, batch, batch_idx):
            output = super().validation_step(batch, batch_idx)
            self.count_exposure(self.validation_exposure, batch)
            return output

        def validation_inference(self, records, model):
            # The same author text-generation pipeline, EOS and greedy decoding.
            generator = author.LLM_Generator(model.lm_model, model.tokenizer, self.device)
            original_generate = model.lm_model.generate
            captured = []

            def capture_generate(*args, **kwargs):
                result = original_generate(*args, **kwargs)
                inputs = kwargs.get("input_ids", args[0] if args else None)
                if inputs is None or not torch.is_tensor(result) or len(result) != 1:
                    raise RuntimeError("Cannot account for the author pipeline's generated tokens")
                captured.append(
                    {
                        "prompt_tokens": inputs.shape[1],
                        "generated_token_ids": result[0, inputs.shape[1] :].tolist(),
                    }
                )
                return result

            model.lm_model.generate = capture_generate
            try:
                generator.generate(
                    records,
                    pad_token_id=model.tokenizer.pad_token_id,
                    temperature=0.0,
                    do_sample=False,
                    top_p=1.0,
                    max_new_tokens=self.hparams.max_new_tokens,
                )
            finally:
                model.lm_model.generate = original_generate
            if len(captured) != len(records):
                raise RuntimeError("Generation tracing must cover each author query exactly once")
            for record, token_record in zip(records, captured, strict=False):
                record.update(token_record)

        def on_save_checkpoint(self, checkpoint):
            checkpoint["reconstruction_exposure"] = copy.deepcopy(self.exposure)
            checkpoint["reconstruction_validation_exposure"] = copy.deepcopy(
                self.validation_exposure
            )

        def on_load_checkpoint(self, checkpoint):
            if "reconstruction_exposure" in checkpoint:
                self.exposure = copy.deepcopy(checkpoint["reconstruction_exposure"])
            if "reconstruction_validation_exposure" in checkpoint:
                self.validation_exposure = copy.deepcopy(
                    checkpoint["reconstruction_validation_exposure"]
                )

        def on_validation_epoch_end(self):
            # Keep the author evaluation/writer. Optional CSV logging in smoke
            # has no log_table; scientific training uses WandbLogger as published.
            if not hasattr(self.logger, "log_table"):
                outputs = self.validation_step_outputs
                if outputs:
                    val_loss, records = self.validation_epoch_callback(outputs)
                    self.log("val_loss", val_loss.to(self.device), on_epoch=True)
                    metrics = self.validation_eval(records)
                    for key, value in metrics.items():
                        self.log(f"val_{key}", torch.tensor(value).to(self.device), on_epoch=True)
                    self.validation_step_outputs.clear()
            else:
                super().on_validation_epoch_end()

    return ReconstructedPERK


def parameter_digest(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def environment_manifest() -> dict[str, str]:
    result = {}
    for package in (
        "torch",
        "transformers",
        "peft",
        "lightning",
        "pytorch-lightning",
        "tokenizers",
        "numpy",
        "datasets",
        "hydra-core",
        "omegaconf",
        "wandb",
        "liger-kernel",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "absent"
    return result


def seed_all(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def train(
    config: dict[str, Any],
    source_dir: str | Path,
    *,
    device: str = "cuda",
    max_steps: int = -1,
    limit_validation: int | None = None,
    wandb: bool = True,
    resume: str | None = None,
) -> dict[str, Any]:
    import lightning as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
    from omegaconf import OmegaConf

    output = Path(config["output_dir"])
    if (output / "completion.json").exists() and resume is None:
        raise FileExistsError("Completed run exists; do not restart it")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved-config.json", config)
    manifest = {
        "author_source": source_manifest(source_dir),
        "datasets": dataset_hashes(config),
        "environment": environment_manifest(),
        "adapter_sha256": sha256(__file__),
        "device": device,
        "max_steps_override": max_steps,
        "validation_limit_override": limit_validation,
    }
    write_json(output / "execution-manifest.json", manifest)
    seed_all(config["seed"])
    model_type = runtime_class(source_dir)
    model = model_type(OmegaConf.create(config))
    if model.model.config.model_type != "gpt2":
        raise ValueError("This Figure 3 comparison requires the actual GPT-2 model")
    initial_digest = parameter_digest(model)
    if wandb:
        run_logger = WandbLogger(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_name"],
            save_dir=str(output),
        )
    else:
        run_logger = CSVLogger(str(output), name="development")
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch={epoch}-step={step}",
    )
    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min"),
    ]
    trainer_kwargs = {
        "accelerator": "gpu" if device == "cuda" else "cpu",
        "devices": 1,
        "max_epochs": config["num_train_epochs"],
        "max_steps": max_steps,
        "callbacks": callbacks,
        "logger": run_logger,
        "log_every_n_steps": 1,
        "val_check_interval": config["val_check_interval"],
        "reload_dataloaders_every_n_epochs": 1,
        "num_sanity_val_steps": 0,
        "gradient_clip_val": config["max_grad_norm"],
        "accumulate_grad_batches": 1,
        "precision": "bf16-mixed" if config["bf16"] and device == "cuda" else 32,
        # Test-time updates need grad outside Lightning's default inference mode.
        "inference_mode": False,
    }
    if limit_validation is not None:
        trainer_kwargs["limit_val_batches"] = limit_validation
        if max_steps > 0:
            # Development short runs must reach validation before stopping;
            # this override never applies to the full scientific command.
            trainer_kwargs["val_check_interval"] = min(max_steps, len(model.train_data))
    write_json(
        output / "trainer-settings.json",
        {key: value for key, value in trainer_kwargs.items() if key not in ("logger", "callbacks")},
    )
    trainer = pl.Trainer(**trainer_kwargs)
    run_logger.log_hyperparams(config)
    started = time.monotonic()
    trainer.fit(model, ckpt_path=resume)
    # A short development run may end before scheduled validation. Always retain
    # its exact last state, while best-checkpoint selection remains validation-only.
    if not checkpoint.last_model_path:
        trainer.save_checkpoint(str(output / "checkpoints" / "last.ckpt"))
    summary = {
        "status": "training_complete",
        "reproduction_kind": config["reproduction_kind"],
        "initial_parameter_sha256": initial_digest,
        "final_parameter_sha256": parameter_digest(model),
        "global_step": trainer.global_step,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "lora_parameters": sum(p.numel() for name, p in model.named_parameters() if "lora" in name),
        "weighting_parameters": sum(p.numel() for p in model.model.weighting_model.parameters()),
        "inner_learning_rate_parameters": sum(
            p.numel() for p in model.inner_schedular.parameters()
        ),
        "exposure": model.exposure,
        "validation_exposure": model.validation_exposure,
        "wall_seconds": time.monotonic() - started,
        "best_checkpoint": checkpoint.best_model_path,
        "last_checkpoint": checkpoint.last_model_path or str(output / "checkpoints" / "last.ckpt"),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "evaluation_and_reload_status": "pending",
    }
    write_json(output / "completion.json", summary)
    run_logger.finalize("success")
    return summary


def evaluate_checkpoint(
    config: dict[str, Any],
    source_dir: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    split: str = "test",
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    seed_all(config["seed"])
    model_type = runtime_class(source_dir)
    model = model_type.load_from_checkpoint(str(checkpoint), config=OmegaConf.create(config))
    model.to(device)
    before = parameter_digest(model)
    # A fresh evaluation accounts for its own context updates, separately from
    # validation that happened during training and was retained in the checkpoint.
    model.validation_exposure = dict.fromkeys(model.exposure, 0)
    dataset = {"test": model.test_data, "validation": model.dev_data}[split]
    records = []
    losses = []
    destination = Path(config["output_dir"]) / "independent-evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    output_file = destination / f"{split}-predictions.jsonl"
    if output_file.exists():
        raise FileExistsError(output_file)
    with output_file.open("w") as handle:
        for index in range(len(dataset) if limit is None else min(len(dataset), limit)):
            batch = dataset[index]
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model.validation_step(batch, index)
            losses.append(float(output["outer_loss"]))
            for record in output["records"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
            model.validation_step_outputs.clear()
            if parameter_digest(model) != before:
                raise RuntimeError("Inner context adaptation changed persistent model weights")
            write_json(
                destination / "progress.json", {"completed": index + 1, "pool": len(dataset)}
            )
    evaluator = importlib.import_module("perk.evaluate")
    metrics, _ = evaluator.evaluate(records)
    summary = {
        "status": "passed",
        "split": split,
        "rows": len(records),
        "full_pool": len(dataset),
        "metrics": metrics,
        "mean_outer_loss": sum(losses) / len(losses),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_sha256": before,
        "predictions_sha256": sha256(output_file),
        "persistent_parameters_unchanged": True,
        "evaluation_exposure": model.validation_exposure,
        "reproduction_kind": config["reproduction_kind"],
    }
    write_json(destination / f"{split}-summary.json", summary)
    return summary
