"""Contracts that matter for the published PERK computation and its execution."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_memory_editability.perk_reproduction import (
    AUTHOR_COMMIT,
    dataset_hashes,
    evaluate_checkpoint,
    load_author,
    parameter_digest,
    resolve_config,
    runtime_class,
    seed_all,
    source_manifest,
    train,
    write_json,
)

AUTHOR_SOURCE = Path(
    os.environ.get(
        "PERK_AUTHOR_SOURCE",
        "/ossfs/workspace/research-recovery/perk-paper-reproduction-v1/reference/perk",
    )
)


@pytest.fixture(scope="module")
def author():
    if not AUTHOR_SOURCE.exists():
        pytest.skip("Frozen PERK author source is required")
    for package in ("lightning", "peft", "transformers", "omegaconf", "datasets"):
        pytest.importorskip(package)
    return load_author(AUTHOR_SOURCE)


@pytest.fixture()
def miniature(tmp_path, author):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    seed_all(317)
    model_dir = tmp_path / "tiny-gpt2"
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[UNK]",
                "[EOS]",
                "doc_0",
                "doc_1",
                ":",
                "Mary",
                "John",
                "went",
                "to",
                "the",
                "kitchen",
                "hallway",
                ".",
                "Where",
                "is",
                "?",
                "You",
                "have",
                "memorized",
                "2",
                "documents",
                "Answer",
                "<",
                ">",
                "/",
                "support",
                "output",
                "bathroom",
            ]
        )
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="[EOS]",
        pad_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=128,
    )
    tokenizer.save_pretrained(model_dir)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=128,
            n_embd=16,
            n_layer=2,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).save_pretrained(model_dir)
    folder = tmp_path / "data" / "babilong" / "qa2_8k"
    folder.mkdir(parents=True)
    rows = [
        {
            "context": ["Mary went to the kitchen.", "John went to the hallway."],
            "question": "Where is Mary?",
            "answer": "kitchen",
            "support": ["Mary went to the kitchen."],
        },
        {
            "context": ["Mary went to the bathroom.", "John went to the kitchen."],
            "question": "Where is Mary?",
            "answer": "bathroom",
            "support": ["Mary went to the bathroom."],
        },
    ]
    for split in ("train", "validation", "test"):
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return resolve_config(
        {
            "reproduction_kind": "development_smoke",
            "model_name_or_path": str(model_dir),
            "data_dir": str(tmp_path / "data"),
            "num_workers": 0,
            "lora_rank": 2,
            "lora_alpha": 2,
            "bf16": False,
            "max_new_tokens": 4,
            "val_check_interval": 1.0,
            "do_eval": True,
        },
        tmp_path / "output",
    )


def test_labels_cannot_claim_unreported_exact_recipe(tmp_path):
    with pytest.raises(ValueError, match="exact paper replication"):
        resolve_config({"data_dir": "unused", "reproduction_kind": "exact_replication"}, tmp_path)
    with pytest.raises(ValueError, match="Credentials"):
        resolve_config({"data_dir": "unused", "wandb_api_key": "placeholder"}, tmp_path)


def test_one_problem_and_independent_chunk_contract(tmp_path):
    for field, value in (("train_batch_size", 2), ("packing", True), ("n_gpu", 2)):
        with pytest.raises(ValueError):
            resolve_config({"data_dir": "unused", field: value}, tmp_path)


def test_original_source_retained_and_adamw_body_unchanged(author):
    manifest = source_manifest(AUTHOR_SOURCE)
    assert manifest["commit"] == AUTHOR_COMMIT
    optimizer = importlib.import_module("higher.optim")
    original = (AUTHOR_SOURCE / "higher/optim.py").read_text()
    execution = Path(optimizer.__file__).read_text()
    section = "class DifferentiableAdamW"
    end = "class DifferentiableAdamax"
    assert original.split(section)[1].split(end)[0] == execution.split(section)[1].split(end)[0]


def test_training_coverage_and_evaluation_no_resampling(miniature):
    from omegaconf import OmegaConf

    model = runtime_class(AUTHOR_SOURCE)(OmegaConf.create(miniature))
    assert len(model.train_data) == len(model.dev_data) == len(model.test_data) == 2
    assert [r["guid"] for r in model.dev_data.data_rows] == ["qa2_8k_0", "qa2_8k_1"]
    assert dataset_hashes(miniature)["train"]["rows"] == 2


def test_base_frozen_and_real_gpt2_metagradients(miniature):
    import torch
    from omegaconf import OmegaConf

    model = runtime_class(AUTHOR_SOURCE)(OmegaConf.create(miniature))
    for name, param in model.model.lm_model.named_parameters():
        assert param.requires_grad == ("lora" in name)
    initial = parameter_digest(model)
    batch = model.train_data[0]
    model.eval()
    # Author step explicitly turns adaptation/train dropout on. Fix its random
    # state for deterministic metagradient checks.
    seed_all(411)
    loss = model.step(batch)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter_digest(model) == initial
    groups = {
        "adapter": [p for n, p in model.model.named_parameters() if "lora" in n],
        "weighting": list(model.model.weighting_model.parameters()),
        "learned_lr": list(model.inner_schedular.parameters()),
    }
    for parameters in groups.values():
        gradients = [p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_author_adamw_matches_torch_and_learning_rate_finite_difference(author):
    import torch
    from torch import nn

    higher = importlib.import_module("higher")

    class Quadratic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.7, -0.4], dtype=torch.float64))

        def forward(self):
            return ((self.weight - torch.tensor([0.2, 0.8], dtype=torch.float64)) ** 2).sum()

    reference = Quadratic()
    opt = torch.optim.AdamW(reference.parameters(), lr=0.03, betas=(0.9, 0.95), weight_decay=0.01)
    for _ in range(4):
        opt.zero_grad()
        reference().backward()
        opt.step()

    def adapted(lr):
        model = Quadratic()
        inner = torch.optim.AdamW(model.parameters(), lr=0.03, betas=(0.9, 0.95), weight_decay=0.01)
        with higher.innerloop_ctx(model, inner, copy_initial_weights=False) as (fast, diffopt):
            for _ in range(4):
                diffopt.step(fast(), override={"lr": [lr]})
            return fast(), fast.weight

    lr = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    loss, weights = adapted(lr)
    torch.testing.assert_close(weights, reference.weight, rtol=1e-10, atol=1e-12)
    derivative = torch.autograd.grad(loss, lr)[0]
    epsilon = 1e-6
    plus, _ = adapted(torch.tensor(0.03 + epsilon, dtype=torch.float64))
    minus, _ = adapted(torch.tensor(0.03 - epsilon, dtype=torch.float64))
    numeric = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(derivative, numeric, rtol=1e-7, atol=1e-8)


def test_real_checkpoint_reloads_and_persistent_weights_survive_adaptation(miniature, tmp_path):
    import torch
    from omegaconf import OmegaConf

    summary = train(
        miniature, AUTHOR_SOURCE, device="cpu", max_steps=1, limit_validation=1, wandb=False
    )
    model_type = runtime_class(AUTHOR_SOURCE)
    restored = model_type.load_from_checkpoint(
        summary["last_checkpoint"], config=OmegaConf.create(miniature)
    )
    assert parameter_digest(restored) == summary["final_parameter_sha256"]
    assert restored.exposure == summary["exposure"]
    assert restored.exposure["problems"] == 1
    before = parameter_digest(restored)
    loss = restored.step(restored.train_data[0])["loss"]
    assert torch.isfinite(loss)
    assert parameter_digest(restored) == before
    reference = evaluate_checkpoint(
        miniature, AUTHOR_SOURCE, summary["last_checkpoint"], device="cpu", limit=1
    )
    config_file = tmp_path / "reload-config.json"
    write_json(config_file, miniature)
    subprocess_output = tmp_path / "new-process"
    command = [
        sys.executable,
        "scripts/run_perk_reproduction.py",
        "evaluate",
        "--config",
        str(config_file),
        "--source-dir",
        str(AUTHOR_SOURCE),
        "--output",
        str(subprocess_output),
        "--device",
        "cpu",
        "--checkpoint",
        summary["last_checkpoint"],
        "--limit-evaluation",
        "1",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    replay = json.loads(
        (subprocess_output / "independent-evaluation/test-summary.json").read_text()
    )
    assert replay["predictions_sha256"] == reference["predictions_sha256"]
    assert replay["parameter_sha256"] == reference["parameter_sha256"]
    assert replay["mean_outer_loss"] == reference["mean_outer_loss"]
    write_json(tmp_path / "checkpoint-contract.json", summary)
