"""Conditional P3 control: block cross-fact document attention during training.

The frozen document tensors, absolute positions, loss masks, optimizer, QA stream,
and inference path are unchanged. Only document attention edges are removed.
This module is prepared for conditional execution; importing it does not train.
"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import bios_cross_train as frozen_trainer
from .bios_cross import CHAINS, CONDITIONS, audit, documents, make_cross_world, qa_schedule
from .bios_data import array_hash, write_json
from .bios_model import Attention, CausalLM
from .bios_organization_train import atomic_numpy_save
from .bios_path_diagnostics import autonomous_two_step

RECORD_LENGTH = 6
DOCUMENT_LENGTH = 60
CHECKPOINTS = (0, 1280, 2560, 5120, 10240, 15360)


def fact_causal_mask(length=DOCUMENT_LENGTH, record_length=RECORD_LENGTH, device=None):
    """SDPA Boolean True means an allowed edge: same record and causal order."""
    if length <= 0 or record_length <= 0 or length % record_length:
        raise ValueError("Document length must contain complete fact records")
    position = torch.arange(length, device=device)
    return (position[:, None] // record_length == position[None, :] // record_length) & (
        position[None, :] <= position[:, None]
    )


class FactIsolatedAttention(Attention):
    """An existing Attention object's parameters are retained without reinitializing."""

    def forward(self, x):
        if not self.document_attention_isolated:
            return super().forward(x)
        batch, length, width = x.shape
        if length != DOCUMENT_LENGTH:
            raise ValueError("Document-only attention entered with a non-document tensor")
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).unbind(2)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=self.fact_mask,
            is_causal=False,
        )
        return self.proj(output.transpose(1, 2).reshape(batch, length, width))


class FactIsolatedCausalLM(CausalLM):
    """Use the original initializer, forward, position IDs, and state-dict schema.

    In the frozen trainer the only 60-token training input is a document; QA
    training inputs contain six tokens. Evaluation always uses the original
    attention path, including an unusually long inference input. No dropout,
    extra trainable parameter, random draw, or reset of position IDs is added.
    """

    def __init__(self, config):
        super().__init__(config)
        mask = fact_causal_mask()
        for block in self.blocks:
            attention = block.attention
            attention.__class__ = FactIsolatedAttention
            attention.document_attention_isolated = False
            attention.register_buffer("fact_mask", mask, persistent=False)

    def forward(self, tokens, positions=None):
        isolate = self.training and tokens.shape[1] == DOCUMENT_LENGTH
        if self.training and tokens.shape[1] not in (RECORD_LENGTH, DOCUMENT_LENGTH):
            raise ValueError("Unexpected training input length; document/QA dispatch is unsafe")
        if isolate:
            if positions is None or positions.shape[1] != 20:
                raise ValueError("Document training must preserve all twenty supervised positions")
            expected = torch.arange(DOCUMENT_LENGTH, device=tokens.device).view(10, 6)[:, 3:5]
            if not torch.equal(positions, expected.reshape(1, 20).expand(len(tokens), -1)):
                raise ValueError("Document supervision positions changed")
        for block in self.blocks:
            block.attention.document_attention_isolated = isolate
        try:
            return super().forward(tokens, positions)
        finally:
            for block in self.blocks:
                block.attention.document_attention_isolated = False


@contextmanager
def isolated_model_factory():
    """Patch only this worker's factory while retaining the frozen trainer body."""
    original = frozen_trainer.CausalLM
    if original is not CausalLM:
        raise RuntimeError("The frozen trainer already has a modified model factory")
    frozen_trainer.CausalLM = FactIsolatedCausalLM
    try:
        yield
    finally:
        frozen_trainer.CausalLM = original


def context_sources():
    directory = Path(__file__).parent
    return {
        **frozen_trainer.source_hashes(),
        **{
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("bios_context_control.py", "bios_path_diagnostics.py")
        },
    }


def validate_study(study):
    expected = {
        "width": 256,
        "layers": 8,
        "heads": 4,
        "steps": 15360,
        "checkpoints": list(CHECKPOINTS),
        "lr": 0.0001,
        "documents_per_step": 16,
        "facts_per_document": 10,
        "QA_per_chain_per_step": 20,
        "document_QA_weights": [0.8, 0.2],
        "worlds": [0, 1],
        "seeds": [0, 1],
        "conditions": list(CONDITIONS),
        "learning_runs": 12,
        "edit_cases": 0,
    }
    for key, value in expected.items():
        if study.get(key) != value:
            raise ValueError(f"Context control changed its frozen study setting: {key}")


def run(args):
    study = json.loads(Path(args.config).read_text())
    validate_study(study)
    if args.world not in study["worlds"] or args.seed not in study["seeds"]:
        raise ValueError("Context control is restricted to the development matrix")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        world = make_cross_world(args.world)
        world_audit = audit(world)
        docs, schedule = documents(world, args.condition), qa_schedule(world, study["steps"])
        contract = {
            "protocol": "v2.8-p3-context-control",
            "world": args.world,
            "seed": args.seed,
            "condition": args.condition,
            "study": study,
            "sources": context_sources(),
            "intervention": {
                "training_document_attention": "causal within each six-token fact only",
                "absolute_positions": "unchanged 0 through 59; never reset within facts",
                "document_tokens_batches_and_supervision": "unchanged",
                "QA_training_attention": "original causal attention",
                "all_inference_attention": "original causal attention",
                "mask_sha256": array_hash(fact_causal_mask().numpy()),
            },
            "data": {
                "truth_sha256": world_audit["truth_sha256"],
                "prompts_sha256": world_audit["prompts_sha256"],
                "documents_sha256": array_hash(docs),
                "qa_sha256": array_hash(schedule),
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            },
        }
        path = output / "launch-contract.json"
        if path.exists():
            old = json.loads(path.read_text())
            for key in contract.keys() - {"environment"}:
                if old[key] != contract[key]:
                    raise ValueError(f"Context control contract changed: {key}")
        else:
            write_json(path, contract)
        if (output / "complete.json").exists():
            return
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        controlled_study = {
            **study,
            "context_contract": contract["sources"],
            "context_intervention": contract["intervention"],
        }
        try:
            with isolated_model_factory():
                model, _ = frozen_trainer.learning(args, world, output, controlled_study, device)
            # Inference must be the original unrestricted forward path.
            model.eval()
            with np.load(output / f"predictions-{study['steps']}.npz") as saved:
                direct = {key: saved[key] for key in ("prediction", "ended", "correct")}
            for chain, name in enumerate(CHAINS):
                arrays = autonomous_two_step(model, world, chain, device)
                ids = world.derived_ids[chain]
                arrays.update({f"direct_{key}": value[ids] for key, value in direct.items()})
                atomic_numpy_save(output / f"two-step-{name}.npz", **arrays)
            write_json(
                output / "complete.json",
                {
                    "status": "complete",
                    "phase": "P3-context-control-learning-only",
                    "learning_steps": study["steps"],
                    "edit_cases": 0,
                    "finished": time.time(),
                },
            )
        except Exception as error:
            write_json(
                output / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "time": time.time(),
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--config", default="configs/bios-context-control-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
