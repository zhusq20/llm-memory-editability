"""Paired parameter-sharing interventions in the project's complete GPT-2.

Every arm starts from the same seeded reference model. Repeating its prefix and
copying that prefix to independent blocks give exactly the same initial function
at the same executed depth. The intervention is the parameter constraint during
learning, not an extra embedding, readout, or change to the Transformer block.
"""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn
from transformers import GPT2Config

from .grokking_reproduction import ReproductionGPT

ARMS = ("shared", "untied", "mlp_shared", "shallow")


def construct(spec, device):
    """Build an arm; reconstruction must use the same spec before loading state.

    ``base_unique_blocks`` selects the reference prefix and ``repeats`` its
    logical unroll count. ``shallow`` executes that prefix once. ``mlp_shared``
    shares only the MLP module, leaving both LayerNorms and attention independent
    at each occurrence. The input/output embedding is tied in every arm.

    All arms initialize ``initialization_reference_layers`` (default eight)
    before selecting the prefix. Thus GPT-2's residual projection initialization
    scale, shared prefix, embedding, and final LayerNorm do not depend on the arm
    or unroll count. This is an intervention initialization, not native independent
    initialization for the deeper untied model.
    """
    arm = spec["arm"]
    if arm not in ARMS:
        raise ValueError(f"Unknown Loop learning arm: {arm}")
    base = int(spec["base_unique_blocks"])
    repeats = int(spec["repeats"])
    reference = int(spec.get("initialization_reference_layers", 8))
    if base < 1 or repeats < 1 or reference < base:
        raise ValueError("Need positive base/repeats and reference depth >= base depth")
    options = spec["model"]
    if options["hidden_size"] % options["attention_heads"]:
        raise ValueError("Hidden width must be divisible by the attention head count")
    config = GPT2Config(
        vocab_size=options["vocab_size"],
        n_positions=options.get("positions", 1024),
        n_embd=options["hidden_size"],
        n_head=options["attention_heads"],
        n_layer=reference,
        resid_pdrop=options["dropout"],
        embd_pdrop=options["dropout"],
        attn_pdrop=options["dropout"],
        activation_function="gelu_new",
        use_cache=False,
    )
    torch.manual_seed(spec["initialization"])
    model = ReproductionGPT(config, repeats=1)
    prefix = list(model.transformer.h)[:base]
    if arm in {"shared", "shallow"}:
        model.transformer.h = nn.ModuleList(prefix)
        model.repeats = repeats if arm == "shared" else 1
    else:
        blocks = [copy.deepcopy(block) for _ in range(repeats) for block in prefix]
        if arm == "mlp_shared":
            for index, block in enumerate(blocks):
                block.mlp = blocks[index % base].mlp
        model.transformer.h = nn.ModuleList(blocks)
    model.config.n_layer = len(model.transformer.h)
    model.loop_learning_info = {
        "arm": arm,
        "base_unique_blocks": base,
        "logical_repeats": repeats,
        "initialization_reference_layers": reference,
        "initialization": int(spec["initialization"]),
        "initialization_policy": "seeded reference prefix; identical occurrence copies",
        "mlp_sharing_includes_layernorm": False,
    }
    return model.to(device)


def architecture_manifest(model):
    """Count unique storage separately from repeated execution and module use."""
    blocks = list(model.transformer.h)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        **model.loop_learning_info,
        "unique_parameters": parameters,
        "trainable_unique_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        # These dense arms execute every unique parameter in a full forward.
        "active_unique_parameters": parameters,
        "stored_blocks": len(blocks),
        "execution_repeats": model.repeats,
        "executed_blocks": len(blocks) * model.repeats,
        "unique_block_modules": len({id(block) for block in blocks}),
        "unique_attention_modules": len({id(block.attn) for block in blocks}),
        "unique_mlp_modules": len({id(block.mlp) for block in blocks}),
        "executed_block_parameter_uses": model.repeats
        * sum(parameter.numel() for block in blocks for parameter in block.parameters()),
        "hidden_size": model.config.n_embd,
        "attention_heads": model.config.n_head,
        "vocab_size": model.config.vocab_size,
        "tied_input_output_embedding": True,
        "flops_scope": (
            "Training matrix multiply estimate from actual padded input, readout positions, "
            "and executed depth; excludes optimizer, normalization, nonlinearities, "
            "evaluation and other overhead. It is not measured hardware FLOPs."
        ),
    }


def estimate_training_flops(model, executed_input_tokens, readout_positions, attention_pairs):
    """Mirror realworld_composition.update's forward/backward matmul estimate.

    ``attention_pairs`` is sum(batch_size * padded_sequence_length**2) over
    microbatches; ``readout_positions`` includes loss-masked padded answer slots.
    Parameter sharing reduces storage, but does not reduce this executed cost.
    """
    if min(executed_input_tokens, readout_positions, attention_pairs) < 0:
        raise ValueError("FLOPs counters must be nonnegative")
    width = model.config.n_embd
    blocks = len(model.transformer.h) * model.repeats
    return (
        6 * 12 * width * width * blocks * executed_input_tokens
        + 6 * width * model.config.vocab_size * readout_positions
        + 12 * blocks * width * attention_pairs
    )


def execution_digest(model):
    """Hash weights in execution order, canonicalizing the sharing convention.

    Identical digests verify identical expanded weights, not behavior on all
    possible inputs. Dropout randomness and runtime numerics remain separate.
    This intentionally differs from a raw state_dict checkpoint digest.
    """
    digest = hashlib.sha256()

    def add(name, tensor):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{list(value.shape)}".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())

    for name, parameter in model.transformer.wte.named_parameters():
        add("embedding." + name, parameter)
    for name, parameter in model.transformer.wpe.named_parameters():
        add("position." + name, parameter)
    for repeat in range(model.repeats):
        for index, block in enumerate(model.transformer.h):
            occurrence = repeat * len(model.transformer.h) + index
            for name, parameter in block.named_parameters():
                add(f"occurrence.{occurrence}.{name}", parameter)
    for name, parameter in model.transformer.ln_f.named_parameters():
        add("final_norm." + name, parameter)
    return digest.hexdigest()
