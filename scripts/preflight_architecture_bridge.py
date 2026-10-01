"""Read-only model/cache preflight for the architecture-bridge-v1 experiment.

Run with the batch-specific environment and CUDA_VISIBLE_DEVICES=2. No parameter
updates are performed. The official multimodal checkpoint is loaded through its
text-only class, so the vision encoder is deliberately not instantiated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("data/architecture-bridge-v1/models/qwen3.5-0.8b-base")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/development-artifacts/architecture-bridge-v1/model-preflight.json"),
    )
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("This batch permits only physical GPU 2: set CUDA_VISIBLE_DEVICES=2")

    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    torch.set_num_threads(4)
    torch.manual_seed(1729)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.12, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "tokenizers", "huggingface-hub", "safetensors"]
        },
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_memory_fraction_limit": 0.12 if args.device.startswith("cuda") else None,
        "model_path": str(args.model.resolve()),
        "model_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "vision_encoder_loaded": False,
        "parameter_updates": 0,
        "script_sha256": digest(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report["tokenizer_class"] = type(tokenizer).__name__
    load_start = time.monotonic()
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    report["loading_info"] = json.loads(json.dumps(loading, default=sorted))
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Checkpoint did not fully initialize text model: {loading}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    report["load_wall_seconds"] = time.monotonic() - load_start
    report["model_class"] = type(model).__name__
    report["text_parameter_count"] = sum(p.numel() for p in model.parameters())
    report["layer_types"] = model.config.layer_types
    report["mlp_paths"] = [f"model.layers.{i}.mlp" for i in range(len(model.model.layers))]
    report["forward_signature"] = str(inspect.signature(model.forward))
    report["implementation"] = {
        "path": inspect.getfile(type(model)),
        "sha256": digest(Path(inspect.getfile(type(model)))),
    }
    report["checkpoint_sample_checks"] = {}
    shard = next(args.model.glob("*.safetensors"))
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        for path in [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.23.mlp.down_proj.weight",
        ]:
            official_key = path.replace("model.", "model.language_model.", 1)
            expected = checkpoint.get_tensor(official_key).float()
            actual = model.get_parameter(path).detach().cpu()
            report["checkpoint_sample_checks"][path] = bool(torch.equal(expected, actual))
    if not all(report["checkpoint_sample_checks"].values()):
        raise RuntimeError("Text-only checkpoint remapping verification failed")
    print("Text-only model loaded; beginning cache checks", flush=True)

    def tokens(text: str):
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
            args.device
        )

    clean_ids = tokens("Known fact: Ada lives in Paris. Bridge entity: Paris")
    corrupt_ids = tokens("Known fact: Ada lives in Paris. Bridge entity: Rome")
    suffix = tokens("\nQuestion: Where does Ada live?\nAnswer:")
    assert clean_ids.shape == corrupt_ids.shape
    prefix_length = clean_ids.shape[1]
    report["prompts"] = {
        "clean_prefix_ids": clean_ids.tolist(),
        "corrupt_prefix_ids": corrupt_ids.tolist(),
        "suffix_ids": suffix.tolist(),
        "prefix_length": prefix_length,
        "changed_positions": (clean_ids != corrupt_ids).nonzero().tolist(),
        "note": "Engineering examples only; not experimental observations.",
    }

    def forward(ids, cache=None, explicit_positions=False):
        past = 0 if cache is None else cache.get_seq_length()
        kwargs = {}
        if explicit_positions:
            kwargs["position_ids"] = torch.arange(past, past + ids.shape[1], device=args.device)[
                None, :
            ]
        return model(
            input_ids=ids,
            attention_mask=torch.ones(
                (1, past + ids.shape[1]), dtype=torch.long, device=args.device
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            **kwargs,
        )

    def comparison(left, right, tolerance):
        delta = (left.float() - right.float()).abs()
        return {
            "max_absolute_logit_difference": delta.max().item(),
            "rms_logit_difference": delta.square().mean().sqrt().item(),
            "greedy_token_equal": bool(torch.equal(left.argmax(-1), right.argmax(-1))),
            "absolute_tolerance": tolerance,
            "passed": bool(delta.max().item() <= tolerance),
        }

    def cache_tensors(cache):
        for i, layer in enumerate(cache.layers):
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                value = getattr(layer, name, None)
                if isinstance(value, torch.Tensor):
                    yield f"layers.{i}.{name}", value
                elif isinstance(value, dict):
                    for index, tensor in value.items():
                        if isinstance(tensor, torch.Tensor):
                            yield f"layers.{i}.{name}.{index}", tensor
                elif isinstance(value, (list, tuple)):
                    for index, tensor in enumerate(value):
                        if isinstance(tensor, torch.Tensor):
                            yield f"layers.{i}.{name}.{index}", tensor

    with torch.no_grad():
        checks = {}
        clean = forward(clean_ids)
        clean_cache = clean.past_key_values
        corrupt_cache = forward(corrupt_ids).past_key_values
        report["cache_type"] = type(clean_cache).__module__ + "." + type(clean_cache).__name__
        report["cache_layers"] = [type(layer).__name__ for layer in clean_cache.layers]
        report["cache_tensors"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in cache_tensors(clean_cache)
        }
        cloned = copy.deepcopy(clean_cache)
        originals = dict(cache_tensors(clean_cache))
        copies = dict(cache_tensors(cloned))
        checks["deepcopy"] = {
            "values_equal": all(
                torch.equal(value, copies[name]) for name, value in originals.items()
            ),
            "storage_independent": all(
                value.data_ptr() != copies[name].data_ptr() for name, value in originals.items()
            ),
        }
        full = forward(torch.cat([clean_ids, suffix], dim=1)).logits
        continued = forward(suffix, copy.deepcopy(clean_cache)).logits
        checks["fp32_prefill_suffix_vs_full"] = comparison(full, continued, 1e-4)
        explicit = forward(suffix, copy.deepcopy(clean_cache), explicit_positions=True).logits
        checks["implicit_vs_explicit_positions"] = comparison(continued, explicit, 1e-6)
        single_cache = copy.deepcopy(clean_cache)
        for index in range(suffix.shape[1]):
            single = forward(suffix[:, index : index + 1], single_cache)
            single_cache = single.past_key_values
        checks["fp32_tokenwise_suffix_vs_full"] = comparison(full, single.logits, 2e-4)
        # The caller owns and clones all caches: the model updates them in place.
        report["prefill_cache_original_length_after_cloned_continuations"] = (
            clean_cache.get_seq_length()
        )

        layer_index = 10
        captured = {}

        def capture(_module, _inputs, output):
            captured["value"] = output[:, -1, :].detach().clone()

        hook = model.model.layers[layer_index].mlp.register_forward_hook(capture)
        forward(clean_ids)
        hook.remove()

        def patch(_module, _inputs, output):
            output = output.clone()
            output[:, -1, :] = captured["value"]
            return output

        hook = model.model.layers[layer_index].mlp.register_forward_hook(patch)
        patched_cache = forward(corrupt_ids).past_key_values
        hook.remove()
        before = copy.deepcopy(patched_cache)
        patched_cache.layers[layer_index] = copy.deepcopy(clean_cache.layers[layer_index])
        # Also exercise a full-attention layer transfer separately.
        kv_transferred = copy.deepcopy(corrupt_cache)
        kv_transferred.layers[11] = copy.deepcopy(clean_cache.layers[11])
        transferred_output = forward(suffix, patched_cache)
        kv_output = forward(suffix, kv_transferred)
        source = dict(cache_tensors(clean_cache))
        donor_before_decode = copy.deepcopy(before)
        donor_before_decode.layers[layer_index] = copy.deepcopy(clean_cache.layers[layer_index])
        checks["mlp_and_same_layer_gdn_cache_transfer"] = {
            "layer_index": layer_index,
            "layer_type": model.config.layer_types[layer_index],
            "donor_tensors_equal_before_decode": all(
                torch.equal(tensor, source[name])
                for name, tensor in cache_tensors(donor_before_decode)
                if name.startswith(f"layers.{layer_index}.")
            ),
            "all_logits_finite": bool(torch.isfinite(transferred_output.logits).all()),
            "donor_length_unchanged": clean_cache.get_seq_length() == prefix_length,
            "next_greedy_token": transferred_output.logits.argmax(-1).tolist(),
        }
        checks["attention_kv_cache_transfer"] = {
            "layer_index": 11,
            "all_logits_finite": bool(torch.isfinite(kv_output.logits).all()),
        }
        next_ids = transferred_output.logits[:, -1, :].argmax(-1, keepdim=True)
        final_cache = transferred_output.past_key_values
        decoded = []
        for _ in range(4):
            decoded.append(next_ids.item())
            step = forward(next_ids, final_cache)
            final_cache = step.past_key_values
            next_ids = step.logits[:, -1, :].argmax(-1, keepdim=True)
        checks["custom_greedy_decode"] = {
            "tokens": decoded,
            "text": tokenizer.decode(decoded),
            "final_cache_length": final_cache.get_seq_length(),
            "all_logits_finite": bool(torch.isfinite(step.logits).all()),
        }
        # BF16 is separately assessed; tolerance is a numerical engineering check,
        # not a criterion for accepting an experimental mechanism hypothesis.
        model.to(dtype=torch.bfloat16)
        bf_cache = forward(clean_ids).past_key_values
        bf_full = forward(torch.cat([clean_ids, suffix], dim=1)).logits
        bf_continued = forward(suffix, bf_cache).logits
        checks["bf16_prefill_suffix_vs_full"] = comparison(bf_full, bf_continued, 0.15)

    report["checks"] = checks
    report["all_required_checks_passed"] = all(
        checks[name]["passed"]
        for name in (
            "fp32_prefill_suffix_vs_full",
            "implicit_vs_explicit_positions",
            "fp32_tokenwise_suffix_vs_full",
        )
    ) and all(checks["deepcopy"].values())
    report["all_required_checks_passed"] = report["all_required_checks_passed"] and all(
        checks["mlp_and_same_layer_gdn_cache_transfer"][name]
        for name in (
            "donor_tensors_equal_before_decode",
            "all_logits_finite",
            "donor_length_unchanged",
        )
    )
    report["all_required_checks_passed"] = (
        report["all_required_checks_passed"]
        and checks["attention_kv_cache_transfer"]["all_logits_finite"]
        and checks["custom_greedy_decode"]["all_logits_finite"]
    )
    report["cache_contract"] = {
        "in_place_mutation": True,
        "safe_branching": "copy.deepcopy(cache) under torch.no_grad()",
        "gdn_recurrent": "cache.layers[L].recurrent_states[0]",
        "gdn_convolution": "cache.layers[L].conv_states[0]",
        "full_attention_keys": "cache.layers[L].keys",
        "full_attention_values": "cache.layers[L].values",
        "same_length_transfer": "cache.layers[L] = copy.deepcopy(donor_cache.layers[L])",
        "position_ids": "Optional for batch-one unpadded text; inferred from cache length",
        "cache_position": "Not needed by this pinned model forward",
        "attention_mask": "Length equals cached prefix plus current input tokens",
        "limitations": "No padded/multimodal/batched/beam or gradient-path validation",
    }
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    report["wall_seconds"] = time.monotonic() - start
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "checks": checks}, indent=2), flush=True)
    if not report["all_required_checks_passed"]:
        raise SystemExit("Required cache equivalence checks failed; inspect report")


if __name__ == "__main__":
    main()
