"""Independent FP32 cache and down-projection autograd checks, with no updates."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/development-artifacts/architecture-bridge-v1"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def difference(left, right):
    delta = (left - right).abs()
    return {
        "max_absolute_logit_difference": delta.max().item(),
        "rms_logit_difference": delta.square().mean().sqrt().item(),
        "greedy_token_equal": bool(torch.equal(left.argmax(-1), right.argmax(-1))),
        "passed_at_1e_4_absolute": delta.max().item() <= 1e-4,
    }


def qwen3_cache_checks(model, tokenizer):
    ids = tokenizer("Alpha beta gamma delta epsilon.", return_tensors="pt").input_ids.cuda()
    prefix, suffix = ids[:, :-2], ids[:, -2:]

    def forward(current, cache=None, explicit=False):
        length = current.shape[1] + (cache.get_seq_length() if cache is not None else 0)
        kwargs = {}
        if explicit:
            kwargs["position_ids"] = torch.arange(
                length - current.shape[1], length, device=current.device
            )[None, :]
        return model(
            input_ids=current,
            attention_mask=torch.ones((1, length), dtype=torch.long, device=current.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            **kwargs,
        )

    with torch.no_grad():
        full = forward(ids).logits
        cache = forward(prefix).past_key_values
        saved = copy.deepcopy(cache)
        copied = copy.deepcopy(cache)
        same = all(
            torch.equal(getattr(layer, field), getattr(copied.layers[i], field))
            for i, layer in enumerate(cache.layers)
            for field in ("keys", "values")
        )
        independent = all(
            getattr(layer, field).data_ptr() != getattr(copied.layers[i], field).data_ptr()
            for i, layer in enumerate(cache.layers)
            for field in ("keys", "values")
        )
        resumed = forward(suffix, copy.deepcopy(cache)).logits
        explicit = forward(suffix, copy.deepcopy(cache), explicit=True).logits
        report = {
            "prefix_ids": prefix.tolist(),
            "suffix_ids": suffix.tolist(),
            "cache_class": type(cache).__module__ + "." + type(cache).__name__,
            "layer_types": [type(layer).__name__ for layer in cache.layers],
            "full_vs_resumed": difference(full, resumed),
            "implicit_vs_explicit_positions": difference(resumed, explicit),
            "deepcopy_values_equal": same,
            "deepcopy_storage_independent": independent,
            "layers": {},
        }
        for layer_index in (6, 14):
            transplanted = copy.deepcopy(cache)
            transplanted.layers[layer_index] = copy.deepcopy(saved.layers[layer_index])
            continuation = forward(suffix, transplanted).logits
            layer = cache.layers[layer_index]
            report["layers"][str(layer_index)] = {
                "mlp_path": f"model.layers.{layer_index}.mlp",
                "down_path": f"model.layers.{layer_index}.mlp.down_proj",
                "keys_interface": f"past_key_values.layers[{layer_index}].keys",
                "values_interface": f"past_key_values.layers[{layer_index}].values",
                "keys_shape": list(layer.keys.shape),
                "values_shape": list(layer.values.shape),
                "same_donor_transplant_vs_resumed": difference(resumed, continuation),
            }
        report["original_cache_length_preserved"] = cache.get_seq_length() == prefix.shape[1]
    report["all_checks_passed"] = (
        report["full_vs_resumed"]["passed_at_1e_4_absolute"]
        and report["implicit_vs_explicit_positions"]["passed_at_1e_4_absolute"]
        and same
        and independent
        and report["original_cache_length_preserved"]
        and all(
            value["same_donor_transplant_vs_resumed"]["passed_at_1e_4_absolute"]
            for value in report["layers"].values()
        )
    )
    return report


def down_gradient_check(model, tokenizer, layer_index):
    target = model.model.layers[layer_index].mlp.down_proj.weight
    original = target.detach().clone()
    target.requires_grad_(True)
    ids = tokenizer("A small bird rests on a tree.", return_tensors="pt").input_ids.cuda()
    output = model(input_ids=ids, labels=ids, use_cache=False)
    loss = output.loss
    loss.backward()
    gradient = target.grad
    report = {
        "layer_index": layer_index,
        "target": f"model.layers.{layer_index}.mlp.down_proj.weight",
        "input_ids": ids.tolist(),
        "loss": loss.item(),
        "gradient_shape": list(gradient.shape),
        "gradient_norm": gradient.norm().item(),
        "gradient_finite": bool(torch.isfinite(gradient).all()),
        "gradient_nonzero": bool(torch.count_nonzero(gradient).item() > 0),
        "target_weight_unchanged": bool(torch.equal(original, target.detach())),
        "other_parameter_gradients_absent": all(
            parameter.grad is None for parameter in model.parameters() if parameter is not target
        ),
        "optimizer_steps": 0,
        "weights_saved": False,
    }
    target.grad = None
    target.requires_grad_(False)
    report["requires_grad_restored_to_frozen"] = not target.requires_grad
    report["gradient_cleared"] = target.grad is None
    report["all_checks_passed"] = all(
        report[key]
        for key in (
            "gradient_finite",
            "gradient_nonzero",
            "target_weight_unchanged",
            "other_parameter_gradients_absent",
            "requires_grad_restored_to_frozen",
            "gradient_cleared",
        )
    )
    return report


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("Only physical GPU 2 is permitted")
    torch.cuda.set_per_process_memory_fraction(0.12, 0)
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = {}
    models = (
        ("qwen3", Qwen3ForCausalLM, "data/hebbian-learning-v1/source/qwen3-0.6b-base", 14),
        ("qwen35", Qwen3_5ForCausalLM, "data/architecture-bridge-v1/models/qwen3.5-0.8b-base", 10),
    )
    for name, cls, model_path, layer_index in models:
        start = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(ROOT / model_path, local_files_only=True)
        model, loading = cls.from_pretrained(
            ROOT / model_path,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not loading["missing_keys"] and not loading["mismatched_keys"]
        model.cuda().eval().requires_grad_(False)
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable,
            "model_class": type(model).__name__,
            "model_path": str(ROOT / model_path),
            "dtype": "float32",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_config_sha256": sha256(ROOT / model_path / "config.json"),
            "implementation_sha256": sha256(inspect.getfile(cls)),
            "script_sha256": sha256(__file__),
            "loading_info": json.loads(json.dumps(loading, default=sorted)),
            "forward_signature": str(inspect.signature(model.forward)),
        }
        if name == "qwen3":
            record["cache_checks"] = qwen3_cache_checks(model, tokenizer)
        record["down_gradient_check"] = down_gradient_check(model, tokenizer, layer_index)
        torch.cuda.synchronize()
        record["wall_seconds"] = time.monotonic() - start
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        record["all_checks_passed"] = record["down_gradient_check"]["all_checks_passed"] and (
            record.get("cache_checks", {}).get("all_checks_passed", True)
        )
        records[name] = record
        (OUT / f"{name}-preflight.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    manifest = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "recommended_python": sys.executable,
        "secondary_environment": str(ROOT / "data/architecture-bridge-v1/environment"),
        "secondary_status": "Installation and imports passed; not used for model experiments",
        "install_specification": "transformers==5.17.0",
        "install_option_local": "--no-compile",
        "fallback_paths": [
            "/home/siqizhu4/.local/share/llm-memory/runtime-cu128",
            str(ROOT / ".venv/lib/python3.12/site-packages"),
            str(ROOT / "src"),
        ],
        "original_environment_versions_after_install": {
            "transformers": "4.57.3",
            "tokenizers": "0.22.2",
            "huggingface_hub": "0.36.2",
        },
        "primary_versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")
        },
        "pip_freeze_all": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--all"], text=True
        ).splitlines(),
        "pip_check": subprocess.check_output(
            [sys.executable, "-m", "pip", "check"], text=True
        ).strip(),
        "setup_notes": [
            "Shared filesystem installation was slow; completed without interruption.",
            "All preflights used the local disk environment with identical pinned packages.",
            "Initial model preflight import needed the existing torchgen runtime fallback path.",
            "Second preflight reached serialization; loading_info sets needed JSON conversion.",
            "Both failed attempts are retained as model-preflight-attempt-1/2.log.",
            "FP32 equivalence checks passed; BF16 reference-kernel discrepancy was retained.",
        ],
        "optimized_gdn_kernels_installed": False,
        "active_install_processes": 0,
        "additional_preflights_passed": all(
            value["all_checks_passed"] for value in records.values()
        ),
    }
    (OUT / "environment-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["additional_preflights_passed"]:
        raise SystemExit("An independent preflight failed")


if __name__ == "__main__":
    main()
