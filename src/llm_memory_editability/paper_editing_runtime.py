"""Original ROME/MEMIT calls, exact rollback, raw generation, and independent replay."""

from __future__ import annotations

import gc
import importlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from .experiment_tracking import read_json as read_json
from .experiment_tracking import write_json
from .paper_reproduction import sha


def verify_inputs(spec):
    for filename, expected in spec["input_hashes"].items():
        assert sha(filename) == expected, filename
    assert sha(spec["data_file"]) == spec["data_sha256"]


def answer_text(raw, mode):
    if mode == "cot":
        match = re.search(r"(?:^|\n)Answer:\s*([^\n]+)", raw)
        return match.group(1).strip() if match else ""
    return raw.strip().split("\n", 1)[0].strip()


def exact_alias_match(text, answers):
    # The authors' released MQuAKE notebook uses equality with the label/aliases.
    return text in answers


def cloze_alias_match(text, answers):
    """The generated object must start the continuation, with an entity-name boundary."""
    text = text.lstrip()
    return any(
        alias
        and text.startswith(alias)
        and (len(text) == len(alias) or not text[len(alias)].isalnum())
        for alias in answers
    )


def ripple_match(text, answers):
    # Paper §5.1: at least one gold object (and one of that object's aliases).
    return any(alias in text for group in answers for alias in group if alias)


def source_ripple_match(text, answers):
    # Preserve the released executor's stricter all-objects rule as a diagnostic.
    return bool(answers) and all(any(a and a in text for a in group) for group in answers)


def canonical_block_arguments(module, args, kwargs):
    """Expose keyword hidden_states to the authors' positional-input tracing hook."""
    if not args and "hidden_states" in kwargs:
        remaining = dict(kwargs)
        hidden = remaining.pop("hidden_states")
        return (hidden,), remaining
    return args, kwargs


class Generation:
    def __init__(self, model, tokenizer, spec, device):
        self.model, self.tokenizer = model, tokenizer
        self.spec, self.device = spec, device
        self.edited = False
        self.calls = []
        self.input_tokens = self.output_tokens = self.queries = 0

    def reset_calls(self):
        self.calls = []

    def generate(self, prompts, *, max_new_tokens, mode="line"):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        previous_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        finally:
            self.tokenizer.padding_side = previous_padding
        width = encoded["input_ids"].shape[1]
        limit = getattr(self.model.config, "n_positions", None)
        if limit is None:
            limit = self.model.config.max_position_embeddings
        assert width + max_new_tokens <= limit, "Do not silently truncate benchmark prompts"
        tokenizer = self.tokenizer

        class AnswerStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                suffixes = tokenizer.batch_decode(input_ids[:, width:], skip_special_tokens=True)
                if mode == "cot":
                    return all(re.search(r"(?:^|\n)Answer:[^\n]+\n", s) for s in suffixes)
                if mode == "line":
                    return all("\n" in s.lstrip("\n") for s in suffixes)
                return False

        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([AnswerStop()])
                if mode != "ripple"
                else None,
            )
        generated = output[:, width:].cpu().tolist()
        raw_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        lengths = encoded["attention_mask"].sum(dim=1).cpu().tolist()
        items = []
        for prompt, raw, tokens, length in zip(prompts, raw_texts, generated, lengths, strict=True):
            items.append(
                {
                    "prompt": prompt,
                    "raw_generation": raw,
                    "generated_token_ids": tokens,
                    "prompt_tokens": length,
                    "answer_text": answer_text(raw, mode),
                    "hit_generation_limit": len(tokens) == max_new_tokens,
                }
            )
        call = {
            "edited": self.edited,
            "max_new_tokens": max_new_tokens,
            "mode": mode,
            "items": items,
        }
        self.calls.append(call)
        self.queries += len(items)
        self.input_tokens += sum(lengths)
        self.output_tokens += sum(len(ids) for ids in generated)
        return items


class Runtime:
    """Use the frozen author implementation; observers do not change its optimizer."""

    def __init__(self, spec, out, device):
        import numpy as np
        import torch
        import yaml
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec, self.out, self.device = spec, Path(out), device
        self.started = time.monotonic()
        verify_inputs(spec)
        random.seed(spec["seed"])
        np.random.seed(spec["seed"])
        torch.manual_seed(spec["seed"])
        torch.cuda.manual_seed_all(spec["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.source = Path(spec["editor_source"])
        work = self.out / "author-workdir"
        work.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load((self.source / "globals.yml").read_text())
        config.update(
            RESULTS_DIR=str(self.out),
            DATA_DIR=str(self.out / "author-data"),
            STATS_DIR=spec["stats_dir"],
            HPARAMS_DIR=str(self.source / "hparams"),
            KV_DIR=str(self.out / "author-kv"),
        )
        (work / "globals.yml").write_text(yaml.safe_dump(config))
        os.chdir(work)
        sys.path.insert(0, str(self.source))
        self.author = importlib.import_module(
            "rome.rome_main" if spec["method"] == "ROME" else "memit.memit_main"
        )
        hparam_class = (
            self.author.ROMEHyperParams
            if spec["method"] == "ROME"
            else self.author.MEMITHyperParams
        )
        self.hparams = hparam_class.from_json(spec["hparams_file"])
        write_json(self.out / "resolved-hparams.json", vars(self.hparams))
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model_dir"], local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"  # The authors' editing token indices assume this.
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                spec["model_dir"],
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(device)
        )
        # Cache naming is based on the canonical model id, not the local download directory.
        self.model.config._name_or_path = spec["model_name"]
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        compatibility = {"author_edit_padding": "right", "evaluation_padding": "left"}
        if self.model.config.model_type == "gptj":
            sample = self.tokenizer("The capital of France is", return_tensors="pt").to(device)
            with torch.inference_mode():
                before = self.model(**sample).logits.detach().clone()
            for block in self.model.transformer.h:
                block.register_forward_pre_hook(canonical_block_arguments, with_kwargs=True)
            with torch.inference_mode():
                after = self.model(**sample).logits
            assert torch.equal(before, after), "Input routing must not change the Transformer"
            compatibility.update(keyword_input_hook=True, full_model_logits_unchanged=True)
        write_json(self.out / "compatibility-audit.json", compatibility)
        self.generation = Generation(self.model, self.tokenizer, spec, device)
        self.parameters = dict(self.model.named_parameters())
        self.names = {
            self.hparams.rewrite_module_tmp.format(layer) + ".weight"
            for layer in self.hparams.layers
        }
        self.baseline = {name: self.parameters[name].detach().cpu().clone() for name in self.names}
        self.edited = False
        self.optimizer_updates = self.edit_requests = self.rollback_count = 0
        self.before_hashes = self.parameter_hashes()
        write_json(self.out / "base-parameter-hashes.json", self.before_hashes)
        write_json(self.out / "resolved-model-config.json", self.model.config.to_dict())
        print(
            json.dumps(
                {
                    "state": "model_loaded",
                    "model": spec["model_name"],
                    "layers": sorted(self.hparams.layers),
                }
            ),
            flush=True,
        )

    def parameter_hashes(self):
        import hashlib

        hashes = {}
        for name, parameter in self.parameters.items():
            array = parameter.detach().cpu().contiguous().numpy()
            hashes[name] = hashlib.sha256(memoryview(array)).hexdigest()
        return hashes

    def assert_original(self):
        import torch

        assert not self.edited
        for name, baseline in self.baseline.items():
            assert torch.equal(self.parameters[name].detach().cpu(), baseline), name

    def edit(self, requests, destination):
        import torch

        self.assert_original()
        assert self.tokenizer.padding_side == "right", "Keep the author's editing token positions"
        captured = []
        name = "execute_rome" if self.spec["method"] == "ROME" else "execute_memit"
        original_execute = getattr(self.author, name)
        original_adam = torch.optim.Adam.step

        def observe_execute(*args, **kwargs):
            deltas = original_execute(*args, **kwargs)
            captured.append(
                {k: tuple(v.detach().cpu().clone() for v in pair) for k, pair in deltas.items()}
            )
            return deltas

        def observe_adam(optimizer, *args, **kwargs):
            result = original_adam(optimizer, *args, **kwargs)
            self.optimizer_updates += 1
            return result

        setattr(self.author, name, observe_execute)
        torch.optim.Adam.step = observe_adam
        try:
            apply = (
                self.author.apply_rome_to_model
                if self.spec["method"] == "ROME"
                else self.author.apply_memit_to_model
            )
            returned_model, original_weights = apply(
                self.model, self.tokenizer, requests, self.hparams, return_orig_weights=True
            )
            assert returned_model is self.model and set(original_weights) == self.names
            for key, old in original_weights.items():
                assert torch.equal(old.detach().cpu(), self.baseline[key]), key
            for key in self.names:
                assert torch.isfinite(self.parameters[key]).all().item(), key
            assert captured
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(captured, destination)
            write_json(
                destination.with_suffix(".json"),
                {
                    "requests": requests,
                    "sha256": sha(destination),
                    "sequential_updates": len(captured),
                },
            )
            context = self.author.CONTEXT_TEMPLATES_CACHE
            if context is not None and not (self.out / "author-context-templates.json").exists():
                write_json(self.out / "author-context-templates.json", context)
            self.edit_requests += len(requests)
            self.edited = self.generation.edited = True
            del original_weights
            return {"file": str(destination), "sha256": sha(destination)}
        except BaseException:
            self.edited = True
            self.restore()
            raise
        finally:
            setattr(self.author, name, original_execute)
            torch.optim.Adam.step = original_adam

    def replay(self, delta_file, expected_sha):
        import torch

        self.assert_original()
        assert sha(delta_file) == expected_sha
        deltas = torch.load(delta_file, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for delta in deltas:
                assert set(delta) == self.names
                for name, pair in delta.items():
                    left, right = (v.to(self.device) for v in pair)
                    update = (
                        left.unsqueeze(1) @ right.unsqueeze(0)
                        if self.spec["method"] == "ROME"
                        else left @ right.T
                    )
                    weight = self.parameters[name]
                    update = self.author.upd_matrix_match_shape(update, weight.shape)
                    weight[...] += update.float() if self.spec["method"] == "MEMIT" else update
        self.edited = self.generation.edited = True

    def restore(self):
        import torch

        with torch.no_grad():
            for name, original in self.baseline.items():
                self.parameters[name][...] = original.to(self.device)
        self.edited = self.generation.edited = False
        self.assert_original()
        self.rollback_count += 1

    def finish(self):
        self.assert_original()
        assert self.parameter_hashes() == self.before_hashes, "An unedited parameter changed"
        write_json(
            self.out / "restoration-audit.json",
            {
                "passed": True,
                "exact_rollbacks": self.rollback_count,
                "all_parameter_hashes_unchanged": True,
            },
        )

    def release_covariances(self):
        import torch

        for module_name in ("rome.compute_u", "memit.memit_main"):
            module = sys.modules.get(module_name)
            if module is not None:
                for attr in ("COV_CACHE", "COV_INV_CACHE", "INV_COV_CACHE", "inv_mom2_cache"):
                    cache = getattr(module, attr, None)
                    if isinstance(cache, dict):
                        cache.clear()
        gc.collect()
        torch.cuda.empty_cache()


def start_run(spec, out, device):
    import torch

    out = Path(out)
    assert not (out / "run.json").exists(), "Do not silently restart an existing experiment"
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "run.json",
        {
            "spec": spec,
            "model": spec["model_name"],
            "tracking_group": spec["tracking_group"],
            "job_type": "paper-reproduction",
            "tags": ["paper-reproduction", spec["benchmark"], spec["method"]],
            "dtype": "FP32; original author target-vector optimizer and covariance precision",
            "gpu_name": torch.cuda.get_device_name(device),
            "initial_model_sha256": spec["model_weights_sha256"],
            "world_sha256": spec["data_sha256"],
            "learning_step_unit": (
                "independent editing case; Adam steps optimize the edit target vector"
            ),
        },
    )
    write_json(out / "learning.json", [])
    write_json(out / "status.json", {"state": "loading_original_model", "step": 0})


def record_progress(out, runtime, history, step, metrics):
    record = {
        "step": step,
        "optimizer_updates": runtime.optimizer_updates,
        "examples": step,
        "edit_requests": runtime.edit_requests,
        "generation_queries": runtime.generation.queries,
        "executed_input_tokens": runtime.generation.input_tokens,
        "generated_tokens": runtime.generation.output_tokens,
        "exact_rollbacks": runtime.rollback_count,
        "wall_seconds": time.monotonic() - runtime.started,
        "metrics": metrics,
    }
    history.append(record)
    write_json(Path(out) / "learning.json", history)
    write_json(Path(out) / "status.json", {"state": "editing_and_evaluating", **record})
    print(json.dumps(record), flush=True)


def replay_calls(runtime, calls):
    count = 0
    for call in calls:
        assert call["edited"] == runtime.edited
        items = runtime.generation.generate(
            [row["prompt"] for row in call["items"]],
            max_new_tokens=call["max_new_tokens"],
            mode=call["mode"],
        )
        assert [r["generated_token_ids"] for r in items] == [
            r["generated_token_ids"] for r in call["items"]
        ], "Fresh-process replay changed raw generation"
        count += len(items)
    return count
