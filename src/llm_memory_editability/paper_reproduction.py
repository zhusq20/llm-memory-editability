"""Run Ye et al. Figure 3 with the authors' notebook and training implementation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

from .experiment_tracking import read_json, write_json

TOKEN = re.compile(r"<e_\d+>|<r_\d+>|</a>")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def notebook_functions(notebook):
    """Execute only the original function definitions; no notebook file writes."""
    import numpy as np
    from tqdm.auto import tqdm

    cells = read_json(notebook)["cells"]
    namespace = {"np": np, "tqdm": tqdm}
    for index in (1, 3):
        parsed = ast.parse("".join(cells[index]["source"]))
        definitions = ast.Module(
            body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)],
            type_ignores=[],
        )
        exec(compile(definitions, f"{notebook}:cell-{index}", "exec"), namespace)
    return namespace


def validate_data(groups, training, *, expected_id, expected_ood, expected_ii):
    """Check graph truth, structural roles, held-out queries and the actual sample count."""
    id_rows, ood_rows = groups["ID Triples"], groups["OOD Triples"]
    assert len(id_rows) == expected_id and len(ood_rows) == expected_ood
    atomic = {}
    membership = {}
    for kind, rows in (("I", id_rows), ("O", ood_rows)):
        for row in rows:
            parts = TOKEN.findall(row["target_text"])
            assert len(parts) == 4 and parts[-1] == "</a>"
            h, relation, tail, _ = parts
            key = (h, relation)
            assert key not in atomic
            assert row["input_text"] == h + relation
            atomic[key], membership[key] = tail, kind
    assert len(training["only_ii"]) == expected_ii
    assert training["ii_plus_id"] == id_rows + training["only_ii"]
    train_queries = {row["input_text"] for row in training["only_ii"]}
    assert len(train_queries) == expected_ii
    assert all(len(TOKEN.findall(row["input_text"])) == 3 for row in training["only_ii"])
    checked = 0
    for name, rows in groups.items():
        if name in {"ID Triples", "OOD Triples"}:
            continue
        role = "II" if name == "Train-II" else name.removeprefix("Test-")
        seen = set()
        for row in rows:
            h, r1, r2 = TOKEN.findall(row["input_text"])
            bridge = atomic[(h, r1)]
            tail = atomic[(bridge, r2)]
            assert row["target_text"] == h + r1 + r2 + tail + "</a>"
            assert membership[(h, r1)] + membership[(bridge, r2)] == role
            assert row["input_text"] not in seen
            seen.add(row["input_text"])
            assert (row["input_text"] in train_queries) == (name == "Train-II")
            checked += 1
    return {"passed": True, "atomic_truths": len(atomic), "composition_truths": checked}


def prepare(notebook, destination, *, seed=42, entities=2000, relations=200, degree=20):
    import numpy as np

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    assert os.environ.get("PYTHONHASHSEED") == "0", "Fix the authors' set iteration order"
    random.seed(seed)
    np.random.seed(seed)
    functions = notebook_functions(notebook)
    values = functions["build_base_dataset"](entities, relations, out_degree=degree)
    entity_tokens, relation_tokens, id_rows, ood_rows, ii, test_ii, io, oi, oo = values
    choose = functions["choose"]
    # Notebook passes 7.2 * N as float, which choose() interprets as a ratio.
    # Table 2 specifies 273600. Pass an integer COUNT rather than changing phi.
    expected_ii = round(7.2 * len(id_rows))
    candidate_count = len(ii)
    evaluation = {
        "ID Triples": choose(id_rows, 3000),
        "OOD Triples": choose(ood_rows, 3000),
        "Test-II": choose(test_ii, 3000),
        "Test-IO": choose(io, 3000),
        "Test-OI": choose(oi, 3000),
        "Test-OO": choose(oo, 3000),
    }
    ii = choose(ii, expected_ii)
    evaluation["Train-II"] = choose(ii, 3000)
    training = {"only_ii": ii, "ii_plus_id": id_rows + ii}
    full_groups = {"ID Triples": id_rows, "OOD Triples": ood_rows, "Train-II": ii}
    full_groups.update(
        {name: rows for name, rows in evaluation.items() if name.startswith("Test-")}
    )
    audit = validate_data(
        full_groups,
        training,
        expected_id=round(entities * degree * 0.95),
        expected_ood=round(entities * degree * 0.05),
        expected_ii=expected_ii,
    )
    probes = [dict(row, type=name) for name, rows in evaluation.items() for row in rows]
    vocab = entity_tokens + relation_tokens + ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
    for arm, rows in training.items():
        for filename, data in {
            "train.json": rows,
            "valid.json": evaluation["Test-OO"],
            "test.json": probes,
            "vocab.json": vocab,
        }.items():
            write_json(destination / arm / filename, data)
    write_json(destination / "atomic-truth.json", full_groups["ID Triples"] + ood_rows)
    audit.update(
        seed=seed,
        python_hash_seed=0,
        notebook_sha256=sha(notebook),
        candidate_train_ii=candidate_count,
        selected_train_ii=expected_ii,
        training_counts={arm: len(rows) for arm, rows in training.items()},
        evaluation_counts={name: len(rows) for name, rows in evaluation.items()},
        same_ii_and_probes=True,
        integer_count_fix="Paper Table 2 rather than notebook float-as-ratio sampling",
        files={
            str(path.relative_to(destination)): sha(path) for path in destination.rglob("*.json")
        },
    )
    write_json(destination / "data-audit.json", audit)
    return audit


def score_predictions(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(
            row["type"], {"total": 0, "paper_correct": 0, "strict_correct": 0}
        )
        group["total"] += 1
        group["paper_correct"] += row["model_output"] == row["target_text"]
        group["strict_correct"] += row["strict_answer_eos"]
    for group in groups.values():
        group["paper_accuracy"] = group["paper_correct"] / group["total"]
        group["strict_answer_eos_accuracy"] = group["strict_correct"] / group["total"]
    return groups


def install_author_imports():
    import sys

    # The fixed image contains the authors' two bundled packages, not stock HF GPT-2.
    sys.path[:0] = [
        "/opt/paper-upstream/transformers/src",
        "/opt/paper-upstream/simpletransformers",
    ]
    import transformers

    assert transformers.__version__ == "4.37.0.dev0"
    assert str(Path(transformers.__file__)).startswith("/opt/paper-upstream/")


@contextmanager
def decode_unmapped_tokens(tokenizer):
    """Make padded output classes printable failures without changing generation logits."""
    original = tokenizer._convert_id_to_token

    def convert(index):
        token = original(index)
        return token if token is not None else f"<unmapped_{index}>"

    tokenizer._convert_id_to_token = convert
    try:
        yield
    finally:
        tokenizer._convert_id_to_token = original


def predict_with_raw(model, probes, destination):
    """Keep original generation/scoring and also preserve the unmodified token outputs."""
    module = model.model.module if hasattr(model.model, "module") else model.model
    generate = module.generate
    raw = []

    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        raw.extend(result.detach().cpu().tolist())
        return result

    module.generate = capture
    try:
        # Original predict() modifies its input rows in place.
        original = getattr(model, "predict_original", model.predict)
        with decode_unmapped_tokens(model.lm_tokenizer):
            original([dict(row) for row in probes], str(destination))
    finally:
        module.generate = generate
    rows = read_json(Path(destination) / "all_items.json")
    assert len(raw) == len(rows)
    for row, ids in zip(rows, raw, strict=True):
        unpadded = [token for token in ids if token != model.lm_tokenizer.pad_token_id]
        expected = model.lm_tokenizer.encode(row["target_text"])
        # Stop at the answer delimiter; tokens afterwards are outside the task.
        row["strict_answer_eos"] = unpadded[: len(expected)] == expected
        row["raw_token_ids"] = ids
        with decode_unmapped_tokens(model.lm_tokenizer):
            row["raw_output"] = model.lm_tokenizer.decode(ids, skip_special_tokens=False)
    write_json(Path(destination) / "raw-predictions.json", rows)
    return score_predictions(rows)


class Observer:
    """Record updates without changing the authors' optimizer, gradients or data order."""

    def __init__(self, out, spec, training_count):
        self.out, self.spec = Path(out), spec
        self.started = time.monotonic()
        self.evaluation_seconds = 0.0
        self.examples = self.atomics = self.compositions = 0
        self.history = []
        self.step = 0
        self.skipped_updates = 0
        self.training_count = training_count
        self.latest_training = {}
        self.scaler_state = None

    def after_update(self, model, step, inputs, loss, optimizer, scheduler, scaler, skipped):
        self.step = step
        count = len(inputs["target_ids"])
        atomic_count = int((inputs["lm_labels"][:, 2] != -100).sum().item())
        self.examples += count
        self.atomics += atomic_count
        self.compositions += count - atomic_count
        finite_loss = float(loss.item())
        assert math.isfinite(finite_loss), f"Non-finite training loss at step {step}"
        self.skipped_updates += int(skipped)
        self.scaler_state = scaler.state_dict() if scaler is not None else None
        self.latest_training = {"loss": finite_loss, "learning_rate": scheduler.get_last_lr()[0]}
        save_steps = self.spec.get("save_steps", 50000)
        dense_limit = self.spec.get("save_step_dense", 40000)
        dense_interval = self.spec.get("save_step_dense_interval", 2000)
        checkpoint = (save_steps > 0 and step % save_steps == 0) or (
            dense_limit > 0 and step <= dense_limit and step % dense_interval == 0
        )
        # Do not publish this step before its evaluation: the sidecar has a monotonic cursor.
        if not checkpoint and (step in {1, 10} or step % self.spec.get("log_steps", 100) == 0):
            self.record(step, **self.latest_training)

    def record(self, step, **values):
        wall = time.monotonic() - self.started
        entry = {
            "step": step,
            "wall_seconds": wall,
            "training_seconds": wall - self.evaluation_seconds,
            "examples": self.examples,
            "atomic_examples": self.atomics,
            "composition_examples": self.compositions,
            "supervised_tokens": 2 * self.examples,
            "executed_input_tokens": 10 * self.examples,
            "data_epochs": self.examples / self.training_count,
            "amp_skipped_updates": self.skipped_updates,
            "successful_updates": step - self.skipped_updates,
            **values,
        }
        # Evaluation at a logging step replaces that record, avoiding duplicate W&B steps.
        if self.history and self.history[-1]["step"] == step:
            entry = {**self.history.pop(), **entry}
        self.history.append(entry)
        write_json(self.out / "learning.json", self.history)
        write_json(self.out / "status.json", {"state": "training", **entry})
        print(json.dumps({"step": step, "examples": self.examples, **values}), flush=True)


def model_args(spec, out):
    return {
        "reprocess_input_data": True,
        "overwrite_output_dir": False,
        "max_seq_length": 10,
        "max_length": 10,
        "max_gen_length": None,
        "block_size": 10,
        "train_batch_size": 1024,
        "eval_batch_size": 512,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "num_train_epochs": 20,
        "save_eval_checkpoints": False,
        "save_steps": spec.get("save_steps", 50000),
        "use_multiprocessing": False,
        "output_dir": str(out / "model-output"),
        "manual_seed": spec["seed"],
        "fp16": True,
        "truncation": True,
        "dataloader_num_workers": 0,
        "use_multiprocessed_decoding": False,
        "save_best_model": False,
        "save_model_every_epoch": False,
        "save_epoch_interval": 0,
        "scheduler": "constant_schedule_with_warmup",
        "weight_decay": 0.1,
        "evaluate_during_training": True,
        "predict_during_training": True,
        "mlm": False,
        "warmup_steps": 2000,
        "max_steps": spec["max_steps"],
        "n_layer": 8,
        "n_inner": None,
        "n_head": None,
        "memory_dim": 1536,
        # File locations and console verbosity only; training defaults remain intact.
        "silent": True,
        "cache_dir": str(out / "cache"),
        "tensorboard_dir": str(out / "tensorboard"),
        "no_cache": True,
    }


def run(spec, out, device):
    import numpy as np
    import pandas as pd
    import torch

    install_author_imports()
    from simpletransformers.seq2seq import Seq2SeqModel

    out = Path(out)
    assert not (out / "model-output").exists(), "No automatic restart or overwrite"
    dataset = Path(spec["dataset"])
    for filename, expected in spec["data_files"].items():
        assert sha(dataset / filename) == expected
    training = read_json(dataset / "train.json")
    probes = read_json(dataset / "test.json")
    observer = Observer(out, spec, len(training))

    class RecordedModel(Seq2SeqModel):
        def predict(self, *args, **kwargs):
            # The inherited train() invokes this only at a saved checkpoint.
            start = time.monotonic()
            destination = Path(args[1] if len(args) > 1 else kwargs["output_dir"])
            metrics = predict_with_raw(self, probes, destination)
            observer.evaluation_seconds += time.monotonic() - start
            observer.record(observer.step, metrics=metrics, **observer.latest_training)
            return metrics

        def save_model(self, output_dir, optimizer=None, scheduler=None, **kwargs):
            super().save_model(output_dir, optimizer, scheduler, **kwargs)
            torch.save(
                {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all(),
                    "step": observer.step,
                    "amp_scaler": observer.scaler_state,
                },
                Path(output_dir) / "rng-state.pt",
            )

    # Avoid override recursion when predict_with_raw calls the authors' predict().
    original_predict = Seq2SeqModel.predict
    RecordedModel.predict_original = original_predict
    model = RecordedModel(
        "gpt2",
        spec["reference_model"],
        args=model_args(spec, out),
        ddp_args={
            "local_rank": -1,
            "rank": -1,
            "gpu": None,
            "world_size": -1,
            "dist_url": "env://",
            "dist_backend": "nccl",
        },
        new_tokens=read_json(dataset / "vocab.json"),
        init_weights=True,
    )
    model.paper_observer = observer
    model._move_model_to_device()
    assert model.language_model.config.n_layer == 8
    assert model.language_model.config.n_embd == 768
    assert model.language_model.config.n_head == 12
    assert model.language_model.config.vocab_size == 52464
    assert not getattr(model.language_model.config, "add_recurrence", False)
    metadata = {
        "spec": spec,
        "model": "Author GPT-2, independent 8 layers",
        "tracking_group": spec.get("tracking_group", "implicit-reasoning-paper-reproduction-v1"),
        "job_type": "paper-reproduction",
        "tags": [
            "paper-reproduction",
            "Ye-et-al-NeurIPS-2025",
            spec.get("original_experiment", "Figure-3"),
        ],
        "parameters": sum(p.numel() for p in model.language_model.parameters()),
        "vocab_size": 52464,
        "dtype": "FP32 parameters, author FP16 AMP",
        "gpu_name": torch.cuda.get_device_name(device),
        "world_sha256": spec["world_sha256"],
        "learning_step_unit": "author iterations; successful updates and AMP skips also recorded",
    }
    torch.save(model.language_model.state_dict(), out / "initial-state.pt")
    metadata["initial_model_sha256"] = sha(out / "initial-state.pt")
    write_json(out / "run.json", metadata)
    write_json(out / "learning.json", [])
    write_json(out / "resolved-model-args.json", vars(model.args))
    write_json(out / "resolved-model-config.json", model.language_model.config.to_dict())
    model.lm_tokenizer.save_pretrained(out / "tokenizer")
    start = time.monotonic()
    metrics = predict_with_raw(model, probes, out / "predictions-0000000")
    observer.evaluation_seconds += time.monotonic() - start
    observer.record(0, metrics=metrics)
    steps, _ = model.train_model(
        train_data=pd.DataFrame(training),
        eval_data=pd.DataFrame(read_json(dataset / "valid.json")),
        test_data=probes,
        output_dir=str(out / "model-output"),
        save_step_dense=spec.get("save_step_dense", 40000),
        save_step_dense_interval=spec.get("save_step_dense_interval", 2000),
        show_running_loss=False,
    )
    assert steps == spec["max_steps"], (steps, spec["max_steps"])
    final = out / "model-output" / f"checkpoint-{steps}"
    assert (final / "raw-predictions.json").exists(), "Final step must be a checkpoint node"
    write_json(out / "complete.json", {"step": steps, "final_checkpoint": str(final)})
    write_json(out / "status.json", {"state": "awaiting_independent_reload", "step": steps})


def audit(out, device):
    """Fresh-process reload of authors' saved final model and the complete frozen test pool."""
    import torch

    install_author_imports()
    from simpletransformers.seq2seq import Seq2SeqModel

    out = Path(out)
    spec = read_json(out / "run.json")["spec"]
    final = Path(read_json(out / "complete.json")["final_checkpoint"])
    model = Seq2SeqModel(
        "gpt2",
        str(final),
        args=model_args(spec, out),
        ddp_args={
            "local_rank": -1,
            "rank": -1,
            "gpu": None,
            "world_size": -1,
            "dist_url": "env://",
            "dist_backend": "nccl",
        },
        init_weights=False,
    )
    model._move_model_to_device()
    predict_with_raw(model, read_json(Path(spec["dataset"]) / "test.json"), out / "reload-audit")
    written = read_json(final / "raw-predictions.json")
    reloaded = read_json(out / "reload-audit/raw-predictions.json")
    assert [row["raw_token_ids"] for row in written] == [row["raw_token_ids"] for row in reloaded]
    assert score_predictions(written) == score_predictions(reloaded)
    assert torch.isfinite(next(model.language_model.parameters())).all().item()
    write_json(
        out / "audit.json",
        {
            "passed": True,
            "step": spec["max_steps"],
            "queries": len(written),
            "raw_generation_exact_match": True,
            "metrics": score_predictions(reloaded),
        },
    )
    write_json(out / "status.json", {"state": "complete", "step": spec["max_steps"]})
