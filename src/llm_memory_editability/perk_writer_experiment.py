"""Calibrate and run the authorized fixed-reader writer comparison.

The original PERK remains immutable. The complete author model, Higher AdamW,
and four-step write unroll are reused. This controlled small-world extension is
not a BabiLong paper reproduction and does not test persistent sequential edits.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from llm_memory_editability.perk_reproduction import (
    load_author,
    parameter_digest,
    resolve_config,
    seed_all,
    sha256,
    write_json,
)
from llm_memory_editability.perk_writer_data import OBJECTS, validate_world


def read_rows(folder: Path, split: str) -> list[dict]:
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / f"{split}.jsonl"
    if sha256(path) != manifest["files"][split]["sha256"]:
        raise ValueError(f"Changed {split} data")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != manifest["files"][split]["rows"]:
        raise ValueError("Pool coverage changed")
    for row in rows:
        validate_world(row)
    return rows


def load_writer(plan: dict, output: Path, inner_rate: float):
    import torch
    from omegaconf import OmegaConf

    from llm_memory_editability.perk_writer_control import ControlledPERKWriter

    seed_all(plan["control_initialization_seed"])
    parent = Path(plan["prerequisite"])
    cfg = json.loads((parent / "frozen/configs/perk.json").read_text())
    cfg.update(
        learning_rate=plan["outer_learning_rate"],
        inner_lr=inner_rate,
        max_new_tokens=plan["max_new_tokens"],
        num_workers=0,
    )
    config = resolve_config(cfg, output)
    config.update(
        reproduction_kind="controlled_writer_extension",
        paper_target="Fixed-reader writer comparison, not original BabiLong reproduction",
    )
    author = load_author(plan["source"])

    class ExplicitWorldModule(author.MetaLMModule):
        def load_dataset(self):
            pass

    module = ExplicitWorldModule(OmegaConf.create(config))
    selection = json.loads((parent / "runs/perk/checkpoint-selection.json").read_text())
    checkpoint = Path(selection["checkpoint"])
    expected = selection["files"].get(str(checkpoint))
    if expected is None or sha256(checkpoint) != expected:
        raise ValueError("Selected parent checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(saved["state_dict"], strict=True)
    del saved
    # Keep the trained reader/LoRA initialization. All three conditions start
    # with the same fresh uniform writer, not inherited learned token weights.
    seed_all(plan["control_initialization_seed"])
    module.model.weighting_model.fc1.reset_parameters()
    with torch.no_grad():
        module.model.weighting_model.fc2.weight.zero_()
        module.model.weighting_model.fc2.bias.zero_()
        for rate in module.inner_schedular.parameters():
            rate.fill_(inner_rate)
    writer = ControlledPERKWriter(module)
    writer.to("cuda")
    writer.eval()
    return writer, {
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": expected,
        "initial_reader_sha256": parameter_digest(module.model.lm_model),
        "initial_writer_sha256": parameter_digest(writer),
    }


def collate(writer, world: dict, queries: list[dict]) -> dict:
    tokenizer = writer.author_module.tokenizer
    prefix = f"You have memorized {len(world['context'])} documents."
    materials = [(f"doc_{index}:", text) for index, text in enumerate(world["context"])]
    outer = [
        (
            f"{prefix}\n{q['question']}\nAnswer:",
            f"<support>{chr(10).join(q['support'])}</support>\n<output>{q['answer']}</output>",
        )
        for q in queries
    ]
    result = encode_training_rows(tokenizer, outer)
    result.update(
        {f"train_{key}": value for key, value in encode_training_rows(tokenizer, materials).items()}
    )
    return result


def encode_training_rows(tokenizer, sequences: list[tuple[str, str]]) -> dict:
    """Author text format with right padding and no truncation in this extension.

    The extension has three outer questions per step. Right padding keeps GPT-2
    absolute prompt positions independent of other questions' support lengths.
    Padding and prompt tokens are masked; material/answer and EOS stay supervised.
    """
    import torch

    rows = []
    for prompt, completion in sequences:
        ids = tokenizer.encode(
            f"{prompt} {completion}{tokenizer.eos_token}", add_special_tokens=True
        )
        prefix = tokenizer.encode(prompt, add_special_tokens=True)
        if ids[: len(prefix)] != prefix or len(ids) > tokenizer.model_max_length:
            raise ValueError("Prompt boundary or complete reader-window contract failed")
        rows.append((ids, len(prefix)))
    width = max(len(ids) for ids, _ in rows)
    inputs = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
    masks = torch.zeros_like(inputs)
    labels = torch.full_like(inputs, -100)
    for index, (ids, prefix_len) in enumerate(rows):
        inputs[index, : len(ids)] = torch.tensor(ids)
        masks[index, : len(ids)] = 1
        labels[index, prefix_len : len(ids)] = inputs[index, prefix_len : len(ids)]
    return {"input_ids": inputs, "attention_mask": masks, "labels": labels}


def paired_batch(writer, world: dict, step: int):
    from llm_memory_editability.perk_writer_control import PairedWriterBatch

    by_id = {q["id"]: q for q in world["atomic"]}
    # Alternate the two atomic relations, with exactly three outer questions.
    keys = [
        f"owner:{obj}" if step % 2 == 0 else f"location:{world['owners'][obj]}" for obj in OBJECTS
    ]
    first = collate(writer, world, [by_id[key] for key in keys])
    second = collate(writer, world, world["compositional"])
    first = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in first.items()}
    second = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in second.items()}
    return PairedWriterBatch.from_author_batches(first, second, material_id=world["world_id"])


def all_queries(world: dict, alternate: bool = True) -> list[dict]:
    queries = [dict(q, phrasing="standard") for q in world["atomic"] + world["compositional"]]
    if alternate:
        queries.extend(
            dict(q, question=q["alternate"], id=q["id"] + ":alternate", phrasing="alternate")
            for q in world["atomic"] + world["compositional"]
        )
    return queries


def summarize(records: list[dict]) -> dict:
    groups = {}
    for record in records:
        key = f"{record['kind']}/{record['phrasing']}"
        values = groups.setdefault(key, {"correct": 0, "queries": 0, "worlds": set()})
        values["correct"] += record["correct"]
        values["queries"] += 1
        values["worlds"].add(record["world_id"])
    return {
        key: {
            "correct": value["correct"],
            "queries": value["queries"],
            "worlds": len(value["worlds"]),
            "accuracy": value["correct"] / value["queries"],
        }
        for key, value in groups.items()
    }


def read_world(writer, world: dict, *, reference: bool, max_new_tokens: int) -> list[dict]:
    import torch

    from llm_memory_editability.perk_writer_control import AdaptedPERKReadView

    queries = all_queries(world, alternate=not reference)
    tokenizer = writer.author_module.tokenizer
    prompts = [
        f"You have memorized {len(world['context'])} documents.\n{q['question']}\nAnswer:"
        for q in queries
    ]
    if reference:
        prompts = ["\n".join(world["context"]) + "\n" + prompt for prompt in prompts]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    encoded = {k: v.cuda() for k, v in encoded.items()}
    seed_all(int(world["world_id"][:8], 16))
    material_batch = collate(writer, world, world["compositional"])
    material = {
        k: material_batch[f"train_{k}"].cuda() for k in ("input_ids", "attention_mask", "labels")
    }
    evaluator = importlib.import_module("perk.evaluate")

    def generate(view):
        before = view.adapter_values()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = view.generate(
                encoded["input_ids"],
                encoded["attention_mask"],
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        after = view.adapter_values()
        if before.keys() != after.keys() or any(
            not torch.equal(v, after[k]) for k, v in before.items()
        ):
            raise RuntimeError("Read-only questions mutated the fast adapter")
        return ids

    if reference:
        # Explicit materials test the common reader's operation, not writing.
        ids = generate(AdaptedPERKReadView(writer.author_module.model, torch.tensor(0.0)))
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16), writer.adapt_material(material) as view:
            ids = generate(view)
    records = []
    for q, prompt, tokens in zip(queries, prompts, ids, strict=True):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        target = f"<support>{chr(10).join(q['support'])}</support>\n<output>{q['answer']}</output>"
        native = {"guid": world["world_id"], "prompt": prompt, "answer": target, "output": text}
        metrics, _ = evaluator.evaluate([native])
        correct = metrics["accuracy"] == 1.0
        records.append(
            {
                "world_id": world["world_id"],
                "query_id": q["id"],
                "kind": q["kind"],
                "phrasing": q["phrasing"],
                "held_binding": q["held_binding"],
                "prerequisites": q.get("prerequisites", []),
                "prompt": prompt,
                "answer": q["answer"],
                "scoring_target": target,
                "output": text,
                "token_ids": tokens,
                "correct": bool(correct),
                "reference_material_in_query": reference,
            }
        )
    return records


def evaluate(
    writer, rows: list[dict], destination: Path, *, reference: bool, max_new_tokens: int
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "predictions.jsonl"
    if output.exists():
        raise FileExistsError(output)
    records = []
    start = time.monotonic()
    before = parameter_digest(writer)
    with output.open("w") as handle:
        for index, world in enumerate(rows):
            values = read_world(writer, world, reference=reference, max_new_tokens=max_new_tokens)
            for record in values:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.extend(values)
            write_json(
                destination / "progress.json", {"worlds_completed": index + 1, "worlds": len(rows)}
            )
    if before != parameter_digest(writer):
        raise RuntimeError("Evaluation changed persistent reader or writer parameters")
    result = {
        "passed": True,
        "worlds": len(rows),
        "queries": len(records),
        "generated_tokens": sum(len(row["token_ids"]) for row in records),
        "metrics": summarize(records),
        "predictions_sha256": sha256(output),
        "seconds": time.monotonic() - start,
        "persistent_parameters_unchanged": True,
        "reference_material_in_query": reference,
    }
    write_json(destination / "summary.json", result)
    return result


def calibration(plan: dict, base: Path) -> dict:
    import gc

    import torch

    rows = read_rows(base / "data", "development")
    trials = []
    selected = None
    for index, rate in enumerate(plan["calibration_inner_rates"]):
        folder = base / "calibration" / f"rate-{index}"
        writer, provenance = load_writer(plan, folder, rate)
        reference = evaluate(
            writer,
            rows,
            folder / "reference",
            reference=True,
            max_new_tokens=plan["max_new_tokens"],
        )
        baseline = evaluate(
            writer,
            rows,
            folder / "baseline",
            reference=False,
            max_new_tokens=plan["max_new_tokens"],
        )
        kinds = ("atomic_location", "atomic_owner", "compositional")
        reference_ok = all(
            reference["metrics"][f"{kind}/standard"]["accuracy"]
            >= plan["reader_reference_min_accuracy"]
            for kind in kinds
        )
        atomic_ok = all(
            baseline["metrics"][f"{kind}/standard"]["accuracy"]
            >= plan["baseline_atomic_min_accuracy"]
            for kind in kinds[:2]
        )
        trial = {
            "inner_rate": rate,
            "reference_ok": reference_ok,
            "atomic_write_ok": atomic_ok,
            "reference": reference,
            "baseline": baseline,
            "provenance": provenance,
        }
        trials.append(trial)
        write_json(folder / "trial.json", trial)
        del writer
        gc.collect()
        torch.cuda.empty_cache()
        # First satisfactory fact-writing setting; never choose by propagation.
        if reference_ok and atomic_ok:
            selected = rate
            break
        if not reference_ok:
            # Changing write rate cannot fix a failed unadapted reader reference.
            break
    result = {
        "passed": selected is not None,
        "selected_inner_rate": selected,
        "trials": trials,
        "selection_rule": (
            "first reader-reference and per-role atomic-write pass; no composed-score selection"
        ),
        "scientific_training_started": False,
    }
    write_json(base / "calibration-result.json", result)
    return result


def controls(writer) -> dict:
    return {name: param.detach().cpu().clone() for name, param in writer.outer_named_parameters()}


def validation_losses(writer, rows: list[dict], destination: Path) -> dict:
    """Monitor a reserved pool with teacher forcing after query-free writing."""
    import torch

    destination.mkdir(parents=True)
    values = []
    before = parameter_digest(writer)
    for world in rows:
        seed_all(int(world["world_id"][:8], 16))
        batch = collate(writer, world, world["atomic"])
        material = {
            key: batch[f"train_{key}"].cuda() for key in ("input_ids", "attention_mask", "labels")
        }
        losses = {}
        with torch.autocast("cuda", dtype=torch.bfloat16), writer.adapt_material(material) as view:
            for family in ("atomic", "compositional"):
                outer = collate(writer, world, world[family])
                checked = {
                    key: outer[key].cuda() for key in ("input_ids", "attention_mask", "labels")
                }
                losses[family] = float(view.read(checked)["loss"])
        values.append({"world_id": world["world_id"], **losses})
    if before != parameter_digest(writer):
        raise RuntimeError("Validation modified persistent state")
    (destination / "world-losses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in values)
    )
    result = {
        "worlds": len(rows),
        "checkpoint_selection": False,
        "write_subsequence_presentations": len(rows) * 3 * 4,
        **{
            f"validation_{family}_loss": sum(row[family] for row in values) / len(values)
            for family in ("atomic", "compositional")
        },
    }
    write_json(destination / "summary.json", result)
    return result


def restore_controls(writer, state: dict) -> None:
    import torch

    params = dict(writer.outer_named_parameters())
    if params.keys() != state.keys():
        raise ValueError("Control checkpoint fields changed")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(param.device))


def run_arm(
    plan: dict,
    base: Path,
    condition: str,
    seed: int,
    *,
    stage: str = "science",
    updates: int | None = None,
    limit_worlds: int | None = None,
) -> dict:
    import random

    import torch
    import wandb

    calibration_result = json.loads((base / "calibration-result.json").read_text())
    if not calibration_result["passed"]:
        raise ValueError("Reader/write calibration did not pass")
    task = f"{condition}-seed{seed}"
    root = base / ("runs" if stage == "science" else "smoke") / task
    root.mkdir(parents=True, exist_ok=True)
    writer, provenance = load_writer(plan, root, calibration_result["selected_inner_rate"])
    start = time.monotonic()
    training = read_rows(base / "data", "train")
    validation = read_rows(base / "data", "validation")
    test = read_rows(base / "data", "development" if stage != "science" else "test")
    if limit_worlds is not None:
        test = test[:limit_worlds]
    optimizer = writer.outer_optimizer()
    budget = (
        0
        if condition == "fixed"
        else (updates if updates is not None else plan["outer_updates_per_arm"])
    )
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        id=f"{plan['batch']}-{stage}-{task}",
        name=f"{plan['batch']}-{stage}-{task}",
        config={
            **plan,
            "condition": condition,
            "training_seed": seed,
            "stage": stage,
            "inner_rate": calibration_result["selected_inner_rate"],
            "actual_outer_budget": budget,
            **provenance,
        },
    )
    exposure = {}
    history = root / "history.jsonl"
    if history.exists():
        raise FileExistsError(history)
    try:
        order_rng = random.Random(seed)
        order = []
        for _ in range(plan["outer_epochs"]):
            order.extend(order_rng.sample(range(len(training)), len(training)))
        if budget > len(order):
            raise ValueError("Outer budget exceeds the declared material stream")
        with history.open("w") as handle:
            for step in range(budget):
                seed_all(seed * 100000 + step)
                batch = paired_batch(writer, training[order[step]], step)
                writer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = writer.meta_step(batch, condition)
                output["loss"].backward()
                grads = [p.grad for _, p in writer.outer_named_parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError("Write-control meta gradients are missing or nonfinite")
                torch.nn.utils.clip_grad_norm_([p for _, p in writer.outer_named_parameters()], 1.0)
                optimizer.step()
                factor = min((step + 1) / max(1, int(budget * 0.03)), 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = plan["outer_learning_rate"] * factor
                if not all(torch.isfinite(p).all() for _, p in writer.outer_named_parameters()):
                    raise RuntimeError("Write-control values are nonfinite")
                for key, value in output["exposure"].items():
                    if isinstance(value, int):
                        exposure[key] = exposure.get(key, 0) + value
                record = {
                    "outer_updates": step + 1,
                    "outer_loss": float(output["outer_loss"]),
                    "inner_loss": float(output["inner_loss"]),
                    "material_id": output["exposure"]["material_id"],
                    "material_sha256": output["exposure"]["material_sha256"],
                    "outer_sha256": output["exposure"]["outer_sha256"],
                    **exposure,
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                run.log(record, step=step + 1)
                write_json(root / "progress.json", {**record, "budget": budget})
                if (step + 1) % plan["validation_interval"] == 0:
                    torch.save(
                        {"controls": controls(writer), "outer_updates": step + 1},
                        root / f"controls-step{step + 1}.pt",
                    )
                    monitored = validation_losses(
                        writer, validation, root / f"validation-step{step + 1}"
                    )
                    run.log(
                        {
                            key: value
                            for key, value in monitored.items()
                            if key.startswith("validation_")
                        },
                        step=step + 1,
                    )
                del output, batch
        if provenance["initial_reader_sha256"] != parameter_digest(
            writer.author_module.model.lm_model
        ):
            raise RuntimeError("Outer training changed the fixed reader or LoRA initialization")
        torch.save(
            {"controls": controls(writer), "outer_updates": budget, "provenance": provenance},
            root / "controls-final.pt",
        )
        summary = evaluate(
            writer,
            test,
            root / "evaluation",
            reference=False,
            max_new_tokens=plan["max_new_tokens"],
        )
        result = {
            "passed": True,
            "task": task,
            "stage": stage,
            "outer_updates": budget,
            "exposure": exposure,
            "evaluation": summary,
            "provenance": provenance,
            "checkpoint_sha256": sha256(root / "controls-final.pt"),
            "seconds": time.monotonic() - start,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "fixed_reader_unchanged": True,
            "wandb_run_id": run.id,
            "wandb_url": run.url,
        }
        run.summary.update(
            {
                "outer_updates": budget,
                "fixed_reader_unchanged": True,
                "evaluation_worlds": summary["worlds"],
                "evaluation_queries": summary["queries"],
            }
        )
        for key, metric in summary["metrics"].items():
            run.summary[f"eval/{key}/accuracy"] = metric["accuracy"]
        write_json(root / "science-complete.json", result)
        return result
    finally:
        run.finish()


def replay_arm(
    plan: dict, base: Path, condition: str, seed: int, *, stage: str = "science"
) -> dict:
    import torch

    task = f"{condition}-seed{seed}"
    root = base / ("runs" if stage == "science" else "smoke") / task
    original = json.loads((root / "science-complete.json").read_text())
    checkpoint = root / "controls-final.pt"
    if sha256(checkpoint) != original["checkpoint_sha256"]:
        raise ValueError("Scientific control checkpoint changed")
    rate = json.loads((base / "calibration-result.json").read_text())["selected_inner_rate"]
    writer, provenance = load_writer(plan, root / "replay", rate)
    if provenance != original["provenance"]:
        raise ValueError("Replay did not start from the same reader and writer initialization")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restore_controls(writer, state["controls"])
    rows = read_rows(base / "data", "test" if stage == "science" else "development")
    if stage != "science":
        rows = rows[: original["evaluation"]["worlds"]]
    summary = evaluate(
        writer, rows, root / "replay", reference=False, max_new_tokens=plan["max_new_tokens"]
    )
    for key in ("worlds", "queries", "generated_tokens", "metrics", "predictions_sha256"):
        if summary[key] != original["evaluation"][key]:
            raise ValueError(f"Independent replay differs: {key}")
    result = {
        "passed": True,
        "checkpoint_sha256": sha256(checkpoint),
        "predictions_sha256": summary["predictions_sha256"],
        "worlds": summary["worlds"],
        "queries": summary["queries"],
        "tokens_and_scores_exact": True,
    }
    write_json(root / "audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "run", "replay"])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--condition", choices=["atomic", "compositional", "fixed"])
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--stage", choices=["science", "smoke"], default="science")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--limit-worlds", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.stage == "science" and (args.updates is not None or args.limit_worlds is not None):
        parser.error("Development limits cannot enter the scientific run")
    if args.mode == "calibrate":
        import wandb

        run = wandb.init(
            entity="zhusq20",
            project="llm-memory-editability",
            id=f"{plan['batch']}-calibration",
            name=f"{plan['batch']}-calibration",
            config={**plan, "stage": "reader_and_write_calibration"},
        )
        try:
            result = calibration(plan, args.base)
            run.summary.update(
                {
                    "calibration_passed": result["passed"],
                    "selected_inner_rate": result["selected_inner_rate"],
                    "scientific_training_started": False,
                }
            )
        finally:
            run.finish()
        if not result["passed"]:
            raise SystemExit(3)
    elif args.mode == "run":
        result = run_arm(
            plan,
            args.base,
            args.condition,
            args.seed,
            stage=args.stage,
            updates=args.updates,
            limit_worlds=args.limit_worlds,
        )
    else:
        result = replay_arm(plan, args.base, args.condition, args.seed, stage=args.stage)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
