#!/usr/bin/env python3
"""Benchmark physical batches, restore the exact source state, then continue."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import signal
import time
import traceback
from pathlib import Path

import torch
from knowledge_batch_resume import choose_batch, lr_scale, next_indices, validate_checkpoint
from knowledge_checkpoint_io import CheckpointTransport

from llm_memory_editability import knowledge_method_factorial as F


def verify_restoration(model, optimizer, saved):
    parameters = dict(model.named_parameters())
    for name, tensor in saved["weights"].items():
        if not torch.equal(parameters[name].detach().cpu(), tensor):
            raise AssertionError(f"Restored weight differs: {name}")
    actual = optimizer.state_dict()
    if actual["param_groups"] != saved["optimizer"]["param_groups"]:
        raise AssertionError("Restored optimizer groups differ")
    for key, values in saved["optimizer"]["state"].items():
        for field, value in values.items():
            other = actual["state"][key][field]
            equal = (
                torch.equal(value.cpu(), other.cpu())
                if isinstance(value, torch.Tensor)
                else value == other
            )
            if not equal:
                raise AssertionError(f"Restored optimizer state differs: {key}/{field}")
    if set(actual["state"]) != set(saved["optimizer"]["state"]):
        raise AssertionError("Restored optimizer state coverage differs")


def update(model, optimizer, parameters, batch, arm, config, teacher):
    optimizer.zero_grad(set_to_none=True)
    loss = F.batch_loss(model, batch, arm, config, teacher)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, config["max_grad_norm"], error_if_nonfinite=True
    )
    optimizer.step()
    return float(loss.detach()), float(norm)


def benchmark(model, tok, optimizer, parameters, teacher, rows, config, arm, exposure, output, cap):
    from check_knowledge_method_factorial_cuda import input_batch

    pressure, provenance = input_batch(tok, config, arm, True)
    indices = next_indices(len(rows), config["epochs"], config["seed"], exposure, 256)
    # A fixed set for every candidate; no evaluation answers or scores are consulted.
    encoded = [F.encode_sample(tok, rows[i], arm, config["max_sequence_tokens"]) for i in indices]
    measurements = []
    for size in [4, 8, 16, 32, 64, 128]:
        record = {"batch_size": size, "passed": False, "memory_cap_mib": cap}
        try:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            pressure_batch = {
                side: {key: tensor[:1].repeat(size, 1) for key, tensor in values.items()}
                for side, values in pressure.items()
            }
            for _ in range(2):
                update(model, optimizer, parameters, pressure_batch, arm, config, teacher)
            torch.cuda.synchronize()
            record["pressure_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            del pressure_batch
            batches = [
                F.shared.collate(encoded[i : i + size], tok.pad_token_id)
                for i in range(0, len(encoded), size)
            ]
            update(model, optimizer, parameters, batches[0], arm, config, teacher)
            torch.cuda.synchronize()
            began = time.perf_counter()
            for batch in batches:
                update(model, optimizer, parameters, batch, arm, config, teacher)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            record.update(
                passed=True,
                samples=len(encoded),
                seconds=seconds,
                samples_per_second=len(encoded) / seconds,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                engineering_updates=3 + len(batches),
            )
            del batches
        except torch.OutOfMemoryError as error:
            record.update(error="bounded_allocator_oom", detail=str(error)[:500])
        measurements.append(record)
        F.write(
            output,
            {
                "arm": arm,
                "pressure_input": provenance,
                "benchmark_sample_ids": [rows[i]["id"] for i in indices],
                "measurements": measurements,
            },
        )
        if not record["passed"]:
            break
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    selected = choose_batch(measurements)
    return selected, measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--memory-cap-mib", type=int, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--previous-observed-step", type=int, default=0)
    args = parser.parse_args()
    extension = Path(__file__).resolve().parent
    manifest = json.loads((extension / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((extension / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Continuation source changed: {name}")
    cfg = json.loads(args.config.read_text())
    out = args.batch / "runs" / args.arm
    work = args.batch / "batchsize-20261022" / "runs" / args.arm
    work.mkdir(parents=True, exist_ok=True)
    if (work / "started.json").exists() or (out / "complete.json").exists():
        raise FileExistsError("Existing continuation or completed arm must not be restarted")
    F.write(work / "started.json", {"time": time.time(), "source": str(args.source_checkpoint)})
    status = args.batch / "stage-status" / f"{args.arm}-train.json"
    F.write(status, {"state": "running", "stage": "batch_probe", "name": args.arm})
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_mib * 2**20 / total_memory)
    F.shared.seed_all(cfg["seed"])
    rows_path = args.batch / "shared/train.jsonl"
    data_hash = F.shared.digest(rows_path)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows) != cfg["expected_training_rows"]:
        raise ValueError("Training data coverage changed")
    model, tok = F.load_model(cfg)
    model = F.configure_arm(model, cfg, args.arm)
    teacher = F.load_teacher(model, cfg, args.arm)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    if args.source_checkpoint:
        saved = torch.load(
            args.source_checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        validate_checkpoint(saved, cfg, args.arm, data_hash)
        F.shared.restore(model, saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        F.shared.restore_rng(saved["rng"])
        verify_restoration(model, optimizer, saved)
    else:
        if (out / "resume.pt").exists() or (out / "progress.json").exists():
            raise ValueError("An existing trained arm cannot start without its checkpoint")
        saved = {
            "weights": F.shared.state(model),
            "optimizer": optimizer.state_dict(),
            "rng": F.shared.rng_state(),
            "step": 0,
            "exposure": 0,
            "answer_tokens": 0,
            "forward_tokens": 0,
            "config": cfg,
            "data_sha256": data_hash,
            "arm": args.arm,
        }
    source_step, exposure = saved["step"], saved["exposure"]
    F.write(
        work / "restore-before-probe.json",
        {
            "passed": True,
            "step": source_step,
            "exposure": exposure,
            "weights_and_optimizer_exact": True,
            "source": str(args.source_checkpoint),
            "data_sha256": data_hash,
        },
    )
    model.train()
    selected, measurements = benchmark(
        model,
        tok,
        optimizer,
        parameters,
        teacher,
        rows,
        cfg,
        args.arm,
        exposure,
        work / "benchmark.json",
        args.memory_cap_mib,
    )
    # Engineering updates are discarded; restore weights, Adam moments and RNG again.
    optimizer.zero_grad(set_to_none=True)
    if args.source_checkpoint:
        # Adam's CPU scalar step may alias load_state_dict's source tensors.
        # Reopen the immutable archive so probe updates cannot alter that counter.
        del saved
        saved = torch.load(
            args.source_checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        validate_checkpoint(saved, cfg, args.arm, data_hash)
    F.shared.restore(model, saved["weights"])
    optimizer.load_state_dict(saved["optimizer"])
    verify_restoration(model, optimizer, saved)
    F.shared.restore_rng(saved["rng"])
    step, answer_tokens, forward_tokens = (
        source_step,
        saved["answer_tokens"],
        saved["forward_tokens"],
    )
    log_offset = max(0, args.previous_observed_step - source_step) + 100
    chosen = next(x for x in measurements if x["batch_size"] == selected)
    amendment = {
        "batch_size": selected,
        "original_batch_size": cfg["batch_size"],
        "source_step": source_step,
        "source_exposure": exposure,
        "previous_observed_step": args.previous_observed_step,
        "observed_recomputed_updates": max(0, args.previous_observed_step - source_step),
        "additional_unreported_updates_bound": cfg["log_steps"] - 1,
        "target_exposure": len(rows) * cfg["epochs"],
        "data_sha256": data_hash,
        "weights_and_optimizer_restored_exactly": True,
        "rng_restored": True,
        "learning_rate_basis": "original schedule indexed by cumulative sample exposure",
        "learning_rate_multiplier": 1,
        "samples_per_second": chosen["samples_per_second"],
        "pressure_peak_reserved_mib": chosen["peak_reserved_mib"],
        "memory_cap_mib": args.memory_cap_mib,
        "log_step_offset": log_offset,
        "checkpoint_sample_interval": 60000,
        "checkpoint_max_seconds": 1800,
        "engineering_updates_committed": 0,
    }
    F.write(work / "selection.json", amendment)
    torch.cuda.set_per_process_memory_fraction(
        (chosen["peak_reserved_mib"] + 6144) * 2**20 / total_memory
    )
    source_rng = saved["rng"]
    del saved
    gc.collect()
    torch.cuda.empty_cache()
    run = F.wandb_run(cfg, args.batch, "train", args.arm)
    run.config.update({"batch_continuation_20261022": amendment}, allow_val_change=False)
    F.write(
        status,
        {
            "state": "running",
            "stage": "train",
            "name": args.arm,
            "batch_size": selected,
            "source_exposure": exposure,
        },
    )
    pause = [False]
    signal.signal(signal.SIGTERM, lambda *_: pause.__setitem__(0, True))
    transport = CheckpointTransport(F.shared.atomic_torch_save)
    last_save, last_save_time, last_log = exposure, time.monotonic(), exposure
    began = time.monotonic()
    record = {}
    target = len(rows) * cfg["epochs"]
    F.shared.restore_rng(source_rng)
    try:
        while exposure < target:
            ids = next_indices(len(rows), cfg["epochs"], cfg["seed"], exposure, selected)
            encoded = [
                F.encode_sample(tok, rows[i], args.arm, cfg["max_sequence_tokens"]) for i in ids
            ]
            batch = F.shared.collate(encoded, tok.pad_token_id)
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rate"] * lr_scale(exposure, cfg)
            loss, norm = update(model, optimizer, parameters, batch, args.arm, cfg, teacher)
            step += 1
            exposure += len(ids)
            answer_tokens += int(batch["student"]["answer_mask"].sum())
            forward_tokens += int(batch["student"]["attention_mask"].sum())
            if args.arm.startswith("pd_"):
                forward_tokens += int(batch["teacher"]["attention_mask"].sum())
            record = {
                "step": step,
                "optimizer_updates": step,
                "exposure": exposure,
                "answer_tokens": answer_tokens,
                "forward_tokens": forward_tokens,
                "batch_size": selected,
                "loss": loss,
                "grad_norm": norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds_this_process": time.monotonic() - began,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "continuation": "batchsize-20261022",
            }
            if (
                exposure - last_log >= 2000
                or exposure == target
                or step == source_step + 1
                or pause[0]
            ):
                F.write(out / "progress.json", record)
                F.write(work / "progress.json", record)
                with (work / "curve.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                run.log(record, step=step + log_offset)
                last_log = exposure
            if (
                exposure - last_save >= 60000
                or time.monotonic() - last_save_time >= 1800
                or exposure == target
                or pause[0]
            ):
                transport.save(
                    {
                        "weights": F.shared.state(model),
                        "optimizer": optimizer.state_dict(),
                        "rng": F.shared.rng_state(),
                        "step": step,
                        "exposure": exposure,
                        "answer_tokens": answer_tokens,
                        "forward_tokens": forward_tokens,
                        "config": cfg,
                        "data_sha256": data_hash,
                        "arm": args.arm,
                        "continuation": amendment,
                    },
                    work / "resume.pt",
                )
                last_save, last_save_time = exposure, time.monotonic()
            if pause[0]:
                transport.close()
                F.write(status, {"state": "paused_at_saved_checkpoint", **record})
                return
        transport.close()
        F.shared.atomic_torch_save(F.shared.state(model), out / "weights.pt")
        model.eval()
        samples = [
            F.encode_sample(tok, row, args.arm, cfg["max_sequence_tokens"]) for row in rows[:2]
        ]
        inputs = F.shared.collate(samples, tok.pad_token_id)["student"]
        with torch.no_grad(), F.autocast():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            ).logits
        F.shared.atomic_torch_save(
            {
                "input_ids": inputs["input_ids"].cpu(),
                "attention_mask": inputs["attention_mask"].cpu(),
                "greedy": logits.argmax(-1).cpu(),
                "logits": logits[:, -1, :1024].float().cpu(),
            },
            out / "reload-reference.pt",
        )
        F.write(
            out / "complete.json",
            {
                **record,
                "weights_sha256": F.shared.digest(out / "weights.pt"),
                "smoke": False,
                "continuation_amendment": amendment,
            },
        )
        F.write(
            status, {"state": "complete", "stage": "train", "name": args.arm, "exposure": exposure}
        )
    finally:
        transport.close()
        run.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
