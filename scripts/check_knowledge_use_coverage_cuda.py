"""Exercise real prepared inputs, discarded updates, and exact independent reload."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
import traceback
from pathlib import Path

import run_knowledge_use_coverage as R
import torch
from knowledge_checkpoint_io import CheckpointTransport, snapshot

from llm_memory_editability.knowledge_use_coverage_budget import pressure_example

K, F = R.K, R.F


def versions(model, frozen=False):
    return {n: p._version for n, p in model.named_parameters() if not frozen or not p.requires_grad}


def selected_logits(model, batch):
    model.eval()
    with torch.no_grad(), K.autocast():
        out = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
        ).logits
    return {"logits": out[batch["answer_mask"]][:, :1024].cpu(), "greedy": out.argmax(-1).cpu()}


def restore_optimizer(optimizer, saved):
    # load_state_dict installs state tensors; copy avoids mutating the comparison snapshot.
    optimizer.load_state_dict(copy.deepcopy(saved))


def exercise(config, batch, cohort, objective):
    began = time.perf_counter()
    out = batch / "engineering" / cohort / objective
    if (out / "started.json").exists():
        raise FileExistsError(f"Existing engineering attempt: {out}")
    K.write(out / "started.json", {"utc_unix": time.time(), "scientific_updates": 0})
    K.seed_all(config["seed"])
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("CUDA matrix backward failed")
    del x
    model, tok = K.load_model(config)
    model.config.pad_token_id = tok.pad_token_id
    model = F.configure_arm(model, config, f"{objective}_{config['scope']}")
    teacher = R.teacher_for_all(config)
    F._validate_independent_teacher(model, teacher)
    prepared = batch / "data" / cohort
    receipt = R.read(prepared / "prepare-complete.json")
    all_rows = []
    for name, digest in receipt["sha256"].items():
        if K.digest(prepared / name) != digest:
            raise ValueError(f"Data hash differs: {name}")
        if name.startswith("prepared-") and name != "prepared-replay.jsonl":
            all_rows.extend(R.read_rows(prepared / name))
    encoded = R.encoded_rows(tok, all_rows, objective, config)
    replay = R.encoded_rows(
        tok, R.read_rows(prepared / "prepared-replay.jsonl"), "pd", config, replay=True
    )
    longest, pressure_provenance = pressure_example(encoded)
    protected, replay_provenance = pressure_example(replay)
    # Simultaneously maximize input padding and the full-vocabulary target logits.
    pressure = [longest] * config["microbatch_size"]
    protected_pressure = [protected] * config["microbatch_size"]
    fixed = K.collate(encoded[:2], tok.pad_token_id)
    teacher_before = selected_logits(teacher, fixed["teacher"])
    teacher_versions, frozen_versions = versions(teacher), versions(model, True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    torch.cuda.reset_peak_memory_stats()

    def update():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, counts = R.loss_backward(model, teacher, pressure, objective, config)
        keep, keep_counts = R.loss_backward(
            model, teacher, protected_pressure, "pd", config, config["retention_kl_weight"]
        )
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise AssertionError(f"Missing trainable gradients: {missing}")
        norm = torch.nn.utils.clip_grad_norm_(
            params, config["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        if versions(teacher) != teacher_versions or versions(model, True) != frozen_versions:
            raise AssertionError("A frozen parameter changed")
        counts.update(keep_counts)
        return {
            "loss": loss,
            "retention_kl": keep,
            "grad_norm": float(norm),
            "counts": dict(counts),
        }

    first = update()
    saved_weights, saved_optimizer, saved_rng = (
        K.state(model),
        snapshot(optimizer.state_dict()),
        K.rng_state(),
    )
    second = update()
    expected = K.state(model)
    K.restore(model, saved_weights)
    restore_optimizer(optimizer, saved_optimizer)
    K.restore_rng(saved_rng)
    repeated = update()
    if second != repeated:
        raise AssertionError("Restored optimizer/RNG update metrics differ")
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.equal(p.detach().cpu(), expected[n]):
            raise AssertionError(f"Restored optimizer update differs: {n}")
    teacher_after = selected_logits(teacher, fixed["teacher"])
    if any(not torch.equal(teacher_before[k], teacher_after[k]) for k in teacher_before):
        raise AssertionError("Frozen teacher outputs changed")
    peak = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    if total - peak < 14 * 1024**3:
        raise MemoryError("Longest-input run leaves less than 6 GiB task plus 8 GiB card reserve")
    reference = selected_logits(model, fixed["student"])
    reference.update(
        {
            k: v.cpu()
            for k, v in fixed["student"].items()
            if k in {"input_ids", "attention_mask", "answer_mask"}
        }
    )
    transport = CheckpointTransport(K.atomic_torch_save)
    try:
        transport.save(
            {
                "weights": expected,
                "optimizer": snapshot(optimizer.state_dict()),
                "rng": K.rng_state(),
                "config": config,
            },
            out / "resume.pt",
        )
        K.atomic_torch_save(reference, out / "reference.pt")
    finally:
        transport.close()
    K.write(
        out / "exercise.json",
        {
            "passed": True,
            "scientific_updates": 0,
            "engineering_optimizer_updates": 3,
            "parameters": F.parameter_contract(model, f"{objective}_{config['scope']}"),
            "pressure": pressure_provenance,
            "replay_pressure": replay_provenance,
            "student_tokens": len(longest["student"]) + len(longest["answer"]) - 1,
            "teacher_tokens": len(longest["teacher"]) + len(longest["answer"]) - 1,
            "microbatch": len(pressure),
            "first": first,
            "second": second,
            "restoration_exact": True,
            "teacher_unchanged": True,
            "peak_reserved_bytes": peak,
            "total_memory_bytes": total,
            "seconds": time.perf_counter() - began,
            "weights_sha256": K.digest(out / "resume.pt"),
        },
    )


def reload(config, batch, cohort, objective):
    out = batch / "engineering" / cohort / objective
    saved = torch.load(out / "resume.pt", weights_only=False, map_location="cpu")
    if saved["config"] != config:
        raise AssertionError("Saved preflight configuration differs")
    model, _ = K.load_model(config)
    model = F.configure_arm(model, config, f"{objective}_{config['scope']}")
    K.restore(model, saved["weights"])
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.equal(p.detach().cpu(), saved["weights"][n]):
            raise AssertionError(f"Independent reload differs: {n}")
    reference = torch.load(out / "reference.pt", weights_only=False, map_location="cpu")
    inputs = {k: reference[k].cuda() for k in ("input_ids", "attention_mask", "answer_mask")}
    actual = selected_logits(model, inputs)
    passed = all(torch.equal(actual[k], reference[k]) for k in actual)
    if not passed:
        raise AssertionError("Independent logits or greedy reload differs")
    if K.digest(out / "resume.pt") != R.read(out / "exercise.json")["weights_sha256"]:
        raise AssertionError("Engineering checkpoint hash differs")
    K.write(
        out / "reload.json",
        {
            "passed": True,
            "full_trainable_weights_exact": True,
            "logits_and_greedy_exact": True,
            "new_process": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--cohort", choices=["development", "formal", "natural"], required=True)
    parser.add_argument("--stage", choices=["all", "exercise", "reload"], default="all")
    parser.add_argument("--objective", choices=["pd", "sft"])
    args = parser.parse_args()
    config = R.read(args.config)
    if args.cohort == "natural":
        from run_knowledge_use_coverage_natural import natural_config

        config = natural_config(config)
    destination = args.batch / "engineering" / args.cohort
    try:
        if args.stage == "all":
            if (destination / "preflight.json").exists():
                raise FileExistsError("Existing preflight receipt")
            for objective in ("pd", "sft"):
                for stage in ("exercise", "reload"):
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--config",
                            str(args.config),
                            "--batch",
                            str(args.batch),
                            "--cohort",
                            args.cohort,
                            "--stage",
                            stage,
                            "--objective",
                            objective,
                        ],
                        check=True,
                    )
            K.write(
                destination / "preflight.json",
                {
                    "passed": True,
                    "status": "passed",
                    "config": config,
                    "scientific_updates": 0,
                    "engineering_optimizer_updates": 6,
                    "utc_unix": time.time(),
                },
            )
        elif args.stage == "exercise":
            exercise(config, args.batch, args.cohort, args.objective)
        else:
            reload(config, args.batch, args.cohort, args.objective)
    except BaseException:
        K.write(
            destination / f"failure-{args.stage}-{args.objective}.json",
            {"traceback": traceback.format_exc(), "scientific_updates": 0},
        )
        raise


if __name__ == "__main__":
    main()
