#!/usr/bin/env python3
"""Evaluate existing final checkpoints, then compare paired round predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.experiment_tracking import read_json, write_json  # noqa: E402
from llm_memory_editability.round_evaluation import (  # noqa: E402
    correctness_summary,
    inference_matmul_flops,
    paired_summary,
    round_logits,
)


def evaluate(config, name, output):
    import torch
    import wandb
    from transformers import GPT2Config

    from llm_memory_editability.loop_hop_data import load, pack_rows
    from llm_memory_editability.rule_gate import RuleGateGPT
    from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.16)
    arm = config["arms"][name]
    source = Path(arm["source"])
    out = output / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise FileExistsError(out / "complete.json")
    complete = read_json(source / "complete.json")
    if complete["step"] != config["checkpoint_step"] or complete["state"] != "complete":
        raise ValueError("Only the fixed completed endpoint is admissible")
    ck = torch.load(
        source / f"checkpoint-{config['checkpoint_step']:07d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _, panels, meta = load(config["input_root"])
    if ck["step"] != config["checkpoint_step"] or ck["world_sha256"] != meta["world_sha256"]:
        raise ValueError("Wrong endpoint or world")
    if meta["panels_sha256"] != config["panels_sha256"] or set(panels) != set(config["panels"]):
        raise ValueError("Wrong evaluation pools")
    hf = GPT2Config(
        **config["model"],
        n_layer=1,
        vocab_size=meta["vocab_size"],
        bos_token_id=meta["pad_token"],
        eos_token_id=meta["end_marker"],
        pad_token_id=meta["pad_token"],
        use_cache=False,
    )
    rule = arm.get("rule")
    model = (
        (
            RuleGateGPT(hf, 4, rule=rule, tau=0.1)
            if rule
            else SupervisedAlignmentGPT(hf, 4, discoloop=True)
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(ck["model"], strict=True)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.requires_grad_(False)
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    tracking = config["tracking"]
    wb = wandb.init(
        entity=tracking["entity"],
        project=tracking["project"],
        group=config["batch"],
        name=name,
        id=arm["wandb_id"],
        resume="never",
        job_type="evaluation",
        dir=str(out),
        config={"evaluation": config, "arm": name, "training_updates": 0},
        settings=wandb.Settings(disable_git=True, x_stats_gpu_device_ids=[0]),
    )
    write_json(out / "tracking.json", {"id": wb.id, "url": wb.url})
    result = {
        "arm": name,
        "step": ck["step"],
        "panels": {},
        "checks": {},
        "source": str(source),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "training_updates": 0,
        "model_sha256_recorded_by_training": ck["model_sha256"],
        "inference_rounds": config["rounds"],
    }
    started = time.perf_counter()
    max_rounds = max(config["rounds"])
    try:
        with torch.inference_mode():
            for panel in config["panels"]:
                rows = panels[panel]
                tokens, positions, labels = pack_rows(rows, meta)
                answer = np.empty((len(rows), max_rounds), np.int32)
                stop = np.empty_like(answer)
                for start in range(0, len(rows), config["batch_size"]):
                    end = min(start + config["batch_size"], len(rows))
                    x = torch.as_tensor(tokens[start:end], device="cuda")
                    pos = torch.as_tensor(positions[start:end], device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for r, logits in enumerate(round_logits(model, x, pos, max_rounds), 1):
                            pred = logits[:, 0].argmax(-1)
                            # Exact native-prefix check includes all 16 rounds, both readout slots.
                            if start == 0 and panel == config["panels"][0]:
                                model.repeats = r
                                if not torch.equal(logits, model(x, pos)):
                                    raise AssertionError(f"Native logits mismatch at R{r}")
                            answer[start:end, r - 1] = pred.cpu().numpy()
                            # Preserve the original generated-answer EOS scoring.
                            generated = x.clone()
                            generated[torch.arange(len(x), device="cuda"), pos[:, 1]] = pred
                            model.repeats = r
                            stop[start:end, r - 1] = (
                                model(generated, pos)[:, 1].argmax(-1).cpu().numpy()
                            )
                for r in (1, 2, 3, 4):
                    with np.load(source / "grid" / f"r{r}" / f"{panel}.npz") as old:
                        if not all(
                            (
                                np.array_equal(old["rows"], rows),
                                np.array_equal(old["answer"], answer[:, r - 1]),
                                np.array_equal(old["stop"], stop[:, r - 1]),
                            )
                        ):
                            raise AssertionError(f"Existing grid disagreement: {panel}/R{r}")
                correct = answer == labels[:, :1]
                exact = correct & (stop == meta["end_marker"])
                result["panels"][panel] = {
                    "answer": correctness_summary(correct),
                    "answer_eos": correctness_summary(exact),
                }
                np.savez_compressed(
                    out / f"{panel}.npz",
                    rows=rows,
                    labels=labels,
                    answer=answer,
                    stop=stop,
                    correct=correct,
                    exact=exact,
                )
                write_json(
                    out / "progress.json",
                    {
                        "state": "evaluating",
                        "panel": panel,
                        "panels_done": len(result["panels"]),
                        "panels_total": len(panels),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                )
                print(
                    json.dumps(
                        {"arm": name, "panel": panel, "answer_accuracy": correct.mean(0).tolist()}
                    ),
                    flush=True,
                )
            # Use the same fixed batch and complete answer+EOS path for wall-time comparisons.
            tokens, positions, _ = pack_rows(
                panels[config["benchmark_panel"]][: config["batch_size"]], meta
            )
            x, pos = (
                torch.as_tensor(tokens, device="cuda"),
                torch.as_tensor(positions, device="cuda"),
            )
            result["benchmark"] = {
                "panel": config["benchmark_panel"],
                "batch_size": len(x),
                "sequence_length": x.shape[1],
                "rounds": {},
            }
            for r in config["rounds"]:
                model.repeats = r
                timings = []
                for rep in range(config["benchmark_warmup"] + config["benchmark_repeats"]):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = model(x, pos)[:, 0].argmax(-1)
                        generated = x.clone()
                        generated[torch.arange(len(x), device="cuda"), pos[:, 1]] = pred
                        model(generated, pos)[:, 1].argmax(-1)
                    torch.cuda.synchronize()
                    if rep >= config["benchmark_warmup"]:
                        timings.append(time.perf_counter() - begin)
                result["benchmark"]["rounds"][str(r)] = {
                    "batch_seconds": timings,
                    "median_ms_per_item": float(np.median(timings) * 1000 / len(x)),
                    "estimated_matmul_flops_per_item": inference_matmul_flops(
                        hf.n_embd, hf.vocab_size, x.shape[1], r, rule
                    ),
                }
            if any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()):
                raise AssertionError("Evaluation changed parameters")
        result["checks"] = {
            "native_logits_r1_16": True,
            "all_saved_r1_4_answer_and_eos": True,
            "weights_unchanged": True,
            "items_per_round": sum(len(v) for v in panels.values()),
        }
        result["elapsed_seconds"] = time.perf_counter() - started
        for r in config["rounds"]:
            record = {
                "inference/round": r,
                **{
                    f"eval/{p}/{s}": result["panels"][p][s]["accuracy"][r - 1]
                    for p in panels
                    for s in ["answer", "answer_eos"]
                },
                **{
                    f"cost/{k}": v
                    for k, v in result["benchmark"]["rounds"][str(r)].items()
                    if k != "batch_seconds"
                },
                **{
                    f"transitions/{p}/{key}": result["panels"][p]["answer"][key][r - 1]
                    for p in panels
                    for key in ("gained_vs_reference_n", "lost_vs_reference_n")
                },
            }
            wb.log(record, step=r)
        wb.summary["primary_r8_graph_b_three_hop"] = result["panels"]["ood_3"]["answer"][
            "accuracy"
        ][7]
        wb.summary["checks_passed"] = True
        write_json(out / "summary.json", result)
        wb.finish()
        write_json(
            out / "complete.json",
            {
                "state": "complete",
                "training_updates": 0,
                "weights_written": 0,
                "checks_passed": True,
            },
        )
    except BaseException:
        wb.finish(exit_code=1)
        raise


def report(config, output):
    runs = {name: read_json(output / "runs" / name / "summary.json") for name in config["arms"]}
    for name in runs:
        if read_json(output / "runs" / name / "complete.json")["state"] != "complete":
            raise ValueError("Incomplete evaluation")
    compared = {}
    for name in ("stability-u", "readiness-u"):
        compared[name] = {}
        for panel in config["panels"]:
            with (
                np.load(output / "runs" / "disco" / f"{panel}.npz") as base,
                np.load(output / "runs" / name / f"{panel}.npz") as new,
            ):
                if not np.array_equal(base["rows"], new["rows"]):
                    raise ValueError("Paired item order differs")
                compared[name][panel] = {
                    key: paired_summary(new[key], base[key]) for key in ("correct", "exact")
                }
    curves = {}
    for name, arm in config["arms"].items():
        source = Path(arm["source"])
        history = read_json(source / "learning.json")
        cost = read_json(source / "run.json")
        curves[name] = [
            {
                "step": r["step"],
                "metrics": r["metrics"],
                "examples": r.get("examples"),
                "executed_input_tokens": r.get("executed_input_tokens"),
                "training_seconds": r.get("training_seconds"),
                "estimated_total_matmul_flops": r["estimated_matmul_training_flops"]
                + r["step"] * cost.get("rule_forward_only_matmul_flops_per_step", 0),
            }
            for r in history
        ]
    write_json(
        output / "comparison.json",
        {
            "runs": runs,
            "paired_vs_disco": compared,
            "training_curves": curves,
            "scope": (
                "One observed world and initialization; exploratory paired evaluation; "
                "fixed R8 primary"
            ),
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, panel in zip(axes.flat, ["ood_2", "ood_3", "id_4", "ood_4"], strict=True):
        for name, run in runs.items():
            ax.plot(
                config["rounds"],
                100 * np.array(run["panels"][panel]["answer"]["accuracy"]),
                marker=".",
                label=name,
            )
        ax.axvline(4, color="gray", ls=":")
        ax.axvline(8, color="gray", ls="--")
        ax.set(title=panel, xlabel="Inference rounds", ylabel="Answer accuracy (%)", ylim=(-2, 102))
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.savefig(output / "round-accuracy.png", dpi=180)
    fig.savefig(output / "round-accuracy.pdf")
    plt.close(fig)
    write_json(
        output / "completion.json",
        {
            "state": "complete",
            "arms": list(runs),
            "training_updates": 0,
            "rounds": config["rounds"],
            "primary_round": 8,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evaluate", "report"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm")
    args = parser.parse_args()
    cfg = read_json(args.config)
    if args.mode == "evaluate":
        evaluate(cfg, args.arm, args.output)
    else:
        report(cfg, args.output)


if __name__ == "__main__":
    main()
