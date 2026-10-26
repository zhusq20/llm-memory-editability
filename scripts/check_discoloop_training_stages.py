"""Compare existing foundation/common checkpoints with the prior edited readouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.ttcd_editing import tensor_digest
from llm_memory_editability.ttcd_verified_base import foundation_inputs, load_reader


def stage_summary(records):
    result = {}
    for kind in ("atom", "two_hop"):
        result[kind] = {}
        for stage in records[0][kind]["stages"]:
            values = [r[kind]["stages"][stage] for r in records]
            answers = [v for item in values for v in item["answer"]["argmax"]]
            old = [v for r in records for v in r[kind]["old"]]
            new = [v for r in records for v in r[kind]["expected_new"]]
            counts = {
                "n": len(answers),
                "old": sum(a == b for a, b in zip(answers, old, strict=True)),
                "new": sum(a == b for a, b in zip(answers, new, strict=True)),
            }
            if kind == "two_hop":
                counts["bridge_correct"] = sum(
                    v for item in values for v in item["bridge"]["expected_correct"]
                )
            result[kind][stage] = counts
    return result


@torch.no_grad()
def native_parity(reader, fast, rows, metadata):
    tokens, positions, _ = (torch.as_tensor(v) for v in encode_rows(np.asarray(rows), metadata))
    actual = reader(tokens, positions, fast=fast)
    native = reader.backbone(tokens, positions)
    if not torch.equal(actual, native):
        raise AssertionError("A0 wrapper must preserve native backbone logits exactly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("audited_diagnostic_helper", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    config = json.loads(args.config.read_text())
    prior_summary = json.loads((args.prior / "summary.json").read_text())
    prior_cloud = json.loads((args.prior / "cloud-verification.json").read_text())
    if not prior_cloud["passed"] or not prior_summary["all_saved_answer_and_EOS_exact"]:
        raise AssertionError("The prior full diagnostic must have passed")
    prior_files = sorted(args.prior.glob("fact-*.json"))
    if len(prior_files) != 32:
        raise AssertionError("Preserve all32 original facts")
    prior = [json.loads(p.read_text()) for p in prior_files]
    bound = {str(p): helper.sha256(p) for p in prior_files}
    stages = {
        "writer_trained_A0": [
            {k: v for k, v in r["before"].items() if k != "first_hop_atom"} for r in prior
        ],
        "writer_trained_edited": [
            {k: v for k, v in r["after"].items() if k != "first_hop_atom"} for r in prior
        ],
    }
    import wandb

    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        group="discoloop-training-stage-diagnostic-v1",
        name="foundation-common-writer/edit-rounds",
        job_type="diagnostic",
        dir=str(args.out),
        config={"base_parameter_updates": 0, "fast_state_updates": 0, "facts": 32, "device": "cpu"},
        settings=wandb.Settings(disable_git=True),
    )
    helper.save_json(args.out / "wandb.json", {"id": run.id, "url": run.url})
    for stage in ("foundation_A0", "common_A0"):
        if stage == "foundation_A0":
            reader, world, _, provenance = foundation_inputs(config, "cpu")
            checkpoint = Path(config["foundation"]["checkpoint"])
        else:
            reader, world = load_reader(config, "common", "cpu")
            checkpoint = Path(config["common"]["checkpoint"])
            provenance = {"checkpoint_sha256": helper.sha256(checkpoint)}
        bound[str(checkpoint)] = provenance["checkpoint_sha256"]
        reader.requires_grad_(False)
        model_before = tensor_digest(reader.backbone.state_dict())
        fast = reader.empty_fast()
        fast_before = tensor_digest(fast)
        records = []
        for index, previous in enumerate(prior):
            fact = previous["fact"]
            atom = world.atoms[[fact["atom"]]]
            chains = world.chains[fact["D"]]
            bridges = world.atoms[world.first[fact["D"]], -1]
            record = {"fact": fact, "stage": stage}
            for kind, rows, expected, bridge in (
                ("atom", atom, [fact["new_tail"]], None),
                ("two_hop", chains, [fact["new_tail"]] * len(chains), bridges),
            ):
                native_parity(reader, fast, rows, world.metadata)
                record[kind] = helper.inspect_rows(
                    reader,
                    fast,
                    rows,
                    world.metadata,
                    expected,
                    kind=kind,
                    bridges=bridge,
                )
            helper.save_json(args.out / f"{stage}-fact-{index:03d}.json", record)
            records.append(record)
            step = index + 1 + (32 if stage == "common_A0" else 0)
            run.log({"completed_stage_facts": step}, step=step)
            print(json.dumps({"stage": stage, "completed_facts": index + 1}), flush=True)
        if (
            tensor_digest(reader.backbone.state_dict()) != model_before
            or tensor_digest(fast) != fast_before
        ):
            raise AssertionError("Model or A0 changed")
        stages[stage] = records
    for path, digest in bound.items():
        if helper.sha256(path) != digest:
            raise AssertionError(f"Input changed: {path}")
    summary = {
        "schema": "discoloop-training-stage-diagnostic-v1",
        "facts": 32,
        "two_hop_queries": 253,
        "scope": (
            "Same32 edited keys and253 D queries; foundation is the original-recipe300k rebuild, "
            "not the unavailable historical checkpoint. No full40000-atom claim."
        ),
        "base_parameter_updates": 0,
        "fast_state_updates": 0,
        "native_backbone_wrapper_logits_exact": True,
        "observer_logits_exact": True,
        "models_and_A0_unchanged": True,
        "input_hashes_unchanged": True,
        "results": {name: stage_summary(records) for name, records in stages.items()},
        "input_sha256": bound,
        "wall_seconds": time.monotonic() - started,
        "interpretation_limit": (
            "Intermediate tied-decoder observations, not causal isolation of the failure mechanism."
        ),
    }
    helper.save_json(args.out / "summary.json", summary)
    run.summary.update(
        {
            k: v
            for k, v in summary.items()
            if k not in ["input_sha256", "scope", "interpretation_limit"]
        }
    )
    run.save(str(args.out / "summary.json"), base_path=str(args.out), policy="now")
    run.finish()


if __name__ == "__main__":
    main()
