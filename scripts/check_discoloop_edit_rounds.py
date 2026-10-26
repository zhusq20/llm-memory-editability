"""Read-only round/layer readouts of an audited DiscoLoop and saved fast states.

Imports must come from the parent's frozen source. No writer or optimizer is
called. Intermediate logits are tied-decoder readouts, not proof of retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.ttcd_editing import load_fast, tensor_digest
from llm_memory_editability.ttcd_verified_base import load_reader


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class RoundTrace:
    """Observe existing computation; hooks return None and never replace tensors."""

    def __init__(self, reader):
        self.reader = reader
        self.inputs = {}
        self.norm_inputs = []
        self.norm_outputs = []
        self.handles = []

    def __enter__(self):
        def norm_hook(module, args, output):
            self.norm_inputs.append(args[0].detach().clone())
            self.norm_outputs.append(output.detach().clone())

        for index, block in enumerate(self.reader.backbone.transformer.h):
            self.inputs[index] = []

            def observe(module, args, index=index):
                self.inputs[index].append(args[0].detach().clone())

            self.handles.append(block.ln_1.register_forward_pre_hook(observe))
        self.handles.append(self.reader.backbone.transformer.ln_f.register_forward_hook(norm_hook))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def states(self):
        if self.reader.backbone.repeats != 2 or len(self.norm_inputs) != 2:
            raise AssertionError("Expected exactly two rounds and one DiscoLoop boundary")
        if any(len(values) != 2 for values in self.inputs.values()):
            raise AssertionError("Every physical layer must execute exactly twice")
        layers = len(self.inputs)
        states = {"round1_end": self.norm_inputs[0], "round2_input": self.inputs[0][1]}
        for layer in range(layers):
            states[f"round2_layer{layer + 1}"] = (
                self.inputs[layer + 1][1] if layer + 1 < layers else self.norm_inputs[1]
            )
        return states


def scores(logits, expected, old=None, new=None):
    probability = logits.float().softmax(-1)
    top = probability.topk(5, dim=-1)
    result = {
        "argmax": logits.argmax(-1).cpu().tolist(),
        "expected": expected.cpu().tolist(),
        "expected_correct": (logits.argmax(-1) == expected).cpu().tolist(),
        "expected_probability": probability.gather(-1, expected[:, None])[:, 0].cpu().tolist(),
        "expected_rank": (logits > logits.gather(-1, expected[:, None]))
        .sum(-1)
        .add(1)
        .cpu()
        .tolist(),
        "top5_tokens": top.indices.cpu().tolist(),
        "top5_probabilities": top.values.cpu().tolist(),
    }
    if old is not None and new is not None:
        result.update(
            old_correct=(logits.argmax(-1) == old).cpu().tolist(),
            new_correct=(logits.argmax(-1) == new).cpu().tolist(),
            new_minus_old=(logits.gather(-1, new[:, None]) - logits.gather(-1, old[:, None]))[:, 0]
            .cpu()
            .tolist(),
        )
    return result


@torch.no_grad()
def inspect_rows(reader, fast, rows, metadata, expected, *, kind, bridges=None):
    """Use the parent's full query layout and greedy answer/EOS procedure."""
    device = next(reader.parameters()).device
    arrays = encode_rows(np.asarray(rows, dtype=np.int64), metadata)
    tokens, positions, labels = (torch.as_tensor(v, device=device) for v in arrays)
    old = labels[:, 0]
    new = torch.as_tensor(expected, device=device) + metadata["entity_offset"]
    before = tensor_digest(fast)
    ordinary = reader(tokens, positions, fast=fast)
    with RoundTrace(reader) as trace:
        observed = reader(tokens, positions, fast=fast)
    if not torch.equal(ordinary, observed):
        raise AssertionError("Observation changed final logits")
    states = trace.states()
    index = torch.arange(len(tokens), device=device)
    answer = observed[:, 0].argmax(-1)
    tokens[index, positions[:, 1]] = answer
    stop = reader(tokens, positions, fast=fast)[:, 1].argmax(-1)
    result = {
        "kind": kind,
        "rows": np.asarray(rows).tolist(),
        "answer_position": positions[:, 0].cpu().tolist(),
        "answer": answer.cpu().tolist(),
        "stop": stop.cpu().tolist(),
        "expected_new": new.cpu().tolist(),
        "old": old.cpu().tolist(),
        "final_new_joint": ((answer == new) & (stop == metadata["end_marker"])).cpu().tolist(),
        "final_old_joint": ((answer == old) & (stop == metadata["end_marker"])).cpu().tolist(),
        "hook_logits_exact": True,
        "stages": {},
    }
    model = reader.backbone
    for stage, state in states.items():
        logits = F.linear(model.transformer.ln_f(state), model.transformer.wte.weight)
        stage_result = {"answer": scores(logits[index, positions[:, 0]], new, old, new)}
        if bridges is not None:
            bridge = torch.as_tensor(bridges, device=device) + metadata["entity_offset"]
            stage_result["bridge"] = scores(logits[:, 1], bridge)
        result["stages"][stage] = stage_result
    if bridges is not None:
        # This is the boundary's actual probability-weighted embedding calculation.
        stage1_logits = F.linear(trace.norm_outputs[0], model.transformer.wte.weight)
        probability = (stage1_logits.float() / model.discoloop_temperature).softmax(-1)
        soft = F.linear(probability, model.transformer.wte.weight.T)[:, 1]
        expected_embedding = model.transformer.wte.weight[
            torch.as_tensor(bridges, device=device) + metadata["entity_offset"]
        ]

        def rms(x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + model.discoloop_eps)

        channel = rms(soft)
        actual_input = states["round2_input"][:, 1]
        reconstructed = states["round1_end"][:, 1] + model.discoloop_alpha * channel
        if not torch.equal(actual_input, reconstructed):
            raise AssertionError("Reconstructed bridge boundary differs from actual input")
        result["bridge_channel"] = {
            "soft_embedding_cosine_to_true": F.cosine_similarity(soft, expected_embedding)
            .cpu()
            .tolist(),
            "normalized_embedding_relative_error": (
                (channel - rms(expected_embedding)).norm(dim=-1)
                / rms(expected_embedding).norm(dim=-1)
            )
            .cpu()
            .tolist(),
            "residual_to_channel_norm_ratio": (
                states["round1_end"][:, 1].norm(dim=-1)
                / (model.discoloop_alpha * channel).norm(dim=-1)
            )
            .cpu()
            .tolist(),
            "boundary_reconstruction_exact": True,
        }
    if tensor_digest(fast) != before:
        raise AssertionError("Diagnostic mutated a fast state")
    return result


def compare_saved(observed, saved):
    for key in ("answer", "stop"):
        if observed[key] != saved[key]:
            raise AssertionError(f"Saved CUDA FP32 {key} differs from diagnostic")
    actual = observed["stages"]["round2_layer4"]["answer"]["new_minus_old"]
    delta = [abs(a - b) for a, b in zip(actual, saved["new_minus_old"], strict=True)]
    return {
        "answer_and_EOS_exact": True,
        "max_new_minus_old_score_difference": max(delta),
    }


def aggregate(records):
    summary = {}
    for condition in ("before", "after"):
        summary[condition] = {}
        for kind in ("atom", "first_hop_atom", "two_hop"):
            items = [r[condition][kind] for r in records]
            stages = {}
            for stage in items[0]["stages"]:
                answer = [v for item in items for v in item["stages"][stage]["answer"]["argmax"]]
                new = [v for item in items for v in item["expected_new"]]
                old = [v for item in items for v in item["old"]]
                stages[stage] = {
                    "n": len(answer),
                    "new": sum(a == b for a, b in zip(answer, new, strict=True)),
                    "old": sum(a == b for a, b in zip(answer, old, strict=True)),
                    "other": sum(a not in (b, c) for a, b, c in zip(answer, new, old, strict=True)),
                }
                if kind == "two_hop":
                    bridge = [
                        v
                        for item in items
                        for v in item["stages"][stage]["bridge"]["expected_correct"]
                    ]
                    stages[stage]["bridge_correct"] = sum(bridge)
                    prob = [
                        v
                        for item in items
                        for v in item["stages"][stage]["bridge"]["expected_probability"]
                    ]
                    stages[stage]["bridge_probability_mean"] = float(np.mean(prob))
                    stages[stage]["bridge_probability_min"] = min(prob)
            summary[condition][kind] = stages
        channels = [r[condition]["two_hop"]["bridge_channel"] for r in records]
        summary[condition]["bridge_channel"] = {
            key: {
                "mean": float(np.mean(values)),
                "min": min(values),
                "max": max(values),
            }
            for key in channels[0]
            if key != "boundary_reconstruction_exact"
            for values in [[v for channel in channels for v in channel[key]]]
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text())
    receipt = json.loads((args.parent / "audit.json").read_text())
    cloud = json.loads((args.parent / "result-cloud-audit.json").read_text())
    if not receipt["passed"] or not cloud["passed"]:
        raise AssertionError("Parent independent and cloud audits must pass")
    reader, world = load_reader(config, "single", args.device)
    reader.requires_grad_(False)
    run = None
    if args.tracking != "disabled":
        import wandb

        run = wandb.init(
            entity="zhusq20",
            project="llm-memory-editability",
            group="discoloop-edit-round-diagnostic-v1",
            name="audited-single/read-only-rounds",
            job_type="diagnostic",
            dir=str(args.out),
            mode=args.tracking,
            config={
                "parent_batch": config["batch"],
                "parent_arm": "single",
                "parent_checkpoint_sha256": sha256(config["models"]["single"]["checkpoint"]),
                "base_parameter_updates": 0,
                "fast_state_updates": 0,
                "device": args.device,
                "precision": "fp32",
            },
            settings=wandb.Settings(disable_git=True),
        )
        save_json(
            args.out / "wandb.json",
            {"id": run.id, "url": run.url, "mode": args.tracking},
        )
    model_before = tensor_digest(reader.backbone.state_dict())
    files = {}

    def bind(path):
        files[str(path)] = sha256(path)

    for path in [args.config, args.parent / "audit.json", args.parent / "panel.json"]:
        bind(path)
    bind(Path(config["models"]["single"]["checkpoint"]))
    baseline_path = args.parent / "states/baseline/fast.pt"
    bind(baseline_path)
    baseline, _ = load_fast(baseline_path, reader)
    baseline_record_path = baseline_path.with_name("record.json")
    bind(baseline_record_path)
    base_record = json.loads(baseline_record_path.read_text())["readout"]["predictions"]
    panel = json.loads((args.parent / "panel.json").read_text())
    records = []
    parity = []
    for index, fact in enumerate(panel["facts"]):
        directory = args.parent / f"states/single/f{index:03d}"
        for filename in ("fast.pt", "record.json"):
            bind(directory / filename)
        fast, spec = load_fast(directory / "fast.pt", reader)
        saved = json.loads((directory / "record.json").read_text())
        if spec != saved["spec"] or spec["indices"] != [index]:
            raise AssertionError("Saved state/record/panel identity differs")
        atom = world.atoms[[fact["atom"]]]
        chains = world.chains[fact["D"]]
        first_atoms = world.atoms[world.first[fact["D"]]]
        bridges = first_atoms[:, -1]
        if not np.all(bridges == atom[0, 0]):
            raise AssertionError("The edited key must be precisely the second-hop fact")
        record = {"fact_index": index, "fact": fact, "before": {}, "after": {}}
        for condition, state in (("before", baseline), ("after", fast)):
            for kind, rows, expected, bridge_values in (
                ("atom", atom, [fact["new_tail"]], None),
                ("first_hop_atom", first_atoms, first_atoms[:, -1], None),
                ("two_hop", chains, [fact["new_tail"]] * len(chains), bridges),
            ):
                observed = inspect_rows(
                    reader,
                    state,
                    rows,
                    world.metadata,
                    expected,
                    kind=kind,
                    bridges=bridge_values,
                )
                record[condition][kind] = observed
                source = saved["readout"]["predictions"] if condition == "after" else base_record
                pool = {
                    "atom": "E",
                    "first_hop_atom": "prerequisite_atoms",
                    "two_hop": "D",
                }[kind]
                lookup = {tuple(row): i for i, row in enumerate(source[pool]["rows"])}
                indices = [lookup[tuple(row)] for row in np.asarray(rows).tolist()]
                subset = {key: [values[i] for i in indices] for key, values in source[pool].items()}
                parity.append(compare_saved(observed, subset))
        records.append(record)
        if run is not None:
            run.log({"completed_facts": index + 1}, step=index + 1)
        save_json(args.out / f"fact-{index:03d}.json", record)
        save_json(args.out / "progress.json", {"completed_facts": index + 1, "target": 32})
        print(
            json.dumps({"completed_facts": index + 1, "seconds": time.monotonic() - started}),
            flush=True,
        )
    if tensor_digest(reader.backbone.state_dict()) != model_before:
        raise AssertionError("Backbone changed")
    for path, digest in files.items():
        if sha256(path) != digest:
            raise AssertionError(f"Input file changed: {path}")
    summary = {
        "schema": "discoloop-edit-round-diagnostic-v1",
        "post_hoc": True,
        "parent_batch": config["batch"],
        "parent_arm": "single",
        "facts": len(records),
        "two_hop_queries": sum(len(r["fact"]["D"]) for r in records),
        "device": args.device,
        "torch": torch.__version__,
        "precision": "fp32",
        "base_parameter_updates": 0,
        "fast_state_updates": 0,
        "backbone_unchanged": True,
        "all_input_files_unchanged": True,
        "all_hook_logits_exact": True,
        "all_saved_answer_and_EOS_exact": True,
        "max_saved_score_difference": max(v["max_new_minus_old_score_difference"] for v in parity),
        "results": aggregate(records),
        "input_sha256": files,
        "wall_seconds": time.monotonic() - started,
        "interpretation_limit": (
            "Tied-decoder intermediate readouts describe states; "
            "no causal intervention was performed. "
            "Repeated queries belong to one graph/initialization."
        ),
    }
    save_json(args.out / "summary.json", summary)
    if run is not None:
        run.summary.update(
            {
                "facts": summary["facts"],
                "two_hop_queries": summary["two_hop_queries"],
                "base_parameter_updates": 0,
                "fast_state_updates": 0,
                "all_saved_answer_and_EOS_exact": True,
                "results": summary["results"],
            }
        )
        run.save(str(args.out / "summary.json"), base_path=str(args.out), policy="now")
        run.finish()
    print(
        json.dumps({"complete": True, "summary": str(args.out / "summary.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
