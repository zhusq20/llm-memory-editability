"""Post-hoc, parameter-only inspection of already audited CE stream states.

No model is constructed, no forward/backward runs, and no optimizer is created.
The original experiment, result JSON and saved tensors are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch

STEPS = (1, 2, 3, 4, 8, 16, 32, 64)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@torch.no_grad()
def parameter_metrics(base, previous, current, radius):
    if set(base) != set(previous) or set(base) != set(current):
        raise ValueError("Parameter names differ")
    total = saturated = previous_boundary = boundary_flip = 0
    previous_sq = current_sq = update_sq = cross = update_previous = 0.0
    maximum_update = 0.0
    for name, anchor in base.items():
        old, new = previous[name], current[name]
        if old.shape != anchor.shape or new.shape != anchor.shape:
            raise ValueError("Parameter shapes differ")
        if anchor.dtype != torch.float32 or old.dtype != anchor.dtype or new.dtype != anchor.dtype:
            raise ValueError("Original states must remain FP32")
        low, high = anchor - radius, anchor + radius
        if not ((new >= low) & (new <= high)).all():
            raise ValueError("An original-theta0 elementwise bound was violated")
        old_delta, new_delta, update = old - anchor, new - anchor, new - old
        previous_sq += float(old_delta.square().sum(dtype=torch.float64))
        current_sq += float(new_delta.square().sum(dtype=torch.float64))
        update_sq += float(update.square().sum(dtype=torch.float64))
        cross += float((old_delta * new_delta).sum(dtype=torch.float64))
        update_previous += float((update * old_delta).sum(dtype=torch.float64))
        maximum_update = max(maximum_update, float(update.abs().max()))
        saturated += int(((new == low) | (new == high)).sum())
        previous_boundary += int(((old == low) | (old == high)).sum())
        boundary_flip += int(
            (((old == low) & (new == high)) | ((old == high) & (new == low))).sum()
        )
        total += anchor.numel()
    return {
        "elements": total,
        "offset_rms": math.sqrt(current_sq / total),
        "fact_update_rms": math.sqrt(update_sq / total),
        "fact_update_l2": math.sqrt(update_sq),
        "maximum_fact_update": maximum_update,
        "saturated_elements": saturated,
        "saturated_fraction": saturated / total,
        "previous_offset_cosine": cross / math.sqrt(previous_sq * current_sq)
        if previous_sq * current_sq
        else None,
        "update_previous_offset_cosine": update_previous / math.sqrt(previous_sq * update_sq)
        if previous_sq * update_sq
        else None,
        "previous_offset_projection_retained": cross / previous_sq if previous_sq else None,
        "update_over_previous_offset_l2": math.sqrt(update_sq / previous_sq)
        if previous_sq
        else None,
        "previous_boundary_elements": previous_boundary,
        "opposite_boundary_elements": boundary_flip,
        "opposite_boundary_fraction_all": boundary_flip / total,
        "opposite_boundary_fraction_previous_boundary": boundary_flip / previous_boundary
        if previous_boundary
        else None,
    }


def behavioral_summary(records, panel):
    pairs = [p for p in panel["pairs"] if p["split"] == "evaluation"]
    facts = [p[role] for p in pairs for role in ("A", "B")]
    indices = {f["fact_id"]: i for i, f in enumerate(facts)}
    age_rows, trajectories, replacements, threshold_updates = [], [], [], []
    for age in (0, 1, 2, 3, 4, 8, 16, 32, 48, 63):
        answers, full = [], []
        for stream, rows in records.items():
            for initial in range(64 - age):
                index = indices[panel["streams"][stream][initial]]
                item = rows[initial + age]["readout"]["atoms"]
                answers.append(item["answer_correct"][index])
                full.append(item["final_correct"][index])
        age_rows.append(
            {"age": age, "n": len(full), "correct": sum(full), "answer_correct": sum(answers)}
        )
    for stream, rows in records.items():
        for i, row in enumerate(rows):
            atoms, info = row["readout"]["atoms"], row["write_info"]
            supplied = [indices[fid] for fid in row["written"]]
            current_target = atoms["expected"][supplied[-1]]
            prior = supplied[:-1]
            threshold_updates.append(
                next((h["step"] - 1 for h in info["history"] if h["new_ce_before"] < 1e-3), 20)
            )
            trajectories.append(
                {
                    "stream": stream,
                    "step": i + 1,
                    "written_correct": sum(atoms["final_correct"][j] for j in supplied),
                    "written_answer_correct": sum(atoms["answer_correct"][j] for j in supplied),
                    "n": len(supplied),
                    "saturated_fraction": info["constraint_saturated_elements"]
                    / info["trainable_parameters"],
                }
            )
            if i:
                before = rows[i - 1]["readout"]["atoms"]["final_correct"]
                lost = [j for j in prior if before[j] and not atoms["final_correct"][j]]
                replacements.append(
                    {
                        "stream": stream,
                        "step": i + 1,
                        "previously_correct_lost": len(lost),
                        "lost_now_current_tail": sum(
                            atoms["answer"][j] == current_target for j in lost
                        ),
                        "prior_answer_current_tail": sum(
                            atoms["answer"][j] == current_target for j in prior
                        ),
                    }
                )
    return {
        "age_retention": age_rows,
        "trajectories": trajectories,
        "answer_replacements": replacements,
        "updates_before_first_logged_CE_below_1e_3": {
            "values": threshold_updates,
            "quantiles_min_q25_median_q75_max": np.quantile(
                threshold_updates, [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            "censor_value_if_never_observed": 20,
        },
        "saturated_fraction_all_128_readouts_median": float(
            np.median([r["saturated_fraction"] for r in trajectories])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    torch.set_num_threads(4)
    plan = read(args.batch / "source-v2/configs/ce-cross-sample-double-new-v1.json")
    panel = read(args.batch / "source-v2" / plan["panel"])
    original = args.batch / "report/output/result.json"
    receipt = read(args.batch / "report/output/cloud-verification.json")
    if not receipt["passed"] or sha(original) != receipt["result_sha256"]:
        raise ValueError("Original full cloud result must verify")
    result = {
        "analysis": "post-hoc descriptive inspection; no causal ablations",
        "optimizer_updates": 0,
        "model_forward_calls": 0,
        "model_constructions": 0,
        "torch_version": torch.__version__,
        "source_script_sha256": sha(Path(__file__)),
        "original_result_sha256": sha(original),
        "parameter_inspection_steps_per_stream": STEPS,
        "architectures": {},
    }
    for architecture, arm in plan["architectures"].items():
        directory = args.batch / architecture / "output/evaluate"
        scientific = read(directory / "result.json")
        records = {}
        for stream in ("forward", "reverse"):
            records[stream] = []
            for step in range(1, 65):
                path = directory / "records" / f"{stream}-{step:03}.json"
                if sha(path) != scientific["record_hashes"][path.name]:
                    raise ValueError("Original readout changed")
                records[stream].append(read(path))
        if sha(arm["checkpoint"]) != arm["checkpoint_sha256"]:
            raise ValueError("Original foundation changed")
        saved = torch.load(arm["checkpoint"], map_location="cpu", weights_only=False)
        first = torch.load(
            directory / "states/forward-001.pt", map_location="cpu", weights_only=True
        )
        base = {name: saved["model"][name] for name in first["values"]}
        del first, saved
        if sum(t.numel() for t in base.values()) != arm["editable_parameters"]:
            raise ValueError("Editable parameter count changed")
        inspected, bound_hashes = [], {}
        for stream, rows in records.items():
            needed = sorted({j for step in STEPS for j in (step - 1, step) if j > 0})
            states = {}
            for step in needed:
                row = rows[step - 1]
                path = directory / row["state"]
                actual_sha = sha(path)
                if actual_sha != row["state_sha256"]:
                    raise ValueError("An audited saved state changed")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["foundation_digest"] != scientific["foundation_digest"]:
                    raise ValueError("Saved state belongs to another foundation")
                states[step] = payload["values"]
                bound_hashes[row["key"]] = actual_sha
            for step in STEPS:
                values = parameter_metrics(
                    base,
                    states[step - 1] if step > 1 else base,
                    states[step],
                    plan["adaptation"]["parameter_constraint"]["maximum_absolute_element_change"],
                )
                row = rows[step - 1]
                if (
                    values["saturated_elements"]
                    != row["write_info"]["constraint_saturated_elements"]
                ):
                    raise ValueError("Tensor inspection and original saturation log differ")
                inspected.append({"stream": stream, "step": step, **values})
                print(
                    json.dumps({"architecture": architecture, "stream": stream, "step": step}),
                    flush=True,
                )
            del states
        result["architectures"][architecture] = {
            "learning_rate": arm["learning_rate"],
            "editable_parameters": arm["editable_parameters"],
            "behavior": behavioral_summary(records, panel),
            "parameter_inspection": inspected,
            "verified_state_hashes": bound_hashes,
        }
        save(args.out / "progress.json", {"completed_architectures": list(result["architectures"])})
    from torch.optim.adam import _single_tensor_adam

    (args.out / "pytorch-adam-source.py").write_text(inspect.getsource(_single_tensor_adam))
    result["pytorch_adam_source_sha256"] = sha(args.out / "pytorch-adam-source.py")
    result["passed"] = True
    save(args.out / "result.json", result)
    print(
        json.dumps({"passed": True, "optimizer_updates": 0, "model_forward_calls": 0}), flush=True
    )


if __name__ == "__main__":
    main()
