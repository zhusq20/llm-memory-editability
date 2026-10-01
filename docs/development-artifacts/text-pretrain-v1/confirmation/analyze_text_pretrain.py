"""Pure-prefix interchange tests at fixed text-pretraining checkpoints."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_pretrain import (
    build_world,
    composite_sentence,
    construct,
    evaluate,
    sha,
)


def select_donors(world, seed):
    rng = np.random.default_rng(seed + 333)
    background_heads = set(world["background_train"][:, 0])
    first = [r for r in world["atomic_first"] if r[0] in background_heads]
    second = {(b, r): t for b, r, t in world["atomic_second"]}
    reserved = {
        tuple(r[[0, 4]]) for r in np.concatenate([world["target_test"], world["background_test"]])
    }
    result = {"same_bridge": [], "different_bridge": []}
    for i, row in enumerate(world["target_test"]):
        h, r1, b, r2, t = row
        for kind in result:
            candidates = []
            for hh, rr, bb in first:
                if rr != r1 or hh == h:
                    continue
                tt = second[bb, r2]
                if kind == "same_bridge" and bb == b:
                    candidates.append([hh, rr, bb, r2, tt])
                if kind == "different_bridge" and bb != b and tt != t and (hh, tt) in reserved:
                    candidates.append([hh, rr, bb, r2, tt])
            if candidates:
                result[kind].append((i, candidates[rng.integers(len(candidates))]))
    return result


@torch.inference_mode()
def first_logits(model, rows, device, patch=None):
    outputs = []
    for start in range(0, len(rows), 256):
        x = torch.tensor(
            [composite_sentence(r)[:-3] for r in rows[start : start + 256]], device=device
        )
        p = None if patch is None else dict(patch, value=patch["value"][start : start + 256])
        outputs.append(model(x, patch=p)[:, -1].cpu())
    return torch.cat(outputs).numpy()


@torch.inference_mode()
def analyze(run_path, output, device, node, position=3):
    run_path, output = Path(run_path), Path(output)
    if (output / "summary.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((run_path / "spec.json").read_text())
    world = build_world(spec)
    checkpoint = run_path / f"weights-{node:06d}.pt"
    model = construct(spec, device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    before = torch.nn.utils.parameters_to_vector(model.parameters()).clone()
    selected = select_donors(world, spec["world"])
    output_arrays, results = {}, []
    started = time.monotonic()
    for kind, pairs in selected.items():
        if not pairs:
            results.append(dict(kind=kind, n=0))
            continue
        indices = np.array([p[0] for p in pairs])
        donor = np.array([p[1] for p in pairs])
        rows = world["target_test"][indices].copy()
        # Recipient input remains original; only the scored counterfactual answer changes.
        original_targets = rows[:, -1].copy()
        rows[:, -1] = donor[:, -1]
        cache = {}
        pure_prefix = torch.tensor(
            [[2, int(r[0]), 3, int(r[1]), 3 if position == 4 else 0, 0, 0] for r in donor],
            device=device,
        )
        model(pure_prefix, cache=cache)
        self_cache = {}
        model(
            torch.tensor(
                [[2, int(r[0]), 3, int(r[1]), 3 if position == 4 else 0, 0, 0] for r in rows],
                device=device,
            ),
            cache=self_cache,
        )
        baseline, baseline_preds = evaluate(model, rows, device, composite=True)
        normal, normal_preds = evaluate(model, donor, device, composite=True)
        base_logits = first_logits(model, rows, device)
        output_arrays[kind + "_indices"] = indices
        output_arrays[kind + "_donor"] = donor
        output_arrays[kind + "_original_target"] = original_targets
        for name, predictions in (("baseline", baseline_preds), ("normal", normal_preds)):
            for key, val in predictions.items():
                output_arrays[f"{kind}_{name}_{key}"] = val
        records, identity_checks = [], []
        for layer in range(model.effective_depth):
            identity = dict(
                layer=layer,
                position=position,
                component="full",
                value=self_cache[layer]["full"][:, position],
            )
            _score, pred = evaluate(model, rows, device, composite=True, patch=identity)
            assert np.array_equal(pred["answer"], baseline_preds["answer"])
            assert np.array_equal(pred["stops"], baseline_preds["stops"])
            error = float(np.max(np.abs(pred["probability"] - baseline_preds["probability"])))
            assert error < 1e-5
            identity_checks.append(error)
            for component in ("attention", "mlp", "full"):
                patch = dict(
                    layer=layer,
                    position=position,
                    component=component,
                    value=cache[layer][component][:, position],
                )
                score, predictions = evaluate(model, rows, device, composite=True, patch=patch)
                logits = first_logits(model, rows, device, patch)
                n = np.arange(len(rows))
                other = original_targets.copy()
                same = other == rows[:, -1]
                competitor = base_logits.copy()
                competitor[n, rows[:, -1]] = -np.inf
                other[same] = competitor.argmax(-1)[same]
                margin = logits[n, rows[:, -1]] - logits[n, other]
                original_full = (
                    (predictions["answer"] == original_targets)
                    & (predictions["stops"][:, 0] == 5)
                    & (predictions["stops"][:, 1] == 1)
                )
                prefix = f"{kind}_L{layer}_{component}"
                for key, val in predictions.items():
                    output_arrays[prefix + "_" + key] = val
                output_arrays[prefix + "_margin"] = margin
                output_arrays[prefix + "_logits"] = logits
                records.append(
                    dict(
                        layer=layer,
                        component=component,
                        **score,
                        original_answer_retained=float(original_full.mean()),
                        target_competitor_margin=float(margin.mean()),
                    )
                )
        results.append(
            dict(
                kind=kind,
                n=len(rows),
                coverage=len(rows) / len(world["target_test"]),
                baseline=baseline,
                normal_counterfactual=normal,
                normal_counterfactual_heldout_fraction=float(
                    np.mean(
                        [
                            tuple(r[[0, 4]]) in {tuple(x[[0, 4]]) for x in world["background_test"]}
                            for r in donor
                        ]
                    )
                ),
                self_patch_probability_error=identity_checks,
                conditions=records,
            )
        )
    assert torch.equal(before, torch.nn.utils.parameters_to_vector(model.parameters()))
    np.savez_compressed(output / "predictions.npz", **output_arrays)
    write_json(
        output / "summary.json",
        dict(
            spec=spec,
            node=node,
            checkpoint_sha256=sha(checkpoint),
            script_sha256=sha(__file__),
            donor_selection="seeded background prefix per eligible query; no score selection",
            donor_valid_tokens=position + 1,
            patched_position=position,
            source_never_sees_r2_or_answer=True,
            seconds=time.monotonic() - started,
            results=results,
        ),
    )
    print(json.dumps(dict(run=run_path.name, node=node, results=results)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node", type=int, default=32000)
    parser.add_argument("--position", type=int, choices=[3, 4], default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.08, args.device)
    analyze(args.run, args.output, args.device, args.node, args.position)
