"""Offline donor-state interventions on frozen crossover models.

Truth is used to match donor/recipient pairs and score outputs, never supplied as
an answer or bridge to the model. This is a causal diagnostic, not an editor.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .bios_cross import CHAINS, make_cross_world
from .bios_data import EOS, array_hash, rng_for, write_json
from .bios_path_diagnostics import (
    INDISTINGUISHABLE,
    classify_answers,
    file_sha256,
    free_generate_values,
)

PAIR_TYPES = (
    "same_group_different_actual",
    "different_group_same_actual_different_default",
    "same_group_same_actual_different_person",
    "different_group_different_actual_different_default",
)
POSITIONS = (2, 3, 4)
ANSWER_ROLES = ("recipient_default", "recipient_actual", "donor_default", "donor_actual")


def make_donor_plan(world, chain, per_population=64):
    """Select using truth and train/heldout membership only, with fixed RNG.

    A missing donor stays -1. No prediction- or success-based replacement occurs.
    The same world yields identical pairs across sizes, conditions and model seeds.
    """
    heldout = world.person[world.heldout_ids[chain]]
    train = world.person[world.train_ids[chain]]
    rng = rng_for(world.seed, 971, chain)
    recipients = []
    for exceptional in (False, True):
        pool = heldout[world.exceptions[chain, heldout] == exceptional]
        if len(pool) < per_population:
            raise ValueError("Not enough heldout recipients for predeclared sample")
        recipients.extend(rng.choice(pool, per_population, replace=False))
    recipients = np.asarray(recipients, dtype=np.int64)
    group = world.memberships[chain]
    default = world.answers[world.root_ids[chain, group]]
    actual = world.answers[world.actual_ids[chain]]
    recipient_rows, donor_rows, type_rows, candidate_counts = [], [], [], []
    for recipient in recipients:
        same_group = group[train] == group[recipient]
        same_actual = actual[train] == actual[recipient]
        same_default = default[train] == default[recipient]
        masks = (
            same_group & ~same_actual,
            ~same_group & same_actual & ~same_default,
            same_group & same_actual & (train != recipient),
            ~same_group & ~same_actual & ~same_default,
        )
        for pair_type, mask in enumerate(masks):
            candidates = train[mask]
            recipient_rows.append(recipient)
            donor_rows.append(rng.choice(candidates) if len(candidates) else -1)
            type_rows.append(pair_type)
            candidate_counts.append(len(candidates))
    donor_rows = np.asarray(donor_rows, dtype=np.int64)
    complete = (donor_rows.reshape(-1, len(PAIR_TYPES)) >= 0).all(axis=1)
    return {
        "case_id": np.arange(len(donor_rows), dtype=np.int64),
        "recipient": np.asarray(recipient_rows, dtype=np.int64),
        "donor": donor_rows,
        "pair_type": np.asarray(type_rows, dtype=np.int8),
        "candidate_count": np.asarray(candidate_counts, dtype=np.int32),
        "common_recipient": np.repeat(complete, len(PAIR_TYPES)),
        "recipient_old_exception": world.exceptions[chain, recipient_rows],
    }


def intervention_state(recipient_state, donor_state, intervention, seed):
    """Random control matches the donor-minus-recipient perturbation norm per row."""
    if recipient_state.shape != donor_state.shape:
        raise ValueError("Activation shapes do not match")
    delta = donor_state.astype(np.float64) - recipient_state.astype(np.float64)
    norms = np.linalg.norm(delta, axis=-1)
    if intervention == "donor":
        patched = donor_state.copy()
    elif intervention == "sham":
        patched = recipient_state.copy()
    elif intervention == "matched_random":
        noise = np.random.default_rng(seed).standard_normal(delta.shape)
        noise /= np.maximum(np.linalg.norm(noise, axis=-1, keepdims=True), 1e-30)
        patched = (recipient_state + noise * norms[:, None]).astype(recipient_state.dtype)
    else:
        raise ValueError("Unknown intervention")
    actual_norms = np.linalg.norm(patched.astype(np.float64) - recipient_state, axis=-1)
    return patched, norms, actual_norms


def cache_clean_states(model, prompts, lengths, device, batch_size=256):
    """Capture post-block residuals at fixed positions for clean prompt passes."""
    import torch

    from .bios_train import precision

    states = [[] for _ in model.blocks]
    hooks = []
    for layer, block in enumerate(model.blocks):

        def capture(module, inputs, output, layer=layer):
            states[layer].append(output[:, POSITIONS, :].detach().float().cpu().numpy())

        hooks.append(block.register_forward_hook(capture))
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                p = torch.as_tensor(prompts[start : start + batch_size], device=device)
                n = torch.as_tensor(lengths[start : start + batch_size], device=device)
                with precision(device):
                    model(p, (n - 1)[:, None])
    finally:
        for hook in hooks:
            hook.remove()
    return np.stack([np.concatenate(layer_states) for layer_states in states])


def generate_with_state_patch(
    model, prompts, lengths, replacements, layer, position, device, batch_size=256
):
    """Apply the same pre-answer state patch in value and EOS generation passes.

    Positions <= 4 cannot depend on the generated token at position 5 under the
    causal mask, so the clean prompt cache contains no teacher-forced answer.
    """
    import torch

    from .bios_train import precision

    if position not in POSITIONS or np.any(lengths != 5):
        raise ValueError("This diagnostic is restricted to five-token derived queries")
    predictions, ended = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            p = torch.as_tensor(prompts[start : start + batch_size], device=device)
            n = torch.as_tensor(lengths[start : start + batch_size], device=device)
            replacement = torch.as_tensor(replacements[start : start + batch_size], device=device)

            def patch(module, inputs, output, replacement=replacement):
                changed = output.clone()
                changed[:, position] = replacement.to(output.dtype)
                return changed

            hook = model.blocks[layer].register_forward_hook(patch)
            try:
                with precision(device):
                    first = model(p, (n - 1)[:, None])[:, 0].float().argmax(-1)
                    continuation = torch.zeros((len(p), 6), device=device, dtype=torch.long)
                    continuation[:, :5] = p
                    continuation[:, 5] = first
                    second = model(continuation, n[:, None])[:, 0].argmax(-1)
                predictions.append(first.cpu().numpy())
                ended.append(second.eq(EOS).cpu().numpy())
            finally:
                hook.remove()
    return {"prediction": np.concatenate(predictions), "ended": np.concatenate(ended)}


def _candidate_values(world, chain, recipients, donors):
    default = world.answers[world.root_ids[chain, world.memberships[chain]]]
    actual = world.answers[world.actual_ids[chain]]
    return np.column_stack(
        [default[recipients], actual[recipients], default[donors], actual[donors]]
    )


def _node_rows(arrays, identity):
    rows = []
    for pair_type, label in enumerate(PAIR_TYPES):
        for population in ("all", "ordinary", "old_exception"):
            for common_only in (False, True):
                selected = arrays["pair_type"] == pair_type
                if population != "all":
                    selected &= arrays["recipient_old_exception"] == (population == "old_exception")
                if common_only:
                    selected &= arrays["common_recipient"]
                n = int(selected.sum())
                before = arrays["baseline_correct"][selected]
                after = arrays["correct"][selected]
                row = {
                    **identity,
                    "pair_type": label,
                    "population": population,
                    "common_recipients_only": common_only,
                    "n": n,
                    "baseline_correct": int(before.sum()),
                    "correct": int(after.sum()),
                    "known_became_wrong": int((before & ~after).sum()),
                    "wrong_became_correct": int((~before & after).sum()),
                    "termination_error": int((~arrays["ended"][selected]).sum()),
                    "prediction_changed": int(
                        (
                            arrays["prediction"][selected]
                            != arrays["baseline_prediction"][selected]
                        ).sum()
                    ),
                    "indistinguishable": int(
                        (arrays["class_code"][selected] == INDISTINGUISHABLE).sum()
                    ),
                    "mean_perturbation_norm": float(arrays["perturbation_norm"][selected].mean())
                    if n
                    else None,
                }
                for i, role in enumerate(ANSWER_ROLES):
                    row[f"matches_{role}"] = int(
                        (
                            ((arrays["matched_role_bits"][selected] & (1 << i)) != 0)
                            & arrays["ended"][selected]
                        ).sum()
                    )
                rows.append(row)
    return rows


def diagnose_causal_run(run, output, step=15360, device="cuda", batch_size=256, per_population=64):
    """One model, both relations, all layers/positions with fixed donor controls."""
    import torch

    from .bios_model import CausalLM, ModelConfig
    from .bios_organization_train import atomic_numpy_save

    repository = Path(__file__).resolve().parents[2]
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError("Never write causal diagnostics inside frozen training runs")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    if config["world"] not in (0, 1) or config["model"]["width"] not in (256, 768):
        raise ValueError("This development diagnostic is frozen to worlds 0/1, widths 256/768")
    checkpoint = run / f"model-{step}.pt"
    device = torch.device(device)
    world = make_cross_world(config["world"], repository / "data/bios-organization-v1")
    if array_hash(world.answers) != config["truth_sha256"]:
        raise ValueError("World truth mismatch")
    identity = {
        "run": str(run),
        "checkpoint": str(checkpoint),
        "step": step,
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(run / "config.json"),
        "diagnostic_source_sha256": file_sha256(__file__),
        "generation_helper_sha256": file_sha256(
            repository / "src/llm_memory_editability/bios_path_diagnostics.py"
        ),
        "world": config["world"],
        "seed": config["seed"],
        "condition": config["condition"],
        "width": config["model"]["width"],
        "per_population": per_population,
        "positions": list(POSITIONS),
        "device_type": device.type,
        "batch_size": batch_size,
        "torch": torch.__version__,
        "offline_truth_matching": True,
        "role": "candidate_localization" if config["world"] == 0 else "development_replication",
    }
    config_path = output / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != identity:
            raise ValueError("Causal diagnostic identity changed")
    else:
        write_json(config_path, identity)
    if (output / "complete.json").exists():
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CausalLM(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    rows, plan_rows = [], []
    source_output_hashes = {}
    for chain, name in enumerate(CHAINS):
        dest = output / name
        dest.mkdir(exist_ok=True)
        plan = make_donor_plan(world, chain, per_population)
        plan_path = dest / "plan.npz"
        if plan_path.exists():
            with np.load(plan_path) as existing:
                if any(not np.array_equal(existing[k], v) for k, v in plan.items()):
                    raise ValueError("Saved donor plan differs")
        else:
            atomic_numpy_save(plan_path, **plan)
        source_output_hashes[str(plan_path.relative_to(output))] = file_sha256(plan_path)
        valid = plan["donor"] >= 0
        selected = {k: v[valid] for k, v in plan.items()}
        recipients, donors = selected["recipient"], selected["donor"]
        for pair_type, label in enumerate(PAIR_TYPES):
            for exceptional in (False, True):
                matching = (plan["pair_type"] == pair_type) & (
                    plan["recipient_old_exception"] == exceptional
                )
                plan_rows.append(
                    {
                        "chain": name,
                        "pair_type": label,
                        "population": "old_exception" if exceptional else "ordinary",
                        "requested": int(matching.sum()),
                        "valid": int((matching & valid).sum()),
                        "missing": int((matching & ~valid).sum()),
                    }
                )
        people = np.unique(np.concatenate([recipients, donors]))
        ids = world.derived_ids[chain, people]
        cache_path = dest / "clean-cache.npz"
        if cache_path.exists():
            with np.load(cache_path) as cache:
                states = cache["states"]
                baseline = {k: cache[k] for k in ("prediction", "ended")}
                if not np.array_equal(cache["people"], people):
                    raise ValueError("Clean cache people mismatch")
        else:
            states = cache_clean_states(
                model, world.prompts[ids], world.lengths[ids], device, batch_size
            )
            baseline = free_generate_values(
                model, world.prompts[ids], world.lengths[ids], device, batch_size
            )
            atomic_numpy_save(cache_path, people=people, states=states, **baseline)
        source_output_hashes[str(cache_path.relative_to(output))] = file_sha256(cache_path)
        ri, di = np.searchsorted(people, recipients), np.searchsorted(people, donors)
        recipient_queries = world.derived_ids[chain, recipients]
        candidates = _candidate_values(world, chain, recipients, donors)
        base = {
            **selected,
            "recipient_query": recipient_queries,
            "donor_query": world.derived_ids[chain, donors],
            "candidate_tokens": candidates,
            "baseline_prediction": baseline["prediction"][ri],
            "baseline_ended": baseline["ended"][ri],
            "baseline_correct": (baseline["prediction"][ri] == candidates[:, 0])
            & baseline["ended"][ri],
        }
        for layer in range(len(model.blocks)):
            for position_index, position in enumerate(POSITIONS):
                receiver_states = states[layer, ri, position_index]
                donor_states = states[layer, di, position_index]
                seed = [world.seed, 972, chain, layer, position]
                for intervention in ("donor", "matched_random", "sham"):
                    node = dest / f"layer-{layer}-position-{position}-{intervention}.npz"
                    if node.exists():
                        with np.load(node) as existing:
                            arrays = dict(existing)
                        if not np.array_equal(arrays["case_id"], selected["case_id"]):
                            raise ValueError("Saved node case mismatch")
                    else:
                        replacement, donor_norm, actual_norm = intervention_state(
                            receiver_states, donor_states, intervention, seed
                        )
                        generated = generate_with_state_patch(
                            model,
                            world.prompts[recipient_queries],
                            world.lengths[recipient_queries],
                            replacement,
                            layer,
                            position,
                            device,
                            batch_size,
                        )
                        classes, bits = classify_answers(
                            generated["prediction"], generated["ended"], candidates
                        )
                        arrays = {
                            **base,
                            **generated,
                            "class_code": classes,
                            "matched_role_bits": bits,
                            "correct": (generated["prediction"] == candidates[:, 0])
                            & generated["ended"],
                            "donor_delta_norm": donor_norm,
                            "perturbation_norm": actual_norm,
                        }
                        atomic_numpy_save(node, **arrays)
                    source_output_hashes[str(node.relative_to(output))] = file_sha256(node)
                    node_identity = {
                        "world": world.seed,
                        "seed": config["seed"],
                        "width": config["model"]["width"],
                        "condition": config["condition"],
                        "chain": name,
                        "layer": layer,
                        "position": position,
                        "intervention": intervention,
                        "terminal_control": layer == len(model.blocks) - 1,
                    }
                    rows.extend(_node_rows(arrays, node_identity))
                print(
                    json.dumps(
                        {
                            "event": "causal_node_complete",
                            "run": str(run),
                            "chain": name,
                            "layer": layer,
                            "position": position,
                        }
                    ),
                    flush=True,
                )
    for filename, contents in (("summary.csv", rows), ("pair-coverage.csv", plan_rows)):
        with (output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contents[0]))
            writer.writeheader()
            writer.writerows(contents)
    write_json(
        output / "schema.json",
        {
            "pair_types": PAIR_TYPES,
            "answer_roles": ANSWER_ROLES,
            "state": "post-transformer-block residual stream",
            "positions": {
                "2": "membership relation",
                "3": "default relation",
                "4": "answer marker",
            },
            "world_roles": {
                "0": "candidate localization",
                "1": "development replication, not confirmation",
            },
            "random_control": "Random vector matched per case to ||donor_state-recipient_state||_2",
            "matching": "True group/actual/default answers select pairs offline; prompts contain "
            "no true answer or true bridge entity.",
            "missing": "No donor replacement across types; no prediction-based recipient selection",
            "common_recipients": "Auxiliary rows require all four donor types; truth-selected",
            "last_layer_controls": "Positions 2/3 cannot affect other tokens after final block; "
            "position 4 directly changes value readout and is not routing evidence.",
            "causal_scope": "Single-state replacement establishes an intervention effect, not a "
            "unique mechanism or strict necessity. Same-point restoration equals sham.",
            "eos": "Value and EOS are independently generated with identical intervention; "
            "non-terminated values are failures even if an answer candidate matches.",
            "overlap": "All matched roles retained; multiple roles remain indistinguishable",
        },
    )
    write_json(output / "artifacts.json", source_output_hashes)
    write_json(
        output / "complete.json",
        {
            "complete": True,
            "identity": identity,
            "node_files": 2 * len(model.blocks) * len(POSITIONS) * 3,
            "plan_rows": plan_rows,
            "artifacts_sha256": file_sha256(output / "artifacts.json"),
        },
    )
