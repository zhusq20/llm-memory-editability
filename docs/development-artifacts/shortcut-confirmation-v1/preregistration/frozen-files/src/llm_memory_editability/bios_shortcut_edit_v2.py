"""Prepared, resumable E39 MLP edits of matched low/high 15360-step parents."""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .bios_cross import CHAINS, CONDITIONS, make_cross_world
from .bios_cross_continue import (
    capture_state,
    edit_update,
    file_hash,
    restore_state,
    validate_optimizer,
)
from .bios_cross_train import evaluate, source_hashes, tensor_queries
from .bios_data import array_hash, rng_for, write_json
from .bios_model import CausalLM, ModelConfig, select_parameters
from .bios_organization_train import atomic_numpy_save, atomic_torch_save, state_hash
from .bios_shortcut_control import high_exception_world, shortcut_sources
from .bios_shortcut_matched_edit import KINDS, PHASES, make_matched_edit_pair
from .bios_train import precision

CHECKPOINTS = (0, 32, 128, 512)
STUDY = {
    "protocol": "v2.8-p3-shortcut-matched-E39",
    "width": 256,
    "parent_step": 15360,
    "phases": list(PHASES),
    "worlds": [0, 1],
    "seeds": [0, 1],
    "conditions": list(CONDITIONS),
    "chains": list(CHAINS),
    "kinds": list(KINDS),
    "scope": "mlp",
    "mlp_layers": [3, 4, 5],
    "lr": 0.00003,
    "weight_decay": 0.1,
    "edit_batch": 128,
    "replay_batch": 128,
    "retention_kl": 1.0,
    "clip_norm": 1.0,
    "steps": 512,
    "checkpoints": list(CHECKPOINTS),
    "resume_every": 32,
    "models": 24,
    "edit_cases": 96,
    "E": 39,
    "D": 96,
    "R": 4096,
    "primary_reference_D": 18,
    "primary_reference_D_heldout": 9,
    "status": "prepared conditional branch; no best-step selection",
}


def validate_study(study):
    if study != STUDY:
        raise ValueError("E39 editing configuration differs from the frozen prepared contract")


def editing_sources():
    directory = Path(__file__).parent
    return {
        **source_hashes(),
        **{
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "bios_cross_continue.py",
                "bios_shortcut_control.py",
                "bios_shortcut_matched_edit.py",
                "bios_shortcut_edit_v2.py",
            )
        },
    }


def sampling_stream(world_seed, chain, steps=512, batch=128, e_size=39, r_size=4096):
    """No phase, organization, initialization, or update-type dependence."""
    rng = rng_for(world_seed, 932, chain)
    return rng.integers(e_size, size=(steps, batch)), rng.integers(r_size, size=(steps, batch))


def score_arrays(arrays, truth):
    if any(arrays[key].shape != truth.shape for key in ("prediction", "ended", "correct")):
        raise ValueError("Prediction shape mismatch")
    if arrays["ended"].dtype != np.bool_ or arrays["correct"].dtype != np.bool_:
        raise ValueError("EOS and correctness must be Boolean arrays")
    np.testing.assert_array_equal(
        arrays["correct"], (arrays["prediction"] == truth) & arrays["ended"]
    )
    return arrays


def query_metrics(correct, ids):
    n = len(ids)
    count = int(correct[ids].sum())
    return {"n": n, "correct": count, "accuracy": count / n if n else None}


def retention_metrics(correct, old_correct, ids):
    n = len(ids)
    known_ids = ids[old_correct[ids]]
    broken = int((~correct[known_ids]).sum())
    return {
        "n": n,
        "known": len(known_ids),
        "coverage": len(known_ids) / n if n else None,
        "broken": broken,
        "damage": broken / len(known_ids) if len(known_ids) else None,
    }


def edit_metrics(pair, phase, kind, arrays, old_correct):
    correct = score_arrays(arrays, pair[f"{phase}_{kind}"])["correct"]
    if old_correct.shape != correct.shape or old_correct.dtype != np.bool_:
        raise ValueError("Old-correct coverage array mismatch")
    result = {
        key: query_metrics(correct, pair[key])
        for key in (
            "E",
            "E_roots",
            "E_actual",
            "D",
            "D_trained",
            "D_heldout",
            "paired_reference_D",
            "paired_reference_D_heldout",
            "D_edited_actual",
            "D_edited_aligned",
            "D_original_exception",
            "D_newly_exception",
            "D_remaining_ordinary_unedited",
        )
    }
    if kind == "exception":
        result["exception_conflict_D"] = query_metrics(correct, pair["exception_conflict_D"])
        result["exception_conflict_D_heldout"] = query_metrics(
            correct, pair["exception_conflict_D_heldout"]
        )
    result["factual_conflict_D"] = query_metrics(
        correct, pair[f"{phase}_{kind}_factual_conflict_D"]
    )
    for pool in ("U_full", "U_heldout"):
        ids = pair[pool]
        result[pool] = {
            **retention_metrics(correct, old_correct, ids),
            "strata": {
                str(group): retention_metrics(
                    correct, old_correct, ids[pair["U_strata"][ids] == group]
                )
                for group in range(5)
            },
        }
    return result


def load_parent(parent, phase):
    config = json.loads((parent / "config.json").read_text())
    low = make_cross_world(config["world"])
    expected_model = dict(vocab_size=low.vocab_size, width=256, layers=8, heads=4, context=128)
    if config["model"] != expected_model or config["study"]["steps"] != 15360:
        raise ValueError("E39 requires original width256/15360 parents, never P2 continuations")
    if any(
        config["study"][key] != value
        for key, value in {
            "lr": 1e-4,
            "documents_per_step": 16,
            "facts_per_document": 10,
            "QA_per_chain_per_step": 20,
            "document_QA_weights": [0.8, 0.2],
        }.items()
    ):
        raise ValueError("E39 parent exposure or learning budget differs")
    if (
        config["world"] not in (0, 1)
        or config["seed"] not in (0, 1)
        or config["condition"] not in CONDITIONS
    ):
        raise ValueError("Parent is outside the fixed development matrix")
    if config["sources"] != source_hashes():
        raise ValueError("Parent frozen trainer sources changed")
    if config["torch"] != torch.__version__ or config["numpy"] != np.__version__:
        raise ValueError("Parent numerical software differs")
    if phase == "low" and config["study"]["protocol"] != "v2.6-development-crossover":
        raise ValueError("Low parent is not an original unrestricted-context run")
    if phase == "high" and config["study"].get("shortcut_contract") != shortcut_sources():
        raise ValueError("High parent manipulation/source provenance changed")
    high, _, _, _ = high_exception_world(low)
    world = low if phase == "low" else high
    if config["truth_sha256"] != array_hash(world.answers):
        raise ValueError("Parent truth differs from its declared prevalence phase")
    with np.load(parent / "predictions-15360.npz") as saved:
        predictions = score_arrays(
            {key: saved[key] for key in ("prediction", "ended", "correct", "value_nll")},
            world.answers,
        )
    checkpoint = torch.load(parent / "model-15360.pt", map_location="cpu", weights_only=False)
    complete = json.loads((parent / "learning-complete.json").read_text())
    if (
        checkpoint["step"] != 15360
        or checkpoint["config"] != config["model"]
        or complete["status"] != "complete"
    ):
        raise ValueError("Parent learning has not completed the specified checkpoint")
    if state_hash(checkpoint["model"]) != complete["model_sha256"]:
        raise ValueError("Parent checkpoint differs from completed learning weights")
    provenance = {
        "directory": str(parent),
        "files_sha256": {
            name: file_hash(parent / name)
            for name in (
                "config.json",
                "model-15360.pt",
                "predictions-15360.npz",
                "learning-complete.json",
            )
        },
        "model_sha256": complete["model_sha256"],
    }
    return config, checkpoint, predictions, low, high, world, provenance


def run_case(
    model,
    baseline,
    data,
    world,
    pair,
    phase,
    kind,
    chain,
    output,
    identity_hash,
    parent_predictions,
    device,
):
    dest = output / f"{CHAINS[chain]}-{kind}-mlp"
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(baseline)
    selected = select_parameters(model, "mlp", 3)
    e_choices, r_choices = sampling_stream(world.seed, chain)
    saved_sets = {key: value for key, value in pair.items() if isinstance(value, np.ndarray)}
    saved_sets.update(
        old_correct=parent_predictions["correct"],
        edit_sampling=e_choices,
        replay_sampling=r_choices,
    )
    atomic_numpy_save(dest / "sets.npz", **saved_sets)
    write_json(dest / "data-contract.json", pair["contract"])
    target = pair[f"{phase}_{kind}"]
    new_data = tensor_queries(world, device, target)
    model.eval()
    references = []
    with torch.no_grad(), precision(device):
        for begin in range(0, len(pair["R"]), 256):
            ids = torch.as_tensor(pair["R"][begin : begin + 256], device=device)
            references.append(model(data["tokens"][ids], data["positions"][ids]).detach())
    references = torch.cat(references)
    optimizer = torch.optim.AdamW(
        selected, lr=STUDY["lr"], weight_decay=0.1, fused=device.type == "cuda"
    )
    first, timeline, seconds = 0, [], 0.0
    if (dest / "resume.pt").exists():
        saved = torch.load(dest / "resume.pt", map_location="cpu", weights_only=False)
        if saved["identity_sha256"] != identity_hash or saved["case"] != dest.name:
            raise ValueError("E39 resume identity mismatch")
        if not 0 <= saved["step"] <= 512 or saved["step"] % 32:
            raise ValueError("E39 resume step mismatch")
        if saved["step"]:
            validate_optimizer(saved["optimizer"], saved["step"], STUDY["lr"])
        restore_state(model, optimizer, saved, device)
        first, timeline, seconds = saved["step"], saved["timeline"], saved["seconds"]
    for step in range(first, 513):
        if step in CHECKPOINTS and (not timeline or timeline[-1]["step"] != step):
            arrays = evaluate(model, new_data, target)
            if step == 0:
                for key in ("prediction", "ended"):
                    np.testing.assert_array_equal(arrays[key], parent_predictions[key])
            point = {
                "step": step,
                "edit_seconds": seconds,
                **edit_metrics(pair, phase, kind, arrays, parent_predictions["correct"]),
            }
            timeline.append(point)
            atomic_numpy_save(dest / f"predictions-{step}.npz", **arrays)
            write_json(dest / "trajectory.json", timeline)
        if step % 32 == 0:
            atomic_torch_save(
                capture_state(
                    model,
                    optimizer,
                    device,
                    step=step,
                    timeline=timeline,
                    seconds=seconds,
                    identity_sha256=identity_hash,
                    case=dest.name,
                ),
                dest / "resume.pt",
            )
        if step == 512:
            break
        ei = torch.as_tensor(pair["E"][e_choices[step]], device=device)
        ri = torch.as_tensor(pair["R"][r_choices[step]], device=device)
        choices = torch.as_tensor(r_choices[step], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        edit_update(model, optimizer, selected, data, new_data, references, ei, ri, choices, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - started
    atomic_torch_save(
        {"model": model.state_dict(), "config": model.config_dict(), "step": 512},
        dest / "model-final.pt",
    )
    write_json(
        dest / "complete.json",
        {
            "status": "complete",
            "phase": phase,
            "chain": CHAINS[chain],
            "kind": kind,
            "scope": "mlp",
            "final": timeline[-1],
            "model_sha256": state_hash(model.state_dict()),
        },
    )
    print(
        json.dumps(
            {
                "event": "E39_edit_complete",
                "case": dest.name,
                "E": timeline[-1]["E"]["accuracy"],
                "paired_reference_heldout": timeline[-1]["paired_reference_D_heldout"]["accuracy"],
            }
        ),
        flush=True,
    )


def run(args):
    validate_study(json.loads(Path(args.config).read_text()))
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("E39 output must be separate from its immutable parent")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        config, checkpoint, predictions, low, high, world, provenance = load_parent(
            parent, args.phase
        )
        pairs = [make_matched_edit_pair(low, high, chain) for chain in range(2)]
        identity = {
            "phase": args.phase,
            "world": world.seed,
            "seed": config["seed"],
            "condition": config["condition"],
            "study": STUDY,
            "sources": editing_sources(),
            "parent": provenance,
            "data_contracts": [pair["contract"] for pair in pairs],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "device_type": torch.device(args.device).type,
        }
        identity_path = output / "config.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("E39 editing identity or sources changed")
        if not identity_path.exists():
            write_json(identity_path, identity)
        if (output / "complete.json").exists():
            return
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        model = CausalLM(ModelConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        data = tensor_queries(world, device)
        try:
            baseline = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            observed = evaluate(model, data, world.answers)
            for key in ("prediction", "ended", "correct"):
                np.testing.assert_array_equal(observed[key], predictions[key])
            for chain in range(2):
                for kind in KINDS:
                    run_case(
                        model,
                        baseline,
                        data,
                        world,
                        pairs[chain],
                        args.phase,
                        kind,
                        chain,
                        output,
                        file_hash(identity_path),
                        predictions,
                        device,
                    )
            write_json(
                output / "complete.json",
                {
                    "status": "complete",
                    "phase": args.phase,
                    "edit_cases": 4,
                    "steps": 512,
                    "finished": time.time(),
                },
            )
        except Exception as error:
            write_json(
                output / "failure.json",
                {"type": type(error).__name__, "message": str(error), "time": time.time()},
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/bios-shortcut-edit-v1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
