"""H5: original E93 exception edits of all weights except tied token/readout weights."""

import argparse
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import torch

from .bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from .bios_cross_continue import (
    capture_state,
    edit_metrics,
    edit_update,
    file_hash,
    restore_state,
    validate_optimizer,
)
from .bios_cross_train import evaluate, source_hashes, tensor_queries
from .bios_data import array_hash, rng_for, write_json
from .bios_model import CausalLM, ModelConfig
from .bios_organization_train import atomic_numpy_save, atomic_torch_save, state_hash
from .bios_train import precision

CHECKPOINTS = (0, 32, 128, 512)
SCOPE = "all-freeze-embedding"
STUDY = {
    "protocol": "v2.8-h5-readout-control-E93",
    "widths": [256, 768],
    "parent_step": 15360,
    "worlds": [0, 1],
    "seeds": [0, 1],
    "conditions": list(CONDITIONS),
    "chains": list(CHAINS),
    "kind": "exception",
    "scope": SCOPE,
    "frozen": "token.weight, shared by input embedding and output F.linear readout",
    "trainable": "all other parameters, including absolute position embedding",
    "lr": 3e-5,
    "weight_decay": 0.1,
    "edit_batch": 128,
    "replay_batch": 128,
    "retention_kl": 1.0,
    "clip_norm": 1.0,
    "steps": 512,
    "checkpoints": list(CHECKPOINTS),
    "resume_every": 32,
    "models": 24,
    "edit_cases": 48,
    "E": 93,
    "D": 96,
    "R": 4096,
    "reference_scopes": ["mlp", "all"],
    "selection": "all fixed512 endpoints; no best step",
}


def validate_study(study):
    if study != STUDY:
        raise ValueError("H5 configuration differs from the fixed contract")


def control_sources():
    return {
        **source_hashes(),
        "bios_cross_continue.py": file_hash(Path(__file__).with_name("bios_cross_continue.py")),
        Path(__file__).name: file_hash(__file__),
    }


def select_control_parameters(model):
    """The ordinary model passes this same Parameter directly to F.linear."""
    excluded = id(model.token.weight)
    selected = []
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) != excluded)
        parameter.grad = None
        if parameter.requires_grad:
            selected.append(parameter)
    assert model.position.weight.requires_grad and not model.token.weight.requires_grad
    return selected


def sampling_stream(world_seed, chain, steps=512, batch=128):
    rng = rng_for(world_seed, 908, chain)
    return rng.integers(93, size=(steps, batch)), rng.integers(4096, size=(steps, batch))


def score_arrays(arrays, truth):
    for key in ("prediction", "ended", "correct"):
        if arrays[key].shape != truth.shape:
            raise ValueError("H5 prediction array shape mismatch")
    if arrays["ended"].dtype != np.bool_ or arrays["correct"].dtype != np.bool_:
        raise ValueError("H5 EOS/correct flags must be Boolean")
    np.testing.assert_array_equal(
        arrays["correct"], (arrays["prediction"] == truth) & arrays["ended"]
    )
    return arrays


def validate_parent_config(config, world):
    width = config["model"]["width"]
    if width not in STUDY["widths"] or config["world"] not in (0, 1):
        raise ValueError("H5 parent outside original size/world matrix")
    expected = dict(
        vocab_size=world.vocab_size, width=width, layers=8, heads=width // 64, context=128
    )
    if config["model"] != expected or config["study"]["protocol"] != "v2.6-development-crossover":
        raise ValueError("H5 needs original unrestricted low-prevalence architecture")
    expected_study = dict(
        steps=15360,
        lr=1e-4,
        documents_per_step=16,
        facts_per_document=10,
        QA_per_chain_per_step=20,
        document_QA_weights=[0.8, 0.2],
    )
    if any(config["study"][key] != value for key, value in expected_study.items()):
        raise ValueError("H5 parent is not the original learning budget")
    if config["seed"] not in (0, 1) or config["condition"] not in CONDITIONS:
        raise ValueError("H5 parent outside original initialization/organization matrix")
    if config["sources"] != source_hashes() or config["truth_sha256"] != array_hash(world.answers):
        raise ValueError("H5 parent source or truth mismatch")
    if config["torch"] != torch.__version__ or config["numpy"] != np.__version__:
        raise ValueError("H5 numerical software differs from parent")


def reference_contract(parent, world, old_arrays):
    hashes = {}
    for chain, name in enumerate(CHAINS):
        pair = edit_pair(world, chain)
        assert len(pair["E"]) == 93 and len(pair["D"]) == 96 and len(pair["replay"]) == 4096
        e, r = sampling_stream(world.seed, chain)
        for scope in ("mlp", "all"):
            dest = parent / "edits" / f"{name}-exception-{scope}"
            with np.load(dest / "sets.npz") as actual:
                for key, value in pair.items():
                    np.testing.assert_array_equal(actual[key], value)
                np.testing.assert_array_equal(actual["old_correct"], old_arrays["correct"])
                np.testing.assert_array_equal(actual["edit_sampling"], e)
                np.testing.assert_array_equal(actual["replay_sampling"], r)
            points = json.loads((dest / "trajectory.json").read_text())
            if [point["step"] for point in points] != list(CHECKPOINTS):
                raise ValueError("Original MLP/All reference checkpoint grid differs")
            if json.loads((dest / "complete.json").read_text())["status"] != "complete":
                raise ValueError("Original MLP/All reference incomplete")
            for step in CHECKPOINTS:
                with np.load(dest / f"predictions-{step}.npz") as saved:
                    arrays = score_arrays(dict(saved), pair["exception"])
                if step == 0:
                    for key in ("prediction", "ended"):
                        np.testing.assert_array_equal(arrays[key], old_arrays[key])
            for filename in (
                "sets.npz",
                "trajectory.json",
                "complete.json",
                *(f"predictions-{step}.npz" for step in CHECKPOINTS),
            ):
                hashes[str((dest / filename).relative_to(parent))] = file_hash(dest / filename)
    return hashes


def load_parent(parent):
    config = json.loads((parent / "config.json").read_text())
    world = make_cross_world(config["world"])
    validate_parent_config(config, world)
    with np.load(parent / "predictions-15360.npz") as saved:
        arrays = score_arrays(dict(saved), world.answers)
    references = reference_contract(parent, world, arrays)
    checkpoint = torch.load(parent / "model-15360.pt", map_location="cpu", weights_only=False)
    complete = json.loads((parent / "learning-complete.json").read_text())
    if checkpoint["step"] != 15360 or checkpoint["config"] != config["model"]:
        raise ValueError("H5 parent checkpoint identity differs")
    if (
        complete["status"] != "complete"
        or state_hash(checkpoint["model"]) != complete["model_sha256"]
    ):
        raise ValueError("H5 parent learning checkpoint is not verified complete")
    provenance = dict(
        directory=str(parent),
        model_sha256=complete["model_sha256"],
        reference_files_sha256=references,
        files_sha256={
            name: file_hash(parent / name)
            for name in (
                "config.json",
                "model-15360.pt",
                "predictions-15360.npz",
                "learning-complete.json",
            )
        },
    )
    return config, world, arrays, checkpoint, provenance


def run_case(model, baseline, data, world, parent_arrays, chain, output, identity_hash, device):
    dest = output / f"{CHAINS[chain]}-exception-{SCOPE}"
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(baseline)
    selected = select_control_parameters(model)
    frozen = baseline["token.weight"]
    pair = edit_pair(world, chain)
    e, r = sampling_stream(world.seed, chain)
    atomic_numpy_save(
        dest / "sets.npz",
        **pair,
        old_correct=parent_arrays["correct"],
        edit_sampling=e,
        replay_sampling=r,
    )
    new_data = tensor_queries(world, device, pair["exception"])
    model.eval()
    references = []
    with torch.no_grad(), precision(device):
        for begin in range(0, len(pair["replay"]), 256):
            ids = torch.as_tensor(pair["replay"][begin : begin + 256], device=device)
            references.append(model(data["tokens"][ids], data["positions"][ids]).detach())
    references = torch.cat(references)
    optimizer = torch.optim.AdamW(selected, lr=3e-5, weight_decay=0.1, fused=device.type == "cuda")
    first, timeline, seconds = 0, [], 0.0
    if (dest / "resume.pt").exists():
        saved = torch.load(dest / "resume.pt", map_location="cpu", weights_only=False)
        if saved["identity_sha256"] != identity_hash or saved["case"] != dest.name:
            raise ValueError("H5 resume identity differs")
        if not 0 <= saved["step"] <= 512 or saved["step"] % 32:
            raise ValueError("H5 resume step differs")
        if saved["step"]:
            validate_optimizer(saved["optimizer"], saved["step"], 3e-5)
        restore_state(model, optimizer, saved, device)
        first, timeline, seconds = saved["step"], saved["timeline"], saved["seconds"]
    for step in range(first, 513):
        if step in CHECKPOINTS:
            if not torch.equal(model.token.weight.detach().cpu(), frozen):
                raise ValueError("Frozen input/output token weights changed")
            if not timeline or timeline[-1]["step"] != step:
                arrays = score_arrays(
                    evaluate(model, new_data, pair["exception"]), pair["exception"]
                )
                if step == 0:
                    for key in ("prediction", "ended"):
                        np.testing.assert_array_equal(arrays[key], parent_arrays[key])
                timeline.append(
                    dict(
                        step=step,
                        edit_seconds=seconds,
                        **edit_metrics(world, pair, arrays, parent_arrays["correct"]),
                    )
                )
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
        ei = torch.as_tensor(pair["E"][e[step]], device=device)
        ri = torch.as_tensor(pair["replay"][r[step]], device=device)
        choices = torch.as_tensor(r[step], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        edit_update(model, optimizer, selected, data, new_data, references, ei, ri, choices, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - start
    atomic_torch_save(
        dict(model=model.state_dict(), config=model.config_dict(), step=512),
        dest / "model-final.pt",
    )
    write_json(
        dest / "complete.json",
        dict(
            status="complete",
            scope=SCOPE,
            kind="exception",
            chain=CHAINS[chain],
            final=timeline[-1],
            model_sha256=state_hash(model.state_dict()),
            frozen_token_sha256=state_hash({"token.weight": frozen}),
        ),
    )
    print(
        json.dumps(dict(event="H5_edit_complete", case=dest.name, final=timeline[-1])), flush=True
    )


def run(args):
    validate_study(json.loads(Path(args.config).read_text()))
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("H5 output must be separate from immutable parent")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        config, world, arrays, checkpoint, provenance = load_parent(parent)
        device = torch.device(args.device)
        identity = dict(
            world=world.seed,
            seed=config["seed"],
            condition=config["condition"],
            width=config["model"]["width"],
            study=STUDY,
            sources=control_sources(),
            parent=provenance,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            numpy=np.__version__,
            device_type=device.type,
        )
        path = output / "config.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Frozen H5 worker contract changed")
        if not path.exists():
            write_json(path, identity)
        if (output / "complete.json").exists():
            return
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        model = CausalLM(ModelConfig(**checkpoint["config"])).to(device)
        model.load_state_dict(checkpoint["model"])
        baseline = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        data = tensor_queries(world, device)
        try:
            observed = score_arrays(evaluate(model, data, world.answers), world.answers)
            for key in ("prediction", "ended", "correct"):
                np.testing.assert_array_equal(observed[key], arrays[key])
            for chain in range(2):
                run_case(
                    model, baseline, data, world, arrays, chain, output, file_hash(path), device
                )
            write_json(
                output / "complete.json",
                dict(status="complete", edit_cases=2, steps=512, scope=SCOPE, finished=time.time()),
            )
        except Exception as error:
            write_json(
                output / "failure.json",
                dict(type=type(error).__name__, message=str(error), time=time.time()),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/bios-readout-control-v1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
