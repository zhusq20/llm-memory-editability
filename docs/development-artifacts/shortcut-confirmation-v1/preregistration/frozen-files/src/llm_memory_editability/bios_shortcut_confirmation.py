"""Locked independent-world behavioral confirmation, never a route-mediation claim.

The original learning implementation and the audited E39 update implementation
are called directly. An external prospective lock is mandatory before execution.
"""

import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .bios_cross import CHAINS, CONDITIONS, make_cross_world
from .bios_cross_train import learning, source_hashes
from .bios_data import write_json
from .bios_organization_train import atomic_numpy_save, state_hash
from .bios_path_diagnostics import autonomous_two_step
from .bios_shortcut_confirmation_sets import make_confirmation_edit_pair
from .bios_shortcut_control import high_exception_world
from .bios_shortcut_edit_v2 import STUDY as DEVELOPMENT_EDIT_STUDY
from .bios_shortcut_edit_v2 import run_case
from .bios_shortcut_matched_edit import KINDS, PHASES

PROTOCOL = "v2.9-shortcut-behavior-confirmation"
WORLDS = tuple(range(100, 108))
LEARNING_CHECKPOINTS = (0, 1280, 2560, 5120, 10240, 15360)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def confirmation_sources():
    directory = Path(__file__).parent
    return {
        **source_hashes(),
        "scripts/prepare_bios_shortcut_confirmation.py": file_hash(
            directory.parents[1] / "scripts/prepare_bios_shortcut_confirmation.py"
        ),
        **{
            name: file_hash(directory / name)
            for name in (
                "bios_shortcut_confirmation.py",
                "bios_shortcut_confirmation_sets.py",
                "bios_shortcut_control.py",
                "bios_shortcut_matched_edit.py",
                "bios_shortcut_edit_v2.py",
                "bios_cross_continue.py",
                "bios_path_diagnostics.py",
            )
        },
    }


def validate_config(config):
    expected = {
        "protocol": PROTOCOL,
        "worlds": list(WORLDS),
        "seeds": [0, 1],
        "conditions": list(CONDITIONS),
        "phases": list(PHASES),
        "chains": list(CHAINS),
        "supports": [0, 1],
        "width": 256,
        "layers": 8,
        "heads": 4,
        "steps": 15360,
        "checkpoints": list(LEARNING_CHECKPOINTS),
        "lr": 0.0001,
        "documents_per_step": 16,
        "facts_per_document": 10,
        "QA_per_chain_per_step": 20,
        "document_QA_weights": [0.8, 0.2],
        "learning_runs": 96,
        "edit_cases": 768,
        "edit_steps": 512,
        "edit_checkpoints": [0, 32, 128, 512],
        "edit_lr": 0.00003,
        "retention_kl": 1.0,
        "edit_scope": "mlp",
        "edit_mlp_layers": [3, 4, 5],
        "edit_batch": 128,
        "replay_batch": 128,
        "edit_weight_decay": 0.1,
        "E": 39,
        "D": 96,
        "R": 4096,
        "base_world_root": "data/bios-shortcut-confirmation-v1",
        "source_root": "data/bios-source",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Prospective confirmation contract differs: {key}")
    for key in ("steps", "checkpoints", "lr", "retention_kl", "E", "D", "R"):
        source = {"steps": "edit_steps", "checkpoints": "edit_checkpoints", "lr": "edit_lr"}
        if DEVELOPMENT_EDIT_STUDY[key] != config[source.get(key, key)]:
            raise ValueError(f"Imported E39 scientific update differs: {key}")
    if config.get("edit_sampling") != "shared original world932chain stream across supports":
        raise ValueError("The two supports retain the original paired sampling stream")


def validate_frozen_design(config_path, lock_path):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("status") != "frozen_before_any_confirmation_model_or_outcome":
        raise ValueError("A prospective confirmation lock is required")
    if lock["config_sha256"] != file_hash(config_path):
        raise ValueError("Confirmation configuration changed after locking")
    if lock["sources"] != confirmation_sources():
        raise ValueError("Confirmation scientific sources changed after locking")
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__}
    if lock["environment"] != environment:
        raise ValueError("Confirmation numerical environment changed after locking")
    if not lock.get("analysis_sources"):
        raise ValueError("The prospective analysis implementation must also be locked")
    root = Path(__file__).resolve().parents[2]
    for relative, checksum in lock["analysis_sources"].items():
        if file_hash(root / relative) != checksum:
            raise ValueError(f"Prospective analysis source changed: {relative}")
    if not lock.get("data_source_files"):
        raise ValueError("Original data source files must be prospectively locked")
    for relative, checksum in lock["data_source_files"].items():
        if file_hash(root / relative) != checksum:
            raise ValueError(f"Original data source changed: {relative}")
    return config, lock


def validate_lock(config_path, lock_path, args):
    config, lock = validate_frozen_design(config_path, lock_path)
    if torch.device(args.device).type != "cuda" or lock.get("device_type") != "cuda":
        raise ValueError("Confirmation requires CUDA BF16 execution, never mixed CPU precision")
    if args.world not in WORLDS or args.seed not in (0, 1):
        raise ValueError("Only the eight reserved worlds and two initializations are allowed")
    if args.phase not in PHASES or args.condition not in CONDITIONS:
        raise ValueError("Condition outside the prospective matrix")
    expected_output = (
        Path(lock["output_root"])
        / args.phase
        / "width-256"
        / f"world-{args.world}-seed-{args.seed}-{args.condition}"
    ).resolve()
    if Path(args.output).resolve() != expected_output:
        raise ValueError("Output differs from the prospective matrix")
    return config, lock


def validate_prepared_world(config, config_path, lock_path, world):
    if world not in WORLDS:
        raise ValueError("Requested world is outside the prospective preparation matrix")
    root = Path(__file__).resolve().parents[2] / config["base_world_root"]
    receipt = json.loads((root / "preparation-audit.json").read_text())
    lock = json.loads(Path(lock_path).read_text())
    if (
        receipt.get("complete") is not True
        or receipt["config_sha256"] != file_hash(config_path)
        or receipt["lock_sha256"] != file_hash(lock_path)
        or receipt["worlds"] != list(WORLDS)
    ):
        raise ValueError("Base worlds lack a complete prospectively locked preparation")
    if receipt.get("source_files_sha256") != lock["data_source_files"]:
        raise ValueError("Prepared world source files differ from the prospective lock")
    filenames = ("world.npz", "metadata.json", "audit.json")
    expected_files = {f"world-{seed}/{name}" for seed in WORLDS for name in filenames}
    if set(receipt.get("files_sha256", {})) != expected_files:
        raise ValueError("Prepared world file manifest differs from the prospective matrix")
    for name in filenames:
        relative = f"world-{world}/{name}"
        if receipt["files_sha256"].get(relative) != file_hash(root / relative):
            raise ValueError(f"Prepared world changed: {relative}")
    metadata = json.loads((root / f"world-{world}/metadata.json").read_text())
    if type(metadata.get("seed")) is not int or metadata["seed"] != world:
        raise ValueError("Prepared world metadata seed differs from the requested world")
    return root, receipt


def case_identity(worker_identity, chain, support, kind, pair):
    payload = {
        "worker_identity": worker_identity,
        "chain": chain,
        "support": support,
        "kind": kind,
        "pair_contract": pair["contract"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def expected_weight_artifacts():
    names = [f"learning/model-{step}.pt" for step in LEARNING_CHECKPOINTS]
    names.append("learning/resume.pt")
    names.extend(
        f"edits/support-{support}/{chain}-{kind}-mlp/{filename}"
        for support in (0, 1)
        for chain in CHAINS
        for kind in KINDS
        for filename in ("model-final.pt", "resume.pt")
    )
    return names


def scientific_artifacts(output):
    for relative in expected_weight_artifacts():
        if not (output / relative).is_file():
            raise ValueError(f"Required confirmation weight artifact is missing: {relative}")
    paths = [output / "launch-contract.json", output / "manipulation.npz"]
    paths.extend(p for p in (output / "learning").iterdir() if p.is_file())
    paths.extend(p for p in (output / "edits").rglob("*") if p.is_file())
    return {
        str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(paths)
    }


def run(args):
    config, _ = validate_lock(args.config, args.lock, args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        base_root, data_receipt = validate_prepared_world(
            config, args.config, args.lock, args.world
        )
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("A healthy CUDA device with BF16 support is required")
        identity = {
            "protocol": PROTOCOL,
            "world": args.world,
            "seed": args.seed,
            "condition": args.condition,
            "phase": args.phase,
            "config_sha256": file_hash(args.config),
            "preregistration_lock_sha256": file_hash(args.lock),
            "sources": confirmation_sources(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "device_type": "cuda",
            "precision": "BF16 autocast; FP32 model and optimizer",
            "prepared_worlds_audit_sha256": file_hash(base_root / "preparation-audit.json"),
            "world_files_sha256": {
                name: checksum
                for name, checksum in data_receipt["files_sha256"].items()
                if name.startswith(f"world-{args.world}/")
            },
        }
        path = output / "launch-contract.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Confirmation worker identity changed")
        if not path.exists():
            write_json(path, identity)
        if (output / "complete.json").exists():
            return
        try:
            low = make_cross_world(args.world, source_root=base_root)
            high, manipulation, selected, donors = high_exception_world(low)
            world = low if args.phase == "low" else high
            pairs = {
                (chain, support): make_confirmation_edit_pair(low, high, chain, support)
                for chain in range(2)
                for support in (0, 1)
            }
            for chain in range(2):
                for key in ("groups", "E", "D", "selected_people"):
                    if np.intersect1d(pairs[chain, 0][key], pairs[chain, 1][key]).size:
                        raise ValueError(f"The two supports overlap: {key}")
            atomic_numpy_save(
                output / "manipulation.npz",
                selections=selected,
                donors=donors,
                low_answers=low.answers,
                high_answers=high.answers,
                high_exceptions=high.exceptions,
            )
            study = {
                **config,
                "confirmation_sources": identity["sources"],
                "phase": args.phase,
                "manipulation": manipulation,
            }
            learning_output = output / "learning"
            learning_output.mkdir(exist_ok=True)
            device = torch.device(args.device)
            model, data = learning(args, world, learning_output, study, device)
            with np.load(learning_output / "predictions-15360.npz") as saved:
                baseline_predictions = {
                    key: saved[key] for key in ("prediction", "ended", "correct", "value_nll")
                }
            np.testing.assert_array_equal(
                baseline_predictions["correct"],
                (baseline_predictions["prediction"] == world.answers)
                & baseline_predictions["ended"],
            )
            completed = json.loads((learning_output / "learning-complete.json").read_text())
            if completed["model_sha256"] != state_hash(model.state_dict()):
                raise ValueError("Learning baseline differs from completed weights")
            for chain, name in enumerate(CHAINS):
                arrays = autonomous_two_step(model, world, chain, device)
                ids = world.derived_ids[chain]
                arrays.update(
                    {f"direct_{key}": value[ids] for key, value in baseline_predictions.items()}
                )
                atomic_numpy_save(learning_output / f"two-step-{name}.npz", **arrays)
            baseline = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            identity_hash = file_hash(path)
            for support in (0, 1):
                edit_output = output / "edits" / f"support-{support}"
                for chain in range(2):
                    for kind in KINDS:
                        run_case(
                            model,
                            baseline,
                            data,
                            world,
                            pairs[chain, support],
                            args.phase,
                            kind,
                            chain,
                            edit_output,
                            case_identity(
                                identity_hash, chain, support, kind, pairs[chain, support]
                            ),
                            baseline_predictions,
                            device,
                        )
            write_json(
                output / "complete.json",
                {
                    "status": "complete",
                    "protocol": PROTOCOL,
                    "learning_steps": 15360,
                    "edit_cases": 8,
                    "edit_steps": 512,
                    "lock_sha256": file_hash(args.lock),
                    "artifacts": scientific_artifacts(output),
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
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
