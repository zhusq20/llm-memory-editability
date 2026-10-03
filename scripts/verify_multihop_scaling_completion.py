"""Independently verify matrix identity, archived data, initialization and exposure."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import utc
from llm_memory_editability.storage_composition import file_hash

NODES = [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
CHECKPOINTS = [0, 8000, 32000, 64000]
ARCHITECTURES = [(2, 1), (4, 1), (1, 2), (1, 4)]
SPLITS = ("atomic",) + tuple(
    f"{pool}_{hop}" for hop in (2, 3, 4) for pool in ("train", "familiar", "strict")
)
COMMON = tuple(name for name in SPLITS if not name.startswith("train_"))
TRAIN = ("atomic", "train_2", "train_3", "train_4")


def name(spec):
    return (
        f"w{spec['world_seed']}-i{spec['initialization']}-d{spec['width']}"
        f"-phi{spec['phi']:g}-l{spec['layers']}-r{spec['repeats']}-s{spec['steps']}"
    )


def matrix(config):
    phase = config["phase"]
    if phase == "development":
        worlds, initializations = [761011], [762011]
    elif phase == "confirmation":
        worlds, initializations = [761101, 761102, 761103], [762101, 762102]
    else:
        raise AssertionError("Unknown phase")
    expected = {
        (world, initial, width, phi, layers, repeats)
        for world, initial, width, phi, (layers, repeats) in itertools.product(
            worlds, initializations, [128, 256], [1.0, 4.0], ARCHITECTURES
        )
    }
    actual = [
        (s["world_seed"], s["initialization"], s["width"], s["phi"], s["layers"], s["repeats"])
        for s in config["specs"]
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise AssertionError("Frozen factorial matrix differs or contains duplicates")
    for spec in config["specs"]:
        if (
            spec["steps"] != 64000
            or spec["nodes"] != NODES
            or spec["checkpoint_nodes"] != CHECKPOINTS
            or spec["batch_size"] != 128
            or spec["stream_seed"] != spec["world_seed"] + 2000
        ):
            raise AssertionError("Training budget, nodes or stream pairing differ")
    return len(expected)


def array_hash(arrays, metadata, common=False):
    digest = hashlib.sha256()
    for key in COMMON if common else SPLITS:
        rows = arrays[key]
        digest.update(key.encode())
        digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
        digest.update(rows.astype("<i8", copy=False).tobytes())
    if common:
        digest.update(json.dumps(metadata["id_mask"]).encode())
    else:
        digest.update(
            json.dumps(
                {key: metadata[key] for key in ("world_seed", "entities", "relations", "id_mask")},
                sort_keys=True,
            ).encode()
        )
    return digest.hexdigest()


def independent_truth(arrays, metadata):
    atoms = arrays["atomic"]
    lookup = {(int(h), int(r)): (i, int(t)) for i, (h, r, t) in enumerate(atoms)}
    if len(lookup) != 256 or atoms.shape != (256, 3) or atoms.dtype != np.int64:
        raise AssertionError("Atomic graph contract differs")
    ids = np.asarray(metadata["id_mask"], dtype=bool)
    if ids.shape != (256,) or ids.sum() != 192:
        raise AssertionError("Atomic ID/OOD partition differs")
    training_facts = set()
    strict_facts = set()
    for key in SPLITS:
        rows = arrays[key]
        width = 3 if key == "atomic" else int(key.rsplit("_", 1)[1]) + 2
        if rows.dtype != np.int64 or rows.ndim != 2 or rows.shape[1] != width:
            raise AssertionError("Archived compact data shape/dtype differs")
        queries = {tuple(row[:-1]) for row in rows}
        if len(queries) != len(rows):
            raise AssertionError("Repeated complete queries")
        for row in rows:
            current = int(row[0])
            facts = []
            for relation in row[1:-1]:
                index, current = lookup[current, int(relation)]
                facts.append(index)
            if current != row[-1]:
                raise AssertionError("Saved target differs from independently traversed graph")
            if key.startswith(("train_", "familiar_")) and not ids[facts].all():
                raise AssertionError("Non-ID fact appears in familiar composition")
            if key.startswith("strict_"):
                if ids[facts].any():
                    raise AssertionError("ID fact appears in strict composition")
                strict_facts.update(facts)
            elif key.startswith("train_"):
                training_facts.update(facts)
    if strict_facts & training_facts:
        raise AssertionError("Strict atomic facts acquired a composition role")
    for hop in (2, 3, 4):
        sets = [
            {tuple(row[:-1]) for row in arrays[f"{pool}_{hop}"]}
            for pool in ("train", "familiar", "strict")
        ]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise AssertionError("Complete query leakage between archived pools")


def state_hash(state, prefix=None):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if prefix is None or key.startswith(prefix):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def replay_counts(size, seed):
    """Independent NumPy shuffled epochs, cached once per factual sampling stream."""
    rng = np.random.default_rng(seed)
    remaining = np.empty(0, dtype=np.int64)
    counts = np.zeros(size, dtype=np.int64)
    nodes = {}
    previous = 0
    for node in NODES:
        needed = (node - previous) * 32
        while needed:
            if not len(remaining):
                remaining = rng.permutation(size)
            n = min(needed, len(remaining))
            counts += np.bincount(remaining[:n], minlength=size)
            remaining = remaining[n:]
            needed -= n
        nodes[node] = counts.copy()
        previous = node
    return nodes, {"size": size, "rng": rng.bit_generator.state, "remaining": remaining.copy()}


def verify(config_path, out, partial=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    expected_runs = matrix(config)
    artifacts, results = Path(config["artifacts"]), Path(config["results"])
    if file_hash(config_path) != file_hash(artifacts / "frozen-config.json"):
        raise AssertionError("Config changed after source freezing")
    if file_hash(config["design"]) != config["design_sha256"]:
        raise AssertionError("Design changed after freezing")
    for path, digest in config["source"].items():
        if file_hash(path) != digest or file_hash(artifacts / "source" / path) != digest:
            raise AssertionError("Frozen source changed: " + path)
    datasets, counter_cache, initials, pairs = {}, {}, {}, {}
    records, missing, failures = [], [], []
    for spec in config["specs"]:
        folder = results / name(spec)
        for stage in ("train", "audit"):
            path = folder / f"{stage}-process-status.json"
            if path.exists():
                status = json.loads(path.read_text())
                if status["returncode"] != 0:
                    failures.append({"run": folder.name, **status})
        if not (folder / "audit.json").exists():
            missing.append(folder.name)
            if partial:
                continue
            raise AssertionError("Required independently audited run absent: " + folder.name)
        result = json.loads((folder / "complete.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        recorded = json.loads((folder / "spec.json").read_text())
        if (
            result["spec"] != spec
            or recorded["spec"] != spec
            or result["source"] != config["source"]
            or recorded["source"] != config["source"]
            or not audit["passed"]
            or audit["checkpoint_nodes"] != CHECKPOINTS
        ):
            raise AssertionError("Run identity or GPU reload audit differs: " + folder.name)
        metadata = json.loads((folder / "world-metadata.json").read_text())
        with np.load(folder / "world.npz") as archive:
            if set(archive.files) != set(SPLITS):
                raise AssertionError("Saved dataset keys differ")
            arrays = {key: archive[key] for key in SPLITS}
        digest, common = array_hash(arrays, metadata), array_hash(arrays, metadata, True)
        if (
            digest != spec["frozen_data_sha256"]
            or digest != result["data_sha256"]
            or digest != recorded["data_sha256"]
            or digest != metadata["dataset_sha256"]
            or common != spec["common_evaluation_sha256"]
            or common != metadata["common_evaluation_sha256"]
        ):
            raise AssertionError("Independent saved-data hash differs")
        data_key = spec["world_seed"], spec["phi"]
        if data_key not in datasets:
            independent_truth(arrays, metadata)
            datasets[data_key] = arrays, metadata
        else:
            reference, reference_metadata = datasets[data_key]
            if reference_metadata != metadata:
                raise AssertionError("Paired world metadata differ")
            for key in SPLITS:
                np.testing.assert_array_equal(reference[key], arrays[key])
        history = json.loads((folder / "learning.json").read_text())
        if [row["step"] for row in history] != NODES or history[-1] != result["endpoint"]:
            raise AssertionError("Learning curve nodes or fixed endpoint differ")
        for row in history:
            node = row["step"]
            if row["supervised_tokens"] != node * 832 or row["padded_input_tokens"] != node * 1024:
                raise AssertionError("Actual token accounting differs")
            if node and not np.isfinite(row["loss"]):
                raise AssertionError("Nonfinite optimization loss")
            if not (folder / f"predictions-{node:06d}.npz").is_file():
                raise AssertionError("Saved prediction node missing")
            with np.load(folder / f"exposures-{node:06d}.npz") as actual:
                if set(actual.files) != set(TRAIN):
                    raise AssertionError("Exposure counter strata differ")
                for j, key in enumerate(TRAIN):
                    cache_key = len(arrays[key]), spec["stream_seed"] + j
                    if cache_key not in counter_cache:
                        counter_cache[cache_key] = replay_counts(*cache_key)
                    np.testing.assert_array_equal(actual[key], counter_cache[cache_key][0][node])
        initial = torch.load(folder / "model-000000.pt", map_location="cpu", weights_only=False)
        if initial["spec"] != spec or initial["step"] != 0:
            raise AssertionError("Initialization checkpoint identity differs")
        actual_initial = state_hash(initial["model"])
        if (
            actual_initial != spec["initial_model_sha256"]
            or actual_initial != result["initial_model_sha256"]
        ):
            raise AssertionError("Actual initial tensor hash differs from frozen specification")
        init_key = spec["initialization"], spec["width"], spec["layers"], spec["repeats"]
        if init_key in initials and initials[init_key] != actual_initial:
            raise AssertionError("Initialization differs across world/support pairs")
        initials[init_key] = actual_initial
        block_key = spec["initialization"], spec["width"], spec["layers"] * spec["repeats"]
        block_hash = state_hash(initial["model"], "blocks.0.")
        if block_key in pairs and pairs[block_key] != block_hash:
            raise AssertionError("First block differs between matched ordinary and Loop compute")
        pairs[block_key] = block_hash
        del initial
        for node in CHECKPOINTS:
            if not (folder / f"model-{node:06d}.pt").is_file():
                raise AssertionError("Required saved model node missing")
        if file_hash(folder / "latest.pt") != result["checkpoint_sha256"]:
            raise AssertionError("Final optimizer/model checkpoint changed")
        latest = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
        if (
            latest["spec"] != spec
            or latest["step"] != 64000
            or latest["source"] != config["source"]
            or latest["data_sha256"] != digest
        ):
            raise AssertionError("Latest-state model/data/source identity differs")
        with np.load(folder / "exposures.npz") as actual:
            for j, key in enumerate(TRAIN):
                cache_key = len(arrays[key]), spec["stream_seed"] + j
                expected, expected_state = counter_cache[cache_key]
                np.testing.assert_array_equal(actual[key], expected[64000])
                np.testing.assert_array_equal(latest["counts"][j], expected[64000])
                np.testing.assert_array_equal(
                    latest["streams"][j]["remaining"], expected_state["remaining"]
                )
                if (
                    latest["streams"][j]["rng"] != expected_state["rng"]
                    or latest["streams"][j]["size"] != expected_state["size"]
                ):
                    raise AssertionError("Sampling RNG state differs from independent replay")
        del latest
        records.append(
            {
                "run": folder.name,
                "dataset_sha256": digest,
                "common_evaluation_sha256": common,
                "initial_model_sha256": actual_initial,
                "latest_checkpoint_sha256": result["checkpoint_sha256"],
                "reload_audit_sha256": file_hash(folder / "audit.json"),
                "learning_sha256": file_hash(folder / "learning.json"),
                "steps": 64000,
                "saved_nodes": NODES,
                "checkpoint_nodes": CHECKPOINTS,
                "atomic_accuracy": result["endpoint"]["metrics"]["atomic"]["accuracy"],
                "all_exposure_nodes_and_rng_replay_exact": True,
            }
        )
    worlds = {key[0] for key in datasets}
    for world in worlds:
        low, high = datasets.get((world, 1.0)), datasets.get((world, 4.0))
        if low is None or high is None:
            if partial:
                continue
            raise AssertionError("Support arm absent")
        for key in COMMON:
            np.testing.assert_array_equal(low[0][key], high[0][key])
        if low[1]["id_mask"] != high[1]["id_mask"]:
            raise AssertionError("Atomic ID/OOD partition changed with support")
        for hop in (2, 3, 4):
            np.testing.assert_array_equal(low[0][f"train_{hop}"], high[0][f"train_{hop}"][:192])
    if not partial:
        execution = json.loads((artifacts / "execution.json").read_text())
        if (
            not execution["passed"]
            or len(execution["records"]) != expected_runs
            or {r["run"] for r in execution["records"]} != {r["run"] for r in records}
            or not all(r["passed"] for r in execution["records"])
        ):
            raise AssertionError("Completed execution matrix differs")
    record = {
        "utc": utc(),
        "phase": config["phase"],
        "scope": "Independent CPU completion verification",
        "config_sha256": file_hash(config_path),
        "verification_source_sha256": file_hash(__file__),
        "expected_runs": expected_runs,
        "verified_runs": len(records),
        "missing_runs": missing,
        "process_failures": failures,
        "complete": len(records) == expected_runs and not failures,
        "frozen_sources_config_design_exact": True,
        "archived_truth_data_common_pools_nested_support_exact": True,
        "paired_initial_and_same_compute_first_block_exact": True,
        "all_actual_exposure_nodes_and_rng_state_exact": True,
        "completed_updates": len(records) * 64000,
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in ("expected_runs", "verified_runs", "complete", "process_failures")
            }
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    verify(args.config, args.out, args.allow_partial)


if __name__ == "__main__":
    main()
