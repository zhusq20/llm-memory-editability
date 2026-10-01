"""Independent CPU audit of immutable confirmation inputs and step-zero weights.

Run from the repository root. No project training/data builder is imported, no
CUDA API is called, and training streams/exposure files are outside this audit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
ARTIFACT = ROOT / "docs/development-artifacts/grok-usage-v1"
CONFIG_PATH = ROOT / "configs/grok-usage-confirmation-v1.json"
LOCK_PATH = ARTIFACT / "confirmation-lock.json"
OUTPUT = ARTIFACT / "confirmation-input-audit.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def array_record(value):
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        raw_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    )


def tensor_record(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        elements=value.numel(),
    )


def main():
    cfg, lock = read(CONFIG_PATH), read(LOCK_PATH)
    checks, runs, world_runs = [], [], defaultdict(list)

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    expected_runs = sorted(cfg["runs"])
    output_root = ROOT / cfg["output_root"] / cfg["base"]["phase"]
    observed_runs = sorted(p.name for p in output_root.iterdir() if p.is_dir())
    check(
        "confirmation_matrix",
        len(expected_runs) == 12 and expected_runs == observed_runs,
        expected=expected_runs,
        observed=observed_runs,
    )
    check("lock_expected_runs", sorted(lock["expected_runs"]) == expected_runs)
    check("configuration_path", lock["config"] == relative(CONFIG_PATH))
    check("configuration_hash", sha(CONFIG_PATH) == lock["files"][relative(CONFIG_PATH)])
    frozen = datetime.fromisoformat(lock["frozen_utc"])
    development = []
    for arm in ("A", "B", "repA", "repB"):
        path = ROOT / cfg["output_root"] / "development" / f"development-{arm}/complete.json"
        item = read(path)
        finished = datetime.fromisoformat(item["finished_utc"])
        check(
            f"development_finished_before_lock/{arm}",
            finished < frozen,
            finished_utc=item["finished_utc"],
            frozen_utc=lock["frozen_utc"],
        )
        development.append(
            dict(arm=arm, path=relative(path), sha256=sha(path), finished_utc=item["finished_utc"])
        )
    required_arrays = [
        "atomic",
        "eval_atomic",
        "eval_atomic_BG",
        "eval_atomic_A",
        "eval_atomic_B",
        "eval_test_composite",
    ] + [f"eval_test_{a}_{b}" for a in ("BG", "A", "B") for b in ("BG", "A", "B")]
    reference_model, reference_fingerprints, reference_run = None, None, None
    for run_id in expected_runs:
        folder = output_root / run_id
        expected_spec = {**cfg["base"], **cfg["runs"][run_id]}
        meta_path = folder / "metadata.json"
        meta = read(meta_path)
        check(f"metadata_spec/{run_id}", meta["spec"] == expected_spec)
        check(f"metadata_source_lock/{run_id}", meta["source_lock"] == lock)
        check(
            f"started_after_lock/{run_id}",
            datetime.fromisoformat(meta["started_utc"]) > frozen,
            started_utc=meta["started_utc"],
            frozen_utc=lock["frozen_utc"],
        )
        archived = {}
        source_root = folder / "source"
        source_paths = sorted(
            relative.relative_to(source_root).as_posix()
            for relative in source_root.rglob("*")
            if relative.is_file()
        )
        check(
            f"source_snapshot_file_set/{run_id}",
            source_paths == sorted(lock["files"]),
            expected_count=len(lock["files"]),
            observed_count=len(source_paths),
        )
        for filename, expected_hash in lock["files"].items():
            path = source_root / filename
            actual_hash = sha(path)
            check(f"source_snapshot_hash/{run_id}/{filename}", actual_hash == expected_hash)
            archived[filename] = actual_hash
        world_path = folder / "world.npz"
        with np.load(world_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in required_arrays}
        check(
            f"atomic_evaluation_identity/{run_id}",
            np.array_equal(arrays["atomic"], arrays["eval_atomic"]),
        )
        atomic_parts = np.concatenate([arrays[f"eval_atomic_{part}"] for part in ("BG", "A", "B")])
        test_parts = np.concatenate(
            [arrays[f"eval_test_{a}_{b}"] for a in ("BG", "A", "B") for b in ("BG", "A", "B")]
        )
        check(
            f"atomic_partition/{run_id}",
            Counter(map(tuple, atomic_parts)) == Counter(map(tuple, arrays["atomic"])),
        )
        check(
            f"nine_grid_partition/{run_id}",
            Counter(map(tuple, test_parts)) == Counter(map(tuple, arrays["eval_test_composite"])),
        )
        weights_path = folder / "weights-0000000.pt"
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        check(f"zero_step/{run_id}", state["step"] == 0)
        check(f"zero_step_spec/{run_id}", state["spec"] == expected_spec)
        model = state["model"]
        fingerprints = {name: tensor_record(value) for name, value in model.items()}
        if reference_model is None:
            reference_model, reference_fingerprints, reference_run = model, fingerprints, run_id
        check(
            f"zero_model_tensor_keys/{run_id}",
            set(model) == set(reference_model),
            reference_run=reference_run,
        )
        for name, value in model.items():
            numeric_equal = name in reference_model and torch.equal(value, reference_model[name])
            bitwise_equal = fingerprints.get(name) == reference_fingerprints.get(name)
            check(
                f"zero_model_tensor_identity/{run_id}/{name}",
                numeric_equal and bitwise_equal,
                numeric_equal=numeric_equal,
                bitwise_equal=bitwise_equal,
            )
        model_hash = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
        record = dict(
            run_id=run_id,
            world_seed=expected_spec["world_seed"],
            arm=expected_spec["arm"],
            initialization=expected_spec["initialization"],
            started_utc=meta["started_utc"],
            metadata=dict(path=relative(meta_path), sha256=sha(meta_path)),
            source_snapshot_sha256=archived,
            world=dict(
                path=relative(world_path),
                sha256=sha(world_path),
                arrays={k: array_record(v) for k, v in arrays.items()},
            ),
            zero_weights=dict(
                path=relative(weights_path),
                file_sha256=sha(weights_path),
                canonical_model_sha256=model_hash,
                tensors=fingerprints,
            ),
        )
        runs.append(record)
        world_runs[expected_spec["world_seed"]].append((run_id, expected_spec["arm"], arrays))
    cross_arm = []
    for world_seed, members in sorted(world_runs.items()):
        check(
            f"four_arms/{world_seed}",
            sorted(arm for _, arm, _ in members) == ["A", "B", "repA", "repB"],
        )
        baseline_id, _, baseline = members[0]
        for run_id, arm, arrays in members[1:]:
            for name in required_arrays:
                same = arrays[name].dtype == baseline[name].dtype and np.array_equal(
                    arrays[name], baseline[name]
                )
                check(f"cross_arm_array_identity/{world_seed}/{arm}/{name}", same)
                cross_arm.append(
                    dict(
                        world_seed=world_seed,
                        reference_run=baseline_id,
                        compared_run=run_id,
                        array=name,
                        identical=bool(same),
                    )
                )
    failures = [c for c in checks if not c["passed"]]
    result = dict(
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        status="pass" if not failures else "fail",
        method={
            "scope": (
                "Immutable metadata, source snapshots, world arrays, "
                "zero-step tensors and chronology"
            ),
            "implementation": (
                "Independent Python/NumPy/PyTorch CPU loader; no project training/data "
                "construction imports and no CUDA calls"
            ),
            "model_comparison": (
                "Every tensor key, shape, dtype, value and raw-byte SHA256 matched to the first "
                "confirmation run across all worlds"
            ),
            "data_comparison": (
                "Exact shape/dtype/ordered-array comparison for atomic, full test, nine test cells "
                "and three atomic cohorts within each world"
            ),
            "chronology": (
                "Recorded ISO UTC: all four development complete.finished_utc < confirmation "
                "frozen_utc < every metadata.started_utc"
            ),
            "exclusions": "No training-exposure, endpoint scoring, optimizer or later-weight audit",
        },
        audit_source=dict(path=relative(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        config=dict(path=relative(CONFIG_PATH), sha256=sha(CONFIG_PATH)),
        lock=dict(path=relative(LOCK_PATH), sha256=sha(LOCK_PATH), frozen_utc=lock["frozen_utc"]),
        counts=dict(
            runs=len(runs),
            worlds=len(world_runs),
            locked_files_per_run=len(lock["files"]),
            source_snapshot_hashes=sum(len(r["source_snapshot_sha256"]) for r in runs),
            shared_arrays_per_world=len(required_arrays),
            cross_arm_array_comparisons=len(cross_arm),
            tensors_per_model=len(reference_model),
            zero_tensor_checks=len(reference_model) * len(runs),
            nonreference_zero_tensor_comparisons=len(reference_model) * (len(runs) - 1),
            total_checks=len(checks),
            failed_checks=len(failures),
        ),
        development_completion_records=development,
        time_bounds=dict(
            latest_development_finished=max(r["finished_utc"] for r in development),
            frozen_utc=lock["frozen_utc"],
            earliest_confirmation_started=min(r["started_utc"] for r in runs),
            latest_confirmation_started=max(r["started_utc"] for r in runs),
        ),
        runs=runs,
        cross_arm_array_comparisons=cross_arm,
        checks=checks,
        failures=failures,
    )
    temporary = OUTPUT.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(OUTPUT)
    print(
        json.dumps(
            dict(
                status=result["status"], counts=result["counts"], time_bounds=result["time_bounds"]
            ),
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
