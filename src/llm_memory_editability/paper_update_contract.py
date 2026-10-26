"""Read-only compatibility checks for reusing the completed paper-gate arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BATCH = "paper-update-gate-multihop-v1"
BATCH_INITIAL_UPDATES = {BATCH: 0.9, "paper-update-gate-multihop-u95-v1": 0.95}
BASELINE_FILES = (
    "run.json",
    "complete.json",
    "audit.json",
    "grid.json",
    "grid-audit.json",
    "exposure.npz",
    "learning.json",
    "canonical-result.json",
    "cloud-audit.json",
    "checkpoint-0000000.pt",
    "checkpoint-0300000.pt",
)
CORE_SOURCE_FILES = (
    "scripts/run_loop_hop.py",
    "scripts/run_loop_hop_depth.py",
    "scripts/run_supervised_alignment.py",
    "scripts/run_dual_gate.py",
    "src/llm_memory_editability/dual_gate.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "src/llm_memory_editability/loop_hop_data.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/bios_model.py",
    "src/llm_memory_editability/experiment_tracking.py",
)
PANELS = ("atomic_id", "atomic_ood") + tuple(
    f"{kind}_{hop}" for hop in (2, 3, 4) for kind in ("train", "id", "ood")
)
RECIPE_FIELDS = (
    "data",
    "model",
    "training",
    "dropout_seed",
    "evaluation_nodes",
    "evaluation_batch_size",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    """Limit this comparison to two changes from P; compatibility is checked below."""
    _require(config.get("batch") in BATCH_INITIAL_UPDATES, "Unexpected experiment batch")
    initial_update = BATCH_INITIAL_UPDATES[config["batch"]]
    expected = {
        f"{mode}-update": {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "update_mode": mode,
            "initial_update": initial_update,
        }
        for mode in ("fixed", "token")
    }
    _require(
        config.get("arms") == expected, "Only the two matched fixed/token update gates are included"
    )
    _require(config.get("training", {}).get("steps") == 300000, "Budget must stay at 300000")
    baseline = config.get("baseline", {})
    _require(
        set(BASELINE_FILES) <= set(baseline.get("files", {})),
        "Missing pinned baseline artifacts",
    )
    _require(
        set(CORE_SOURCE_FILES) == set(baseline.get("source_files", {})),
        "The twelve unchanged paper/native source files must be pinned",
    )
    for field in ("root", "run", "config", "initial_model_sha256", "model_sha256"):
        _require(bool(baseline.get(field)), f"Missing baseline {field}")


def verify_input(config):
    """Verify the existing data; never generate, replace or copy it."""
    root = Path(config["input_root"])
    metadata = read_json(root / "complete.json")
    _require(read_json(root / "audit.json").get("passed") is True, "Input audit did not pass")
    for name, key in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        expected = config["input_origin"][key]
        _require(file_sha256(root / name) == expected == metadata[key], f"Input changed: {name}")
    for key, value in config["data"].items():
        _require(metadata.get(key) == value, f"Input recipe mismatch: {key}")
    _require(metadata["training_examples"] == 40000, "Expected the existing 40000-example world")
    expected_panels = {p: 1000 if p.startswith("train_") else 5000 for p in PANELS}
    _require(metadata["panels"] == expected_panels, "Evaluation panel coverage changed")
    for key, expected in {
        "entity_offset": 3,
        "relation_offset": 1003,
        "pad_token": 0,
        "end_marker": 2,
        "vocab_size": 1013,
        "sequence_length": 6,
        "atomic_examples": 10000,
    }.items():
        _require(metadata.get(key) == expected, f"Input token/shape contract changed: {key}")
    return {
        "passed": True,
        "input_root": str(root),
        "world_sha256": metadata["world_sha256"],
        "panels_sha256": metadata["panels_sha256"],
        "training_examples": metadata["training_examples"],
        "evaluation_queries": sum(metadata["panels"].values()),
    }


def verify_baseline(config, baseline_dir=None, source_root=ROOT, full_hashes=True):
    """Check P's recipe, immutable artifacts and prior audits without new evaluation.

    ``full_hashes=False`` omits only large checkpoint content hashes. The receipt
    records that omission and must not be presented as a full checkpoint audit.
    ``baseline_dir`` permits a read-only run mount; root/config remain explicit.
    """
    validate_config(config)
    baseline = config["baseline"]
    root = Path(baseline["root"])
    run = Path(baseline_dir) if baseline_dir is not None else Path(baseline["run"])
    prior_path = Path(baseline["config"])
    prior = read_json(prior_path)
    prior_sha = file_sha256(prior_path)
    _require(prior.get("batch") == "dual-gate-multihop-v1", "Expected completed P batch")
    _require(
        prior["arms"]["paper"]
        == {"architecture": "discoloop", "unique_layers": 1, "repeats": 4, "gate_variant": "paper"},
        "Reference is not the original one-layer four-round paper gate",
    )
    for field in RECIPE_FIELDS:
        _require(config[field] == prior[field], f"Baseline scientific recipe mismatch: {field}")
    for field in ("world_sha256", "panels_sha256"):
        _require(
            config["input_origin"][field] == prior["input_origin"][field],
            f"Baseline data identity mismatch: {field}",
        )
    freeze = read_json(root / "freeze.json")
    source_image = baseline.get("source_image", config["runtime"]["image"])
    _require(
        source_image == prior["runtime"]["image"] == freeze["image"],
        "Baseline original image differs",
    )
    runtime_changed = config["runtime"]["image"] != source_image
    if runtime_changed:
        amendment = config["runtime"].get("amendment", {})
        _require(
            bool(amendment.get("reason", "").strip())
            and amendment.get("original_image") == source_image
            and amendment.get("actual_image") == config["runtime"]["image"]
            and isinstance(amendment.get("requires_baseline_replay"), bool),
            "Replacement image needs an explicit amendment stating its replay requirement",
        )
    for relative, expected in baseline["source_files"].items():
        _require(
            file_sha256(Path(source_root) / relative)
            == file_sha256(root / "source" / relative)
            == freeze["files"][f"source/{relative}"]
            == expected,
            f"Unchanged source differs: {relative}",
        )
    checked_files = []
    for relative, expected in baseline["files"].items():
        path = run / relative
        _require(path.is_file(), f"Missing baseline artifact: {relative}")
        if full_hashes or path.suffix != ".pt":
            _require(file_sha256(path) == expected, f"Baseline artifact changed: {relative}")
            checked_files.append(relative)

    complete = read_json(run / "complete.json")
    recorded_run = read_json(run / "run.json")
    audit = read_json(run / "audit.json")
    grid = read_json(run / "grid.json")
    grid_audit = read_json(run / "grid-audit.json")
    _require(
        complete["state"] == "complete"
        and complete["step"] == complete["budget"] == 300000
        and complete["engineering_only"] is False,
        "Reference is not a completed scientific 300k endpoint",
    )
    for label, document in (("complete", complete), ("run", recorded_run), ("grid", grid)):
        _require(document["arm"] == "paper", f"Reference arm mismatch: {label}")
        _require(document["config_sha256"] == prior_sha, f"Reference config changed: {label}")
        _require(
            document["world_sha256"] == config["input_origin"]["world_sha256"],
            f"Reference world changed: {label}",
        )
        _require(
            document["initial_model_sha256"] == baseline["initial_model_sha256"],
            f"Reference initialization changed: {label}",
        )
    for document in (recorded_run, grid):
        _require(
            document["panels_sha256"] == config["input_origin"]["panels_sha256"],
            "Reference panels changed",
        )
    _require(
        complete["model_sha256"]
        == grid["model_sha256"]
        == grid_audit["model_sha256"]
        == baseline["model_sha256"],
        "Reference terminal model identity differs",
    )
    _require(
        complete["final_checkpoint_sha256"] == baseline["files"]["checkpoint-0300000.pt"],
        "Terminal checkpoint identity differs",
    )
    _require(
        audit.get("passed") is True
        and audit.get("exposure_match") is True
        and audit["step"] == 300000
        and audit["arm"] == "paper",
        "Independent endpoint reload did not pass",
    )
    _require(
        set(audit["panels"]) == set(PANELS)
        and all(p["passed"] and p["answer_and_eos_ids_exact"] for p in audit["panels"].values()),
        "Incomplete independent endpoint coverage",
    )
    expected_grid = {f"r{r}/{p}" for r in range(1, 5) for p in PANELS}
    _require(
        grid_audit.get("passed") is True
        and set(grid_audit["findings"]) == expected_grid
        and all(v is True for v in grid_audit["findings"].values())
        and grid_audit["grid_sha256"] == baseline["files"]["grid.json"],
        "Independent R1-4 grid reload did not pass",
    )
    _require(
        grid["step"] == 300000
        and grid["training_repeats"] == 4
        and grid["unique_layers"] == 1
        and set(grid["metrics"]) == {"1", "2", "3", "4"}
        and all(set(m) == set(PANELS) for m in grid["metrics"].values())
        and all(
            grid["metrics"]["4"][panel].get(key) == value
            for panel, metrics in complete["endpoint"]["metrics"].items()
            for key, value in metrics.items()
        ),
        "Reference inference grid/endpoint mismatch",
    )
    history = read_json(run / "learning.json")
    _require([row["step"] for row in history] == config["evaluation_nodes"], "Incomplete history")
    _require(history[-1] == complete["endpoint"], "History terminal record differs")
    canonical = read_json(run / "canonical-result.json")
    for filename in ("complete.json", "audit.json", "grid.json", "grid-audit.json"):
        _require(
            canonical[filename] == read_json(run / filename), f"Canonical mismatch: {filename}"
        )
    cloud = read_json(run / "cloud-audit.json")
    _require(
        cloud.get("passed") is True
        and cloud["canonical_sha256"] == baseline["files"]["canonical-result.json"],
        "Prior cloud artifact verification did not pass",
    )
    with np.load(run / "exposure.npz", allow_pickle=False) as saved:
        counts = saved["counts"]
        _require(
            counts.shape == (40000,)
            and np.issubdtype(counts.dtype, np.integer)
            and np.all(counts >= 0)
            and int(counts.sum()) == complete["endpoint"]["examples"],
            "Reference exposure identity/count mismatch",
        )
        exposure_examples = int(counts.sum())
    return {
        "passed": True,
        "baseline_run": str(run),
        "baseline_root": str(root),
        "config_sha256": prior_sha,
        "new_baseline_updates": 0,
        "initial_model_sha256": baseline["initial_model_sha256"],
        "model_sha256": baseline["model_sha256"],
        "checked_source_files": list(CORE_SOURCE_FILES),
        "checked_files": checked_files,
        "history_steps": [row["step"] for row in history],
        "exposure_examples": exposure_examples,
        "full_checkpoint_hashes": bool(full_hashes),
        "new_model_evaluations": 0,
        "source_image": source_image,
        "image": config["runtime"]["image"],
        "requires_baseline_runtime_replay": (
            config["runtime"]["amendment"]["requires_baseline_replay"] if runtime_changed else False
        ),
        "runtime_equivalence_verified": False,
    }
