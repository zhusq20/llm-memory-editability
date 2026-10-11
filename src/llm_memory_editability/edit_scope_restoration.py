"""Read-only restoration of the two executions of saved joint atomic edits.

The first digit selects pre/post-edit weights for the first iteration, the second
for the second iteration. No new edit, optimizer, case selection or fitting occurs.
All raw readouts use the completed development batch's native scorer and pools.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .edit_scope_development import (
    CODE_FILES as NATIVE_CODE_FILES,
)
from .edit_scope_development import (
    _verify_source,
    _write_prediction,
    make_edit_model,
    parameter_digest,
)
from .grok_depth import utc, write_json
from .path_overlap import World, load_model, sha256

MODES = ("00", "11", "10", "01")
ROLES = ("first", "second")
LEARNING_RATES = (1e-5, 1e-4)
CODE_FILES = (
    "src/llm_memory_editability/edit_scope_restoration.py",
    "scripts/run_edit_scope_restoration.py",
    *NATIVE_CODE_FILES,
    "src/llm_memory_editability/path_overlap.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "src/llm_memory_editability/grok_depth.py",
)


def _read(path):
    return json.loads(Path(path).read_text())


def _selected_names(parent):
    return {name for name, _ in parent.named_parameters() if name.startswith("transformer.h.")}


def _frozen_native_digest(parent):
    """Match the old joint arm's non-block hash without changing the parent."""
    digest = hashlib.sha256()
    selected = _selected_names(parent)
    for name, parameter in parent.named_parameters():
        if name not in selected:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_restoration_model(parent):
    """Eight independent blocks, same native function, all parameters frozen."""
    model = make_edit_model(parent, "round1")
    return model.eval().requires_grad_(False)


@torch.no_grad()
def apply_restoration(model, parent, selected, mode):
    """Copy the literal joint changes to the indicated execution, never optimize."""
    if mode not in MODES:
        raise ValueError(f"Unknown restoration mode {mode!r}")
    if parent.repeats != 2 or len(parent.transformer.h) != 4:
        raise ValueError("Expected the unchanged Loop 4x2 parent")
    if model.repeats != 1 or len(model.transformer.h) != 8:
        raise ValueError("Expected an eight-block restoration model")
    if set(selected) != _selected_names(parent):
        raise ValueError("Selected joint state has the wrong parameter names")
    original = dict(parent.named_parameters())
    for name, parameter in model.named_parameters():
        if not name.startswith("transformer.h."):
            if parameter.requires_grad:
                raise ValueError("Restoration must freeze every parameter")
            continue
        _, _, index, suffix = name.split(".", 3)
        index = int(index)
        source_name = f"transformer.h.{index % 4}.{suffix}"
        source = selected[source_name] if mode[index // 4] == "1" else original[source_name]
        if source.shape != parameter.shape or source.dtype != parameter.dtype:
            raise ValueError(f"Selected parameter shape/dtype mismatch: {source_name}")
        parameter.copy_(source.to(device=parameter.device))
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Restoration readouts require eval mode, no gradients")


def compare_predictions(reference, observed):
    """Exact native-array gate, including answer/EOS, ranks, margins and bridges."""
    if set(reference) != set(observed):
        return {"passed": False, "reason": "prediction array names differ", "mismatched": []}
    mismatched = [
        key
        for key in reference
        if reference[key].dtype != observed[key].dtype
        or not np.array_equal(reference[key], observed[key], equal_nan=False)
    ]
    return {"passed": not mismatched, "mismatched": mismatched, "comparison": "exact arrays"}


def paired_contrasts(metrics):
    """Within-case differences; cases, not queries, are the averaging unit."""
    if set(metrics) != set(MODES):
        raise ValueError("All four restoration states are required")
    result = {}
    for pool in metrics["00"]:
        result[pool] = {}
        for metric in metrics["00"][pool]:
            if metric == "n":
                if len({metrics[mode][pool][metric] for mode in MODES}) != 1:
                    raise ValueError("Restoration pools changed between modes")
                continue
            values = {mode: metrics[mode][pool][metric] for mode in MODES}
            if any(value is None for value in values.values()):
                continue
            a, b, c, d = (values[mode] for mode in MODES)
            result[pool][metric] = {
                "joint_minus_original": b - a,
                "first_only_minus_original": c - a,
                "second_only_minus_original": d - a,
                "restore_first_loss": b - d,
                "restore_second_loss": b - c,
                "interaction": b - c - d + a,
            }
    return result


def bridge_metrics(predictions, metadata):
    answer = predictions["first_iteration/D"]
    n = len(answer)
    return {
        "n": n,
        "old_bridge_accuracy": float(np.mean(answer == predictions["first_iteration/D_old_bridge"]))
        if n
        else None,
        "new_bridge_accuracy": float(np.mean(answer == predictions["first_iteration/D_new_bridge"]))
        if n
        else None,
        "E_old_bridge_accuracy": float(
            np.mean(
                predictions["first_iteration/E"]
                == predictions["E/old_answers"] + metadata["entity_offset"]
            )
        ),
        "E_new_bridge_accuracy": float(
            np.mean(
                predictions["first_iteration/E"]
                == predictions["E/new_answers"] + metadata["entity_offset"]
            )
        ),
    }


def _source_manifest(source_batch, role, rate):
    """Only consume the complete unchanged bank, never recompute selection."""
    source_batch = Path(source_batch).resolve()
    arm = f"{role}-joint"
    run = source_batch / "runs" / arm
    config_path = source_batch / "frozen/config.json"
    config, summary, bank = (
        _read(config_path),
        _read(run / "summary.json"),
        _read(run / "case-bank.json"),
    )
    complete, audit = _read(run / "complete.json"), _read(run / "audit.json")
    checks = (
        complete.get("complete"),
        complete.get("independent_audit_complete"),
        audit.get("passed"),
        audit.get("checked") == audit.get("expected") == 64,
        sha256(run / "summary.json") == complete["summary_sha256"],
        sha256(run / "audit.json") == complete["audit_sha256"],
        sha256(config_path) == summary["config_sha256"],
        sha256(run / "case-bank.json") == summary["case_bank_sha256"],
        summary["arm"] == arm,
        bank["role"] == role,
        all(case["role"] == role for case in bank["cases"]),
        {kind: sum(case["group"] == kind for case in bank["cases"]) for kind in ("id", "ood")}
        == {"id": 16, "ood": 16},
        summary["case_count"] == 32 == len(bank["cases"]),
        summary["settings"]["learning_rates"] == list(LEARNING_RATES),
        summary["settings"]["save_endpoints"] == [100],
    )
    if not all(checks):
        raise ValueError("Source joint arm is incomplete or its frozen provenance changed")
    records = [
        record
        for record in summary["records"]
        if record["step"] == 100 and record["learning_rate"] == rate
    ]
    cases = {case["case_id"]: case for case in bank["cases"]}
    if len(records) != 32 or {r["case_id"] for r in records} != set(cases):
        raise ValueError("Shard must contain exactly one saved endpoint for every source case")
    project = Path(__file__).resolve().parents[2]
    if any(sha256(project / name) != value for name, value in summary["code_sha256"].items()):
        raise ValueError("Native scoring source differs from the completed joint arm")
    inputs, endpoints = {}, []
    for path in (
        config_path,
        run / "summary.json",
        run / "case-bank.json",
        run / "complete.json",
        run / "audit.json",
    ):
        inputs[str(path)] = sha256(path)
    for record in records:
        state = run / record["selected_state_path"]
        prediction = run / record["prediction_path"]
        baseline = prediction.with_name("predictions-000.npz")
        if (
            sha256(state) != record["selected_state_sha256"]
            or sha256(prediction) != record["prediction_sha256"]
        ):
            raise ValueError("Saved joint state or raw native prediction changed")
        paths = {"selected": state, "11": prediction, "00": baseline}
        endpoints.append(
            {
                "case": cases[record["case_id"]],
                "record": record,
                "paths": {key: str(path) for key, path in paths.items()},
                "hashes": {key: sha256(path) for key, path in paths.items()},
            }
        )
    return config, summary, endpoints, inputs


def _load_selected(endpoint, summary, frozen_hash):
    path = endpoint["paths"]["selected"]
    if sha256(path) != endpoint["hashes"]["selected"]:
        raise ValueError("Source saved selected state changed")
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "scope": "joint",
        "case_id": endpoint["case"]["case_id"],
        "step": 100,
        "learning_rate": endpoint["record"]["learning_rate"],
        "case_bank_sha256": summary["case_bank_sha256"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "config_sha256": summary["config_sha256"],
        "world_sha256": summary["world_sha256"],
        "frozen_parameter_sha256": frozen_hash,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved selected state belongs to a different endpoint or parent")
    return state["selected"]


def _raw(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def summarize_records(records):
    """Full bank plus original ID/OOD groups, equal weight per case."""
    output = {}
    for group in ("all", "id", "ood"):
        selected = [row for row in records if group == "all" or row["group"] == group]
        entry = {
            "case_count": len(selected),
            "modes": {},
            "paired_contrasts": {},
            "pool_coverage": {},
        }
        if not selected:
            output[group] = entry
            continue
        for mode in MODES:
            entry["modes"][mode] = {}
            for pool, metrics in selected[0]["metrics"][mode].items():
                entry["modes"][mode][pool] = {
                    key: float(
                        np.mean(
                            [
                                r["metrics"][mode][pool][key]
                                for r in selected
                                if r["metrics"][mode][pool][key] is not None
                            ]
                        )
                    )
                    if any(r["metrics"][mode][pool][key] is not None for r in selected)
                    else None
                    for key in metrics
                }
        for pool in selected[0]["metrics"]["00"]:
            counts = [r["metrics"]["00"][pool]["n"] for r in selected]
            entry["pool_coverage"][pool] = {
                "cases_with_queries": sum(n > 0 for n in counts),
                "query_count": sum(counts),
            }
            metric_names = {key for r in selected for key in r["contrasts"][pool]}
            entry["paired_contrasts"][pool] = {}
            for key in sorted(metric_names):
                contributing = [
                    r["contrasts"][pool][key] for r in selected if key in r["contrasts"][pool]
                ]
                entry["paired_contrasts"][pool][key] = {
                    contrast: float(np.mean([row[contrast] for row in contributing]))
                    for contrast in contributing[0]
                }
        output[group] = entry
    return output


def run_shard(source_batch, role, learning_rate, out, device="cuda:0", replay_of=None, limit=None):
    """Four fixed role/rate shards; all native gates precede all hybrid readouts."""
    if role not in ROLES or learning_rate not in LEARNING_RATES:
        raise ValueError("Only the two unchanged roles and learning rates are supported")
    if limit not in (None, 1):
        raise ValueError("Only --limit 1 is supported for smoke; full shards contain 32 cases")
    out, device = Path(out), torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    owned = (
        "run.json",
        "provenance.json",
        "summary.json",
        "complete.json",
        "failure.json",
        "cases",
        "native-gates.json",
    )
    if any((out / name).exists() for name in owned):
        raise RuntimeError("Output contains a scientific attempt; retain the previous attempt")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    request = {
        "source_batch": str(Path(source_batch).resolve()),
        "role": role,
        "learning_rate": learning_rate,
        "limit": limit,
    }
    write_json(
        out / "run.json",
        {**request, "started_utc": utc(), "pid": os.getpid(), "optimization_updates": 0},
    )
    try:
        config, source, endpoints, input_hashes = _source_manifest(
            source_batch, role, learning_rate
        )
        if limit is not None:
            endpoints = endpoints[:limit]
        expected = len(endpoints)
        world = World(Path(config["source"]["root"]) / "data")
        parent, spec, checkpoint = load_model(
            config["source"]["root"],
            config["source"]["run"],
            config["source"]["step"],
            world.metadata,
            device,
        )
        _verify_source(config, world, spec, checkpoint)
        if sha256(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint differs from the saved edit")
        if sha256(Path(config["source"]["root"]) / "data/world.npz") != source["world_sha256"]:
            raise ValueError("World differs from the saved edit")
        model, frozen_hash = make_restoration_model(parent), _frozen_native_digest(parent)
        parent_hash = parameter_digest(parent)
        precision, records = source["settings"]["precision"], []
        provenance = {
            "request": request,
            "source_input_sha256": input_hashes,
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_world_sha256": source["world_sha256"],
            "source_case_bank_sha256": source["case_bank_sha256"],
            "source_endpoints": endpoints,
            "code_sha256": {
                name: sha256(Path(__file__).resolve().parents[2] / name) for name in CODE_FILES
            },
            "optimization_updates": 0,
            "single_world_single_initialization": True,
            "posthoc_mechanism_readout": True,
            "precision": precision,
            "smoke": limit is not None,
            "scoring": "unchanged native full-vocabulary answer/EOS, ranks and margins",
        }
        write_json(out / "provenance.json", provenance)
        # These gates cover every endpoint in the shard before any hybrid outcome.
        for endpoint in endpoints:
            case = endpoint["case"]
            selected = _load_selected(endpoint, source, frozen_hash)
            folder = out / "cases" / case["case_id"]
            folder.mkdir(parents=True)
            record = {
                "case_id": case["case_id"],
                "group": case["group"],
                "learning_rate": learning_rate,
                "source_hashes": endpoint["hashes"],
                "metrics": {},
                "predictions": {},
                "gates": {},
            }
            for mode in ("00", "11"):
                apply_restoration(model, parent, selected, mode)
                path = folder / f"predictions-{mode}.npz"
                metrics = _write_prediction(path, model, world, case, device, precision)
                fresh = _raw(path)
                metrics["bridge"] = bridge_metrics(fresh, world.metadata)
                source_path = endpoint["paths"][mode]
                if sha256(source_path) != endpoint["hashes"][mode]:
                    raise ValueError("Source native reference changed during replay")
                gate = compare_predictions(_raw(source_path), fresh)
                record["gates"][mode] = gate
                write_json(folder / f"gate-{mode}.json", gate)
                if not gate["passed"]:
                    raise RuntimeError(f"Exact native replay gate failed: {case['case_id']}/{mode}")
                record["metrics"][mode] = metrics
                record["predictions"][mode] = {
                    "path": str(path.relative_to(out)),
                    "sha256": sha256(path),
                }
            records.append(record)
            write_json(
                out / "progress.json",
                {
                    "phase": "native-gates",
                    "completed_cases": len(records),
                    "expected_cases": expected,
                    "optimization_updates": 0,
                    "updated_utc": utc(),
                },
            )
        write_json(
            out / "native-gates.json",
            {
                "passed": True,
                "checked": 2 * expected,
                "exact_arrays": True,
                "before_any_hybrid": True,
            },
        )
        for endpoint, record in zip(endpoints, records, strict=True):
            selected = _load_selected(endpoint, source, frozen_hash)
            folder = out / "cases" / record["case_id"]
            for mode in ("10", "01"):
                apply_restoration(model, parent, selected, mode)
                path = folder / f"predictions-{mode}.npz"
                metrics = _write_prediction(path, model, world, endpoint["case"], device, precision)
                metrics["bridge"] = bridge_metrics(_raw(path), world.metadata)
                record["metrics"][mode] = metrics
                record["predictions"][mode] = {
                    "path": str(path.relative_to(out)),
                    "sha256": sha256(path),
                }
            record["contrasts"] = paired_contrasts(record["metrics"])
            write_json(folder / "readout.json", record)
            write_json(
                out / "progress.json",
                {
                    "phase": "hybrid-readouts",
                    "completed_cases": sum("contrasts" in r for r in records),
                    "expected_cases": expected,
                    "optimization_updates": 0,
                    "updated_utc": utc(),
                },
            )
        if parameter_digest(parent) != parent_hash:
            raise RuntimeError("Parent weights changed")
        summary = {
            "request": request,
            "provenance_sha256": sha256(out / "provenance.json"),
            "case_count": expected,
            "endpoint_count": expected,
            "mode_readouts": 4 * expected,
            "evaluation_answer_eos_query_exposures": sum(
                metrics["n"]
                for row in records
                for pools in row["metrics"].values()
                for pool, metrics in pools.items()
                if pool != "bridge"
            ),
            "evaluation_first_iteration_query_exposures": sum(
                1 + pools["bridge"]["n"] for row in records for pools in row["metrics"].values()
            ),
            "optimization_updates": 0,
            "smoke": limit is not None,
            "records": records,
            "aggregates": summarize_records(records),
            "seconds": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "completed_utc": utc(),
        }
        write_json(out / "summary.json", summary)
        if replay_of is not None:
            original = Path(replay_of)
            prior = _read(original / "summary.json")
            if prior["request"] != request or prior["case_count"] != len(records):
                raise ValueError("Audit request or case coverage differs")
            checks = []
            for previous, fresh in zip(prior["records"], records, strict=True):
                if (
                    previous["case_id"] != fresh["case_id"]
                    or previous["metrics"] != fresh["metrics"]
                    or previous["contrasts"] != fresh["contrasts"]
                ):
                    raise RuntimeError("Independent replay changed case metrics or contrasts")
                for mode in MODES:
                    path = original / previous["predictions"][mode]["path"]
                    if sha256(path) != previous["predictions"][mode]["sha256"]:
                        raise ValueError("Recorded restoration prediction changed")
                    check = compare_predictions(
                        _raw(path), _raw(out / fresh["predictions"][mode]["path"])
                    )
                    checks.append({"case_id": fresh["case_id"], "mode": mode, **check})
            audit = {
                "passed": len(checks) == 4 * expected and all(c["passed"] for c in checks),
                "checked": len(checks),
                "checks": checks,
                "optimization_updates": 0,
                "source_summary_sha256": sha256(original / "summary.json"),
                "completed_utc": utc(),
            }
            write_json(out / "audit.json", audit)
            if not audit["passed"]:
                raise RuntimeError("Independent restoration replay failed")
        write_json(
            out / "complete.json",
            {
                "complete": True,
                "passed": True,
                "summary_sha256": sha256(out / "summary.json"),
                "native_gates_passed": True,
                "independent_replay": replay_of is not None,
                "optimization_updates": 0,
                "completed_utc": utc(),
            },
        )
        return summary
    except Exception as error:
        write_json(
            out / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "optimization_updates": 0,
                "utc": utc(),
            },
        )
        raise


def audit_shard(out, audit_out, device="cuda:0"):
    """Replay all four modes into a new directory, without touching source files."""
    out = Path(out)
    complete, summary, provenance = (
        _read(out / "complete.json"),
        _read(out / "summary.json"),
        _read(out / "provenance.json"),
    )
    if (
        not complete.get("complete")
        or sha256(out / "summary.json") != complete["summary_sha256"]
        or sha256(out / "provenance.json") != summary["provenance_sha256"]
    ):
        raise ValueError("Restoration run is incomplete or its provenance changed")
    project = Path(__file__).resolve().parents[2]
    if any(sha256(project / name) != value for name, value in provenance["code_sha256"].items()):
        raise ValueError("Restoration replay source code changed")
    return run_shard(**summary["request"], out=audit_out, device=device, replay_of=out)
