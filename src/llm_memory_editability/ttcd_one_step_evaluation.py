"""One-step-trained endpoints: single, continuous and independent-delta writes.

The same immutable backbone and facts serve every writing method and both hop
roles. New two-hop answers are readouts only. A separate audit rewrites every
state from the audited endpoint; it never relies on a saved predecessor A.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import ttcd_continual_eval as original
from . import ttcd_editing as fast_io
from . import ttcd_merge_hops as merge
from . import ttcd_one_step as one
from .grok_depth import write_json
from .path_overlap import World, sha256
from .ttcd_merge_hop_panel import build_panel

BATCH = "ttcd-one-step-training-v1"
METHODS = ("single", "continuous", "independent_merge")
ROLES = ("first", "second")


def validate_settings(task, settings):
    if (
        settings["batch"] != BATCH
        or settings["writer_steps"] != 1
        or settings["training_writer_steps"] != 1
        or settings["methods"] != list(METHODS)
        or settings["roles"] != list(ROLES)
        or task["training_method"] != "stream"
        or task["architecture_name"] not in ("transformer", "loop", "discoloop")
        or task["training_config"]["writer"]["steps"] != 1
    ):
        raise ValueError("Use one one-step continuous-trained endpoint per architecture")


def prepare(task, settings, out):
    """Graph-only preparation can precede endpoint training and auditing."""
    validate_settings(task, settings)
    config = task["training_config"]
    path = Path(config["continual"]["panel_path"])
    if sha256(path) != settings["panel_sha256"]:
        raise ValueError("Original fact/replacement panel changed")
    world = World(config["data_dir"])
    parent = original._panel(config, world)
    one.validate_panel(parent, world)
    panel = build_panel(parent, world, settings, config["phi"])
    write_json(Path(out) / "panel.json", panel)
    return {
        "passed": True,
        "panel_sha256": sha256(Path(out) / "panel.json"),
        "first_hop_coverage": panel["first_hop_coverage"],
        "second_hop_queries": sum(len(f["D"]) for f in panel["facts"]),
        "model_queries": 0,
        "new_base_parameter_updates": 0,
    }


def _engine(task):
    from . import ttcd_one_step_training as training

    return training.build_engine(task)


def source_context(task, settings, out, device):
    """The adapter binds new weights/audit and unchanged common-parent hashes."""
    validate_settings(task, settings)
    config = task["training_config"]
    reader, world = _engine(task).load_reader(config, "stream", device)
    base_panel = original._panel(config, world)
    one.validate_panel(base_panel, world)
    if sha256(config["continual"]["panel_path"]) != settings["panel_sha256"]:
        raise ValueError("Original formal panel bytes changed")
    panel = json.loads((Path(out) / "panel.json").read_text())
    if panel != build_panel(base_panel, world, settings, config["phi"]):
        raise ValueError("Prepared hop panel changed")
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("All backbone parameters must remain frozen during evaluation")
    return reader, world, config, panel


def identity(task, settings, out):
    config = task["training_config"]
    panel = json.loads((Path(out) / "panel.json").read_text())
    return {
        "schema": BATCH,
        "architecture": task["architecture_name"],
        "settings": settings,
        "training_writer_steps": 1,
        "evaluation_writer_steps": 1,
        "training_endpoint": _engine(task).evaluation_identity(config, "stream", panel),
        "panel_sha256": sha256(Path(out) / "panel.json"),
        "overlay_sha256": sha256(__file__),
        "new_base_parameter_updates": 0,
    }


def _one_write(reader, fast, config, world, fact, device):
    if config["writer"]["steps"] != 1:
        raise ValueError("Each fact must use exactly one writer step")
    result, history = original.write_fact(reader, fast, config, world, fact, device)
    if len(history["updates"]) != 1:
        raise AssertionError("A fact did not produce exactly one committed update")
    return result, history


def preflight(task, settings, out, device):
    """Engineering writes use development atoms; no formal readouts or selection."""
    from .grokking_reproduction import encode_rows

    reader, world, config, _ = source_context(task, settings, out, device)
    before = fast_io.tensor_digest(reader.backbone.state_dict())
    arrays = encode_rows(world.atoms[:8], world.metadata)
    tokens, positions, _ = (torch.as_tensor(a, device=device) for a in arrays)
    with torch.no_grad():
        torch.testing.assert_close(
            reader.backbone(tokens, positions),
            reader(tokens, positions, fast=reader.empty_fast()),
            rtol=0,
            atol=0,
        )
    cases = [c for c in original.case_bank(config) if c["split"] == "calibration"]
    increments = []
    for group in ("id", "ood"):
        case = next(c for c in cases if c["group"] == group)
        fast, _ = _one_write(reader, reader.empty_fast(), config, world, case, device)
        increments.append(fast)
    digests = [fast_io.tensor_digest(delta) for delta in increments]
    merged = reader.empty_fast()
    for delta in increments:
        merged = merge.add_increment(merged, delta)
    for key in merged:
        torch.testing.assert_close(
            merged[key], increments[0][key] + increments[1][key], rtol=0, atol=0
        )
    path = Path(out) / "development-merged.pt"
    fast_io.save_fast(path, reader, merged, {"development_only": True})
    loaded, _ = fast_io.load_fast(path, reader)
    if fast_io.tensor_digest(loaded) != fast_io.tensor_digest(merged):
        raise AssertionError("Development merged state reload differs")
    if digests != [fast_io.tensor_digest(delta) for delta in increments]:
        raise AssertionError("Merging changed an independent donor")
    if before != fast_io.tensor_digest(reader.backbone.state_dict()):
        raise AssertionError("Development writes changed backbone parameters")
    return {
        "passed": True,
        "native_A0_exact": True,
        "development_one_step_writes": 2,
        "donors_and_backbone_unchanged": True,
        "merged_save_reload_exact": True,
        "formal_model_queries": 0,
        "training_writer_steps": 1,
        "evaluation_writer_steps": 1,
        "new_base_parameter_updates": 0,
    }


def calibrate(task, settings, out, device):
    """Report calibration accuracy; stop on numerical/identity failures only."""
    reader, world, config, _ = source_context(task, settings, out, device)
    before = fast_io.tensor_digest(reader.backbone.state_dict())
    expected_identity = identity(task, settings, out)
    result = original.calibrate(
        config,
        "stream",
        Path(out) / "development-calibration",
        device,
        reader_loader=lambda *_: (reader, world),
        identity_factory=lambda *_: expected_identity,
    )
    if before != fast_io.tensor_digest(reader.backbone.state_dict()):
        raise AssertionError("Calibration changed the backbone")
    result.update(
        accuracy_gate_passed=result["passed"],
        passed=True,
        accuracy_gate_advisory_only=True,
        model_or_fact_selection=False,
        training_writer_steps=1,
        evaluation_writer_steps=1,
        new_base_parameter_updates=0,
    )
    write_json(Path(out) / "calibration.json", result)
    return result


def state_plan(panel):
    n = len(panel["facts"])
    plans = {
        "baseline": {
            "kind": "baseline",
            "indices": list(range(n)),
            "t": 0,
            "rotation": None,
        }
    }
    for index in range(n):
        plans[f"single/f{index:03d}"] = {
            "kind": "single",
            "indices": [index],
            "t": 1,
            "rotation": None,
        }
    for method in ("continuous", "independent_merge"):
        for rotation, order in enumerate(panel["orders"]):
            for t in range(1, n + 1):
                plans[f"{method}/r{rotation:03d}/t{t:03d}"] = {
                    "kind": method,
                    "indices": order[:t],
                    "t": t,
                    "rotation": rotation,
                }
    return plans


def _state_transitions(reader, config, world, panel, plans, device):
    """Rebuild from A0, without reading saved states or prediction artifacts."""
    donors, donor_digests = [], []
    fast, previous = None, None
    for key, spec in plans.items():
        kind = spec["kind"]
        if kind == "baseline":
            fast, history = reader.empty_fast(), None
        elif kind == "single":
            fast, history = _one_write(
                reader,
                reader.empty_fast(),
                config,
                world,
                panel["facts"][spec["indices"][0]],
                device,
            )
            donors.append(fast)
            donor_digests.append(fast_io.tensor_digest(fast))
        else:
            branch = (kind, spec["rotation"])
            if branch != previous:
                fast, previous = reader.empty_fast(), branch
            index = spec["indices"][-1]
            if kind == "continuous":
                fast, history = _one_write(
                    reader, fast, config, world, panel["facts"][index], device
                )
            elif kind == "independent_merge":
                fast = merge.add_increment(fast, donors[index])
                history = {
                    "independent_donors": [f"single/f{i:03d}" for i in spec["indices"]],
                    "new_writer_updates": 0,
                    "rule": "sum immutable A_i from A0",
                }
            else:
                raise ValueError("Unknown writing method")
        yield key, spec, fast, history
    if donor_digests != [fast_io.tensor_digest(delta) for delta in donors]:
        raise AssertionError("A later write mutated an independent donor")


def _read(reader, fast, world, panel, spec, device):
    base = original.read_state(
        reader,
        fast,
        world,
        panel,
        spec["indices"],
        [panel["facts"][i]["new_tail"] for i in spec["indices"]],
        device,
    )
    first = merge.read_first(reader, fast, world, panel, spec["indices"], device)
    return base, first


def _require_calibration(task, settings, out):
    receipt = json.loads((Path(out) / "calibration.json").read_text())
    expected = identity(task, settings, out)
    if receipt.get("passed") is not True or receipt["identity"] != expected:
        raise ValueError("Numerical development calibration or input identity changed")
    return expected


def _analyze(panel, records, metadata):
    result, paired = merge.analyze(panel, records, metadata)
    result["schema"] = BATCH
    result["training_writer_steps"] = result["evaluation_writer_steps"] = 1
    return result, paired


def evaluate(task, settings, out, device, log=None):
    started = time.monotonic()
    out = Path(out)
    reader, world, config, panel = source_context(task, settings, out, device)
    expected_identity = _require_calibration(task, settings, out)
    before = fast_io.tensor_digest(reader.backbone.state_dict())
    plans, records = state_plan(panel), []
    for key, spec, fast, history in _state_transitions(reader, config, world, panel, plans, device):
        directory = out / "states" / key
        directory.mkdir(parents=True, exist_ok=False)
        fast_io.save_fast(directory / "fast.pt", reader, fast, spec)
        base, first = _read(reader, fast, world, panel, spec, device)
        record = {
            "key": key,
            "spec": spec,
            "write_history": history,
            "readout": base,
            "first": first,
            "fast_digest": fast_io.tensor_digest(fast),
        }
        write_json(directory / "record.json", record)
        records.append(record)
        write_json(
            out / "evaluation-progress.json",
            {"states": len(records), "expected": len(plans), "last": key},
        )
        if log:
            log(len(records), record)
    if before != fast_io.tensor_digest(reader.backbone.state_dict()):
        raise AssertionError("Formal evaluation changed the backbone")
    summary, paired = _analyze(panel, records, world.metadata)
    summary.update(
        identity=expected_identity,
        state_keys=list(plans),
        states=len(plans),
        independent_donors=len(panel["facts"]),
        continuous_transitions=len(panel["orders"]) * len(panel["facts"]),
        merged_states=len(panel["orders"]) * len(panel["facts"]),
        new_base_parameter_updates=0,
        wall_seconds=time.monotonic() - started,
        compute=dict(reader.compute),
        base_parameters=sum(p.numel() for p in reader.parameters()),
        fast_elements=sum(v.numel() for v in reader.empty_fast().values()),
        formal_updated_two_hop_supervision=False,
    )
    write_json(out / "summary.json", summary)
    write_json(out / "paired-queries.json", paired)
    return summary


def audit(task, settings, out, device):
    """Separate-process rewrite and full token/score replay of every state."""
    out = Path(out)
    reader, world, config, panel = source_context(task, settings, out, device)
    expected_identity = _require_calibration(task, settings, out)
    before = fast_io.tensor_digest(reader.backbone.state_dict())
    plans = state_plan(panel)
    summary = json.loads((out / "summary.json").read_text())
    if (
        summary["identity"] != expected_identity
        or summary["state_keys"] != list(plans)
        or summary["states"] != len(plans)
    ):
        raise ValueError("Evaluation identity or state matrix changed")
    paths = list((out / "states").rglob("record.json"))
    keys = {str(p.parent.relative_to(out / "states")) for p in paths}
    if keys != set(plans) or len(paths) != len(plans):
        raise ValueError("Incomplete or extra evaluation states")
    records, checked_queries, writes = [], 0, 0
    for key, spec, rebuilt, history in _state_transitions(
        reader, config, world, panel, plans, device
    ):
        directory = out / "states" / key
        record = json.loads((directory / "record.json").read_text())
        fast, provenance = fast_io.load_fast(directory / "fast.pt", reader)
        if record["key"] != key or record["spec"] != spec or provenance != spec:
            raise ValueError("State provenance changed")
        digest = fast_io.tensor_digest(fast)
        if digest != record["fast_digest"] or digest != fast_io.tensor_digest(rebuilt):
            raise AssertionError("State differs from independently rebuilt one-step transition")
        original._compare_nested(history, record["write_history"], "write_history")
        writes += spec["kind"] in ("single", "continuous")
        base, first = _read(reader, fast, world, panel, spec, device)
        original._compare_readouts(base, record["readout"])
        original._compare_nested(first, record["first"], "first_hop")
        checked_queries += sum(len(p["answer"]) for p in base["predictions"].values())
        checked_queries += sum(len(p["answer"]) for p in first["predictions"].values())
        records.append({**record, "readout": base, "first": first})
    analysis, paired = _analyze(panel, records, world.metadata)
    original._compare_nested(analysis, {k: summary[k] for k in analysis}, "summary")
    original._compare_nested(paired, json.loads((out / "paired-queries.json").read_text()))
    if before != fast_io.tensor_digest(reader.backbone.state_dict()):
        raise AssertionError("Audit changed the backbone")
    return {
        "passed": True,
        "identity": expected_identity,
        "states": len(plans),
        "queries": checked_queries,
        "one_step_writes_rebuilt": writes,
        "independent_donors_rewritten": len(panel["facts"]),
        "all_continuous_trajectories_rebuilt_from_A0": True,
        "merged_states_rebuilt_from_independent_donors": True,
        "answer_and_EOS_exact": True,
        "all_hop_and_retention_readouts_replayed": True,
        "backbone_unchanged": True,
        "new_base_parameter_updates": 0,
    }


def paired_training_difference(one_step, four_step):
    """Same query/order/prefix; report both-baseline-correct coverage separately."""

    def indexed(rows):
        values = {(tuple(r["query"]), r["rotation"], r["t"]): r for r in rows}
        if len(values) != len(rows):
            raise ValueError("Duplicate training-comparison query/order/prefix")
        return values

    new, old = indexed(one_step), indexed(four_step)
    if new.keys() != old.keys():
        raise ValueError("Training comparisons must use identical query/order/prefix pairs")
    result = {}
    for condition in ("all", "both_baseline_correct"):
        pairs = [
            (old[key], row)
            for key, row in new.items()
            if condition == "all" or (row["baseline_correct"] and old[key]["baseline_correct"])
        ]
        differences = [int(b["new_correct"]) - int(a["new_correct"]) for a, b in pairs]
        result[condition] = {
            "n": len(pairs),
            "coverage": len(pairs) / len(new) if new else None,
            "one_step_training": one.fraction(b["new_correct"] for _, b in pairs),
            "four_step_training": one.fraction(a["new_correct"] for a, _ in pairs),
            "one_step_minus_four_step_training_pp": 100 * sum(differences) / len(differences)
            if differences
            else None,
            "improved": differences.count(1),
            "regressed": differences.count(-1),
            "unchanged": differences.count(0),
        }
    return result


def _control_report(paths, values, control_outputs):
    reports = {}
    for architecture, data in values.items():
        directory = (control_outputs or {}).get(architecture)
        required = (
            "summary.json",
            "paired-queries.json",
            "result-audit.json",
            "result-cloud-audit.json",
        )
        missing = [
            name for name in required if directory is None or not (Path(directory) / name).is_file()
        ]
        if missing:
            reports[architecture] = {"status": "pending", "missing": missing}
            continue
        directory = Path(directory)
        old_summary = json.loads((directory / "summary.json").read_text())
        new_summary = json.loads((Path(paths[architecture]) / "summary.json").read_text())
        old_identity = old_summary.get("identity", {})
        old_settings = old_identity.get("settings", {})
        reasons = []
        for stage in ("audit", "cloud-audit"):
            if (
                json.loads((directory / f"result-{stage}.json").read_text()).get("passed")
                is not True
            ):
                reasons.append(f"{stage} has not passed")
        if (
            old_summary.get("schema") != merge.BATCH
            or old_settings.get("parent_writer_steps") != 4
            or old_settings.get("writer_steps") != 1
            or old_identity.get("architecture") != architecture
            or old_identity.get("panel_sha256") != new_summary["identity"]["panel_sha256"]
        ):
            reasons.append(
                "Control must be four-step training, one-step deployment, same architecture/panel"
            )
        if reasons:
            reports[architecture] = {"status": "unavailable", "reasons": reasons}
            continue
        old = json.loads((directory / "paired-queries.json").read_text())
        reports[architecture] = {
            "status": "complete",
            "training_writer_steps": 4,
            "evaluation_writer_steps": 1,
            "source_sha256": {name: sha256(directory / name) for name in required},
            "roles": {
                role: {
                    method: paired_training_difference(
                        data["roles"][role][method], old["roles"][role][method]
                    )
                    for method in METHODS
                }
                for role in ROLES
            },
            "all_control_results": old_summary,
        }
    return {
        "status": "complete"
        if all(r["status"] == "complete" for r in reports.values())
        else "pending",
        "architectures": reports,
        "comparison": "One-step versus four-step continuous training; both deploy one-step writers",
        "four_step_training_four_step_deployment": "Preserved under original parent batches",
    }


def aggregate(paths, out, control_outputs=None):
    """Report all queries plus the common baseline-correct subset, even if empty."""
    if set(paths) != {"transformer", "loop", "discoloop"}:
        raise ValueError("All three architectures are required")
    values = {
        name: json.loads((Path(path) / "paired-queries.json").read_text())
        for name, path in paths.items()
    }
    reports = {}
    for role in ROLES:
        universes = [
            {tuple(r["query"]) for r in data["baseline"][role]} for data in values.values()
        ]
        if any(u != universes[0] for u in universes):
            raise ValueError("Architecture query panels differ")
        common = set.intersection(
            *[
                {tuple(r["query"]) for r in data["baseline"][role] if r["correct"]}
                for data in values.values()
            ]
        )
        reports[role] = {
            "all_queries": len(universes[0]),
            "common_baseline_correct": len(common),
            "coverage": len(common) / len(universes[0]) if universes[0] else None,
            "architectures": {
                name: {
                    method: merge.summarize_rows(r for r in rows if tuple(r["query"]) in common)
                    for method, rows in data["roles"][role].items()
                }
                for name, data in values.items()
            },
        }
    result = {
        "schema": BATCH,
        "roles": reports,
        "training_writer_steps": 1,
        "evaluation_writer_steps": 1,
        "new_base_parameter_updates": 0,
        "all_results": {
            name: json.loads((Path(path) / "summary.json").read_text())
            for name, path in paths.items()
        },
        "paired_method_differences": {
            name: {
                role: merge.paired_difference(
                    data["roles"][role]["continuous"], data["roles"][role]["independent_merge"]
                )
                for role in ROLES
            }
            for name, data in values.items()
        },
        "four_step_training_control": _control_report(paths, values, control_outputs),
    }
    write_json(Path(out) / "comparison.json", result)
    return result
