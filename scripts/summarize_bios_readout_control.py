"""Independently rescore H5/MLP/All, auditing frozen tied weights via mmap reads."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from llm_memory_editability.bios_cross_continue import edit_metrics, file_hash, validate_optimizer
from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_organization_train import state_hash
from llm_memory_editability.bios_readout_control import (
    CHECKPOINTS,
    SCOPE,
    STUDY,
    control_sources,
    sampling_stream,
    score_arrays,
    validate_parent_config,
)

ROOT = Path(__file__).resolve().parents[1]


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def csv_write(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def additional_metrics(world, pair, observed, old):
    result = edit_metrics(world, pair, observed, old)
    for name, ids in (
        ("E", pair["E"]),
        ("E_default", np.intersect1d(pair["E"], world.root_ids)),
        ("E_actual", np.intersect1d(pair["E"], world.actual_ids)),
        ("D", pair["D"]),
        ("D_heldout", np.intersect1d(pair["D"], world.heldout_ids)),
        ("D_conflict", pair["conflict_D"]),
        ("D_conflict_heldout", np.intersect1d(pair["conflict_D"], world.heldout_ids)),
    ):
        result[f"{name}_count"] = int(observed["correct"][ids].sum())
        result[f"{name}_n"] = len(ids)
    for pool, ids in (("full", np.flatnonzero(pair["strata"] >= 0)), ("heldout", pair["heldout"])):
        known = ids[old[ids]]
        broken = int((~observed["correct"][known]).sum())
        result[f"U_{pool}_n"] = len(ids)
        result[f"U_{pool}_known"] = len(known)
        result[f"U_{pool}_coverage"] = len(known) / len(ids)
        result[f"U_{pool}_broken"] = broken
        result[f"U_{pool}_damage"] = broken / len(known) if len(known) else None
    return flatten(result)


def summarize(source):
    source = Path(source).resolve()
    out = source / "summary"
    out.mkdir(parents=True, exist_ok=True)
    launch = json.loads((source / "launch-contract.json").read_text())
    if launch["sources"] != control_sources() or launch["runner_sha256"] != file_hash(
        ROOT / "scripts/run_bios_readout_control.py"
    ):
        raise ValueError("H5 frozen sources or runner changed")
    jobs = launch["jobs"]
    expected = {
        (width, world, seed, condition)
        for width in (256, 768)
        for world in (0, 1)
        for seed in (0, 1)
        for condition in CONDITIONS
    }
    if (
        len(jobs) != 24
        or {(j["width"], j["world"], j["seed"], j["condition"]) for j in jobs} != expected
    ):
        raise ValueError("H5 launch matrix incomplete or duplicated")
    worlds = {world: make_cross_world(world) for world in (0, 1)}
    rows, missing, ledger = [], [], {}
    cases, reference_cases, frozen_checks, models = 0, 0, 0, 0
    for job in jobs:
        run = source / job["id"]
        if not (run / "complete.json").exists():
            missing.append(str(run))
            continue
        config = json.loads((run / "config.json").read_text())
        if config["study"] != STUDY or config["sources"] != launch["sources"]:
            raise ValueError("H5 worker study/source identity differs")
        for key in ("width", "world", "seed", "condition"):
            if config[key] != job[key]:
                raise ValueError(f"H5 worker matrix identity differs: {key}")
        parent, world = Path(job["parent"]), worlds[job["world"]]
        if config["parent"]["directory"] != str(parent.resolve()):
            raise ValueError("H5 parent identity differs")
        validate_parent_config(json.loads((parent / "config.json").read_text()), world)
        for name in ("config.json", "predictions-15360.npz", "learning-complete.json"):
            if file_hash(parent / name) != config["parent"]["files_sha256"][name]:
                raise ValueError(f"H5 parent metadata changed: {name}")
        for name, wanted in config["parent"]["reference_files_sha256"].items():
            if file_hash(parent / name) != wanted:
                raise ValueError(f"Original reference changed: {name}")
        old = score_arrays(arrays(parent / "predictions-15360.npz"), world.answers)
        parent_checkpoint = torch.load(
            parent / "model-15360.pt", map_location="cpu", weights_only=False, mmap=True
        )
        frozen = parent_checkpoint["model"]["token.weight"].clone()
        del parent_checkpoint
        frozen_hash = state_hash({"token.weight": frozen})
        for chain, chain_name in enumerate(CHAINS):
            pair = edit_pair(world, chain)
            e, r = sampling_stream(world.seed, chain)
            for scope in ("mlp", "all", SCOPE):
                dest = (
                    run / f"{chain_name}-exception-{SCOPE}"
                    if scope == SCOPE
                    else parent / "edits" / f"{chain_name}-exception-{scope}"
                )
                completion = json.loads((dest / "complete.json").read_text())
                sets = arrays(dest / "sets.npz")
                for key, value in pair.items():
                    np.testing.assert_array_equal(sets[key], value)
                for key, value in (
                    ("old_correct", old["correct"]),
                    ("edit_sampling", e),
                    ("replay_sampling", r),
                ):
                    np.testing.assert_array_equal(sets[key], value)
                timeline = json.loads((dest / "trajectory.json").read_text())
                if [point["step"] for point in timeline] != list(CHECKPOINTS):
                    raise ValueError("H5/reference trajectory grid differs")
                for point in timeline:
                    step = point["step"]
                    path = dest / f"predictions-{step}.npz"
                    observed = score_arrays(arrays(path), pair["exception"])
                    ledger[str(path)] = file_hash(path)
                    metrics = edit_metrics(world, pair, observed, old["correct"])
                    for key in metrics:
                        if key in point and point[key] != metrics[key]:
                            raise ValueError(
                                f"Stored H5/reference metric differs: {dest}/{step}/{key}"
                            )
                    if step == 0:
                        for key in ("prediction", "ended"):
                            np.testing.assert_array_equal(observed[key], old[key])
                    rows.append(
                        dict(
                            **{k: job[k] for k in ("width", "world", "seed", "condition")},
                            chain=chain_name,
                            scope=scope,
                            step=step,
                            **additional_metrics(world, pair, observed, old["correct"]),
                        )
                    )
                if scope == SCOPE:
                    if (
                        completion["scope"] != SCOPE
                        or completion["frozen_token_sha256"] != frozen_hash
                    ):
                        raise ValueError("H5 claimed frozen weights differ from parent")
                    if completion["final"] != timeline[-1]:
                        raise ValueError("H5 final completion differs from trajectory")
                    final = torch.load(
                        dest / "model-final.pt", map_location="cpu", weights_only=False, mmap=True
                    )
                    resumed = torch.load(
                        dest / "resume.pt", map_location="cpu", weights_only=False, mmap=True
                    )
                    if final["step"] != 512 or resumed["step"] != 512:
                        raise ValueError("H5 final checkpoint step differs")
                    if resumed["identity_sha256"] != file_hash(run / "config.json"):
                        raise ValueError("H5 resume source identity differs")
                    if not torch.equal(final["model"]["token.weight"], frozen) or not torch.equal(
                        resumed["model"]["token.weight"], frozen
                    ):
                        raise ValueError("Frozen H5 readout actually changed")
                    validate_optimizer(resumed["optimizer"], 512, STUDY["lr"])
                    del final, resumed
                    frozen_checks += 1
                    cases += 1
                else:
                    reference_cases += 1
        models += 1
    complete = models == 24 and cases == 48 and reference_cases == 96 and len(rows) == 576
    paired = []
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ("width", "world", "seed", "condition", "chain", "step"))
        grouped[key][row["scope"]] = row
    metrics = (
        "E",
        "E_default",
        "E_actual",
        "D_heldout",
        "D_conflict_heldout",
        "U_full_damage",
        "U_heldout_damage",
        "U_full_0_rate",
    )
    for key, group in sorted(grouped.items()):
        if set(group) != {"mlp", "all", SCOPE}:
            raise ValueError("Unpaired H5 comparison")
        row = dict(zip(("width", "world", "seed", "condition", "chain", "step"), key, strict=True))
        for metric in metrics:
            for scope in ("mlp", "all", SCOPE):
                row[f"{scope}_{metric}"] = group[scope][metric]
            for reference in ("mlp", "all"):
                a, b = group[SCOPE][metric], group[reference][metric]
                row[f"freeze_minus_{reference}_{metric}"] = (
                    a - b if a is not None and b is not None else None
                )
        paired.append(row)
    audit = dict(
        complete=complete,
        models=models,
        new_cases=cases,
        reference_cases=reference_cases,
        all_checkpoints=len(rows),
        frozen_readout_checks=frozen_checks,
        missing=missing,
        all_predictions_rescored=True,
        full_model_weights_hashed=False,
        frozen_weights_and_optimizer_steps_checked=True,
        source_sha256=file_hash(__file__),
    )
    csv_write(out / "editing.csv", rows)
    csv_write(out / "paired-scopes.csv", paired)
    write_json(out / "audit.json", audit)
    write_json(out / "sources.json", ledger)
    lines = [
        "# H5 冻结输入/输出 token 参数",
        "",
        str(audit),
        "",
        "输入词嵌入与输出读出绑定，因此此干预同时冻结两者；位置嵌入及其余参数仍训练。"
        "这不是等参数量比较，也不能单独把差异归于输出读出；可训练梯度集合改变还会影响总范数裁剪。"
        "所有比较使用同一原始父节点、E93/R和采样流，固定512步，不选择最好检查点。",
        "",
        "| Width | Scope | Cases | E | Heldout conflict D | Full U damage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for width in (256, 768):
        for scope in ("mlp", "all", SCOPE):
            group = [
                r for r in rows if r["width"] == width and r["scope"] == scope and r["step"] == 512
            ]
            if group:
                values = [
                    mean(r[k] for r in group if r[k] is not None)
                    for k in ("E", "D_conflict_heldout", "U_full_damage")
                ]
                lines.append(
                    f"| {width} | {scope} | {len(group)} | "
                    + " | ".join(f"{v * 100:.3f}%" for v in values)
                    + " |"
                )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if not complete or missing:
        raise ValueError("H5 matrix incomplete")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/bios-mechanism-dev-v1/h5-readout-control")
    print(json.dumps(summarize(parser.parse_args().source), indent=2))
