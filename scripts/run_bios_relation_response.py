"""Freeze, complete and diagnose one paired four-cell experiment."""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability import bios_direction as engine
from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest, load_parent
from llm_memory_editability.bios_direction_cross import bounded_representation_geometry
from llm_memory_editability.bios_relation_response import (
    CELLS,
    LAYER,
    calibrate_controls,
    check_reused_task,
    derivatives,
    factorial_tasks,
    first_adam_update,
    isolated_directions,
    make_probe,
    values,
)

ART = ROOT / "docs/development-artifacts/relation-response-v1"
OUT = ROOT / "results/bios-relation-response-v1"
CONFIG = ROOT / "configs/bios-relation-response-v1.json"
OLD_ART = ROOT / "docs/development-artifacts/direction-cross-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def parent(world, seed):
    return (
        ROOT
        / "results/bios-cross-scale-dev-v1/width-256"
        / f"world-{world}-seed-{seed}-neither/model-15360.pt"
    )


def old_directory(world, seed):
    return (
        ROOT / "results/bios-direction-cross-v1/main" / f"world-{world}-seed-{seed}/func-soft-adam"
    )


def selected():
    return json.loads((OLD_ART / "selected.json").read_text())["choices"]["func-soft-adam"]


def freeze():
    config = json.loads(CONFIG.read_text())
    files = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/report_bios_relation_response.py",
        ROOT / "tests/test_bios_relation_response.py",
        OLD_ART / "selected.json",
    ]
    files += [
        ROOT / "src/llm_memory_editability" / name
        for name in (
            "bios_relation_response.py",
            "bios_direction.py",
            "bios_direction_cross.py",
            "bios_cross.py",
            "bios_cross_train.py",
            "bios_model.py",
            "bios_data.py",
        )
    ]
    reuse = []
    for world, seed in itertools.product((0, 1), repeat=2):
        files.append(parent(world, seed))
        directory = old_directory(world, seed)
        receipt = json.loads((directory / "complete.json").read_text())
        for name, expected in receipt["files"].items():
            assert digest(directory / name) == expected, (directory, name)
            files.append(directory / name)
        records = json.loads((directory / "metrics.json").read_text())
        tasks = factorial_tasks(world, seed, "cpu")
        for p in range(6):
            for offset in (0, 1):
                task, record = tasks[4 * p + offset], records[2 * p + offset]
                check_reused_task(task, record["task"])
                assert record["config"] == selected()
                assert record["accepted"] == 1024 and not record["failed"]
                reuse.append(
                    dict(
                        task=task.name,
                        source=str(directory.relative_to(ROOT)),
                        index=2 * p + offset,
                    )
                )
    lock = dict(
        created=now(),
        config=config,
        selected=selected(),
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
        reuse=reuse,
        environment=dict(torch=torch.__version__, numpy=np.__version__),
    )
    target = ART / "lock.json"
    if target.exists():
        previous = json.loads(target.read_text())
        assert previous["files"] == lock["files"]
        return
    write_json(target, lock)
    write_json(
        ART / "tasks.json",
        [
            t.manifest()
            for w, s in itertools.product((0, 1), repeat=2)
            for t in factorial_tasks(w, s, "cpu")
        ],
    )
    (ART / "preregistration.md").write_text((ROOT / "docs/experimental-protocol.md").read_text())
    for p in files:
        if "results/" not in str(p.relative_to(ROOT)):
            destination = ART / "source" / p.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(p.read_bytes())
    print(
        json.dumps(dict(event="frozen", reused=len(reuse), new=config["new_trajectories"])),
        flush=True,
    )


def verify(world, seed):
    lock = json.loads((ART / "lock.json").read_text())
    for path, expected in lock["files"].items():
        # Each worker reads its parents and all code; other workers' sources
        # were checked together in freeze, avoiding needless shared-disk reads.
        if (
            path.startswith("results/")
            and f"world-{world}-seed-{seed}/" not in path
            and f"world-{world}-seed-{seed}-" not in path
        ):
            continue
        assert digest(ROOT / path) == expected, path


def predict(world, seed, device):
    destination = OUT / f"world-{world}-seed-{seed}/prediction.json"
    if destination.exists():
        return
    model = load_parent(parent(world, seed), device)
    w0 = model.blocks[LAYER].mlp.down.weight.detach().clone()
    tasks = factorial_tasks(world, seed, device)
    config = json.loads(CONFIG.read_text())
    predictions, saved_gradients = [], []
    for p in range(6):
        four = tasks[4 * p : 4 * p + 4]
        task = four[0]
        probe = make_probe(model, task)
        logits, baseline, gradients = derivatives(probe, w0, task.metadata)
        geometry, calibration = calibrate_controls(
            probe, w0, task.metadata, gradients, config["pre_control_magnitudes"]
        )
        first = []
        for variant in four:
            delta = first_adam_update(model, variant, selected())
            predicted = (gradients.double() * delta.double()).sum((-1, -2))
            with torch.no_grad():
                _, after = values(probe, w0 + delta, task.metadata)
            first.append(
                dict(
                    cell=variant.metadata["cell"],
                    delta_norm=float(delta.norm()),
                    predicted_ab=(predicted[:, 0] - predicted[:, 1]).cpu().tolist(),
                    observed_ab=((after - baseline)[:, 0] - (after - baseline)[:, 1])
                    .cpu()
                    .tolist(),
                )
            )
        predictions.append(
            dict(
                metadata=task.metadata,
                baseline_ac_bc=baseline.cpu().tolist(),
                baseline_prediction=logits.argmax(-1).cpu().tolist(),
                geometry=geometry,
                calibration=calibration,
                first_step=first,
            )
        )
        saved_gradients.append(gradients.cpu())
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(gradients=torch.stack(saved_gradients), parent=w0.cpu()),
        destination.with_suffix(".pt"),
    )
    write_json(
        destination,
        dict(
            created=now(),
            parent_sha256=digest(parent(world, seed)),
            gradients_sha256=digest(destination.with_suffix(".pt")),
            cases=predictions,
            prospective_cells=["ba", "bb"],
            already_observed_cells=["aa", "ab"],
        ),
    )
    print(json.dumps(dict(event="predictions_frozen", world=world, seed=seed, cases=6)), flush=True)


def edit(world, seed, device):
    for w, s in itertools.product((0, 1), repeat=2):
        assert (OUT / f"world-{w}-seed-{s}/prediction.json").exists()
    prediction = OUT / f"world-{world}-seed-{seed}/prediction.json"
    directory = OUT / f"world-{world}-seed-{seed}/new"
    if (directory / "complete.json").exists():
        return
    write_json(
        directory / "launch.json",
        dict(
            created=now(),
            prediction_sha256=digest(prediction),
            device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        ),
    )
    model = load_parent(parent(world, seed), device)
    tasks = [t for t in factorial_tasks(world, seed, device) if t.metadata["cell"] in ("ba", "bb")]
    engine.representation_geometry = bounded_representation_geometry
    records = engine.edit_batch(
        model, LAYER, tasks, "func-soft-adam", [selected()] * 12, 1024, directory
    )
    print(
        json.dumps(dict(event="new_edits_complete", world=world, seed=seed, cases=len(records))),
        flush=True,
    )


@torch.no_grad()
def answer(model, prompt):
    tokens = prompt[None]
    value = model(tokens)[:, -1].argmax(-1)
    ended = model(torch.cat((tokens, value[:, None]), -1))[:, -1].argmax(-1).eq(3)
    return int(value), bool(ended)


@torch.no_grad()
def two_step(model, task, weight, world):
    w = model.blocks[LAYER].mlp.down.weight
    original = w.clone()
    w.copy_(weight)
    try:
        meta = task.metadata
        chain, person, group = (meta[k] for k in ("chain", "person", "group"))
        membership = int(world.membership_ids[chain, person])
        root = int(world.root_ids[chain, group])
        first, first_eos = answer(
            model, torch.tensor(world.prompts[membership, :4], device=w.device)
        )
        prompt = torch.tensor(world.prompts[root, :4], device=w.device)
        prompt[1] = first
        second, second_eos = answer(model, prompt)
        focal = int(world.derived_ids[chain, person])
        direct, direct_eos = answer(model, torch.tensor(world.prompts[focal], device=w.device))
        return dict(
            membership_correct=bool(first == int(world.answers[membership]) and first_eos),
            autonomous=bool(first_eos and second_eos and second == meta["root_target"]),
            direct_value=direct,
            direct_eos=direct_eos,
        )
    finally:
        w.copy_(original)


def analyze(world, seed, device):
    directory = OUT / f"world-{world}-seed-{seed}"
    destination = directory / "analysis.json"
    if destination.exists():
        return
    model = load_parent(parent(world, seed), device)
    old, new = old_directory(world, seed), directory / "new"
    states, records = {}, {}
    for name, path in (("old", old), ("new", new)):
        receipt = json.loads((path / "complete.json").read_text())
        for filename, expected in receipt["files"].items():
            assert digest(path / filename) == expected
        states[name] = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        records[name] = json.loads((path / "metrics.json").read_text())
        assert torch.equal(states[name]["parent"], model.blocks[LAYER].mlp.down.weight.cpu())
    launch = json.loads((new / "launch.json").read_text())
    prediction = json.loads((directory / "prediction.json").read_text())
    assert digest(directory / "prediction.json") == launch["prediction_sha256"]
    assert prediction["created"] < launch["created"]
    data_world = make_cross_world(world, ROOT / "data/bios-organization-v1")
    tasks = factorial_tasks(world, seed, device)
    analysis = []
    for task in tasks:
        cell, p = task.metadata["cell"], task.metadata["pair_index"]
        source = "old" if cell in ("aa", "ab") else "new"
        offset = 0 if cell in ("aa", "ba") else 1
        index = 2 * p + offset
        record, state = records[source][index], states[source]
        if source == "old":
            check_reused_task(task, record["task"])
        else:
            assert task.manifest() == record["task"]
        probe = make_probe(model, task)
        nodes = sorted(map(int, state["checkpoints"]))
        trajectory = []
        for ni, node in enumerate(nodes):
            weight = state["checkpoints"][str(node)][index].to(device)
            logits, contrasts, gradients = derivatives(probe, weight, task.metadata)
            _, geometry = isolated_directions(gradients)
            row = dict(
                step=node,
                margins_ab=(contrasts[:, 0] - contrasts[:, 1]).cpu().tolist(),
                contrasts_ac_bc=contrasts.cpu().tolist(),
                prediction=logits.argmax(-1).cpu().tolist(),
                geometry=geometry,
            )
            if ni + 1 < len(nodes):
                next_weight = state["checkpoints"][str(nodes[ni + 1])][index].to(device)
                delta = next_weight - weight
                predicted = (gradients.double() * delta.double()).sum((-1, -2))
                with torch.no_grad():
                    _, after = values(probe, next_weight, task.metadata)
                observed = after - contrasts
                row["interval"] = dict(
                    next_step=nodes[ni + 1],
                    delta_norm=float(delta.norm()),
                    predicted_ab=(predicted[:, 0] - predicted[:, 1]).cpu().tolist(),
                    observed_ab=(observed[:, 0] - observed[:, 1]).cpu().tolist(),
                    note=(
                        "Observed interval displacement; post-hoc linearization, "
                        "not a prospective optimizer forecast."
                    ),
                )
            trajectory.append(row)
        final_weight = state["weights"][index].to(device)
        fresh = engine.evaluate_weights(model, LAYER, [task], final_weight[None])[0]
        stored = record["timeline"][-1]
        for key in ("prediction", "e_joint", "joint"):
            assert fresh[key] == stored[key], (task.name, key)
        for name in task.sets:
            for key in ("correct", "broken", "old_known", "n"):
                assert fresh["sets"][name][key] == stored["sets"][name][key], (task.name, name, key)
        geometry, calibration = calibrate_controls(
            probe, final_weight, task.metadata, gradients, [0.01]
        )
        first = prediction["cases"][p]["first_step"][CELLS.index(cell)]
        observed_first = trajectory[0]["interval"]["observed_ab"]
        analysis.append(
            dict(
                metadata=task.metadata,
                source=source,
                source_index=index,
                accepted=record["accepted"],
                final=fresh,
                two_step=two_step(model, task, final_weight, data_world),
                trajectory=trajectory,
                final_geometry=geometry,
                final_calibration=calibration,
                first_step_frozen=first,
                first_step_replay_max_difference=max(
                    abs(a - b) for a, b in zip(first["observed_ab"], observed_first, strict=True)
                ),
            )
        )
    write_json(
        destination, dict(created=now(), cases=analysis, all_endpoint_generation_verified=True)
    )
    print(
        json.dumps(dict(event="analysis_complete", world=world, seed=seed, cases=len(analysis))),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "predict", "edit", "analyze"))
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.03, device)
    verify(args.world, args.seed)
    {"predict": predict, "edit": edit, "analyze": analyze}[args.command](
        args.world, args.seed, device
    )


if __name__ == "__main__":
    main()
