"""Fail-closed CPU audit of the locked 96-parent / 768-edit confirmation matrix.

No reserved world is generated until the prospective lock and every producer
completion record pass. Never trains, reads weights, selects models, or computes
inferential statistics; the independently audited endpoint CSVs feed that stage.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from summarize_bios_shortcut_control import (
    chain_rows,
    cohorts,
    score_direct,
    score_two_step,
)
from summarize_bios_shortcut_edit_v2 import flatten

from llm_memory_editability.bios_cross import (
    CHAINS,
    CONDITIONS,
    documents,
    make_cross_world,
    qa_schedule,
)
from llm_memory_editability.bios_cross_continue import expected_exposure
from llm_memory_editability.bios_cross_train import learning_metrics, source_hashes
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_shortcut_confirmation import (
    LEARNING_CHECKPOINTS,
    PROTOCOL,
    WORLDS,
    confirmation_sources,
    validate_config,
    validate_frozen_design,
    validate_prepared_world,
)
from llm_memory_editability.bios_shortcut_confirmation_sets import make_confirmation_edit_pair
from llm_memory_editability.bios_shortcut_control import high_exception_world
from llm_memory_editability.bios_shortcut_edit_v2 import (
    CHECKPOINTS,
    edit_metrics,
    sampling_stream,
    score_arrays,
)
from llm_memory_editability.bios_shortcut_matched_edit import KINDS, PHASES

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Sources:
    def __init__(self):
        self.hashes = {}

    def bytes(self, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        previous = self.hashes.setdefault(str(path), digest)
        if previous != digest:
            raise ValueError(f"Source changed between reads: {path}")
        return content

    def json(self, path):
        return json.loads(self.bytes(path))

    def arrays(self, path):
        with np.load(io.BytesIO(self.bytes(path)), allow_pickle=False) as saved:
            return dict(saved)

    def recheck(self):
        for path, expected in self.hashes.items():
            if sha(path) != expected:
                raise ValueError(f"Source changed after analysis: {path}")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix():
    return [
        dict(phase=phase, world=world, seed=seed, condition=condition)
        for phase in PHASES
        for world in WORLDS
        for seed in (0, 1)
        for condition in CONDITIONS
    ]


def run_path(source, identity):
    return (
        source
        / identity["phase"]
        / "width-256"
        / (f"world-{identity['world']}-seed-{identity['seed']}-{identity['condition']}")
    )


def validate_endpoint_rows(learning, editing):
    """Check keys before constructing dictionaries; duplicates must never overwrite."""
    common = ("world", "seed", "condition", "chain", "prevalence")
    expected_learning = {
        (w, s, c, chain, p)
        for w in WORLDS
        for s in (0, 1)
        for c in CONDITIONS
        for chain in CHAINS
        for p in PHASES
    }
    keys = [tuple(r[k] for k in common) for r in learning]
    if len(keys) != len(set(keys)) or set(keys) != expected_learning:
        raise ValueError("Learning endpoint matrix is incomplete or duplicated")
    if any(
        "support" in r
        or r["step"] != 15360
        or r["cohort"] != "original_exception"
        or r["split"] != "heldout"
        or r["n"] != 64
        or not 0 <= r["correct"] <= 64
        for r in learning
    ):
        raise ValueError("Learning endpoint denominator / cohort changed")
    expected_editing = {
        (*key, support, kind) for key in expected_learning for support in (0, 1) for kind in KINDS
    }
    edit_keys = [tuple(r[k] for k in (*common, "support", "kind")) for r in editing]
    if len(edit_keys) != len(set(edit_keys)) or set(edit_keys) != expected_editing:
        raise ValueError("Editing endpoint matrix is incomplete or duplicated")
    if any(r["step"] != 512 or r["n"] != 9 or not 0 <= r["correct"] <= 9 for r in editing):
        raise ValueError("Editing endpoint denominator / checkpoint changed")


def validate_lock(config_path, lock_path, sources):
    checked_config, checked_lock = validate_frozen_design(config_path, lock_path)
    config, lock = sources.json(config_path), sources.json(lock_path)
    if config != checked_config or lock != checked_lock:
        raise ValueError("Prospective lock changed between validation and reading")
    validate_config(config)
    if lock.get("status") != "frozen_before_any_confirmation_model_or_outcome":
        raise ValueError("A prospective lock is required before reading confirmation outcomes")
    if lock["config_sha256"] != sha(config_path) or lock["sources"] != confirmation_sources():
        raise ValueError("Prospective configuration or scientific sources changed")
    analysis = lock.get("analysis_sources", {})
    own = str(Path(__file__).resolve().relative_to(ROOT))
    required_analysis = {
        own,
        "scripts/summarize_bios_shortcut_control.py",
        "scripts/summarize_bios_shortcut_edit_v2.py",
        "scripts/archive_bios_mechanism_stage.py",
    }
    if not required_analysis <= set(analysis):
        raise ValueError(
            "Summary, imported helpers and archive helper must be prospectively locked"
        )
    for name, expected in analysis.items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Prospective analysis source changed: {name}")
        sources.bytes(path)
    for name, expected in lock["data_source_files"].items():
        path = ROOT / name
        sources.bytes(path)
        if sources.hashes[str(path.resolve())] != expected:
            raise ValueError("Original data source differs from prospective lock")
    return config, lock


def audit_archive(index_path, lock, sources):
    index = sources.json(index_path)
    if any(
        index.get(key) is not True
        for key in ("complete", "stage_complete", "every_member_verified")
    ):
        raise ValueError("Full independently verified weight archive is required")
    if Path(index["source"]).resolve() != Path(lock["output_root"]).resolve():
        raise ValueError("Archive source is not the prospectively locked output root")
    if index["files"] != len(index["entries"]) or index["bytes"] != sum(
        entry["bytes"] for entry in index["entries"].values()
    ):
        raise ValueError("Archive member totals mismatch")
    return index


def require_archived(index, relative, metadata):
    if index["entries"].get(relative) != metadata:
        raise ValueError(f"Producer seal differs from verified archive: {relative}")


def required_artifacts():
    required = {
        "launch-contract.json",
        "manipulation.npz",
        "learning/config.json",
        "learning/schedule.npz",
        "learning/learning.json",
        "learning/learning-complete.json",
        "learning/resume.pt",
    }
    for step in LEARNING_CHECKPOINTS:
        required.update((f"learning/predictions-{step}.npz", f"learning/model-{step}.pt"))
    required.update(f"learning/two-step-{chain}.npz" for chain in CHAINS)
    for support in (0, 1):
        for chain in CHAINS:
            for kind in KINDS:
                edit_prefix = f"edits/support-{support}/{chain}-{kind}-mlp"
                required.update(
                    f"{edit_prefix}/{name}"
                    for name in (
                        "complete.json",
                        "sets.npz",
                        "data-contract.json",
                        "trajectory.json",
                        "model-final.pt",
                        "resume.pt",
                    )
                )
                required.update(f"{edit_prefix}/predictions-{step}.npz" for step in CHECKPOINTS)
    return required


def audit_completion(run, identity, lock_hash, sources, archive_index, source_root, environment):
    complete = sources.json(run / "complete.json")
    run_prefix = str(run.relative_to(source_root))
    require_archived(
        archive_index,
        f"{run_prefix}/complete.json",
        {
            "bytes": (run / "complete.json").stat().st_size,
            "sha256": sources.hashes[str((run / "complete.json").resolve())],
        },
    )
    expected = dict(
        status="complete",
        protocol=PROTOCOL,
        learning_steps=15360,
        edit_cases=8,
        edit_steps=512,
        lock_sha256=lock_hash,
    )
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Wrong producer completion: {run}")
    launch = sources.json(run / "launch-contract.json")
    if any(launch[k] != v for k, v in identity.items()):
        raise ValueError(f"Run-directory / producer identity mismatch: {run}")
    if launch["preregistration_lock_sha256"] != lock_hash:
        raise ValueError("Worker was launched under a different prospective lock")
    if launch["sources"] != confirmation_sources() or launch["protocol"] != PROTOCOL:
        raise ValueError("Worker scientific source mismatch")
    if any(launch[key] != value for key, value in environment.items()):
        raise ValueError("Worker numerical environment differs from prospective lock")
    required = required_artifacts()
    if not required <= set(complete["artifacts"]):
        raise ValueError("Producer artifact manifest omits required scientific data")
    weights = []
    for relative, metadata in complete["artifacts"].items():
        require_archived(archive_index, f"{run_prefix}/{relative}", metadata)
        path = (run / relative).resolve()
        if run.resolve() not in path.parents or path.stat().st_size != metadata["bytes"]:
            raise ValueError("Artifact path / size mismatch")
        if path.suffix == ".pt":
            weights.append({"path": str(path), **metadata})
            continue
        sources.bytes(path)
        if sources.hashes[str(path)] != metadata["sha256"]:
            raise ValueError(f"Producer artifact hash mismatch: {path}")
    if len(weights) != 23:
        raise ValueError("Each completed parent requires exactly 23 archived weight files")
    return launch, weights


def audit_shared_initialization(seen, identity, config):
    """The same world/seed initializes all six organization/prevalence cells."""
    key = (identity["world"], identity["seed"])
    controlled = {k: config[k] for k in ("model", "initial_sha256", "torch", "numpy", "precision")}
    if key in seen and seen[key] != controlled:
        raise ValueError("Initialization differs across phase/organization within world/seed")
    seen[key] = controlled


def audit_learning(run, identity, config, low, high, manipulation, selected, donors, sources):
    world = low if identity["phase"] == "low" else high
    saved = sources.arrays(run / "manipulation.npz")
    expected = dict(
        selections=selected,
        donors=donors,
        low_answers=low.answers,
        high_answers=high.answers,
        high_exceptions=high.exceptions,
    )
    if set(saved) != set(expected):
        raise ValueError("Manipulation artifact keys differ")
    for key, value in expected.items():
        np.testing.assert_array_equal(saved[key], value)
    learning = run / "learning"
    conf = sources.json(learning / "config.json")
    for key in ("world", "seed", "condition"):
        if conf[key] != identity[key]:
            raise ValueError(f"Learning identity differs: {key}")
    expected_study = {
        **config,
        "confirmation_sources": confirmation_sources(),
        "phase": identity["phase"],
        "manipulation": manipulation,
    }
    if conf["study"] != expected_study or conf["sources"] != source_hashes():
        raise ValueError("Learning study / source differs from prospective contract")
    if any(conf["model"][k] != config[k] for k in ("width", "layers", "heads")) or (
        conf["truth_sha256"] != array_hash(world.answers)
    ):
        raise ValueError("Learning model size / truth mismatch")
    schedule = dict(documents=documents(world, identity["condition"]), qa=qa_schedule(world, 15360))
    saved = sources.arrays(learning / "schedule.npz")
    for key, value in schedule.items():
        np.testing.assert_array_equal(saved[key], value)
        if conf[f"{key}_sha256"] != array_hash(value):
            raise ValueError("Learning schedule identity mismatch")
    if conf["prompts_sha256"] != array_hash(world.prompts):
        raise ValueError("Learning prompt hash mismatch")
    timeline = sources.json(learning / "learning.json")
    if [point["step"] for point in timeline] != list(LEARNING_CHECKPOINTS):
        raise ValueError("Learning checkpoint grid differs")
    done = sources.json(learning / "learning-complete.json")
    if done["status"] != "complete" or done["final"] != timeline[-1]:
        raise ValueError("Learning completion / timeline mismatch")
    return conf, schedule, timeline


def learning_fact_rows(identity, low, high, phase, arrays):
    world = low if phase == "low" else high
    rows = []
    for chain, label in enumerate(CHAINS):
        ids = world.derived_ids[chain]
        for _, cohort, mask in cohorts(low, high, chain, phase):
            if cohort in ("ordinary", "exception"):
                continue
            take = mask & np.isin(ids, world.heldout_ids[chain])
            root = world.root_ids[chain, world.memberships[chain]]
            row = {
                **identity,
                "phase": phase,
                "chain": label,
                "cohort": cohort,
                "split": "heldout",
                "n": int(take.sum()),
            }
            for name, query in (
                ("actual", world.actual_ids[chain]),
                ("membership", world.membership_ids[chain]),
                ("root", root),
            ):
                row[f"{name}_correct"] = int(arrays["correct"][query[take]].sum())
                row[f"{name}_accuracy"] = float(arrays["correct"][query[take]].mean())
                row[f"{name}_nll"] = float(arrays["value_nll"][query[take]].mean())
            rows.append(row)
    return rows


def world_aggregates(learning, editing):
    """All-world descriptive rows, prior to any inferential or subgroup comparison."""
    output = []
    for world in WORLDS:
        for phase in PHASES:
            learn = [r for r in learning if r["world"] == world and r["prevalence"] == phase]
            output.append(
                dict(
                    world=world,
                    prevalence=phase,
                    outcome="learning_original_exception",
                    cases=len(learn),
                    accuracy=float(np.mean([r["correct"] / r["n"] for r in learn])),
                )
            )
            for kind in KINDS:
                cases = [
                    r
                    for r in editing
                    if r["world"] == world and r["prevalence"] == phase and r["kind"] == kind
                ]
                output.append(
                    dict(
                        world=world,
                        prevalence=phase,
                        outcome=f"editing_{kind}_fixed_reference",
                        cases=len(cases),
                        accuracy=float(np.mean([r["correct"] / r["n"] for r in cases])),
                    )
                )
    return output


def summarize(config_path, lock_path, source, output, archive_index_path, require_complete=False):
    sources = Sources()
    config, lock = validate_lock(config_path, lock_path, sources)
    source = Path(source or lock["output_root"]).resolve()
    output = Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Confirmation summary must not overlap raw outcomes")
    expected = matrix()
    missing = [
        str(run_path(source, identity) / "complete.json")
        for identity in expected
        if not (run_path(source, identity) / "complete.json").exists()
    ]
    output.mkdir(parents=True, exist_ok=True)
    if missing:
        audit = dict(
            complete=False,
            expected_models=96,
            complete_models=96 - len(missing),
            missing=missing,
            reserved_worlds_generated=False,
            outputs_sha256={},
        )
        write_json(output / "audit.json", audit)
        write_json(output / "sources.json", sources.hashes)
        if require_complete:
            raise ValueError(f"Confirmation matrix incomplete: {len(missing)} missing parents")
        return audit
    archive_index = audit_archive(archive_index_path, lock, sources)
    prepared = {}
    for w in WORLDS:
        base_root, receipt = validate_prepared_world(config, config_path, lock_path, w)
        sources.json(base_root / "preparation-audit.json")
        files = {
            name: value
            for name, value in receipt["files_sha256"].items()
            if name.startswith(f"world-{w}/")
        }
        for name, expected_hash in files.items():
            sources.bytes(base_root / name)
            if sources.hashes[str((base_root / name).resolve())] != expected_hash:
                raise ValueError("Prepared base world differs from its receipt")
        prepared[w] = (
            base_root,
            {
                **lock["environment"],
                "device_type": "cuda",
                "precision": "BF16 autocast; FP32 model and optimizer",
                "prepared_worlds_audit_sha256": sha(base_root / "preparation-audit.json"),
                "world_files_sha256": files,
            },
        )
    lock_hash, config_hash = sha(lock_path), sha(config_path)
    declared_weights = []
    for identity in expected:
        launch, weights = audit_completion(
            run_path(source, identity),
            identity,
            lock_hash,
            sources,
            archive_index,
            source,
            prepared[identity["world"]][1],
        )
        if launch["config_sha256"] != config_hash:
            raise ValueError("Worker config differs from prospective lock")
        declared_weights.extend(weights)
    if len(declared_weights) != 2208:
        raise ValueError("Full confirmation requires 2208 independently archived weight files")
    # This is the first line allowed to materialize reserved worlds.
    worlds, pairs = {}, {}
    for w in WORLDS:
        low = make_cross_world(w, source_root=prepared[w][0])
        high, manipulation, selected, donors = high_exception_world(low)
        worlds[w] = low, high, manipulation, selected, donors
        for chain in range(2):
            for support in (0, 1):
                pairs[w, chain, support] = make_confirmation_edit_pair(low, high, chain, support)
            for key in ("groups", "E", "D", "selected_people"):
                if np.intersect1d(pairs[w, chain, 0][key], pairs[w, chain, 1][key]).size:
                    raise ValueError(f"Prospectively disjoint supports overlap: {key}")
    learning_rows, learning_facts, learning_metrics_rows, two_rows, edit_rows = [], [], [], [], []
    learning_endpoint, editing_endpoint, paired_initial, shared_initial = [], [], {}, {}
    learning_count, edit_count, edit_checkpoints, two_count = 0, 0, 0, 0
    for identity in expected:
        phase, w = identity["phase"], identity["world"]
        run = run_path(source, identity)
        low, high, manipulation, selected, donors = worlds[w]
        world = low if phase == "low" else high
        conf, schedule, timeline = audit_learning(
            run, identity, config, low, high, manipulation, selected, donors, sources
        )
        audit_shared_initialization(shared_initial, identity, conf)
        pair_key = (w, identity["seed"], identity["condition"])
        controls = {
            k: conf[k]
            for k in (
                "model",
                "initial_sha256",
                "documents_sha256",
                "qa_sha256",
                "prompts_sha256",
                "torch",
                "numpy",
                "precision",
            )
        }
        if pair_key in paired_initial and paired_initial[pair_key]["controls"] != controls:
            raise ValueError(
                "Low/high paired initialization / document / numerical controls differ"
            )
        terminal = None
        for point in timeline:
            step = point["step"]
            arrays = score_direct(
                sources.arrays(run / "learning" / f"predictions-{step}.npz"), world.answers
            )
            exposure = expected_exposure(
                world, schedule["documents"], schedule["qa"], step, config["lr"]
            )
            for key, value in zip(("exposure", "slots", "weighted"), exposure, strict=True):
                np.testing.assert_allclose(arrays[key], value, rtol=1e-11, atol=1e-12)
            if step == 0:
                if pair_key in paired_initial:
                    for key in ("prediction", "ended"):
                        np.testing.assert_array_equal(arrays[key], paired_initial[pair_key][key])
                else:
                    paired_initial[pair_key] = {
                        "controls": controls,
                        **{k: arrays[k].copy() for k in ("prediction", "ended")},
                    }
            measured = learning_metrics(world, arrays)
            for key, value in measured.items():
                if (value is None and point[key] is not None) or (
                    value is not None and not np.isclose(value, point[key])
                ):
                    raise ValueError(f"Learning timeline mismatch: {run}/{step}/{key}")
            node = {**identity, "step": step}
            learning_metrics_rows.append({**node, **measured})
            learning_facts.extend(learning_fact_rows(node, low, high, phase, arrays))
            for chain in range(2):
                rows = chain_rows(node, low, high, chain, phase, arrays, "direct")
                learning_rows.extend(rows)
                if step == 15360:
                    row = next(
                        r
                        for r in rows
                        if r["cohort_kind"] == "fixed"
                        and r["cohort"] == "original_exception"
                        and r["split"] == "heldout"
                    )
                    learning_endpoint.append(
                        {
                            **{k: identity[k] for k in ("world", "seed", "condition")},
                            "chain": CHAINS[chain],
                            "prevalence": phase,
                            "step": 15360,
                            "cohort": "original_exception",
                            "split": "heldout",
                            "n": row["n"],
                            "correct": row["correct"],
                        }
                    )
            learning_count += 1
            if step == 15360:
                terminal = arrays
        for chain, label in enumerate(CHAINS):
            a = score_two_step(
                sources.arrays(run / "learning" / f"two-step-{label}.npz"), world, chain, terminal
            )
            two_rows.extend(
                chain_rows({**identity, "step": 15360}, low, high, chain, phase, a, "two_step")
            )
            two_count += 1
            for support in (0, 1):
                pair = pairs[w, chain, support]
                e_choices, r_choices = sampling_stream(w, chain)
                for kind in KINDS:
                    dest = run / "edits" / f"support-{support}" / f"{label}-{kind}-mlp"
                    done = sources.json(dest / "complete.json")
                    expected_done = dict(
                        status="complete", phase=phase, chain=label, kind=kind, scope="mlp"
                    )
                    if any(done[k] != v for k, v in expected_done.items()):
                        raise ValueError("Edit completion identity differs")
                    if sources.json(dest / "data-contract.json") != pair["contract"]:
                        raise ValueError("Confirmation support data contract differs")
                    saved_sets = sources.arrays(dest / "sets.npz")
                    target_sets = {k: v for k, v in pair.items() if isinstance(v, np.ndarray)}
                    target_sets.update(
                        old_correct=terminal["correct"],
                        edit_sampling=e_choices,
                        replay_sampling=r_choices,
                    )
                    if set(saved_sets) != set(target_sets):
                        raise ValueError("Edit support array keys differ")
                    for key, value in target_sets.items():
                        np.testing.assert_array_equal(saved_sets[key], value)
                    trajectory = sources.json(dest / "trajectory.json")
                    if [p["step"] for p in trajectory] != list(CHECKPOINTS):
                        raise ValueError("Editing checkpoint grid differs")
                    for point in trajectory:
                        step = point["step"]
                        arrays = score_arrays(
                            sources.arrays(dest / f"predictions-{step}.npz"),
                            pair[f"{phase}_{kind}"],
                        )
                        if step == 0:
                            for key in ("prediction", "ended"):
                                np.testing.assert_array_equal(arrays[key], terminal[key])
                        measured = edit_metrics(pair, phase, kind, arrays, terminal["correct"])
                        if any(point[key] != value for key, value in measured.items()):
                            raise ValueError("Stored edit metrics do not reproduce")
                        row = {
                            **identity,
                            "chain": label,
                            "support": support,
                            "kind": kind,
                            "step": step,
                            **flatten(measured),
                        }
                        edit_rows.append(row)
                        edit_checkpoints += 1
                        if step == 512:
                            if done["final"] != point:
                                raise ValueError("Edit completion / terminal point mismatch")
                            primary = measured["paired_reference_D_heldout"]
                            editing_endpoint.append(
                                {
                                    **{
                                        k: row[k]
                                        for k in (
                                            "world",
                                            "seed",
                                            "condition",
                                            "chain",
                                            "support",
                                            "kind",
                                            "step",
                                        )
                                    },
                                    "prevalence": phase,
                                    "n": primary["n"],
                                    "correct": primary["correct"],
                                    "cohort": "paired_reference",
                                    "split": "heldout",
                                    **flatten(measured),
                                }
                            )
                    edit_count += 1
    validate_endpoint_rows(learning_endpoint, editing_endpoint)
    if (learning_count, two_count, edit_count, edit_checkpoints) != (576, 192, 768, 3072):
        raise ValueError("Full learning/editing checkpoint counts differ")
    aggregates = world_aggregates(learning_endpoint, editing_endpoint)
    sources.recheck()
    tables = {
        "learning-strata.csv": learning_rows,
        "learning-facts.csv": learning_facts,
        "learning-metrics.csv": learning_metrics_rows,
        "two-step-strata.csv": two_rows,
        "editing-all-nodes.csv": edit_rows,
        "learning-endpoint.csv": learning_endpoint,
        "editing-endpoint.csv": editing_endpoint,
        "world-descriptive.csv": aggregates,
    }
    for name, rows in tables.items():
        write_csv(output / name, rows)
    audit = dict(
        complete=True,
        expected_models=96,
        complete_models=96,
        learning_checkpoints=learning_count,
        two_step_chains=two_count,
        edit_cases=edit_count,
        edit_checkpoints=edit_checkpoints,
        missing=[],
        model_weights_read=False,
        weight_archive_verified=True,
        archive_index_sha256=sha(archive_index_path),
        reserved_worlds_generated=True,
        outputs_sha256={name: sha(output / name) for name in tables},
        lock_sha256=lock_hash,
        config_sha256=config_hash,
        physical_source_root=str(source),
        script_sha256=sha(Path(__file__)),
        world_aggregate_rows=len(aggregates),
        weights_status="All producer weight seals match the independently verified full archive; "
        "current source sizes checked, no repeated torch.load",
        support_sampling="Shared original world932chain stream; supports are nested, "
        "not independent worlds",
        root_measure="Person-weighted corresponding root; never a truth bridge "
        "for two-step inference",
    )
    write_json(output / "declared-weights.json", declared_weights)
    write_json(output / "sources.json", sources.hashes)
    write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-index", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                args.config,
                args.lock,
                args.source,
                args.output,
                args.archive_index,
                args.require_complete,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
