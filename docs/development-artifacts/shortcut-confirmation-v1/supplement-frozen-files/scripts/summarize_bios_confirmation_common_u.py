"""Supplemental paired U audit, frozen before confirmation effect inspection.

This never trains, reads weights, chooses checkpoints, or adds hypothesis tests.
It requires the complete frozen primary audit and independently verified archive
before loading any confirmation world. The phase-specific primary U is unchanged.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import summarize_bios_shortcut_confirmation as primary

from llm_memory_editability.bios_shortcut_matched_edit import U_STRATA

PROTOCOL = "confirmation-common-U-supplement-v1"
CONTROLS = ("common_known", "same_truth_common_known")
POOLS = ("U_full", "U_unseen", "U_heldout")
SUPPLEMENT_FILES = (
    "scripts/summarize_bios_confirmation_common_u.py",
    "tests/test_bios_confirmation_common_u.py",
)


def supplemental_sources():
    return {name: primary.sha(primary.ROOT / name) for name in SUPPLEMENT_FILES}


def freeze_receipt(config_path, lock_path, receipt_path):
    """Record code identity only; do not inspect any model, result, or new world."""
    sources = primary.Sources()
    primary.validate_lock(config_path, lock_path, sources)
    receipt = {
        "protocol": PROTOCOL,
        "status": "frozen_before_confirmation_effect_inspection",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": primary.sha(config_path),
        "lock_sha256": primary.sha(lock_path),
        "supplemental_sources": supplemental_sources(),
        "scope": "Secondary descriptive common-known and unchanged-truth U only; "
        "no primary endpoint, estimator, hypothesis, model, support, or step changes",
        "checkpoints": list(primary.CHECKPOINTS),
        "controls": list(CONTROLS),
        "pools": list(POOLS),
        "baseline": "Each phase parent15360; both baseline correct with value and EOS",
        "unchanged_truth": "Both edit targets equal their immutable original truths on U; "
        "same_truth_common_known additionally requires low/high original truth equality",
        "unseen": "Intersection of full U, excluding both phases E/D/R",
        "zero_denominator": "None, never zero damage; count invalid cases explicitly",
    }
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def validate_receipt(config_path, lock_path, receipt_path, sources):
    receipt = sources.json(receipt_path)
    expected = {
        "protocol": PROTOCOL,
        "status": "frozen_before_confirmation_effect_inspection",
        "config_sha256": primary.sha(config_path),
        "lock_sha256": primary.sha(lock_path),
        "supplemental_sources": supplemental_sources(),
        "checkpoints": list(primary.CHECKPOINTS),
        "controls": list(CONTROLS),
        "pools": list(POOLS),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Supplemental pre-inspection receipt or source identity changed")
    for name in SUPPLEMENT_FILES:
        sources.bytes(primary.ROOT / name)
    return receipt


def checked_predictions(arrays, truth):
    truth = np.asarray(truth)
    if any(
        np.asarray(arrays[key]).shape != truth.shape for key in ("prediction", "ended", "correct")
    ):
        raise ValueError("Prediction shape differs from immutable truth")
    if arrays["ended"].dtype != np.bool_ or arrays["correct"].dtype != np.bool_:
        raise ValueError("Prediction masks must be Boolean")
    correct = (arrays["prediction"] == truth) & arrays["ended"]
    np.testing.assert_array_equal(correct, arrays["correct"])
    return correct


def checked_ids(value, n):
    ids = np.asarray(value)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
        or np.any((ids < 0) | (ids >= n))
    ):
        raise ValueError("Invalid or duplicated query IDs")
    return ids


def case_rows(identity, truths, sets, baselines, predictions):
    """Score one low/high pair without trusting stored correctness or pool labels."""
    kind, n = identity["kind"], len(truths["low"])
    if len(truths["high"]) != n:
        raise ValueError("Low/high query layouts differ")
    old, correct, phase_pools = {}, {}, {}
    excluded, replay = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    for phase in primary.PHASES:
        saved = sets[phase]
        old[phase] = checked_predictions(baselines[phase], truths[phase])
        np.testing.assert_array_equal(saved["old_correct"], old[phase])
        e, d, r, u, held = (
            checked_ids(saved[key], n) for key in ("E", "D", "R", "U_full", "U_heldout")
        )
        changed = np.union1d(e, d)
        np.testing.assert_array_equal(np.sort(u), np.setdiff1d(np.arange(n), changed))
        if not np.isin(held, u).all() or np.intersect1d(held, r).size:
            raise ValueError("Heldout U must be unchanged and excluded from replay")
        target = saved[f"{phase}_{kind}"]
        if np.asarray(target).shape != (n,):
            raise ValueError("Editing truth shape differs")
        np.testing.assert_array_equal(target[u], truths[phase][u])
        correct[phase] = checked_predictions(predictions[phase], target)
        phase_pools[phase] = {"U_full": u, "U_heldout": held}
        excluded, replay = np.union1d(excluded, changed), np.union1d(replay, r)
    np.testing.assert_array_equal(sets["low"]["U_strata"], sets["high"]["U_strata"])
    strata = sets["low"]["U_strata"]
    if np.asarray(strata).shape != (n,):
        raise ValueError("Local U strata shape differs")
    full = np.setdiff1d(
        np.intersect1d(phase_pools["low"]["U_full"], phase_pools["high"]["U_full"]), excluded
    )
    pools = {
        "U_full": full,
        "U_unseen": np.setdiff1d(full, replay),
        "U_heldout": np.intersect1d(
            full, np.intersect1d(phase_pools["low"]["U_heldout"], phase_pools["high"]["U_heldout"])
        ),
    }
    both_known, same_truth = old["low"] & old["high"], truths["low"] == truths["high"]
    rows = []
    for pool, common_ids in pools.items():
        for stratum in (-1, 0, 1, 2, 3, 4):
            ids = common_ids if stratum == -1 else common_ids[strata[common_ids] == stratum]
            for control in CONTROLS:
                eligible = ids if control == "common_known" else ids[same_truth[ids]]
                known = eligible[both_known[eligible]]
                row = {
                    **identity,
                    "pool": pool,
                    "stratum": stratum,
                    "stratum_name": "all" if stratum == -1 else U_STRATA[str(stratum)],
                    "control": control,
                    "pool_n": len(ids),
                    "eligible_n": len(eligible),
                    "known": len(known),
                    "joint_known_coverage": len(known) / len(eligible) if len(eligible) else None,
                    "joint_known_fraction_of_pool": len(known) / len(ids) if len(ids) else None,
                }
                for phase in primary.PHASES:
                    broken = int((~correct[phase][known]).sum())
                    row[f"{phase}_baseline_known"] = int(old[phase][eligible].sum())
                    row[f"{phase}_broken"] = broken
                    row[f"{phase}_damage"] = broken / len(known) if len(known) else None
                row["high_minus_low_damage"] = (
                    row["high_damage"] - row["low_damage"] if len(known) else None
                )
                rows.append(row)
    return rows


def aggregate_rows(rows, by_world):
    keys = ("world",) if by_world else ()
    keys += ("kind", "step", "pool", "stratum", "stratum_name", "control")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    counts = (
        "pool_n",
        "eligible_n",
        "known",
        "low_baseline_known",
        "high_baseline_known",
        "low_broken",
        "high_broken",
    )
    rates = (
        "joint_known_coverage",
        "joint_known_fraction_of_pool",
        "low_damage",
        "high_damage",
        "high_minus_low_damage",
    )
    for key, group in sorted(groups.items()):
        row = {**dict(zip(keys, key, strict=True)), "paired_cases": len(group)}
        row.update({name: sum(item[name] for item in group) for name in counts})
        for name in rates:
            valid = [item[name] for item in group if item[name] is not None]
            row[f"{name}_case_macro"] = float(np.mean(valid)) if valid else None
            row[f"{name}_valid_cases"] = len(valid)
            if not by_world:
                world_values = []
                for world in sorted({item["world"] for item in group}):
                    values = [
                        item[name]
                        for item in group
                        if item["world"] == world and item[name] is not None
                    ]
                    if values:
                        world_values.append(float(np.mean(values)))
                row[f"{name}_world_macro"] = float(np.mean(world_values)) if world_values else None
                row[f"{name}_valid_worlds"] = len(world_values)
        for phase in primary.PHASES:
            row[f"{phase}_damage_pooled"] = (
                row[f"{phase}_broken"] / row["known"] if row["known"] else None
            )
        row["high_minus_low_damage_pooled"] = (
            (row["high_broken"] - row["low_broken"]) / row["known"] if row["known"] else None
        )
        row["joint_known_coverage_pooled"] = (
            row["known"] / row["eligible_n"] if row["eligible_n"] else None
        )
        output.append(row)
    return output


def audit_primary_summary(summary_dir, lock_path, config_path, source, archive_path, sources):
    audit = sources.json(summary_dir / "audit.json")
    expected = dict(
        complete=True,
        expected_models=96,
        complete_models=96,
        learning_checkpoints=576,
        two_step_chains=192,
        edit_cases=768,
        edit_checkpoints=3072,
        missing=[],
        weight_archive_verified=True,
        lock_sha256=primary.sha(lock_path),
        config_sha256=primary.sha(config_path),
        archive_index_sha256=primary.sha(archive_path),
        physical_source_root=str(source),
        script_sha256=primary.sha(primary.__file__),
    )
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Complete frozen primary summary with matching archive is required")
    names = {
        "learning-strata.csv",
        "learning-facts.csv",
        "learning-metrics.csv",
        "two-step-strata.csv",
        "editing-all-nodes.csv",
        "learning-endpoint.csv",
        "editing-endpoint.csv",
        "world-descriptive.csv",
    }
    if set(audit.get("outputs_sha256", {})) != names:
        raise ValueError("Primary summary output manifest differs")
    for name, expected_hash in audit["outputs_sha256"].items():
        sources.bytes(summary_dir / name)
        if sources.hashes[str((summary_dir / name).resolve())] != expected_hash:
            raise ValueError("Primary summary output changed")
    recorded = sources.json(summary_dir / "sources.json")
    for path, expected_hash in recorded.items():
        sources.bytes(path)
        if sources.hashes[str(Path(path).resolve())] != expected_hash:
            raise ValueError("Primary summary raw source changed")
    return recorded


def summarize(config_path, lock_path, source, summary_dir, archive_path, receipt_path, output):
    sources = primary.Sources()
    config, lock = primary.validate_lock(config_path, lock_path, sources)
    validate_receipt(config_path, lock_path, receipt_path, sources)
    source = Path(source or lock["output_root"]).resolve()
    summary_dir, output = Path(summary_dir).resolve(), Path(output).resolve()
    if source != Path(lock["output_root"]).resolve():
        raise ValueError("Source differs from locked output root")
    if any(
        output == p or output in p.parents or p in output.parents for p in (source, summary_dir)
    ):
        raise ValueError("Supplement output must not overlap raw or primary summary trees")
    expected = primary.matrix()
    if len(expected) != 96 or any(
        not (primary.run_path(source, item) / "complete.json").is_file() for item in expected
    ):
        raise ValueError("All 96 completed parents are required before supplemental analysis")
    archive = primary.audit_archive(archive_path, lock, sources)
    recorded = audit_primary_summary(
        summary_dir, lock_path, config_path, source, archive_path, sources
    )
    prepared = {}
    for w in primary.WORLDS:
        base, receipt = primary.validate_prepared_world(config, config_path, lock_path, w)
        sources.json(base / "preparation-audit.json")
        files = {
            name: checksum
            for name, checksum in receipt["files_sha256"].items()
            if name.startswith(f"world-{w}/")
        }
        for name in files:
            sources.bytes(base / name)
        prepared[w] = (
            base,
            {
                **lock["environment"],
                "device_type": "cuda",
                "precision": "BF16 autocast; FP32 model and optimizer",
                "prepared_worlds_audit_sha256": primary.sha(base / "preparation-audit.json"),
                "world_files_sha256": files,
            },
        )
    weights = []
    for identity in expected:
        launch, declared = primary.audit_completion(
            primary.run_path(source, identity),
            identity,
            primary.sha(lock_path),
            sources,
            archive,
            source,
            prepared[identity["world"]][1],
        )
        if launch["config_sha256"] != primary.sha(config_path):
            raise ValueError("Producer configuration changed")
        weights.extend(declared)
    if len(weights) != 2208:
        raise ValueError("All 2208 archived weights are required")
    # Only now may confirmation world truths and paired predictions be materialized.
    rows, pairs_count = [], 0
    for w in primary.WORLDS:
        low = primary.make_cross_world(w, source_root=prepared[w][0])
        high, *_ = primary.high_exception_world(low)
        truths = {"low": low.answers, "high": high.answers}
        pairs = {
            (chain, support): primary.make_confirmation_edit_pair(low, high, chain, support)
            for chain in (0, 1)
            for support in (0, 1)
        }
        for seed in (0, 1):
            for condition in primary.CONDITIONS:
                paths = {
                    phase: primary.run_path(
                        source, dict(phase=phase, world=w, seed=seed, condition=condition)
                    )
                    for phase in primary.PHASES
                }
                baseline = {
                    phase: sources.arrays(path / "learning/predictions-15360.npz")
                    for phase, path in paths.items()
                }
                for chain, label in enumerate(primary.CHAINS):
                    for support in (0, 1):
                        pair = pairs[chain, support]
                        for kind in primary.KINDS:
                            dest = {
                                phase: path / f"edits/support-{support}/{label}-{kind}-mlp"
                                for phase, path in paths.items()
                            }
                            saved = {
                                phase: sources.arrays(path / "sets.npz")
                                for phase, path in dest.items()
                            }
                            for phase in primary.PHASES:
                                if (
                                    sources.json(dest[phase] / "data-contract.json")
                                    != pair["contract"]
                                ):
                                    raise ValueError("Support data contract changed")
                                for key, value in pair.items():
                                    if isinstance(value, np.ndarray):
                                        np.testing.assert_array_equal(saved[phase][key], value)
                            for step in primary.CHECKPOINTS:
                                predictions = {
                                    phase: sources.arrays(path / f"predictions-{step}.npz")
                                    for phase, path in dest.items()
                                }
                                if step == 0:
                                    for phase in primary.PHASES:
                                        for key in ("prediction", "ended"):
                                            np.testing.assert_array_equal(
                                                predictions[phase][key], baseline[phase][key]
                                            )
                                identity = dict(
                                    world=w,
                                    seed=seed,
                                    condition=condition,
                                    chain=label,
                                    support=support,
                                    kind=kind,
                                    step=step,
                                )
                                rows.extend(
                                    case_rows(identity, truths, saved, baseline, predictions)
                                )
                            pairs_count += 1
    if pairs_count != 384 or len(rows) != 55296:
        raise ValueError("Supplemental paired-case/checkpoint/strata matrix is incomplete")
    for path, checksum in sources.hashes.items():
        if source in Path(path).parents and recorded.get(path) != checksum:
            raise ValueError("Supplemental raw source is absent or different in primary audit")
    sources.recheck()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite supplemental audit results")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "paired-common-u.csv": rows,
        "world-common-u.csv": aggregate_rows(rows, True),
        "overall-common-u.csv": aggregate_rows(rows, False),
    }
    for name, values in tables.items():
        primary.write_csv(output / name, values)
    primary.write_json(output / "sources.json", sources.hashes)
    audit = dict(
        protocol=PROTOCOL,
        complete=True,
        parent_models=96,
        edit_cases=768,
        paired_cases=pairs_count,
        paired_checkpoints=pairs_count * 4,
        rows=len(rows),
        receipt_sha256=primary.sha(receipt_path),
        lock_sha256=primary.sha(lock_path),
        primary_audit_sha256=primary.sha(summary_dir / "audit.json"),
        archive_index_sha256=primary.sha(archive_path),
        declared_weights=len(weights),
        model_weights_read=False,
        weight_archive_verified=True,
        sources_sha256=primary.sha(output / "sources.json"),
        supplementary_sources=supplemental_sources(),
        outputs_sha256={name: primary.sha(output / name) for name in tables},
        inference="Descriptive only; repeated queries/supports/seeds are not independent worlds",
        limitation=(
            "Conditions on both baselines being correct after prevalence-specific learning; "
            "not a primary causal effect of prevalence on retention and does not replace "
            "phase-specific primary U. Case macros average only nonzero-known cases; world "
            "macros average available world case means, with valid counts reported. Archive "
            "byte verification reused via sealed independent index; weights not loaded again."
        ),
    )
    primary.write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--freeze-receipt", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--archive-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze_receipt:
        result = freeze_receipt(args.config, args.lock, args.receipt)
    else:
        if any(value is None for value in (args.summary, args.archive_index, args.output)):
            parser.error("Scoring requires --summary, --archive-index and --output")
        result = summarize(
            args.config,
            args.lock,
            args.source,
            args.summary,
            args.archive_index,
            args.receipt,
            args.output,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
