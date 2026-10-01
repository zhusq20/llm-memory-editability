"""Aggregate P0 two-step results and gate P4 replication behind a world-0 candidate lock.

Discovery mode never reads world 1 causal diagnostic files. A later lock operation
records one explicit candidate before replication mode may access world 1.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_causal_paths import PAIR_TYPES, make_donor_plan
from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_path_diagnostics import classify_answers, file_sha256

ROOT = Path(__file__).resolve().parents[1]
MECHANISM = ROOT / "results/bios-mechanism-dev-v1"
PRIMARY_POSITIONS = (2, 3)
DONOR_STRATA = (
    "default_only",
    "actual_only",
    "default_actual_overlap",
    "other",
    "termination_error",
)


def donor_prediction_strata(prediction, ended, candidates):
    """Describe clean donor outputs without dropping unsuccessful or ambiguous donors."""
    default_hit = prediction == candidates[:, 2]
    actual_hit = prediction == candidates[:, 3]
    result = np.full(len(prediction), "other", dtype="U32")
    result[default_hit & ~actual_hit] = "default_only"
    result[actual_hit & ~default_hit] = "actual_only"
    result[default_hit & actual_hit] = "default_actual_overlap"
    result[~ended] = "termination_error"
    return result


def verify_causal_producer(root, run, world, ledger):
    """Check the producer's existing seals and identity before accepting any outcomes.

    The checkpoint path, step and producer-recorded digest are checked without
    rereading weights. Output hashes must agree with the original producer seal,
    rather than merely creating a fresh seal over whichever files are present.
    """
    complete_path = root / "complete.json"
    ledger[str(complete_path)] = file_sha256(complete_path)
    complete = json.loads(complete_path.read_text())
    identity = complete["identity"]
    expected = {
        **{key: run[key] for key in ("width", "world", "seed", "condition")},
        "step": 15360,
        "run": str(run["source"].resolve()),
        "checkpoint": str((run["source"] / "model-15360.pt").resolve()),
        "per_population": 64,
        "positions": [2, 3, 4],
        "offline_truth_matching": True,
    }
    if (
        not complete["complete"]
        or complete["node_files"] != 144
        or any(identity.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"Causal producer identity mismatch: {root}")
    if len(identity.get("checkpoint_sha256", "")) != 64:
        raise ValueError("Missing producer checkpoint digest")
    producer_config = root / "config.json"
    ledger[str(producer_config)] = file_sha256(producer_config)
    if json.loads(producer_config.read_text()) != identity:
        raise ValueError("Producer config and completion identity differ")
    original_config = run["source"] / "config.json"
    ledger[str(original_config)] = file_sha256(original_config)
    if ledger[str(original_config)] != identity["config_sha256"]:
        raise ValueError("Original model configuration changed")
    config = json.loads(original_config.read_text())
    if any(config[key] != run[key] for key in ("world", "seed", "condition")) or (
        config["model"]["width"] != run["width"]
        or config["truth_sha256"] != array_hash(world.answers)
    ):
        raise ValueError("Original model/world identity mismatch")
    for filename, key in (
        ("bios_causal_paths.py", "diagnostic_source_sha256"),
        ("bios_path_diagnostics.py", "generation_helper_sha256"),
    ):
        code = ROOT / "src/llm_memory_editability" / filename
        ledger[str(code)] = file_sha256(code)
        if ledger[str(code)] != identity[key]:
            raise ValueError("Frozen producer source changed")
    artifacts_path = root / "artifacts.json"
    ledger[str(artifacts_path)] = file_sha256(artifacts_path)
    if ledger[str(artifacts_path)] != complete["artifacts_sha256"]:
        raise ValueError("Producer artifacts manifest changed")
    artifacts = json.loads(artifacts_path.read_text())
    expected_files = {
        f"{chain}/{filename}"
        for chain in CHAINS
        for filename in ["plan.npz", "clean-cache.npz"]
        + [
            f"layer-{layer}-position-{position}-{intervention}.npz"
            for layer in range(8)
            for position in (2, 3, 4)
            for intervention in ("donor", "matched_random", "sham")
        ]
    }
    if set(artifacts) != expected_files:
        raise ValueError("Producer artifact grid differs from the frozen diagnostic")
    return artifacts


def load_verified_causal_arrays(root, relative, artifacts, ledger, keys=None):
    path = root / relative
    digest = file_sha256(path)
    if artifacts[relative] != digest:
        raise ValueError(f"Producer artifact hash mismatch: {path}")
    ledger[str(path)] = digest
    with np.load(path) as saved:
        return {key: saved[key] for key in (keys if keys is not None else saved.files)}


def verify_causal_node(arrays, selected, world, chain, clean, intervention):
    """Rescore all labels independently of stored candidate/baseline fields."""
    for key, value in selected.items():
        if not np.array_equal(arrays[key], value):
            raise ValueError(f"Causal node differs from frozen donor plan: {key}")
    recipients, donors = selected["recipient"], selected["donor"]
    people = clean["people"]
    if not np.array_equal(people, np.unique(np.concatenate((recipients, donors)))):
        raise ValueError("Clean-cache person set differs from donor plan")
    ri = np.searchsorted(people, recipients)
    default = world.answers[world.root_ids[chain, world.memberships[chain]]]
    actual = world.answers[world.actual_ids[chain]]
    candidates = np.column_stack(
        (default[recipients], actual[recipients], default[donors], actual[donors])
    )
    expected = {
        "recipient_query": world.derived_ids[chain, recipients],
        "donor_query": world.derived_ids[chain, donors],
        "candidate_tokens": candidates,
        "baseline_prediction": clean["prediction"][ri],
        "baseline_ended": clean["ended"][ri],
        "baseline_correct": (clean["prediction"][ri] == default[recipients]) & clean["ended"][ri],
        "correct": (arrays["prediction"] == default[recipients]) & arrays["ended"],
    }
    classes, bits = classify_answers(arrays["prediction"], arrays["ended"], candidates)
    expected.update(class_code=classes, matched_role_bits=bits)
    if intervention == "sham":
        expected.update(prediction=clean["prediction"][ri], ended=clean["ended"][ri])
    for key, value in expected.items():
        if not np.array_equal(arrays[key], value):
            raise ValueError(f"Causal node scoring/sham mismatch: {key}")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_runs(widths=(64, 128, 256, 768), worlds=(0, 1)):
    manifest = json.loads((ROOT / "configs/bios-cross-scale-development-v1.json").read_text())
    for size in manifest["sizes"]:
        if size["width"] not in widths:
            continue
        for world in worlds:
            for seed in (0, 1):
                for condition in CONDITIONS:
                    name = f"world-{world}-seed-{seed}-{condition}"
                    yield {
                        "width": size["width"],
                        "world": world,
                        "seed": seed,
                        "condition": condition,
                        "run_name": name,
                        "source": ROOT / size["output"] / name,
                    }


def two_step_subsets(world, chain, case):
    people = np.arange(world.derived_ids.shape[1])
    query = world.derived_ids[chain]
    for split, ids in (
        ("all", query),
        ("trained", world.train_ids[chain]),
        ("heldout", world.heldout_ids[chain]),
    ):
        population = np.isin(query, ids)
        for label, mask in (
            ("all", np.ones(len(query), dtype=bool)),
            ("ordinary", ~world.exceptions[chain]),
            ("old_exception", world.exceptions[chain]),
        ):
            yield f"QA/{split}/{label}", people[population & mask]
    if case.startswith("learning-"):
        return
    edit_chain, kind, _ = case.split("-")
    if edit_chain != CHAINS[chain]:
        return
    pair = edit_pair(world, chain)
    target = pair[kind]
    conflict = target[query] != target[world.actual_ids[chain]]
    for split, ids in (
        ("all", pair["D"]),
        ("trained", np.intersect1d(pair["D"], world.train_ids[chain])),
        ("heldout", np.intersect1d(pair["D"], world.heldout_ids[chain])),
    ):
        selected = np.isin(query, ids)
        for label, mask in (
            ("all", np.ones(len(query), dtype=bool)),
            ("manipulated_conflict_cohort", np.isin(query, pair["conflict_D"])),
            ("factual_conflict", conflict),
            ("old_exception", world.exceptions[chain]),
        ):
            yield f"D/{split}/{label}", people[selected & mask]


def summarize_two_step(source, output, require_complete=False):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    worlds = {w: make_cross_world(w, ROOT / "data/bios-organization-v1") for w in (0, 1)}
    rows, missing, ledger = [], [], {}
    counts = {"models": 0, "states": 0, "chain_files": 0, "derived_queries": 0}
    for run in expected_runs():
        dest = source / f"width-{run['width']}" / run["run_name"]
        cases = [f"learning-{step}" for step in (5120, 10240, 15360)] + [
            f"{chain}-{kind}-{scope}"
            for chain in CHAINS
            for kind in ("coherent", "exception")
            for scope in ("mlp", "all")
        ]
        world = worlds[run["world"]]
        model_states = 0
        for case in cases:
            state = dest / case
            complete = state / "complete.json"
            if not complete.exists():
                missing.append(str(state))
                continue
            record = json.loads(complete.read_text())
            if not record["complete"] or record["oracle_bridging"]:
                raise ValueError(f"Invalid completion record: {complete}")
            ledger[str(complete)] = file_sha256(complete)
            if case.startswith("learning-"):
                target, phase, step = world.answers, "learning", int(case.split("-")[1])
                edit_chain, kind, scope = "", "", ""
            else:
                edit_chain, kind, scope = case.split("-")
                target = edit_pair(world, CHAINS.index(edit_chain))[kind]
                phase, step = "editing", 512
            for chain, name in enumerate(CHAINS):
                path = state / f"{name}.npz"
                ledger[str(path)] = file_sha256(path)
                with np.load(path) as data:
                    a = dict(data)
                query = world.derived_ids[chain]
                if not np.array_equal(a["query_id"], query):
                    raise ValueError(f"Incorrect query order: {path}")
                valid_bridge = a["bridge_ended"] & np.isin(
                    a["bridge_prediction"], world.prompts[world.root_ids[chain], 1]
                )
                correct_bridge = a["bridge_ended"] & (
                    a["bridge_prediction"] == target[world.membership_ids[chain]]
                )
                if not np.array_equal(valid_bridge, a["bridge_valid"]) or not np.array_equal(
                    correct_bridge, a["bridge_correct"]
                ):
                    raise ValueError(f"First-hop scoring mismatch: {path}")
                correct = (a["prediction"] == target[query]) & a["ended"] & a["bridge_valid"]
                direct = (a["direct_prediction"] == target[query]) & a["direct_ended"]
                if not np.array_equal(correct, a["correct"]) or not np.array_equal(
                    direct, a["direct_correct"]
                ):
                    raise ValueError(f"Two-step scoring mismatch: {path}")
                for subset, ids in two_step_subsets(world, chain, case):
                    before, after = direct[ids], correct[ids]
                    n = len(ids)
                    rows.append(
                        {
                            **{k: run[k] for k in ("width", "world", "seed", "condition")},
                            "phase": phase,
                            "step": step,
                            "edit_chain": edit_chain,
                            "kind": kind,
                            "scope": scope,
                            "test_chain": name,
                            "subset": subset,
                            "n": n,
                            "direct_correct": int(before.sum()),
                            "two_step_correct": int(after.sum()),
                            "direct_accuracy": float(before.mean()) if n else None,
                            "two_step_accuracy": float(after.mean()) if n else None,
                            "bridge_valid": int(a["bridge_valid"][ids].sum()),
                            "bridge_correct": int(a["bridge_correct"][ids].sum()),
                            "both_correct": int((before & after).sum()),
                            "direct_only": int((before & ~after).sum()),
                            "two_step_only": int((~before & after).sum()),
                            "both_wrong": int((~before & ~after).sum()),
                            "second_termination_error": int(
                                (a["bridge_valid"][ids] & ~a["ended"][ids]).sum()
                            ),
                            "invalid_bridge": int((~a["bridge_valid"][ids]).sum()),
                        }
                    )
                counts["chain_files"] += 1
                counts["derived_queries"] += len(query)
            counts["states"] += 1
            model_states += 1
        counts["models"] += model_states == len(cases)
    complete = not missing and counts["states"] == 528 and counts["chain_files"] == 1056
    if require_complete and not complete:
        raise ValueError(f"Incomplete two-step matrix: {counts}; missing {len(missing)} states")
    write_csv(output / "two-step-strata.csv", rows)
    write_json(output / "two-step-sources.json", ledger)
    write_json(
        output / "two-step-audit.json",
        {"complete": complete, **counts, "missing": missing, "errors": []},
    )
    lines = [
        "# 自主两步诊断",
        "",
        f"完成状态：{complete}；模型{counts['models']}/48，"
        f"状态{counts['states']}/528，链文件{counts['chain_files']}/1056。",
        "",
        "第一步自行生成组织，第二步使用该预测组织。真值桥接未用于模型输入。"
        "外部两次调用改善不证明原一次前向存在同一组合算法。",
        "",
        "| 宽度 | 编辑 | 案例 | 冲突留出直接答题 | 自主两步 | 首跳正确 | 无效首跳 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in rows:
        if (
            row["phase"] == "editing"
            and row["kind"] == "exception"
            and row["subset"] == "D/heldout/manipulated_conflict_cohort"
        ):
            groups[row["width"], row["scope"]].append(row)
    for (width, scope), group in sorted(groups.items()):
        values = [
            np.mean([r[key] / r["n"] for r in group])
            for key in ("direct_correct", "two_step_correct", "bridge_correct", "invalid_bridge")
        ]
        lines.append(
            f"| {width} | {scope} | {len(group)} | "
            + " | ".join(f"{100 * value:.2f}%" for value in values)
            + " |"
        )
    lines += ["", "案例等权平均；两个世界是开发数据。缺失状态不作零分填补。", ""]
    (output / "two-step-report.md").write_text("\n".join(lines))
    return counts


def causal_effect_rows(source, world, candidate=None):
    """Read only the requested world and optionally only the frozen candidate."""
    rows, ledger, missing, donor_rows = [], {}, [], []
    truth = make_cross_world(world, ROOT / "data/bios-organization-v1")
    for run in expected_runs(widths=(256, 768), worlds=(world,)):
        root = Path(source) / f"width-{run['width']}" / run["run_name"]
        complete = root / "complete.json"
        if not complete.exists():
            missing.append(str(root))
            continue
        artifacts = verify_causal_producer(root, run, truth, ledger)
        for chain_index, chain in enumerate(CHAINS):
            plan = load_verified_causal_arrays(root, f"{chain}/plan.npz", artifacts, ledger)
            expected_plan = make_donor_plan(truth, chain_index, 64)
            if set(plan) != set(expected_plan) or any(
                not np.array_equal(plan[key], value) for key, value in expected_plan.items()
            ):
                raise ValueError("Producer donor plan differs from the frozen world and RNG")
            valid = plan["donor"] >= 0
            selected = {key: value[valid] for key, value in plan.items()}
            clean = load_verified_causal_arrays(
                root,
                f"{chain}/clean-cache.npz",
                artifacts,
                ledger,
                keys=("people", "prediction", "ended"),
            )
            clean_people, clean_prediction, clean_ended = (
                clean[key] for key in ("people", "prediction", "ended")
            )
            for layer in range(8):
                for position in (2, 3, 4):
                    if candidate and (layer, position) != (
                        candidate["layer"],
                        candidate["position"],
                    ):
                        continue
                    arrays = {}
                    for intervention in ("donor", "matched_random", "sham"):
                        relative = f"{chain}/layer-{layer}-position-{position}-{intervention}.npz"
                        arrays[intervention] = load_verified_causal_arrays(
                            root, relative, artifacts, ledger
                        )
                        verify_causal_node(
                            arrays[intervention], selected, truth, chain_index, clean, intervention
                        )
                    d, random, sham = (arrays[k] for k in ("donor", "matched_random", "sham"))
                    donor_index = np.searchsorted(clean_people, d["donor"])
                    if not np.array_equal(clean_people[donor_index], d["donor"]):
                        raise ValueError("Clean cache is missing donor predictions")
                    donor_strata = donor_prediction_strata(
                        clean_prediction[donor_index],
                        clean_ended[donor_index],
                        d["candidate_tokens"],
                    )
                    for pair_type, pair_label in enumerate(PAIR_TYPES):
                        for donor_stratum in ("all", *DONOR_STRATA):
                            included = d["pair_type"] == pair_type
                            if donor_stratum != "all":
                                included &= donor_strata == donor_stratum
                            row = {
                                **{k: run[k] for k in ("width", "world", "seed", "condition")},
                                "chain": chain,
                                "layer": layer,
                                "position": position,
                                "terminal_control": layer == 7,
                                "answer_state_transfer": position == 4,
                                "pair_type": pair_label,
                                "donor_prediction": donor_stratum,
                                "requested_pairs": int((plan["pair_type"] == pair_type).sum()),
                                "valid_pairs": int((d["pair_type"] == pair_type).sum()),
                                "missing_pairs": int(
                                    ((plan["pair_type"] == pair_type) & ~valid).sum()
                                ),
                                "n": int(included.sum()),
                                "recipient_baseline_correct": int(
                                    d["baseline_correct"][included].sum()
                                ),
                                "recipient_old_exception": int(
                                    d["recipient_old_exception"][included].sum()
                                ),
                            }
                            for intervention, a in arrays.items():
                                row[f"{intervention}_correct"] = int(a["correct"][included].sum())
                                row[f"{intervention}_termination_error"] = int(
                                    (~a["ended"][included]).sum()
                                )
                                for role, label in enumerate(
                                    (
                                        "recipient_default",
                                        "recipient_actual",
                                        "donor_default",
                                        "donor_actual",
                                    )
                                ):
                                    row[f"{intervention}_matches_{label}"] = int(
                                        (
                                            (a["prediction"] == d["candidate_tokens"][:, role])
                                            & a["ended"]
                                            & included
                                        ).sum()
                                    )
                            donor_rows.append(row)
                    for comparison in (random, sham):
                        for key in (
                            "case_id",
                            "recipient",
                            "donor",
                            "candidate_tokens",
                            "baseline_correct",
                        ):
                            if not np.array_equal(d[key], comparison[key]):
                                raise ValueError("Paired intervention metadata differ")
                    for mechanism, pair_type, answer_role in (
                        ("actual_path", 0, 3),
                        ("default_path", 1, 2),
                    ):
                        if candidate and mechanism != candidate["mechanism"]:
                            continue
                        target = d["candidate_tokens"][:, answer_role]
                        distinct = (target != d["candidate_tokens"][:, 0]) & (
                            target != d["candidate_tokens"][:, 1]
                        )
                        for known_only in (False, True):
                            eligible = (d["pair_type"] == pair_type) & distinct
                            if known_only:
                                eligible &= d["baseline_correct"]
                            n = int(eligible.sum())
                            hits = {
                                name: (a["prediction"] == target) & a["ended"]
                                for name, a in arrays.items()
                            }
                            rows.append(
                                {
                                    **{k: run[k] for k in ("width", "world", "seed", "condition")},
                                    "chain": chain,
                                    "layer": layer,
                                    "position": position,
                                    "terminal_control": layer == 7,
                                    "mechanism": mechanism,
                                    "pair_type": PAIR_TYPES[pair_type],
                                    "known_only": known_only,
                                    "requested_pairs": int((plan["pair_type"] == pair_type).sum()),
                                    "valid_pairs": int((d["pair_type"] == pair_type).sum()),
                                    "missing_pairs": int(
                                        ((plan["pair_type"] == pair_type) & ~valid).sum()
                                    ),
                                    "answer_distinct_pairs": int(
                                        ((d["pair_type"] == pair_type) & distinct).sum()
                                    ),
                                    "n": n,
                                    "donor_hit": int(hits["donor"][eligible].sum()),
                                    "random_hit": int(hits["matched_random"][eligible].sum()),
                                    "sham_hit": int(hits["sham"][eligible].sum()),
                                    "donor_minus_random": float(
                                        (
                                            hits["donor"][eligible].astype(float)
                                            - hits["matched_random"][eligible]
                                        ).mean()
                                    )
                                    if n
                                    else None,
                                    "donor_minus_sham": float(
                                        (
                                            hits["donor"][eligible].astype(float)
                                            - hits["sham"][eligible]
                                        ).mean()
                                    )
                                    if n
                                    else None,
                                }
                            )
    return rows, ledger, missing, donor_rows


def rank_candidates(rows, known_only=False):
    groups = defaultdict(list)
    for row in rows:
        if (
            row["layer"] < 7
            and row["position"] in PRIMARY_POSITIONS
            and row["known_only"] == known_only
        ):
            groups[row["mechanism"], row["layer"], row["position"]].append(row)
    ranks = []
    for (mechanism, layer, position), group in groups.items():
        available = [r for r in group if r["n"]]
        ranks.append(
            {
                "mechanism": mechanism,
                "layer": layer,
                "position": position,
                "known_only": known_only,
                "available_model_chain_blocks": len(available),
                "expected_model_chain_blocks": 24,
                "query_count": sum(r["n"] for r in group),
                "score": float(np.mean([r["donor_minus_random"] for r in available]))
                if available
                else None,
                "donor_minus_sham": float(np.mean([r["donor_minus_sham"] for r in available]))
                if available
                else None,
            }
        )
    ranked = []
    for mechanism in ("actual_path", "default_path"):
        candidates = sorted(
            (r for r in ranks if r["mechanism"] == mechanism),
            key=lambda r: (
                r["score"] is None,
                -r["score"] if r["score"] is not None else 0,
                r["layer"],
                r["position"],
            ),
        )
        ranked.extend({**row, "rank": rank} for rank, row in enumerate(candidates, 1))
    return ranked


def discovery(source, output, require_complete=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, ledger, missing, donor_rows = causal_effect_rows(source, world=0)
    if require_complete and missing:
        raise ValueError(f"Missing world-0 models: {missing}")
    write_csv(output / "world-0-effects.csv", rows)
    write_csv(output / "world-0-donor-prediction-strata.csv", donor_rows)
    ranked = rank_candidates(rows, known_only=False)
    write_csv(output / "world-0-candidate-ranks.csv", ranked)
    write_csv(output / "world-0-known-only-diagnostic-ranks.csv", rank_candidates(rows, True))
    write_json(output / "world-0-sources.json", ledger)
    write_json(
        output / "world-0-audit.json",
        {
            "complete": not missing and sum(Path(p).name == "complete.json" for p in ledger) == 12,
            "worlds_read": [0],
            "primary_positions": list(PRIMARY_POSITIONS),
            "primary_known_only": False,
            "analysis_revision": "Before inspecting discovery effect/rank values "
            "or locking a candidate, "
            "restrict relation-routing candidates to positions 2/3 based on causal time order. "
            "Position 4 may transfer an already-formed answer even before the last layer; "
            "retain all position-4 outputs as answer-state-transfer diagnostics. "
            "Donor clean-output strata include incorrect, ambiguous and nonterminated donors; "
            "no donor-output based sample deletion or ranking filter. "
            "Before viewing effects, primary ranking also changed to all answer-distinct "
            "matched recipients; baseline-correct ranking is conditional diagnostic only.",
            "missing": missing,
            "ranking": "Separate hypotheses; equal model-chain mean of donor-minus-"
            "matched-random target hits, among all answer-distinct matched cases. "
            "Positions 2/3 only; final-layer controls excluded. "
            "Missing denominators never imputed.",
            "source_manifest_sha256": file_sha256(output / "world-0-sources.json"),
            "analysis_script_sha256": file_sha256(__file__),
            "producer_audit": "Original complete/config identity, producer artifacts manifest "
            "and original hashes verified; frozen pairing, candidate labels, baseline correctness, "
            "full value/EOS scoring and sham outputs independently checked. "
            "Checkpoint identity/digest checked against producer record without rereading weights.",
        },
    )


def lock_candidate(discovery_dir, path, mechanism, layer, position, rationale):
    discovery_dir, path = Path(discovery_dir), Path(path)
    audit = json.loads((discovery_dir / "world-0-audit.json").read_text())
    if not audit["complete"] or audit["worlds_read"] != [0]:
        raise ValueError("Candidate lock requires complete world-0-only discovery")
    if position not in PRIMARY_POSITIONS or audit.get("primary_positions") != list(
        PRIMARY_POSITIONS
    ):
        raise ValueError("Relation-routing candidates require positions 2/3 under revised analysis")
    if audit.get("primary_known_only") is not False:
        raise ValueError("Primary ranking must use all answer-distinct matched recipients")
    ranks_path = discovery_dir / "world-0-candidate-ranks.csv"
    with ranks_path.open() as stream:
        selected = [
            r
            for r in csv.DictReader(stream)
            if r["mechanism"] == mechanism
            and int(r["layer"]) == layer
            and int(r["position"]) == position
        ]
    if (
        len(selected) != 1
        or not selected[0]["score"]
        or not np.isfinite(float(selected[0]["score"]))
    ):
        raise ValueError("Candidate has no finite world-0 evidence")
    verify_source_ledger(discovery_dir / "world-0-sources.json")
    candidate = {
        "mechanism": mechanism,
        "layer": layer,
        "position": position,
        "known_only": False,
        "rationale": rationale,
        "discovery_dir": str(discovery_dir.resolve()),
        "discovery_ranks_sha256": file_sha256(ranks_path),
        "discovery_sources_sha256": file_sha256(discovery_dir / "world-0-sources.json"),
        "discovery_audit_sha256": file_sha256(discovery_dir / "world-0-audit.json"),
        "selected_world0_row": selected[0],
    }
    if path.exists():
        previous = json.loads(path.read_text())
        previous.pop("locked_at_utc", None)
        if previous != candidate:
            raise ValueError("Cannot overwrite an existing candidate lock")
        return
    # Exclusive creation prevents silently replacing an earlier selection.
    candidate["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(candidate, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def verify_source_ledger(path):
    for source, digest in json.loads(Path(path).read_text()).items():
        if file_sha256(source) != digest:
            raise ValueError(f"World-0 source changed after discovery: {source}")


def replication(source, output, lock, require_complete=False):
    output, lock = Path(output), Path(lock)
    candidate = json.loads(lock.read_text())
    discovery_dir = Path(candidate["discovery_dir"])
    for filename, key in (
        ("world-0-candidate-ranks.csv", "discovery_ranks_sha256"),
        ("world-0-sources.json", "discovery_sources_sha256"),
        ("world-0-audit.json", "discovery_audit_sha256"),
    ):
        if file_sha256(discovery_dir / filename) != candidate[key]:
            raise ValueError("World-0 discovery changed after candidate lock")
    verify_source_ledger(discovery_dir / "world-0-sources.json")
    if candidate["position"] not in PRIMARY_POSITIONS:
        raise ValueError("Locked routing candidate cannot be the answer-marker state")
    rows, ledger, missing, donor_rows = causal_effect_rows(source, world=1, candidate=candidate)
    if require_complete and missing:
        raise ValueError(f"Missing world-1 models: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "world-1-locked-candidate-effects.csv", rows)
    write_csv(output / "world-1-locked-candidate-donor-strata.csv", donor_rows)
    write_json(output / "world-1-sources.json", ledger)
    primary = [r for r in rows if not r["known_only"] and r["n"]]
    known = [r for r in rows if r["known_only"] and r["n"]]
    write_json(
        output / "world-1-audit.json",
        {
            "complete": not missing and sum(Path(p).name == "complete.json" for p in ledger) == 12,
            "worlds_read": [1],
            "candidate_lock": str(lock.resolve()),
            "candidate_lock_sha256": file_sha256(lock),
            "candidate": candidate,
            "missing": missing,
            "available_model_chain_blocks": len(primary),
            "mean_donor_minus_random": float(np.mean([r["donor_minus_random"] for r in primary]))
            if primary
            else None,
            "mean_donor_minus_sham": float(np.mean([r["donor_minus_sham"] for r in primary]))
            if primary
            else None,
            "known_only_diagnostic_mean_donor_minus_random": float(
                np.mean([r["donor_minus_random"] for r in known])
            )
            if known
            else None,
            "interpretation": "World 1 is held back from selection, but is an already-used "
            "development world. This is replication, not new-world confirmation.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    p0 = subparsers.add_parser("two-step")
    p0.add_argument("--source", type=Path, default=MECHANISM / "p0/two-step")
    p0.add_argument("--output", type=Path, default=MECHANISM / "p0/summary")
    p0.add_argument("--require-complete", action="store_true")
    for name in ("discovery", "replication"):
        command = subparsers.add_parser(name)
        command.add_argument("--source", type=Path, default=MECHANISM / "p4/localization")
        command.add_argument("--output", type=Path, default=MECHANISM / f"p4/{name}")
        command.add_argument("--require-complete", action="store_true")
        if name == "replication":
            command.add_argument("--lock", type=Path, required=True)
    lock = subparsers.add_parser("lock")
    lock.add_argument("--discovery", type=Path, default=MECHANISM / "p4/discovery")
    lock.add_argument("--output", type=Path, required=True)
    lock.add_argument("--mechanism", choices=("actual_path", "default_path"), required=True)
    lock.add_argument("--layer", type=int, required=True)
    lock.add_argument("--position", type=int, choices=PRIMARY_POSITIONS, required=True)
    lock.add_argument("--rationale", required=True)
    args = parser.parse_args()
    if args.mode == "two-step":
        summarize_two_step(args.source, args.output, args.require_complete)
    elif args.mode == "discovery":
        discovery(args.source, args.output, args.require_complete)
    elif args.mode == "lock":
        lock_candidate(
            args.discovery, args.output, args.mechanism, args.layer, args.position, args.rationale
        )
    else:
        replication(args.source, args.output, args.lock, args.require_complete)


if __name__ == "__main__":
    main()
