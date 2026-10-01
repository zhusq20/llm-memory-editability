"""Read-only answer-source diagnostics for the frozen two-chain experiments.

Matching an answer to a candidate is a behavioral description, not evidence that
the model used that internal path. Multiple matching roles are never tie-broken.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from .bios_data import EOS, array_hash, write_json

LEARNING_ROLES = ("default", "actual", "other_default", "other_actual")
EDIT_ROLES = (
    "new_default",
    "new_actual",
    "old_default",
    "old_actual",
    "other_default",
    "other_actual",
)
# Codes are shared across files. Role names differ between learning and editing.
NOT_APPLICABLE, TERMINATION_ERROR, OTHER, INDISTINGUISHABLE = -1, 0, 1, 2
ROLE_CODE_START = 3


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_answers(prediction, ended, candidates):
    """Return exclusive categories plus *all* matched role bits, including overlaps.

    An unfinished generation is a termination error even if its value matches.
    Correctness is deliberately separate: a correct answer can have ambiguous roles.
    """
    prediction, ended, candidates = map(np.asarray, (prediction, ended, candidates))
    if (
        candidates.ndim != 2
        or prediction.shape != ended.shape
        or candidates.shape[0] != len(prediction)
    ):
        raise ValueError("Incompatible prediction/candidate shapes")
    if not 1 <= candidates.shape[1] <= 8:
        raise ValueError("Candidate role bitset requires one to eight roles")
    matched = prediction[:, None] == candidates
    n_matches = matched.sum(axis=1)
    bits = (matched.astype(np.uint8) * (1 << np.arange(candidates.shape[1]))).sum(axis=1)
    category = np.full(len(prediction), OTHER, dtype=np.int8)
    unique = n_matches == 1
    category[unique] = ROLE_CODE_START + matched[unique].argmax(axis=1)
    category[n_matches > 1] = INDISTINGUISHABLE
    category[~ended.astype(bool)] = TERMINATION_ERROR
    return category, bits.astype(np.uint8)


def candidate_answers(world, chain, target=None):
    """Candidate values for all people, with no dependence on model predictions."""
    other = 1 - chain
    old_default = world.answers[world.root_ids[chain, world.memberships[chain]]]
    old_actual = world.answers[world.actual_ids[chain]]
    current = world.answers if target is None else target
    other_default = current[world.root_ids[other, world.memberships[other]]]
    other_actual = current[world.actual_ids[other]]
    if target is None:
        roles = LEARNING_ROLES
        values = (old_default, old_actual, other_default, other_actual)
    else:
        roles = EDIT_ROLES
        values = (
            target[world.root_ids[chain, world.memberships[chain]]],
            target[world.actual_ids[chain]],
            old_default,
            old_actual,
            other_default,
            other_actual,
        )
    return roles, np.column_stack(values)


def score_saved_arrays(arrays, answers):
    prediction, ended = arrays["prediction"], arrays["ended"]
    if prediction.shape != answers.shape or ended.shape != answers.shape:
        raise ValueError("Saved prediction length differs from truth")
    if not np.isin(ended, [False, True]).all():
        raise ValueError("Invalid termination flags")
    correct = (prediction == answers) & ended.astype(bool)
    if not np.array_equal(correct, arrays["correct"]):
        raise ValueError("Saved correctness differs from value-plus-EOS scoring")
    return correct


def annotate_predictions(world, arrays, target=None):
    """Full-query arrays plus candidate matrices aligned with derived_query_ids."""
    answers = world.answers if target is None else target
    correct = score_saved_arrays(arrays, answers)
    category = np.full(len(answers), NOT_APPLICABLE, dtype=np.int8)
    bits = np.zeros(len(answers), dtype=np.uint8)
    candidates = []
    for chain in range(2):
        roles, values = candidate_answers(world, chain, target)
        ids = world.derived_ids[chain]
        category[ids], bits[ids] = classify_answers(
            arrays["prediction"][ids], arrays["ended"][ids], values
        )
        candidates.append(values)
    return {
        "query_id": np.arange(len(answers), dtype=np.int32),
        "prediction": np.asarray(arrays["prediction"], dtype=np.int32),
        "ended": np.asarray(arrays["ended"], dtype=bool),
        "correct": correct,
        "class_code": category,
        "matched_role_bits": bits,
        "derived_query_ids": world.derived_ids.astype(np.int32),
        "candidate_tokens": np.stack(candidates).astype(np.int32),
        "candidate_roles": np.array(roles),
    }


def learning_subsets(world):
    """Overlapping diagnostic groups; every row retains its own denominator."""
    yield "all_queries", np.arange(len(world.answers))
    yield "base", np.arange(world.n_base)
    yield "independent_attributes", np.flatnonzero(np.isin(world.relation, [3, 4, 5, 6]))
    for chain, name in enumerate(CHAINS):
        for relation, ids in (
            ("membership", world.membership_ids[chain]),
            ("default", world.root_ids[chain]),
            ("actual", world.actual_ids[chain]),
        ):
            yield f"{name}/base/{relation}", ids
        for split, ids in (
            ("all", world.derived_ids[chain]),
            ("trained", world.train_ids[chain]),
            ("heldout", world.heldout_ids[chain]),
        ):
            yield f"{name}/qa/{split}/all", ids
            for label, mask in (
                ("ordinary", ~world.exceptions[chain]),
                ("old_exception", world.exceptions[chain]),
            ):
                yield (
                    f"{name}/qa/{split}/{label}",
                    np.intersect1d(ids, world.derived_ids[chain, mask]),
                )


def editing_subsets(world, pair, chain, target):
    yield "E/all", pair["E"]
    yield "E/default", np.intersect1d(pair["E"], world.root_ids[chain])
    yield "E/actual", np.intersect1d(pair["E"], world.actual_ids[chain])
    # Membership in the 45-person manipulated cohort differs from factual conflict:
    # six old exceptions can also conflict; coherent edits have no *new* conflicts.
    actual_conflict = target[world.derived_ids[chain]] != target[world.actual_ids[chain]]
    for split, ids in (
        ("all", pair["D"]),
        ("trained", np.intersect1d(pair["D"], world.train_ids[chain])),
        ("heldout", np.intersect1d(pair["D"], world.heldout_ids[chain])),
    ):
        yield f"D/{split}/all", ids
        for label, group in (
            ("manipulated_conflict_cohort", pair["conflict_D"]),
            ("factual_conflict", world.derived_ids[chain, actual_conflict]),
            ("no_factual_conflict", world.derived_ids[chain, ~actual_conflict]),
            ("old_exception", world.derived_ids[chain, world.exceptions[chain]]),
            ("ordinary", world.derived_ids[chain, ~world.exceptions[chain]]),
        ):
            yield f"D/{split}/{label}", np.intersect1d(ids, group)
    for pool_name, pool in (
        ("full", np.flatnonzero(pair["strata"] >= 0)),
        ("editor_heldout", pair["heldout"]),
    ):
        yield f"U/{pool_name}/all", pool
        for stratum in range(4):
            yield (
                f"U/{pool_name}/stratum_{stratum}",
                np.intersect1d(pool, np.flatnonzero(pair["strata"] == stratum)),
            )


def summary_rows(world, annotations, subsets, identity, old_correct=None):
    """Counts are sufficient to reaggregate without hiding empty or ambiguous cells."""
    classes, bits = annotations["class_code"], annotations["matched_role_bits"]
    for label, ids in subsets:
        ids = np.asarray(ids)
        correct = annotations["correct"][ids]
        row = {
            **identity,
            "subset": label,
            "n": len(ids),
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()) if len(ids) else None,
            "termination_error": int((~annotations["ended"][ids]).sum()),
            "role_not_applicable": int((classes[ids] == NOT_APPLICABLE).sum()),
            "other_answer": int((classes[ids] == OTHER).sum()),
            "indistinguishable": int((classes[ids] == INDISTINGUISHABLE).sum()),
            "indistinguishable_wrong": int(((classes[ids] == INDISTINGUISHABLE) & ~correct).sum()),
        }
        # Stable CSV schema, even though learning uses only four roles.
        for role in dict.fromkeys((*LEARNING_ROLES, *EDIT_ROLES)):
            role_names = list(annotations["candidate_roles"])
            if role in role_names:
                index = role_names.index(role)
                row[f"unique_{role}"] = int((classes[ids] == ROLE_CODE_START + index).sum())
                row[f"matches_{role}"] = int(
                    (((bits[ids] & (1 << index)) != 0) & annotations["ended"][ids]).sum()
                )
            else:
                row[f"unique_{role}"] = row[f"matches_{role}"] = 0
        if old_correct is not None:
            known = old_correct[ids]
            row["old_known"] = int(known.sum())
            row["old_known_coverage"] = float(known.mean()) if len(ids) else None
            row["old_known_now_wrong"] = int((known & ~correct).sum())
            row["damage_rate"] = (
                (float((known & ~correct).sum() / known.sum()) if known.any() else None)
                if label.startswith("U/")
                else None
            )
        else:
            row.update(
                old_known=None, old_known_coverage=None, old_known_now_wrong=None, damage_rate=None
            )
        yield row


def _write_csv(path, rows):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_output(output, source_roots):
    output = Path(output).resolve()
    if any(output == root or root in output.parents for root in source_roots):
        raise ValueError("Diagnostics must not be written into source experiment directories")
    output.mkdir(parents=True, exist_ok=True)
    return output


def analyze_saved(repository, output, manifest_path="configs/bios-cross-scale-development-v1.json"):
    """Re-score the complete matrix using CPU arrays; never invoke old summary writers."""
    repository = Path(repository).resolve()
    manifest_path = repository / manifest_path
    manifest = json.loads(manifest_path.read_text())
    source_roots = [(repository / size["output"]).resolve() for size in manifest["sizes"]]
    output = _safe_output(output, source_roots)
    ledger = {}

    def record(path):
        path = Path(path).resolve()
        key = str(path.relative_to(repository)) if repository in path.parents else str(path)
        digest = file_sha256(path)
        if key in ledger and ledger[key] != digest:
            raise ValueError(f"Source changed during analysis: {path}")
        ledger[key] = digest

    record(manifest_path)
    worlds, pairs = {}, {}
    learning_rows, editing_rows, error_rows = [], [], []
    counts = {
        "models": 0,
        "learning_checkpoints": 0,
        "edit_cases": 0,
        "edit_checkpoints": 0,
        "scored_query_predictions": 0,
        "classified_derived_predictions": 0,
    }
    for size, root in zip(manifest["sizes"], source_roots, strict=True):
        contract_path = root / "launch-contract.json"
        record(contract_path)
        contract = json.loads(contract_path.read_text())
        study = contract["study"]
        expected_runs = {
            (w, seed, condition)
            for w in study["worlds"]
            for seed in study["seeds"]
            for condition in CONDITIONS
        }
        seen_runs = set()
        for config_path in sorted(root.glob("world-*/config.json")):
            record(config_path)
            config = json.loads(config_path.read_text())
            if config["sources"] != contract["sources"] or config["study"] != study:
                raise ValueError(f"Run contract mismatch: {config_path}")
            for name, digest in config["sources"].items():
                path = repository / "src/llm_memory_editability" / name
                if file_sha256(path) != digest:
                    raise ValueError(f"Frozen source changed: {name}")
                record(path)
            run = config_path.parent
            key = (config["world"], config["seed"], config["condition"])
            if key in seen_runs:
                raise ValueError("Duplicate model")
            seen_runs.add(key)
            if not (run / "complete.json").exists():
                raise ValueError(f"Incomplete run: {run}")
            record(run / "complete.json")
            world_seed = config["world"]
            if world_seed not in worlds:
                worlds[world_seed] = make_cross_world(
                    world_seed, repository / "data/bios-organization-v1"
                )
            world = worlds[world_seed]
            if config["truth_sha256"] != array_hash(world.answers) or config[
                "prompts_sha256"
            ] != array_hash(world.prompts):
                raise ValueError("Generated data mismatch")
            common = {
                "width": size["width"],
                "parameters": config["parameters"],
                "world": world_seed,
                "seed": config["seed"],
                "condition": config["condition"],
            }
            out_run = output / "queries" / f"width-{size['width']}" / run.name
            out_run.mkdir(parents=True, exist_ok=True)
            baseline = None
            for step in study["checkpoints"]:
                prediction_path = run / f"predictions-{step}.npz"
                record(prediction_path)
                with np.load(prediction_path, allow_pickle=False) as loaded:
                    arrays = {name: loaded[name] for name in ("prediction", "ended", "correct")}
                annotated = annotate_predictions(world, arrays)
                np.savez_compressed(out_run / f"learning-{step}.npz", **annotated)
                identity = {
                    **common,
                    "phase": "learning",
                    "chain": "",
                    "kind": "",
                    "scope": "",
                    "step": step,
                }
                learning_rows.extend(
                    summary_rows(world, annotated, learning_subsets(world), identity)
                )
                counts["learning_checkpoints"] += 1
                counts["scored_query_predictions"] += len(world.answers)
                counts["classified_derived_predictions"] += world.derived_ids.size
                if step == study["steps"]:
                    baseline = arrays["correct"]
            if baseline is None:
                raise ValueError("Missing learning endpoint")
            for chain, name in enumerate(CHAINS):
                pair_key = (world_seed, chain)
                if pair_key not in pairs:
                    pairs[pair_key] = edit_pair(world, chain)
                pair = pairs[pair_key]
                for kind in ("coherent", "exception"):
                    target = pair[kind]
                    for scope in ("mlp", "all"):
                        case = f"{name}-{kind}-{scope}"
                        dest = run / "edits" / case
                        record(dest / "complete.json")
                        record(dest / "sets.npz")
                        with np.load(dest / "sets.npz", allow_pickle=False) as loaded:
                            for field, expected in pair.items():
                                if not np.array_equal(loaded[field], expected):
                                    raise ValueError(f"Edit set mismatch: {dest}/{field}")
                            if not np.array_equal(loaded["old_correct"], baseline):
                                raise ValueError(f"Edit baseline mismatch: {dest}")
                        for step in study["edit_checkpoints"]:
                            prediction_path = dest / f"predictions-{step}.npz"
                            record(prediction_path)
                            with np.load(prediction_path, allow_pickle=False) as loaded:
                                arrays = {k: loaded[k] for k in ("prediction", "ended", "correct")}
                            annotated = annotate_predictions(world, arrays, target)
                            np.savez_compressed(out_run / f"{case}-{step}.npz", **annotated)
                            identity = {
                                **common,
                                "phase": "editing",
                                "chain": name,
                                "kind": kind,
                                "scope": scope,
                                "step": step,
                            }
                            editing_rows.extend(
                                summary_rows(
                                    world,
                                    annotated,
                                    editing_subsets(world, pair, chain, target),
                                    identity,
                                    baseline,
                                )
                            )
                            # Readable per-D query ledger, with every ambiguity explicitly retained.
                            roles, values = candidate_answers(world, chain, target)
                            for query in pair["D"]:
                                person = world.person[query]
                                matches = [
                                    role
                                    for i, role in enumerate(roles)
                                    if annotated["matched_role_bits"][query] & (1 << i)
                                ]
                                code = int(annotated["class_code"][query])
                                category = {
                                    TERMINATION_ERROR: "termination_error",
                                    OTHER: "other",
                                    INDISTINGUISHABLE: "indistinguishable",
                                }
                                error_rows.append(
                                    {
                                        **identity,
                                        "query": int(query),
                                        "person": int(person),
                                        "heldout": bool(query in world.heldout_ids[chain]),
                                        "old_exception": bool(world.exceptions[chain, person]),
                                        "manipulated_conflict_cohort": bool(
                                            query in pair["conflict_D"]
                                        ),
                                        "factual_conflict": bool(
                                            values[person, 0] != values[person, 1]
                                        ),
                                        "prediction": int(arrays["prediction"][query]),
                                        "ended": bool(arrays["ended"][query]),
                                        "correct": bool(arrays["correct"][query]),
                                        "class": category.get(
                                            code,
                                            roles[code - ROLE_CODE_START]
                                            if code >= ROLE_CODE_START
                                            else "",
                                        ),
                                        "matched_roles": "|".join(matches),
                                        **{
                                            role: int(values[person, i])
                                            for i, role in enumerate(roles)
                                        },
                                    }
                                )
                            counts["edit_checkpoints"] += 1
                            counts["scored_query_predictions"] += len(world.answers)
                            counts["classified_derived_predictions"] += world.derived_ids.size
                        counts["edit_cases"] += 1
            counts["models"] += 1
            print(json.dumps({"event": "p0_model_scored", **common, **counts}), flush=True)
        if seen_runs != expected_runs:
            raise ValueError(f"Incomplete or unexpected model grid: width {size['width']}")
    if (
        counts["models"] != manifest["learning_runs"]
        or counts["edit_cases"] != manifest["edit_cases"]
    ):
        raise ValueError("Incomplete matrix")
    _write_csv(output / "learning-strata.csv", learning_rows)
    _write_csv(output / "editing-strata.csv", editing_rows)
    _write_csv(output / "propagation-queries.csv", error_rows)
    # Verify that reading/analysis did not mutate any source and that no parallel
    # process changed this historical input batch while it was being analyzed.
    for name, digest in ledger.items():
        if file_sha256(repository / name) != digest:
            raise ValueError(f"Source changed during analysis: {name}")
    write_json(output / "sources.json", ledger)
    write_json(
        output / "schema.json",
        {
            "class_codes": {
                "not_applicable_to_base_query": -1,
                "termination_error": 0,
                "other": 1,
                "indistinguishable": 2,
                "unique_role_start": 3,
            },
            "learning_roles": LEARNING_ROLES,
            "editing_roles": EDIT_ROLES,
            "matched_role_bits": "bit i indicates value matches role i; EOS checked separately",
            "candidate_tokens_shape": "chain, person, role; "
            "maps to derived_query_ids[chain, person]",
            "interpretation": "Candidate answer agreement is not causal evidence for a path. "
            "Overlapping roles are never assigned a preferred interpretation.",
            "subset_rows_overlap": True,
            "manipulated_conflict_cohort": "Original 45 people, also tagged in coherent controls; "
            "factual_conflict is measured from actual target answers.",
        },
    )
    code_hashes = {
        str(path.relative_to(repository)): file_sha256(path)
        for path in (Path(__file__).resolve(), repository / "scripts/analyze_bios_paths.py")
    }
    audit = {
        "complete": True,
        **counts,
        "source_files": len(ledger),
        "sources_sha256": file_sha256(output / "sources.json"),
        "analysis_sources": code_hashes,
        "errors": [],
        "source_data_modified": False,
        "source_hashes_rechecked_after_analysis": True,
    }
    write_json(output / "audit.json", audit)
    _report(output, audit, learning_rows, editing_rows)
    return audit


def _report(output, audit, learning_rows, editing_rows):
    lines = [
        "# P0：已有预测的行为诊断",
        "",
        f"完成{audit['models']}个模型、{audit['edit_cases']}个编辑；"
        f"重计分{audit['learning_checkpoints']}个学习和{audit['edit_checkpoints']}个编辑检查点。",
        "",
        "候选答案相同的角色记为不可区分；答案重合不能证明内部路径。"
        "终止失败独立于答案值记录。下表为案例等权均值，非独立查询推断。",
        "",
        "| 宽度 | 编辑 | 案例 | 冲突留出D | 新实际城市匹配 | 不可区分 | 终止失败 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in editing_rows:
        if (
            row["kind"] == "exception"
            and row["step"] == 512
            and row["subset"] == "D/heldout/manipulated_conflict_cohort"
        ):
            grouped[row["width"], row["scope"]].append(row)
    for (width, scope), rows in sorted(grouped.items()):
        rates = [
            np.mean([r[key] / r["n"] for r in rows])
            for key in ("correct", "matches_new_actual", "indistinguishable", "termination_error")
        ]
        lines.append(
            f"| {width} | {scope} | {len(rows)} | "
            + " | ".join(f"{100 * rate:.2f}%" for rate in rates)
            + " |"
        )
    lines += [
        "",
        "新实际城市匹配是允许与旧实际/另一条链答案重合的描述性比例；"
        "唯一匹配和全部角色位图均另存，不能据此宣称捷径已证实。",
        "",
        "学习分层见 learning-strata.csv；编辑分层见 editing-strata.csv；"
        "每条传播查询见 propagation-queries.csv；全部逐查询数组见 queries/。",
        "自主两步前向诊断尚须单独执行并记录模型权重来源。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def free_generate_values(model, prompts, lengths, device, batch_size=512):
    """Exactly the frozen value-plus-EOS generation convention, without targets."""
    import torch

    from .bios_train import precision

    predictions, ended = [], []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(prompts), batch_size):
            p = torch.as_tensor(prompts[begin : begin + batch_size], device=device)
            n = torch.as_tensor(lengths[begin : begin + batch_size], device=device)
            with precision(device):
                first = model(p, (n - 1)[:, None])[:, 0].float().argmax(-1)
                continuation = torch.zeros((len(p), 6), dtype=torch.long, device=device)
                continuation[:, :5] = p
                continuation[torch.arange(len(p), device=device), n] = first
                second = model(continuation, n[:, None])[:, 0].argmax(-1)
            predictions.append(first.cpu().numpy())
            ended.append(second.eq(EOS).cpu().numpy())
    return {"prediction": np.concatenate(predictions), "ended": np.concatenate(ended)}


def autonomous_two_step(model, world, chain, device, batch_size=512, answers=None):
    """Generate bridge identity, then query that predicted identity, never the true one.

    Truth is accessed only for scoring after both model calls. A wrong but valid
    bridge may coincidentally lead to a correct city; keep that outcome and the
    first-hop correctness separately. Invalid/unterminated bridges fail outright.
    """
    membership_ids = world.membership_ids[chain]
    first = free_generate_values(
        model, world.prompts[membership_ids], world.lengths[membership_ids], device, batch_size
    )
    allowed_subjects = world.prompts[world.root_ids[chain], 1]
    valid = first["ended"] & np.isin(first["prediction"], allowed_subjects)
    second_prediction = np.full(len(membership_ids), -1, dtype=np.int64)
    second_ended = np.zeros(len(membership_ids), dtype=bool)
    if valid.any():
        # Root queries have the same grammar and relation token. Do not choose a
        # template using true membership; replace the subject using generated data.
        root_template = world.prompts[world.root_ids[chain, 0]]
        prompts = np.broadcast_to(root_template, (int(valid.sum()), 5)).copy()
        prompts[:, 1] = first["prediction"][valid]
        lengths = np.full(len(prompts), world.lengths[world.root_ids[chain, 0]], dtype=np.int64)
        second = free_generate_values(model, prompts, lengths, device, batch_size)
        second_prediction[valid], second_ended[valid] = second["prediction"], second["ended"]
    target = world.answers if answers is None else answers
    return {
        "query_id": world.derived_ids[chain],
        "bridge_prediction": first["prediction"],
        "bridge_ended": first["ended"],
        "bridge_valid": valid,
        "bridge_correct": (first["prediction"] == target[membership_ids]) & first["ended"],
        "prediction": second_prediction,
        "ended": second_ended,
        "correct": valid & second_ended & (second_prediction == target[world.derived_ids[chain]]),
    }


def diagnose_two_step_run(
    run, output, device="cuda", learning_steps=(15360,), include_edits=False, batch_size=512
):
    """Independent, resumable forward-only jobs; no optimizer or training mutations."""
    import torch

    from .bios_model import CausalLM, ModelConfig

    run, output = Path(run).resolve(), Path(output).resolve()
    repository = Path(__file__).resolve().parents[2]
    output = _safe_output(output, [run])
    config = json.loads((run / "config.json").read_text())
    world = make_cross_world(config["world"], repository / "data/bios-organization-v1")
    if array_hash(world.answers) != config["truth_sha256"]:
        raise ValueError("Two-step world differs from trained world")
    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    jobs = [
        (
            f"learning-{step}",
            run / f"model-{step}.pt",
            world.answers,
            run / f"predictions-{step}.npz",
        )
        for step in learning_steps
    ]
    if include_edits:
        for chain, name in enumerate(CHAINS):
            pair = edit_pair(world, chain)
            for kind in ("coherent", "exception"):
                for scope in ("mlp", "all"):
                    dest = run / "edits" / f"{name}-{kind}-{scope}"
                    jobs.append(
                        (
                            dest.name,
                            dest / "model-final.pt",
                            pair[kind],
                            dest / f"predictions-{config['study']['edit_steps']}.npz",
                        )
                    )
    for label, checkpoint, answers, direct_path in jobs:
        dest = output / label
        dest.mkdir(parents=True, exist_ok=True)
        identity = {
            "run": str(run),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "direct_predictions_sha256": file_sha256(direct_path),
            "config_sha256": file_sha256(run / "config.json"),
            "diagnostic_source_sha256": file_sha256(__file__),
            "answer_sha256": array_hash(answers),
            "batch_size": batch_size,
            "device_type": device.type,
            "torch": torch.__version__,
            "precision": "BF16 autocast; FP32 weights" if device.type == "cuda" else "FP32",
        }
        complete_path = dest / "complete.json"
        if complete_path.exists():
            saved = json.loads(complete_path.read_text())
            if saved["identity"] != identity:
                raise ValueError(f"Changed diagnostic identity: {dest}")
            continue
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = CausalLM(ModelConfig(**saved["config"]))
        model.load_state_dict(saved["model"])
        model.to(device).eval()
        with np.load(direct_path) as data:
            direct = {k: data[k] for k in ("prediction", "ended", "correct")}
        score_saved_arrays(direct, answers)
        rows = []
        for chain, name in enumerate(CHAINS):
            arrays = autonomous_two_step(model, world, chain, device, batch_size, answers)
            ids = world.derived_ids[chain]
            arrays.update({f"direct_{k}": v[ids] for k, v in direct.items()})
            np.savez_compressed(dest / f"{name}.npz", **arrays)
            for split, mask in (
                ("all", np.ones(len(ids), dtype=bool)),
                ("trained", np.isin(ids, world.train_ids[chain])),
                ("heldout", np.isin(ids, world.heldout_ids[chain])),
            ):
                for population, population_mask in (
                    ("all", np.ones(len(ids), dtype=bool)),
                    ("ordinary", ~world.exceptions[chain]),
                    ("old_exception", world.exceptions[chain]),
                ):
                    selected = mask & population_mask
                    n = int(selected.sum())
                    rows.append(
                        {
                            "chain": name,
                            "split": split,
                            "population": population,
                            "n": n,
                            **{
                                key: int(arrays[key][selected].sum())
                                for key in (
                                    "correct",
                                    "direct_correct",
                                    "bridge_valid",
                                    "bridge_correct",
                                )
                            },
                        }
                    )
        _write_csv(dest / "summary.csv", rows)
        write_json(
            complete_path,
            {
                "identity": identity,
                "complete": True,
                "queries": int(world.derived_ids.size),
                "oracle_bridging": False,
            },
        )
        del model, saved
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps({"event": "two_step_complete", "run": str(run), "case": label}), flush=True
        )
