"""MQuAKE Table 3: GPT-J, original ROME/MEMIT, independent edits per instance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paper_editing_runtime import (
    Runtime,
    answer_text,
    cloze_alias_match,
    exact_alias_match,
    read_json,
    record_progress,
    replay_calls,
    start_run,
    write_json,
)


def queries(case, prompts, edited):
    hops = case["new_single_hops"] if edited else case["single_hops"]
    triples = case["orig"]["new_triples"] if edited else case["orig"]["triples"]
    assert len(hops) == len(triples)
    single = []
    for hop, triple in zip(hops, triples, strict=True):
        prefix = prompts["relations"][triple[1]]
        single.append(
            {
                "prompt": prefix + "\nQ: " + hop["question"] + " A:",
                "answers": [hop["answer"], *hop["answer_alias"]],
            }
        )
    edits = []
    labeled = case["orig"]["new_triples_labeled"] if edited else case["orig"]["triples_labeled"]
    for request in case["requested_rewrite"]:
        target = request["target_new"] if edited else request["target_true"]
        associated = [
            hop
            for hop, triple, labels in zip(hops, triples, labeled, strict=True)
            if labels[0] == request["subject"]
            and triple[1] == request["relation_id"]
            and triple[2] == target["id"]
        ]
        if edited:
            assert associated, "Every requested edit must belong to the new gold fact chain"
        # Later edits may refer to a new bridge, e.g. editing India's capital
        # after changing a person's citizenship. Their old objects are supplied
        # in target_true but need not occur in the original fact chain.
        aliases = sorted({alias for hop in associated for alias in hop["answer_alias"]})
        edits.append(
            {
                "prompt": request["prompt"].format(request["subject"]),
                "answers": [target["str"], *aliases],
            }
        )
    answer = case["new_answer"] if edited else case["answer"]
    aliases = case["new_answer_alias"] if edited else case["answer_alias"]
    return {
        "edit": edits,
        "single": single,
        "multi": [
            {"prompt": prompts["multi"] + "\nQ: " + q + " A:", "answers": [answer, *aliases]}
            for q in case["questions"]
        ],
        "cot": [
            {
                "prompt": prompts["cot"] + "\n\nQuestion: " + q + "\nThoughts:",
                "answers": [answer, *aliases],
            }
            for q in case["questions"]
        ],
    }


def evaluate_case(runtime, case, prompts, edited):
    recipe = queries(case, prompts, edited)
    values = {}
    runtime.generation.reset_calls()
    for kind, items in recipe.items():
        predictions = runtime.generation.generate(
            [q["prompt"] for q in items],
            max_new_tokens=runtime.spec["cot_max_new_tokens"]
            if kind == "cot"
            else runtime.spec["answer_max_new_tokens"],
            mode="cot" if kind == "cot" else "line",
        )
        values[kind] = [
            dict(
                prediction,
                gold_answers=q["answers"],
                correct=(cloze_alias_match if kind == "edit" else exact_alias_match)(
                    prediction["answer_text"], q["answers"]
                ),
                strict_full_line_correct=exact_alias_match(prediction["answer_text"], q["answers"]),
            )
            for prediction, q in zip(predictions, items, strict=True)
        ]
    return {"predictions": values, "calls": runtime.generation.calls}


def phase_scores(predictions):
    return {
        "edit": [row["correct"] for row in predictions["edit"]],
        "instance": all(row["correct"] for row in predictions["single"]),
        "multi": any(row["correct"] for row in predictions["multi"]),
        "cot": any(row["correct"] for row in predictions["cot"]),
    }


def aggregate(records):
    n = len(records)
    if not n:
        return {}
    groups = {}
    for phase in ("baseline", "edited"):
        scores = [phase_scores(row[phase]["predictions"]) for row in records]
        edits = sum(len(row["edit"]) for row in scores)
        group = {
            "cases": n,
            "edits": edits,
            "edit_wise_success": sum(sum(row["edit"]) for row in scores) / edits,
            "instance_wise_accuracy": sum(row["instance"] for row in scores) / n,
            "multi_hop_accuracy": sum(row["multi"] for row in scores) / n,
            "multi_hop_cot_accuracy": sum(row["cot"] for row in scores) / n,
        }
        eligible = [s for s in scores if s["instance"] and all(s["edit"])]
        group["all_facts_and_edits_known_cases"] = len(eligible)
        group["all_facts_and_edits_known_coverage"] = len(eligible) / n
        if eligible:
            group["conditional_multi_hop_accuracy"] = sum(s["multi"] for s in eligible) / len(
                eligible
            )
            group["conditional_multi_hop_cot_accuracy"] = sum(s["cot"] for s in eligible) / len(
                eligible
            )
        groups[phase] = group
    for hops in (2, 3, 4):
        subset = [row for row in records if row["hops"] == hops]
        if subset:
            scores = [phase_scores(row["edited"]["predictions"]) for row in subset]
            groups[f"edited_{hops}_hop"] = {
                "cases": len(subset),
                "multi_hop_accuracy": sum(s["multi"] for s in scores) / len(subset),
                "multi_hop_cot_accuracy": sum(s["cot"] for s in scores) / len(subset),
            }
    return groups


def load_prompts(spec):
    root = Path(spec["prompt_dir"])
    return {
        "relations": read_json(root / "rel-prompts.json"),
        "multi": (root / "multihop-prompts.txt").read_text().strip(),
        "cot": (root / "multihop-cot-prompts.txt").read_text().strip(),
    }


def rescore(case, prompts, record):
    assert record["case_id"] == case["case_id"]
    assert (
        record["case_sha256"]
        == hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    )
    for phase, edited in (("baseline", False), ("edited", True)):
        recipe = queries(case, prompts, edited)
        for kind, expected in recipe.items():
            actual = record[phase]["predictions"][kind]
            assert len(actual) == len(expected)
            call_index = list(recipe).index(kind)
            call = record[phase]["calls"][call_index]
            assert call["edited"] == edited
            for row, query in zip(actual, expected, strict=True):
                assert row["prompt"] == query["prompt"] and row["gold_answers"] == query["answers"]
                assert row["answer_text"] == answer_text(
                    row["raw_generation"], "cot" if kind == "cot" else "line"
                )
                assert row["correct"] == (
                    cloze_alias_match if kind == "edit" else exact_alias_match
                )(row["answer_text"], query["answers"])
                assert row["strict_full_line_correct"] == exact_alias_match(
                    row["answer_text"], query["answers"]
                )
            assert [r["generated_token_ids"] for r in actual] == [
                r["generated_token_ids"] for r in call["items"]
            ]


def run(spec, out, device):
    out = Path(out)
    start_run(spec, out, device)
    runtime = Runtime(spec, out, device)
    cases = read_json(spec["data_file"])
    assert len(cases) == spec["total_source_cases"]
    cases = cases[: spec["max_cases"]]
    prompts = load_prompts(spec)
    records, history = [], []
    record_progress(out, runtime, history, 0, {})
    for index, case in enumerate(cases, 1):
        directory = out / "cases" / f"{index:05d}"
        directory.mkdir(parents=True, exist_ok=False)
        runtime.assert_original()
        baseline = evaluate_case(runtime, case, prompts, False)
        requests = [
            dict(request, case_id=case["case_id"] * 10 + i)
            for i, request in enumerate(case["requested_rewrite"])
        ]
        delta = runtime.edit(requests, directory / "deltas.pt")
        try:
            edited = evaluate_case(runtime, case, prompts, True)
        finally:
            runtime.restore()
        record = {
            "case_id": case["case_id"],
            "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
            "hops": len(case["single_hops"]),
            "baseline": baseline,
            "edited": edited,
            "delta": delta,
            "rollback_exact": True,
        }
        rescore(case, prompts, record)
        write_json(directory / "record.json", record)
        records.append(record)
        record_progress(out, runtime, history, index, aggregate(records))
    runtime.finish()
    write_json(out / "summary.json", aggregate(records))
    write_json(
        out / "complete.json", {"cases": len(records), "state": "awaiting_independent_replay"}
    )
    runtime.release_covariances()


def audit(out, device):
    """Fresh model, every saved edit, every raw generation, and metric denominators."""
    out = Path(out)
    spec = read_json(out / "run.json")["spec"]
    runtime = Runtime(spec, out / "reload-audit", device)
    cases = read_json(spec["data_file"])[: spec["max_cases"]]
    prompts = load_prompts(spec)
    records, queries_checked = [], 0
    for index, case in enumerate(cases, 1):
        record = read_json(out / "cases" / f"{index:05d}" / "record.json")
        rescore(case, prompts, record)
        queries_checked += replay_calls(runtime, record["baseline"]["calls"])
        runtime.replay(record["delta"]["file"], record["delta"]["sha256"])
        try:
            queries_checked += replay_calls(runtime, record["edited"]["calls"])
        finally:
            runtime.restore()
        runtime.generation.reset_calls()
        records.append(record)
        write_json(out / "audit-progress.json", {"cases": index, "queries": queries_checked})
    runtime.finish()
    metrics = aggregate(records)
    assert metrics == read_json(out / "summary.json")
    write_json(
        out / "audit.json",
        {
            "passed": True,
            "cases": len(records),
            "queries": queries_checked,
            "all_raw_generations_exact": True,
            "all_saved_edits_replayed": True,
            "metrics": metrics,
        },
    )
    write_json(out / "status.json", {"state": "complete", "step": len(records)})
