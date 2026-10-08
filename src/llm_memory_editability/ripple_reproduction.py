"""RippleEdits Tables 3–5 with the authors' evaluator and bundled GPT-2 XL ROME."""

from __future__ import annotations

import ast
import importlib
import re
import sys
import types
from collections import defaultdict
from pathlib import Path

from .paper_editing_runtime import (
    Runtime,
    read_json,
    record_progress,
    replay_calls,
    ripple_match,
    source_ripple_match,
    start_run,
    write_json,
)

AXES = {
    "Relation_Specificity": "RS",
    "Logical_Generalization": "LG",
    "Subject_Aliasing": "SA",
    "Compositionality_I": "CI",
    "Compositionality_II": "CII",
    "Forgetfulness": "PV",
}


def relation_templates(source):
    tree = ast.parse((Path(source) / "src/relation.py").read_text())
    relation = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    return {
        node.targets[0].id: ast.literal_eval(node.value)[1]
        for node in relation.body
        if isinstance(node, ast.Assign)
    }


def fact_labels(data, templates):
    """Recover the exact dated labels from the published fact, without Wikidata calls."""
    before, after = templates[data["relation"]].split("<subject>")
    matched = re.fullmatch(
        re.escape(before) + r"(.*?)" + re.escape(after) + r" (.*)\.", data["prompt"]
    )
    assert matched is not None, data
    subject, target = matched.groups()
    return subject, target, before + subject + after


def frozen_aliases(cases):
    aliases = defaultdict(set)
    for case in cases:
        for field in AXES:
            for test in case[field]:
                for query in test["test_queries"] + test["condition_queries"]:
                    ids = (
                        query["second_hop_target_ids"]
                        if query["query_type"] == "two_hop"
                        else query["target_ids"]
                    )
                    assert len(ids) == len(query["answers"])
                    for target, answer in zip(ids, query["answers"], strict=True):
                        if isinstance(target, str):
                            aliases[target].update([answer["value"], *answer["aliases"]])
    return {key: sorted(values) for key, values in aliases.items()}


class FrozenQuery:
    def __init__(self, data):
        self.data = data

    def get_query_prompt(self):
        return self.data["prompt"]

    def get_answers(self):
        return [
            [s for s in [answer["value"], *answer["aliases"]] if len(s) > 1 or s.isdigit()]
            for answer in self.data["answers"]
        ]

    def to_dict(self):
        return self.data


class FrozenFact:
    def __init__(self, data, templates, aliases):
        self.data = data
        self.subject, self.target, self.prompt = fact_labels(data, templates)
        self.aliases = aliases.get(data["target_id"], [])

    def get_subject_label(self):
        return self.subject

    def get_target_label(self):
        return self.target

    def get_fact_prompt(self):
        return self.prompt

    def get_fact_query(self):
        return FrozenQuery(
            {
                "prompt": self.prompt,
                "answers": [{"value": self.target, "aliases": self.aliases}],
                "query_type": "regular",
                "subject_id": self.data["subject_id"],
                "relation": self.data["relation"],
                "target_ids": [self.data["target_id"]],
                "phrase": None,
            }
        )


def author_evaluator(source, cases):
    """Load the original evaluator definitions and use dated JSON prompts and answers."""
    source = Path(source)
    sys.path[:0] = [str(source / "src"), str(source)]

    def unavailable(*args, **kwargs):
        raise RuntimeError("A reproduction must not query live Wikidata or an external API")

    wikidata = types.ModuleType("wikidata")
    wiki_utils = types.ModuleType("wikidata.utils")
    for name in ("get_label", "get_aliases", "subject_relation_to_targets"):
        setattr(wiki_utils, name, unavailable)
    sys.modules["wikidata"] = wikidata
    sys.modules["wikidata.utils"] = wiki_utils
    # Only the two pure comparison helpers are needed from utils.py. Its other
    # imports require the authors' private OpenAI key, which local evaluation never uses.
    utils = types.ModuleType("utils")
    parsed = ast.parse((source / "src/utils.py").read_text())
    body = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"normalize_text", "compute_exact_match"}
    ]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source / "src/utils.py"), "exec"),
        utils.__dict__,
    )
    sys.modules["utils"] = utils
    query_module = importlib.import_module("query")
    query_module.Query.from_dict = staticmethod(FrozenQuery)
    # The released testcase imports src.query while fact imports query.
    sys.modules["src.query"] = query_module
    fact_module = importlib.import_module("fact")
    templates, aliases = relation_templates(source), frozen_aliases(cases)
    fact_module.Fact.from_dict = staticmethod(lambda data: FrozenFact(data, templates, aliases))
    benchmark = importlib.import_module("benchmark")
    runner = importlib.import_module("testrunner")
    namespace = {
        "defaultdict": defaultdict,
        "Example": benchmark.Example,
        "TestsAxis": benchmark.TestsAxis,
        "TestRunner": runner.TestRunner,
        "TestResult": runner.TestResult,
        "ExampleResult": runner.ExampleResult,
    }
    parsed = ast.parse((source / "src/evaluation.py").read_text())
    body = [node for node in parsed.body if isinstance(node, ast.ClassDef)]
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source / "src/evaluation.py"), "exec"),
        namespace,
    )
    return benchmark, runner, namespace["Evaluator"]


class QueryExecutor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.events = []

    def execute_query(self, query, answer_length=30):
        row = self.runtime.generation.generate(
            [query.get_query_prompt()], max_new_tokens=20, mode="ripple"
        )[0]
        answers = query.get_answers()
        correct = ripple_match(row["raw_generation"], answers)
        self.events.append(
            {
                "query": query.to_dict(),
                "answers": answers,
                "edited": self.runtime.edited,
                "raw_generation": row["raw_generation"],
                "correct": correct,
                "source_all_objects_correct": source_ripple_match(row["raw_generation"], answers),
            }
        )
        return correct


class ModelEditor:
    def __init__(self, runtime, destination):
        self.runtime, self.destination = runtime, destination
        self.delta = None

    def edit_model(self, fact):
        subject = fact.get_subject_label()
        requests = [
            {
                "prompt": fact.get_fact_prompt().replace(subject, "{}"),
                "subject": subject,
                "target_new": {"str": fact.get_target_label()},
            }
        ]
        assert requests[0]["prompt"].count("{}") == 1
        self.delta = self.runtime.edit(requests, self.destination)

    def restore_model(self):
        self.runtime.restore()


def evaluate_axis(evaluator_class, example, tests, executor, editor):
    evaluator = evaluator_class(executor, editor)
    captured = {}
    original = evaluator._test_runner.run_testcases

    def observe(*args, **kwargs):
        flag, results = original(*args, **kwargs)
        captured.update(
            fact_result=flag.name,
            outcomes={
                key.name: [tests.index(value) for value in values]
                for key, values in results.items()
            },
        )
        return flag, results

    evaluator._test_runner.run_testcases = observe
    accuracy, executed, size, succeeded = evaluator.average_acc(example, tests)
    return {
        "paper_accuracy": accuracy,
        "executed_fraction": executed,
        "test_count": int(size),
        "edit_success": succeeded,
        **captured,
    }


def aggregate(records):
    valid = [row for row in records if not row.get("excluded")]
    metrics = {
        "pool": {
            "source_cases_processed": len(records),
            "valid_label_cases": len(valid),
            "invalid_label_cases": len(records) - len(valid),
        }
    }
    for axis in AXES.values():
        rows = [record["axes"][axis]["result"] for record in valid]
        eligible = [row for row in rows if row["edit_success"] and row["executed_fraction"] > 0]
        total_tests = sum(row["test_count"] for row in rows)
        executed = sum(
            len(row["outcomes"]["PASSED"]) + len(row["outcomes"]["FAILED"]) for row in rows
        )
        group = {
            "valid_label_cases": len(valid),
            "successful_edit_cases": sum(row["edit_success"] for row in rows),
            "successful_edit_eligible_cases": len(eligible),
            "paper_case_coverage": len(eligible) / len(valid) if valid else 0,
            "all_test_cases": total_tests,
            "known_precondition_test_cases": executed,
            "precondition_filtered_test_cases": total_tests - executed,
            "known_precondition_test_coverage": executed / total_tests if total_tests else 0,
        }
        if eligible:
            group["paper_accuracy"] = sum(row["paper_accuracy"] for row in eligible) / len(eligible)
        metrics[axis] = group
    observed = [
        metrics[axis]["paper_accuracy"]
        for axis in AXES.values()
        if "paper_accuracy" in metrics[axis]
    ]
    if len(observed) == 6:
        metrics["pool"]["paper_six_axis_average"] = sum(observed) / 6
    return metrics


def testcase_lists(example):
    return dict(
        zip(
            AXES,
            [
                example.making_up_tests,
                example.logical_constraints,
                example.subject_paraphrasing_tests,
                example.two_hop_tests,
                example.forward_two_hop_tests,
                example.prev_storage_tests,
            ],
            strict=True,
        )
    )


def run(spec, out, device):
    out = Path(out)
    start_run(spec, out, device)
    runtime = Runtime(spec, out, device)
    all_cases = read_json(spec["data_file"])
    assert len(all_cases) == spec["total_source_cases"]
    benchmark, _, evaluator = author_evaluator(spec["benchmark_source"], all_cases)
    cases = all_cases[spec.get("start_case", 0) :][: spec["max_cases"]]
    records, history = [], []
    record_progress(out, runtime, history, 0, {})
    for index, case in enumerate(cases, 1):
        directory = out / "cases" / f"{index:05d}"
        directory.mkdir(parents=True, exist_ok=False)
        example = benchmark.Example.from_dict(case)
        record = {"source_index": index - 1 + spec.get("start_case", 0), "case": case, "axes": {}}
        if not example.fact.get_subject_label() or not example.fact.get_target_label():
            record.update(
                excluded=True, reason="Empty subject or target label in released fact prompt"
            )
        else:
            for field, tests in testcase_lists(example).items():
                axis = AXES[field]
                executor = QueryExecutor(runtime)
                editor = ModelEditor(runtime, directory / f"{axis}-deltas.pt")
                runtime.generation.reset_calls()
                try:
                    result = evaluate_axis(evaluator, example, tests, executor, editor)
                finally:
                    if runtime.edited:
                        runtime.restore()
                record["axes"][axis] = {
                    "result": result,
                    "events": executor.events,
                    "calls": runtime.generation.calls,
                    "delta": editor.delta,
                    "rollback_exact": True,
                }
        write_json(directory / "record.json", record)
        records.append(record)
        record_progress(out, runtime, history, index, aggregate(records))
    runtime.finish()
    write_json(out / "summary.json", aggregate(records))
    write_json(
        out / "complete.json", {"cases": len(records), "state": "awaiting_independent_replay"}
    )
    runtime.release_covariances()


class ReplayExecutor:
    def __init__(self, events):
        self.events, self.index, self.edited = events, 0, False

    def execute_query(self, query, answer_length=30):
        event = self.events[self.index]
        self.index += 1
        assert query.to_dict() == event["query"] and query.get_answers() == event["answers"]
        assert event["edited"] == self.edited
        correct = ripple_match(event["raw_generation"], query.get_answers())
        assert correct == event["correct"]
        return correct


class ReplayEditor:
    def __init__(self, executor):
        self.executor = executor

    def edit_model(self, fact):
        assert not self.executor.edited
        self.executor.edited = True

    def restore_model(self):
        self.executor.edited = False


def audit(out, device):
    """Re-run original eligibility/aggregation, and reload every edited model and query."""
    out = Path(out)
    spec = read_json(out / "run.json")["spec"]
    runtime = Runtime(spec, out / "reload-audit", device)
    all_cases = read_json(spec["data_file"])
    benchmark, _, evaluator = author_evaluator(spec["benchmark_source"], all_cases)
    cases = all_cases[spec.get("start_case", 0) :][: spec["max_cases"]]
    records, checked = [], 0
    for index, case in enumerate(cases, 1):
        record = read_json(out / "cases" / f"{index:05d}" / "record.json")
        assert record["case"] == case
        example = benchmark.Example.from_dict(case)
        if record.get("excluded"):
            assert not example.fact.get_subject_label() or not example.fact.get_target_label()
        else:
            for field, tests in testcase_lists(example).items():
                saved = record["axes"][AXES[field]]
                executor = ReplayExecutor(saved["events"])
                result = evaluate_axis(evaluator, example, tests, executor, ReplayEditor(executor))
                assert result == saved["result"] and executor.index == len(saved["events"])
                calls = saved["calls"]
                assert len(calls) == len(saved["events"])
                for event, call in zip(saved["events"], calls, strict=True):
                    assert call["mode"] == "ripple" and call["max_new_tokens"] == 20
                    assert call["items"][0]["prompt"] == event["query"]["prompt"]
                    assert call["items"][0]["raw_generation"] == event["raw_generation"]
                checked += replay_calls(runtime, [c for c in calls if not c["edited"]])
                runtime.replay(saved["delta"]["file"], saved["delta"]["sha256"])
                try:
                    checked += replay_calls(runtime, [c for c in calls if c["edited"]])
                finally:
                    runtime.restore()
                runtime.generation.reset_calls()
        records.append(record)
        write_json(out / "audit-progress.json", {"cases": index, "queries": checked})
    runtime.finish()
    metrics = aggregate(records)
    assert metrics == read_json(out / "summary.json")
    write_json(
        out / "audit.json",
        {
            "passed": True,
            "cases": len(records),
            "queries": checked,
            "all_raw_generations_exact": True,
            "all_saved_edits_replayed": True,
            "original_evaluator_rescored": True,
            "metrics": metrics,
        },
    )
    write_json(out / "status.json", {"state": "complete", "step": len(records)})
