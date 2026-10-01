"""Aggregate complete frozen endpoint diagnostics without selecting cases."""

import csv
from collections import defaultdict
from pathlib import Path

from llm_memory_editability.bios_original_data import load_json, write_json
from llm_memory_editability.bios_original_diagnostics import parse_date

ROOT = Path(__file__).resolve().parent
plan = load_json(ROOT / "plan.json")
completed = [load_json(ROOT / "endpoints" / e["id"] / "complete.json") for e in plan["endpoints"]]
rows = []
extraction_rows = []
pooled = defaultdict(lambda: defaultdict(int))
for result in completed:
    if result["diagnostics"] is None:
        continue
    endpoint = result["endpoint"]
    extractions = load_json(ROOT / "endpoints" / endpoint["id"] / "extractions.json")
    for view in plan["views"]:
        selected = [r for r in extractions if r["view"] == view]
        parsed = [(parse_date(r["generated"]), parse_date(r["target"])) for r in selected]
        extraction_rows.append(
            dict(
                world=endpoint["world"],
                condition=endpoint["condition"],
                view=view,
                n=len(selected),
                full_date_correct=sum(r["correct"] for r in selected),
                parseable=sum(r["parseable"] for r in selected),
                month_correct=sum(p is not None and p[1] == t[1] for p, t in parsed),
                year_correct=sum(p is not None and p[0] == t[0] for p, t in parsed),
            )
        )
    for key, values in result["diagnostics"]["totals"].items():
        task, view, mode = key.split("/")
        values = dict(values)
        if mode == "autonomous_rule":
            values["parseable_end_to_end_correct"] = values["inputs_parseable_correct"]
            values["parseable_end_to_end_total"] = values["n"]
            values["malformed_input_correct"] = (
                values["correct"] - values["inputs_parseable_correct"]
            )
        rows.append(
            dict(
                world=endpoint["world"],
                condition=endpoint["condition"],
                task=task,
                view=int(view[-1]),
                mode=mode,
                **values,
            )
        )
        bucket = pooled[f"{endpoint['condition']}/{key}"]
        for field, value in values.items():
            bucket[field] += value
with (ROOT / "endpoint-summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
    writer.writeheader()
    writer.writerows(rows)
with (ROOT / "extraction-summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(extraction_rows[0]))
    writer.writeheader()
    writer.writerows(extraction_rows)
summary = dict(
    completed=len(completed),
    expected=len(plan["endpoints"]),
    audit_rows=sum(r["audit"]["rows"] for r in completed),
    audit_passed=sum(r["audit"]["status"] == "passed" for r in completed),
    generation_mismatches=sum(r["audit"]["generation_mismatches"] for r in completed),
    max_abs_full_logp_difference=max(r["audit"]["max_abs_full_logp_difference"] for r in completed),
    diagnostic_rows=sum(r["diagnostics"]["rows"] for r in completed if r["diagnostics"]),
    extraction_rows=sum(r["diagnostics"]["extraction_rows"] for r in completed if r["diagnostics"]),
    endpoint_seconds=sum(r["seconds"] for r in completed),
    peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in completed),
    pooled=dict(pooled),
    aggregation="Each cell pools 24 people per world; equivalent to equal-world mean. "
    "Views and reversed partner queries are repeated observations, not new people.",
)
replays = sorted((ROOT / "rule-replay").glob("*.json"))
summary["rule_replay_endpoints"] = len(replays)
summary["rule_replay"] = {p.stem: load_json(p)["summary"] for p in replays}
write_json(ROOT / "summary.json", summary)
print({k: v for k, v in summary.items() if k not in ("pooled", "rule_replay")})
