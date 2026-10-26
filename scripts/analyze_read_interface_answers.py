"""knowledge-read-interface-v2 stage R1: classify saved U1-U3 answers (zero training).

Re-reads the full generations of knowledge-read-interface-development-v1 and the layer-write
arms of knowledge-read-interface-followup-v1. Only main facts (no calibration) written by the
method (U0 strict correct) enter; every written fact has the same number of rows per use, so
row rates equal fact means.

U2 asks for country(v). A wrong U2 answer is "named_v" when it names the city v (the fact fired
but the country operation was dropped), "country_h" when it gives the subject's own country (the
fact did not fire), else "other". A wrong U3 answer (gold v) is "h_or_c" when it names h or C.
For CoT rows, "v_written" records whether the reply mentions v anywhere.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from llm_memory_editability import read_interface_data as D

PERSISTENT = Path(
    "/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability"
)
SOURCES = {
    "dev": "knowledge-read-interface-development-v1/runs",
    "second": "knowledge-read-interface-development-v1/runs-confirm",
    "layer-dev": "knowledge-read-interface-followup-v1/layer-write/dev",
    "layer-second": "knowledge-read-interface-followup-v1/layer-write/second",
}
MODELS = ("qwen3-1.7b", "qwen3.5-2b")
U1_ORDER = {"U1.0": "h_first", "U1.1": "h_first", "U1.2": "h_first", "U1.3": "r_first"}


def classify(row, fact, graders):
    if row["strict"]:
        return "correct"
    answer = row.get("answer") or ""
    cities = graders["city"].mentions(answer)
    countries = graders["country"].mentions(answer)
    if row["use"].startswith("U2"):
        if fact["v"] in cities:
            return "named_v"
        if countries and countries[0] == fact["country_h"]:
            return "country_h"
        return "other"
    if fact["h"] in cities or fact["country_h"] in countries:
        return "h_or_c"
    return "other"


def summarize(rows):
    n = len(rows)
    return {"rows": n, **{k: v / n for k, v in sorted(Counter(rows).items())}} if n else {"rows": 0}


def analyze(run_dir, graders):
    facts = {f["fid"]: f for f in json.loads((run_dir / "facts.json").read_text())["facts"]}
    out = {}
    for path in sorted((run_dir / "eval").glob("*.json")):
        method = path.stem
        uses = json.loads(path.read_text())["uses"]
        main = [r for r in uses if facts[r["fid"]]["split"] != "calibration"]
        if method in ("base", "context"):
            written = {r["fid"] for r in main}
        else:
            written = {r["fid"] for r in main if r["use"] == "U0" and r["strict"]}
        kinds, cot_v, u1 = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in main:
            if r["fid"] not in written:
                continue
            fact = facts[r["fid"]]
            use = r["use"].split(".")[0]
            if use in ("U2", "U3"):
                key = f"{use}/{r['mode']}"
                kinds[key].append(classify(r, fact, graders))
                if r["mode"] == "cot":
                    cot_v[key].append(fact["v"] in graders["city"].mentions(r.get("reply") or ""))
            elif r["use"] in U1_ORDER:
                u1[r["use"]].append(bool(r["strict"]))
        out[method] = {
            "written_facts": len(written),
            "answer_types": {k: summarize(v) for k, v in sorted(kinds.items())},
            "cot_v_written": {k: sum(v) / len(v) for k, v in sorted(cot_v.items())},
            "u1_by_template": {
                k: {"order": U1_ORDER[k], "strict": sum(v) / len(v)} for k, v in sorted(u1.items())
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PERSISTENT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    graders = D.graders()
    result = {
        "description": __doc__,
        "sources": {k: str(args.root / v) for k, v in SOURCES.items()},
        "results": {
            source: {model: analyze(args.root / rel / model, graders) for model in MODELS}
            for source, rel in SOURCES.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
