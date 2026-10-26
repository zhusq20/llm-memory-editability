"""Outcome-independent facts and paired contexts for the bounded ICA analysis.

The original ``adapt``/``test`` labels are provenance, not claims about whether
the model trained on a fact. Calibration targets retain their existing split.
Natural retention questions are returned separately and never select facts,
donors, distractors, or components.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

from llm_memory_editability import read_interface_data as D

DEFAULT_SEED = 20261023
USES = ("U0", "U1.0", "U2.0", "U3.0")


@dataclass(frozen=True)
class FactSelection:
    main: tuple[D.Fact, ...]
    calibration: tuple[D.Fact, ...]
    seed: int


@dataclass(frozen=True)
class WrongValue:
    fact: D.Fact
    source_fid: str


@dataclass(frozen=True)
class ContextPair:
    clean: str
    corrupt: str
    wrong_fact: D.Fact
    wrong_source_fid: str
    distractor_fids: tuple[str, ...]
    target_index: int


def _rank(seed: int, purpose: str, *ids: str) -> str:
    payload = json.dumps([seed, purpose, *ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bank(facts: Iterable[D.Fact]) -> tuple[D.Fact, ...]:
    facts = tuple(facts)
    if len({f.fid for f in facts}) != len(facts):
        raise ValueError("Fact IDs must be unique")
    return facts


def select_facts(
    facts: Iterable[D.Fact], n: int = 48, seed: int = DEFAULT_SEED
) -> FactSelection:
    """Select exactly ``n`` main facts in whole subject groups, independent of input order.

    All existing calibration facts are reserved. A SHA256 order over the seed
    and original IDs chooses main subject groups; exact-size subset selection
    preserves groups even if a future fact bank has unequal group sizes.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    facts = _bank(facts)
    calibration = tuple(
        sorted(
            (f for f in facts if f.split == "calibration"),
            key=lambda f: (_rank(seed, "calibration", f.fid), f.fid),
        )
    )
    calibration_subjects = {f.h for f in calibration}
    groups = defaultdict(list)
    for f in facts:
        if f.split != "calibration":
            if f.h in calibration_subjects:
                raise ValueError(f"Subject occurs in calibration and main splits: {f.h}")
            groups[f.h].append(f)
    ordered = [tuple(sorted(group, key=lambda f: f.fid)) for group in groups.values()]
    ordered.sort(key=lambda group: _rank(seed, "main-subject", *(f.fid for f in group)))

    # The first subset reaching a size is retained, so no dictionary iteration
    # order or content-derived score can affect a later tie.
    subsets: dict[int, tuple[int, ...]] = {0: ()}
    for i, group in enumerate(ordered):
        for size, indices in sorted(tuple(subsets.items()), reverse=True):
            total = size + len(group)
            if total <= n and total not in subsets:
                subsets[total] = (*indices, i)
    if n not in subsets:
        raise ValueError(f"Cannot select exactly {n} main facts while preserving subject groups")
    main = tuple(f for i in subsets[n] for f in ordered[i])
    return FactSelection(main=main, calibration=calibration, seed=seed)


def evaluation_rows(fact: D.Fact) -> list[dict]:
    """Reuse the four historical direct prompts, golds, and grading exclusions."""
    rows = {
        row["use"]: row
        for row in D.use_queries(fact)
        if row["mode"] == "direct" and row["use"] in USES
    }
    if set(rows) != set(USES):
        raise ValueError(f"Missing evaluation uses for {fact.fid}")
    # Matches read_interface_stages.exclusions without importing torch.
    return [
        {
            **rows[use],
            "fact_id": fact.fid,
            "split": fact.split,
            "exclude": sorted({fact.h, fact.country_h} | {rows[use]["subject"] or fact.h}),
        }
        for use in USES
    ]


def retention_row(fact: D.Fact, seed: int = DEFAULT_SEED) -> dict:
    """One unrelated natural city-country question; never a feature-selection row.

    The screening query ID is preserved. No historical screening outcomes are
    read: the query is selected solely by a hash of the seed, fact ID and qid.
    """
    blocked_cities = {fact.h, fact.v}
    blocked_countries = {fact.country_h, fact.country_v}
    candidates = [
        row
        for row in D.screening_queries()
        if row["qid"].startswith("city_country/")
        and row["qid"].removeprefix("city_country/") not in blocked_cities
        and row["gold"] not in blocked_countries
    ]
    if not candidates:
        raise ValueError(f"No unrelated retention query for {fact.fid}")
    row = min(candidates, key=lambda r: (_rank(seed, "retention", fact.fid, r["qid"]), r["qid"]))
    return {
        **row,
        "prompt": row["prompt"] + D.DIRECT,
        "fact_id": fact.fid,
        "fid": fact.fid,
        "split": fact.split,
        "use": "retention",
        "role": "retention",
        "mode": "direct",
        "exclude": [],
        "feature_selection_eligible": False,
    }


def wrong_value_fact(
    fact: D.Fact, facts: Iterable[D.Fact], seed: int = DEFAULT_SEED
) -> WrongValue:
    """Replace only the target value and its country, using a recorded main donor.

    Prefer a same-relation donor when available. The donor's subject is never
    substituted into the target. Calibration facts cannot serve as donors.
    """
    candidates = [
        f
        for f in _bank(facts)
        if f.split != "calibration"
        and f.fid != fact.fid
        and f.v != fact.v
        and f.country_v != fact.country_v
    ]
    same_relation = [f for f in candidates if f.relation == fact.relation]
    candidates = same_relation or candidates
    if not candidates:
        raise ValueError(f"No wrong value with a different country for {fact.fid}")
    source = min(candidates, key=lambda f: (_rank(seed, "wrong-value", fact.fid, f.fid), f.fid))
    return WrongValue(
        fact=replace(fact, v=source.v, country_v=source.country_v), source_fid=source.fid
    )


def line(fact: D.Fact) -> str:
    return "Fact: " + D.context_statement(fact)


def controlled_contexts(
    fact: D.Fact, facts: Iterable[D.Fact], seed: int = DEFAULT_SEED
) -> ContextPair:
    """Three distractors plus a target, with only the target value corrupted.

    Distractors have the target relation, distinct subjects and values, and
    neither target value. Calibration targets may use main facts as unrelated
    distractors/donors; calibration facts never supply main contexts. The
    returned IDs preserve that provenance. No question or response is used to
    choose the distractors, their shared order, or the corrupt value.
    """
    facts = _bank(facts)
    wrong = wrong_value_fact(fact, facts, seed)
    candidates = sorted(
        (
            f
            for f in facts
            if f.split != "calibration"
            and f.fid != fact.fid
            and f.relation == fact.relation
            and f.h != fact.h
            and f.v not in {fact.v, wrong.fact.v}
        ),
        key=lambda f: (_rank(seed, "distractor", fact.fid, f.fid), f.fid),
    )
    distractors = []
    used_subjects, used_values = {fact.h}, {fact.v, wrong.fact.v}
    for candidate in candidates:
        if candidate.h in used_subjects or candidate.v in used_values:
            continue
        distractors.append(candidate)
        used_subjects.add(candidate.h)
        used_values.add(candidate.v)
        if len(distractors) == 3:
            break
    if len(distractors) != 3:
        raise ValueError(f"Need three distinct same-relation distractors for {fact.fid}")
    ordered = sorted(
        [fact, *distractors],
        key=lambda f: (_rank(seed, "context-order", fact.fid, f.fid), f.fid),
    )
    target_index = next(i for i, f in enumerate(ordered) if f.fid == fact.fid)
    clean_lines = [line(f) for f in ordered]
    corrupt_lines = list(clean_lines)
    corrupt_lines[target_index] = line(wrong.fact)
    return ContextPair(
        clean="\n".join(clean_lines),
        corrupt="\n".join(corrupt_lines),
        wrong_fact=wrong.fact,
        wrong_source_fid=wrong.source_fid,
        distractor_fids=tuple(f.fid for f in ordered if f.fid != fact.fid),
        target_index=target_index,
    )


def grade(text: str, row: dict, grader_bank: dict | None = None) -> dict:
    """Apply the existing strict/lenient entity grader to a returned row."""
    grader_bank = D.graders() if grader_bank is None else grader_bank
    return grader_bank[row["type"]].grade(text, row["gold"], exclude=row["exclude"])
