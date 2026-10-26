"""Offline data contract for the existing Knowledge A/B/C coverage comparison.

The constructor and training functions never read evaluation rows. Gold answers are
symbolic QA checks, not sampled teacher responses. All new facts in a world are written
into one model. Real city/country premises reuse the project's local frozen table.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from dataclasses import asdict, dataclass

from .read_interface_data import city_table

SCHEMA_VERSION = "knowledge-use-coverage-v1"
BACKGROUND_SOURCE = "src/llm_memory_editability/read_interface_data.py::city_table"


def natural_retention_clusters(records):
    """Map frozen retention IDs to country blocks, never treating templates as IID.

    Only the released question metadata define these blocks; predictions and
    correctness are ignored. Unidentifiable or conflicting mappings remain None.
    """
    country_by_city = {}
    for row in records:
        identity = row.get("original_source_id", row.get("id", ""))
        if identity.startswith(("ret/capital_rev/", "ret/city_country/")):
            city = identity.rsplit("/", 1)[-1]
            gold = row.get("gold")
            if isinstance(gold, str) and gold.strip():
                country_by_city.setdefault(city, set()).add(gold.strip())
    result = {}
    for row in records:
        identity = row.get("original_source_id", row.get("id", ""))
        parts = identity.split("/")
        country = None
        if identity.startswith("ret/currency/") and len(parts) == 3 and parts[-1]:
            country = parts[-1]
        elif identity.startswith(("ret/", "nat2/", "nat2short/")):
            candidates = country_by_city.get(parts[-1], set())
            if len(candidates) == 1:
                country = next(iter(candidates))
        result[row["id"]] = f"natural-retention-country/{country}" if country else None
    return result


CITY_NAMES = (
    "Paris",
    "Berlin",
    "Rome",
    "Madrid",
    "Lisbon",
    "Dublin",
    "Amsterdam",
    "Vienna",
    "Stockholm",
    "Oslo",
    "Copenhagen",
    "Helsinki",
    "Tokyo",
    "Beijing",
    "Seoul",
    "Bangkok",
    "Hanoi",
    "Ottawa",
    "Brasília",
    "Buenos Aires",
    "Santiago",
    "Lima",
    "Bogotá",
    "Cairo",
    "Nairobi",
    "Canberra",
    "Wellington",
)
SYLLABLES = (
    "Al",
    "Bel",
    "Cor",
    "Del",
    "El",
    "Fen",
    "Gal",
    "Hal",
    "Il",
    "Jar",
    "Kel",
    "Lor",
    "Mar",
    "Nel",
    "Or",
    "Pel",
    "Quel",
    "Ral",
    "Sel",
    "Tal",
    "Ul",
    "Val",
    "Wel",
    "Xel",
    "Yar",
    "Zel",
    "Av",
    "Bren",
    "Cyr",
    "Drin",
    "Evr",
    "Fjor",
)

# The first template is the explicitly identified trained-question diagnostic.
TRAIN_ATOMIC = (
    "What is the {relation} of {subject}?",
    "Name the {relation} assigned to {subject}.",
    "Which {kind} is listed as the {relation} for {subject}?",
    "Tell me {subject}'s {relation}.",
    "According to the registry, what is {subject}'s {relation}?",
    "Identify the {relation} recorded for {subject}.",
    "For {subject}, give the name of its {relation}.",
    "The registry associates {subject} with which {relation}?",
)
TEST_ATOMIC = (
    "I am checking {subject}'s entry. Which {kind} fills its {relation} field?",
    "A directory lookup for {subject}'s {relation} returns what name?",
)
DIRECT = " Return only the answer."
STEP = (
    " Work through the relevant facts step by step. End with a line 'Final answer:'"
    " followed by only the answer."
)
RULE = (
    "This is a fictional research registry. Each organization has exactly one assigned office,"
    " each office is assigned to exactly one organization, and each office has one base city."
)


@dataclass(frozen=True)
class FactPair:
    id: str
    organization: str
    office: str
    city: str
    country: str
    reference_city: str
    reference_country: str
    other_city: str
    other_country: str

    @property
    def assignment_id(self):
        return f"{self.id}/assignment"

    @property
    def location_id(self):
        return f"{self.id}/location"


@dataclass(frozen=True)
class World:
    cohort: str
    seed: int
    pairs: tuple[FactPair, ...]

    def manifest(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "cohort": self.cohort,
            "seed": self.seed,
            "pair_count": len(self.pairs),
            "new_fact_count": 2 * len(self.pairs),
            "background_source": BACKGROUND_SOURCE,
            "city_selection": {
                "rule": "local table has a distinct city in the same country; no model screening",
                "candidate_cities_before_filter": len(CITY_NAMES),
                "candidate_cities_after_filter": len(_eligible_cities()),
                "used_target_cities": sorted({p.city for p in self.pairs}),
            },
            "pairs": [asdict(p) for p in self.pairs],
            "rewrite_permutation": [i ^ 1 for i in range(len(self.pairs))],
            "rewrite_analysis_unit": "disjoint two-pair swap block",
            "answer_source": "symbolic gold for validation; teacher sampling is separate",
        }


def _background_id(city):
    return "background/city-country/" + city


def _eligible_cities():
    cities = city_table()
    return [
        c
        for c in CITY_NAMES
        if any(
            other != c and row["country"] == cities[c]["country"] for other, row in cities.items()
        )
    ]


def make_world(cohort="development", pair_count=32, seed=20261025):
    """Create 2 * pair_count additions; dev/formal new entity pools are disjoint.

    Adjacent pairs have different cities and countries, so the fixed office swap
    changes both dependent city and country answers. Shared background cities are
    intentionally permitted across cohorts. Only development may select settings.
    """
    if cohort not in {"development", "formal"}:
        raise ValueError("cohort must be development or formal")
    if pair_count < 2 or pair_count % 2 or pair_count > 128:
        raise ValueError("pair_count must be even and between 2 and 128")
    rng = random.Random(f"{SCHEMA_VERSION}/{cohort}/{seed}")
    parity = int(cohort == "formal")
    names = sorted({a + b.lower() + "en" for a in SYLLABLES for b in SYLLABLES})
    # Partition the final surface form, not its syllable decomposition.
    names = [
        name for name in names if int(hashlib.sha256(name.encode()).hexdigest(), 16) % 2 == parity
    ]
    names = rng.sample(names, 2 * pair_count)
    cities = city_table()
    pairs = []
    city_pool = _eligible_cities()
    rng.shuffle(city_pool)
    for i in range(pair_count):
        candidates = [
            c for c in city_pool if not i % 2 or cities[c]["country"] != pairs[-1].country
        ]
        city = candidates[rng.randrange(len(candidates))]
        country = cities[city]["country"]
        other = rng.choice([c for c in city_pool if cities[c]["country"] != country])
        # The same-country comparison uses a real second city when available.
        same_country = [c for c, row in cities.items() if row["country"] == country and c != city]
        reference = rng.choice(same_country)
        pairs.append(
            FactPair(
                id=f"{cohort}/pair-{i:03d}",
                organization=names[i] + " Research Institute",
                office=names[pair_count + i] + " Field Office",
                city=city,
                country=country,
                reference_city=reference,
                reference_country=country,
                other_city=other,
                other_country=cities[other]["country"],
            )
        )
    return World(cohort, seed, tuple(pairs))


def _target(world, i, stage):
    if stage not in {"addition", "rewrite"}:
        raise ValueError("stage must be addition or rewrite")
    return world.pairs[i if stage == "addition" else i ^ 1]


def _context(world, i, stage="addition"):
    p, target = world.pairs[i], _target(world, i, stage)
    facts = {
        p.assignment_id: f"{p.organization}'s assigned office is {target.office}.",
        p.location_id: f"{p.office}'s base city is {p.city}.",
        target.location_id: f"{target.office}'s base city is {target.city}.",
    }
    for q in (p, target):
        for city, country in (
            (q.city, q.country),
            (q.reference_city, q.reference_country),
            (q.other_city, q.other_country),
        ):
            facts[_background_id(city)] = f"{city} is located in {country}."
    return RULE + "\n" + "\n".join(facts.values()), list(facts)


def _row(
    world,
    i,
    key,
    question,
    gold,
    category,
    split,
    fact_ids,
    *,
    stage="addition",
    variant="direct",
    aliases=(),
    chain=(),
    extra=None,
):
    context, context_fact_ids = _context(world, i, stage)
    row = {
        "id": f"{world.pairs[i].id}/{stage}/{split}/{key}/{variant}",
        "fact_ids": list(fact_ids),
        "question": question,
        "context": context,
        "gold": gold,
        "aliases": sorted({gold, *aliases}),
        "category": category,
        "split": split,
        "variant": variant,
        "mode": variant,
        "stage": stage,
        "cohort": world.cohort,
        "pair_id": world.pairs[i].id,
        "cluster_id": (
            world.pairs[i].id if stage == "addition" else f"{world.cohort}/swap-{i // 2:03d}"
        ),
        "context_fact_ids": context_fact_ids,
        "background_fact_ids": [f for f in fact_ids if f.startswith("background/")],
        "chain": list(chain),
        "gold_source": "offline_symbolic_truth",
    }
    if extra:
        row.update(extra)
    return row


def _atomic_question(subject, kind, relation, template=0, *, heldout=False):
    templates = TEST_ATOMIC if heldout else TRAIN_ATOMIC
    return templates[template].format(subject=subject, kind=kind, relation=relation)


def _atomic(world, i, which, template, *, split="train", heldout=False, stage="addition"):
    p, target = world.pairs[i], _target(world, i, stage)
    if which == "assignment":
        subject, kind, relation, gold, fid = (
            p.organization,
            "office",
            "assigned office",
            target.office,
            p.assignment_id,
        )
    else:
        subject, kind, relation, gold, fid = (p.office, "city", "base city", p.city, p.location_id)
    question = _atomic_question(subject, kind, relation, template, heldout=heldout)
    aliases = city_table()[gold]["aliases"] if which == "location" else []
    return _row(
        world,
        i,
        f"{which}-{'test' if heldout else 'train'}-{template}",
        question + DIRECT,
        gold,
        "atomic_unseen" if heldout else "atomic_seen",
        split,
        [fid],
        stage=stage,
        aliases=aliases,
        extra={
            "template_bank": "test" if heldout else "train",
            "previous_gold": p.office if which == "assignment" else p.city,
            "answer_changed": which == "assignment" and stage == "rewrite",
        },
    )


def training_rows(world, arm):
    """A has 2 unique QA/pair; B and C have 16. Token matching is external.

    C contains 8 atomic and 8 use QA. A/B/C receive identical source contexts
    for each pair. No evaluation function or evaluation template is consulted.
    Responses must be sampled and frozen separately for PD/SFT.
    """
    if arm not in {"A", "B", "C"}:
        raise ValueError("arm must be A, B, or C")
    rows = []
    n = {"A": 1, "B": 8, "C": 4}[arm]
    for i, p in enumerate(world.pairs):
        for which in ("assignment", "location"):
            rows.extend(_atomic(world, i, which, t) for t in range(n))
        if arm != "C":
            continue
        for t, question in enumerate(
            (
                f"In which country is {p.office}'s base city located?",
                f"Name the country containing the base city of {p.office}.",
            )
        ):
            rows.append(
                _row(
                    world,
                    i,
                    f"country-{t}",
                    question + DIRECT,
                    p.country,
                    "composition",
                    "train",
                    [p.location_id, _background_id(p.city)],
                )
            )
        foil = world.pairs[i ^ 1].office
        for label, candidate, answer in (("true", p.office, "yes"), ("false", foil, "no")):
            rows.append(
                _row(
                    world,
                    i,
                    f"assignment-judge-{label}",
                    f"Is {candidate} the assigned office of {p.organization}?" + DIRECT,
                    answer,
                    "judgment",
                    "train",
                    [p.assignment_id],
                )
            )
        for label, candidate, answer in (("true", p.city, "yes"), ("false", p.other_city, "no")):
            rows.append(
                _row(
                    world,
                    i,
                    f"location-judge-{label}",
                    f"Is {p.office} based in {candidate}?" + DIRECT,
                    answer,
                    "judgment",
                    "train",
                    [p.location_id],
                )
            )
        for label, city, answer in (
            ("same", p.reference_city, "yes"),
            ("different", p.other_city, "no"),
        ):
            rows.append(
                _row(
                    world,
                    i,
                    f"compare-{label}",
                    f"Are {p.office}'s base city and {city} in the same country?" + DIRECT,
                    answer,
                    "comparison",
                    "train",
                    [p.location_id, _background_id(p.city), _background_id(city)],
                )
            )
    for row in rows:
        row["arm"] = arm
        row["category"] = "atomic" if row["category"] == "atomic_seen" else row["category"]
    return rows


def rewrite_training_rows(world):
    """Identical atomic-only second update for every arm, with no dependent gold.

    Contexts here contain only the new assignment statement plus the schema rule.
    The first update's office/city facts remain in model weights. A staged trainer
    must continue the corresponding first-update model, not reset to the base.
    """
    rows = []
    for i, p in enumerate(world.pairs):
        row = _atomic(world, i, "assignment", 0, stage="rewrite")
        row["context"] = (
            RULE + f"\n{p.organization}'s assigned office is {world.pairs[i ^ 1].office}."
        )
        row["context_fact_ids"] = [p.assignment_id]
        row["category"] = "rewrite_atomic"
        rows.append(row)
    return rows


def _modes(row):
    direct = dict(row)
    step = dict(row)
    step["id"] = step["id"].removesuffix("/direct") + "/step_by_step"
    step["question"] = step["question"].removesuffix(DIRECT) + STEP
    step["variant"] = step["mode"] = "step_by_step"
    return [direct, step]


def _chain(*questions):
    # No gold bridge or candidate list is provided. Substitute the model's own
    # preceding answer, not a reference entity, into each successive question.
    return [{"question": q + DIRECT, "answer_slot": f"answer{i}"} for i, q in enumerate(questions)]


def evaluation_rows(world, stage="addition"):
    """Frozen evaluation families, each direct and explicit step by step.

    The same rows support closed-book and source-present evaluation; the caller
    must label which mode it ran. Except atomic_seen, stems are absent from all
    training arms. 'seen_use' means trained only in C. Other families are held out
    in all arms. Rewrite golds follow the fixed permutation, with old gold retained.
    """
    rows = []
    for i, p in enumerate(world.pairs):
        target = _target(world, i, stage)
        for which in ("assignment", "location"):
            rows.extend(_modes(_atomic(world, i, which, 0, split="evaluation", stage=stage)))
            for t in range(len(TEST_ATOMIC)):
                rows.extend(
                    _modes(
                        _atomic(world, i, which, t, split="evaluation", heldout=True, stage=stage)
                    )
                )
        cases = [
            (
                "country",
                f"Name the nation in which the registered base city of {p.office} lies.",
                p.country,
                "seen_use_new_wording",
                [p.location_id, _background_id(p.city)],
                _chain(f"What is {p.office}'s base city?", "In which country is {answer0}?"),
                p.country,
            ),
            (
                "assignment-judge-true",
                f"Does the registry assign {p.organization} to {p.office}?",
                "yes" if stage == "addition" else "no",
                "seen_use_new_wording",
                [p.assignment_id],
                (),
                "yes",
            ),
            (
                "assignment-judge-false",
                f"Does the registry assign {p.organization} to {world.pairs[i ^ 1].office}?",
                "no" if stage == "addition" else "yes",
                "seen_use_new_wording",
                [p.assignment_id],
                (),
                "no",
            ),
            (
                "location-judge-true",
                f"Would {p.city} be a correct base-city entry for {p.office}?",
                "yes",
                "seen_use_new_wording",
                [p.location_id],
                (),
                "yes",
            ),
            (
                "location-judge-false",
                f"Would {p.other_city} be a correct base-city entry for {p.office}?",
                "no",
                "seen_use_new_wording",
                [p.location_id],
                (),
                "no",
            ),
            (
                "compare-same",
                f"Do {p.reference_city} and the city hosting {p.office} share a country?",
                "yes",
                "seen_use_new_wording",
                [p.location_id, _background_id(p.city), _background_id(p.reference_city)],
                (),
                "yes",
            ),
            (
                "compare-different",
                f"Do {p.other_city} and the city hosting {p.office} share a country?",
                "no",
                "seen_use_new_wording",
                [p.location_id, _background_id(p.city), _background_id(p.other_city)],
                (),
                "no",
            ),
            (
                "double-new",
                f"In which city is the assigned office of {p.organization} based?",
                target.city,
                "double_new_composition",
                [p.assignment_id, target.location_id],
                _chain(
                    f"What office is assigned to {p.organization}?",
                    "What is the base city of {answer0}?",
                ),
                p.city,
            ),
            (
                "new-new-background",
                f"Which country contains the base city of the office assigned to {p.organization}?",
                target.country,
                "heldout_relation_composition",
                [p.assignment_id, target.location_id, _background_id(target.city)],
                _chain(
                    f"What office is assigned to {p.organization}?",
                    "What is the base city of {answer0}?",
                    "In which country is {answer1}?",
                ),
                p.country,
            ),
            (
                "inverse-assignment",
                f"In this registry, which organization has {p.office} as its assigned office?",
                target.organization,
                "heldout_use_category",
                [target.assignment_id],
                (),
                p.organization,
            ),
        ]
        for key, question, gold, category, fact_ids, chain, old_gold in cases:
            aliases = city_table()[gold]["aliases"] if category == "double_new_composition" else []
            row = _row(
                world,
                i,
                key,
                question + DIRECT,
                gold,
                category,
                "evaluation",
                fact_ids,
                stage=stage,
                aliases=aliases,
                chain=chain,
                extra={
                    "previous_gold": old_gold,
                    "answer_changed": gold != old_gold,
                    "template_bank": "test",
                    "trained_use_in_arms": ["C"] if category == "seen_use_new_wording" else [],
                },
            )
            if key == "inverse-assignment" and stage == "rewrite":
                # Keep the queried office and row identity fixed across stages;
                # its new owning organization is described in the partner context.
                row["context"], row["context_fact_ids"] = _context(world, i ^ 1, stage)
            rows.extend(_modes(row))
    return rows


def background_rows(world):
    """Old-knowledge premises, measured before updates without filtering the world."""
    cities = city_table()
    names = sorted({name for p in world.pairs for name in (p.city, p.reference_city, p.other_city)})
    return [
        {
            "id": f"{world.cohort}/premise/{city}/{mode}",
            "fact_ids": [_background_id(city)],
            "question": f"Which country contains the city of {city}?" + suffix,
            "context": f"{city} is located in {cities[city]['country']}.",
            "gold": cities[city]["country"],
            "aliases": [cities[city]["country"]],
            "category": "background_premise",
            "split": "evaluation",
            "variant": mode,
            "mode": mode,
            "cohort": world.cohort,
            "pair_id": None,
            "cluster_id": _background_id(city),
            "chain": [],
            "gold_source": "local_frozen_city_table",
        }
        for city in names
        for mode, suffix in (("direct", DIRECT), ("step_by_step", STEP))
    ]


def retention_rows(world, stage="addition"):
    """Related unchanged facts; background exposure is disclosed, not called unseen.

    Office locations are new at stage one and become old unchanged knowledge at
    rewrite. Background facts were available in all teacher contexts. General old
    QA (including the existing 317-question pool) must be attached independently.
    """
    rows = background_rows(world)
    for row in rows:
        row["category"] = "retention_related_background"
        row["background_seen_in_teacher_context"] = True
        row["split"] = "retention"
        row["id"] = row["id"].replace("/premise/", "/retention/")
    if stage == "rewrite":
        for row in evaluation_rows(world, stage):
            if row["category"] == "atomic_unseen" and row["fact_ids"][0].endswith("/location"):
                row["category"] = "retention_unchanged_new_location"
                row["split"] = "retention"
                row["id"] = row["id"].replace("/evaluation/", "/retention/")
                rows.append(row)
    return rows


def replay_rows(count=32):
    """Fallback arithmetic protection prompts R, never a broad ability benchmark.

    Use frozen base-model continuations and context-free KL, not the symbolic gold
    as hard replay labels. Prefer a validated historical general QA replay pool.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rows = []
    for i in range(count):
        a, b = 31 + 2 * i, 7 + i
        addition = i % 2 == 0
        expression = f"{a} {'+' if addition else '-'} {b}"
        gold = str(a + b if addition else a - b)
        rows.append(
            {
                "id": f"replay/arithmetic-{i:03d}",
                "fact_ids": [f"arithmetic/{expression}"],
                "question": f"Calculate {expression}." + DIRECT,
                "context": "",
                "gold": gold,
                "aliases": [gold],
                "category": "arithmetic_replay",
                "split": "replay",
                "variant": "direct",
                "mode": "direct",
                "chain": [],
                "gold_source": "offline_symbolic_truth_not_training_target",
            }
        )
    return rows


def general_retention_rows(count=32):
    """Small disjoint arithmetic U supplement; retain the existing natural old-QA U."""
    if count < 1:
        raise ValueError("count must be positive")
    rows = []
    for i in range(count):
        a, b = 12 + i, 3 + i % 7
        gold = str(a * b)
        rows.append(
            {
                "id": f"retention/arithmetic-{i:03d}",
                "fact_ids": [f"arithmetic/{a} * {b}"],
                "question": f"What is {a} multiplied by {b}?" + DIRECT,
                "context": "",
                "gold": gold,
                "aliases": [gold],
                "category": "retention_arithmetic",
                "split": "retention",
                "variant": "direct",
                "mode": "direct",
                "chain": [],
                "gold_source": "offline_symbolic_truth",
            }
        )
    return rows


def validate_retention_split(replay, retention, target_fact_ids=()):
    """Reject question/fact reuse between protected R and independently held-out U."""
    replay_questions = {r["question"].strip() for r in replay}
    retention_questions = {r["question"].strip() for r in retention}
    replay_facts = {f for r in replay for f in r["fact_ids"]}
    retention_facts = {f for r in retention for f in r["fact_ids"]}
    if replay_questions & retention_questions or replay_facts & retention_facts:
        raise ValueError("replay R and retention U overlap")
    if set(target_fact_ids) & (replay_facts | retention_facts):
        raise ValueError("changed target facts cannot be preservation constraints")


def audit_world(world):
    """Check actual dependencies/contexts, template isolation and rewrite truth."""
    arms = {a: training_rows(world, a) for a in "ABC"}
    for i, p in enumerate(world.pairs):
        expected = _context(world, i)[0]
        for rows in arms.values():
            selected = [r for r in rows if r["pair_id"] == p.id]
            assert selected and all(r["context"] == expected for r in selected)
            assert all(set(r["fact_ids"]) <= set(r["context_fact_ids"]) for r in selected)
        assert p.city != world.pairs[i ^ 1].city
        assert p.country != world.pairs[i ^ 1].country
    training_stems = {r["question"].removesuffix(DIRECT) for rows in arms.values() for r in rows}
    tests = evaluation_rows(world)
    for row in tests:
        if row["category"] != "atomic_seen":
            stem = row["question"].removesuffix(DIRECT).removesuffix(STEP)
            assert stem not in training_stems
        assert set(row["fact_ids"]) <= set(row["context_fact_ids"])
        assert all("gold" not in step for step in row["chain"])
    for rows in [*arms.values(), tests, evaluation_rows(world, "rewrite")]:
        assert len({r["id"] for r in rows}) == len(rows)
    rewrites = rewrite_training_rows(world)
    assert all(
        len(r["fact_ids"]) == 1 and r["fact_ids"][0].endswith("/assignment") for r in rewrites
    )
    validate_retention_split(replay_rows(), general_retention_rows())
    return {
        "passed": True,
        "pair_count": len(world.pairs),
        "unique_training_rows": {a: len(rows) for a, rows in arms.items()},
        "evaluation_rows_per_stage": len(tests),
        "rewrite_training_rows": len(rewrites),
        "background_equal_across_arms": True,
        "dual_new_training_rows": 0,
        "teacher_generation_performed": False,
    }


def exact_answer_match(text, row):
    """Conservative final-answer scoring; a mention in reasoning is not success.

    This deliberately does not guess that arbitrary prose is semantically correct.
    Store original generations for separately reported semantic/format review.
    """
    return score_answer(row, text)["correct"]


def _normalize_answer(value):
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(value.casefold().strip(" \t\r\n.,!;:*_`\"'").split())


def extract_answer(text):
    """Last explicit answer field; unmarked multiline text remains unparsed.

    A chain runner uses this value even when wrong; it must never replace it with
    row gold. Empty/multiline outputs without an answer field should fail the chain.
    """
    markers = list(re.finditer(r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*", text, re.I))
    if markers:
        answer = text[markers[-1].end() :].strip()
        return answer, "explicit_answer" if answer and "\n" not in answer else "unparsed"
    answer = text.strip()
    return answer, "single_line" if answer and "\n" not in answer else "unparsed"


def score_answer(row, text):
    """Score final aliases conservatively and expose conflicting answer candidates.

    No substring score: 'yes or no', 'Paris or Berlin', or an intermediate gold
    mention cannot pass. Exact aliases with punctuation/case/accent variation pass.
    Other natural prose remains a format failure for a separate semantic audit.
    """
    answer, status = extract_answer(text)
    normalized = _normalize_answer(answer)
    gold_aliases = {_normalize_answer(a) for a in row["aliases"]}
    gold = row["gold"]
    cities = city_table()
    if gold in {"yes", "no"}:
        candidates = {"yes": ["yes"], "no": ["no"]}
    elif gold in cities:
        candidates = {city: [city, *info["aliases"]] for city, info in cities.items()}
    elif gold.endswith(" Field Office") or gold.endswith(" Research Institute"):
        names = re.findall(r"\b[A-Za-z]+ (?:Field Office|Research Institute)\b", answer)
        candidates = {name: [name] for name in names}
        candidates[gold] = row["aliases"]
    elif gold.isdigit() or gold.lstrip("-").isdigit():
        names = re.findall(r"(?<!\w)-?\d+(?!\w)", answer)
        candidates = {name: [name] for name in names}
    else:
        candidates = {info["country"]: [info["country"]] for info in cities.values()}
    mentions = []
    for canonical, aliases in candidates.items():
        if any(
            re.search(r"(?<!\w)" + re.escape(_normalize_answer(a)) + r"(?!\w)", normalized)
            for a in aliases
        ):
            mentions.append(canonical)
    conflicts = sorted(name for name in mentions if _normalize_answer(name) not in gold_aliases)
    candidate_aliases = {
        _normalize_answer(alias) for aliases in candidates.values() for alias in aliases
    }
    correct = status != "unparsed" and normalized in gold_aliases and not conflicts
    return {
        "correct": correct,
        "strict_correct": correct,
        "parsed_answer": answer,
        "parse_status": status,
        "conflicting_candidates": conflicts,
        "format_valid": status != "unparsed" and normalized in candidate_aliases,
        "scoring": "exact_final_alias",
    }
