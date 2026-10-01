"""Pinned six-attribute bioS reconstruction for plan 14.27 (not the old graph)."""

from __future__ import annotations

import ast
import calendar
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

SOURCE = Path("data/bios-source")
MATERIAL = SOURCE / "data-synthetic-pretrain/Capo-bioS-bioR"
SOURCE_CODE = MATERIAL / "Capo-bioS-bioR.py"
MONTHS = list(calendar.month_name)[1:]
ATTRS = ("date", "birthcity", "university", "field", "workcity", "company")
TASKS = (*ATTRS, "year", "parity", "company_city", "city_company", "comparison")

# View 0 of the first six tasks is the question wording in Part 3.1, section 2.2.
# Additional tasks and views are an explicit measurement adaptation.
QUESTIONS = {
    "date": [
        "What is the birth date of {n}?",
        "On what date was {n} born?",
        "Give {n}'s full date of birth.",
        "When was {n} born?",
        "State the birthday, including the year, of {n}.",
        "What is {n}'s complete birth date?",
    ],
    "birthcity": [
        "What is the birth city of {n}?",
        "In which city was {n} born?",
        "Give {n}'s city of birth.",
        "Where was {n} born?",
        "Name the birthplace city of {n}.",
        "What city is {n}'s birthplace?",
    ],
    "university": [
        "Which university did {n} study?",
        "Where did {n} attend university?",
        "Give the university attended by {n}.",
        "At which university did {n} study?",
        "Name {n}'s university.",
        "What university educated {n}?",
    ],
    "field": [
        "What major did {n} study?",
        "What was {n}'s field of study?",
        "Give {n}'s university major.",
        "In what subject did {n} major?",
        "Name the discipline studied by {n}.",
        "Which subject did {n} study at university?",
    ],
    "company": [
        "Which company did {n} work for?",
        "Who was {n}'s employer?",
        "Give the company employing {n}.",
        "At what company did {n} work?",
        "Name {n}'s employer.",
        "What is the name of the company where {n} worked?",
    ],
    "workcity": [
        "Where did {n} work?",
        "In which city did {n} work?",
        "Give {n}'s work city.",
        "What city was {n}'s employer based in?",
        "Name the city of {n}'s workplace.",
        "What was {n}'s city of employment?",
    ],
    "year": [
        "What is the birth year of {n}?",
        "In which year was {n} born?",
        "Give only {n}'s birth year.",
        "Name the year in {n}'s date of birth.",
        "What year marks {n}'s birth?",
        "State the year of birth for {n}.",
    ],
    "parity": [
        "Was {n} born in an even-numbered month?",
        "Is {n}'s birth month number even?",
        "Does the month of {n}'s birth have an even number?",
        "Is the calendar month when {n} was born even-numbered?",
        "For {n}, is the birth month one of months 2, 4, 6, 8, 10, or 12?",
        "Is the number of {n}'s birth month divisible by two?",
    ],
    "company_city": [
        "Give {n}'s company, then work city.",
        "Name the employer of {n}, followed by its city.",
        "For {n}, list company before workplace city.",
        "What company employed {n}, and in what city?",
        "State {n}'s company and city, in that order.",
        "Report {n}'s employer first and work city second.",
    ],
    "city_company": [
        "Give {n}'s work city, then company.",
        "Name the work city of {n}, followed by the employer.",
        "For {n}, list workplace city before company.",
        "In what city did {n} work, and for what company?",
        "State {n}'s work city and company, in that order.",
        "Report {n}'s work city first and employer second.",
    ],
    "comparison": [
        "Who was born earlier, {a} or {b}?",
        "Which was born first: {a} or {b}?",
        "Between {a} and {b}, name the older person.",
        "Whose date of birth is earlier, {a}'s or {b}'s?",
        "Compare the birthdays of {a} and {b}. Who is older?",
        "Of {a} and {b}, who has the earlier birthday including the year?",
    ],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_source():
    manifest = load_json("configs/bios-source-manifest.json")
    for row in manifest["files"]:
        if digest(SOURCE / row["path"]) != row["sha256"]:
            raise ValueError(f"Official source changed: {row['path']}")
    return manifest


def source_templates():
    """Read literal templates without importing or changing the upstream source."""
    tree = ast.parse(SOURCE_CODE.read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_text_simple3"
    )
    values = {}
    for node in fn.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("sentence_structures"):
                values[int(name[-1]) - 1] = ast.literal_eval(node.value)
    return [values[i] for i in range(6)]


def date(p):
    return f"{MONTHS[p['month'] - 1]} {p['day']}, {p['year']}"


def date_key(p):
    return p["year"], p["month"], p["day"]


def render_sentences(p, templates, ids, order=range(6)):
    fields = dict(
        birthday=date(p),
        birthcity=p["birthcity"],
        university=p["university"],
        field=p["field"],
        company1city=p["workcity"],
        company1name=p["company"],
    )
    return [
        " " + templates[a][ids[a]].format(name=p["name"] if j == 0 else p["pronoun"], **fields)
        for j, a in enumerate(order)
    ]


def make_people(seed, count):
    if count % 10:
        raise ValueError("Population must permit exact 50/10/40 splits")
    rng = random.Random(seed)
    pools = {
        key: (MATERIAL / "fields" / f"{key}.txt").read_text().splitlines()
        for key in (
            "first_name",
            "middle_name",
            "last_name",
            "city",
            "university",
            "field",
            "company",
        )
    }
    names, people = set(), []
    for i in range(count):
        while True:
            name_ids = [
                rng.randrange(len(pools[k])) for k in ("first_name", "middle_name", "last_name")
            ]
            name = " ".join(
                pools[k][v]
                for k, v in zip(("first_name", "middle_name", "last_name"), name_ids, strict=True)
            )
            if name not in names:
                names.add(name)
                break
        indices = {
            k: rng.randrange(len(pools[k])) for k in ("city", "university", "field", "company")
        }
        company, city = [s.strip() for s in pools["company"][indices["company"]].split(";", 1)]
        people.append(
            dict(
                id=i,
                name=name,
                name_ids=name_ids,
                attribute_ids=indices,
                year=rng.randrange(1900, 2100),
                month=rng.randrange(1, 13),
                day=rng.randrange(1, 29),
                birthcity=pools["city"][indices["city"]],
                university=pools["university"][indices["university"]],
                field=pools["field"][indices["field"]],
                company=company,
                workcity=city,
                pronoun="He" if i % 2 == 0 else "She",
            )
        )
    ids = list(range(count))
    random.Random(seed + 1000000).shuffle(ids)
    for j, i in enumerate(ids):
        people[i]["split"] = (
            "train" if j < count // 2 else ("dev" if j < 6 * count // 10 else "test")
        )
    # One balanced comparison row per person. Both participants stay in the same split.
    for split in ("train", "dev", "test"):
        indices = [p["id"] for p in people if p["split"] == split]
        random.Random(seed + len(indices)).shuffle(indices)
        for j in range(0, len(indices), 2):
            a, b = indices[j : j + 2]
            if date_key(people[a]) == date_key(people[b]):
                for k in range(j + 2, len(indices)):
                    c = indices[k]
                    if date_key(people[a]) != date_key(people[c]):
                        indices[j + 1], indices[k] = indices[k], indices[j + 1]
                        b = c
                        break
                else:
                    raise ValueError("Equal-date terminal pair; choose a new prespecified seed")
            people[a]["partner"] = b
            people[b]["partner"] = a
    return people


def answer(p, task, other=None):
    if task == "date":
        return date(p)
    if task == "year":
        return str(p["year"])
    if task == "parity":
        return "yes" if p["month"] % 2 == 0 else "no"
    if task == "company_city":
        return f"{p['company']}; {p['workcity']}"
    if task == "city_company":
        return f"{p['workcity']}; {p['company']}"
    if task == "comparison":
        return min((p, other), key=date_key)["name"]
    return p[task]


def question(p, task, view=0, other=None, cot=False):
    q = QUESTIONS[task][view].format(n=p["name"], a=p["name"], b=other["name"] if other else "")
    return "Question: " + q + (" Explain briefly." if cot else "") + "\nAnswer:"


def reasoning(p, task, other=None):
    value = answer(p, task, other)
    if task == "parity":
        return (
            f"{date(p)}; {MONTHS[p['month'] - 1]} is month {p['month']}; "
            f"{p['month']} is {'even' if p['month'] % 2 == 0 else 'odd'}; Answer: {value}"
        )
    if task == "comparison":
        return f"{p['name']}: {date(p)}; {other['name']}: {date(other)}; Answer: {value}"
    if task == "year":
        return f"{date(p)}; the year is {p['year']}; Answer: {value}"
    if task in ("company_city", "city_company"):
        return f"company: {p['company']}; work city: {p['workcity']}; Answer: {value}"
    return f"{task}: {value}; Answer: {value}"


def edit_cases(people, seed):
    rng = random.Random(seed + 3000000)
    groups = [
        [p for p in people if p["split"] == "test" and p["month"] % 2 == parity]
        for parity in (0, 1)
    ]
    selected = rng.sample(groups[0], 8) + rng.sample(groups[1], 8)
    result = []
    for p in selected:
        month = rng.choice([m for m in range(1, 13) if m % 2 != p["month"] % 2])
        year = rng.choice([y for y in range(1900, 2100) if y != p["year"]])
        result.append(
            dict(
                person_id=p["id"],
                old_month=p["month"],
                old_year=p["year"],
                new_month=month,
                new_year=year,
                day=p["day"],
                cells=[
                    dict(month=m, year=y)
                    for m, y in (
                        (p["month"], p["year"]),
                        (month, p["year"]),
                        (p["month"], year),
                        (month, year),
                    )
                ],
            )
        )
    return result


def prepare_world(cfg, seed, tokenizer):
    root = Path(cfg["data_root"]) / f"world-{seed}"
    if (root / "audit.json").exists():
        audit = load_json(root / "audit.json")
        for f, h in audit["files"].items():
            if digest(root / f) != h:
                raise ValueError(f"Data changed: {root / f}")
        return audit
    root.mkdir(parents=True, exist_ok=True)
    manifest = verify_source()
    source_hash = digest(SOURCE_CODE)
    templates = source_templates()
    people = make_people(seed, cfg["people"])
    write_json(root / "people.json", people)
    write_json(root / "edit-cases.json", edit_cases(people, seed))
    write_json(root / "questions.json", QUESTIONS)
    rng = random.Random(seed + 2000000)
    template_ids = np.empty((len(people), 5, 6), dtype=np.int16)
    # Store separately tokenized full-name/pronoun sentences; no splitting at periods.
    # This preserves abbreviations in company names and city names.
    all_ids, lengths = [], []
    with (root / "biographies.jsonl").open("w") as out:
        for p in people:
            for variant in range(5):
                ids = [rng.randrange(len(t)) for t in templates]
                template_ids[p["id"], variant] = ids
                sentences = render_sentences(p, templates, ids)
                out.write(
                    json.dumps(
                        dict(
                            person_id=p["id"],
                            variant=variant,
                            template_ids=ids,
                            order=list(ATTRS),
                            source_sha256=source_hash,
                            text="".join(sentences),
                        )
                    )
                    + "\n"
                )
                for a in range(6):
                    # Force this attribute first, and then second, to obtain both name forms.
                    full = render_sentences(p, templates, ids, [a])[0]
                    pro = render_sentences(p, templates, ids, [(a + 1) % 6, a])[1]
                    for text in (full, pro):
                        enc = tokenizer.encode(text).ids
                        all_ids.append(enc)
                        lengths.append(len(enc))
    max_len = max(lengths)
    packed = np.zeros((len(all_ids), max_len), dtype=np.uint16)
    for i, tokens in enumerate(all_ids):
        packed[i, : len(tokens)] = tokens
    np.save(root / "sentence-tokens.npy", packed.reshape(len(people), 5, 6, 2, max_len))
    np.save(
        root / "sentence-lengths.npy",
        np.array(lengths, dtype=np.int16).reshape(len(people), 5, 6, 2),
    )
    np.save(root / "template-ids.npy", template_ids)
    assert len({p["name"] for p in people}) == len(people)
    for p in people:
        assert people[p["partner"]]["split"] == p["split"]
        assert date_key(p) != date_key(people[p["partner"]])
    counts = Counter(p["split"] for p in people)
    for task in TASKS:
        assert len(QUESTIONS[task]) == len(set(QUESTIONS[task])) == 6
    files = [
        "people.json",
        "edit-cases.json",
        "questions.json",
        "biographies.jsonl",
        "sentence-tokens.npy",
        "sentence-lengths.npy",
        "template-ids.npy",
    ]
    audit = dict(
        seed=seed,
        people=len(people),
        splits=dict(counts),
        source_revision=manifest["revision"],
        source_sha256=digest(SOURCE_CODE),
        template_counts=dict(zip(ATTRS, map(len, templates), strict=True)),
        variants=5,
        facts_per_person=6,
        passes=540,
        tasks_per_train_person=11,
        train_rows=counts["train"] * 11,
        qa_training_views=[0],
        globally_held_out_views=list(range(1, 6)),
        equal_date_comparisons=0,
        cross_split_comparisons=0,
        comparison_first_answer_fraction={
            s: np.mean(
                [date_key(p) < date_key(people[p["partner"]]) for p in people if p["split"] == s]
            ).item()
            for s in counts
        },
        city_collisions=sum(p["birthcity"] == p["workcity"] for p in people),
        max_sentence_tokens=max_len,
        files={f: digest(root / f) for f in files},
    )
    write_json(root / "audit.json", audit)
    return audit


class BioStream:
    """One deterministic pass presents every person exactly once, including the tail."""

    def __init__(self, root, seed, condition):
        self.tokens = np.load(Path(root) / "sentence-tokens.npy", mmap_mode="r")
        self.lengths = np.load(Path(root) / "sentence-lengths.npy", mmap_mode="r")
        self.seed, self.condition = seed, condition
        self.count = self.tokens.shape[0]

    def pass_tokens(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 1427]))
        people = rng.permutation(self.count)
        stream = [50256]
        order_hash = hashlib.sha256()
        for pid in people:
            variant = 0 if self.condition == "S" else (epoch + int(pid)) % 5
            order = rng.permutation(6) if self.condition == "MP" else np.arange(6)
            order_hash.update(np.array([pid, variant, *order], dtype=np.int64).tobytes())
            for j, attr in enumerate(order):
                form = int(j != 0)
                size = self.lengths[pid, variant, attr, form]
                stream.extend(self.tokens[pid, variant, attr, form, :size])
            stream.append(50256)
        return np.asarray(stream, dtype=np.int64), order_hash.hexdigest()

    def exact_token_budget(self, passes):
        # Permuting only changes which sentence uses the full name; sum its exact sizes.
        return sum(len(self.pass_tokens(e)[0]) - 1 for e in range(passes))
