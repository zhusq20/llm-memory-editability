"""bioS-Work-v1 truth, equal-exposure curricula and paired counterfactual worlds.

Official files remain unmodified in data/bios-source; this is a new adapter.
The symbolic experiment does not imply that the natural-language audit is complete.
"""

import calendar
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REVISION = "211b28f2e9d453114ca5a1cbbb2d5a632ac60fcf"
SOURCE_FOLDER = "data-synthetic-pretrain/Capo-bioS-bioR"
RELATIONS = (
    "employer",
    "default_city",
    "actual_city",
    "birth_date",
    "birth_city",
    "university",
    "major",
)
N_PEOPLE, N_COMPANIES, N_BASE, N_QUERIES = 2048, 64, 12352, 14400
PAD, BOS, ANS, EOS = range(4)


def rng_for(seed, purpose, index=0):
    return np.random.default_rng(np.random.SeedSequence([seed, purpose, index]))


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass
class World:
    seed: int
    prompts: np.ndarray
    lengths: np.ndarray
    answers: np.ndarray
    relation: np.ndarray
    person: np.ndarray
    company: np.ndarray
    employers: np.ndarray
    defaults: np.ndarray
    actual: np.ndarray
    exceptions: np.ndarray
    city_tokens: np.ndarray
    token_labels: list
    names: dict

    @property
    def vocab_size(self):
        return len(self.token_labels)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        np.savez_compressed(directory / "world.npz", **arrays)
        write_json(
            directory / "metadata.json",
            {"seed": self.seed, "token_labels": self.token_labels, "names": self.names},
        )
        write_json(directory / "audit.json", audit_world(self))


def load_world(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    with np.load(directory / "world.npz", allow_pickle=False) as arrays:
        return World(**metadata, **dict(arrays))


def make_world(seed, source_root="data/bios-source"):
    source_root = Path(source_root)
    manifest = json.loads((source_root / "manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("Unexpected official source revision")
    for entry in manifest["files"]:
        if (
            hashlib.sha256((source_root / entry["path"]).read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError("Source hash mismatch: " + entry["path"])

    def field(name):
        return (source_root / SOURCE_FOLDER / "fields" / (name + ".txt")).read_text().splitlines()

    cities, universities, majors = field("city"), field("university"), field("field")
    first, middle, last = field("first_name"), field("middle_name"), field("last_name")
    names_rng, group_rng, value_rng = (rng_for(seed, i) for i in (11, 12, 13))

    def unique_names(count):
        result, used = [], set()
        while len(result) < count:
            value = f"{names_rng.choice(first)} {names_rng.choice(middle)} {names_rng.choice(last)}"
            if value.casefold() not in used:
                result.append(value)
                used.add(value.casefold())
        return result

    person_names = unique_names(2 * N_PEOPLE)
    syllables = ("va", "ne", "ri", "lo", "sa", "tu", "mi", "ze", "ka", "po", "da", "fi")
    company_names, used = [], set()
    while len(company_names) < 2 * N_COMPANIES:
        value = "".join(names_rng.choice(syllables, 4)).capitalize() + " Labs"
        if value not in used:
            used.add(value)
            company_names.append(value)
    employers = group_rng.permutation(np.repeat(np.arange(N_COMPANIES), 32))
    work_cities = value_rng.choice(len(cities), 32, replace=False)
    defaults = value_rng.permutation(np.repeat(work_cities, 2))
    exceptions = np.zeros(N_PEOPLE, dtype=bool)
    actual = defaults[employers].copy()
    for c in range(N_COMPANIES):
        selected = group_rng.choice(np.flatnonzero(employers == c), 2, replace=False)
        exceptions[selected] = True
        actual[selected] = value_rng.choice(work_cities[work_cities != defaults[c]], 2)
    attributes_rng = rng_for(seed, 14)
    years = attributes_rng.integers(1900, 2100, N_PEOPLE)
    months = attributes_rng.integers(1, 13, N_PEOPLE)
    days = attributes_rng.integers(1, 29, N_PEOPLE)
    dates = [
        f"{calendar.month_name[m]} {d}, {y}" for y, m, d in zip(years, months, days, strict=True)
    ]
    birth_cities = attributes_rng.integers(0, len(cities), N_PEOPLE)
    university = attributes_rng.integers(0, len(universities), N_PEOPLE)
    major = attributes_rng.integers(0, len(majors), N_PEOPLE)
    labels = (
        [f"person:{i}" for i in range(N_PEOPLE)]
        + [f"company:{i}" for i in range(N_COMPANIES)]
        + [f"city:{i}" for i in range(len(cities))]
        + [f"date:{d}" for d in sorted(set(dates))]
        + [f"university:{i}" for i in range(len(universities))]
        + [f"major:{i}" for i in range(len(majors))]
        + [f"relation:{r}" for r in RELATIONS]
    )
    labels = list(rng_for(seed, 15).permutation(labels))
    token_labels = ["<PAD>", "<BOS>", "<ANS>", "<EOS>"] + labels
    token = {label: i for i, label in enumerate(token_labels)}
    city_tokens = np.array([token[f"city:{i}"] for i in range(len(cities))])
    prompts = np.full((N_QUERIES, 5), PAD, dtype=np.int64)
    lengths = np.full(N_QUERIES, 4, dtype=np.int64)
    answers = np.empty(N_QUERIES, dtype=np.int64)
    relation = np.empty(N_QUERIES, dtype=np.int64)
    person = np.full(N_QUERIES, -1, dtype=np.int64)
    company = np.empty(N_QUERIES, dtype=np.int64)
    values = [
        [token[f"company:{c}"] for c in employers],
        city_tokens[defaults],
        city_tokens[actual],
        [token[f"date:{d}"] for d in dates],
        city_tokens[birth_cities],
        [token[f"university:{u}"] for u in university],
        [token[f"major:{m}"] for m in major],
    ]
    offset = 0
    for rel, rel_values in enumerate(values):
        for i, value in enumerate(rel_values):
            q = offset + i
            is_company = rel == 1
            subject = token[f"company:{i}" if is_company else f"person:{i}"]
            prompts[q, :4] = [BOS, subject, token[f"relation:{RELATIONS[rel]}"], ANS]
            answers[q], relation[q] = value, rel
            company[q] = i if is_company else employers[i]
            if not is_company:
                person[q] = i
        offset += len(rel_values)
    for i in range(N_PEOPLE):
        q = N_BASE + i
        prompts[q] = [
            BOS,
            token[f"person:{i}"],
            token["relation:employer"],
            token["relation:default_city"],
            ANS,
        ]
        answers[q] = city_tokens[defaults[employers[i]]]
        lengths[q], relation[q], person[q], company[q] = 5, 7, i, employers[i]
    names = {
        "people": person_names[:N_PEOPLE],
        "people_aliases": person_names[N_PEOPLE:],
        "companies": company_names[:N_COMPANIES],
        "company_aliases": company_names[N_COMPANIES:],
        "company_source_ids": names_rng.choice(
            len(field("company")), N_COMPANIES, replace=False
        ).tolist(),
        "cities": cities,
        "universities": universities,
        "majors": majors,
        "work_city_ids": work_cities.tolist(),
    }
    world = World(
        seed,
        prompts,
        lengths,
        answers,
        relation,
        person,
        company,
        employers,
        defaults,
        actual,
        exceptions,
        city_tokens,
        token_labels,
        names,
    )
    audit_world(world)
    return world


def curriculum(world, order, total_steps, batch_size=128):
    if order not in {"SA", "AS"} or total_steps < 5280 or batch_size != 128:
        raise ValueError("Protocol curriculum requires SA/AS, >=5280 steps and batch 128")
    s = np.flatnonzero(np.isin(world.relation[:N_BASE], [0, 1]))
    a = np.flatnonzero(~np.isin(world.relation[:N_BASE], [0, 1]))
    parts = {}
    for name, ids, repeats, purpose in (("S", s, 160, 21), ("A", a, 33, 22)):
        rng = rng_for(world.seed, purpose)
        parts[name] = np.concatenate([rng.permutation(ids) for _ in range(repeats)])
        assert len(parts[name]) == 337920
    rng = rng_for(world.seed, 23)
    n_mix = (total_steps - 5280) * batch_size
    # Exact 80:20 across each five-step block; final partial block is recorded.
    n_derived = n_mix // 5
    n_atomic = n_mix - n_derived
    atomic = np.concatenate([rng.permutation(N_BASE) for _ in range(n_atomic // N_BASE + 1)])[
        :n_atomic
    ]
    derived = N_BASE + np.tile(rng.permutation(N_PEOPLE), n_derived // N_PEOPLE + 1)[:n_derived]
    mixed = np.empty(n_mix, dtype=np.int64)
    derived_positions = np.arange(4, n_mix, 5)
    mixed[derived_positions] = derived
    is_atomic = np.ones(n_mix, dtype=bool)
    is_atomic[derived_positions] = False
    mixed[is_atomic] = atomic
    return np.concatenate([parts[order[0]], parts[order[1]], mixed]).reshape(
        total_steps, batch_size
    )


def paired_edit(world, support=0, k=1):
    if k not in (1, 4, 16):
        raise ValueError("Protocol scales are 1, 4 and 16")
    rng = rng_for(world.seed, 31, support + 100 * k)
    # Draw whole disjoint supports until every triple has three different defaults.
    for _ in range(100000):
        companies = rng.choice(N_COMPANIES, 3 * k, replace=False).reshape(k, 3)
        if all(len(set(world.defaults[t])) == 3 for t in companies):
            break
    else:
        raise RuntimeError("No valid support")
    coherent, exception = world.answers.copy(), world.answers.copy()
    for triple in companies:
        old = world.defaults[triple]
        for j, c in enumerate(triple):
            next_city, alternate = old[(j + 1) % 3], old[(j + 2) % 3]
            root = 2048 + c
            coherent[root] = exception[root] = world.city_tokens[next_city]
            members = rng.permutation(np.flatnonzero((world.employers == c) & ~world.exceptions))
            rows = 2112 + members
            coherent[rows] = world.city_tokens[next_city]
            exception[rows[:15]], exception[rows[15:]] = (
                world.city_tokens[next_city],
                world.city_tokens[alternate],
            )
            derived = N_BASE + np.flatnonzero(world.employers == c)
            coherent[derived] = exception[derived] = world.city_tokens[next_city]
    edit_mask = coherent != world.answers
    assert np.array_equal(edit_mask, exception != world.answers)
    e = np.flatnonzero(edit_mask & (np.arange(N_QUERIES) < N_BASE))
    d = np.flatnonzero(edit_mask & (np.arange(N_QUERIES) >= N_BASE))
    u = ~edit_mask
    selected = np.isin(world.company, companies.ravel())
    old_exception = (world.relation == 2) & world.exceptions[np.maximum(world.person, 0)]
    strata = np.full(N_QUERIES, -1, dtype=np.int64)
    strata[u & selected & old_exception] = 0
    strata[u & selected & ~old_exception] = 1
    strata[u & ~selected & np.isin(world.relation, [1, 2, 7])] = 2
    strata[u & ~selected & np.isin(world.relation, [0, 3, 4, 5, 6])] = 3
    assert np.array_equal(strata >= 0, u)
    assert len(e) == 93 * k and len(d) == 96 * k
    assert np.array_equal(
        np.bincount(coherent[e], minlength=world.vocab_size),
        np.bincount(exception[e], minlength=world.vocab_size),
    )
    split_rng = rng_for(world.seed, 32, support + 100 * k)
    replay_pool, heldout = [], []
    for group in range(4):
        ids = split_rng.permutation(
            np.flatnonzero((strata == group) & (np.arange(N_QUERIES) < N_BASE))
        )
        n_test = max(1, int(np.ceil(0.2 * len(ids))))
        heldout.extend(ids[:n_test])
        replay_pool.extend(ids[n_test:])
    heldout.extend(np.flatnonzero(u & (np.arange(N_QUERIES) >= N_BASE)))
    replay = split_rng.choice(replay_pool, min(4096, len(replay_pool)), replace=False)
    return {
        "companies": companies,
        "coherent": coherent,
        "exception": exception,
        "E": e,
        "D": d,
        "strata": strata,
        "replay": replay,
        "heldout": np.array(sorted(heldout)),
        "support": support,
        "k": k,
        "selection": "prespecified_unmatched",
    }


def audit_world(world):
    assert world.prompts.shape == (N_QUERIES, 5)
    assert len(set(map(tuple, world.prompts))) == N_QUERIES
    assert np.array_equal(np.bincount(world.employers), np.full(N_COMPANIES, 32))
    assert np.array_equal(np.unique(world.defaults, return_counts=True)[1], np.full(32, 2))
    assert world.exceptions.sum() == 128
    assert np.array_equal(
        np.bincount(world.employers[world.exceptions], minlength=64), np.full(64, 2)
    )
    assert np.array_equal(world.actual != world.defaults[world.employers], world.exceptions)
    assert np.array_equal(world.answers[2112:4160], world.city_tokens[world.actual])
    assert np.array_equal(
        world.answers[N_BASE:], world.city_tokens[world.defaults[world.employers]]
    )
    for key, alias in (("people", "people_aliases"), ("companies", "company_aliases")):
        names = world.names[key] + world.names[alias]
        assert len(set(n.casefold().strip() for n in names)) == len(names)
    baseline = {}
    for relation in range(7):
        values = world.answers[world.relation == relation]
        answer = int(np.bincount(values, minlength=world.vocab_size).argmax())
        baseline[RELATIONS[relation]] = {
            "answer_token": answer,
            "accuracy": float(np.mean(values == answer)),
        }
    return {
        "version": "bioS-Work-v1",
        "world_seed": world.seed,
        "source_revision": REVISION,
        "base_facts": N_BASE,
        "derived_queries": N_PEOPLE,
        "people": N_PEOPLE,
        "companies": N_COMPANIES,
        "old_exceptions": 128,
        "vocab_size": world.vocab_size,
        "truth_sha256": array_hash(world.answers),
        "queries_sha256": array_hash(world.prompts),
        "majority_baseline": baseline,
        "oracle_default_baseline": {"overall": 0.9375, "old_exception": 0.0, "nonexception": 1.0},
        "symbolic_truth_audit": "passed",
        "natural_language_audit": "not_complete; not used in symbolic run",
    }
