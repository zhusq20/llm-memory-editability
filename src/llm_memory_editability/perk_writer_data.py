"""Truth-backed episodic worlds for the controlled PERK writer extension.

This is a small adaptation of bAbI's movement/pickup semantics, not the published
QA2 pool or a reproduction. A world is identified by its complete factual state,
not sentence ordering. Questions and support are rendered from that state.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

PEOPLE = ("Mary", "John", "Daniel", "Sandra", "Bill", "Fred")
ROOMS = ("kitchen", "hallway", "bedroom", "bathroom", "garden", "office")
OBJECTS = ("apple", "football", "milk")


def world_id(locations: dict, owners: dict) -> str:
    state = json.dumps({"locations": locations, "owners": owners}, sort_keys=True)
    return hashlib.sha256(state.encode()).hexdigest()


def held_location(person: str, room: str) -> bool:
    return ROOMS.index(room) in {PEOPLE.index(person), (PEOPLE.index(person) + 1) % 6}


def held_owner(obj: str, person: str) -> bool:
    return PEOPLE.index(person) in {OBJECTS.index(obj), OBJECTS.index(obj) + 3}


def make_world(rng: random.Random) -> dict:
    locations = dict(zip(PEOPLE, rng.sample(ROOMS, len(ROOMS)), strict=True))
    owners = dict(zip(OBJECTS, rng.sample(PEOPLE, len(OBJECTS)), strict=True))
    people_order = rng.sample(list(PEOPLE), len(PEOPLE))
    object_order = rng.sample(list(OBJECTS), len(OBJECTS))
    # All movements precede pickups: "there" has one unambiguous location.
    events = [f"{person} went to the {locations[person]}." for person in people_order]
    events.extend(f"{owners[obj]} picked up the {obj} there." for obj in object_order)
    move = {person: events[people_order.index(person)] for person in PEOPLE}
    pickup = {obj: events[6 + object_order.index(obj)] for obj in OBJECTS}
    atomic, composed = [], []
    for person in PEOPLE:
        atomic.append(
            {
                "id": f"location:{person}",
                "kind": "atomic_location",
                "head": person,
                "question": f"Where is {person}?",
                "alternate": f"In which room is {person}?",
                "answer": locations[person],
                "support": [move[person]],
                "held_binding": held_location(person, locations[person]),
            }
        )
    for obj in OBJECTS:
        owner = owners[obj]
        atomic.append(
            {
                "id": f"owner:{obj}",
                "kind": "atomic_owner",
                "head": obj,
                "question": f"Who has the {obj}?",
                "alternate": f"Who is carrying the {obj}?",
                "answer": owner,
                "support": [pickup[obj]],
                "held_binding": held_owner(obj, owner),
            }
        )
        composed.append(
            {
                "id": f"composition:{obj}",
                "kind": "compositional",
                "head": obj,
                "question": f"Where is the {obj}?",
                "alternate": f"In which room is the {obj}?",
                "answer": locations[owner],
                "support": [move[owner], pickup[obj]],
                "prerequisites": [f"owner:{obj}", f"location:{owner}"],
                "held_binding": held_owner(obj, owner) or held_location(owner, locations[owner]),
            }
        )
    return {
        "world_id": world_id(locations, owners),
        "locations": locations,
        "owners": owners,
        "events": events,
        "context": [
            "\n".join(f"{i + 1}: {events[i]}" for i in range(start, start + 3))
            for start in (0, 3, 6)
        ],
        "atomic": atomic,
        "compositional": composed,
    }


def validate_world(row: dict) -> None:
    if set(row["locations"]) != set(PEOPLE) or set(row["owners"]) != set(OBJECTS):
        raise ValueError("Incomplete world state")
    if row["world_id"] != world_id(row["locations"], row["owners"]):
        raise ValueError("World identity differs from truth")
    if len(row["events"]) != 9 or len(set(row["events"])) != 9:
        raise ValueError("Every fact must appear exactly once")
    questions = row["atomic"] + row["compositional"]
    if len({q["id"] for q in questions}) != 12:
        raise ValueError("Question coverage changed")
    by_id = {q["id"]: q for q in questions}
    for q in questions:
        if not set(q["support"]) <= set(row["events"]):
            raise ValueError("Support is not in the material")
        if q["kind"] == "atomic_location":
            expected = row["locations"][q["head"]]
        elif q["kind"] == "atomic_owner":
            expected = row["owners"][q["head"]]
        else:
            owner = row["owners"][q["head"]]
            expected = row["locations"][owner]
            if [by_id[x]["answer"] for x in q["prerequisites"]] != [owner, expected]:
                raise ValueError("Composition prerequisites disagree with truth")
        if q["answer"] != expected:
            raise ValueError("Question answer disagrees with complete world state")
    material = "\n".join(row["context"])
    if any(event not in material for event in row["events"]) or "Answer:" in material:
        raise ValueError("Material lost a fact or contains question supervision")


def build_pools(counts: dict[str, int], seed: int = 110317) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    seen, pools = set(), {}
    for split, count in counts.items():
        if split not in {"development", "train", "validation", "test"} or count < 1:
            raise ValueError("Invalid split or empty requested pool")
        rows, attempts = [], 0
        while len(rows) < count:
            attempts += 1
            if attempts > count * 10000:
                raise ValueError("Requested pool cannot be generated under the binding split")
            row = make_world(rng)
            held = any(q["held_binding"] for q in row["atomic"])
            # Training and development never contain the reserved atomic pairs.
            # Test worlds require a composed chain with at least one such pair.
            if (split == "test") != held or row["world_id"] in seen:
                continue
            if split == "test" and not any(q["held_binding"] for q in row["compositional"]):
                continue
            validate_world(row)
            seen.add(row["world_id"])
            rows.append(row)
        pools[split] = rows
    return pools


def save_pools(destination: Path, counts: dict[str, int], seed: int = 110317) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    pools = build_pools(counts, seed)
    files = {}
    for split, rows in pools.items():
        path = destination / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        "kind": "controlled_babi_semantics_extension",
        "seed": seed,
        "files": files,
        "held_binding_scope": "current writer outer training; not original PERK pretraining",
        "split_unit": "complete factual world, independent of event ordering",
    }
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
