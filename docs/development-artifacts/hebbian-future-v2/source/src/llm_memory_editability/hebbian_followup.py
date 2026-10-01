"""Fixed-content worlds and conservative full-entity answer parsing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

import numpy as np

from .hebbian_future import ROOT, chain_world

CONFIG = ROOT / "configs/hebbian-future-v2.json"
ART = ROOT / "docs/development-artifacts/hebbian-future-v2"
DATA = ROOT / "data/hebbian-future-v2"
RESULTS = ROOT / "results/hebbian-future-v2"


def stable(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixed_world(world, rho, cfg):
    """Change only nonfocal home values as rho changes; focal wrong values stay fixed."""
    w = chain_world(world, rho, cfg)
    rng = np.random.default_rng(world + 918273)
    people = rng.permutation(cfg["people"]).reshape(cfg["organizations"], 8)
    hq = rng.permutation(np.arange(cfg["organizations"]) % cfg["cities"])
    city_order = rng.permutation(cfg["cities"])
    matches = int(rho * 4)
    for group, persons in enumerate(people):
        for split in range(2):
            selected = persons[split * 4 : split * 4 + 4]
            # Offset of slot j is j at EVERY rho until made coherent. Slot 3
            # is always the same competing city, in both train and held-out.
            offsets = np.arange(4)
            offsets[:matches] = 0
            w["home"][selected] = city_order[(hq[group] + offsets) % 4]
    w["home_y"] = w["city_start"] + w["home"]
    return w


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def inventory_from_facts(facts):
    inventory = defaultdict(dict)
    for fact in facts:
        values = inventory[fact["relation_id"]]
        item = values.setdefault(fact["target_id"], {"label": fact["answer"], "aliases": []})
        item["aliases"] = sorted(set(item["aliases"] + fact.get("aliases", []) + [fact["answer"]]))
    return dict(inventory)


def parse_entity(text, inventory):
    """Read a leading complete alias, abstaining on collisions/negation/lists.

    Plain completion makes a leading answer meaningful. The parser does not
    convert every nonmatch into factual error. No model-assisted grading.
    """
    text = normalize(text).lstrip(" \"'`“”‘’([{:")
    if not text:
        return {"status": "unrecognized", "entity": None}
    if re.match(r"(?:not|no|neither|unknown|i don't|i do not)\b", text):
        return {"status": "unrecognized", "entity": None}
    matches = []
    for entity, item in inventory.items():
        for alias in item["aliases"]:
            a = normalize(alias)
            if text.startswith(a) and (len(text) == len(a) or not text[len(a)].isalnum()):
                matches.append((len(a), entity))
    if not matches:
        return {"status": "unrecognized", "entity": None}
    length = max(n for n, _ in matches)
    entities = {entity for n, entity in matches if n == length}
    if len(entities) != 1:
        return {"status": "ambiguous", "entity": None}
    suffix = text[length:].lstrip()
    if re.match(r"(?:and|or|/|,?\s*not)\b", suffix):
        return {"status": "ambiguous", "entity": None}
    return {"status": "recognized", "entity": next(iter(entities))}


def candidate_set(fact, inventory, seed):
    items = inventory[fact["relation_id"]]
    own_aliases = {normalize(a) for a in items[fact["target_id"]]["aliases"]}
    ordered = sorted(items, key=lambda x: stable(f"{seed}:{fact['case_id']}:{x}"))
    selected = [fact["target_id"]]
    aliases = set(own_aliases)
    for entity in ordered:
        other = {normalize(a) for a in items[entity]["aliases"]}
        if entity in selected or aliases.intersection(other):
            continue
        selected.append(entity)
        aliases.update(other)
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("Need four distinct, unambiguous candidates")
    # Correct position varies independently of model outputs.
    return sorted(selected, key=lambda x: stable(f"position:{seed}:{fact['case_id']}:{x}"))
