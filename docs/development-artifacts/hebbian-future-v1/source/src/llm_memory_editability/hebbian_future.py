"""Exploratory kernel prediction and balanced multi-hop worlds, September 2026."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/hebbian-future-v1.json"
ART = ROOT / "docs/development-artifacts/hebbian-future-v1"
DATA = ROOT / "data/hebbian-future-v1"
RESULTS = ROOT / "results/hebbian-future-v1"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def chain_world(world, correlation, cfg):
    """Match marginals, subjects, memberships and query exposure across correlations.

    Eight people per organization: four combination-trained, four held out.
    Within EACH split and organization exactly rho*4 have home==HQ. Other home
    cities cycle identically across the balanced HQ city groups, keeping both
    training/held-out home-city histograms uniform. ID permutations are shared.
    """
    n, groups, cities = (cfg[x] for x in ("people", "organizations", "cities"))
    assert n == groups * 8 and groups % cities == 0
    rng = np.random.default_rng(world + 918273)
    people = rng.permutation(n).reshape(groups, 8)
    headquarters = rng.permutation(np.arange(groups) % cities)
    city_order = rng.permutation(cities)
    membership = np.empty(n, dtype=int)
    home = np.empty(n, dtype=int)
    train = np.zeros(n, dtype=bool)
    common_conflict = np.zeros(n, dtype=bool)
    matches = int(correlation * 4)
    for group, persons in enumerate(people):
        membership[persons] = group
        for split in range(2):
            subset = persons[split * 4 : split * 4 + 4]
            common_conflict[subset[-1]] = True
            offsets = np.r_[np.zeros(matches, dtype=int), 1 + np.arange(4 - matches) % 3]
            home[subset] = city_order[(headquarters[group] + offsets) % cities]
            if split == 0:
                train[subset] = True
    headquarters = city_order[headquarters]
    # Shared vocabulary and query positions; labels are next-token + EOS.
    person_start, group_start, city_start = 10, 10 + n, 10 + n + groups
    vocab = city_start + cities
    member = np.stack(
        [np.ones(n, int), person_start + np.arange(n), np.full(n, 3), np.full(n, 2)], axis=1
    )
    root = np.stack(
        [
            np.ones(groups, int),
            group_start + np.arange(groups),
            np.full(groups, 4),
            np.full(groups, 2),
        ],
        axis=1,
    )
    resident = member.copy()
    resident[:, 2] = 5
    composite = member.copy()
    composite[:, 2] = 6
    return {
        "world": world,
        "correlation": correlation,
        "vocab": vocab,
        "membership": membership,
        "home": home,
        "headquarters": headquarters,
        "train_people": train,
        "common_conflict": common_conflict,
        "member_x": member,
        "root_x": root,
        "home_x": resident,
        "composite_x": composite,
        "member_y": group_start + membership,
        "root_y": city_start + headquarters,
        "home_y": city_start + home,
        "composite_y": city_start + headquarters[membership],
        "group_start": group_start,
        "city_start": city_start,
    }


def training_arrays(world):
    mask = world["train_people"]
    x = np.concatenate(
        [world[k + "_x"] for k in ("member", "root", "home")] + [world["composite_x"][mask]]
    )
    y = np.concatenate(
        [world[k + "_y"] for k in ("member", "root", "home")] + [world["composite_y"][mask]]
    )
    return x, y


def ridge_predict(train_x, train_y, test_x, penalty=10.0):
    """Fixed ridge linear probability model; statistics fitted on dev only."""
    mean, scale = train_x.mean(0), np.maximum(train_x.std(0), 1e-8)
    a = np.c_[np.ones(len(train_x)), (train_x - mean) / scale]
    b = np.c_[np.ones(len(test_x)), (test_x - mean) / scale]
    reg = np.eye(a.shape[1]) * penalty
    reg[0, 0] = 0
    beta = np.linalg.solve(a.T @ a + reg, a.T @ train_y)
    return np.clip(b @ beta, 0, 1)


def auc(y, scores):
    positive, negative = np.asarray(scores)[y == 1], np.asarray(scores)[y == 0]
    if not len(positive) or not len(negative):
        return None
    return float(
        (positive[:, None] > negative).mean() + 0.5 * (positive[:, None] == negative).mean()
    )


def bootstrap_mean(values, seed, repeats=1000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]
