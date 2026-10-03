"""Role balance and disjoint target/replay/keep/unused partitions in the repair."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from llm_memory_editability.grokking_dynamics_mechanism import select_cases, taught_atoms
from llm_memory_editability.latent_scaling import build_world


def test_balanced_replay_covers_both_roles_without_changing_targets_or_d_truth():
    root = Path(__file__).parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "balanced_replay", root / "scripts/followup_grokking_edit_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    world = build_world(
        {
            "world": 103,
            "heads_n": 32,
            "bridges_n": 32,
            "tails_n": 16,
            "familiar_n": 8,
            "strict_n": 4,
            "holdout_fraction": 0.25,
            "low_extra": "anchors",
            "anchor_n": 4,
            "composition_count": "all",
        }
    )
    atoms = taught_atoms(world)
    for old in select_cases(world):
        new = module.balanced(old, world)
        assert old["new_fact"] == new["new_fact"]
        for key in ["D_familiar", "D_strict", "U_familiar", "U_strict"]:
            np.testing.assert_array_equal(old["tasks"][key], new["tasks"][key])
        for key in ["R_atomic", "Kdev_atomic"]:
            rows = new["tasks"][key]
            assert len(rows) == 32 and int(((rows[:, 1] >= 13) & (rows[:, 1] <= 16)).sum()) == 16
        r, k, u = map(set, [new["replay_indices"], new["keep_indices"], new["unused_indices"]])
        e = {new["atomic_index"]}
        assert not (e & r or e & k or e & u or r & k or r & u or k & u)
        assert e | r | k | u == set(range(len(atoms)))
