"""Target-complete edits on the original two-chain, eight-layer task."""

import numpy as np
import torch

from .bios_cross import edit_pair, make_cross_world
from .bios_cross_train import tensor_queries
from .bios_direction import ROOT, Task, stable_order

CROSS_ARMS = ("func-soft-adam", "repr-hard-adam", "func-hard-gn", "func-soft-gn")


def cross_tasks(world_id, seed, device):
    world = make_cross_world(world_id, ROOT / "data/bios-organization-v1")
    tasks = []
    for chain in (0, 1):
        edits = edit_pair(world, chain)
        for group_index, group in enumerate(edits["groups"]):
            members = np.flatnonzero(world.memberships[chain] == group)
            eligible = [
                int(p)
                for p in members
                if not world.exceptions[chain, p]
                and world.derived_ids[chain, p] in world.heldout_ids[chain]
            ]
            person = stable_order(eligible, "focal", world_id, chain, int(group))[0]
            root, actual = int(world.root_ids[chain, group]), int(world.actual_ids[chain, person])
            c = int(world.answers[root])
            a = int(
                world.city_tokens[world.defaults[chain, edits["groups"][(group_index + 1) % 3]]]
            )
            b = int(
                world.city_tokens[world.defaults[chain, edits["groups"][(group_index + 2) % 3]]]
            )
            assert len({a, b, c}) == 3 and int(world.answers[actual]) == c
            e = [root, actual]
            d = world.derived_ids[chain, members].tolist()
            focal = int(world.derived_ids[chain, person])
            assert focal in world.heldout_ids[chain]
            membership = int(world.membership_ids[chain, person])
            local = np.flatnonzero(
                (world.person == person) & (np.arange(len(world.answers)) < world.n_base)
            ).tolist()
            local = [i for i in local if i not in e + [membership]]
            # Near facts exclude all direct edits and all affected compositions.
            near_pool = list(
                set(
                    world.actual_ids[chain, members].tolist()
                    + world.membership_ids[chain, members].tolist()
                )
                - set(e + [membership] + local)
            )
            near = stable_order(near_pool, "near", world_id, chain, int(group))[:48]
            reserved = set(e + d + local + near + [membership])
            base_pool = stable_order(
                [i for i in range(world.n_base) if i not in reserved],
                "base",
                world_id,
                chain,
                int(group),
            )
            r = [membership] + base_pool[:47]
            v = base_pool[47:95]
            far_base = base_pool[95:175]
            other_d = stable_order(
                [int(i) for i in world.heldout_ids.flatten() if i not in reserved],
                "far-d",
                world_id,
                chain,
                int(group),
            )[:16]
            far = far_base + other_d
            u = local + near + far
            included = e + r + v + u + d
            assert len(included) == len(set(included))
            mapping = {old: new for new, old in enumerate(included)}
            sets = {
                "E": e,
                "R": r,
                "V": v,
                "U": u,
                "local": local,
                "U_near": near,
                "U_far": far,
                "D": d,
                "D_heldout": [i for i in d if i in world.heldout_ids[chain]],
                "D_focal": [focal],
            }
            for kind in ("coherent", "independent"):
                truth = world.answers.copy()
                truth[root], truth[actual], truth[d] = a, a if kind == "coherent" else b, a
                assert set(np.flatnonzero(truth != world.answers)) == set(e + d)
                data = tensor_queries(world, device, truth)
                old = torch.stack(
                    [
                        torch.as_tensor(world.answers[included], device=device),
                        torch.full((len(included),), 3, device=device),
                    ],
                    -1,
                )
                tasks.append(
                    Task(
                        f"world-{world_id}-seed-{seed}-chain-{chain}-group-{int(group)}-{kind}",
                        data["tokens"][included],
                        data["positions"][included],
                        data["labels"][included],
                        old,
                        {k: [mapping[i] for i in ids] for k, ids in sets.items()},
                        dict(
                            world=world_id,
                            seed=seed,
                            chain=chain,
                            group=int(group),
                            group_index=group_index,
                            person=person,
                            kind=kind,
                            a=a,
                            b=b,
                            c=c,
                            pair=[mapping[actual], mapping[root]],
                            original_ids=included,
                        ),
                    )
                )
    return tasks
