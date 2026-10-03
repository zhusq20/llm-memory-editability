"""Check conditional identities and training-support graphs; never load model weights."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, message, atol=1e-10):
    require(np.allclose(a, b, atol=atol, rtol=1e-10), message)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def incidence(n, edges):
    matrix = np.zeros((len(edges), n))
    for i, (left, right) in enumerate(edges):
        matrix[i, left], matrix[i, right] = 1.0, -1.0
    return matrix


def math_checks():
    rng = np.random.default_rng(752101)
    checks = {}
    features = rng.normal(size=(5, 12))
    readout = rng.normal(size=(3, 5))
    query = rng.normal(size=(5, 7))
    covariance = features @ features.T / features.shape[1]
    outputs = readout @ features
    hebbian = outputs @ features.T @ np.linalg.inv(covariance) @ query / features.shape[1]
    checks["full_rank_hebbian_identity"] = {
        "max_abs_error": close(hebbian, readout @ query, "Whitened identity failed")
    }

    features = np.diag([1.0, 2.0, 0.0, 0.0])[:, :2]
    covariance = features @ features.T / features.shape[1]
    projector = covariance @ np.linalg.pinv(covariance)
    readout = np.array([[1.0, -2.0, 3.0, 4.0]])
    query = np.array([0.0, 0.0, 1.0, 0.0])
    hebbian = readout @ covariance @ np.linalg.pinv(covariance) @ query
    remainder = readout @ (np.eye(4) - projector) @ query
    checks["singular_hebbian_decomposition"] = {
        "max_abs_error": close(hebbian + remainder, readout @ query, "Singular identity"),
        "omitted_output_without_remainder": float(remainder[0]),
    }
    require(abs(remainder[0]) > 1, "Example must expose a nonzero omitted direction")

    eigenvalues, vectors = np.linalg.eigh(covariance)
    positive = eigenvalues > 1e-12
    smallest = float(eigenvalues[positive].min())
    initial_error = rng.normal(size=(3, 4))

    def flow(time):
        return initial_error @ (vectors * np.exp(-time * eigenvalues)) @ vectors.T

    time, delta = 2.0, 1e-5
    derivative = (flow(time + delta) - flow(time - delta)) / (2 * delta)
    residual = close(derivative, -flow(time) @ covariance, "Flow does not solve ODE", 1e-8)
    null_residual = close(
        flow(time) @ (np.eye(4) - projector),
        initial_error @ (np.eye(4) - projector),
        "Nullspace changed during flow",
    )
    slacks = []
    for query in rng.normal(size=(100, 4)):
        bound = np.linalg.norm(initial_error, 2) * (
            np.linalg.norm((np.eye(4) - projector) @ query)
            + np.exp(-smallest * time) * np.linalg.norm(projector @ query)
        )
        slacks.append(float(bound - np.linalg.norm(flow(time) @ query)))
    require(min(slacks) >= -1e-10, "Spectral query bound violated")
    checks["spectral_flow_and_unidentifiable_directions"] = {
        "finite_difference_ode_error": residual,
        "nullspace_invariance_error": null_residual,
        "minimum_bound_slack_over_100_queries": min(slacks),
    }
    duplicated = np.tile(features, (1, 8))
    checks["repetition_preserves_support"] = {
        "normalized_gram_error": close(
            duplicated @ duplicated.T / duplicated.shape[1], covariance, "Repeated Gram"
        ),
        "rank_before": int(np.linalg.matrix_rank(features)),
        "rank_after": int(np.linalg.matrix_rank(duplicated)),
    }

    # Two bipartite supports have eight edges and identical degree two at every role.
    disconnected = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 6), (2, 7), (3, 6), (3, 7)]
    connected = [(0, 4), (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 4)]
    bd, bc = incidence(8, disconnected), incidence(8, connected)
    close(np.diag(bd.T @ bd), np.diag(bc.T @ bc), "Role degrees differ")
    gauge_shift = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    unseen_contrast = np.eye(8)[0] - np.eye(8)[6]
    close(bd @ gauge_shift, np.zeros(8), "Component shift changes observed edges")
    require(unseen_contrast @ gauge_shift == 1, "Unobserved contrast must change")
    require(np.linalg.norm(bc @ gauge_shift) > 0, "Connecting support misses shift")
    require(np.linalg.matrix_rank(bd) == 6, "Disconnected incidence rank")
    require(np.linalg.matrix_rank(bc) == 7, "Connected incidence rank")
    checks["coverage_and_degree_do_not_imply_identifiability"] = {
        "edges_each": 8,
        "degree_each_role": 2,
        "disconnected_rank": 6,
        "connected_rank": 7,
        "unseen_contrast_change_at_zero_training_residual": 1.0,
    }

    certified = []
    for lipschitz in [0.8, 1.0, 1.2]:
        actual, ideal, initial, epsilon = 0.2, 0.1, 0.1, 0.02
        for repeats in range(1, 9):
            actual, ideal = lipschitz * actual + epsilon, lipschitz * ideal
            total = sum(lipschitz**j for j in range(repeats))
            error_bound = lipschitz**repeats * initial + epsilon * total
            close(abs(actual - ideal), error_bound, "Finite-horizon equality failed")
            certified.append(error_bound)
    checks["finite_horizon_error_bound"] = {
        "cases": len(certified),
        "regimes": ["contractive", "nonexpansive", "expansive"],
        "scope": "Affine scalar examples attaining the conditional upper bound",
    }

    canonical, internal = np.array([1.0, 0.0]), np.array([1.0, 3.0])

    def atomic_gap(z):
        return 2 * z[0]

    def consumer_gap(z):
        return 1 - z[1]

    require(atomic_gap(canonical) > 0 and atomic_gap(internal) > 0, "Atomic recall")
    require(consumer_gap(canonical) > 0 > consumer_gap(internal), "Consumer failure")
    checks["atomic_correctness_does_not_certify_a_consumer"] = {
        "atomic_gaps": [float(atomic_gap(canonical)), float(atomic_gap(internal))],
        "consumer_gaps": [float(consumer_gap(canonical)), float(consumer_gap(internal))],
    }

    theta = 0.6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    gaps = {str(r): float(2 * (np.linalg.matrix_power(rotation, r) @ canonical)[0]) for r in [2, 3]}
    close(np.linalg.norm(rotation, 2), 1.0, "Rotation norm")
    require(gaps["2"] > 0 > gaps["3"], "Fixed weights should lose correct classification")
    checks["more_repeats_need_not_improve_a_final_readout"] = {
        "operator_norm": 1.0,
        "same_weight_gaps": gaps,
    }
    # Fixed decoding, inputs and execution schedules cannot create new weight states.
    facts, values, bits, parameters = 4, 4, 2, 3
    world_count = values**facts
    weight_states = 2 ** (bits * parameters)
    require(weight_states < world_count, "Counting example needs insufficient weight bits")
    checks["finite_precision_capacity_counting"] = {
        "independent_fact_worlds": world_count,
        "weight_states_at_any_fixed_repeat_count": weight_states,
        "required_bits": facts * np.log2(values),
        "available_bits": bits * parameters,
    }
    return {
        "scope": "Conditional mathematics and counterexamples; no LM mechanism claim",
        "checks": checks,
    }


def graph_partition(vertices, edges, indices):
    parent = list(range(len(vertices)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    forest = []
    adjacency = [set() for _ in vertices]
    for i in indices:
        left, right = edges[i]
        adjacency[left].add(right)
        adjacency[right].add(left)
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b
            forest.append(int(i))
    dsu = [find(i) for i in range(len(vertices))]
    # Independent breadth-first traversal audits the union-find partition.
    bfs, component = [-1] * len(vertices), 0
    for root in range(len(vertices)):
        if bfs[root] >= 0:
            continue
        queue = deque([root])
        bfs[root] = component
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if bfs[neighbour] < 0:
                    bfs[neighbour] = component
                    queue.append(neighbour)
        component += 1
    pairs = {(a, b) for a, b in zip(dsu, bfs, strict=True)}
    require(len(pairs) == len(set(dsu)) == component, "Graph partitions disagree")
    return bfs, forest, sum(not neighbours for neighbours in adjacency)


def disconnect_with_degree_preserving_switches(pool, selected_indices):
    """Alter only available training edges; preserve every left/right role degree."""
    lookup = {
        ((int(h), int(r1)), (int(b), int(r2))): i for i, (h, r1, b, r2, _t) in enumerate(pool)
    }
    groups = {}
    for i, row in enumerate(pool):
        groups.setdefault(int(row[2]), set()).add(i)
    selected = set(selected_indices)
    switches = []

    def components(indices):
        vertices = {("l", int(pool[i, 0]), int(pool[i, 1])) for i in indices} | {
            ("r", int(pool[i, 2]), int(pool[i, 3])) for i in indices
        }
        parent = {v: v for v in vertices}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for i in indices:
            h, r1, b, r2, _t = pool[i]
            left, right = find(("l", int(h), int(r1))), find(("r", int(b), int(r2)))
            parent[left] = right
        return len({find(v) for v in vertices})

    for _bridge, indices in sorted(groups.items()):
        local = selected & indices
        count = components(local)
        while True:
            found = False
            for i, j in itertools.combinations(sorted(local), 2):
                h1, r11, b1, r21, _t1 = pool[i]
                h2, r12, b2, r22, _t2 = pool[j]
                a = lookup.get(((int(h1), int(r11)), (int(b2), int(r22))))
                b = lookup.get(((int(h2), int(r12)), (int(b1), int(r21))))
                if a is None or b is None or a == b or a in selected or b in selected:
                    continue
                candidate = (local - {i, j}) | {a, b}
                new_count = components(candidate)
                if new_count > count:
                    selected = (selected - {i, j}) | {a, b}
                    local, count, found = candidate, new_count, True
                    switches.append({"removed": [int(i), int(j)], "added": [int(a), int(b)]})
                    break
            if not found:
                break
    return sorted(selected), switches


def training_graph_check(item, confirmation_root):
    world_id = item["world"]
    path = confirmation_root / f"w{world_id}-i741101-d128-l1-r1-nall-s128000/world.npz"
    with np.load(path) as world:
        pool, test = world["available_composite"].copy(), world["familiar_test"].copy()
    vertices = sorted(
        {("l", int(h), int(r)) for h, r, _b, _r2, _t in pool}
        | {("r", int(b), int(r)) for _h, _r1, b, r, _t in pool}
    )
    index = {v: i for i, v in enumerate(vertices)}
    edges = [
        (index["l", int(h), int(r1)], index["r", int(b), int(r2)]) for h, r1, b, r2, _t in pool
    ]
    full_partition, forest, _ = graph_partition(vertices, edges, range(len(edges)))
    full_components = len(set(full_partition))
    require(len(forest) == len(vertices) - full_components, "Spanning forest cardinality")
    require(len(forest) <= 512 <= len(pool), "512 forest selection must be feasible")
    forest_set = set(forest)
    selected = forest + [i for i in range(len(pool)) if i not in forest_set][: 512 - len(forest)]
    selected = sorted(selected)
    disconnected, switches = disconnect_with_degree_preserving_switches(pool, selected)
    require(bool(switches), "Degree-matched disconnected comparison must exist")
    require(len(disconnected) == len(set(disconnected)) == 512, "Unique chain count changed")

    def degrees(indices):
        return np.bincount([v for i in indices for v in edges[i]], minlength=len(vertices))

    close(degrees(selected), degrees(disconnected), "Degree-preserving selection failed")
    for column in range(pool.shape[1]):
        maximum = int(pool[:, column].max()) + 1
        close(
            np.bincount(pool[selected, column], minlength=maximum),
            np.bincount(pool[disconnected, column], minlength=maximum),
            "Entity/relation marginal count changed",
        )
    cases = {
        "prefix256": list(range(256)),
        "role_matched384": item["fixed_count_selections"]["384"][
            "selected_available_composite_indices"
        ],
        "role_matched512": item["fixed_count_selections"]["512"][
            "selected_available_composite_indices"
        ],
        "forest_matched512": selected,
        "degree_matched_disconnected512": disconnected,
        "all": list(range(len(pool))),
    }
    summaries = {}
    for name, indices in cases.items():
        partition, _forest, isolated = graph_partition(vertices, edges, indices)
        same_full_partition = (
            len({(a, b) for a, b in zip(partition, full_partition, strict=True)})
            == len(set(partition))
            == full_components
        )
        connected, covered = 0, 0
        used = {vertex for i in indices for vertex in edges[i]}
        for h, r1, b, r2, _t in test:
            left, right = index.get(("l", int(h), int(r1))), index.get(("r", int(b), int(r2)))
            if left is not None and right is not None:
                covered += int(left in used and right in used)
                connected += int(partition[left] == partition[right])
        summaries[name] = {
            "chain_count": len(indices),
            "components_including_uncovered_roles": len(set(partition)),
            "isolated_roles": isolated,
            "incidence_rank": len(vertices) - len(set(partition)),
            "matches_full_component_partition": same_full_partition,
            "test_pairs_with_both_roles_used": covered,
            "test_pairs_in_same_component": connected,
            "test_count": len(test),
        }
    require(summaries["forest_matched512"]["matches_full_component_partition"], "Forest matching")
    require(summaries["forest_matched512"]["isolated_roles"] == 0, "Forest role coverage")
    require(len(forest) > 384, "384 infeasibility statement no longer holds")
    return {
        "world": world_id,
        "data_path": str(path),
        "data_sha256": sha256(path),
        "role_count": len(vertices),
        "bridge_count": len(set(map(int, pool[:, 2]))),
        "full_components": full_components,
        "minimum_edges_preserving_full_components": len(forest),
        "cases": summaries,
        "forest_matched512_available_composite_indices": selected,
        "degree_matched_disconnected512_available_composite_indices": disconnected,
        "degree_preserving_switches": switches,
        "paired512_role_degrees_equal": True,
        "paired512_column_entity_relation_counts_equal": True,
        "exposure_scope": "Complete epochs only; actual fixed-step suffix is not yet trained",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confirmation-root", type=Path, default=Path("results/latent-confirmation-v1")
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=Path(
            "docs/development-artifacts/latent-confirmation-v1/role-coverage-followup.json"
        ),
    )
    parser.add_argument("--skip-training-graphs", action="store_true")
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Use a new output directory; do not overwrite reports")
    args.output_dir.mkdir(parents=True)
    math = math_checks()
    (args.output_dir / "math-checks.json").write_text(json.dumps(math, indent=2) + "\n")
    if not args.skip_training_graphs:
        coverage = json.loads(args.coverage_report.read_text())
        with ThreadPoolExecutor(max_workers=3) as executor:
            worlds = list(
                executor.map(
                    lambda x: training_graph_check(x, args.confirmation_root), coverage["worlds"]
                )
            )
        graph = {
            "scope": "Training support topology only; zero added LM training or weight evaluation",
            "selection_basis": "Only available training edges and their frozen order; "
            "no model scores or test pairs select edges",
            "caveat": "Incidence rank is exact for the additive calibration example, "
            "not a measured LM feature Gram rank",
            "source_sha256": sha256(__file__),
            "coverage_report_sha256": sha256(args.coverage_report),
            "worlds": worlds,
        }
        (args.output_dir / "support-graphs.json").write_text(json.dumps(graph, indent=2) + "\n")
    print(
        json.dumps(
            {"math_check_groups_passed": len(math["checks"]), "output_dir": str(args.output_dir)}
        )
    )


if __name__ == "__main__":
    main()
