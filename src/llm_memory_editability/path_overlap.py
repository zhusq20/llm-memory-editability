"""Zero-training path-overlap (Z1) interventions.

The contract is configs/zero-training-path-overlap-v1.json, frozen before any Z1
outcome existed. The bridge-format check (Z2) is withdrawn there and has no code
here. Every forward runs the existing grokking-reproduction-v1 weights in
float32 with explicit causal softmax attention (no autocast, no SDPA, no TF32).
Interventions are forward-time edits: MLP activation gates, attention edge masks
and residual-state replacements. No parameter is ever written.

Executions are counted e = 0..7. A Loop 4x2 runs block e % 4, round e // 4; the
standard 8-layer model runs block e. Unit u = (unique block j, neuron n); its
zero ablation multiplies GELU activation n of block j by zero, which equals
zeroing row n of that block's mlp.c_proj weight (GPT-2 Conv1D stores [m, d]).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from .grokking_reproduction import ReproductionGPT

BATCH = "zero-training-path-overlap-v1"
PANELS_ATOMIC = ("atomic_id", "atomic_ood")
PANELS_TWO_HOP = ("train_ii", "test_ii", "test_io", "test_oi", "test_oo")
CODE_FILES = (
    "src/llm_memory_editability/path_overlap.py",
    "scripts/run_zero_training_path_overlap.py",
    "scripts/read_path_hop_secondary.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Contract ---------------------------------------------------------------------


def _find(text, pattern, cast=float, count=1):
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Contract text no longer matches {pattern!r}: {text!r}")
    values = [cast(v) for v in match.groups()]
    assert len(values) == count
    return values[0] if count == 1 else values


def contract_parameters(contract):
    """Parse every numeric setting from the frozen contract text.

    The contract states most settings in prose. Each is read by an anchored
    pattern so that any later edit of the contract fails loudly here.
    """
    if contract["batch"] != BATCH:
        raise ValueError(f"Expected the {BATCH} contract, got {contract['batch']!r}")
    if "z2" in contract or "z2" not in contract.get("withdrawn", {}):
        raise ValueError("Z2 must be recorded as withdrawn and absent from the contract")
    z1 = contract["z1"]
    sel = z1["selection_on_single_hop_only"]
    two = z1["two_hop_measure"]
    nodes, interp = z1["primary_nodes"], z1["interpretation"]
    ported = z1["ported_diagnostics_descriptive"]
    same_bridge = next(d for d in two["descriptive"] if d.startswith("same-bridge"))
    rep = contract["execution"]["replication_check"]
    per_node = re.fullmatch(r"(.+?) in (\S+)", contract["outputs"]["per_node"])
    if per_node is None:
        raise ValueError(f"Contract per-node outputs no longer parse: {contract['outputs']}")
    p = {
        "runs": list(contract["source"]["runs"]),
        "nodes": list(contract["source"]["nodes"]),
        "code_files": tuple(c.strip() for c in contract["execution"]["code"].split(";")),
        "outputs": tuple(name.strip() for name in per_node.group(1).split(",")),
        "output_dir": contract["execution"]["output"],
        "replication": {
            "min_agreement": _find(rep, r">= (0\.\d+) on every panel"),
            "from_step": _find(rep, r"every node with step >= (\d+)", int),
            "final_test_ii": _find(rep, r"and >= (0\.\d+) on test_ii at 1\.5M"),
            "early_stop": _find(rep, r"the node stops only below (0\.\d+)"),
        },
        "z1": {
            "n_id": _find(z1["fact_sample"], r"(\d+) ID facts", int),
            "n_ood": _find(z1["fact_sample"], r"(\d+) OOD facts", int),
            "fact_seed": _find(z1["fact_sample"], r"drawn with seed (\d+)", int),
            "min_chains": _find(z1["fact_sample"], r">= (\d+) eligible-kind chains", int),
            "chains_per_fact": _find(z1["fact_sample"], r"Up to (\d+) such chains", int),
            "k_grid": [int(k) for k in sel["k_grid"]],
            "same_answer": _find(sel["selectivity_controls"]["same_answer"], r"up to (\d+)", int),
            "random_atoms": _find(sel["selectivity_controls"]["random"], r"^(\d+) random", int),
            "random_seed": _find(sel["selectivity_controls"]["random"], r"seed (\d+)", int),
            "retention": _find(sel["selective"], r"retention >= (0\.\d+)"),
            "empty_control_passes": "empty after the baseline filter counts as passed"
            in sel["selective"],
            "control_seed": _find(two["R_scope"], r"seed from \((\d+), fact index\)", int),
            "bootstrap": _find(
                two["delta"],
                r"bootstrap (\d+) draws, seed (\d+), percentile \[([\d.]+), ([\d.]+)\]",
                float,
                4,
            ),
            "same_bridge": _find(same_bridge, r"up to (\d+) chains", int),
            "scopes": z1["scopes"],
            "loop_nodes": _find(
                nodes["loop"], r"test_io >= (0\.\d+) and atomic_ood >= (0\.\d+)", float, 2
            ),
            "loop_node_count": _find(nodes["loop"], r"\((\d+) nodes\)", int),
            "standard_nodes": _find(nodes["standard"], r"test_ii >= (0\.\d+)"),
            "standard_node_count": _find(nodes["standard"], r"\((\d+) nodes\)", int),
            "power": _find(nodes["power"], r">= (\d+) selective facts", int),
            "loop_supports": _find(
                interp["loop_supports"],
                r"Delta_round2 >= (0\.\d+) and Delta_round1 <= (0\.\d+)",
                float,
                2,
            ),
            "loop_against": _find(interp["loop_against"], r"Delta_round2 <= (0\.\d+)"),
            "standard_separate": _find(interp["standard_separate_copy"], r"Delta_all <= (0\.\d+)"),
            "standard_shared": _find(interp["standard_shared"], r"Delta_all >= (0\.\d+)"),
            "windows": _find(
                ported["attention_window_masking"],
                r"window (\d+): \[c-1, c\+1\]; also window (\d+)",
                int,
                2,
            ),
            "patch": _find(
                ported["bridge_patching"], r"(\d+) ID and (\d+) OOD tuples \(seed (\d+)\)", int, 3
            ),
        },
    }
    draws, seed, low, high = p["z1"]["bootstrap"]
    p["z1"]["bootstrap"] = {"iterations": int(draws), "seed": int(seed), "interval": [low, high]}
    n_id, n_ood, seed = p["z1"]["patch"]
    p["z1"]["patch"] = {"id": n_id, "ood": n_ood, "seed": seed}
    return p


def load_contract(path):
    contract = json.loads(Path(path).read_text())
    return contract, contract_parameters(contract)


# Data -------------------------------------------------------------------------


class World:
    """Graph arrays plus token offsets of grokking-reproduction-v1."""

    def __init__(self, data_dir):
        data_dir = Path(data_dir)
        self.metadata = json.loads((data_dir / "complete.json").read_text())
        world = dict(np.load(data_dir / "world.npz"))
        self.panels = dict(np.load(data_dir / "panels.npz"))
        self.atoms = world["atoms"].astype(np.int64)
        self.ood = world["ood"].astype(bool)
        self.chains = world["chains"].astype(np.int64)
        self.first = world["first"].astype(np.int64)
        self.second = world["second"].astype(np.int64)
        self.kind = world["kind"].astype(np.int64)
        self.train_order = world["train_order"].astype(np.int64)
        self.eo = int(self.metadata["entity_offset"])
        self.ro = int(self.metadata["relation_offset"])
        self.entities = int(self.metadata["entities"])
        self.lookup = np.full((self.entities, int(self.metadata["relations"])), -1, np.int64)
        self.lookup[self.atoms[:, 0], self.atoms[:, 1]] = np.arange(len(self.atoms))

    def support(self, phi):
        mask = np.zeros(len(self.chains), dtype=bool)
        mask[self.train_order[: round(phi * self.metadata["atomic_id"])]] = True
        return mask

    def atomic_tokens(self, atom_index):
        a = self.atoms[np.asarray(atom_index, dtype=np.int64)]
        return np.column_stack((a[:, 0] + self.eo, a[:, 1] + self.ro))

    def chain_tokens(self, chain_index):
        c = self.chains[np.asarray(chain_index, dtype=np.int64)]
        return np.column_stack((c[:, 0] + self.eo, c[:, 1] + self.ro, c[:, 2] + self.ro))

    def row_tokens(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.shape[1] == 3:
            return np.column_stack((rows[:, 0] + self.eo, rows[:, 1] + self.ro))
        return np.column_stack((rows[:, 0] + self.eo, rows[:, 1] + self.ro, rows[:, 2] + self.ro))

    def row_facts(self, rows):
        """(first atom, second atom, bridge) of panel chain rows (h, r1, r2, t)."""
        rows = np.asarray(rows, dtype=np.int64)
        first = self.lookup[rows[:, 0], rows[:, 1]]
        bridge = self.atoms[first, 2]
        second = self.lookup[bridge, rows[:, 2]]
        assert np.all(first >= 0) and np.all(second >= 0)
        assert np.array_equal(self.atoms[second, 2], rows[:, 3])
        return first, second, bridge


def load_model(root, run, step, metadata, device):
    path = Path(root) / "development" / run / f"checkpoint-{step:07d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == step, "Wrong checkpoint training step"
    assert payload["spec"]["name"] == run, "Checkpoint belongs to another run"
    cfg = GPT2Config.from_pretrained(Path(root) / "reference/gpt2", local_files_only=True)
    cfg.vocab_size = metadata["vocab_size"]
    cfg.n_layer = payload["spec"]["unique_layers"]
    cfg.use_cache = False
    model = ReproductionGPT(cfg, payload["spec"]["repeats"])
    model.load_state_dict(payload["model"])
    model.to(device).eval().requires_grad_(False)
    return model, payload["spec"], path


# Forward with interventions ---------------------------------------------------


class Executor:
    """Float32 replica of ReproductionGPT.forward with forward-time edits."""

    def __init__(self, model):
        if model.training:
            raise ValueError("Interventions require eval mode")
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Freeze model parameters")
        t = model.transformer
        self.model, self.t = model, t
        self.blocks = list(t.h)
        self.unique = len(self.blocks)
        self.repeats = model.repeats
        self.executions = self.unique * self.repeats
        self.width = model.config.n_embd
        self.heads = model.config.n_head
        self.neurons = self.blocks[0].mlp.c_fc.weight.shape[1]
        self.device = t.wte.weight.device
        self.embedding = t.wte.weight

    def block_of(self, execution):
        return execution % self.unique

    def scope(self, name):
        """Executions of a named scope (contract z1.scopes)."""
        everything = range(self.executions)
        if name == "all":
            return frozenset(everything)
        if name in ("round1", "round2"):
            if self.repeats < 2:
                raise ValueError("Rounds exist only for looped models")
            return frozenset(e for e in everything if e // self.unique == int(name[-1]) - 1)
        raise ValueError(name)

    def embed(self, tokens):
        return self.t.wte(tokens) + self.t.wpe(torch.arange(tokens.shape[1], device=self.device))

    def run(
        self, x, executions, *, gates=None, gate_scope=(), edges=None, edits=None, capture=False
    ):
        """Run the given executions on residual x.

        gates: [B, unique, neurons] multipliers on GELU activations, applied in
        gate_scope. edges: {execution: [(source, target), ...]} attention edges set
        to zero then renormalized (Zhang et al. Appendix E). edits: {execution:
        callable(x) -> x} applied to the output of that execution. capture
        returns the output of every executed execution.
        """
        boundaries = []
        length = x.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=self.device).tril()
        dh = self.width // self.heads
        for e in executions:
            block = self.blocks[self.block_of(e)]
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(self.width, dim=-1)
            shape = (len(x), length, self.heads, dh)
            q, k, v = (a.reshape(shape).transpose(1, 2) for a in (q, k, v))
            scores = (q @ k.transpose(-1, -2)) / dh**0.5
            weights = scores.masked_fill(~causal, float("-inf")).softmax(-1)
            if edges and edges.get(e):
                mask = torch.zeros(length, length, dtype=torch.bool, device=self.device)
                for source, target in edges[e]:
                    mask[source, target] = True
                weights = weights.masked_fill(mask, 0.0)
                weights = weights / weights.sum(-1, keepdim=True)
            y = (weights @ v).transpose(1, 2).reshape(len(x), length, self.width)
            x = x + block.attn.c_proj(y)
            a = block.mlp.act(block.mlp.c_fc(block.ln_2(x)))
            if gates is not None and e in gate_scope:
                a = a * gates[:, self.block_of(e)][:, None, :]
            x = x + block.mlp.c_proj(a)
            if edits and e in edits:
                x = edits[e](x)
            if capture:
                boundaries.append(x)
        return (x, boundaries) if capture else x

    def readout(self, state):
        return F.linear(self.t.ln_f(state), self.embedding)

    def logits(self, tokens, position, **kwargs):
        x = self.run(self.embed(tokens), range(self.executions), **kwargs)
        return self.readout(x[:, position])

    def gates_for(self, units, count):
        """Dense zero-ablation gates from flat unit indices (block * neurons + n), -1 pads."""
        flat = torch.ones(count, self.unique * self.neurons + 1, device=self.device)
        units = units.clone()
        units[units < 0] = self.unique * self.neurons
        flat.scatter_(1, units, 0.0)
        return flat[:, :-1].reshape(count, self.unique, self.neurons)


def margin(logits, target):
    """logit_t - max over v != t, full vocabulary."""
    own = logits.gather(1, target[:, None]).squeeze(1)
    other = logits.scatter(1, target[:, None], float("-inf")).max(-1).values
    return own - other


@torch.no_grad()
def predict(ex, tokens, position, target=None, *, units=None, scope="all", edges=None, batch=1024):
    """Answers (and margins when target is given) under optional unit ablation."""
    tokens = torch.as_tensor(np.asarray(tokens), device=ex.device)
    answers, margins = [], []
    for start in range(0, len(tokens), batch):
        part = tokens[start : start + batch]
        kwargs = {"edges": edges}
        if units is not None:
            u = torch.as_tensor(np.asarray(units[start : start + batch]), device=ex.device)
            kwargs["gates"] = ex.gates_for(u, len(part))
            kwargs["gate_scope"] = ex.scope(scope) if isinstance(scope, str) else frozenset(scope)
        logits = ex.logits(part, position, **kwargs)
        answers.append(logits.argmax(-1).cpu())
        if target is not None:
            t = torch.as_tensor(np.asarray(target[start : start + batch]), device=ex.device)
            margins.append(margin(logits, t).cpu())
    answer = torch.cat(answers).numpy() if answers else np.zeros(0, np.int64)
    if target is None:
        return answer
    return answer, (torch.cat(margins).numpy() if margins else np.zeros(0, np.float32))


def attribution(ex, tokens, target, *, top, batch=128):
    """Top units by s_u = -d margin / d g_u at g = 1, gate shared over all executions."""
    tokens = torch.as_tensor(np.asarray(tokens), device=ex.device)
    target = torch.as_tensor(np.asarray(target), device=ex.device)
    ranked = []
    everything = ex.scope("all")
    for start in range(0, len(tokens), batch):
        part, t = tokens[start : start + batch], target[start : start + batch]
        g = torch.ones(len(part), ex.unique, ex.neurons, device=ex.device, requires_grad=True)
        logits = ex.logits(part, 1, gates=g, gate_scope=everything)
        (grad,) = torch.autograd.grad(margin(logits, t).sum(), g)
        score = -grad.reshape(len(part), -1)
        ranked.append(score.topk(top, dim=-1).indices.cpu())
    return torch.cat(ranked).numpy() if ranked else np.zeros((0, top), np.int64)


def pad_units(rows, width):
    out = np.full((len(rows), width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


# Z1: fact sample, selection and two-hop transfer -----------------------------


def _grouped(keys, values):
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]
    starts = np.searchsorted(keys, np.arange(keys.max() + 2 if len(keys) else 1))
    return lambda k: values[starts[k] : starts[k + 1]] if k + 1 < len(starts) else values[:0]


def sample_facts(world, phi, p):
    """Deterministic per-run fact sample (contract z1.fact_sample).

    One generator seeded with the fact seed is consumed in a fixed order: ID
    facts, OOD facts, untrained chains per fact, trained II chains of ID facts,
    same-answer controls, same-bridge other-relation chains.
    """
    q = p["z1"]
    rng = np.random.default_rng(q["fact_seed"])
    support = world.support(phi)
    index = np.arange(len(world.chains))
    eligible = {
        "id": (world.kind == 0) & ~support,
        "ood": world.kind == 1,
    }
    facts, kinds, chains = [], [], []
    by_kind = {}
    for kind, n in (("id", q["n_id"]), ("ood", q["n_ood"])):
        counts = np.bincount(world.second[eligible[kind]], minlength=len(world.atoms))
        candidates = np.flatnonzero(counts >= q["min_chains"])
        assert np.all(world.ood[candidates] == (kind == "ood"))
        by_kind[kind] = np.sort(rng.choice(candidates, n, replace=False))
    for kind in ("id", "ood"):
        lookup = _grouped(world.second[eligible[kind]], index[eligible[kind]])
        for f in by_kind[kind]:
            pool = lookup(f)
            chosen = rng.choice(pool, min(q["chains_per_fact"], len(pool)), replace=False)
            facts.append(int(f))
            kinds.append(kind)
            chains.append(np.sort(chosen))
    trained_mask = (world.kind == 0) & support
    trained_lookup = _grouped(world.second[trained_mask], index[trained_mask])
    trained = []
    for f, kind in zip(facts, kinds, strict=True):
        pool = trained_lookup(f) if kind == "id" else np.zeros(0, np.int64)
        trained.append(
            np.sort(rng.choice(pool, min(q["chains_per_fact"], len(pool)), replace=False))
        )
    tail_lookup = _grouped(world.atoms[:, 2], np.arange(len(world.atoms)))
    same_answer = []
    for f in facts:
        b, _, t = world.atoms[f]
        pool = tail_lookup(t)
        pool = pool[(world.atoms[pool, 0] != b) & (pool != f)]
        same_answer.append(
            np.sort(rng.choice(pool, min(q["same_answer"], len(pool)), replace=False))
        )
    first_lookup = _grouped(world.first, index)
    same_bridge = []
    for f, own in zip(facts, chains, strict=True):
        pool = np.unique(np.concatenate([first_lookup(a) for a in np.unique(world.first[own])]))
        pool = pool[world.second[pool] != f]
        same_bridge.append(
            np.sort(rng.choice(pool, min(q["same_bridge"], len(pool)), replace=False))
        )
    random_order = np.random.default_rng(q["random_seed"]).permutation(len(world.atoms))
    random_controls = []
    for f in facts:
        b = world.atoms[f, 0]
        keep = random_order[world.atoms[random_order, 0] != b][: q["random_atoms"]]
        random_controls.append(keep)
    same_subject = []
    for f in facts:
        b = world.atoms[f, 0]
        group = np.flatnonzero(world.atoms[:, 0] == b)
        same_subject.append(group[group != f])
    digest = hashlib.sha256(
        json.dumps([facts, kinds, [c.tolist() for c in chains]]).encode()
    ).hexdigest()
    return {
        "facts": np.asarray(facts, dtype=np.int64),
        "kinds": np.asarray(kinds),
        "chains": chains,
        "trained": trained,
        "same_subject": same_subject,
        "same_answer": same_answer,
        "random": random_controls,
        "same_bridge": same_bridge,
        "sha256": digest,
    }


def random_units(units, neurons, unique, seed, atom):
    """Same per-block counts as `units`, uniform without replacement, excluding them."""
    rng = np.random.default_rng([seed, int(atom)])
    units = np.asarray(units, dtype=np.int64)
    chosen = []
    for block in range(unique):
        need = int(np.sum(units // neurons == block))
        if need:
            pool = np.setdiff1d(np.arange(block * neurons, (block + 1) * neurons), units)
            chosen.append(rng.choice(pool, need, replace=False))
    return np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)


def _flat(pairs):
    """Flatten [(fact position, array of items)] into aligned owner/item arrays."""
    owner = (
        np.concatenate([np.full(len(v), i, np.int64) for i, v in pairs])
        if pairs
        else np.zeros(0, np.int64)
    )
    items = (
        np.concatenate([np.asarray(v, np.int64) for _, v in pairs])
        if pairs
        else np.zeros(0, np.int64)
    )
    return owner, items


def bootstrap_mean(values, spec):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(spec["seed"])
    draws = np.sort(values[rng.integers(0, len(values), (spec["iterations"], len(values)))].mean(1))
    low, high = spec["interval"]
    return {
        "low": float(draws[int(low * (len(draws) - 1))]),
        "high": float(draws[int(high * (len(draws) - 1))]),
        "valid_draws": int(len(draws)),
    }


CONTROLS = ("same_subject", "same_answer", "random")


def _ablate(ex, owner, items, rows, scope, tokens_fn, position, targets):
    """Answers and margins of items, item j under the units rows[owner[j]]."""
    if not len(owner):
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    width = max(1, max(len(rows[o]) for o in owner))
    padded = pad_units([rows[o] for o in owner], width)
    return predict(ex, tokens_fn(items), position, targets, units=padded, scope=scope)


def _per_fact(n, owner, values, facts_index):
    out = np.full(n, np.nan)
    for i in facts_index:
        mask = owner == i
        if mask.any():
            out[i] = float(np.mean(values[mask]))
    return out


def z1_selection(ex, world, sample, atom_answer, p):
    """Z1 selection at one node, from single-hop prompts only.

    Finds S_f (the k_star units) by attribution and the k grid, draws its random
    control units and measures selectivity on the control atoms. No two-hop
    query is evaluated here; z1_node and the selection-only stage both call it.
    """
    q = p["z1"]
    neurons, unique = ex.neurons, ex.unique
    facts = sample["facts"]
    n = len(facts)
    empty = np.zeros(0, np.int64)
    tails = world.atoms[facts, 2] + world.eo
    atom_correct = atom_answer == world.atoms[:, 2] + world.eo
    baseline_ok = atom_correct[facts]

    # Selection on the single-hop prompt only: attribution, then the k grid.
    top = max(q["k_grid"])
    ranked = np.full((n, top), -1, dtype=np.int64)
    ok = np.flatnonzero(baseline_ok)
    if len(ok):
        ranked[ok] = attribution(ex, world.atomic_tokens(facts[ok]), tails[ok], top=top)
    k_star = np.full(n, -1, dtype=np.int64)
    for k in q["k_grid"]:
        pending = ok[k_star[ok] < 0]
        if not len(pending):
            break
        answer = predict(ex, world.atomic_tokens(facts[pending]), 1, units=ranked[pending, :k])
        k_star[pending[answer != tails[pending]]] = k
    localized = np.flatnonzero(k_star > 0)
    units = [ranked[i, : k_star[i]] if k_star[i] > 0 else empty for i in range(n)]
    controls = [
        random_units(units[i], neurons, unique, q["control_seed"], facts[i])
        if k_star[i] > 0
        else empty
        for i in range(n)
    ]

    # Selectivity controls under S_f in all executions.
    retention = {}
    for name in CONTROLS:
        owner, items = _flat(
            [(int(i), sample[name][i][atom_correct[sample[name][i]]]) for i in localized]
        )
        targets = world.atoms[items, 2] + world.eo
        answer, _ = _ablate(ex, owner, items, units, "all", world.atomic_tokens, 1, targets)
        retention[name] = _per_fact(n, owner, answer == targets, localized)
    threshold = q["retention"]

    # A control group left empty by the baseline filter passes and is flagged (contract).
    empty_value = 1.0 if q["empty_control_passes"] else -1.0

    def passes(names):
        out = k_star > 0
        for name in names:
            out = out & (np.nan_to_num(retention[name], nan=empty_value) >= threshold)
        return out

    return {
        "tails": tails,
        "atom_correct": atom_correct,
        "baseline_ok": baseline_ok,
        "k_star": k_star,
        "localized": localized,
        "units": units,
        "controls": controls,
        "retention": retention,
        "selective": passes(CONTROLS),
        "secondary": passes(("same_subject", "random")),
    }


def selection_records(sample, sel):
    """Per-fact selection fields, identical in z1.json and in the selection-only stage."""
    records = []
    for i in range(len(sample["facts"])):
        k = sel["k_star"][i]
        records.append(
            {
                "atom": int(sample["facts"][i]),
                "kind": str(sample["kinds"][i]),
                "baseline_correct": bool(sel["baseline_ok"][i]),
                "k_star": int(k) if k > 0 else None,
                "retention": {
                    name: None
                    if np.isnan(sel["retention"][name][i])
                    else float(sel["retention"][name][i])
                    for name in CONTROLS
                },
                "control_empty": {
                    name: bool(k > 0 and not np.any(sel["atom_correct"][sample[name][i]]))
                    for name in CONTROLS
                },
                "selective": bool(sel["selective"][i]),
                "selective_secondary": bool(sel["secondary"][i]),
                "units": sel["units"][i].tolist(),
                "control_units": sel["controls"][i].tolist(),
            }
        )
    return records


def eligible_counts(ex, world, sample, atom_correct):
    """Eligible untrained chains per fact, from unablated answers only.

    A chain is eligible when its two-hop answer and its first fact's atomic
    answer are correct at baseline (contract z1.two_hop_measure.eligible_query).
    The selection-only stage uses this to count powered nodes; no unit is ablated.
    """
    owner, items = _flat(list(enumerate(sample["chains"])))
    targets = world.chains[items, 3] + world.eo
    answer = predict(ex, world.chain_tokens(items), 2)
    eligible = (answer == targets) & atom_correct[world.first[items]]
    return np.bincount(owner[eligible], minlength=len(sample["facts"]))


def summarize_selection(records, eligible, q):
    """Per-kind counts of the selection-only stage, including the contract's power rule."""
    out = {}
    for kind in ("id", "ood"):
        rows = [(r, int(e)) for r, e in zip(records, eligible, strict=True) if r["kind"] == kind]
        baseline = sum(r["baseline_correct"] for r, _ in rows)
        localized = sum(r["k_star"] is not None for r, _ in rows)
        powered_facts = [e for r, e in rows if r["selective"] and e >= 1]
        hist = {}
        for r, _ in rows:
            if r["k_star"]:
                hist[str(r["k_star"])] = hist.get(str(r["k_star"]), 0) + 1
        out[kind] = {
            "sampled": len(rows),
            "baseline_correct": baseline,
            "localized": localized,
            "not_localized": baseline - localized,
            "not_localized_rate": (baseline - localized) / baseline if baseline else None,
            "selective": sum(r["selective"] for r, _ in rows),
            "selective_secondary": sum(r["selective_secondary"] for r, _ in rows),
            "selective_with_eligible": len(powered_facts),
            "eligible_queries_of_selective": int(sum(powered_facts)),
            "powered": len(powered_facts) >= q["power"],
            "k_star_histogram": hist,
        }
    return out


def z1_node(ex, world, sample, atom_answer, p, scopes):
    """All Z1 measurements at one node. Two-hop outcomes never enter selection."""
    q = p["z1"]
    neurons = ex.neurons
    facts = sample["facts"]
    n = len(facts)
    sel = z1_selection(ex, world, sample, atom_answer, p)
    tails, atom_correct = sel["tails"], sel["atom_correct"]
    k_star, localized = sel["k_star"], sel["localized"]
    units, controls = sel["units"], sel["controls"]

    def ablate(owner, items, rows, scope, tokens_fn, position, targets):
        return _ablate(ex, owner, items, rows, scope, tokens_fn, position, targets)

    def per_fact(owner, values, facts_index):
        return _per_fact(n, owner, values, facts_index)

    # Two-hop transfer: untrained (primary) and trained II chains.
    two_hop = {}
    for label, lists in (("untrained", sample["chains"]), ("trained", sample["trained"])):
        owner, items = _flat([(i, lists[i]) for i in range(n)])
        targets = world.chains[items, 3] + world.eo
        base_answer, base_margin = predict(ex, world.chain_tokens(items), 2, targets)
        eligible = (base_answer == targets) & atom_correct[world.first[items]]
        keep = eligible & (k_star[owner] > 0)
        o, it, tg, bm = owner[keep], items[keep], targets[keep], base_margin[keep]
        outcomes = {}
        for scope in scopes:
            for which, rows in (("S", units), ("R", controls)):
                answer, m = ablate(o, it, rows, scope, world.chain_tokens, 2, tg)
                outcomes[(scope, which)] = (answer != tg, m - bm)
        two_hop[label] = {
            "sampled": owner,
            "owner": o,
            "items": it,
            "targets": tg,
            "outcomes": outcomes,
        }

    records = selection_records(sample, sel)
    for i, rec in enumerate(records):
        for label, block in two_hop.items():
            mask = block["owner"] == i
            rec[f"{label}_sampled"] = int(np.sum(block["sampled"] == i))
            rec[f"{label}_eligible"] = int(mask.sum())
            if mask.any():
                for (scope, which), (flip, change) in block["outcomes"].items():
                    rec[f"{label}_{which}_{scope}"] = float(flip[mask].mean())
                    rec[f"{label}_{which}_{scope}_margin"] = float(change[mask].mean())

    # Descriptive items (never part of a verdict).
    base = two_hop["untrained"]
    if len(localized):
        first_items = world.first[base["items"]]
        targets = world.atoms[first_items, 2] + world.eo
        answer, _ = ablate(
            base["owner"], first_items, units, "all", world.atomic_tokens, 1, targets
        )
        values = per_fact(base["owner"], answer == targets, localized)
        owner, items = _flat([(int(i), sample["same_bridge"][i]) for i in localized])
        targets = world.chains[items, 3] + world.eo
        correct = predict(ex, world.chain_tokens(items), 2) == targets
        owner, items, targets = owner[correct], items[correct], targets[correct]
        scope = "round2" if ex.repeats > 1 else "all"
        answer, _ = ablate(owner, items, units, scope, world.chain_tokens, 2, targets)
        bridge_values = per_fact(owner, answer == targets, localized)
        for i in localized:
            records[i]["first_fact_retention"] = None if np.isnan(values[i]) else float(values[i])
            records[i]["same_bridge_retention"] = (
                None if np.isnan(bridge_values[i]) else float(bridge_values[i])
            )
            records[i]["same_bridge_scope"] = scope
        if ex.repeats > 1:
            for scope in ("round1", "round2"):
                answer, _ = ablate(
                    localized,
                    facts[localized],
                    units,
                    scope,
                    world.atomic_tokens,
                    1,
                    tails[localized],
                )
                for i, a in zip(localized, answer, strict=True):
                    records[i][f"atomic_flip_{scope}"] = bool(a != tails[i])
        else:
            half = ex.unique // 2
            for label, lower in (("lower", True), ("upper", False)):
                rows = [u[(u // neurons < half) == lower] for u in units]
                answer, _ = ablate(
                    base["owner"],
                    base["items"],
                    rows,
                    "all",
                    world.chain_tokens,
                    2,
                    base["targets"],
                )
                flips = per_fact(base["owner"], answer != base["targets"], localized)
                for i in localized:
                    records[i][f"untrained_S_all_{label}_blocks"] = (
                        None if np.isnan(flips[i]) else float(flips[i])
                    )
                    records[i][f"units_{label}_blocks"] = int(len(rows[i]))
    return summarize_z1(records, scopes, q)


def summarize_z1(records, scopes, q):
    out = {"scopes": list(scopes), "kinds": {}, "facts": records}
    for kind in ("id", "ood"):
        rows = [r for r in records if r["kind"] == kind]
        hist = {}
        for r in rows:
            if r["k_star"]:
                hist[str(r["k_star"])] = hist.get(str(r["k_star"]), 0) + 1
        summary = {
            "sampled": len(rows),
            "baseline_correct": sum(r["baseline_correct"] for r in rows),
            "localized": sum(r["k_star"] is not None for r in rows),
            "selective": sum(r["selective"] for r in rows),
            "selective_secondary": sum(r["selective_secondary"] for r in rows),
            "control_empty": {
                n: sum(r["control_empty"][n] for r in rows)
                for n in ("same_subject", "same_answer", "random")
            },
            "k_star_histogram": hist,
        }
        for population, flag in (
            ("primary", "selective"),
            ("secondary_population", "selective_secondary"),
        ):
            for label in ("untrained", "trained"):
                chosen = [r for r in rows if r[flag] and r.get(f"{label}_eligible", 0) >= 1]
                block = {
                    "facts_with_eligible": len(chosen),
                    "eligible_queries": int(sum(r[f"{label}_eligible"] for r in chosen)),
                }
                for scope in scopes:
                    t = [r[f"{label}_S_{scope}"] for r in chosen]
                    c = [r[f"{label}_R_{scope}"] for r in chosen]
                    diffs = [a - b for a, b in zip(t, c, strict=True)]
                    block[scope] = {
                        "delta": float(np.mean(diffs)) if diffs else None,
                        "interval": bootstrap_mean(diffs, q["bootstrap"]),
                        "T": float(np.mean(t)) if t else None,
                        "R": float(np.mean(c)) if c else None,
                        "margin_change_S": _mean([r[f"{label}_S_{scope}_margin"] for r in chosen]),
                        "margin_change_R": _mean([r[f"{label}_R_{scope}_margin"] for r in chosen]),
                    }
                summary[f"{population}/{label}"] = block
        sel = [r for r in rows if r["selective"]]
        summary["descriptive_selective"] = {
            "first_fact_retention": _mean([r.get("first_fact_retention") for r in sel]),
            "same_bridge_retention": _mean([r.get("same_bridge_retention") for r in sel]),
            "atomic_flip_round1": _mean([r.get("atomic_flip_round1") for r in sel]),
            "atomic_flip_round2": _mean([r.get("atomic_flip_round2") for r in sel]),
            "untrained_S_all_lower_blocks": _mean(
                [r.get("untrained_S_all_lower_blocks") for r in sel]
            ),
            "untrained_S_all_upper_blocks": _mean(
                [r.get("untrained_S_all_upper_blocks") for r in sel]
            ),
            "retention_mean": {
                n: _mean([r["retention"][n] for r in rows if r["k_star"]])
                for n in ("same_subject", "same_answer", "random")
            },
        }
        out["kinds"][kind] = summary
    return out


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


# Ported Zhang et al. diagnostics ---------------------------------------------


def window_edges(executions, centre, window, edge):
    """Zhang et al. script 8: half_left = w - w//2 - 1, half_right = w//2."""
    lower = max(0, centre - (window - window // 2 - 1))
    upper = min(executions - 1, centre + window // 2)
    return {e: [edge] for e in range(lower, upper + 1)}


@torch.no_grad()
def ported_node(ex, world, phi, p):
    q = p["z1"]
    panels = world.panels
    out = {
        "attention_window_masking": {},
        "layerwise_cosine": {},
        "bridge_patching": {},
        "logit_lens": {},
    }
    E = ex.executions
    for window in q["windows"]:
        result = {}
        for name in PANELS_ATOMIC:
            rows = panels[name]
            accs = []
            for c in range(E):
                answer = predict(
                    ex, world.row_tokens(rows), 1, edges=window_edges(E, c, window, (1, 0))
                )
                accs.append(float(np.mean(answer == rows[:, 2] + world.eo)))
            result[f"{name}/1->0"] = accs
        for name in PANELS_TWO_HOP:
            rows = panels[name]
            for edge in ((2, 1), (1, 0)):
                accs = []
                for c in range(E):
                    answer = predict(
                        ex, world.row_tokens(rows), 2, edges=window_edges(E, c, window, edge)
                    )
                    accs.append(float(np.mean(answer == rows[:, 3] + world.eo)))
                result[f"{name}/{edge[0]}->{edge[1]}"] = accs
        out["attention_window_masking"][f"window_{window}"] = result
    # Layer-by-layer cosine (script 11): boundary 0 is wte without position.
    for family, name in (("id", "train_ii"), ("ood", "test_io")):
        rows = panels[name]
        _, second, _ = world.row_facts(rows)
        facts = np.unique(second)
        atomic = boundary_states(ex, world.atomic_tokens(facts))
        chain = boundary_states(ex, world.row_tokens(rows))
        index = {int(f): i for i, f in enumerate(facts)}
        groups = np.asarray([index[int(f)] for f in second])
        mats = {}
        for label, a_pos, c_pos in (("entity", 0, 1), ("relation", 1, 2)):
            a = atomic[:, :, a_pos]
            c = torch.zeros(a.shape, device=ex.device)
            c.index_add_(1, torch.as_tensor(groups, device=ex.device), chain[:, :, c_pos])
            counts = torch.bincount(torch.as_tensor(groups, device=ex.device), minlength=len(facts))
            c = c / counts[None, :, None]
            a_n, c_n = F.normalize(a, dim=-1), F.normalize(c, dim=-1)
            mats[label] = torch.einsum("ifd,jfd->ijf", a_n, c_n).mean(-1).cpu().tolist()
        out["layerwise_cosine"][family] = {
            **mats,
            "facts": int(len(facts)),
            "chains": int(len(rows)),
        }
    out["bridge_patching"] = bridge_patching(ex, world, phi, p)
    for name in PANELS_TWO_HOP:
        rows = panels[name]
        _, _, bridge = world.row_facts(rows)
        states = boundary_states(ex, world.row_tokens(rows))[1:, :, 1]
        accs = []
        for b in range(states.shape[0]):
            answer = batched_readout(ex, states[b])
            accs.append(float(np.mean(answer == bridge + world.eo)))
        out["logit_lens"][name] = accs
    return out


@torch.no_grad()
def boundary_states(ex, tokens, batch=1024):
    """[boundaries 0..E, N, L, d]; boundary 0 is the token embedding without position."""
    tokens = torch.as_tensor(np.asarray(tokens), device=ex.device)
    parts = []
    for start in range(0, len(tokens), batch):
        t = tokens[start : start + batch]
        _, states = ex.run(ex.embed(t), range(ex.executions), capture=True)
        parts.append(torch.stack([ex.t.wte(t), *states]))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def batched_readout(ex, states, batch=1024):
    return torch.cat(
        [ex.readout(states[s : s + batch]).argmax(-1).cpu() for s in range(0, len(states), batch)]
    ).numpy()


@torch.no_grad()
def bridge_patching(ex, world, phi, p):
    """Zhang et al. script 5 (Section 4.2), ported to ReproductionGPT."""
    spec = p["z1"]["patch"]
    rng = np.random.default_rng(spec["seed"])
    rows = world.panels["train_ii"]
    first, _, bridge = world.row_facts(rows)
    support = world.chains[world.support(phi)]
    by_r2 = {}
    for r2 in np.unique(rows[:, 2]):
        by_r2[int(r2)] = support[support[:, 2] == r2]
    tail_lookup = _grouped(world.atoms[:, 2], np.arange(len(world.atoms)))
    tuples = {"id": [], "ood": []}
    for i, (row, f, b) in enumerate(zip(rows, first, bridge, strict=True)):
        pool = by_r2[int(row[2])]
        pool = pool[pool[:, 3] != row[3]]
        if not len(pool):
            continue
        r2_source = pool[rng.integers(len(pool))]
        prefixes = tail_lookup(b)
        for prefix in prefixes:
            family = "ood" if world.ood[prefix] else "id"
            if family == "id" and prefix == f:
                continue
            tuples[family].append((i, int(prefix), *r2_source.tolist()))
    out = {}
    for family in ("id", "ood"):
        chosen = tuples[family]
        order = rng.permutation(len(chosen))[: spec[family]]
        chosen = np.asarray([chosen[j] for j in order], dtype=np.int64).reshape(-1, 6)
        target_rows = rows[chosen[:, 0]]
        prefix_states = boundary_states(ex, world.atomic_tokens(chosen[:, 1]))[1:, :, 1]
        r2_tokens = world.row_tokens(chosen[:, 2:6])
        r2_states = boundary_states(ex, r2_tokens)[1:, :, 2]
        tokens = torch.as_tensor(world.row_tokens(target_rows), device=ex.device)
        accs = []
        for layer in range(1, ex.executions + 1):
            answers = []
            for s in range(0, len(tokens), 1024):
                t = tokens[s : s + 1024]
                ps, rs = prefix_states[layer - 1, s : s + 1024], r2_states[layer - 1, s : s + 1024]

                def patch(x, ps=ps, rs=rs):
                    x = x.clone()
                    x[:, 1], x[:, 2] = ps, rs
                    return x

                x = ex.run(ex.embed(t), range(ex.executions), edits={layer - 1: patch})
                answers.append(ex.readout(x[:, 2]).argmax(-1).cpu())
            answer = torch.cat(answers).numpy()
            accs.append(float(np.mean(answer == target_rows[:, 3] + world.eo)))
        out[family] = {
            "accuracy_by_layer": accs,
            "tuples": int(len(chosen)),
            "available": len(tuples[family]),
        }
    return out


# Replication check -----------------------------------------------------------


@torch.no_grad()
def replication(ex, world, root, run, step, p):
    rep = p["replication"]
    result, failures = {}, []
    for name in PANELS_ATOMIC + PANELS_TWO_HOP:
        stored = np.load(Path(root) / "development" / run / f"predictions-{step:07d}-{name}.npz")
        rows = world.panels[name]
        assert np.array_equal(stored["rows"], rows), f"Stored rows differ for {name}"
        position = 1 if rows.shape[1] == 3 else 2
        answer = predict(ex, world.row_tokens(rows), position)
        agreement = float(np.mean(answer == stored["answer"]))
        result[name] = {
            "agreement": agreement,
            "fp32_answer_accuracy": float(np.mean(answer == rows[:, -1] + world.eo)),
            "stored_answer_accuracy": float(np.mean(stored["answer"] == rows[:, -1] + world.eo)),
        }
        if step >= rep["from_step"] and agreement < rep["min_agreement"]:
            failures.append(f"{name} agreement {agreement:.4f} < {rep['min_agreement']}")
        if step < rep["from_step"] and agreement < rep["early_stop"]:
            failures.append(
                f"{name} agreement {agreement:.4f} < {rep['early_stop']} "
                f"before step {rep['from_step']}"
            )
        if step == max(p["nodes"]) and name == "test_ii" and agreement < rep["final_test_ii"]:
            failures.append(
                f"test_ii agreement {agreement:.4f} < {rep['final_test_ii']} at final node"
            )
    return result, failures


# Verdicts --------------------------------------------------------------------


def majority_verdict(flags_support, flags_against, powered, total):
    """Majority of powered nodes; inconclusive when fewer than half are powered."""
    if powered < total / 2:
        return "inconclusive"
    if flags_support > powered / 2:
        return "supports"
    if flags_against > powered / 2:
        return "against"
    return "partial"


def z1_loop_flags(block, p):
    """Per-node Loop flags on primary/untrained OOD facts (contract z1.interpretation)."""
    q = p["z1"]
    sup2, sup1 = q["loop_supports"]
    d2, d1 = block["round2"]["delta"], block["round1"]["delta"]
    return {
        "delta_round2": d2,
        "delta_round1": d1,
        "interval_round2": block["round2"]["interval"],
        "support": d2 is not None and d1 is not None and d2 >= sup2 and d1 <= sup1,
        "against": d2 is not None and d2 <= q["loop_against"],
    }


def z1_standard_flags(block, p):
    """Per-node standard-8 flags: support = separate copy, against = shared units."""
    q = p["z1"]
    d = block["all"]["delta"]
    return {
        "delta_all": d,
        "interval_all": block["all"]["interval"],
        "support": d is not None and d <= q["standard_separate"],
        "against": d is not None and d >= q["standard_shared"],
    }
