"""Post-hoc, position-specific diagnostics of completed Z1 interventions.

This module does not select neurons, facts, thresholds or checkpoints. It reuses
the primary design's saved S_f and R_f, reconstructs its frozen query sample,
and checks the original measurements before computing new diagnostics. Every
operation is an eval-mode forward pass; model parameters are never written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .path_overlap import (
    Executor,
    World,
    bootstrap_mean,
    load_contract,
    load_model,
    margin,
    pad_units,
    predict,
    sample_facts,
    sha256,
)

CODE_FILES = (
    "src/llm_memory_editability/path_overlap.py",
    "src/llm_memory_editability/path_overlap_followup.py",
    "scripts/run_path_overlap_followup.py",
)


class PositionExecutor(Executor):
    """The complete Z1 Transformer computation with optional position gates.

    ``gate_positions=None`` preserves the original all-position operation.
    Positions refer to the input sequence: 0=h, 1=r1, 2=r2 for a chain.
    Residual replacements use the inherited ``edits`` interface. Restoring a
    boundary prefix does not undo information already read by position 2.
    """

    def run(
        self,
        x,
        executions,
        *,
        gates=None,
        gate_scope=(),
        edges=None,
        edits=None,
        capture=False,
        gate_positions=None,
    ):
        boundaries = []
        length = x.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=self.device).tril()
        position_mask = None
        if gate_positions is not None:
            positions = tuple(gate_positions)
            if any(p < 0 or p >= length for p in positions) or len(set(positions)) != len(
                positions
            ):
                raise ValueError("Gate positions must be distinct, valid sequence positions")
            position_mask = torch.zeros(1, length, 1, dtype=torch.bool, device=self.device)
            position_mask[:, list(positions)] = True
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
                g = gates[:, self.block_of(e)][:, None, :]
                if position_mask is not None:
                    g = torch.where(position_mask, g, torch.ones_like(g))
                a = a * g
            x = x + block.mlp.c_proj(a)
            if edits and e in edits:
                x = edits[e](x)
            if capture:
                boundaries.append(x)
        return (x, boundaries) if capture else x


@dataclass(frozen=True)
class Mode:
    name: str
    scope: str | None = None
    positions: tuple[int, ...] | None = None
    restore_prefix: bool = False


LOOP_MODES = (
    Mode("baseline"),
    Mode("baseline_prefix_identity", restore_prefix=True),
    Mode("round1_all", "round1"),
    Mode("round2_all", "round2"),
    Mode("round1_last", "round1", (2,)),
    Mode("round1_prefix", "round1", (0, 1)),
    Mode("round1_all_prefix_restored", "round1", restore_prefix=True),
    Mode("round2_last", "round2", (2,)),
)
STANDARD_MODES = (
    Mode("baseline"),
    Mode("baseline_prefix_identity", restore_prefix=True),
    Mode("all8_all", "all"),
    Mode("all8_last", "all", (2,)),
    Mode("all8_prefix", "all", (0, 1)),
)


def prefix_restore(donor):
    """Replace both causal prefix positions with this query's unablated state."""
    if donor.shape[1] != 3:
        raise ValueError("Prefix restoration requires a three-token chain query")

    def restore(x):
        if x.shape != donor.shape:
            raise ValueError("Donor and recipient shapes differ")
        out = x.clone()
        out[:, :2] = donor[:, :2]
        return out

    return restore


@torch.no_grad()
def mode_predict(
    ex, tokens, target, mode, units=None, *, batch=1024, bridge_target=None, bridge_capture=None
):
    """Exact full forward with same-query unablated prefix donor if requested."""
    tokens, target = np.asarray(tokens), np.asarray(target)
    answers, margins = [], []
    bridge_parts = {
        key: []
        for key in ("before_predictions", "before_margins", "after_predictions", "after_margins")
    }
    if (bridge_target is None) != (bridge_capture is None):
        raise ValueError("Bridge readout requires both targets and an output dictionary")
    for start in range(0, len(tokens), batch):
        part = torch.as_tensor(tokens[start : start + batch], device=ex.device)
        kwargs = {}
        if mode.scope is not None:
            if units is None:
                raise ValueError("An ablation mode requires frozen units")
            u = torch.as_tensor(np.asarray(units[start : start + batch]), device=ex.device)
            kwargs.update(
                gates=ex.gates_for(u, len(part)),
                gate_scope=ex.scope(mode.scope),
                gate_positions=mode.positions,
            )
        if mode.restore_prefix or bridge_capture is not None:
            # Loop boundary after its first unique-layer pass. Standard's
            # identity control is after four layers; it is not a Loop round.
            boundary = ex.unique - 1 if ex.repeats > 1 else ex.executions // 2 - 1
            restore = None
            if mode.restore_prefix:
                donor = ex.run(ex.embed(part), range(boundary + 1))
                restore = prefix_restore(donor)

            def at_boundary(x, restore=restore, start=start):
                if bridge_capture is not None:
                    bt = torch.as_tensor(
                        np.asarray(bridge_target[start : start + batch]), device=ex.device
                    )
                    before = ex.readout(x[:, 1])
                    bridge_parts["before_predictions"].append(before.argmax(-1).cpu().numpy())
                    bridge_parts["before_margins"].append(margin(before, bt).cpu().numpy())
                out = restore(x) if restore is not None else x
                if bridge_capture is not None:
                    after = ex.readout(out[:, 1])
                    bridge_parts["after_predictions"].append(after.argmax(-1).cpu().numpy())
                    bridge_parts["after_margins"].append(margin(after, bt).cpu().numpy())
                return out

            kwargs["edits"] = {boundary: at_boundary}
        logits = ex.logits(part, 2, **kwargs)
        t = torch.as_tensor(target[start : start + batch], device=ex.device)
        answers.append(logits.argmax(-1).cpu().numpy())
        margins.append(margin(logits, t).cpu().numpy())
    if bridge_capture is not None:
        for key, parts in bridge_parts.items():
            dtype = np.int64 if key.endswith("predictions") else np.float32
            bridge_capture[key] = np.concatenate(parts) if parts else np.zeros(0, dtype)
    return (
        np.concatenate(answers) if answers else np.zeros(0, np.int64),
        np.concatenate(margins) if margins else np.zeros(0, np.float32),
    )


def _flatten(lists):
    if not lists:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    owners = np.concatenate([np.full(len(items), i, np.int64) for i, items in lists])
    items = np.concatenate([np.asarray(items, np.int64) for _, items in lists])
    return owners, items


def _units_for(records, owners, field):
    rows = [records[int(i)][field] for i in owners]
    return pad_units(rows, max(1, max(map(len, rows), default=0)))


def _fact_values(owners, values, fact_count):
    out = np.full(fact_count, np.nan)
    for i in np.unique(owners):
        out[i] = float(np.mean(values[owners == i]))
    return out


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_frozen_source(config, index, device):
    """Validate provenance and saved selection; never invoke attribution."""
    source = config["source"]
    root, science = Path(source["root"]), Path(source["science_root"])
    node = config["nodes"][index]
    run, step = node["run"], int(node["step"])
    original_dir = science / run / f"{step:07d}"
    original = json.loads((original_dir / "z1.json").read_text())
    provenance = json.loads((original_dir / "hashes.json").read_text())
    contract, p = load_contract(source["contract"])
    if not provenance["complete"] or original["run"] != run or original["step"] != step:
        raise ValueError("Source node is incomplete or belongs to another checkpoint")
    for name, digest in provenance["outputs_sha256"].items():
        if sha256(original_dir / name) != digest:
            raise ValueError(f"Source output hash differs: {name}")
    if sha256(source["contract"]) != provenance["contract_sha256"]:
        raise ValueError("Original Z1 contract hash differs")
    if (
        sha256(Path(__file__).with_name("path_overlap.py"))
        != provenance["code_sha256"]["src/llm_memory_editability/path_overlap.py"]
    ):
        raise ValueError("Original Executor implementation differs from audited Z1 source")
    data = root / "data"
    for name in ("world", "panels"):
        digest = sha256(data / f"{name}.npz")
        if digest != provenance[f"{name}_sha256"] or digest != contract["source"][f"{name}_sha256"]:
            raise ValueError(f"Original {name} hash differs")
    world = World(data)
    model, spec, checkpoint = load_model(root, run, step, world.metadata, device)
    if sha256(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash differs from audited Z1 source")
    if model.repeats not in (1, 2) or len(model.transformer.h) * model.repeats != 8:
        raise ValueError("Follow-up requires the original standard8 or Loop4x2")
    sample = sample_facts(world, spec["phi"], p)
    records = original["primary_design"]["facts"]
    if sample["sha256"] != original["fact_sample_sha256"]:
        raise ValueError("Frozen sample cannot be reconstructed")
    if [r["atom"] for r in records] != sample["facts"].tolist() or [
        r["kind"] for r in records
    ] != sample["kinds"].tolist():
        raise ValueError("Frozen source fact order differs")
    for rec in records:
        units, control = rec["units"], rec["control_units"]
        if len(set(units)) != len(units) or len(units) != len(control):
            raise ValueError("Source S/R units are invalid")
        if set(units) & set(control):
            raise ValueError("Source S and R overlap")
    return world, model, sample, records, original, provenance, original_dir, checkpoint


def _old_float_equal(new, old, label, *, margins=False):
    if old is None:
        if not np.isnan(new):
            raise ValueError(f"Original empty value differs: {label}")
    elif margins:
        # Same allowance as float32 reproducibility; predictions remain exact.
        if not np.isclose(new, old, rtol=2e-5, atol=5e-4):
            raise ValueError(f"Original float32 margin differs: {label}: {new} vs {old}")
    elif new != old:
        raise ValueError(f"Original fact-level value differs: {label}: {new} vs {old}")


def reproduce_original(ex, world, sample, records, original, *, batch=1024):
    """Reproduce saved eligibility and all original scope effects before analysis.

    Z1 saved per-fact outcomes, not query predictions. We therefore additionally
    compare exact argmax against the unchanged original Executor on every frozen
    eligible query, with both S_f and R_f. This limitation is recorded explicitly.
    """
    old_ex = Executor(ex.model)
    atom_index = np.arange(len(world.atoms))
    atoms = predict(old_ex, world.atomic_tokens(atom_index), 1, batch=batch)
    new_atoms = predict(ex, world.atomic_tokens(atom_index), 1, batch=batch)
    if not np.array_equal(atoms, new_atoms):
        raise ValueError("PositionExecutor baseline atomic argmax differs from original Executor")
    atom_correct = atoms == world.atoms[:, 2] + world.eo
    if float(np.mean(atom_correct)) != original["atomic_all_answer_accuracy"]:
        raise ValueError("Original atomic baseline accuracy differs")
    sampled_owner, sampled_chain = _flatten(list(enumerate(sample["chains"])))
    targets = world.chains[sampled_chain, 3] + world.eo
    base, base_margin = predict(old_ex, world.chain_tokens(sampled_chain), 2, targets, batch=batch)
    new_base, new_margin = mode_predict(
        ex, world.chain_tokens(sampled_chain), targets, Mode("baseline"), batch=batch
    )
    if not np.array_equal(base, new_base) or not np.allclose(
        base_margin, new_margin, rtol=2e-5, atol=5e-4
    ):
        raise ValueError("PositionExecutor baseline chain differs from original Executor")
    localized = np.asarray([r["k_star"] is not None for r in records])
    eligible = (base == targets) & atom_correct[world.first[sampled_chain]]
    keep = eligible & localized[sampled_owner]
    owner, chains, targets, bm = (
        sampled_owner[keep],
        sampled_chain[keep],
        targets[keep],
        base_margin[keep],
    )
    counts = np.bincount(owner, minlength=len(records))
    for i, rec in enumerate(records):
        if counts[i] != rec["untrained_eligible"]:
            raise ValueError(f"Original eligible coverage differs for fact {rec['atom']}")
        if bool(atom_correct[rec["atom"]]) != rec["baseline_correct"]:
            raise ValueError("Original fact baseline differs")
    arrays = {
        "atomic_baseline_predictions": atoms,
        "sampled_owner": sampled_owner,
        "sampled_chain_ids": sampled_chain,
        "sampled_baseline_predictions": base,
        "sampled_baseline_margins": base_margin,
        "sampled_eligible": eligible,
        "source_eligible_owner": owner,
        "source_eligible_chain_ids": chains,
        "source_eligible_targets": targets,
    }
    scopes = ("all", "round1", "round2") if ex.repeats > 1 else ("all",)
    source_rates = {}
    for scope in scopes:
        for which, field in (("S", "units"), ("R", "control_units")):
            units = _units_for(records, owner, field)
            answer, m = predict(
                old_ex,
                world.chain_tokens(chains),
                2,
                targets,
                units=units,
                scope=scope,
                batch=batch,
            )
            newer, nm = mode_predict(
                ex,
                world.chain_tokens(chains),
                targets,
                Mode(scope, scope),
                units,
                batch=batch,
            )
            if not np.array_equal(answer, newer) or not np.allclose(m, nm, rtol=2e-5, atol=5e-4):
                raise ValueError(f"Original Executor argmax/margin differs: {scope}/{which}")
            flips = _fact_values(owner, answer != targets, len(records))
            changes = _fact_values(owner, m - bm, len(records))
            for i in np.unique(owner):
                _old_float_equal(flips[i], records[i][f"untrained_{which}_{scope}"], scope)
                _old_float_equal(
                    changes[i],
                    records[i][f"untrained_{which}_{scope}_margin"],
                    scope,
                    margins=True,
                )
            arrays[f"original_{scope}_{which}_predictions"] = answer
            arrays[f"original_{scope}_{which}_margins"] = m
            source_rates[scope, which] = flips
        for kind in ("id", "ood"):
            chosen = np.asarray(
                [
                    i
                    for i, rec in enumerate(records)
                    if rec["kind"] == kind and rec["selective"] and counts[i] > 0
                ],
                dtype=np.int64,
            )
            old_group = original["primary_design"]["kinds"][kind]["primary/untrained"]
            if (
                len(chosen) != old_group["facts_with_eligible"]
                or int(counts[chosen].sum()) != old_group["eligible_queries"]
            ):
                raise ValueError("Original primary aggregate coverage differs")
            for metric, values in (
                ("T", source_rates[scope, "S"]),
                ("R", source_rates[scope, "R"]),
                ("delta", source_rates[scope, "S"] - source_rates[scope, "R"]),
            ):
                _old_float_equal(
                    float(np.mean(values[chosen])) if len(chosen) else np.nan,
                    old_group[scope][metric],
                    f"{kind}/{scope}/{metric}",
                )
    # Saved first-hop atomic retention checks the original all-position damage.
    first = world.first[chains]
    first_target = world.atoms[first, 2] + world.eo
    first_answer = predict(
        old_ex,
        world.atomic_tokens(first),
        1,
        units=_units_for(records, owner, "units"),
        batch=batch,
    )
    first_retention = _fact_values(owner, first_answer == first_target, len(records))
    for i in np.unique(owner):
        _old_float_equal(
            first_retention[i], records[i].get("first_fact_retention"), "first_fact_retention"
        )
    arrays.update(
        first_fact_ids=first,
        first_fact_targets=first_target,
        first_fact_ablation_predictions=first_answer,
    )
    return (
        arrays,
        atom_correct,
        {
            "passed": True,
            "eligible_query_count": int(len(owner)),
            "eligible_fact_count": int(len(np.unique(owner))),
            "source_scope_checks": list(scopes),
            "query_argmax_exact_against_original_executor": True,
            "saved_per_fact_flip_rates_exact": True,
            "saved_first_fact_retention_exact": True,
            "saved_primary_aggregate_deltas_exact": True,
            "original_source_has_query_level_predictions": False,
            "margin_atol": 5e-4,
            "margin_rtol": 2e-5,
        },
    )


def _aggregate(records, owners, targets, outcomes, bootstrap, population):
    summaries, per_fact = {}, {}
    count = len(records)
    for name, pair in outcomes.items():
        s_answer, s_margin = pair["S"]
        r_answer, r_margin = pair["R"]
        t = _fact_values(owners, s_answer != targets, count)
        r = _fact_values(owners, r_answer != targets, count)
        delta = t - r
        per_fact[name] = {"T": t, "R": r, "delta": delta}
        summary = {}
        for kind in ("id", "ood"):
            chosen = np.asarray(
                [i for i, rec in enumerate(records) if rec["kind"] == kind and population[i]],
                dtype=np.int64,
            )
            chosen = chosen[np.isfinite(delta[chosen])]
            summary[kind] = {
                "facts": int(len(chosen)),
                "eligible_queries": int(np.isin(owners, chosen).sum()),
                "T": float(np.mean(t[chosen])) if len(chosen) else None,
                "R": float(np.mean(r[chosen])) if len(chosen) else None,
                "delta": float(np.mean(delta[chosen])) if len(chosen) else None,
                "interval": bootstrap_mean(delta[chosen], bootstrap),
                "margin_mean_S": float(np.mean(s_margin[np.isin(owners, chosen)]))
                if len(chosen)
                else None,
                "margin_mean_R": float(np.mean(r_margin[np.isin(owners, chosen)]))
                if len(chosen)
                else None,
            }
        summaries[name] = summary
    return summaries, per_fact


@torch.no_grad()
def verify_prefix_causality(ex, world, tokens, *, batch=1024):
    """Actual query prefix states must be identical when only r2 changes."""
    boundary = ex.unique - 1 if ex.repeats > 1 else ex.executions // 2 - 1
    for start in range(0, len(tokens), batch):
        original = np.asarray(tokens[start : start + batch])
        alternate = original.copy()
        alternate[:, 2] = world.ro + (
            (alternate[:, 2] - world.ro + 1) % world.metadata["relations"]
        )
        a = torch.as_tensor(original, device=ex.device)
        b = torch.as_tensor(alternate, device=ex.device)
        donor_a = ex.run(ex.embed(a), range(boundary + 1))
        donor_b = ex.run(ex.embed(b), range(boundary + 1))
        if not torch.equal(donor_a[:, :2], donor_b[:, :2]):
            raise ValueError("Boundary prefix donor depends on the future second relation")
    return {"passed": True, "queries_checked": int(len(tokens)), "changed_only_position": 2}


def _boundary_summary(records, owners, targets, captures, population):
    out = {}
    for name, values in captures.items():
        out[name] = {}
        for which, capture in values.items():
            out[name][which] = {}
            for when in ("before", "after"):
                rates = _fact_values(
                    owners, capture[f"{when}_predictions"] == targets, len(records)
                )
                out[name][which][when] = {}
                for kind in ("id", "ood"):
                    indices = np.asarray(
                        [
                            i
                            for i, rec in enumerate(records)
                            if rec["kind"] == kind and population[i] and np.isfinite(rates[i])
                        ],
                        dtype=np.int64,
                    )
                    out[name][which][when][kind] = {
                        "facts": int(len(indices)),
                        "accuracy": float(np.mean(rates[indices])) if len(indices) else None,
                    }
    return out


def run_diagnostic(config, index, device, *, smoke=False):
    world, model, sample, records, original, provenance, source_dir, checkpoint = (
        load_frozen_source(config, index, device)
    )
    ex = PositionExecutor(model)
    batch = int(config.get("execution", {}).get("batch_size", 1024))
    modes = LOOP_MODES if ex.repeats > 1 else STANDARD_MODES
    declared = config.get("analysis", {}).get("modes", {})
    expected = declared.get("loop" if ex.repeats > 1 else "standard")
    if expected is not None and expected != [m.name for m in modes]:
        raise ValueError("Declared diagnostic modes differ from implementation")
    if config["analysis"].get("fact_limit") is not None:
        raise ValueError("Formal diagnostic must reuse the complete saved primary population")
    arrays, atom_correct, reproduction = reproduce_original(
        ex, world, sample, records, original, batch=batch
    )
    owners = arrays["source_eligible_owner"]
    chains = arrays["source_eligible_chain_ids"]
    target = arrays["source_eligible_targets"]
    population = np.asarray([r["selective"] for r in records])
    coverage = {}
    counts = np.bincount(owners, minlength=len(records))
    for kind in ("id", "ood"):
        indices = np.asarray([i for i, r in enumerate(records) if r["kind"] == kind])
        chosen = indices[population[indices] & (counts[indices] > 0)]
        coverage[kind] = {
            "sampled_facts": int(len(indices)),
            "baseline_correct_facts": int(sum(records[i]["baseline_correct"] for i in indices)),
            "localized_facts": int(sum(records[i]["k_star"] is not None for i in indices)),
            "selective_facts": int(population[indices].sum()),
            "selective_with_eligible_facts": int(len(chosen)),
            "eligible_queries": int(counts[chosen].sum()),
        }
    if smoke:
        selected = np.zeros(len(records), bool)
        for kind in ("id", "ood"):
            selected[
                [
                    i
                    for i, r in enumerate(records)
                    if r["kind"] == kind and population[i] and counts[i] > 0
                ][:2]
            ] = True
        population &= selected
    keep = population[owners]
    owners, chains, target = owners[keep], chains[keep], target[keep]
    tokens = world.chain_tokens(chains)
    causal_check = verify_prefix_causality(ex, world, tokens, batch=batch)
    bridge_targets = world.atoms[world.first[chains], 2] + world.eo
    arrays.update(
        primary_owner=owners,
        primary_chain_ids=chains,
        primary_tokens=tokens,
        primary_targets=target,
        primary_boundary_bridge_targets=bridge_targets,
    )
    outcomes, primary_captures = {}, {}
    for mode in modes:
        outcomes[mode.name] = {}
        primary_captures[mode.name] = {}
        for which, field in (("S", "units"), ("R", "control_units")):
            capture = {}
            answer, margins = mode_predict(
                ex,
                tokens,
                target,
                mode,
                _units_for(records, owners, field),
                batch=batch,
                bridge_target=bridge_targets,
                bridge_capture=capture,
            )
            outcomes[mode.name][which] = answer, margins
            primary_captures[mode.name][which] = capture
            arrays[f"primary_{mode.name}_{which}_predictions"] = answer
            arrays[f"primary_{mode.name}_{which}_margins"] = margins
            for key, values in capture.items():
                arrays[f"primary_{mode.name}_{which}_bridge_{key}"] = values
    if not np.array_equal(
        outcomes["baseline"]["S"][0], outcomes["baseline_prefix_identity"]["S"][0]
    ) or not np.allclose(
        outcomes["baseline"]["S"][1],
        outcomes["baseline_prefix_identity"]["S"][1],
        rtol=0,
        atol=0,
    ):
        raise ValueError("Unablated prefix restoration is not an exact identity")
    bootstrap = {
        "iterations": int(config["analysis"]["bootstrap_draws"]),
        "seed": int(config["analysis"]["seed"]),
        "interval": [0.025, 0.975],
    }
    primary, fact_measures = _aggregate(records, owners, target, outcomes, bootstrap, population)

    # Same-bridge, different-r2 queries come from the same previously drawn list.
    # Taking its first four is fixed; no outcome-dependent replacement occurs.
    bridge_owner, bridge_chains = _flatten(
        [(i, sample["same_bridge"][i][:4]) for i in np.flatnonzero(population)]
    )
    bridge_target = world.chains[bridge_chains, 3] + world.eo
    bridge_base, bridge_margin = mode_predict(
        ex, world.chain_tokens(bridge_chains), bridge_target, Mode("baseline"), batch=batch
    )
    bridge_eligible = (bridge_base == bridge_target) & atom_correct[world.first[bridge_chains]]
    arrays.update(
        same_bridge_sampled_owner=bridge_owner,
        same_bridge_sampled_chain_ids=bridge_chains,
        same_bridge_sampled_baseline_predictions=bridge_base,
        same_bridge_sampled_baseline_margins=bridge_margin,
        same_bridge_sampled_targets=bridge_target,
        same_bridge_sampled_eligible=bridge_eligible,
    )
    bridge_owner, bridge_chains, bridge_target = (
        bridge_owner[bridge_eligible],
        bridge_chains[bridge_eligible],
        bridge_target[bridge_eligible],
    )
    arrays.update(
        same_bridge_owner=bridge_owner,
        same_bridge_chain_ids=bridge_chains,
        same_bridge_targets=bridge_target,
    )
    control_bridge_target = world.atoms[world.first[bridge_chains], 2] + world.eo
    arrays["same_bridge_boundary_bridge_targets"] = control_bridge_target
    bridge_outcomes, bridge_captures = {}, {}
    bridge_modes = [
        m for m in modes if m.scope is None or m.scope == ("round1" if ex.repeats > 1 else "all")
    ]
    for mode in bridge_modes:
        bridge_outcomes[mode.name] = {}
        bridge_captures[mode.name] = {}
        for which, field in (("S", "units"), ("R", "control_units")):
            capture = {}
            answer, margins = mode_predict(
                ex,
                world.chain_tokens(bridge_chains),
                bridge_target,
                mode,
                _units_for(records, bridge_owner, field),
                batch=batch,
                bridge_target=control_bridge_target,
                bridge_capture=capture,
            )
            bridge_outcomes[mode.name][which] = answer, margins
            bridge_captures[mode.name][which] = capture
            arrays[f"same_bridge_{mode.name}_{which}_predictions"] = answer
            arrays[f"same_bridge_{mode.name}_{which}_margins"] = margins
            for key, values in capture.items():
                arrays[f"same_bridge_{mode.name}_{which}_bridge_{key}"] = values
    bridge, bridge_facts = _aggregate(
        records, bridge_owner, bridge_target, bridge_outcomes, bootstrap, population
    )
    for label, measures in (("primary", fact_measures), ("same_bridge", bridge_facts)):
        for name, values in measures.items():
            for measure, vector in values.items():
                arrays[f"fact_{label}_{name}_{measure}"] = vector
    arrays.update(
        fact_ids=sample["facts"],
        fact_kinds=sample["kinds"],
        fact_primary_population=population,
        fact_selective_saved=np.asarray([r["selective"] for r in records]),
        fact_k_star=np.asarray([r["k_star"] or -1 for r in records], dtype=np.int64),
        fact_units=pad_units([r["units"] for r in records], max(len(r["units"]) for r in records)),
        fact_control_units=pad_units(
            [r["control_units"] for r in records], max(len(r["control_units"]) for r in records)
        ),
    )
    summary = {
        "batch": config["batch"],
        "run": original["run"],
        "step": original["step"],
        "post_hoc": True,
        "smoke": smoke,
        "optimizer_updates": 0,
        "coverage": coverage,
        "reproduction": reproduction,
        "causal_prefix_check": causal_check,
        "primary": primary,
        "same_bridge": bridge,
        "boundary_readout": {
            "primary": _boundary_summary(
                records, owners, bridge_targets, primary_captures, population
            ),
            "same_bridge": _boundary_summary(
                records, bridge_owner, control_bridge_target, bridge_captures, population
            ),
            "definition": (
                "Position1 residual after executions0-3, with untuned final LN_f+head; "
                "before/after any restoration. Descriptive readout, distinct from final "
                "atomic fact retention."
            ),
        },
        "modes": [m.name for m in modes],
        "bootstrap": bootstrap,
        "aggregation": "Equal weight per frozen fact; queries are repeated measurements",
        "selection": "Saved primary_design units/control_units; no attribution or reselection",
        "same_bridge_selection": (
            "First up to4 of each saved same_bridge list; eligible when baseline chain answer "
            "and first atomic fact are correct; no replacement"
        ),
        "source": {
            "directory": str(source_dir),
            "z1_sha256": sha256(source_dir / "z1.json"),
            "hashes_sha256": sha256(source_dir / "hashes.json"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "world_sha256": provenance["world_sha256"],
            "panels_sha256": provenance["panels_sha256"],
            "fact_sample_sha256": sample["sha256"],
            "saved_units_and_controls_sha256": _json_hash(
                [[r["atom"], r["units"], r["control_units"], r["selective"]] for r in records]
            ),
        },
        "limits": [
            "Post-hoc mechanism diagnostic; no new binary support/against threshold",
            "Same original world/initialization; checkpoints are not independent worlds",
            "Prefix positions 0/1 contain no r2 under causal attention",
            "Last position 2 has r2 and may contain answer information",
            "Restoring the final prefix cannot undo earlier information read by position 2",
            "Small original eligible populations remain descriptive",
        ],
    }
    return summary, arrays


def compare_audit(previous_summary, previous_arrays, summary, arrays):
    """Independent process: exact predictions/units/coverage, tolerant fp32 margins."""
    if set(previous_arrays) != set(arrays):
        raise ValueError("Audit NPZ field set differs")
    for key, values in arrays.items():
        old = previous_arrays[key]
        if old.shape != values.shape:
            raise ValueError(f"Audit shape differs: {key}")
        if "margin" in key:
            equal = np.allclose(old, values, rtol=2e-5, atol=5e-4, equal_nan=True)
        elif np.issubdtype(values.dtype, np.floating):
            equal = np.array_equal(old, values, equal_nan=True)
        else:
            equal = np.array_equal(old, values)
        if not equal:
            raise ValueError(f"Audit output differs: {key}")
    for key in ("coverage", "reproduction", "modes", "bootstrap", "source"):
        if previous_summary[key] != summary[key]:
            raise ValueError(f"Audit summary differs: {key}")
    return {"passed": True, "arrays_checked": len(arrays), "predictions_exact": True}
