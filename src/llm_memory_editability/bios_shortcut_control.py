"""P3: alter shortcut reliability while preserving marginal answer counts.

The archived crossover generator and trainer are imported, never rewritten.
The two chains are manipulated simultaneously; QA identities remain unchanged.
"""

import argparse
import fcntl
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .bios_cross import CONDITIONS, audit, documents, make_cross_world, qa_schedule
from .bios_cross_train import learning, source_hashes
from .bios_data import array_hash, rng_for, write_json
from .bios_organization_train import atomic_numpy_save


def high_exception_world(old):
    """Raise both chains from two to sixteen exceptions per 32-member group."""
    audit(old)
    answers, exceptions = old.answers.copy(), old.exceptions.copy()
    selections = np.empty((2, 64, 14), dtype=np.int64)
    donors = np.empty((2, 14, 64), dtype=np.int64)
    for chain in range(2):
        rng = rng_for(old.seed, 930, chain)
        for group in range(64):
            ordinary = np.flatnonzero((old.memberships[chain] == group) & ~old.exceptions[chain])
            people = []
            for query_pool in (old.train_ids[chain], old.heldout_ids[chain]):
                eligible = ordinary[np.isin(old.derived_ids[chain, ordinary], query_pool)]
                people.extend(rng.permutation(eligible)[:7])
            selections[chain, group] = people
        # Every lane contains one ordinary member per group. A city-deranged
        # bijection preserves the histogram exactly, separately in each QA split.
        for lane in range(14):
            for _ in range(10000):
                donor = rng.permutation(64)
                if np.all(old.defaults[chain, donor] != old.defaults[chain]):
                    break
            else:
                raise RuntimeError("Could not construct a city-deranged permutation")
            donors[chain, lane] = donor
            selected = selections[chain, :, lane]
            answers[old.actual_ids[chain, selected]] = old.city_tokens[old.defaults[chain, donor]]
            exceptions[chain, selected] = True
    world = replace(old, answers=answers, exceptions=exceptions)
    record = audit_high_exception(old, world)
    record["selection_sha256"] = array_hash(selections)
    record["donors_sha256"] = array_hash(donors)
    return world, record, selections, donors


def audit_high_exception(old, new):
    """Fail if the manipulation changes anything outside its declared contract."""
    allowed = old.actual_ids.ravel()
    unchanged = np.setdiff1d(np.arange(len(old.answers)), allowed)
    np.testing.assert_array_equal(old.answers[unchanged], new.answers[unchanged])
    for field in (
        "prompts",
        "lengths",
        "relation",
        "person",
        "memberships",
        "defaults",
        "membership_ids",
        "root_ids",
        "actual_ids",
        "derived_ids",
        "train_ids",
        "heldout_ids",
        "city_tokens",
    ):
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
    if old.token_labels != new.token_labels:
        raise ValueError("Vocabulary changed")
    records = []
    for chain in range(2):
        ids = old.actual_ids[chain]
        np.testing.assert_array_equal(np.sort(old.answers[ids]), np.sort(new.answers[ids]))
        truth_exception = (
            new.answers[ids] != new.city_tokens[new.defaults[chain, new.memberships[chain]]]
        )
        np.testing.assert_array_equal(truth_exception, new.exceptions[chain])
        if np.any(old.exceptions[chain] & ~new.exceptions[chain]):
            raise ValueError("An original exception was removed")
        for group in range(64):
            members = new.memberships[chain] == group
            if int(new.exceptions[chain, members].sum()) != 16:
                raise ValueError("Expected sixteen exceptions per group")
            for pool in (new.train_ids[chain], new.heldout_ids[chain]):
                people = np.flatnonzero(members & np.isin(new.derived_ids[chain], pool))
                if len(people) != 16 or new.exceptions[chain, people].sum() != 8:
                    raise ValueError("QA split no longer has eight exceptions per group")
        for pool in (new.train_ids[chain], new.heldout_ids[chain]):
            people = np.flatnonzero(np.isin(new.derived_ids[chain], pool))
            np.testing.assert_array_equal(
                np.sort(old.answers[ids[people]]), np.sort(new.answers[ids[people]])
            )
        records.append(
            {
                "chain": chain,
                "old_exceptions": int(old.exceptions[chain].sum()),
                "new_exceptions": int(new.exceptions[chain].sum()),
                "changed_actual_facts": int((old.answers[ids] != new.answers[ids]).sum()),
                "answer_histogram_preserved": True,
                "split_histograms_preserved": True,
            }
        )
    for condition in CONDITIONS:
        np.testing.assert_array_equal(documents(old, condition), documents(new, condition))
    np.testing.assert_array_equal(qa_schedule(old, 1280), qa_schedule(new, 1280))
    return {
        "passed": True,
        "world": old.seed,
        "old_truth_sha256": array_hash(old.answers),
        "new_truth_sha256": array_hash(new.answers),
        "prompts_sha256": array_hash(new.prompts),
        "chains": records,
        "changed_total": int((old.answers != new.answers).sum()),
        "document_identities_and_QA_split_preserved": True,
    }


def shortcut_sources():
    diagnostic = Path(__file__).with_name("bios_path_diagnostics.py")
    return {
        **source_hashes(),
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic.name: hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
    }


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = json.loads(Path(args.config).read_text())
        if study["width"] != 256 or study["steps"] != 15360:
            raise ValueError("P3 freezes width 256 and 15360 steps")
        old = make_cross_world(args.world)
        world, manipulation, selected, donors = high_exception_world(old)
        contract = {
            "protocol": "v2.8-p3-shortcut-reliability",
            "study": study,
            "world": args.world,
            "seed": args.seed,
            "condition": args.condition,
            "sources": shortcut_sources(),
            "manipulation": manipulation,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            },
        }
        contract_path = out / "launch-contract.json"
        if contract_path.exists():
            frozen = json.loads(contract_path.read_text())
            for key in (
                "protocol",
                "study",
                "world",
                "seed",
                "condition",
                "sources",
                "manipulation",
            ):
                if frozen[key] != contract[key]:
                    raise ValueError(f"Frozen shortcut contract changed: {key}")
        else:
            write_json(contract_path, contract)
            atomic_numpy_save(
                out / "manipulation.npz",
                selections=selected,
                donors=donors,
                old_answers=old.answers,
                answers=world.answers,
                exceptions=world.exceptions,
                train_ids=world.train_ids,
                heldout_ids=world.heldout_ids,
            )
        if (out / "complete.json").exists():
            return
        # Include adapter provenance in the archived trainer's own run identity.
        study = {**study, "shortcut_contract": contract["sources"]}
        torch.set_num_threads(args.threads)
        try:
            from .bios_path_diagnostics import autonomous_two_step

            device = torch.device(args.device)
            model, _ = learning(args, world, out, study, device)
            with np.load(out / f"predictions-{study['steps']}.npz") as saved:
                direct = {k: saved[k] for k in ("prediction", "ended", "correct")}
            for chain, name in enumerate(("company", "project")):
                arrays = autonomous_two_step(model, world, chain, device)
                ids = world.derived_ids[chain]
                arrays.update({f"direct_{key}": value[ids] for key, value in direct.items()})
                atomic_numpy_save(out / f"two-step-{name}.npz", **arrays)
            write_json(
                out / "complete.json",
                {
                    "status": "complete",
                    "phase": "P3-learning-only",
                    "learning_steps": study["steps"],
                    "edit_cases": 0,
                    "finished": time.time(),
                },
            )
        except Exception as error:
            write_json(
                out / "failure.json",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "time": time.time(),
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--config", default="configs/bios-shortcut-control-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
