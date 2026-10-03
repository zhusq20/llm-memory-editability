"""Confirmation filtering must isolate names without withholding required facts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.realworld_composition_data import sha256, write_json
from llm_memory_editability.realworld_confirmation import (
    audit_confirmation,
    finalize_confirmation,
    isolate_confirmation,
)
from llm_memory_editability.realworld_confirmation_analysis import (
    analyze_confirmation,
    threshold_intervals,
)


def fixtures():
    canonical = {e: e + " name" for e in ["D", "B", "H", "J", "I", "K", "F", "G", "T"]}
    aliases = {"F": [canonical["D"]]}

    def dataset(chains):
        atoms, rows = {}, []
        for identifier, head, bridge in chains:
            edges = [[head, "director", bridge], [bridge, "place of birth", "T"]]
            atom_ids = []
            for subject, relation, tail in edges:
                key = f"{subject}-{relation}-{tail}"
                atom_ids.append(key)
                atoms[key] = dict(
                    id=key,
                    edge=[subject, relation, tail],
                    question=f"{relation} {canonical[subject]}?",
                    answer=canonical[tail],
                    aliases=[],
                    encoded=dict(prefix=[1], target=[2, 50256], input=[1, 2]),
                )
            rows.append(
                dict(
                    id=identifier,
                    edges=edges,
                    surfaces=[
                        [canonical[head], "director", canonical[bridge]],
                        [canonical[bridge], "place of birth", canonical["T"]],
                    ],
                    atom_ids=atom_ids,
                    question=f"Birthplace of the director of {canonical[head]}?",
                    answer=canonical["T"],
                    aliases=[],
                    known_prior_case=False,
                    structural_errors=[],
                    encoded=dict(prefix=[3], target=[2, 50256], input=[3, 2]),
                )
            )
        return dict(
            atoms=list(atoms.values()),
            train_compositions=rows[:1],
            evaluation_compositions=rows[1:],
            panels={},
        )

    dev = dataset([("dev", "D", "B")])
    candidate = dataset([("train", "H", "J"), ("held", "I", "K"), ("collision", "F", "G")])
    return candidate, dev, canonical, aliases


def test_name_collision_filter_preserves_shared_tails_and_rebuilds_exposure_roles():
    candidate, dev, canonical, aliases = fixtures()
    original = copy.deepcopy(candidate)
    data, report = isolate_confirmation(candidate, dev, canonical, aliases)
    assert candidate == original
    assert report["excluded_case_ids"]["evaluation_compositions"] == ["collision"]
    assert [r["id"] for r in data["evaluation_compositions"]] == ["held"]
    assert data["evaluation_compositions"][0]["role"] == "OO"
    assert len(data["atoms"]) == 4
    assert all(r["answer"] == canonical["T"] for r in data["evaluation_compositions"])
    assert audit_confirmation(data, dev, canonical, aliases)["necessary_atomic_coverage"] == 1.0


def test_source_surface_collision_is_excluded_even_without_alias_collision():
    candidate, dev, canonical, aliases = fixtures()
    candidate["evaluation_compositions"][1]["surfaces"][0][2] = canonical["B"]
    data, report = isolate_confirmation(candidate, dev, canonical, {})
    assert "G" in report["excluded_entities"]
    assert all(r["id"] != "collision" for r in data["evaluation_compositions"])


def test_unique_canonical_answer_eligibility_does_not_change_primary_golds():
    candidate, dev, canonical, aliases = fixtures()
    canonical["other-tail"] = canonical["T"]
    data, _ = isolate_confirmation(candidate, dev, canonical, aliases)
    assert not data["evaluation_compositions"][0]["canonical_answer_globally_unambiguous"]
    assert data["evaluation_compositions"][0]["answer"] == canonical["T"]
    assert (
        audit_confirmation(data, dev, canonical, aliases)["canonical_unambiguous_evaluation_count"]
        == 0
    )


@pytest.mark.parametrize(
    "corruption", ["missing_atom", "wrong_role", "leaked_question", "broken_eos"]
)
def test_independent_data_audit_rejects_contract_violations(corruption):
    candidate, dev, canonical, aliases = fixtures()
    data, _ = isolate_confirmation(candidate, dev, canonical, aliases)
    if corruption == "missing_atom":
        data["atoms"].pop()
    elif corruption == "wrong_role":
        data["evaluation_compositions"][0]["role"] = "II"
    elif corruption == "leaked_question":
        data["train_compositions"][0]["question"] = data["evaluation_compositions"][0]["question"]
    else:
        data["atoms"][0]["encoded"]["target"][-1] = 0
    with pytest.raises((AssertionError, KeyError)):
        audit_confirmation(data, dev, canonical, aliases)


def test_threshold_brackets_retain_regression_and_sparse_node_uncertainty():
    history = [
        dict(step=s, metrics={"test_all": {"alias_em": a}})
        for s, a in [(0, 0.6), (10, 0.4), (100, 0.7)]
    ]
    result = threshold_intervals(history, "test_all", 0.5)
    assert result["first_observed_interval"] == [None, 0]
    assert result["sustained_observed_interval"] == [10, 100]
    assert threshold_intervals(history, "test_all", 0.9)["first_observed_interval"] is None


def report_fixture(tmp_path):
    data = {
        "atoms": [dict(id="a", answer="Alice", aliases=[]), dict(id="b", answer="Bob", aliases=[])],
        "train_compositions": [dict(id="train", answer="Bob", aliases=[])],
        "evaluation_compositions": [
            dict(
                id=role,
                role=role,
                answer="Alice",
                aliases=[],
                atom_ids=["a", "b"],
                group="component",
                edges=[["H", "director", "B"], ["B", "birthplace", "T"]],
            )
            for role in ["II", "IO", "OI", "OO"]
        ],
    }
    path = tmp_path / "data.json"
    write_json(path, data)
    config = dict(
        results_root=str(tmp_path / "results"),
        artifact_root=str(tmp_path / "art"),
        phase="confirmation",
        data_file=str(path),
        official_scorer=str(
            Path(
                "docs/development-artifacts/realworld-composition-v1/reference/2wikimultihop_evaluate_v1.1.py"
            ).resolve()
        ),
        runs=[],
        evaluation_nodes=[0, 1],
    )
    for seed in range(3):
        for architecture in ["standard8", "loop4x2", "standard4"]:
            spec = dict(
                name=f"{architecture}-{seed}",
                architecture=architecture,
                initialization=seed,
                steps=1,
            )
            config["runs"].append(spec)
            out = Path(config["results_root"]) / "confirmation" / spec["name"]
            out.mkdir(parents=True)
            (out / "latest.pt").write_text("test weights")
            write_json(
                out / "complete.json",
                dict(
                    step=1, independently_reloaded=True, checkpoint_sha256=sha256(out / "latest.pt")
                ),
            )
            write_json(out / "run.json", dict(common_four_block_initial_sha256=str(seed)))
            raw = {}
            metrics = {}
            for group, records in [
                ("atomic", data["atoms"]),
                ("train_composition", data["train_compositions"]),
                ("test_all", data["evaluation_compositions"]),
            ]:
                raw[group] = [
                    dict(
                        id=r["id"],
                        prediction=r["answer"]
                        if architecture != "loop4x2" or r["id"] != "OO"
                        else "Incorrect",
                        alias_em=float(architecture != "loop4x2" or r["id"] != "OO"),
                    )
                    for r in records
                ]
                metrics[group] = dict(
                    alias_em=sum(r["alias_em"] for r in raw[group]) / len(records)
                )
            for role in ["II", "IO", "OI", "OO"]:
                metrics["test_" + role.lower()] = dict(
                    alias_em=next(p["alias_em"] for p in raw["test_all"] if p["id"] == role)
                )
            metrics["autonomous"] = dict(alias_em=1.0)
            write_json(out / "endpoint-predictions.json", raw)
            write_json(out / "endpoint.json", metrics)
            write_json(out / "learning.json", [dict(step=s, metrics=metrics) for s in [0, 1]])
            for step in [0, 1]:
                np.savez_compressed(
                    out / f"exposure-{step:07d}.npz", counts=np.ones(3, dtype=np.int64) * step
                )
                write_json(out / f"predictions-{step:07d}.json", raw)
    return config


def test_nine_run_finalizer_retains_negative_paired_result_and_knowledge_coverage(tmp_path):
    config = report_fixture(tmp_path)
    result = finalize_confirmation(config)
    assert result["runs"] == 9 and result["independent_dataset_splits"] == 1
    assert result["architecture_scores"]["loop4x2"]["test_oo"]["mean"] == 0
    oo = [r for r in result["paired_comparisons"] if r["pool"] == "OO"]
    assert all(
        r["paired_difference"] == -1 and r["both_models_necessary_atoms_correct_coverage"] == 1
        for r in oo
    )
    assert len(analyze_confirmation(config)["paired_knowledge_trajectories"]) == 24


def test_finalizer_rejects_incorrect_logged_gold_score_and_unpaired_exposure(tmp_path):
    config = report_fixture(tmp_path)
    out = Path(config["results_root"]) / "confirmation" / config["runs"][0]["name"]
    path = out / "endpoint-predictions.json"
    original = json.loads(path.read_text())
    corrupt = copy.deepcopy(original)
    corrupt["test_all"][0]["prediction"] = "Wrong entity"
    write_json(path, corrupt)
    with pytest.raises(AssertionError):
        finalize_confirmation(config)
    write_json(path, original)
    np.savez_compressed(out / "exposure-0000001.npz", counts=np.zeros(3, dtype=np.int64))
    with pytest.raises(AssertionError):
        finalize_confirmation(config)
