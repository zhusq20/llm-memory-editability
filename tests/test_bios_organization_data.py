"""Dataset invariants whose failure would confound the organization comparison."""

import hashlib
import json

import numpy as np
import pytest

from llm_memory_editability.bios_data import (
    ANS,
    EOS,
    N_BASE,
    N_PEOPLE,
    REVISION,
    SOURCE_FOLDER,
    make_world,
)
from llm_memory_editability.bios_organization import (
    CONDITIONS,
    N_PRESENTATIONS,
    PERSONAL_RELATIONS,
    apply_epoch,
    audit_organizations,
    document_epoch,
    english_fact,
    make_documents,
    presentation_ids,
    render_documents,
)


@pytest.fixture(scope="module")
def organization_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("organization-source")
    folder = root / SOURCE_FOLDER / "fields"
    folder.mkdir(parents=True)
    files = []
    for name, count in {
        "first_name": 400,
        "middle_name": 400,
        "last_name": 1000,
        "city": 200,
        "company": 263,
        "university": 300,
        "field": 100,
    }.items():
        path = folder / f"{name}.txt"
        path.write_text("\n".join(f"{name}_{i}" for i in range(count)))
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(json.dumps({"revision": REVISION, "files": files}))
    return make_world(19, root)


@pytest.mark.parametrize("organization_seed", [0, 7])
def test_same_truth_and_exposure_but_different_grouping(organization_world, organization_seed):
    world = organization_world
    before = world.answers.copy()
    documents = {c: make_documents(world, c, organization_seed) for c in CONDITIONS}
    audit = audit_organizations(world, documents)
    np.testing.assert_array_equal(world.answers, before)
    assert len({row["exposure_sha256"] for row in audit["conditions"].values()}) == 1
    assert len({row["answer_histogram_sha256"] for row in audit["conditions"].values()}) == 1
    for c, docs in documents.items():
        counts = np.bincount(docs.ravel(), minlength=N_BASE)
        np.testing.assert_array_equal(counts[world.relation[:N_BASE] == 1], 32)
        np.testing.assert_array_equal(counts[world.relation[:N_BASE] != 1], 1)
        np.testing.assert_array_equal(
            np.sort(presentation_ids(world, docs).ravel()), np.arange(N_PRESENTATIONS)
        )
        people = world.person[docs[:, PERSONAL_RELATIONS]]
        if c in ("A", "B"):
            assert (people == people[:, :1]).all()
        else:
            assert np.all(np.diff(np.sort(people, axis=1), axis=1) != 0)
            assert np.all(np.diff(np.sort(world.company[docs], axis=1), axis=1) != 0)
    assert audit["conditions"]["A"]["default_actual_same_answer_rate"] == 0.9375
    # Prevent a same-city exclusion from making B/C artificially anti-correlated.
    np.testing.assert_array_equal(documents["B"][:, 1:3], documents["C"][:, 1:3])
    assert audit["conditions"]["B"]["default_actual_same_answer_count"] > 0


def test_common_text_presentations_are_only_regrouped(organization_world):
    world = organization_world
    docs = {c: make_documents(world, c) for c in CONDITIONS}
    master = np.empty(N_PRESENTATIONS, dtype=np.int64)
    master[presentation_ids(world, docs["A"]).ravel()] = docs["A"].ravel()
    sentences = [english_fact(world, int(fact), pid % 3) for pid, fact in enumerate(master)]
    expected = sorted(sentences)
    for matrix in docs.values():
        ids = presentation_ids(world, matrix)
        np.testing.assert_array_equal(master[ids], matrix)
        assert sorted(sentences[pid] for pid in ids.ravel()) == expected


def test_shared_epochs_balance_all_fact_positions(organization_world):
    world = organization_world
    counts = []
    for condition in CONDITIONS:
        canonical = make_documents(world, condition)
        slots = np.zeros((N_BASE, 7), dtype=np.int64)
        for epoch in range(7):
            current = apply_epoch(canonical, world.seed, epoch)
            np.testing.assert_array_equal(current, document_epoch(world, condition, epoch))
            # Rotation is the only within-document ordering change.
            np.testing.assert_array_equal(
                world.relation[current], np.tile(np.roll(np.arange(7), epoch), (N_PEOPLE, 1))
            )
            for slot in range(7):
                np.add.at(slots[:, slot], current[:, slot], 1)
        counts.append(slots)
    np.testing.assert_array_equal(counts[0], counts[1])
    np.testing.assert_array_equal(counts[0], counts[2])
    np.testing.assert_array_equal(
        counts[0], np.tile(np.where(world.relation[:N_BASE] == 1, 32, 1)[:, None], (1, 7))
    )
    b, c = (apply_epoch(make_documents(world, cond), world.seed, 8) for cond in ("B", "C"))
    # Shared row permutation leaves the matching default/actual pair aligned.
    np.testing.assert_array_equal(b[:, [2, 3]], c[:, [2, 3]])


def test_rendering_supervises_answer_and_eos_without_cross_document_padding(organization_world):
    world = organization_world
    ids = document_epoch(world, "C", 3)[:5]
    data = render_documents(world, ids)
    assert data["tokens"].shape == (5, 42)
    assert data["positions"].shape == data["labels"].shape == (5, 14)
    facts = data["tokens"].reshape(5, 7, 6)
    np.testing.assert_array_equal(facts[:, :, :4], world.prompts[ids, :4])
    np.testing.assert_array_equal(facts[:, :, 4], world.answers[ids])
    assert (facts[:, :, 3] == ANS).all() and (facts[:, :, 5] == EOS).all()
    row = np.arange(5)[:, None]
    np.testing.assert_array_equal(data["tokens"][row, data["positions"] + 1], data["labels"])
    assert np.all(data["positions"] % 6 == np.tile([3, 4], 7))
    with pytest.raises(ValueError):
        render_documents(world, np.full((1, 7), N_BASE))


def test_audit_rejects_missing_fact_and_accidental_links(organization_world):
    world = organization_world
    docs = {c: make_documents(world, c) for c in CONDITIONS}
    duplicate = {c: matrix.copy() for c, matrix in docs.items()}
    duplicate["C"][0, 2] = duplicate["C"][1, 2]
    with pytest.raises(ValueError, match="presentation counts"):
        audit_organizations(world, duplicate)
    broken = {c: matrix.copy() for c, matrix in docs.items()}
    broken["B"][:, 1] = docs["A"][:, 1]
    with pytest.raises(ValueError, match="accidentally matches"):
        audit_organizations(world, broken)
    with pytest.raises(ValueError, match="Unknown"):
        make_documents(world, "SA")
