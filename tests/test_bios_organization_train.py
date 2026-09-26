"""Scientific objective, accounting, and interrupted-run continuation controls."""

import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_memory_editability import bios_organization_train as training  # noqa: E402
from llm_memory_editability.bios_data import (  # noqa: E402
    N_BASE,
    N_PEOPLE,
    N_QUERIES,
    REVISION,
    SOURCE_FOLDER,
    make_world,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig, matmul_flops  # noqa: E402
from llm_memory_editability.bios_organization import apply_epoch, make_documents  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("organization-training-source")
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
        path = folder / (name + ".txt")
        path.write_text("\n".join(f"{name}_{i}" for i in range(count)))
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(json.dumps({"revision": REVISION, "files": files}))
    return make_world(9, root)


def test_two_context_gradient_is_joint_supervised_token_objective():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = CausalLM(ModelConfig(vocab_size=31, width=16, layers=2, heads=2))
    reference = copy.deepcopy(model)
    documents = {
        "tokens": torch.randint(0, 31, (16, 42)),
        "positions": torch.arange(14)[None, :].expand(16, -1) * 3,
        "labels": torch.randint(0, 31, (16, 14)),
    }
    isolated = {
        "tokens": torch.randint(0, 31, (28, 6)),
        "positions": torch.tensor([3, 4])[None, :].expand(28, -1),
        "labels": torch.randint(0, 31, (28, 2)),
    }
    dl, ql = training.two_context_backward(
        model, documents, isolated, torch.arange(28), torch.device("cpu")
    )
    logits = torch.cat(
        [
            reference(documents["tokens"], documents["positions"]).flatten(0, 1),
            reference(isolated["tokens"], isolated["positions"]).flatten(0, 1),
        ]
    )
    labels = torch.cat([documents["labels"].flatten(), isolated["labels"].flatten()])
    joint = torch.nn.functional.cross_entropy(logits, labels)
    joint.backward()
    torch.testing.assert_close(0.8 * dl + 0.2 * ql, joint)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=1e-5)


def test_document_rows_are_independent_and_future_tokens_are_invisible():
    torch.manual_seed(21)
    model = CausalLM(ModelConfig(vocab_size=37, width=16, layers=2, heads=2))
    tokens = torch.randint(0, 37, (2, 42))
    original = model(tokens)
    changed = tokens.clone()
    changed[1] = (changed[1] + 7) % 37
    torch.testing.assert_close(model(changed)[0], original[0], atol=0, rtol=0)
    changed = tokens.clone()
    changed[:, 25:] = (changed[:, 25:] + 3) % 37
    torch.testing.assert_close(model(changed)[:, :25], original[:, :25], atol=0, rtol=0)


def test_derived_stream_covers_complete_cycles_without_replacement():
    first = training.derived_schedule(4, 1024)
    np.testing.assert_array_equal(first, training.derived_schedule(4, 1024))
    assert not np.array_equal(first, training.derived_schedule(5, 1024))
    assert first.shape == (1024, 28)
    for cycle in first.ravel().reshape(-1, N_PEOPLE):
        np.testing.assert_array_equal(np.sort(cycle), np.arange(N_BASE, N_QUERIES))
    assert len(training.checkpoint_steps(training.DEFAULT_STEPS)) == 7
    assert all(step % (128 * 7) == 0 for step in training.CHECKPOINTS)
    assert training.checkpoint_steps(2) == [0, 2]


def test_checkpoint_exposure_matches_counts_slots_and_warmup_rates(world):
    arrays = []
    for condition in "ABC":
        documents = make_documents(world, condition, organization_seed=0)
        counts = np.zeros(N_BASE, dtype=np.int64)
        slots = np.zeros((N_BASE, 7), dtype=np.int64)
        weighted = np.zeros(N_BASE, dtype=np.float64)
        for epoch in range(7):
            ids = apply_epoch(documents, world.seed, epoch)
            lr = training.learning_rate(epoch * 128, 1e-4)
            assert training.learning_rate(epoch * 128 + 127, 1e-4) == lr
            np.add.at(counts, ids, 1)
            np.add.at(slots, (ids, np.arange(7)[None, :]), 1)
            np.add.at(weighted, ids, lr)
        arrays.append((counts, slots, weighted))
    for other in arrays[1:]:
        for expected, actual in zip(arrays[0], other, strict=True):
            np.testing.assert_array_equal(expected, actual)
    assert np.all(arrays[0][0][world.relation[:N_BASE] == 1] == 32 * 7)
    assert np.all(arrays[0][0][world.relation[:N_BASE] != 1] == 7)
    assert training.learning_rate(2047, 1e-4) == 1e-4
    assert training.learning_rate(2048, 1e-4) == 1e-4


def test_flop_estimate_uses_all_fourteen_document_output_positions():
    config = ModelConfig(vocab_size=5000, width=16, layers=2, heads=2)
    estimate = training.update_flops(config)
    assert estimate == (
        matmul_flops(config, 16, sequence=42, output_positions=14)
        + matmul_flops(config, 28, sequence=6, output_positions=2)
    )
    wrong = matmul_flops(config, 16, sequence=42, output_positions=2)
    correct = matmul_flops(config, 16, sequence=42, output_positions=14)
    assert correct - wrong == 3 * 2 * 16 * 12 * 16 * 5000


def test_resume_restores_optimizer_and_preserves_config(world, tmp_path, monkeypatch):
    world_path = tmp_path / "world"
    world.save(world_path)
    monkeypatch.setattr(training, "CHECKPOINTS", (0, 1, 2))

    def evaluation(model, data, world, batch_size):
        model.eval()
        return {"base_accuracy": 0.0, "derived_accuracy": 0.0}, {
            "prediction": np.zeros(N_QUERIES, dtype=np.int64),
            "ended": np.zeros(N_QUERIES, dtype=bool),
            "correct": np.zeros(N_QUERIES, dtype=bool),
            "value_nll": np.zeros(N_QUERIES),
        }

    monkeypatch.setattr(training, "evaluate", evaluation)

    def arguments(name, *extra):
        return training.parser().parse_args(
            [
                "--world",
                str(world_path),
                "--output",
                str(tmp_path / name),
                "--condition",
                "A",
                "--steps",
                "2",
                "--width",
                "8",
                "--layers",
                "1",
                "--heads",
                "2",
                "--device",
                "cpu",
                "--threads",
                "1",
                *extra,
            ]
        )

    training.train(arguments("baseline"))
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupted_step(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("simulated interruption")
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupted_step)
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        training.train(arguments("resumed"))
    saved_config = (tmp_path / "resumed" / "config.json").read_bytes()
    assert json.loads((tmp_path / "resumed" / "failure.json").read_text())["step"] == 1
    monkeypatch.setattr(torch.optim.AdamW, "step", original_step)
    training.train(arguments("resumed", "--resume"))
    assert (tmp_path / "resumed" / "config.json").read_bytes() == saved_config
    baseline = torch.load(tmp_path / "baseline" / "model-2.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed" / "model-2.pt", weights_only=True)
    assert training.state_hash(baseline["model"]) == training.state_hash(resumed["model"])
    with np.load(tmp_path / "baseline" / "predictions-2.npz") as a:
        with np.load(tmp_path / "resumed" / "predictions-2.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError, match="already contains"):
        training.train(arguments("resumed"))
    with pytest.raises(ValueError, match="mismatch.*condition"):
        changed = arguments("resumed", "--resume")
        changed.condition = "B"
        training.train(changed)
    with pytest.raises(ValueError, match="mismatch.*lr"):
        changed = arguments("resumed", "--resume")
        changed.lr = 2e-4
        training.train(changed)
