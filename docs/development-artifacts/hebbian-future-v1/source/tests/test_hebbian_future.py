import numpy as np

from llm_memory_editability.hebbian_future import (
    CONFIG,
    chain_world,
    read,
    ridge_predict,
    training_arrays,
)


def test_balanced_worlds_and_heldout_composition():
    cfg = read(CONFIG)["chains"]
    for world in cfg["worlds"]:
        previous = None
        for rho in cfg["correlations"]:
            w = chain_world(world, rho, cfg)
            for split in (w["train_people"], ~w["train_people"]):
                assert np.mean(w["home_y"][split] == w["composite_y"][split]) == rho
                assert len(set(np.bincount(w["home"][split], minlength=4))) == 1
                assert (
                    len(set(np.bincount(w["composite_y"][split] - w["city_start"], minlength=4)))
                    == 1
                )
            assert np.all(
                w["home_y"][w["common_conflict"]] != w["composite_y"][w["common_conflict"]]
            )
            x, y = training_arrays(w)
            assert len(x) == 252 and len(y) == 252
            train_composites = {tuple(row) for row in x[x[:, 2] == 6]}
            assert not train_composites.intersection(
                map(tuple, w["composite_x"][~w["train_people"]])
            )
            if previous is not None:
                for key in (
                    "member_x",
                    "member_y",
                    "root_x",
                    "root_y",
                    "composite_x",
                    "composite_y",
                    "train_people",
                    "common_conflict",
                ):
                    np.testing.assert_array_equal(w[key], previous[key])
                np.testing.assert_array_equal(x, training_arrays(previous)[0])
            previous = w


def test_ridge_uses_only_development_statistics():
    train = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    test = np.array([[1.5, 1.0], [100.0, 1.0]])
    together = ridge_predict(train, labels, test)
    separate = np.r_[ridge_predict(train, labels, test[:1]), ridge_predict(train, labels, test[1:])]
    np.testing.assert_allclose(together, separate)


def test_two_step_uses_generated_membership():
    import importlib.util

    import torch

    spec = importlib.util.spec_from_file_location("future_runner", "scripts/run_hebbian_future.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    w = chain_world(300, 0.5, read(CONFIG)["chains"])

    class WrongMembership(torch.nn.Module):
        def forward(self, tokens):
            logits = torch.zeros(len(tokens), tokens.shape[1], w["vocab"])
            if tokens.shape[1] == 5:
                pred = torch.full((len(tokens),), 7)
            else:
                pred = torch.where(
                    tokens[:, 2] == 3,
                    w["group_start"] + 1,
                    w["city_start"] + (tokens[:, 1] - w["group_start"]) % 4,
                )
            logits[torch.arange(len(tokens)), -1, pred] = 10
            return logits

    result = runner.evaluate_chain(WrongMembership(), w, "cpu")
    actual = np.array(result["rows"]["two_step"]["prediction"])
    assert np.all(actual == w["city_start"] + 1)
    assert not np.array_equal(actual, w["composite_y"])
