"""Coordinate/intervention contracts; synthetic checks are not model evidence."""

import pytest
import torch

from llm_memory_editability.ica_coordinates import fit_coordinates


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def mixed_sources(n=12000, seed=31):
    generator = torch.Generator().manual_seed(seed)
    u = torch.rand(n, 3, generator=generator, dtype=torch.float64).clamp(1e-8, 1 - 1e-8)
    sources = torch.stack(
        (
            -torch.log(u[:, 0]) - 1,
            (u[:, 1] - 0.5).sign() * torch.log1p(-2 * (u[:, 1] - 0.5).abs()),
            2 * u[:, 2] - 1,
        ),
        dim=1,
    )
    mixing = torch.tensor([[3.0, 0.5, 0.4], [0.2, 0.6, -0.1], [0.6, 0.2, 1.2]], dtype=torch.float64)
    x = sources @ mixing.T + torch.tensor([10.0, -2.0, 4.0], dtype=torch.float64)
    return x, sources


@pytest.mark.parametrize("method", ["ica", "pca", "random"])
def test_full_reconstruction_whitening_and_duality(method):
    x, _ = mixed_sources(n=3000)
    original = x.clone()
    coordinate = fit_coordinates(x, method, seed=14)
    y = coordinate.encode(x)
    identity = torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(coordinate.decode(y), x, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(coordinate.read.T @ coordinate.write, identity, atol=2e-12, rtol=0)
    torch.testing.assert_close(y.T @ y / len(x), identity, atol=2e-12, rtol=0)
    torch.testing.assert_close(coordinate.mean, x.mean(0), atol=0, rtol=0)
    assert (coordinate.fit_skewness >= 0).all()
    assert coordinate.diagnostics["regularization"] is None
    torch.testing.assert_close(x, original, atol=0, rtol=0)


@pytest.mark.parametrize("fit_dtype", [torch.float32, torch.float64])
def test_ica_recovers_correlated_nongaussian_sources(fit_dtype):
    x, sources = mixed_sources()
    coordinate = fit_coordinates(x, seed=9, fit_dtype=fit_dtype)
    y = coordinate.encode(x)
    normalized_y = (y - y.mean(0)) / y.std(0, correction=0)
    normalized_sources = (sources - sources.mean(0)) / sources.std(0, correction=0)
    correlations = (normalized_y.T @ normalized_sources / len(x)).abs()
    matches = correlations.argmax(dim=1)
    assert matches.unique().numel() == 3
    assert correlations.max(dim=1).values.min() > 0.98
    assert coordinate.diagnostics["converged"]


def test_intervention_uses_write_dual_and_preserves_other_coordinates():
    x, _ = mixed_sources(n=3000)
    coordinate = fit_coordinates(x, seed=4)
    target = x[:7].clone()
    y = coordinate.encode(target)
    delta = torch.arange(1, 8, dtype=torch.float64)
    changed = coordinate.replace_coordinates(target, [1], (y[:, 1] + delta)[:, None])
    expected_coordinates = y.clone()
    expected_coordinates[:, 1] += delta
    torch.testing.assert_close(coordinate.encode(changed), expected_coordinates, atol=2e-12, rtol=0)
    torch.testing.assert_close(
        changed - target, delta[:, None] * coordinate.write[:, 1], atol=2e-12, rtol=0
    )
    wrong = target + delta[:, None] * coordinate.read[:, 1]
    assert (coordinate.encode(wrong) - expected_coordinates).abs().max() > 0.1
    assert not torch.allclose(changed.norm(dim=1), target.norm(dim=1))
    torch.testing.assert_close(target, x[:7], atol=0, rtol=0)
    assert coordinate.ablate(target.float(), [1]).dtype == torch.float32


def test_ablation_and_donor_patch_with_broadcasting():
    x, _ = mixed_sources(n=3000)
    coordinate = fit_coordinates(x, "random", seed=4)
    target, donor = x[:4], x[20]
    y = coordinate.encode(target)
    ablated = coordinate.encode(coordinate.ablate(target, [0, 2]))
    torch.testing.assert_close(
        ablated[:, [0, 2]], torch.zeros(4, 2, dtype=torch.float64), atol=2e-12, rtol=0
    )
    torch.testing.assert_close(ablated[:, 1], y[:, 1], atol=2e-12, rtol=0)
    patched = coordinate.encode(coordinate.patch(target, donor, [0, 2]))
    expected = coordinate.encode(donor)[[0, 2]].expand(4, 2)
    torch.testing.assert_close(patched[:, [0, 2]], expected, atol=2e-12, rtol=0)
    torch.testing.assert_close(patched[:, 1], y[:, 1], atol=2e-12, rtol=0)


def test_sparse_ranking_uses_write_contribution_and_restores_mean():
    generator = torch.Generator().manual_seed(11)
    fit = torch.randn(4000, 2, generator=generator, dtype=torch.float64) * torch.tensor([10.0, 1.0])
    fit += torch.tensor([30.0, -20.0])
    coordinate = fit_coordinates(fit, "pca")
    y = torch.tensor([[0.4, 1.0]], dtype=torch.float64)
    x = coordinate.decode(y)
    assert y[0, 0] < y[0, 1]
    assert (y.abs() * coordinate.write.norm(dim=0))[0].argmax() == 0
    expected = coordinate.decode(torch.tensor([[0.4, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(coordinate.sparse_reconstruct(x, 1), expected, atol=2e-12, rtol=0)
    torch.testing.assert_close(
        coordinate.sparse_reconstruct(x, 0), coordinate.mean[None], atol=0, rtol=0
    )
    torch.testing.assert_close(coordinate.sparse_reconstruct(x, 2), x, atol=2e-12, rtol=0)


def test_heldout_evaluation_does_not_refit_or_reorient():
    x, _ = mixed_sources(n=3000)
    heldout, _ = mixed_sources(n=700, seed=72)
    coordinate = fit_coordinates(x, seed=7)
    original_tensors = {
        name: getattr(coordinate, name).clone()
        for name in ("mean", "read", "write", "rotation", "eigenvalues", "fit_skewness")
    }
    original_diagnostics = dict(coordinate.diagnostics)
    # Deliberately reverse and shift the evaluation distribution: fitting or
    # orienting signs from these samples would change the original transform.
    shifted = -heldout + 100
    metrics = coordinate.evaluate(shifted, k=2)
    centered_y = coordinate.encode(shifted)
    centered_y -= centered_y.mean(0)
    expected = centered_y.pow(4).mean(0) / centered_y.square().mean(0).square() - 3
    torch.testing.assert_close(metrics["excess_kurtosis"], expected, atol=0, rtol=0)
    expected_nmse = (shifted - coordinate.sparse_reconstruct(shifted, 2)).square().sum() / (
        shifted - coordinate.mean
    ).square().sum()
    assert metrics["reconstruction_nmse"] == expected_nmse.item()
    full = coordinate.evaluate(heldout)
    assert full["reconstruction_nmse"] < 1e-24
    assert full["reconstruction_cosine"] == pytest.approx(1.0)
    for name, value in original_tensors.items():
        torch.testing.assert_close(getattr(coordinate, name), value, atol=0, rtol=0)
    assert coordinate.diagnostics == original_diagnostics


def test_rank_deficiency_and_invalid_inputs_are_explicit():
    x = torch.arange(30, dtype=torch.float64).reshape(10, 3)
    with pytest.raises(ValueError, match="rank deficient.*no regularization"):
        fit_coordinates(x)
    with pytest.raises(ValueError, match="more fit samples"):
        fit_coordinates(torch.eye(3))
    x, _ = mixed_sources(n=100)
    x[:, 2] *= 1e-8
    with pytest.raises(ValueError, match="rank deficient"):
        fit_coordinates(x)
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit_coordinates(x)


def test_seed_reproducibility_rng_isolation_and_nonconvergence_reporting():
    x, _ = mixed_sources(n=200)
    global_rng = torch.random.get_rng_state().clone()
    a = fit_coordinates(x, "random", seed=3)
    b = fit_coordinates(x, "random", seed=3)
    c = fit_coordinates(x, "random", seed=4)
    torch.testing.assert_close(a.read, b.read, atol=0, rtol=0)
    assert not torch.allclose(a.read, c.read)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, atol=0, rtol=0)
    unfinished = fit_coordinates(x, seed=3, max_iter=1, tolerance=1e-14, fit_dtype=torch.float64)
    assert unfinished.diagnostics["iterations"] == 1
    assert not unfinished.diagnostics["converged"]
    assert unfinished.diagnostics["duality_max_abs_error"] < 1e-12


def test_invalid_interventions_fail_and_empty_selection_is_identity():
    x, _ = mixed_sources(n=100)
    coordinate = fit_coordinates(x, "pca")
    with pytest.raises(ValueError, match="unique"):
        coordinate.ablate(x, [0, 0])
    with pytest.raises(ValueError, match="out of range"):
        coordinate.ablate(x, [-1])
    with pytest.raises(ValueError, match="integers"):
        coordinate.ablate(x, [0.5])
    with pytest.raises(ValueError, match="broadcast"):
        coordinate.replace_coordinates(x, [0, 1], torch.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        coordinate.replace_coordinates(x, [0], float("nan"))
    with pytest.raises(ValueError, match="k must"):
        coordinate.sparse_reconstruct(x, 4)
    torch.testing.assert_close(coordinate.ablate(x, []), x, atol=0, rtol=0)
