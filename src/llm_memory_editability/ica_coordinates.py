"""Full-rank ICA, PCA, and random coordinates for residual-stream analysis.

For row-vector activations ``x`` of shape ``(..., d)``, coordinates are
``y = (x - mean) @ read`` and reconstruction is ``mean + y @ write.T``.
Columns of ``read`` and ``write`` are dual, generally different directions:
``read.T @ write = I``. Interventions therefore use columns of ``write``.

Covariance, whitening, and stored transforms use float64. Only the FastICA
fixed-point iterations optionally use float32. Covariance is normalized by
the number of fit samples, with no ridge or eigenvalue clipping. Numerically
rank-deficient fit data are rejected instead of claiming full-rank recovery.

The method follows the centered, whitened symmetric FastICA construction in
ICA Lens (arXiv:2606.11722v2, Appendix B.1). These numerical coordinates alone
do not establish semantic interpretation or causal use by a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

CoordinateMethod = Literal["ica", "pca", "random"]


def _matrix(x: Tensor, *, name: str, min_rows: int = 2) -> None:
    if not isinstance(x, Tensor) or x.ndim != 2:
        raise ValueError(f"{name} must be a tensor of shape (samples, features)")
    if x.shape[0] < min_rows or x.shape[1] < 1:
        raise ValueError(f"{name} needs at least {min_rows} samples and one feature")
    if not x.is_floating_point() or not torch.isfinite(x).all().item():
        raise ValueError(f"{name} must contain finite floating-point values")


def _symmetric_decorrelation(w: Tensor) -> Tensor:
    """Return ``(w @ w.T)^(-1/2) @ w`` without clipping singular values."""
    wd = w.to(torch.float64)
    values, vectors = torch.linalg.eigh(wd @ wd.T)
    if not torch.isfinite(values).all().item() or values[0].item() <= 0:
        raise ValueError("FastICA rotation update is singular; use a different seed")
    return ((vectors * values.rsqrt()) @ vectors.T @ wd).to(w.dtype)


def _moments(y: Tensor) -> tuple[Tensor, Tensor]:
    centered = y - y.mean(dim=0)
    variance = centered.square().mean(dim=0)
    skewness = centered.pow(3).mean(dim=0) / variance.pow(1.5)
    kurtosis = centered.pow(4).mean(dim=0) / variance.square() - 3
    undefined = torch.full_like(variance, float("nan"))
    return (
        torch.where(variance > 0, skewness, undefined),
        torch.where(variance > 0, kurtosis, undefined),
    )


@dataclass(frozen=True)
class CoordinateSystem:
    """Fitted transforms; all tensor fields are float64 on the fit device.

    ``mean`` and ``eigenvalues`` have shape ``(d,)``. ``read`` and ``write``
    have shape ``(d, d)`` with one feature direction per column. ``rotation``
    has shape ``(d, d)`` and maps whitened rows as ``z @ rotation.T``.
    Eigenvalues are in descending order. Signs are chosen using fit-data
    skewness only; component ordering is otherwise unchanged.

    Analysis methods return float64 tensors on the argument's device.
    Interventions return the argument's original dtype for use in model
    hooks, so their coordinate guarantees hold up to that dtype's rounding.
    No method refits coordinates or restores an activation's norm.
    """

    method: CoordinateMethod
    mean: Tensor
    read: Tensor
    write: Tensor
    rotation: Tensor
    eigenvalues: Tensor
    fit_skewness: Tensor
    diagnostics: dict[str, int | float | bool | str | None]

    @property
    def n_features(self) -> int:
        return self.mean.numel()

    def _activation(self, x: Tensor, name: str = "x") -> Tensor:
        if not isinstance(x, Tensor) or x.ndim < 1 or x.shape[-1] != self.n_features:
            raise ValueError(f"{name} must have final dimension {self.n_features}")
        if not x.is_floating_point() or not torch.isfinite(x).all().item():
            raise ValueError(f"{name} must contain finite floating-point values")
        return x.to(torch.float64)

    def encode(self, x: Tensor) -> Tensor:
        """Map activations ``(..., d)`` to fitted coordinates ``(..., d)``."""
        xd = self._activation(x)
        return (xd - self.mean.to(x.device)) @ self.read.to(x.device)

    def decode(self, y: Tensor) -> Tensor:
        """Map coordinates ``(..., d)`` back to activation space."""
        yd = self._activation(y, "y")
        return yd @ self.write.to(y.device).T + self.mean.to(y.device)

    def _indices(self, indices: Tensor | list[int] | tuple[int, ...], device) -> Tensor:
        selected = torch.as_tensor(indices, device=device)
        if selected.ndim != 1:
            raise ValueError("indices must be a one-dimensional sequence")
        if selected.numel() == 0:
            return selected.to(torch.long)
        if selected.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("indices must contain integers")
        selected = selected.to(torch.long)
        if selected.min().item() < 0 or selected.max().item() >= self.n_features:
            raise ValueError("coordinate index is out of range")
        if selected.unique().numel() != selected.numel():
            raise ValueError("coordinate indices must be unique")
        return selected

    def replace_coordinates(
        self,
        x: Tensor,
        indices: Tensor | list[int] | tuple[int, ...],
        values: Tensor | float,
    ) -> Tensor:
        """Set selected coordinates while keeping all others unchanged.

        ``values`` broadcasts to ``x.shape[:-1] + (len(indices),)``. The
        activation change uses the write directions, with no norm restoration.
        The input and stored fit tensors are never modified.
        """
        y = self.encode(x)
        selected = self._indices(indices, x.device)
        replacement = torch.as_tensor(values, dtype=torch.float64, device=x.device)
        if not torch.isfinite(replacement).all().item():
            raise ValueError("replacement values must be finite")
        try:
            replacement = torch.broadcast_to(replacement, y[..., selected].shape)
        except RuntimeError as error:
            raise ValueError(
                "replacement values do not broadcast to selected coordinates"
            ) from error
        delta = replacement - y[..., selected]
        updated = x.to(torch.float64) + delta @ self.write.to(x.device)[:, selected].T
        return updated.to(x.dtype)

    def ablate(self, x: Tensor, indices: Tensor | list[int] | tuple[int, ...]) -> Tensor:
        """Set selected centered coordinates to zero, leaving others intact."""
        return self.replace_coordinates(x, indices, 0.0)

    def patch(
        self, x: Tensor, donor: Tensor, indices: Tensor | list[int] | tuple[int, ...]
    ) -> Tensor:
        """Replace selected coordinates with a broadcast-compatible donor."""
        selected = self._indices(indices, x.device)
        donor_y = self.encode(donor).to(x.device)
        return self.replace_coordinates(x, selected, donor_y[..., selected])

    def sparse_reconstruct(self, x: Tensor, k: int) -> Tensor:
        """Keep the top ``k`` contributions per row and restore the fit mean.

        Ranking is ``abs(y_j) * norm(write[:, j])``, not coordinate magnitude
        alone. This is the component-contribution heuristic; nonorthogonal
        write vectors mean it need not minimize reconstruction error globally.
        ``k=0`` returns the fitted mean and ``k=d`` returns full reconstruction.
        """
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= self.n_features:
            raise ValueError(f"k must be an integer in [0, {self.n_features}]")
        y = self.encode(x)
        scores = y.abs() * self.write.to(x.device).norm(dim=0)
        indices = scores.topk(k, dim=-1, sorted=False).indices
        sparse = torch.zeros_like(y).scatter(-1, indices, y.gather(-1, indices))
        return self.decode(sparse)

    @torch.no_grad()
    def evaluate(self, heldout: Tensor, *, k: int | None = None) -> dict[str, Tensor | float | int]:
        """Measure heldout moments/reconstruction without changing the fit.

        Excess kurtosis is the population fourth central moment divided by
        variance squared, minus three, computed on heldout coordinates.
        ``reconstruction_nmse`` divides total squared error by the heldout
        squared norm after subtracting the *fit* mean. Both raw and centered
        mean row-wise cosine are reported; zero-vector cosine is zero.
        A zero NMSE denominator or constant-coordinate kurtosis yields NaN.
        """
        _matrix(heldout, name="heldout")
        y = self.encode(heldout)
        _, kurtosis = _moments(y)
        reconstruction = self.decode(y) if k is None else self.sparse_reconstruct(heldout, k)
        xd = heldout.to(torch.float64)
        centered = xd - self.mean.to(heldout.device)
        centered_reconstruction = reconstruction - self.mean.to(heldout.device)
        denominator = centered.square().sum()
        nmse = (
            (xd - reconstruction).square().sum() / denominator
            if denominator.item() > 0
            else torch.tensor(float("nan"), device=heldout.device)
        )
        return {
            "n_samples": heldout.shape[0],
            "k": self.n_features if k is None else k,
            "excess_kurtosis": kurtosis,
            "mean_absolute_excess_kurtosis": kurtosis.abs().mean().item(),
            "reconstruction_nmse": nmse.item(),
            "reconstruction_cosine": F.cosine_similarity(xd, reconstruction, dim=-1).mean().item(),
            "centered_reconstruction_cosine": F.cosine_similarity(
                centered, centered_reconstruction, dim=-1
            )
            .mean()
            .item(),
        }


@torch.no_grad()
def fit_coordinates(
    x: Tensor,
    method: CoordinateMethod = "ica",
    *,
    seed: int = 0,
    max_iter: int = 1000,
    tolerance: float = 1e-5,
    fit_dtype: torch.dtype = torch.float32,
    rank_rtol: float = 1e-12,
    rank_atol: float = 0.0,
) -> CoordinateSystem:
    """Fit full-rank coordinates on ``x`` with shape ``(samples, features)``.

    ``ica`` uses symmetric FastICA with ``tanh``/logcosh. ``pca`` keeps the
    whitened PCA axes; ``random`` applies a seeded orthogonal rotation in
    that same whitened space. All methods orient signs by fit skewness.

    A covariance eigenvalue at or below ``max(rank_atol, rank_rtol * max_eig)``
    raises ValueError. No regularization is performed. ICA nonconvergence is
    retained in ``diagnostics['converged']``; the last orthogonal iterate is
    returned and must not be described as a converged fit.
    """
    _matrix(x, name="x")
    if method not in ("ica", "pca", "random"):
        raise ValueError("method must be 'ica', 'pca', or 'random'")
    if fit_dtype not in (torch.float32, torch.float64):
        raise ValueError("fit_dtype must be torch.float32 or torch.float64")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between zero and one")
    if not 0 <= rank_rtol < 1 or not 0 <= rank_atol < float("inf"):
        raise ValueError("rank tolerances must be finite and nonnegative, with rank_rtol < 1")
    n, d = x.shape
    if n <= d:
        raise ValueError("full-rank centered covariance needs more fit samples than features")

    xd = x.detach().to(torch.float64)
    mean = xd.mean(dim=0)
    centered = xd - mean
    covariance = centered.T @ centered / n
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    threshold = max(rank_atol, rank_rtol * eigenvalues[-1].item())
    if not torch.isfinite(eigenvalues).all().item() or eigenvalues[0].item() <= threshold:
        rank = int((eigenvalues > threshold).sum().item())
        raise ValueError(
            f"fit covariance is rank deficient at tolerance {threshold:.3g}: "
            f"rank {rank}/{d}; min eigenvalue {eigenvalues[0].item():.3g}; "
            "no regularization was applied"
        )
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    whitening = eigenvectors * eigenvalues.rsqrt()
    unwhitening = eigenvectors * eigenvalues.sqrt()
    z = (centered @ whitening).to(fit_dtype)

    generator = torch.Generator(device=x.device).manual_seed(seed)
    iterations, converged, convergence_delta = 0, True, 0.0
    if method == "pca":
        w = torch.eye(d, dtype=torch.float64, device=x.device)
    else:
        initial = torch.randn(d, d, dtype=torch.float64, device=x.device, generator=generator)
        w = torch.linalg.qr(initial).Q.T.to(fit_dtype)
        if method == "ica":
            converged = False
            for iteration in range(1, max_iter + 1):
                iterations = iteration
                y = z @ w.T
                g = y.tanh()
                update = g.T @ z / n - (1 - g.square()).mean(dim=0)[:, None] * w
                next_w = _symmetric_decorrelation(update)
                convergence_delta = (torch.diagonal(next_w @ w.T).abs() - 1).abs().max().item()
                w = next_w
                if convergence_delta < tolerance:
                    converged = True
                    break

    # A float64 final projection preserves the dual-matrix contract even when
    # the iterative fit used float32 and accumulated orthogonality roundoff.
    w = _symmetric_decorrelation(w.to(torch.float64))
    read, write = whitening @ w.T, unwhitening @ w.T
    skewness, _ = _moments(centered @ read)
    signs = torch.where(skewness < 0, -torch.ones_like(skewness), torch.ones_like(skewness))
    w = w * signs[:, None]
    read, write = read * signs, write * signs
    diagnostics = {
        "n_fit_samples": n,
        "n_features": d,
        "seed": seed,
        "fit_dtype": str(fit_dtype),
        "covariance_dtype": "torch.float64",
        "covariance_normalization": "population (divide by n)",
        "rank_rtol": rank_rtol,
        "rank_atol": rank_atol,
        "rank_threshold": threshold,
        "covariance_condition": (eigenvalues[0] / eigenvalues[-1]).item(),
        "regularization": None,
        "iterations": iterations,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "converged": converged,
        "convergence_delta": convergence_delta,
        "sign_orientation": "fit-data skewness",
        "duality_max_abs_error": (
            read.T @ write - torch.eye(d, dtype=torch.float64, device=x.device)
        )
        .abs()
        .max()
        .item(),
    }
    return CoordinateSystem(
        method, mean, read, write, w, eigenvalues, skewness * signs, diagnostics
    )
