"""Independent-factor reconstruction and paired numerical summaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_path_transfer.py"
SPEC = importlib.util.spec_from_file_location("path_summary", PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_factor_audit_detects_wrong_transfer_sign(tmp_path):
    z = np.array([[[1.0, 2.0], [3.0, -1.0]]])
    delta = np.array([[[2.0, -1.0], [1.0, 4.0]]])
    g = np.einsum("btd,btk->bdk", delta, z)[0]
    s = np.outer(delta[0, 0], z[0, 0])
    u = np.outer(delta[0, 1], z[0, 1])
    path = tmp_path / "factors.npz"
    np.savez(path, evaluation_z=z, evaluation_delta=delta, full_z=z, full_delta=delta)
    row = {
        "arm": "full",
        "probe": 0,
        "kernel": np.sum(g * g),
        "kernel_supervised_supervised": np.sum(s * s),
        "kernel_supervised_unsupervised": np.sum(s * u),
        "kernel_unsupervised_supervised": np.sum(u * s),
        "kernel_unsupervised_unsupervised": np.sum(u * u),
    }
    assert summary.audit_factors(path, [row]) == 0
    row["kernel"] *= -1
    with pytest.raises(ValueError, match="signed transfer"):
        summary.audit_factors(path, [row])


def test_small_unresolved_values_are_retained_in_error():
    rows = [
        {"predicted_change": 0.5, "observed_change": 1.0},
        {"predicted_change": 0.0, "observed_change": 1e-8},
    ]
    result = summary.approximation(rows)
    assert result["n"] == 2
    assert result["resolved"] == 1
    assert result["sign_agreement"] == 1
    assert result["rmse"] == pytest.approx(np.sqrt((0.25 + 1e-16) / 2))


def test_paired_blocks_keep_world_and_seed():
    rows = [
        {"world": w, "seed": s, "value": w + s + x}
        for w in range(2)
        for s in range(2)
        for x in (0, 2)
    ]
    result = summary.aggregate(rows, ["world", "seed"], ["value"])
    assert len(result) == 4
    assert all(row["n"] == 2 for row in result)
    assert [row["value"] for row in result] == [1, 2, 2, 3]
