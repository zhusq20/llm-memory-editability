"""Only binary64 cloud serialization drift is tolerated, never lost results."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "readout_cloud_verifier", ROOT / "scripts/verify_parallel_readouts_cloud.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_accepts_last_digit_json_rounding():
    assert module.metric_matches(11.337857156991959, 11.33785715699196)


def test_rejects_material_metric_change():
    assert not module.metric_matches(0.7890625, 0.7890635)


def test_counts_and_booleans_remain_exact():
    assert module.metric_matches(128, 128)
    assert not module.metric_matches(128, 128.000000000001)
    assert not module.metric_matches(True, 1)
    assert not module.metric_matches(0, False)


def test_missing_and_nonfinite_values_fail():
    assert not module.metric_matches(0.5, None)
    assert not module.metric_matches(0.5, float("nan"))
    assert not module.metric_matches(0.5, float("inf"))
