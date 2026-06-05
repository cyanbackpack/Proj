"""Tests for src/taxonomy/types.py."""

import numpy as np
import pytest

from src.taxonomy.types import (
    ANOMALY_REGISTRY,
    AnomalyLayer,
    AnomalyType,
    marginal_check,
    marginal_check_report,
)


def test_registry_complete():
    assert set(ANOMALY_REGISTRY.keys()) == set(AnomalyType)


def test_c_types_marginal_in_distribution():
    c_types = [t for t in AnomalyType if t.value.startswith("C")]
    for t in c_types:
        assert ANOMALY_REGISTRY[t].marginal_in_distribution, f"{t} must be marginal-in-dist"


def test_c_types_layer():
    c_types = [t for t in AnomalyType if t.value.startswith("C")]
    for t in c_types:
        assert ANOMALY_REGISTRY[t].layer == AnomalyLayer.INTER_METRIC


def test_a_types_layer():
    for t in [AnomalyType.A1, AnomalyType.A2]:
        assert ANOMALY_REGISTRY[t].layer == AnomalyLayer.MARGINAL


def test_b_types_layer():
    for t in [AnomalyType.B1, AnomalyType.B2, AnomalyType.B3]:
        assert ANOMALY_REGISTRY[t].layer == AnomalyLayer.TEMPORAL


# --- marginal_check tests ---

def _make_signal(T: int = 200, C: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, C))


def test_marginal_check_passes_small_signal():
    # Construct a signal with all values in [-1, 1] — guaranteed below any 2.5σ threshold.
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=(200, 3))
    assert marginal_check(x, x.copy(), window=(50, 80)) is True


def test_marginal_check_fails_large_spike():
    x = _make_signal()
    x_inj = x.copy()
    x_inj[50:80, 0] += 10.0  # huge spike in channel 0
    assert marginal_check(x, x_inj, window=(50, 80)) is False


def test_marginal_check_1d_input():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    assert marginal_check(x, x.copy(), window=(10, 30)) is True


def test_marginal_check_report_structure():
    x = _make_signal(C=4)
    report = marginal_check_report(x, x.copy(), window=(10, 50))
    assert "passes" in report
    assert "per_channel_max_z" in report
    assert len(report["per_channel_max_z"]) == 4
    assert report["passes"] is True


def test_marginal_check_constant_channel():
    """Constant channels should not cause division-by-zero."""
    x = _make_signal(C=2)
    x[:, 1] = 5.0  # constant
    x_inj = x.copy()
    assert marginal_check(x, x_inj, window=(0, 10)) is True
