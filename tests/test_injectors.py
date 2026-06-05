"""Tests for src/taxonomy/injectors.py — all anomaly types."""

from __future__ import annotations

import numpy as np
import pytest

from src.taxonomy.injectors import (
    A1GlobalPointInjector,
    A2ContextualPointInjector,
    B1SeasonalInjector,
    B2TrendInjector,
    B3ShapeletInjector,
    C1CorrelationBreakInjector,
    C2PhaseShiftInjector,
    C3RatioAnomalyInjector,
    C4GroupCollectiveInjector,
    C5CausalViolationInjector,
    INJECTOR_REGISTRY,
    MarginalViolationError,
    _iaaft_surrogate,
)
from src.taxonomy.types import AnomalyType
from src.data.generator import VARGenerator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_var_signal(n_channels: int = 4, T: int = 600, seed: int = 0) -> np.ndarray:
    """Generate a stationary VAR(2) signal for injection tests."""
    gen = VARGenerator(n_channels=n_channels, lag_order=2, T=T,
                       spectral_radius=0.7, seed=seed)
    x, _ = gen.generate()
    return x


WINDOW = (100, 200)   # 100-step window


# ------------------------------------------------------------------
# IAAFT surrogate unit tests
# ------------------------------------------------------------------

def test_iaaft_same_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(128)
    s = _iaaft_surrogate(x, rng=rng)
    assert s.shape == x.shape


def test_iaaft_amplitude_spectrum_preserved():
    """Surrogate amplitude spectrum should closely match original."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(256)
    s = _iaaft_surrogate(x, n_iter=16, rng=rng)

    amp_orig = np.abs(np.fft.rfft(x))
    amp_surr = np.abs(np.fft.rfft(s))
    # Allow up to 5% relative error on mean amplitude
    rel_err = np.abs(amp_orig - amp_surr).mean() / (amp_orig.mean() + 1e-12)
    assert rel_err < 0.05, f"Amplitude spectrum mismatch: rel_err={rel_err:.4f}"


def test_iaaft_mean_std_preserved():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(512) * 3 + 1   # non-zero mean / std
    s = _iaaft_surrogate(x, n_iter=8, rng=rng)
    assert abs(s.mean() - x.mean()) < 0.3
    assert abs(s.std() - x.std()) < 0.3


def test_iaaft_phases_randomized():
    """Surrogate should differ from original (phases randomized)."""
    rng = np.random.default_rng(3)
    x = np.sin(np.linspace(0, 10 * np.pi, 256)) + rng.standard_normal(256) * 0.1
    s = _iaaft_surrogate(x, rng=rng)
    assert not np.allclose(x, s, atol=1e-6)


# ------------------------------------------------------------------
# C1 injector: output shape and non-mutation
# ------------------------------------------------------------------

def test_c1_inject_shape():
    x = make_var_signal()
    inj = C1CorrelationBreakInjector(seed=0)
    x_inj = inj.inject(x, WINDOW)
    assert x_inj.shape == x.shape


def test_c1_inject_does_not_mutate_original():
    x = make_var_signal()
    x_orig = x.copy()
    inj = C1CorrelationBreakInjector(seed=0)
    inj.inject(x, WINDOW)
    np.testing.assert_array_equal(x, x_orig)


def test_c1_outside_window_unchanged():
    x = make_var_signal()
    inj = C1CorrelationBreakInjector(seed=0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[:start], x[:start])
    np.testing.assert_array_equal(x_inj[end:], x[end:])


def test_c1_window_changed():
    x = make_var_signal()
    inj = C1CorrelationBreakInjector(seed=0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    assert not np.allclose(x_inj[start:end], x[start:end])


# ------------------------------------------------------------------
# C1: marginal invariant
# ------------------------------------------------------------------

def test_c1_marginal_check_passes():
    x = make_var_signal(T=1000)
    inj = C1CorrelationBreakInjector(seed=42)
    x_inj, report = inj.inject_and_verify(x, WINDOW)
    assert report["passes"], f"Marginal check failed: {report}"


def test_c1_inject_and_verify_returns_report_keys():
    x = make_var_signal()
    inj = C1CorrelationBreakInjector(seed=0)
    _, report = inj.inject_and_verify(x, WINDOW)
    for key in ("passes", "z_threshold", "per_channel_max_z", "window"):
        assert key in report


def test_c1_marginal_violation_raises():
    """A synthetic injector that violates marginals should raise."""
    x = make_var_signal()

    class BadInjector(C1CorrelationBreakInjector):
        def inject(self, x, window):
            x_inj = x.copy()
            x_inj[window[0]:window[1], 0] += 100.0   # huge spike
            return x_inj

    inj = BadInjector(seed=0)
    with pytest.raises(MarginalViolationError):
        inj.inject_and_verify(x, WINDOW, raise_on_violation=True)


# ------------------------------------------------------------------
# C1: correlation actually breaks
# ------------------------------------------------------------------

def test_c1_correlation_reduced():
    """Cross-correlation between channels should decrease after C1 injection."""
    x = make_var_signal(n_channels=2, T=1000, seed=5)
    inj = C1CorrelationBreakInjector(target_channels=[0, 1], seed=5, n_iter=16)
    x_inj = inj.inject(x, WINDOW)

    start, end = WINDOW
    seg_orig = x[start:end]
    seg_inj = x_inj[start:end]

    # Pearson correlation between channels 0 and 1
    def pearson(a: np.ndarray) -> float:
        return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])

    corr_before = abs(pearson(seg_orig))
    corr_after = abs(pearson(seg_inj))
    assert corr_after < corr_before, (
        f"Correlation did not decrease: before={corr_before:.3f}, after={corr_after:.3f}"
    )


# ------------------------------------------------------------------
# Selective channel targeting
# ------------------------------------------------------------------

def test_c1_only_target_channels_modified():
    x = make_var_signal(n_channels=4)
    inj = C1CorrelationBreakInjector(target_channels=[1], seed=0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[start:end, 0], x[start:end, 0])
    np.testing.assert_array_equal(x_inj[start:end, 2], x[start:end, 2])
    np.testing.assert_array_equal(x_inj[start:end, 3], x[start:end, 3])
    assert not np.allclose(x_inj[start:end, 1], x[start:end, 1])


# ===========================================================================
# C2: Phase/lag-shift
# ===========================================================================

def test_c2_shape_unchanged():
    x = make_var_signal()
    inj = C2PhaseShiftInjector(lag_channel=1, tau=10)
    assert inj.inject(x, WINDOW).shape == x.shape


def test_c2_only_lag_channel_modified():
    x = make_var_signal(n_channels=4)
    inj = C2PhaseShiftInjector(lag_channel=2, tau=5)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[start:end, 0], x[start:end, 0])
    np.testing.assert_array_equal(x_inj[start:end, 1], x[start:end, 1])
    np.testing.assert_array_equal(x_inj[start:end, 3], x[start:end, 3])


def test_c2_outside_window_unchanged():
    x = make_var_signal()
    inj = C2PhaseShiftInjector(lag_channel=0, tau=5)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[:start], x[:start])
    np.testing.assert_array_equal(x_inj[end:], x[end:])


def test_c2_marginal_passes():
    x = make_var_signal(T=1000)
    inj = C2PhaseShiftInjector(lag_channel=1, tau=15)
    _, report = inj.inject_and_verify(x, WINDOW)
    assert report["passes"]


def test_c2_lag_relation_broken():
    """After phase shift, cross-correlation peak should shift."""
    x = make_var_signal(n_channels=2, T=1000, seed=7)
    inj = C2PhaseShiftInjector(lag_channel=1, tau=20)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    # The injected segment of ch1 is a rolled version — it shouldn't match original
    assert not np.allclose(x_inj[start:end, 1], x[start:end, 1])


# ===========================================================================
# C3: Ratio anomaly
# ===========================================================================

def test_c3_shape_unchanged():
    x = make_var_signal()
    inj = C3RatioAnomalyInjector(channel_pair=(0, 1))
    assert inj.inject(x, WINDOW).shape == x.shape


def test_c3_rank_reversed():
    """ch_a window should use the same value set with reversed rank order."""
    x = make_var_signal(n_channels=4)
    inj = C3RatioAnomalyInjector(channel_pair=(0, 1))
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    # Injected values must be the same set as original (just reordered)
    np.testing.assert_array_equal(np.sort(x_inj[start:end, 0]), np.sort(x[start:end, 0]))
    # ch_b and other channels untouched
    np.testing.assert_array_equal(x_inj[start:end, 1], x[start:end, 1])
    np.testing.assert_array_equal(x_inj[start:end, 2], x[start:end, 2])


def test_c3_outside_window_unchanged():
    x = make_var_signal()
    inj = C3RatioAnomalyInjector()
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[:start], x[:start])
    np.testing.assert_array_equal(x_inj[end:], x[end:])


def test_c3_marginal_passes():
    x = make_var_signal(T=1000)
    inj = C3RatioAnomalyInjector(channel_pair=(0, 2))
    _, report = inj.inject_and_verify(x, WINDOW)
    assert report["passes"]


# ===========================================================================
# C4: Group-collective
# ===========================================================================

def test_c4_shape_unchanged():
    x = make_var_signal()
    inj = C4GroupCollectiveInjector(group_channels=[0, 1], donor_offset=200)
    assert inj.inject(x, WINDOW).shape == x.shape


def test_c4_non_group_unchanged():
    x = make_var_signal(n_channels=4)
    inj = C4GroupCollectiveInjector(group_channels=[0, 1], donor_offset=200)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[start:end, 2], x[start:end, 2])
    np.testing.assert_array_equal(x_inj[start:end, 3], x[start:end, 3])


def test_c4_marginal_passes():
    x = make_var_signal(T=1000)
    inj = C4GroupCollectiveInjector(group_channels=[0, 1], donor_offset=300)
    _, report = inj.inject_and_verify(x, WINDOW)
    assert report["passes"]


# ===========================================================================
# C5: Causal/precedence violation
# ===========================================================================

def test_c5_shape_unchanged():
    x = make_var_signal()
    inj = C5CausalViolationInjector(cause_channel=0, effect_channel=1)
    assert inj.inject(x, WINDOW).shape == x.shape


def test_c5_effect_channel_unchanged():
    x = make_var_signal(n_channels=4)
    inj = C5CausalViolationInjector(cause_channel=0, effect_channel=1, seed=0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[start:end, 1], x[start:end, 1])
    # Cause channel should differ
    assert not np.allclose(x_inj[start:end, 0], x[start:end, 0])


def test_c5_marginal_passes():
    x = make_var_signal(T=1000)
    inj = C5CausalViolationInjector(cause_channel=0, effect_channel=1, seed=42)
    _, report = inj.inject_and_verify(x, WINDOW)
    assert report["passes"]


# ===========================================================================
# Layer A: A1 GlobalPoint, A2 ContextualPoint
# ===========================================================================

def test_a1_creates_spike():
    x = make_var_signal()
    inj = A1GlobalPointInjector(target_channel=0, n_sigma=5.0, n_points=1)
    x_inj = inj.inject(x, WINDOW)
    # The injected signal should have at least one value far above global std
    sigma = x[:, 0].std()
    mu = x[:, 0].mean()
    z_max = (np.abs(x_inj[:, 0] - mu) / sigma).max()
    assert z_max > 4.0, f"Expected spike > 4σ, got {z_max:.2f}σ"


def test_a1_only_target_channel():
    x = make_var_signal(n_channels=4)
    inj = A1GlobalPointInjector(target_channel=2, n_points=1)
    x_inj = inj.inject(x, WINDOW)
    np.testing.assert_array_equal(x_inj[:, 0], x[:, 0])
    np.testing.assert_array_equal(x_inj[:, 1], x[:, 1])
    np.testing.assert_array_equal(x_inj[:, 3], x[:, 3])


def test_a2_anomalous_points_inserted():
    x = make_var_signal()
    inj = A2ContextualPointInjector(target_channel=0, n_points=3, seed=0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    # Window should differ somewhere
    assert not np.allclose(x_inj[start:end, 0], x[start:end, 0])


def test_a2_outside_unchanged():
    x = make_var_signal()
    inj = A2ContextualPointInjector(target_channel=0, seed=0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[:start], x[:start])
    np.testing.assert_array_equal(x_inj[end:], x[end:])


# ===========================================================================
# Layer B: B1 Seasonal, B2 Trend, B3 Shapelet
# ===========================================================================

def test_b1_channel_modified_in_window():
    x = make_var_signal()
    inj = B1SeasonalInjector(target_channel=0, phase_shift=25)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    assert not np.allclose(x_inj[start:end, 0], x[start:end, 0])


def test_b1_outside_unchanged():
    x = make_var_signal()
    inj = B1SeasonalInjector(target_channel=0, phase_shift=25)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    np.testing.assert_array_equal(x_inj[:start], x[:start])
    np.testing.assert_array_equal(x_inj[end:], x[end:])


def test_b2_ramp_increases_values():
    x = make_var_signal()
    inj = B2TrendInjector(target_channel=0, trend_type="ramp", slope=0.1)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    diff = x_inj[start:end, 0] - x[start:end, 0]
    # Ramp should produce monotonically increasing differences
    assert diff[-1] > diff[0]


def test_b2_step_constant_offset():
    x = make_var_signal()
    inj = B2TrendInjector(target_channel=0, trend_type="step", slope=1.0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    diff = x_inj[start:end, 0] - x[start:end, 0]
    assert np.allclose(diff, diff[0])  # constant offset


def test_b2_invalid_trend_type():
    with pytest.raises(ValueError, match="trend_type must be"):
        B2TrendInjector(trend_type="sine")


@pytest.mark.parametrize("shape", ["bump", "plateau", "zigzag"])
def test_b3_shapes(shape):
    x = make_var_signal()
    inj = B3ShapeletInjector(target_channel=0, shape=shape, amplitude=2.0)
    x_inj = inj.inject(x, WINDOW)
    start, end = WINDOW
    assert not np.allclose(x_inj[start:end, 0], x[start:end, 0])


def test_b3_invalid_shape():
    with pytest.raises(ValueError, match="shape must be one of"):
        B3ShapeletInjector(shape="triangle")


# ===========================================================================
# Registry completeness
# ===========================================================================

def test_injector_registry_covers_all_types():
    assert set(INJECTOR_REGISTRY.keys()) == set(AnomalyType)


def test_all_injectors_are_base_injector_subclasses():
    from src.taxonomy.injectors import BaseInjector
    for t, cls in INJECTOR_REGISTRY.items():
        assert issubclass(cls, BaseInjector), f"{t}: {cls} is not a BaseInjector subclass"
