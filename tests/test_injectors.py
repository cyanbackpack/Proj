"""Tests for src/taxonomy/injectors.py — C1 CorrelationBreak."""

from __future__ import annotations

import numpy as np
import pytest

from src.taxonomy.injectors import (
    C1CorrelationBreakInjector,
    MarginalViolationError,
    _iaaft_surrogate,
)
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
    # Non-target channels in window should be unchanged
    np.testing.assert_array_equal(x_inj[start:end, 0], x[start:end, 0])
    np.testing.assert_array_equal(x_inj[start:end, 2], x[start:end, 2])
    np.testing.assert_array_equal(x_inj[start:end, 3], x[start:end, 3])
    # Target channel should differ
    assert not np.allclose(x_inj[start:end, 1], x[start:end, 1])
