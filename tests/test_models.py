"""Tests for src/models — BaseDetector, ZScore, MovingAverage, KNN, LOF."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.base import BaseDetector
from src.models.baseline import MovingAverageDetector, ZScoreDetector
from src.models.classic import (
    CovarianceAnomalyDetector,
    KNNDetector,
    LagCorrelationDetector,
    LOFDetector,
    MahalanobisDetector,
)
from src.data.generator import VARGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_signal(T: int = 500, C: int = 4, seed: int = 0) -> np.ndarray:
    gen = VARGenerator(n_channels=C, lag_order=2, T=T, spectral_radius=0.7, seed=seed)
    x, _ = gen.generate()
    return x


TRAIN_T = 400
TEST_T = 200
C = 4

TRAIN = make_signal(T=TRAIN_T, C=C, seed=0)
TEST = make_signal(T=TEST_T, C=C, seed=1)


# ---------------------------------------------------------------------------
# BaseDetector: sliding window helpers
# ---------------------------------------------------------------------------

class _DummyDetector(BaseDetector):
    def fit(self, x):
        return self
    def score(self, x):
        return np.zeros(x.shape[0])


def test_sliding_windows_shape():
    det = _DummyDetector()
    w = det._sliding_windows(TRAIN, window_size=10)
    expected_rows = TRAIN_T - 10 + 1
    assert w.shape == (expected_rows, 10 * C)


def test_expand_scores_length():
    det = _DummyDetector()
    ws = np.ones(TEST_T - 10 + 1)
    s = det._expand_scores(ws, TEST_T, window_size=10)
    assert s.shape == (TEST_T,)


def test_expand_scores_all_covered():
    det = _DummyDetector()
    window_size = 30
    window_scores = np.ones(TEST_T - window_size + 1)
    s = det._expand_scores(window_scores, TEST_T, window_size)
    assert np.all(s > 0), "Every timestep should have a non-zero score"


# ---------------------------------------------------------------------------
# ZScoreDetector
# ---------------------------------------------------------------------------

def test_zscore_fit_returns_self():
    det = ZScoreDetector()
    assert det.fit(TRAIN) is det


def test_zscore_score_shape():
    det = ZScoreDetector().fit(TRAIN)
    s = det.score(TEST)
    assert s.shape == (TEST_T,)


def test_zscore_scores_non_negative():
    det = ZScoreDetector().fit(TRAIN)
    assert (det.score(TEST) >= 0).all()


def test_zscore_spike_detected():
    """ZScore should give a very high score at an injected spike."""
    x = TEST.copy()
    x[50, 0] += 20.0  # large spike in channel 0
    det = ZScoreDetector().fit(TRAIN)
    s = det.score(x)
    assert s[50] == s.max(), "Spike position should be the maximum score"


def test_zscore_score_before_fit_raises():
    det = ZScoreDetector()
    with pytest.raises(RuntimeError, match="fit()"):
        det.score(TEST)


def test_zscore_constant_channel_no_error():
    """Constant channels should not cause division by zero."""
    x = TRAIN.copy()
    x[:, 2] = 5.0
    det = ZScoreDetector()
    det.fit(x)
    assert np.isfinite(det.score(TEST)).all()


# ---------------------------------------------------------------------------
# MovingAverageDetector
# ---------------------------------------------------------------------------

def test_ma_fit_returns_self():
    det = MovingAverageDetector(window=10)
    assert det.fit(TRAIN) is det


def test_ma_score_shape():
    det = MovingAverageDetector(window=10).fit(TRAIN)
    assert det.score(TEST).shape == (TEST_T,)


def test_ma_scores_non_negative():
    det = MovingAverageDetector().fit(TRAIN)
    assert (det.score(TEST) >= 0).all()


def test_ma_spike_elevates_score():
    x = TEST.copy()
    x[80, 0] += 15.0
    det = MovingAverageDetector(window=5).fit(TRAIN)
    s = det.score(x)
    s_clean = MovingAverageDetector(window=5).fit(TRAIN).score(TEST)
    assert s[80] > s_clean[80] * 2


def test_ma_before_fit_raises():
    det = MovingAverageDetector()
    with pytest.raises(RuntimeError):
        det.score(TEST)


# ---------------------------------------------------------------------------
# KNNDetector
# ---------------------------------------------------------------------------

def test_knn_fit_returns_self():
    det = KNNDetector(n_neighbors=3, window_size=10)
    assert det.fit(TRAIN) is det


def test_knn_score_shape():
    det = KNNDetector(n_neighbors=3, window_size=10).fit(TRAIN)
    assert det.score(TEST).shape == (TEST_T,)


def test_knn_scores_non_negative():
    det = KNNDetector(n_neighbors=3, window_size=10).fit(TRAIN)
    assert (det.score(TEST) >= 0).all()


def test_knn_spike_elevates_score():
    x = TEST.copy()
    x[60:70, :] += 10.0  # block anomaly
    det = KNNDetector(n_neighbors=3, window_size=10).fit(TRAIN)
    s = det.score(x)
    s_clean = KNNDetector(n_neighbors=3, window_size=10).fit(TRAIN).score(TEST)
    assert s[60:70].mean() > s_clean[60:70].mean()


def test_knn_before_fit_raises():
    det = KNNDetector()
    with pytest.raises(RuntimeError):
        det.score(TEST)


# ---------------------------------------------------------------------------
# LOFDetector
# ---------------------------------------------------------------------------

def test_lof_fit_returns_self():
    det = LOFDetector(n_neighbors=5, window_size=10)
    assert det.fit(TRAIN) is det


def test_lof_score_shape():
    det = LOFDetector(n_neighbors=5, window_size=10).fit(TRAIN)
    assert det.score(TEST).shape == (TEST_T,)


def test_lof_scores_finite():
    det = LOFDetector(n_neighbors=5, window_size=10).fit(TRAIN)
    assert np.isfinite(det.score(TEST)).all()


def test_lof_before_fit_raises():
    det = LOFDetector()
    with pytest.raises(RuntimeError):
        det.score(TEST)


# ---------------------------------------------------------------------------
# MahalanobisDetector
# ---------------------------------------------------------------------------

def test_mahal_fit_returns_self():
    det = MahalanobisDetector(window_size=10)
    assert det.fit(TRAIN) is det


def test_mahal_score_shape():
    det = MahalanobisDetector(window_size=10).fit(TRAIN)
    assert det.score(TEST).shape == (TEST_T,)


def test_mahal_scores_non_negative():
    det = MahalanobisDetector(window_size=10).fit(TRAIN)
    assert (det.score(TEST) >= 0).all()


def test_mahal_scores_finite():
    det = MahalanobisDetector(window_size=10).fit(TRAIN)
    assert np.isfinite(det.score(TEST)).all()


def test_mahal_invalid_estimator():
    with pytest.raises(ValueError, match="covariance_estimator"):
        MahalanobisDetector(covariance_estimator="bad")


def test_mahal_before_fit_raises():
    with pytest.raises(RuntimeError):
        MahalanobisDetector().score(TEST)


def test_mahal_catches_correlation_break():
    """Mahalanobis should score higher when cross-channel correlation breaks.

    Construct a highly-correlated 2-channel signal so the training covariance
    is almost rank-1; then inject a near-zero-correlation window and verify the
    Mahalanobis distance increases.
    """
    rng = np.random.default_rng(0)
    T_tr, T_te = 400, 200
    base = rng.standard_normal(T_tr + T_te)
    noise = rng.standard_normal((T_tr + T_te, 2)) * 0.05
    x_full = np.stack([base, base], axis=1) + noise  # near-perfect correlation

    x_tr = x_full[:T_tr]
    x_clean = x_full[T_tr:]
    # Inject C1: break correlation in window [50, 100]
    from src.taxonomy.injectors import C1CorrelationBreakInjector
    x_inj = C1CorrelationBreakInjector(seed=0, n_iter=16).inject(x_clean, (50, 100))

    det = MahalanobisDetector(window_size=20).fit(x_tr)
    s_clean = det.score(x_clean)
    s_inj = det.score(x_inj)
    assert s_inj[50:100].mean() > s_clean[50:100].mean(), (
        f"Expected higher score in anomaly window: "
        f"inj={s_inj[50:100].mean():.3f}, clean={s_clean[50:100].mean():.3f}"
    )


# ---------------------------------------------------------------------------
# CovarianceAnomalyDetector
# ---------------------------------------------------------------------------

def test_cov_fit_returns_self():
    det = CovarianceAnomalyDetector(window_size=20)
    assert det.fit(TRAIN) is det


def test_cov_score_shape():
    det = CovarianceAnomalyDetector(window_size=20).fit(TRAIN)
    assert det.score(TEST).shape == (TEST_T,)


def test_cov_scores_non_negative():
    det = CovarianceAnomalyDetector(window_size=20).fit(TRAIN)
    assert (det.score(TEST) >= 0).all()


def test_cov_scores_finite():
    det = CovarianceAnomalyDetector(window_size=20).fit(TRAIN)
    assert np.isfinite(det.score(TEST)).all()


def test_cov_before_fit_raises():
    with pytest.raises(RuntimeError):
        CovarianceAnomalyDetector().score(TEST)


def test_cov_elevated_on_correlation_break():
    """CovarianceAnomaly should react to a correlation-break injection."""
    rng = np.random.default_rng(1)
    T_tr, T_te = 400, 200
    base = rng.standard_normal(T_tr + T_te)
    noise = rng.standard_normal((T_tr + T_te, 2)) * 0.05
    x_full = np.stack([base, base], axis=1) + noise

    x_tr = x_full[:T_tr]
    x_clean = x_full[T_tr:]
    from src.taxonomy.injectors import C1CorrelationBreakInjector
    x_inj = C1CorrelationBreakInjector(seed=1, n_iter=16).inject(x_clean, (50, 100))

    det = CovarianceAnomalyDetector(window_size=20).fit(x_tr)
    s_clean = det.score(x_clean)
    s_inj = det.score(x_inj)
    assert s_inj[50:100].mean() > s_clean[50:100].mean(), (
        f"inj={s_inj[50:100].mean():.3f}, clean={s_clean[50:100].mean():.3f}"
    )


# ---------------------------------------------------------------------------
# LagCorrelationDetector
# ---------------------------------------------------------------------------

def test_lag_fit_returns_self():
    det = LagCorrelationDetector(window_size=50, max_lag=10)
    assert det.fit(TRAIN) is det


def test_lag_score_shape():
    det = LagCorrelationDetector(window_size=50, max_lag=10).fit(TRAIN)
    assert det.score(TEST).shape == (TEST_T,)


def test_lag_scores_non_negative():
    det = LagCorrelationDetector(window_size=50, max_lag=10).fit(TRAIN)
    assert (det.score(TEST) >= 0).all()


def test_lag_scores_finite():
    det = LagCorrelationDetector(window_size=50, max_lag=10).fit(TRAIN)
    assert np.isfinite(det.score(TEST)).all()


def test_lag_before_fit_raises():
    det = LagCorrelationDetector()
    with pytest.raises(RuntimeError):
        det.score(TEST)


def test_lag_single_channel_raises():
    with pytest.raises(ValueError, match="2 channels"):
        LagCorrelationDetector().fit(TRAIN[:, :1])


def test_lag_detects_phase_shift():
    """LagCorrelation should score higher after a C2 (phase-shift) injection."""
    rng = np.random.default_rng(0)
    T_tr = 400
    # Two strongly lagged channels: y[t] = x[t-10] + noise
    n_total = T_tr + TEST_T + 10
    base = rng.standard_normal(n_total)
    noise = rng.standard_normal((T_tr + TEST_T, 2)) * 0.1
    x_full = np.stack([base[10:], base[:-10]], axis=1) + noise  # lag-10 relationship

    x_tr = x_full[:T_tr]
    x_te = x_full[T_tr:]

    from src.taxonomy.injectors import C2PhaseShiftInjector
    x_inj = C2PhaseShiftInjector(tau=15).inject(x_te.copy(), (50, 150))

    det = LagCorrelationDetector(window_size=60, max_lag=25).fit(x_tr)
    s_clean = det.score(x_te)
    s_inj = det.score(x_inj)
    assert s_inj[50:150].mean() > s_clean[50:150].mean(), (
        f"Expected higher lag-shift score in anomaly window: "
        f"inj={s_inj[50:150].mean():.4f}, clean={s_clean[50:150].mean():.4f}"
    )


# ---------------------------------------------------------------------------
# AutoEncoderDetector: import guard only (torch may not be installed)
# ---------------------------------------------------------------------------

def test_ae_import_without_torch():
    """If torch is not installed, AE should raise ImportError on instantiation."""
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed — skipping no-torch guard test")
    except ImportError:
        pass
    from src.models.deep.ae import AutoEncoderDetector
    with pytest.raises(ImportError, match="PyTorch"):
        AutoEncoderDetector()
