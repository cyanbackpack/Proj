"""Tests for src/models — BaseDetector, ZScore, MovingAverage, KNN, LOF."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.base import BaseDetector
from src.models.baseline import MovingAverageDetector, ZScoreDetector
from src.models.classic import KNNDetector, LOFDetector
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
