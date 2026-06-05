"""Baseline detectors: ZScore and MovingAverage.

These are per-channel models — they score each channel independently and
take the max across channels.  They serve as the 'lower bound' in the
model × type recall matrix, expected to detect A-type anomalies only.

Reference: Wu & Keogh (IEEE TKDE 2021) — simple baselines.
"""

from __future__ import annotations

import numpy as np

from src.models.base import BaseDetector


class ZScoreDetector(BaseDetector):
    """Score each timestep by its max per-channel |z-score|.

    Fit: compute per-channel mean and std from training data.
    Score: max_c |( x[t,c] - μ_c ) / σ_c|.

    Parameters
    ----------
    eps:
        Floor for std to avoid division by zero on constant channels.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def fit(self, x: np.ndarray) -> "ZScoreDetector":
        self._mu = x.mean(axis=0)
        self._sigma = np.maximum(x.std(axis=0), self.eps)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_mu", self)
        z = np.abs((x - self._mu) / self._sigma)  # (T, C)
        return z.max(axis=1)                        # (T,)


class MovingAverageDetector(BaseDetector):
    """Score each timestep by max per-channel |x - MA(x)|, normalised by σ.

    Fit: compute per-channel std from training data (for normalisation).
    Score: for each channel, compute a causal moving average with window
    ``window`` and return max_c |residual_c[t]| / σ_c.

    Parameters
    ----------
    window:
        Moving-average half-window (in timesteps).
    eps:
        Floor for std.
    """

    def __init__(self, window: int = 20, eps: float = 1e-8) -> None:
        self.window = window
        self.eps = eps

    def fit(self, x: np.ndarray) -> "MovingAverageDetector":
        self._sigma = np.maximum(x.std(axis=0), self.eps)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_sigma", self)
        T, C = x.shape
        ma = np.zeros_like(x)
        for t in range(T):
            lo = max(0, t - self.window + 1)
            ma[t] = x[lo : t + 1].mean(axis=0)
        residuals = np.abs(x - ma) / self._sigma   # (T, C)
        return residuals.max(axis=1)                # (T,)
