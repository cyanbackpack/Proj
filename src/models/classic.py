"""Classic anomaly detectors: KNN and LOF, using sliding windows.

Both models operate on flattened sliding windows of shape (window_size * C,).
Scores are expanded back to per-timestep by averaging overlapping windows.

These are multivariate models (they consume the full flattened window), so
they can, in principle, pick up some inter-metric patterns — but without an
explicit joint model they may still miss subtle C-type anomalies.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from src.models.base import BaseDetector


class KNNDetector(BaseDetector):
    """k-Nearest-Neighbours distance detector over sliding windows.

    Score at each window = mean distance to the k nearest training windows.
    This is a density-free anomaly score that works well for moderate-dim data.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbours.
    window_size:
        Sliding window length (timesteps).
    metric:
        Distance metric passed to :class:`~sklearn.neighbors.NearestNeighbors`.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 50,
        metric: str = "euclidean",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.metric = metric

    def fit(self, x: np.ndarray) -> "KNNDetector":
        windows = self._sliding_windows(x, self.window_size)
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self._nn.fit(windows)
        self._T_train = x.shape[0]
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_nn", self)
        T = x.shape[0]
        windows = self._sliding_windows(x, self.window_size)
        dists, _ = self._nn.kneighbors(windows)
        window_scores = dists.mean(axis=1)
        return self._expand_scores(window_scores, T, self.window_size)


class LOFDetector(BaseDetector):
    """Local Outlier Factor detector over sliding windows.

    LOF measures the local density deviation of each point relative to its
    neighbours.  Scores > 1 indicate anomalous regions; exactly 1 = normal.
    We negate and shift so higher = more anomalous for consistency.

    Parameters
    ----------
    n_neighbors:
        Number of neighbours used in the LOF calculation.
    window_size:
        Sliding window length.
    novelty:
        If True, use novelty detection mode (fit on train, score on test).
        Must be True for our train/test split workflow.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        window_size: int = 50,
        novelty: bool = True,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.novelty = novelty

    def fit(self, x: np.ndarray) -> "LOFDetector":
        windows = self._sliding_windows(x, self.window_size)
        self._lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=self.novelty,
        )
        self._lof.fit(windows)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_lof", self)
        T = x.shape[0]
        windows = self._sliding_windows(x, self.window_size)
        # LOF decision_function returns negative outlier factor; negate for consistency
        lof_scores = -self._lof.decision_function(windows)
        return self._expand_scores(lof_scores, T, self.window_size)
