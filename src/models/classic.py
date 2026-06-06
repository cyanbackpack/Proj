"""Classic anomaly detectors: KNN, LOF, Mahalanobis, CovarianceAnomaly, LagCorrelation.

KNN / LOF operate on flattened sliding windows (window_size * C).
Mahalanobis / CovarianceAnomaly explicitly model joint channel structure,
making them sensitive to C-type (inter-metric) anomalies where per-channel
detectors (ZScore, MA) are expected to fail.
LagCorrelationDetector specifically targets C2 (Phase/lag-shift) anomalies by
tracking the peak-lag of the cross-correlation function between channel pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate as _sig_correlate
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
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


# ---------------------------------------------------------------------------
# Joint-structure detectors (sensitive to C-type inter-metric anomalies)
# ---------------------------------------------------------------------------

class MahalanobisDetector(BaseDetector):
    """Mahalanobis distance detector over sliding windows.

    Learns the multivariate Gaussian of normal windows (mean + covariance).
    Scores each test window by its Mahalanobis distance from that distribution.
    Unlike ZScore (which scores channels independently), Mahalanobis captures
    cross-channel covariance and is therefore sensitive to C-type anomalies
    where marginals stay in-distribution but joint structure breaks.

    Parameters
    ----------
    window_size:
        Sliding window length (timesteps).
    covariance_estimator:
        ``'empirical'`` or ``'ledoit_wolf'`` (shrinkage, more stable for small
        samples or high-dimensional windows).
    """

    def __init__(
        self,
        window_size: int = 50,
        covariance_estimator: str = "ledoit_wolf",
    ) -> None:
        if covariance_estimator not in ("empirical", "ledoit_wolf"):
            raise ValueError(
                f"covariance_estimator must be 'empirical' or 'ledoit_wolf', "
                f"got '{covariance_estimator}'"
            )
        self.window_size = window_size
        self.covariance_estimator = covariance_estimator

    def fit(self, x: np.ndarray) -> "MahalanobisDetector":
        windows = self._sliding_windows(x, self.window_size)
        EstClass = LedoitWolf if self.covariance_estimator == "ledoit_wolf" else EmpiricalCovariance
        self._cov = EstClass().fit(windows)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_cov", self)
        T = x.shape[0]
        windows = self._sliding_windows(x, self.window_size)
        # mahalanobis() returns squared distances; take sqrt for interpretability
        sq_dists = self._cov.mahalanobis(windows)
        window_scores = np.sqrt(np.maximum(sq_dists, 0.0))
        return self._expand_scores(window_scores, T, self.window_size)


class CovarianceAnomalyDetector(BaseDetector):
    """Covariance-deviation detector: flags breaks in cross-channel correlation.

    Explicitly designed to catch C-type anomalies.

    Fit: compute the global covariance matrix Σ_train from training data.
    Score: for each local window of length ``window_size``, compute the local
    covariance Σ_local and return ||Σ_local - Σ_train||_F (Frobenius norm).
    A C-type anomaly (e.g. correlation break, phase shift) changes local
    covariance while leaving per-channel marginals unchanged — this detector
    is specifically tuned to flag that deviation.

    Parameters
    ----------
    window_size:
        Local window used to estimate covariance at each timestep.
    stride:
        Step between consecutive windows (1 = fully overlapping; higher = faster).
    """

    def __init__(self, window_size: int = 50, stride: int = 1) -> None:
        self.window_size = window_size
        self.stride = stride

    def fit(self, x: np.ndarray) -> "CovarianceAnomalyDetector":
        # Global covariance of the training signal
        self._sigma_train = np.cov(x.T)   # (C, C)
        if self._sigma_train.ndim == 0:   # single channel fallback
            self._sigma_train = self._sigma_train.reshape(1, 1)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_sigma_train", self)
        T, C = x.shape
        scores = np.zeros(T)
        counts = np.zeros(T)
        w = self.window_size

        for start in range(0, T - w + 1, self.stride):
            end = start + w
            seg = x[start:end]
            sigma_local = np.cov(seg.T) if C > 1 else np.var(seg).reshape(1, 1)
            diff = sigma_local - self._sigma_train
            frob = float(np.sqrt((diff ** 2).sum()))
            scores[start:end] += frob
            counts[start:end] += 1

        counts = np.where(counts == 0, 1, counts)
        return scores / counts


class LagCorrelationDetector(BaseDetector):
    """Cross-lag correlation shift detector.

    Specifically designed to catch **C2 (Phase/lag-shift)** anomalies.

    Fit: for each channel-pair (i, j), compute the cross-correlation function
    (CCF) at lags 0..max_lag over the training data and record the dominant
    lag (argmax |CCF|).

    Score: for each sliding window, recompute the CCF and measure the shift
    in the dominant lag relative to the training baseline.  A large lag-shift
    for any pair flags a C2-type anomaly.

    Per-window anomaly score = mean over all pairs of
        |dominant_lag_window - dominant_lag_train| / max_lag

    Parameters
    ----------
    window_size:
        Sliding window length (timesteps).  Must be > 2 * max_lag.
    max_lag:
        Maximum lag (in both directions) to search in the CCF.
    stride:
        Step between consecutive windows (1 = fully overlapping; higher = faster).
    """

    def __init__(
        self,
        window_size: int = 100,
        max_lag: int = 20,
        stride: int = 1,
    ) -> None:
        self.window_size = window_size
        self.max_lag = max_lag
        self.stride = stride

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dominant_lags(self, x: np.ndarray) -> np.ndarray:
        """Return (n_pairs,) array of dominant lags for a single segment."""
        T, C = x.shape
        pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.zeros(1)

        lags = np.arange(-self.max_lag, self.max_lag + 1)
        dominant = np.zeros(n_pairs)
        center = T - 1  # index of lag-0 in scipy full-mode output
        for k, (i, j) in enumerate(pairs):
            xi = x[:, i] - x[:, i].mean()
            xj = x[:, j] - x[:, j].mean()
            si = xi.std() or 1.0
            sj = xj.std() or 1.0
            # full CCF has length 2T-1; center = T-1 corresponds to lag 0
            full_ccf = _sig_correlate(xi, xj, mode="full") / (T * si * sj)
            ccf_slice = full_ccf[center - self.max_lag: center + self.max_lag + 1]
            dominant[k] = float(lags[np.argmax(np.abs(ccf_slice))])
        return dominant

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> "LagCorrelationDetector":
        if x.shape[1] < 2:
            raise ValueError("LagCorrelationDetector requires at least 2 channels.")
        self._train_lags = self._dominant_lags(x)   # baseline dominant lags
        self._T_train = x.shape[0]
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted("_train_lags", self)
        T, C = x.shape
        w = self.window_size

        scores = np.zeros(T)
        counts = np.zeros(T)

        for start in range(0, T - w + 1, self.stride):
            end = start + w
            seg = x[start:end]
            win_lags = self._dominant_lags(seg)
            # Mean absolute lag shift, normalised to [0, 1]
            shift = float(np.abs(win_lags - self._train_lags).mean()) / max(self.max_lag, 1)
            scores[start:end] += shift
            counts[start:end] += 1

        counts = np.where(counts == 0, 1, counts)
        return scores / counts
