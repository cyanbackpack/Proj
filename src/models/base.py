"""Abstract base class for all anomaly detectors."""

from __future__ import annotations

import abc

import numpy as np


class BaseDetector(abc.ABC):
    """Common interface for all anomaly detectors.

    Detectors are trained on **normal** data (no anomalies), then applied to
    test sequences.  The :meth:`score` method returns a 1-D array of anomaly
    scores — one value per timestep — where higher means more anomalous.

    Subclasses must implement :meth:`fit` and :meth:`score`.
    """

    @abc.abstractmethod
    def fit(self, x: np.ndarray) -> "BaseDetector":
        """Train the detector on normal data.

        Parameters
        ----------
        x:
            Training signal, shape (T, C).

        Returns
        -------
        self
        """

    @abc.abstractmethod
    def score(self, x: np.ndarray) -> np.ndarray:
        """Return per-timestep anomaly scores.

        Parameters
        ----------
        x:
            Test signal, shape (T, C).

        Returns
        -------
        np.ndarray
            Shape (T,).  Higher = more anomalous.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_fitted(attr: str, obj: object) -> None:
        if not hasattr(obj, attr):
            raise RuntimeError(f"Call fit() before score(). Missing attribute: '{attr}'")

    @staticmethod
    def _sliding_windows(x: np.ndarray, window_size: int) -> np.ndarray:
        """Return shape (T - W + 1, W * C) sliding windows (row-major flatten)."""
        T, C = x.shape
        n = T - window_size + 1
        idx = np.arange(window_size)[None, :] + np.arange(n)[:, None]  # (n, W)
        return x[idx].reshape(n, window_size * C)

    @staticmethod
    def _expand_scores(window_scores: np.ndarray, T: int, window_size: int) -> np.ndarray:
        """Map per-window scores back to per-timestep scores.

        The score for window starting at t is assigned to all timesteps in
        [t, t + window_size).  Overlapping windows are averaged.
        """
        n = len(window_scores)
        counts = np.zeros(T)
        scores = np.zeros(T)
        for i, s in enumerate(window_scores):
            scores[i : i + window_size] += s
            counts[i : i + window_size] += 1
        counts = np.where(counts == 0, 1, counts)
        return scores / counts
