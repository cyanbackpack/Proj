"""Anomaly injectors: one class per anomaly type, all inheriting BaseInjector.

C1 (Correlation-break) uses IAAFT Fourier phase-randomization:
  - Preserves each channel's marginal power spectrum
  - Destroys cross-channel correlation within the injection window
  - Post-injection marginal_check enforced automatically
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from src.taxonomy.types import AnomalyType, marginal_check, marginal_check_report


class MarginalViolationError(Exception):
    """Raised when an injection violates the C-type marginal invariant."""


class BaseInjector(abc.ABC):
    """Abstract base for all anomaly injectors.

    Subclasses must implement :meth:`inject`.  The marginal check is applied
    automatically for types where ``marginal_in_distribution=True`` via
    :meth:`inject_and_verify`.

    Parameters
    ----------
    anomaly_type:
        The AnomalyType this injector produces.
    z_threshold:
        Maximum allowed per-channel z-score inside the injection window.
    """

    anomaly_type: AnomalyType  # must be set by each subclass

    def __init__(self, z_threshold: float = 2.5) -> None:
        self.z_threshold = z_threshold

    @abc.abstractmethod
    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        """Return a copy of *x* with the anomaly injected in *window*.

        Parameters
        ----------
        x:
            Clean signal, shape (T, C).
        window:
            (start, end) — half-open interval, end exclusive.

        Returns
        -------
        np.ndarray
            Modified signal, same shape as *x*.
        """

    def verify_marginal(
        self,
        x_orig: np.ndarray,
        x_injected: np.ndarray,
        window: tuple[int, int],
    ) -> bool:
        """Return True if the injection respects the marginal invariant."""
        return marginal_check(x_orig, x_injected, window, self.z_threshold)

    def inject_and_verify(
        self,
        x: np.ndarray,
        window: tuple[int, int],
        *,
        raise_on_violation: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Inject, then run the marginal check.

        Returns
        -------
        x_inj:
            Injected signal.
        report:
            Output of :func:`marginal_check_report`.

        Raises
        ------
        MarginalViolationError
            If *raise_on_violation* is True and the check fails.
        """
        x_inj = self.inject(x, window)
        report = marginal_check_report(x, x_inj, window, self.z_threshold)
        if raise_on_violation and not report["passes"]:
            raise MarginalViolationError(
                f"{self.anomaly_type} injection violated marginal invariant "
                f"(z_threshold={self.z_threshold}): {report}"
            )
        return x_inj, report


# ---------------------------------------------------------------------------
# C1: Correlation-break via IAAFT Fourier phase-randomization
# ---------------------------------------------------------------------------

class C1CorrelationBreakInjector(BaseInjector):
    """Destroy cross-channel correlation in a window using IAAFT surrogates.

    For each target channel the amplitude spectrum is preserved exactly while
    the Fourier phases inside the window are randomized.  This leaves each
    channel's marginal power spectrum (and therefore its z-score distribution)
    unchanged while eliminating the joint structure.

    Parameters
    ----------
    target_channels:
        Indices of channels whose phases will be randomized.  If None, all
        channels are targeted.
    n_iter:
        IAAFT iterations.  More iterations = closer marginal preservation.
        8 iterations is sufficient for typical use.
    seed:
        RNG seed for phase randomization.
    z_threshold:
        Marginal-check threshold (default 2.5σ).
    """

    anomaly_type = AnomalyType.C1

    def __init__(
        self,
        target_channels: Optional[list[int]] = None,
        n_iter: int = 8,
        seed: int = 0,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.target_channels = target_channels
        self.n_iter = n_iter
        self.seed = seed

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        """Return x with IAAFT surrogate replacing target channels in *window*."""
        rng = np.random.default_rng(self.seed)
        x_inj = x.copy()
        start, end = window
        targets = self.target_channels if self.target_channels is not None else list(range(x.shape[1]))

        for ch in targets:
            segment = x[start:end, ch]
            surrogate = _iaaft_surrogate(segment, n_iter=self.n_iter, rng=rng)
            x_inj[start:end, ch] = surrogate

        return x_inj


# ---------------------------------------------------------------------------
# IAAFT surrogate implementation
# ---------------------------------------------------------------------------

def _iaaft_surrogate(
    x: np.ndarray,
    n_iter: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Iterated Amplitude-Adjusted Fourier Transform (IAAFT) surrogate.

    Produces a surrogate of *x* that:
    - Has (approximately) the same amplitude spectrum as *x*
    - Has the same rank-order distribution as *x*
    - Has randomized Fourier phases → destroyed temporal/cross-channel structure

    Parameters
    ----------
    x:
        1-D real signal of length N.
    n_iter:
        Number of IAAFT refinement iterations.
    rng:
        Random number generator.

    Returns
    -------
    np.ndarray
        Surrogate signal, same shape and dtype as *x*.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(x)
    x_sorted = np.sort(x)
    x_fft_amp = np.abs(np.fft.rfft(x))  # target amplitude spectrum

    # Start from a random shuffle
    surrogate = rng.permutation(x)

    for _ in range(n_iter):
        # Phase-adjust: impose target amplitude spectrum, keep current phases
        s_fft = np.fft.rfft(surrogate)
        phases = np.angle(s_fft)
        s_fft_adjusted = x_fft_amp * np.exp(1j * phases)
        surrogate = np.fft.irfft(s_fft_adjusted, n=N)

        # Rank-adjust: impose target rank-order distribution (marginal preservation)
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = x_sorted[rank_order]

    # End on rank-adjust: guarantees exact same marginal distribution as input.
    # Values in the surrogate are a permutation of values in x, so per-channel
    # z-scores cannot exceed the global max z-score of the original signal.
    return surrogate.astype(x.dtype)
