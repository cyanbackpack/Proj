"""Anomaly injectors: one class per anomaly type, all inheriting BaseInjector.

Layer A (Marginal-visible): A1, A2 — deliberately violate marginal invariant.
Layer B (Temporal-context): B1, B2, B3 — may or may not change marginals.
Layer C (Inter-metric, joint-only-visible): C1–C5 — marginal invariant enforced.

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


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseInjector(abc.ABC):
    """Abstract base for all anomaly injectors.

    Parameters
    ----------
    anomaly_type:
        The AnomalyType this injector produces (set as class attribute).
    z_threshold:
        Maximum allowed per-channel z-score inside the injection window,
        used by :meth:`verify_marginal` and :meth:`inject_and_verify`.
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


# ===========================================================================
# Layer C — Inter-metric (joint-only-visible)
# ===========================================================================

class C1CorrelationBreakInjector(BaseInjector):
    """Destroy cross-channel correlation using IAAFT Fourier phase-randomization.

    Each target channel's amplitude spectrum is preserved while Fourier phases
    are randomized, eliminating joint structure with no marginal change.

    Parameters
    ----------
    target_channels:
        Channels to randomize.  None → all channels.
    n_iter:
        IAAFT iterations (8 is sufficient for typical signals).
    seed:
        RNG seed.
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
        rng = np.random.default_rng(self.seed)
        x_inj = x.copy()
        start, end = window
        targets = self.target_channels if self.target_channels is not None else list(range(x.shape[1]))
        for ch in targets:
            x_inj[start:end, ch] = _iaaft_surrogate(x[start:end, ch], n_iter=self.n_iter, rng=rng)
        return x_inj


class C2PhaseShiftInjector(BaseInjector):
    """Shift the lead-lag relationship between channels by τ timesteps.

    A circular roll of ``lag_channel``'s segment by ``tau`` steps breaks
    the temporal alignment while keeping exact marginal values (the values
    are simply re-ordered, not changed).

    Parameters
    ----------
    lag_channel:
        Index of the channel whose phase is shifted.
    tau:
        Shift amount in timesteps (positive = forward shift).
    """

    anomaly_type = AnomalyType.C2

    def __init__(
        self,
        lag_channel: int = 1,
        tau: int = 10,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.lag_channel = lag_channel
        self.tau = tau

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        x_inj[start:end, self.lag_channel] = np.roll(x[start:end, self.lag_channel], self.tau)
        return x_inj


class C3RatioAnomalyInjector(BaseInjector):
    """Break the ratio relationship between two channels via amplitude-flip.

    Normally channel A and channel B maintain a stable ratio A/B ≈ k.
    The amplitude-flip reflects channel A's window values around their local
    mean: x_inj = 2·μ_win − x.  This preserves the exact value distribution
    of channel A (same mean, std, and set of values, just reversed ordering)
    while inverting the co-movement with channel B, breaking the ratio.

    Parameters
    ----------
    channel_pair:
        (ch_a, ch_b) — ch_a is flipped; ch_b is unchanged.
    """

    anomaly_type = AnomalyType.C3

    def __init__(
        self,
        channel_pair: tuple[int, int] = (0, 1),
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.channel_pair = channel_pair

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        ch_a, _ = self.channel_pair
        segment = x[start:end, ch_a]
        # Rank-reverse: assign the k-th largest value to the position that
        # previously held the k-th smallest, and vice versa.  Uses exactly the
        # same set of values as the original segment → global z-scores unchanged.
        rank_asc = np.argsort(np.argsort(segment))   # rank of each position
        sorted_vals = np.sort(segment)
        x_inj[start:end, ch_a] = sorted_vals[::-1][rank_asc]  # reverse the sorted array
        return x_inj


class C4GroupCollectiveInjector(BaseInjector):
    """Coherent regime shift in a channel subset; other channels stay normal.

    The group channels' window is replaced with values borrowed from a
    donor position elsewhere in the same signal.  Each channel's values
    look individually normal (drawn from the same signal), but the group's
    joint state is no longer aligned with the rest of the channels.

    Parameters
    ----------
    group_channels:
        Indices of the channels that shift together.  None → first half.
    donor_offset:
        Timestep offset used to select the donor segment.  The donor
        window starts at ``start + donor_offset`` (wraps if needed).
    """

    anomaly_type = AnomalyType.C4

    def __init__(
        self,
        group_channels: Optional[list[int]] = None,
        donor_offset: int = 200,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.group_channels = group_channels
        self.donor_offset = donor_offset

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        T = x.shape[0]
        win_len = end - start
        group = self.group_channels if self.group_channels is not None else list(range(x.shape[1] // 2))

        donor_start = (start + self.donor_offset) % T
        donor_end = donor_start + win_len
        idx = np.arange(donor_start, donor_start + win_len) % T

        for ch in group:
            donor_vals = x[idx, ch]
            orig_sorted = np.sort(x[start:end, ch])
            # Apply donor's rank ordering to original window's sorted values.
            # Each position gets the value that has the same rank as the donor —
            # the temporal pattern comes from the donor, but the value set is
            # identical to the original window, so global z-scores are unchanged.
            donor_rank = np.argsort(np.argsort(donor_vals))
            x_inj[start:end, ch] = orig_sorted[donor_rank]

        return x_inj


class C5CausalViolationInjector(BaseInjector):
    """Causal/precedence violation: effect channel fires without its cause.

    The cause channel is replaced with baseline noise (drawn from that
    channel's marginal distribution) while the effect channel is left
    unchanged.  This creates a situation where the effect appears without
    a preceding cause signal.

    Parameters
    ----------
    cause_channel:
        Index of the causal (A) channel to suppress.
    effect_channel:
        Index of the effect (B) channel (kept unchanged).
    seed:
        RNG seed for baseline noise generation.
    """

    anomaly_type = AnomalyType.C5

    def __init__(
        self,
        cause_channel: int = 0,
        effect_channel: int = 1,
        seed: int = 0,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.cause_channel = cause_channel
        self.effect_channel = effect_channel
        self.seed = seed

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        x_inj = x.copy()
        start, end = window
        ch = self.cause_channel
        mu = x[:, ch].mean()
        sigma = x[:, ch].std()
        # Replace cause channel with noise that looks normal but carries no signal
        x_inj[start:end, ch] = rng.normal(mu, sigma, end - start)
        return x_inj


# ===========================================================================
# Layer A — Marginal-visible
# ===========================================================================

class A1GlobalPointInjector(BaseInjector):
    """Inject extreme-value point anomalies (global outliers).

    Adds a large spike of ``n_sigma * global_std`` to ``target_channel``
    at regularly-spaced positions within the window.  Deliberately violates
    the marginal invariant (that is the point).

    Parameters
    ----------
    target_channel:
        Channel to spike.
    n_sigma:
        Spike magnitude in standard deviations (default 5).
    n_points:
        Number of spike positions inside the window.
    """

    anomaly_type = AnomalyType.A1

    def __init__(
        self,
        target_channel: int = 0,
        n_sigma: float = 5.0,
        n_points: int = 1,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.target_channel = target_channel
        self.n_sigma = n_sigma
        self.n_points = n_points

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        ch = self.target_channel
        spike = self.n_sigma * x[:, ch].std()
        positions = np.linspace(start, end - 1, self.n_points, dtype=int)
        x_inj[positions, ch] += spike
        return x_inj


class A2ContextualPointInjector(BaseInjector):
    """Inject contextual point anomalies: normal globally, anomalous locally.

    Within the window the channel is at a local mean M.  The injection
    replaces a subset of points with values at the opposite extreme:
    global_mean + amplitude_factor * (global_mean - local_mean),
    keeping values within the global range while being unexpected locally.

    Parameters
    ----------
    target_channel:
        Channel to perturb.
    amplitude_factor:
        Controls how far from the local mean the injected value is placed.
    n_points:
        Number of anomalous points in the window.
    seed:
        RNG seed for point selection.
    """

    anomaly_type = AnomalyType.A2

    def __init__(
        self,
        target_channel: int = 0,
        amplitude_factor: float = 2.5,
        n_points: int = 3,
        seed: int = 0,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.target_channel = target_channel
        self.amplitude_factor = amplitude_factor
        self.n_points = n_points
        self.seed = seed

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        x_inj = x.copy()
        start, end = window
        ch = self.target_channel
        global_mean = x[:, ch].mean()
        local_mean = x[start:end, ch].mean()
        # Contextual anomaly: flip the local mean direction
        anomaly_value = global_mean + self.amplitude_factor * (global_mean - local_mean)
        win_len = end - start
        positions = rng.choice(win_len, size=min(self.n_points, win_len), replace=False)
        x_inj[start + positions, ch] = anomaly_value
        return x_inj


# ===========================================================================
# Layer B — Temporal-context
# ===========================================================================

class B1SeasonalInjector(BaseInjector):
    """Break the seasonal pattern of a channel by phase-shifting it.

    The channel's window is replaced with values taken from a segment
    shifted by ``phase_shift`` timesteps, simulating a phase anomaly
    in a periodic signal.  Uses circular indexing to avoid boundary issues.

    Parameters
    ----------
    target_channel:
        Channel whose seasonal pattern is disrupted.
    phase_shift:
        Timestep offset for the replacement segment (default: half a period).
    """

    anomaly_type = AnomalyType.B1

    def __init__(
        self,
        target_channel: int = 0,
        phase_shift: int = 25,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        self.target_channel = target_channel
        self.phase_shift = phase_shift

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        T = x.shape[0]
        win_len = end - start
        ch = self.target_channel
        idx = np.arange(start + self.phase_shift, start + self.phase_shift + win_len) % T
        x_inj[start:end, ch] = x[idx, ch]
        return x_inj


class B2TrendInjector(BaseInjector):
    """Inject a level shift (step) or slope change (ramp) into one channel.

    Parameters
    ----------
    target_channel:
        Channel to perturb.
    trend_type:
        ``'ramp'`` adds a linear drift; ``'step'`` adds a DC level shift.
    slope:
        For ``'ramp'``: change per timestep (units of signal std).
        For ``'step'``: constant offset (units of signal std).
    """

    anomaly_type = AnomalyType.B2

    def __init__(
        self,
        target_channel: int = 0,
        trend_type: str = "ramp",
        slope: float = 0.05,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        if trend_type not in ("ramp", "step"):
            raise ValueError(f"trend_type must be 'ramp' or 'step', got '{trend_type}'")
        self.target_channel = target_channel
        self.trend_type = trend_type
        self.slope = slope

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        ch = self.target_channel
        scale = x[:, ch].std()
        win_len = end - start
        if self.trend_type == "ramp":
            drift = np.arange(win_len) * self.slope * scale
        else:  # step
            drift = np.full(win_len, self.slope * scale)
        x_inj[start:end, ch] += drift
        return x_inj


class B3ShapeletInjector(BaseInjector):
    """Replace a channel's window with an unusual local subsequence (shapelet).

    Parameters
    ----------
    target_channel:
        Channel to replace.
    shape:
        ``'bump'`` — Gaussian bump; ``'plateau'`` — flat top; ``'zigzag'`` —
        alternating high/low pattern.
    amplitude:
        Peak amplitude of the shapelet in units of signal std.
    """

    anomaly_type = AnomalyType.B3

    _VALID_SHAPES = ("bump", "plateau", "zigzag")

    def __init__(
        self,
        target_channel: int = 0,
        shape: str = "bump",
        amplitude: float = 2.0,
        z_threshold: float = 2.5,
    ) -> None:
        super().__init__(z_threshold=z_threshold)
        if shape not in self._VALID_SHAPES:
            raise ValueError(f"shape must be one of {self._VALID_SHAPES}, got '{shape}'")
        self.target_channel = target_channel
        self.shape = shape
        self.amplitude = amplitude

    def inject(self, x: np.ndarray, window: tuple[int, int]) -> np.ndarray:
        x_inj = x.copy()
        start, end = window
        ch = self.target_channel
        win_len = end - start
        scale = x[:, ch].std()
        t = np.linspace(-1, 1, win_len)

        if self.shape == "bump":
            pattern = self.amplitude * scale * np.exp(-4 * t ** 2)
        elif self.shape == "plateau":
            pattern = np.full(win_len, self.amplitude * scale)
        else:  # zigzag
            pattern = self.amplitude * scale * np.sign(np.sin(4 * np.pi * t))

        local_mean = x[start:end, ch].mean()
        x_inj[start:end, ch] = local_mean + pattern
        return x_inj


# ===========================================================================
# IAAFT surrogate helper
# ===========================================================================

def _iaaft_surrogate(
    x: np.ndarray,
    n_iter: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Iterated Amplitude-Adjusted Fourier Transform (IAAFT) surrogate.

    Produces a surrogate that has (approximately) the same amplitude spectrum
    as *x*, the same rank-order distribution, and randomized Fourier phases.
    Ending on rank-adjust guarantees exact marginal distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(x)
    x_sorted = np.sort(x)
    x_fft_amp = np.abs(np.fft.rfft(x))

    surrogate = rng.permutation(x)

    for _ in range(n_iter):
        # Phase-adjust: impose target amplitude spectrum, keep current phases
        s_fft = np.fft.rfft(surrogate)
        phases = np.angle(s_fft)
        surrogate = np.fft.irfft(x_fft_amp * np.exp(1j * phases), n=N)
        # Rank-adjust: impose exact marginal distribution
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = x_sorted[rank_order]

    # Final state is rank-adjusted → values are a permutation of x's values.
    return surrogate.astype(x.dtype)


# ===========================================================================
# Registry: anomaly_type → injector class
# ===========================================================================

INJECTOR_REGISTRY: dict[AnomalyType, type[BaseInjector]] = {
    AnomalyType.A1: A1GlobalPointInjector,
    AnomalyType.A2: A2ContextualPointInjector,
    AnomalyType.B1: B1SeasonalInjector,
    AnomalyType.B2: B2TrendInjector,
    AnomalyType.B3: B3ShapeletInjector,
    AnomalyType.C1: C1CorrelationBreakInjector,
    AnomalyType.C2: C2PhaseShiftInjector,
    AnomalyType.C3: C3RatioAnomalyInjector,
    AnomalyType.C4: C4GroupCollectiveInjector,
    AnomalyType.C5: C5CausalViolationInjector,
}
