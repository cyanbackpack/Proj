"""Anomaly type definitions and marginal-validity checking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AnomalyLayer(str, Enum):
    MARGINAL = "marginal"       # (A) detectable from single-channel value
    TEMPORAL = "temporal"       # (B) requires within-channel temporal context
    INTER_METRIC = "inter_metric"  # (C) only detectable via cross-channel structure


class AnomalyType(str, Enum):
    # --- Marginal-visible ---
    A1 = "A1"  # Global Point
    A2 = "A2"  # Contextual Point

    # --- Temporal-context ---
    B1 = "B1"  # Seasonal
    B2 = "B2"  # Trend
    B3 = "B3"  # Shapelet

    # --- Inter-metric (joint-only-visible) ---
    C1 = "C1"  # Correlation-break
    C2 = "C2"  # Phase/lag-shift
    C3 = "C3"  # Ratio anomaly
    C4 = "C4"  # Group-collective
    C5 = "C5"  # Causal/precedence violation


@dataclass(frozen=True)
class AnomalyMeta:
    type_id: AnomalyType
    name: str
    layer: AnomalyLayer
    key_property: str
    marginal_in_distribution: bool  # True → C-type invariant applies


ANOMALY_REGISTRY: dict[AnomalyType, AnomalyMeta] = {
    AnomalyType.A1: AnomalyMeta(
        type_id=AnomalyType.A1,
        name="Global Point",
        layer=AnomalyLayer.MARGINAL,
        key_property="Single-channel extreme value",
        marginal_in_distribution=False,
    ),
    AnomalyType.A2: AnomalyMeta(
        type_id=AnomalyType.A2,
        name="Contextual Point",
        layer=AnomalyLayer.MARGINAL,
        key_property="In-range globally, anomalous in local context",
        marginal_in_distribution=False,  # borderline; treated as out-of-dist locally
    ),
    AnomalyType.B1: AnomalyMeta(
        type_id=AnomalyType.B1,
        name="Seasonal",
        layer=AnomalyLayer.TEMPORAL,
        key_property="Phase/amplitude of periodicity breaks",
        marginal_in_distribution=True,
    ),
    AnomalyType.B2: AnomalyMeta(
        type_id=AnomalyType.B2,
        name="Trend",
        layer=AnomalyLayer.TEMPORAL,
        key_property="Slope change or level shift over time",
        marginal_in_distribution=True,
    ),
    AnomalyType.B3: AnomalyMeta(
        type_id=AnomalyType.B3,
        name="Shapelet",
        layer=AnomalyLayer.TEMPORAL,
        key_property="Unusual local subsequence shape",
        marginal_in_distribution=True,
    ),
    AnomalyType.C1: AnomalyMeta(
        type_id=AnomalyType.C1,
        name="Correlation-break",
        layer=AnomalyLayer.INTER_METRIC,
        key_property="Normally-correlated channels decouple (sign/magnitude change)",
        marginal_in_distribution=True,
    ),
    AnomalyType.C2: AnomalyMeta(
        type_id=AnomalyType.C2,
        name="Phase/lag-shift",
        layer=AnomalyLayer.INTER_METRIC,
        key_property="Leading-lagging relationship shifts in time",
        marginal_in_distribution=True,
    ),
    AnomalyType.C3: AnomalyMeta(
        type_id=AnomalyType.C3,
        name="Ratio anomaly",
        layer=AnomalyLayer.INTER_METRIC,
        key_property="Both channels move but their ratio breaks",
        marginal_in_distribution=True,
    ),
    AnomalyType.C4: AnomalyMeta(
        type_id=AnomalyType.C4,
        name="Group-collective",
        layer=AnomalyLayer.INTER_METRIC,
        key_property="Coherent regime shift in a channel subset, others normal",
        marginal_in_distribution=True,
    ),
    AnomalyType.C5: AnomalyMeta(
        type_id=AnomalyType.C5,
        name="Causal/precedence violation",
        layer=AnomalyLayer.INTER_METRIC,
        key_property="Known A→B causal order violated (B fires without A)",
        marginal_in_distribution=True,
    ),
}


def get_meta(anomaly_type: AnomalyType) -> AnomalyMeta:
    return ANOMALY_REGISTRY[anomaly_type]


def marginal_check(
    x_orig: np.ndarray,
    x_injected: np.ndarray,
    window: tuple[int, int],
    z_threshold: float = 2.5,
) -> bool:
    """Return True if the injection does not introduce new marginal extremes.

    A channel in the injected window passes if its max |z-score| is ≤
    max(z_threshold, original_window_max_z).  This allows naturally occurring
    values above z_threshold that were already present in x_orig[window] to
    remain, while still catching injections that create genuinely new spikes.

    Parameters
    ----------
    x_orig:
        Original (clean) signal, shape (T, C).
    x_injected:
        Signal after injection, shape (T, C).
    window:
        (start, end) indices of the injected anomaly window (end exclusive).
    z_threshold:
        Hard cap on NEW z-score exceedances (default 2.5σ).
    """
    if x_orig.ndim == 1:
        x_orig = x_orig[:, np.newaxis]
        x_injected = x_injected[:, np.newaxis]

    start, end = window
    mu = x_orig.mean(axis=0)
    sigma = x_orig.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)  # guard against constant channels

    z_orig_win = np.abs((x_orig[start:end] - mu) / sigma).max(axis=0)
    z_inj_win = np.abs((x_injected[start:end] - mu) / sigma).max(axis=0)
    # Effective threshold: whichever is larger, the hard cap or what was naturally there
    effective = np.maximum(z_threshold, z_orig_win)
    return bool((z_inj_win <= effective).all())


def marginal_check_report(
    x_orig: np.ndarray,
    x_injected: np.ndarray,
    window: tuple[int, int],
    z_threshold: float = 2.5,
) -> dict:
    """Like marginal_check but returns a dict with per-channel max z-scores."""
    if x_orig.ndim == 1:
        x_orig = x_orig[:, np.newaxis]
        x_injected = x_injected[:, np.newaxis]

    start, end = window
    mu = x_orig.mean(axis=0)
    sigma = x_orig.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)

    z_orig_win = np.abs((x_orig[start:end] - mu) / sigma).max(axis=0)
    z_inj_win = np.abs((x_injected[start:end] - mu) / sigma).max(axis=0)
    effective = np.maximum(z_threshold, z_orig_win)
    per_channel_max = z_inj_win.tolist()
    passes = bool((z_inj_win <= effective).all())

    return {
        "passes": passes,
        "z_threshold": z_threshold,
        "per_channel_max_z": per_channel_max,
        "per_channel_orig_max_z": z_orig_win.tolist(),
        "window": window,
    }
