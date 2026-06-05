"""Evaluation metrics for anomaly detection.

Primary metric: VUS-PR (Volume Under the Surface — Precision-Recall).
Reference: TSB-AD, Liu & Paparrizos, NeurIPS 2024.

PA-F1 (point-adjusted F1) is explicitly excluded — it is known to
artificially inflate scores (Wu & Keogh, IEEE TKDE 2021).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


# ---------------------------------------------------------------------------
# Core metric: VUS-PR
# ---------------------------------------------------------------------------

def vus_pr(
    scores: np.ndarray,
    labels: np.ndarray,
    buffer_sizes: Optional[list[int]] = None,
) -> float:
    """Volume Under the PR Surface swept over a range of temporal buffers.

    For each buffer size *b*, anomaly labels are dilated by *b* timesteps on
    each side (so a detection within *b* steps of a true anomaly counts as a
    true positive).  The AUC-PR is computed for each *b* and then averaged.

    Parameters
    ----------
    scores:
        Per-timestep anomaly scores, shape (T,).  Higher = more anomalous.
    labels:
        Binary ground-truth labels, shape (T,).  1 = anomaly, 0 = normal.
    buffer_sizes:
        List of non-negative integers.  Default: [0, 5, 10, 20, 50].

    Returns
    -------
    float
        VUS-PR in [0, 1].
    """
    if buffer_sizes is None:
        buffer_sizes = [0, 5, 10, 20, 50]

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if labels.sum() == 0:
        return 0.0
    if labels.sum() == len(labels):
        return 1.0

    auc_values = [auc_pr(scores, _apply_buffer(labels, b)) for b in buffer_sizes]
    return float(np.mean(auc_values))


def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the Precision-Recall curve (no temporal buffer).

    Parameters
    ----------
    scores:
        Per-timestep anomaly scores, shape (T,).
    labels:
        Binary labels, shape (T,).

    Returns
    -------
    float
        AUC-PR in [0, 1].
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if labels.sum() == 0:
        return 0.0
    if labels.sum() == len(labels):
        return 1.0

    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


# ---------------------------------------------------------------------------
# Per-type breakdown helpers
# ---------------------------------------------------------------------------

def recall_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    buffer: int = 0,
) -> float:
    """Segment-level recall of a fixed-threshold detector.

    An anomaly *segment* (contiguous run of 1s in *labels*) is considered
    detected if at least one prediction at or above *threshold* falls within
    *buffer* timesteps of the segment.  This avoids penalising a near-perfect
    detector purely for not flagging every single buffered timestep.

    Parameters
    ----------
    scores:
        Per-timestep anomaly scores.
    labels:
        Binary ground-truth labels.
    threshold:
        Score threshold.
    buffer:
        Temporal buffer around each anomaly segment (timesteps).
    """
    labels = np.asarray(labels, dtype=int)
    preds = (np.asarray(scores) >= threshold).astype(int)
    segments = _find_segments(labels)
    if not segments:
        return 0.0
    detected = sum(
        1 for (s, e) in segments
        if preds[max(0, s - buffer) : min(len(preds), e + buffer)].any()
    )
    return detected / len(segments)


def per_type_recall(
    scores_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
    threshold: float,
    buffer: int = 0,
) -> dict[str, float]:
    """Compute per-type recall at a fixed threshold.

    Parameters
    ----------
    scores_dict:
        Mapping of anomaly_type_id → score array.
    labels_dict:
        Mapping of anomaly_type_id → label array.
    threshold:
        Detection threshold.
    buffer:
        Temporal buffer applied to each label set.

    Returns
    -------
    dict
        anomaly_type_id → recall value.
    """
    return {
        k: recall_at_threshold(scores_dict[k], labels_dict[k], threshold, buffer)
        for k in scores_dict
    }


# ---------------------------------------------------------------------------
# Buffer utility
# ---------------------------------------------------------------------------

def _find_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) for each contiguous run of 1s (end exclusive)."""
    segments = []
    in_seg = False
    start = 0
    for t, v in enumerate(labels):
        if v == 1 and not in_seg:
            start = t
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((start, t))
            in_seg = False
    if in_seg:
        segments.append((start, len(labels)))
    return segments


def _apply_buffer(labels: np.ndarray, buffer: int) -> np.ndarray:
    """Dilate binary label array by *buffer* timesteps on each side.

    Parameters
    ----------
    labels:
        Binary array of shape (T,).
    buffer:
        Number of timesteps to expand each anomaly region.

    Returns
    -------
    np.ndarray
        Dilated binary array, same shape as *labels*.
    """
    if buffer == 0:
        return labels.copy()

    T = len(labels)
    result = labels.copy().astype(int)
    # Find anomaly segment boundaries and expand
    in_anomaly = False
    for t in range(T):
        if labels[t] == 1:
            lo = max(0, t - buffer)
            hi = min(T, t + buffer + 1)
            result[lo:hi] = 1
    return result
