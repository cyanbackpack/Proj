"""Tests for src/eval/metrics.py and src/eval/matrix.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.eval.metrics import (
    _apply_buffer,
    auc_pr,
    per_type_recall,
    recall_at_threshold,
    vus_pr,
)
from src.eval.matrix import RecallMatrix, build_from_results
from src.taxonomy.types import AnomalyType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T = 200
RNG = np.random.default_rng(0)


def perfect_scores(labels: np.ndarray) -> np.ndarray:
    """Scores that exactly match labels (perfect detector)."""
    return labels.astype(float)


def random_scores(T: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, T)


def make_labels(T: int, anomaly_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(T, dtype=int)
    n = int(T * anomaly_frac)
    start = T // 4
    labels[start : start + n] = 1
    return labels


# ---------------------------------------------------------------------------
# _apply_buffer
# ---------------------------------------------------------------------------

def test_buffer_zero_unchanged():
    labels = np.array([0, 0, 1, 0, 0])
    np.testing.assert_array_equal(_apply_buffer(labels, 0), labels)


def test_buffer_expands_anomaly():
    labels = np.array([0, 0, 1, 0, 0])
    result = _apply_buffer(labels, 1)
    expected = np.array([0, 1, 1, 1, 0])
    np.testing.assert_array_equal(result, expected)


def test_buffer_clips_at_boundaries():
    labels = np.array([1, 0, 0, 0, 1])
    result = _apply_buffer(labels, 2)
    # First anomaly at 0: expands left (clipped) to [0, 2]
    # Last anomaly at 4: expands right (clipped) to [2, 4]
    assert result[0] == 1
    assert result[2] == 1
    assert result[4] == 1


def test_buffer_all_zeros_unchanged():
    labels = np.zeros(10, dtype=int)
    np.testing.assert_array_equal(_apply_buffer(labels, 5), labels)


def test_buffer_all_ones_unchanged():
    labels = np.ones(10, dtype=int)
    np.testing.assert_array_equal(_apply_buffer(labels, 3), labels)


# ---------------------------------------------------------------------------
# auc_pr
# ---------------------------------------------------------------------------

def test_auc_pr_perfect_detector():
    labels = make_labels(T)
    scores = perfect_scores(labels)
    result = auc_pr(scores, labels)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_auc_pr_all_normal_labels():
    labels = np.zeros(T, dtype=int)
    assert auc_pr(random_scores(T), labels) == 0.0


def test_auc_pr_all_anomaly_labels():
    labels = np.ones(T, dtype=int)
    assert auc_pr(random_scores(T), labels) == 1.0


def test_auc_pr_range():
    labels = make_labels(T)
    result = auc_pr(random_scores(T), labels)
    assert 0.0 <= result <= 1.0


def test_auc_pr_random_worse_than_perfect():
    labels = make_labels(T)
    assert auc_pr(random_scores(T), labels) < auc_pr(perfect_scores(labels), labels)


# ---------------------------------------------------------------------------
# vus_pr
# ---------------------------------------------------------------------------

def test_vus_pr_perfect_detector():
    # A detector scoring exactly 1 at original anomaly positions gets AUC-PR=1
    # at buffer=0 but less at larger buffers (buffered positives are un-scored).
    # VUS-PR is defined as the average, so result < 1 with non-zero buffers.
    labels = make_labels(T)
    result_b0 = vus_pr(perfect_scores(labels), labels, buffer_sizes=[0])
    result_avg = vus_pr(perfect_scores(labels), labels, buffer_sizes=[0, 5, 10])
    assert result_b0 == pytest.approx(1.0, abs=1e-6)
    assert result_avg > 0.7  # still high, but not 1.0 due to buffer expansion


def test_vus_pr_all_normal():
    labels = np.zeros(T, dtype=int)
    assert vus_pr(random_scores(T), labels, buffer_sizes=[0, 5]) == 0.0


def test_vus_pr_range():
    labels = make_labels(T)
    result = vus_pr(random_scores(T), labels, buffer_sizes=[0, 5, 10, 20])
    assert 0.0 <= result <= 1.0


def test_vus_pr_larger_buffer_helps_near_perfect():
    """A detector that's slightly late benefits from larger buffers."""
    labels = make_labels(T, anomaly_frac=0.1)
    # Shift scores by 2 timesteps (slightly late)
    scores = np.zeros(T)
    scores[labels == 1] = 1.0
    scores = np.roll(scores, 3)  # late by 3 steps

    vus_small = vus_pr(scores, labels, buffer_sizes=[0])
    vus_large = vus_pr(scores, labels, buffer_sizes=[5])
    assert vus_large >= vus_small


def test_vus_pr_default_buffer_sizes():
    labels = make_labels(T)
    result = vus_pr(random_scores(T), labels)  # uses default [0, 5, 10, 20, 50]
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# recall_at_threshold
# ---------------------------------------------------------------------------

def test_recall_at_threshold_all_above():
    labels = make_labels(T)
    scores = np.ones(T)  # all flagged
    assert recall_at_threshold(scores, labels, threshold=0.5) == pytest.approx(1.0)


def test_recall_at_threshold_none_above():
    labels = make_labels(T)
    scores = np.zeros(T)  # none flagged
    assert recall_at_threshold(scores, labels, threshold=0.5) == pytest.approx(0.0)


def test_recall_at_threshold_perfect():
    labels = make_labels(T)
    scores = perfect_scores(labels)
    assert recall_at_threshold(scores, labels, threshold=0.5) == pytest.approx(1.0)


def test_recall_at_threshold_with_buffer():
    labels = np.zeros(T, dtype=int)
    labels[50] = 1  # single anomaly segment [50, 51)
    scores = np.zeros(T)
    scores[52] = 1.0  # detection at t=52 (2 steps late)
    # Without buffer: t=52 is outside the segment → segment not detected
    assert recall_at_threshold(scores, labels, threshold=0.5, buffer=0) == 0.0
    # With buffer=3: detection window for the segment is [47, 54), t=52 is inside → TP
    assert recall_at_threshold(scores, labels, threshold=0.5, buffer=3) == 1.0


# ---------------------------------------------------------------------------
# per_type_recall
# ---------------------------------------------------------------------------

def test_per_type_recall_keys():
    types = [AnomalyType.C1, AnomalyType.C2]
    labels = make_labels(T)
    scores_dict = {t.value: random_scores(T) for t in types}
    labels_dict = {t.value: labels for t in types}
    result = per_type_recall(scores_dict, labels_dict, threshold=0.5)
    assert set(result.keys()) == {t.value for t in types}


def test_per_type_recall_values_in_range():
    types = [AnomalyType.A1, AnomalyType.B1, AnomalyType.C1]
    labels = make_labels(T)
    scores_dict = {t.value: random_scores(T, i) for i, t in enumerate(types)}
    labels_dict = {t.value: labels for t in types}
    result = per_type_recall(scores_dict, labels_dict, threshold=0.5)
    for v in result.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# RecallMatrix
# ---------------------------------------------------------------------------

def test_recall_matrix_add_result():
    m = RecallMatrix()
    m.add_result("zscore", AnomalyType.C1, 0.42)
    assert m._results["zscore"]["C1"] == pytest.approx(0.42)


def test_recall_matrix_to_array_shape():
    m = RecallMatrix()
    for t in AnomalyType:
        m.add_result("model_a", t, 0.5)
        m.add_result("model_b", t, 0.7)
    arr, models, types = m.to_array()
    assert arr.shape == (2, 10)
    assert set(models) == {"model_a", "model_b"}
    assert len(types) == 10


def test_recall_matrix_missing_cell_is_nan():
    m = RecallMatrix()
    m.add_result("model_a", AnomalyType.C1, 0.5)
    # C2 not added for model_a
    arr, models, types = m.to_array()
    c2_idx = types.index("C2")
    assert np.isnan(arr[0, c2_idx])


def test_recall_matrix_save_creates_files():
    m = RecallMatrix()
    m.add_result("zscore", AnomalyType.C1, 0.42)
    m.add_result("zscore", AnomalyType.A1, 0.88)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = m.save(tmpdir)
        assert paths["json"].exists()
        assert paths["csv"].exists()


def test_recall_matrix_save_json_content():
    m = RecallMatrix()
    m.add_result("model_x", AnomalyType.B2, 0.65)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = m.save(tmpdir)
        with open(paths["json"]) as f:
            data = json.load(f)
        assert "results" in data
        assert data["results"]["model_x"]["B2"] == pytest.approx(0.65)


def test_build_from_results_roundtrip():
    m = RecallMatrix()
    m.add_result("det_a", AnomalyType.C1, 0.55)
    m.add_result("det_b", AnomalyType.C2, 0.33)
    with tempfile.TemporaryDirectory() as tmpdir:
        m.save(tmpdir)
        loaded = build_from_results(tmpdir)
    assert loaded._results["det_a"]["C1"] == pytest.approx(0.55)
    assert loaded._results["det_b"]["C2"] == pytest.approx(0.33)


def test_recall_matrix_evaluate():
    """Integration: fit ZScore and score an injected signal."""
    from src.models.baseline import ZScoreDetector
    from src.data.generator import VARGenerator

    gen = VARGenerator(n_channels=3, T=600, seed=0)
    x, _ = gen.generate()
    x_train = x[:400]
    x_test = x[400:]

    labels = np.zeros(200, dtype=int)
    labels[50:80] = 1

    matrix = RecallMatrix(buffer_sizes=[0, 5])
    matrix.evaluate(
        "zscore",
        ZScoreDetector(),
        x_train,
        {AnomalyType.C1: (x_test, labels)},
    )
    assert "zscore" in matrix._results
    assert "C1" in matrix._results["zscore"]
    assert 0.0 <= matrix._results["zscore"]["C1"] <= 1.0
