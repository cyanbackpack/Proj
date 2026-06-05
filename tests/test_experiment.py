"""Integration tests for src/experiments/run_experiment.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.experiments.run_experiment import (
    _build_detector,
    _inject_and_label,
    _make_base_signal,
    run_experiment,
)
from src.taxonomy.types import AnomalyType


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

MIN_CFG = {
    "seed": 0,
    "data": {
        "base_path": "data/synthetic/base_0.npz",
        "train_ratio": 0.6,
        "window_size": 20,
    },
    "var": {
        "n_channels": 3,
        "lag_order": 2,
        "T": 300,
        "spectral_radius": 0.7,
        "noise_scale": 1.0,
    },
    "injection": {
        "window_size": 30,
        "n_windows": 2,
    },
    "models": {
        "moving_average": {"window": 10},
        "knn": {"n_neighbors": 3},
        "lof": {"n_neighbors": 5},
    },
    "evaluation": {
        "buffer_sizes": [0, 5],
        "output_dir": "results/",
    },
}


# ---------------------------------------------------------------------------
# _build_detector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["zscore", "moving_average", "knn", "lof"])
def test_build_detector_returns_base_detector(name):
    from src.models.base import BaseDetector
    det = _build_detector(name, MIN_CFG)
    assert isinstance(det, BaseDetector)


def test_build_detector_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        _build_detector("foo", MIN_CFG)


# ---------------------------------------------------------------------------
# _make_base_signal
# ---------------------------------------------------------------------------

def test_make_base_signal_generates_when_missing(tmp_path):
    cfg = {**MIN_CFG, "data": {**MIN_CFG["data"], "base_path": str(tmp_path / "base.npz")}}
    x = _make_base_signal(cfg)
    assert x.shape == (300, 3)
    assert np.isfinite(x).all()


def test_make_base_signal_loads_existing(tmp_path):
    arr = np.ones((100, 2))
    path = tmp_path / "base.npz"
    np.savez(path, x=arr)
    cfg = {**MIN_CFG, "data": {**MIN_CFG["data"], "base_path": str(path)}}
    x = _make_base_signal(cfg)
    np.testing.assert_array_equal(x, arr)


# ---------------------------------------------------------------------------
# _inject_and_label
# ---------------------------------------------------------------------------

def test_inject_and_label_output_shapes():
    from src.data.generator import VARGenerator
    gen = VARGenerator(n_channels=3, T=300, seed=0)
    x, _ = gen.generate()
    x_inj, labels = _inject_and_label(x, AnomalyType.C1, MIN_CFG["injection"], seed=0)
    assert x_inj.shape == x.shape
    assert labels.shape == (300,)
    assert set(labels).issubset({0, 1})


def test_inject_and_label_has_anomalies():
    from src.data.generator import VARGenerator
    gen = VARGenerator(n_channels=3, T=300, seed=0)
    x, _ = gen.generate()
    _, labels = _inject_and_label(x, AnomalyType.A1, MIN_CFG["injection"], seed=0)
    assert labels.sum() > 0


@pytest.mark.parametrize("atype", [AnomalyType.C1, AnomalyType.C3, AnomalyType.A1, AnomalyType.B2])
def test_inject_and_label_all_types(atype):
    from src.data.generator import VARGenerator
    gen = VARGenerator(n_channels=4, T=400, seed=0)
    x, _ = gen.generate()
    x_inj, labels = _inject_and_label(x, atype, MIN_CFG["injection"], seed=0)
    assert x_inj.shape == x.shape
    assert labels.sum() > 0


# ---------------------------------------------------------------------------
# run_experiment (integration)
# ---------------------------------------------------------------------------

def test_run_experiment_returns_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            **MIN_CFG,
            "data": {**MIN_CFG["data"], "base_path": f"{tmpdir}/base.npz"},
            "evaluation": {**MIN_CFG["evaluation"], "output_dir": tmpdir},
        }
        results = run_experiment("zscore", [AnomalyType.C1, AnomalyType.A1], cfg)
    assert "C1" in results
    assert "A1" in results
    assert all(0.0 <= v <= 1.0 for v in results.values())


def test_run_experiment_knn_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            **MIN_CFG,
            "data": {**MIN_CFG["data"], "base_path": f"{tmpdir}/base.npz"},
            "evaluation": {**MIN_CFG["evaluation"], "output_dir": tmpdir},
        }
        results = run_experiment("knn", [AnomalyType.C2], cfg)
    assert "C2" in results
    assert 0.0 <= results["C2"] <= 1.0
