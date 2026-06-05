"""Tests for src/data/generator.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.generator import VARGenerator, _build_companion, _spectral_radius


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def make_gen(**kwargs) -> VARGenerator:
    defaults = dict(n_channels=4, lag_order=2, T=500, spectral_radius=0.7, seed=0)
    defaults.update(kwargs)
    return VARGenerator(**defaults)


# ------------------------------------------------------------------
# Output shape and type
# ------------------------------------------------------------------

def test_output_shape():
    gen = make_gen(n_channels=3, T=300)
    x, meta = gen.generate()
    assert x.shape == (300, 3)


def test_output_dtype():
    x, _ = make_gen().generate()
    assert x.dtype == np.float64


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def test_same_seed_same_output():
    x1, _ = make_gen(seed=7).generate()
    x2, _ = make_gen(seed=7).generate()
    np.testing.assert_array_equal(x1, x2)


def test_different_seeds_different_output():
    x1, _ = make_gen(seed=1).generate()
    x2, _ = make_gen(seed=2).generate()
    assert not np.allclose(x1, x2)


# ------------------------------------------------------------------
# Stability (stationarity)
# ------------------------------------------------------------------

def test_spectral_radius_below_one():
    """Generated coefficient matrices must yield a stable VAR."""
    gen = make_gen(spectral_radius=0.6)
    _, meta = gen.generate()
    coef = [np.array(A) for A in meta["coef_matrices"]]
    companion = _build_companion(coef)
    sr = _spectral_radius(companion)
    assert sr < 1.0, f"spectral radius {sr:.4f} >= 1 — process is non-stationary"


def test_spectral_radius_approximately_target():
    gen = make_gen(spectral_radius=0.75)
    _, meta = gen.generate()
    coef = [np.array(A) for A in meta["coef_matrices"]]
    companion = _build_companion(coef)
    sr = _spectral_radius(companion)
    assert abs(sr - 0.75) < 0.05, f"spectral radius {sr:.4f} too far from target 0.75"


def test_invalid_spectral_radius_raises():
    with pytest.raises(ValueError, match="spectral_radius must be < 1.0"):
        VARGenerator(spectral_radius=1.0)


# ------------------------------------------------------------------
# Signal statistics (approximate)
# ------------------------------------------------------------------

def test_signal_mean_near_zero():
    """Stationary VAR(p) has zero mean for zero-mean innovations."""
    gen = make_gen(T=5000, seed=99)
    x, _ = gen.generate()
    assert np.abs(x.mean(axis=0)).max() < 0.5


def test_signal_finite():
    x, _ = make_gen().generate()
    assert np.isfinite(x).all()


# ------------------------------------------------------------------
# Meta dict completeness
# ------------------------------------------------------------------

def test_meta_keys():
    _, meta = make_gen().generate()
    for key in ("seed", "n_channels", "lag_order", "T", "coef_matrices", "sigma_noise"):
        assert key in meta, f"Missing key: {key}"


def test_meta_coef_shape():
    gen = make_gen(n_channels=4, lag_order=3)
    _, meta = gen.generate()
    assert len(meta["coef_matrices"]) == 3
    for A in meta["coef_matrices"]:
        assert np.array(A).shape == (4, 4)


# ------------------------------------------------------------------
# Custom coefficient matrices
# ------------------------------------------------------------------

def test_custom_coef_matrices():
    C = 3
    A1 = np.eye(C) * 0.3
    A2 = np.eye(C) * 0.1
    gen = VARGenerator(n_channels=C, lag_order=2, T=200, coef_matrices=[A1, A2], seed=0)
    x, _ = gen.generate()
    assert x.shape == (200, C)
