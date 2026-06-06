"""VAR(p) base signal generator.

Usage (CLI):
    python src/data/generator.py --config configs/var_base.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.utils import set_seed


class VARGenerator:
    """Generate multivariate time series from a VAR(p) process.

    Parameters
    ----------
    n_channels:
        Number of variates (C).
    lag_order:
        VAR lag order p.
    T:
        Number of time steps to generate.
    spectral_radius:
        Desired spectral radius of the stacked companion matrix.
        Must be < 1.0 to ensure stationarity.
    noise_scale:
        Standard deviation of the white-noise innovation term.
    seed:
        Random seed for reproducibility.
    coef_matrices:
        Optional list of p coefficient matrices (each C×C).  If None,
        random stable matrices are generated.
    """

    def __init__(
        self,
        n_channels: int = 6,
        lag_order: int = 2,
        T: int = 2000,
        spectral_radius: float = 0.7,
        noise_scale: float = 1.0,
        seed: int = 42,
        coef_matrices: Optional[list[np.ndarray]] = None,
    ) -> None:
        if spectral_radius >= 1.0:
            raise ValueError(f"spectral_radius must be < 1.0, got {spectral_radius}")

        self.n_channels = n_channels
        self.lag_order = lag_order
        self.T = T
        self.spectral_radius = spectral_radius
        self.noise_scale = noise_scale
        self.seed = seed
        self._coef_matrices = coef_matrices

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> tuple[np.ndarray, dict]:
        """Generate a VAR(p) time series.

        Returns
        -------
        x:
            Signal array of shape (T, C).
        meta:
            Dict with coefficient matrices, noise covariance, and seed.
        """
        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        coef_matrices = self._coef_matrices or self._make_stable_coefs(rng)
        sigma_noise = np.eye(self.n_channels) * (self.noise_scale ** 2)

        x = self._simulate(coef_matrices, sigma_noise, rng)

        meta = {
            "seed": self.seed,
            "n_channels": self.n_channels,
            "lag_order": self.lag_order,
            "T": self.T,
            "spectral_radius_target": self.spectral_radius,
            "noise_scale": self.noise_scale,
            "coef_matrices": [A.tolist() for A in coef_matrices],
            "sigma_noise": sigma_noise.tolist(),
        }
        return x, meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_stable_coefs(self, rng: np.random.Generator) -> list[np.ndarray]:
        """Return p random C×C coefficient matrices that produce a stable VAR."""
        C, p = self.n_channels, self.lag_order
        coef_matrices: list[np.ndarray] = []

        for _ in range(p):
            A = rng.standard_normal((C, C)) * 0.3
            coef_matrices.append(A)

        # Rescale so that the companion matrix has the desired spectral radius.
        companion = _build_companion(coef_matrices)
        current_sr = _spectral_radius(companion)
        if current_sr > 1e-8:
            scale = self.spectral_radius / current_sr
            coef_matrices = [A * scale for A in coef_matrices]

        return coef_matrices

    def _simulate(
        self,
        coef_matrices: list[np.ndarray],
        sigma_noise: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        C, p, T = self.n_channels, self.lag_order, self.T
        x = np.zeros((T, C))
        noise = rng.multivariate_normal(np.zeros(C), sigma_noise, size=T)

        for t in range(T):
            x[t] = noise[t]
            for k, A in enumerate(coef_matrices, start=1):
                if t - k >= 0:
                    x[t] += x[t - k] @ A.T

        return x


# ------------------------------------------------------------------
# Utility functions (module-level)
# ------------------------------------------------------------------

def _build_companion(coef_matrices: list[np.ndarray]) -> np.ndarray:
    """Build the (Cp × Cp) companion matrix for stability analysis."""
    C = coef_matrices[0].shape[0]
    p = len(coef_matrices)
    companion = np.zeros((C * p, C * p))
    for k, A in enumerate(coef_matrices):
        companion[:C, k * C : (k + 1) * C] = A
    if p > 1:
        companion[C:, : C * (p - 1)] = np.eye(C * (p - 1))
    return companion


def _spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius (largest absolute eigenvalue) of a matrix."""
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _resolve_path(template: str, seed: int) -> Path:
    return Path(template.replace("{seed}", str(seed)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VAR(p) base signals")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    vcfg = cfg["var"]
    gen = VARGenerator(
        n_channels=vcfg["n_channels"],
        lag_order=vcfg["lag_order"],
        T=vcfg["T"],
        spectral_radius=vcfg["spectral_radius"],
        noise_scale=vcfg["noise_scale"],
        seed=seed,
    )

    x, meta = gen.generate()

    out_path = _resolve_path(cfg["output"]["path"], seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict = {"x": x}
    if cfg["output"].get("save_covariance"):
        save_kwargs["sigma_noise"] = np.array(meta["sigma_noise"])

    np.savez(out_path, **save_kwargs)
    print(f"Saved {x.shape} signal to {out_path}")


if __name__ == "__main__":
    main()
