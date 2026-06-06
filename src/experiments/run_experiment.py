"""Single-model experiment runner.

Generates a VAR base signal, injects anomalies of one or more types,
fits a detector on the normal training split, evaluates with VUS-PR,
and saves the result as a timestamped JSON file.

Usage:
    python src/experiments/run_experiment.py \
        --model zscore \
        --config configs/exp_default.yaml

    python src/experiments/run_experiment.py \
        --model knn \
        --anomaly-types C1 C2 C3 \
        --config configs/exp_default.yaml
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.data.generator import VARGenerator
from src.eval.metrics import vus_pr
from src.models.base import BaseDetector
from src.models.baseline import MovingAverageDetector, ZScoreDetector
from src.models.classic import (
    CovarianceAnomalyDetector,
    KNNDetector,
    LagCorrelationDetector,
    LOFDetector,
    MahalanobisDetector,
)
from src.taxonomy.injectors import INJECTOR_REGISTRY
from src.taxonomy.types import AnomalyType
from src.utils import set_seed


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_detector(model_name: str, cfg: dict) -> BaseDetector:
    model_cfg = cfg.get("models", {}).get(model_name, {})
    if model_name == "zscore":
        return ZScoreDetector()
    if model_name == "moving_average":
        return MovingAverageDetector(window=model_cfg.get("window", 20))
    if model_name == "knn":
        return KNNDetector(
            n_neighbors=model_cfg.get("n_neighbors", 5),
            window_size=cfg["data"]["window_size"],
        )
    if model_name == "lof":
        return LOFDetector(
            n_neighbors=model_cfg.get("n_neighbors", 20),
            window_size=cfg["data"]["window_size"],
        )
    if model_name == "mahalanobis":
        return MahalanobisDetector(
            window_size=cfg["data"]["window_size"],
            covariance_estimator=model_cfg.get("covariance_estimator", "ledoit_wolf"),
        )
    if model_name == "covariance":
        return CovarianceAnomalyDetector(
            window_size=cfg["data"]["window_size"],
            stride=model_cfg.get("stride", 1),
        )
    if model_name == "lag_correlation":
        return LagCorrelationDetector(
            window_size=model_cfg.get("window_size", cfg["data"]["window_size"] * 2),
            max_lag=model_cfg.get("max_lag", 20),
            stride=model_cfg.get("stride", 1),
        )
    if model_name == "ae":
        from src.models.deep.ae import AutoEncoderDetector
        return AutoEncoderDetector(
            window_size=cfg["data"]["window_size"],
            hidden_dims=model_cfg.get("hidden_dims", [64, 16]),
            epochs=model_cfg.get("epochs", 50),
            batch_size=model_cfg.get("batch_size", 64),
            lr=model_cfg.get("lr", 1e-3),
        )
    raise ValueError(f"Unknown model: '{model_name}'. "
                     f"Choose from: zscore, moving_average, knn, lof, "
                     f"mahalanobis, covariance, lag_correlation, ae")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _make_base_signal(cfg: dict) -> np.ndarray:
    """Load or generate the base signal."""
    base_path = Path(cfg["data"]["base_path"])
    if base_path.exists():
        return np.load(base_path)["x"]

    # Generate fresh from VAR config
    var_cfg = cfg.get("var", {})
    gen = VARGenerator(
        n_channels=var_cfg.get("n_channels", 6),
        lag_order=var_cfg.get("lag_order", 2),
        T=var_cfg.get("T", 2000),
        spectral_radius=var_cfg.get("spectral_radius", 0.7),
        noise_scale=var_cfg.get("noise_scale", 1.0),
        seed=cfg["seed"],
    )
    x, _ = gen.generate()
    base_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(base_path, x=x)
    return x


def _inject_and_label(
    x: np.ndarray,
    anomaly_type: AnomalyType,
    injection_cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_injected, labels) for one anomaly type."""
    T = x.shape[0]
    win_size = injection_cfg.get("window_size", 100)
    n_windows = injection_cfg.get("n_windows", 5)

    rng = np.random.default_rng(seed)
    # Space n_windows non-overlapping injection windows evenly
    spacing = T // (n_windows + 1)
    starts = [spacing * (i + 1) for i in range(n_windows)]

    injector_cls = INJECTOR_REGISTRY[anomaly_type]
    sig = inspect.signature(injector_cls.__init__)
    init_kwargs = {"seed": seed} if "seed" in sig.parameters else {}

    # Pass n_points from config when the injector supports it.
    # Use a per-class lookup so configs can tune density independently.
    _N_POINTS_KEY: dict[str, str] = {
        "A1GlobalPointInjector": "a1_n_points",
        "A2ContextualPointInjector": "a2_n_points",
    }
    if "n_points" in sig.parameters:
        cfg_key = _N_POINTS_KEY.get(injector_cls.__name__)
        if cfg_key is not None and cfg_key in injection_cfg:
            init_kwargs["n_points"] = injection_cfg[cfg_key]

    injector = injector_cls(**init_kwargs)

    x_inj = x.copy()
    labels = np.zeros(T, dtype=int)

    for s in starts:
        end = min(s + win_size, T)
        if end - s < win_size // 2:
            continue
        x_inj = injector.inject(x_inj, (s, end))
        labels[s:end] = 1

    return x_inj, labels


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    model_name: str,
    anomaly_types: list[AnomalyType],
    cfg: dict,
) -> dict:
    seed = cfg["seed"]
    set_seed(seed)

    x = _make_base_signal(cfg)
    T = x.shape[0]
    train_end = int(T * cfg["data"]["train_ratio"])
    x_train = x[:train_end]

    detector = _build_detector(model_name, cfg)
    detector.fit(x_train)

    buffer_sizes = cfg["evaluation"].get("buffer_sizes", [0, 5, 10, 20, 50])
    results: dict[str, float] = {}

    for atype in anomaly_types:
        x_inj, labels = _inject_and_label(x, atype, cfg["injection"], seed)
        x_test = x_inj[train_end:]
        labels_test = labels[train_end:]

        scores = detector.score(x_test)
        score = vus_pr(scores, labels_test, buffer_sizes)
        results[atype.value] = score
        print(f"  {atype.value:>4}: VUS-PR = {score:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-model anomaly detection experiment")
    parser.add_argument("--model", required=True,
                        help="Model name: zscore | moving_average | knn | lof | ae")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument(
        "--anomaly-types", nargs="+", default=None,
        help="Anomaly type IDs (e.g. C1 C2 C3). Default: from config injection.types",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    type_ids = args.anomaly_types or cfg["injection"].get("types", [t.value for t in AnomalyType])
    anomaly_types = [AnomalyType(t) for t in type_ids]

    print(f"\nModel: {args.model}")
    print(f"Types: {[t.value for t in anomaly_types]}")
    print(f"Seed:  {cfg['seed']}\n")

    results = run_experiment(args.model, anomaly_types, cfg)

    # Save
    output_dir = Path(cfg["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{ts}_{args.model}.json"
    payload = {
        "timestamp": ts,
        "model": args.model,
        "anomaly_types": [t.value for t in anomaly_types],
        "seed": cfg["seed"],
        "config": args.config,
        "vus_pr": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
