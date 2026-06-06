"""Full model × anomaly-type recall matrix experiment.

Runs every enabled model against every anomaly type and builds the paper's
main result table: a recall matrix showing which models catch which types.

Expected block structure (testable hypothesis):
  - Per-channel models (zscore, moving_average) → high on A-types, low on C-types
  - Joint models (mahalanobis, covariance) → comparably high on A-types,
    substantially higher on C-types

Usage:
    python src/experiments/run_matrix_experiment.py \
        --config configs/exp_default.yaml \
        --output results/

    # Run only specific models:
    python src/experiments/run_matrix_experiment.py \
        --config configs/exp_default.yaml \
        --models zscore mahalanobis covariance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.eval.matrix import RecallMatrix
from src.experiments.run_experiment import _build_detector, _inject_and_label, _make_base_signal
from src.taxonomy.types import AnomalyType
from src.utils import set_seed

# Ordered for display: per-channel → windowed → joint
_DEFAULT_MODEL_ORDER = [
    "zscore",
    "moving_average",
    "knn",
    "lof",
    "mahalanobis",
    "covariance",
]

_ALL_TYPES = list(AnomalyType)


def run_matrix_experiment(
    models: list[str],
    anomaly_types: list[AnomalyType],
    cfg: dict,
    output_dir: str | Path,
) -> RecallMatrix:
    seed = cfg["seed"]
    set_seed(seed)

    x = _make_base_signal(cfg)
    T = x.shape[0]
    train_end = int(T * cfg["data"]["train_ratio"])
    x_train = x[:train_end]

    buffer_sizes = cfg["evaluation"].get("buffer_sizes", [0, 5, 10, 20, 50])
    matrix = RecallMatrix(buffer_sizes=buffer_sizes)

    inj_cfg = cfg["injection"]
    # Pre-build (x_test, labels) for each anomaly type to avoid re-injection per model
    print("Preparing injected test sets…")
    test_pairs: dict[AnomalyType, tuple] = {}
    for atype in anomaly_types:
        x_inj, labels = _inject_and_label(x, atype, inj_cfg, seed)
        test_pairs[atype] = (x_inj[train_end:], labels[train_end:])
        n_anom = labels[train_end:].sum()
        print(f"  {atype.value}: {n_anom} anomalous timesteps in test split")

    print()
    for model_name in models:
        print(f"[{model_name}]")
        try:
            detector = _build_detector(model_name, cfg)
        except Exception as e:
            print(f"  SKIP — {e}")
            continue

        detector.fit(x_train)
        for atype in anomaly_types:
            x_test, labels_test = test_pairs[atype]
            from src.eval.metrics import vus_pr as _vus_pr
            scores = detector.score(x_test)
            score = _vus_pr(scores, labels_test, buffer_sizes)
            matrix.add_result(model_name, atype, score)
            print(f"  {atype.value:>4}: {score:.4f}")
        print()

    # Print and save
    print("=" * 70)
    print("Model × Type Recall Matrix (VUS-PR)")
    print("=" * 70)
    matrix.print_table()

    paths = matrix.save(output_dir, prefix="recall_matrix")
    print(f"\nSaved → {paths['json']}")
    print(f"         {paths['csv']}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Full model × type recall matrix experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model subset (default: all enabled in config)")
    parser.add_argument("--types", nargs="+", default=None,
                        help="Anomaly type subset (default: all 10 types)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.models:
        models = args.models
    else:
        models_cfg = cfg.get("models", {})
        models = [m for m in _DEFAULT_MODEL_ORDER if models_cfg.get(m, {}).get("enabled", True)]

    anomaly_types = (
        [AnomalyType(t) for t in args.types] if args.types else _ALL_TYPES
    )

    run_matrix_experiment(models, anomaly_types, cfg, args.output)


if __name__ == "__main__":
    main()
