"""Model × anomaly-type recall matrix builder.

Builds the main result table of the paper: rows = models, columns = anomaly
types, cells = VUS-PR.  The expected block structure along the A/B/C axis
(C-type anomalies missed by per-channel models, caught by joint models) is
the primary empirical contribution.

Usage (CLI):
    python src/eval/matrix.py --results results/
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.eval.metrics import vus_pr
from src.models.base import BaseDetector
from src.taxonomy.types import AnomalyType, AnomalyLayer, ANOMALY_REGISTRY


# Column ordering: A → B → C within each layer
_TYPE_ORDER = [
    AnomalyType.A1, AnomalyType.A2,
    AnomalyType.B1, AnomalyType.B2, AnomalyType.B3,
    AnomalyType.C1, AnomalyType.C2, AnomalyType.C3, AnomalyType.C4, AnomalyType.C5,
]


class RecallMatrix:
    """Accumulate per-(model, anomaly_type) VUS-PR scores and export.

    Parameters
    ----------
    buffer_sizes:
        Buffer sizes for VUS-PR.  Default: [0, 5, 10, 20, 50].
    """

    def __init__(self, buffer_sizes: Optional[list[int]] = None) -> None:
        self.buffer_sizes = buffer_sizes or [0, 5, 10, 20, 50]
        # {model_name: {anomaly_type_id: vus_pr_score}}
        self._results: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add_result(
        self,
        model_name: str,
        anomaly_type: AnomalyType | str,
        vus_pr_score: float,
    ) -> None:
        """Record a pre-computed VUS-PR score."""
        type_id = AnomalyType(anomaly_type).value if isinstance(anomaly_type, str) else anomaly_type.value
        self._results.setdefault(model_name, {})[type_id] = float(vus_pr_score)

    def evaluate(
        self,
        model_name: str,
        detector: BaseDetector,
        x_train: np.ndarray,
        test_pairs: dict[AnomalyType, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, float]:
        """Fit *detector* on *x_train* and score every (x_test, labels) pair.

        Parameters
        ----------
        model_name:
            Display name used as the row key.
        detector:
            Unfitted BaseDetector instance.
        x_train:
            Normal training signal, shape (T_train, C).
        test_pairs:
            AnomalyType → (x_injected, labels) where labels is shape (T,).

        Returns
        -------
        dict
            anomaly_type_value → vus_pr score for this model.
        """
        detector.fit(x_train)
        row: dict[str, float] = {}
        for atype, (x_test, labels) in test_pairs.items():
            scores = detector.score(x_test)
            score = vus_pr(scores, labels, self.buffer_sizes)
            self.add_result(model_name, atype, score)
            row[atype.value] = score
        return row

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return the raw results dict."""
        return dict(self._results)

    def to_array(
        self,
        model_order: Optional[list[str]] = None,
        type_order: Optional[list[AnomalyType]] = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Return (matrix, model_names, type_ids) as a numpy array.

        Missing cells are filled with NaN.
        """
        models = model_order or sorted(self._results.keys())
        types = type_order or _TYPE_ORDER
        type_ids = [t.value for t in types]

        arr = np.full((len(models), len(type_ids)), np.nan)
        for i, m in enumerate(models):
            for j, tid in enumerate(type_ids):
                if m in self._results and tid in self._results[m]:
                    arr[i, j] = self._results[m][tid]
        return arr, models, type_ids

    def save(self, output_dir: str | Path, prefix: str = "recall_matrix") -> dict[str, Path]:
        """Save results to timestamped JSON and CSV files.

        Returns dict with paths of created files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON: raw nested dict
        json_path = out / f"{prefix}_{ts}.json"
        payload = {
            "timestamp": ts,
            "buffer_sizes": self.buffer_sizes,
            "results": self._results,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        # CSV: rows = models, columns = anomaly types
        csv_path = out / f"{prefix}_{ts}.csv"
        arr, models, type_ids = self.to_array()
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + type_ids)
            for i, m in enumerate(models):
                row = [m] + [f"{v:.4f}" if not np.isnan(v) else "" for v in arr[i]]
                writer.writerow(row)

        return {"json": json_path, "csv": csv_path}

    def print_table(self, type_order: Optional[list[AnomalyType]] = None) -> None:
        """Print the recall matrix to stdout."""
        arr, models, type_ids = self.to_array(type_order=type_order)
        col_w = 8
        header = f"{'model':<20}" + "".join(f"{t:>{col_w}}" for t in type_ids)
        print(header)
        print("-" * len(header))
        for i, m in enumerate(models):
            row_str = f"{m:<20}"
            for v in arr[i]:
                row_str += f"{'N/A':>{col_w}}" if np.isnan(v) else f"{v:>{col_w}.3f}"
            print(row_str)


# ---------------------------------------------------------------------------
# Load results from saved JSON files
# ---------------------------------------------------------------------------

def build_from_results(results_dir: str | Path) -> RecallMatrix:
    """Reconstruct a RecallMatrix from all JSON result files in *results_dir*.

    Handles two file formats:
    - recall_matrix_*.json  — saved by RecallMatrix.save()
    - *_<model>.json        — saved by run_experiment.py (has "model" + "vus_pr" keys)

    Multiple files are merged; later files (lexicographically) overwrite earlier
    ones for the same (model, type) cell.
    """
    matrix = RecallMatrix()
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            data = json.load(f)

        if "results" in data:
            # recall_matrix format
            for model_name, scores in data["results"].items():
                for type_id, score in scores.items():
                    matrix.add_result(model_name, type_id, score)
        elif "vus_pr" in data and "model" in data:
            # run_experiment format
            model_name = data["model"]
            for type_id, score in data["vus_pr"].items():
                matrix.add_result(model_name, type_id, score)

    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Print model × type recall matrix")
    parser.add_argument("--results", required=True, help="Directory with recall_matrix_*.json files")
    args = parser.parse_args()

    matrix = build_from_results(args.results)
    if not matrix._results:
        print(f"No recall_matrix_*.json files found in {args.results}")
        return
    matrix.print_table()


if __name__ == "__main__":
    main()
