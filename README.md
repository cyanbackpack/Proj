# MTS Anomaly Taxonomy

A **detectability-grounded taxonomy of multivariate time series (MTS) anomalies** with a controlled benchmark dataset and systematic empirical analysis of how detection models behave across anomaly types.

**Core thesis:** Existing typed anomaly benchmarks are primarily univariate-centric. The inter-metric (cross-channel) anomaly subtypes in MTS are under-specified, lack injectable ground-truth labels, and have not been empirically linked to model inductive biases.

**Target venue:** NeurIPS / ICML / ICLR Datasets & Benchmarks track, or VLDB/KDD.

---

## Anomaly Taxonomy

### Axis: What structure must a detector model to catch this type?

| Axis | Layer | Description |
|------|-------|-------------|
| **A** | Marginal-visible | Detectable from single-channel value alone |
| **B** | Temporal-context | Requires temporal context within one channel |
| **C** | Inter-metric | Marginals stay in-distribution; only detectable via cross-channel structure |

### Type Definitions

| ID | Name | Layer | Key Property | Marginal in-dist? |
|----|------|-------|--------------|-------------------|
| A1 | Global Point | A | Single-channel extreme value | No |
| A2 | Contextual Point | A | In-range globally, anomalous in local context | Borderline |
| B1 | Seasonal | B | Phase/amplitude of periodicity breaks | Yes |
| B2 | Trend | B | Slope change or level shift over time | Yes |
| B3 | Shapelet | B | Unusual local subsequence shape | Yes |
| **C1** | Correlation-break | **C** | Normally-correlated channels decouple | **Yes** |
| **C2** | Phase/lag-shift | **C** | Leading-lagging relationship shifts in time | **Yes** |
| **C3** | Ratio anomaly | **C** | Both channels move but their ratio breaks | **Yes** |
| **C4** | Group-collective | **C** | Coherent regime shift in a channel subset | **Yes** |
| **C5** | Causal/precedence violation | **C** | Known A→B causal order violated | **Yes** |

**C-type invariant:** after injection, each channel's marginal z-score must not exceed its pre-injection maximum.

---

## Repository Structure

```
├── configs/                   # YAML experiment configs
│   ├── exp_default.yaml       # Default experiment configuration
│   └── var_base.yaml          # VAR signal generation parameters
├── data/
│   ├── synthetic/             # VAR-generated base signals
│   └── critique/              # SMAP/MSL (critique experiment only)
├── src/
│   ├── taxonomy/
│   │   ├── types.py           # AnomalyType enum + marginal-check utilities
│   │   └── injectors.py       # 10 injector classes (one per type)
│   ├── data/
│   │   ├── generator.py       # VAR(p) base signal generator
│   │   └── loader.py          # SMD, SWaT, SMAP, MSL loaders
│   ├── models/
│   │   ├── base.py            # BaseDetector ABC
│   │   ├── baseline.py        # ZScore, MovingAverage
│   │   ├── classic.py         # KNN, LOF, Mahalanobis, Covariance, LagCorrelation
│   │   └── deep/ae.py         # AutoEncoder (PyTorch, optional)
│   ├── eval/
│   │   ├── metrics.py         # VUS-PR implementation
│   │   └── matrix.py          # Model × type recall matrix builder
│   └── experiments/
│       ├── run_experiment.py  # Single-model experiment runner
│       └── run_matrix_experiment.py  # Full model × type matrix
├── notebooks/
│   └── NB01_injection_sanity_check.ipynb
├── tests/                     # pytest test suite (152 tests)
└── results/                   # Saved outputs (JSON + CSV, timestamped)
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic base signal
python src/data/generator.py --config configs/var_base.yaml

# Run a single model experiment
python src/experiments/run_experiment.py \
    --model mahalanobis \
    --config configs/exp_default.yaml

# Run full model × type recall matrix
python src/experiments/run_matrix_experiment.py \
    --config configs/exp_default.yaml \
    --output results/

# Run tests
pytest tests/ -v
```

---

## Models

| Model | Type | C-type sensitive? | Notes |
|-------|------|-------------------|-------|
| `zscore` | Per-channel | No | Baseline; max |z| per timestep |
| `moving_average` | Per-channel | No | Causal MA residual |
| `knn` | Windowed | Partial | Distance to k-NN training windows |
| `lof` | Windowed | Partial | Local Outlier Factor (novelty mode) |
| `mahalanobis` | Joint | **Yes** | Mahalanobis distance with LedoitWolf covariance |
| `covariance` | Joint | **Yes** | Frobenius ||Σ_local − Σ_train|| |
| `lag_correlation` | Joint | **Yes (C2)** | Cross-channel peak-lag shift detection |
| `ae` | Deep | Partial | AutoEncoder reconstruction error (PyTorch) |

---

## Evaluation

**Metric:** VUS-PR (Volume Under the Precision-Recall Surface).  
Computes the average AUC-PR over multiple buffer sizes `[0, 5, 10, 20, 50]`.  
**Never use PA-F1 (point-adjusted F1)** — it artificially inflates scores.

**Model × Type Recall Matrix (VUS-PR)** — main result table:

```
               A1      B1      B2      B3      C1      C2      C3      C4      C5
zscore        0.32    0.30    0.75    0.30    0.31    0.31    0.31    0.31    0.29
moving_avg    0.32    0.30    0.79    0.30    0.31    0.33    0.31    0.31    0.28
knn           0.33    0.55    0.76    0.57    0.46    0.34    0.42    0.48    0.31
lof           0.32    0.46    0.57    0.43    0.35    0.23    0.36    0.34    0.31
mahalanobis   0.35    0.97    0.93    0.93    0.99    0.41    0.99    0.98    0.88
covariance    0.35    0.97    0.94    0.93    0.99    0.39    0.99    0.98    0.87
lag_corr      (run experiment to populate)
```

**Key finding:** The block structure along the A/B/C axis is clearly visible.  
C2 (Phase/lag-shift) remains challenging for all current models — the `lag_correlation` detector specifically addresses this.

---

## Injection Design

### C1 — Correlation Break (IAAFT Surrogate)

Uses Iterated Amplitude-Adjusted Fourier Transform (IAAFT) to phase-randomise one channel within the anomaly window.  
Ends on a **rank-adjust step** that maps output values back to the original window's value set — guaranteeing the marginal distribution is mathematically identical.

### C2 — Phase/Lag Shift

Uses `np.roll` to circularly shift one channel by `τ` timesteps. Since it is a circular permutation, the value set is preserved exactly.

### C3 — Ratio Anomaly

Rank-reverses one channel: the value ranked k-th in ascending order gets the value ranked k-th in descending order. Value set is identical, but the relationship between channels breaks.

### C4 — Group Collective

Maps a donor channel's rank-order onto the original channel's sorted values. Temporal pattern of donor, original marginal distribution.

### C5 — Causal Violation

Replaces the cause channel with Gaussian noise matched to its local mean/std, breaking the A→B causal precedence.

---

## Key References

| Paper | Role |
|-------|------|
| Chandola et al. (ACM CSUR 2009) | Origin of point/contextual/collective taxonomy |
| Lai et al. (NeurIPS 2021) | A/B-type injection baseline; 35 synthetic datasets |
| Wu & Keogh (IEEE TKDE 2021) | Critique of SMAP/MSL; motivation for new benchmark |
| Li et al. / InterFusion (KDD 2021) | Inter-metric modeling baseline |
| Liu & Paparrizos / TSB-AD (NeurIPS 2024) | VUS-PR metric; multivariate benchmark |
| Wang et al. (Sensors 2025) | MTS anomaly survey confirming inter-metric split |

---

## Development Notes

- Python 3.10+, type hints on all public functions
- All random seeds set via `src.utils.set_seed(seed)` — reproducibility is mandatory
- Results saved as JSON + CSV with timestamps; never overwrite
- Do NOT use SMAP/MSL as base for type-labeled data (Wu & Keogh 2021)
- Do NOT use PA-F1 as evaluation metric
