# CLAUDE.md — MTS Anomaly Taxonomy Project

## Project Overview

This is a solo research project aiming to construct a **detectability-grounded taxonomy of multivariate time series (MTS) anomalies**, with a controlled benchmark dataset and a systematic empirical analysis of how detection models behave differently across anomaly types.

**Core thesis:** Existing typed anomaly benchmarks (Lai et al. NeurIPS 2021, TSB-UAD) are primarily univariate-centric. The inter-metric (cross-channel) anomaly subtypes in MTS are under-specified, lack injectable ground-truth labels, and have not been empirically linked to model inductive biases.

**Target venue:** NeurIPS / ICML / ICLR Datasets & Benchmarks track, or VLDB/KDD.

---

## Anomaly Taxonomy (Core Domain Knowledge)

This is the central design artifact. All code, data, and experiments revolve around it.

### Axis: What structure must a detector model to catch this type?

```
(A) Marginal-visible      → detectable from single-channel value alone
(B) Temporal-context      → requires temporal context within one channel
(C) Inter-metric, joint-only-visible  → marginals stay in-distribution; only detectable via cross-channel structure
```

### Type Definitions

| ID | Name | Layer | Key Property | Marginal in-dist? |
|----|------|-------|-------------|-------------------|
| A1 | Global Point | Marginal | Single-channel extreme value | No |
| A2 | Contextual Point | Marginal | In-range globally, anomalous in local context | Borderline |
| B1 | Seasonal | Temporal | Phase/amplitude of periodicity breaks | Yes |
| B2 | Trend | Temporal | Slope change or level shift over time | Yes |
| B3 | Shapelet | Temporal | Unusual local subsequence shape | Yes |
| C1 | Correlation-break | Inter-metric | Normally-correlated channels decouple (sign/magnitude change) | **Yes** |
| C2 | Phase/lag-shift | Inter-metric | Leading-lagging relationship shifts in time | **Yes** |
| C3 | Ratio anomaly | Inter-metric | Both channels move but their ratio breaks | **Yes** |
| C4 | Group-collective | Inter-metric | Coherent regime shift in a channel subset, others normal | **Yes** |
| C5 | Causal/precedence violation | Inter-metric | Known A→B causal order violated (B fires without A) | **Yes** |

**Critical invariant for C-type injection:** each channel's marginal z-score must remain within normal range. This is the defining constraint — verify it after every injection.

### Baseline Coverage (Related Work)
- A1, A2, B1–B3: covered by Lai et al. (NeurIPS 2021) synthetic criterion → use as baseline, do not reinvent
- C1–C5: **the novel contribution** — no existing benchmark provides injectable, separately-labeled inter-metric subtypes
- High-level intra/inter split exists (InterFusion KDD 2021; Wang et al. Sensors 2025 survey) but without fine-grained subtypes or injection tools

---

## Repository Structure

```
mts-anomaly-taxonomy/
├── CLAUDE.md                  # ← this file
├── README.md
├── configs/                   # YAML experiment configs
├── data/
│   ├── synthetic/             # VAR-generated base signals + injected anomalies
│   ├── real/                  # SMD / SWaT normal segments (base for injection)
│   └── critique/              # SMAP, MSL (used ONLY for critique experiment)
├── src/
│   ├── taxonomy/
│   │   ├── types.py           # AnomalyType enum, metadata, marginal-check util
│   │   └── injectors.py       # One injector class per type; all inherit BaseInjector
│   ├── data/
│   │   ├── generator.py       # VAR / linear-Gaussian base signal generation
│   │   └── loader.py          # SMD, SWaT, SMAP, MSL loaders
│   ├── models/
│   │   ├── baseline.py        # ZScore, MovingAverage (reproduces Wu & Keogh one-liner)
│   │   ├── classic.py         # KNN, LOF
│   │   └── deep/              # AE, USAD, InterFusion, GDN, TranAD wrappers
│   ├── eval/
│   │   ├── metrics.py         # VUS-PR implementation; per-type breakdown
│   │   └── matrix.py          # model × type recall matrix builder
│   └── experiments/           # Runnable experiment scripts
├── notebooks/                 # EDA, result visualization
├── tests/
└── results/                   # Saved outputs, tables, figures
```

---

## Key Design Decisions

**Injection strategy for C1 (Correlation-break):**
Use Fourier phase-randomization on one channel within the anomaly window. This preserves the marginal power spectrum while destroying cross-correlation. Reference: IAAFT surrogate method. Always verify post-injection that per-channel z-score stays below threshold (default: 2.5σ).

**Base signal choice:**
- Primary: VAR(p) simulation with known covariance matrix (full ground-truth control)
- Secondary realism check: normal segments of SMD / SWaT as injection base
- SMAP / MSL: strictly for critique experiment only — do NOT use as base for new labels

**Metric policy:**
- Primary: **VUS-PR** (Volume Under the Surface — Precision-Recall). See TSB-AD, NeurIPS 2024.
- Never use point-adjusted F1 (PA-F1). It is known to artificially inflate scores.
- Report per-type recall separately; the model × type recall matrix is the main result table.

**Inductive bias hypothesis (testable):**
C-type anomalies should be missed by per-channel models (AE, ZScore) and caught by joint/graph models (InterFusion, GDN). The recall matrix should show a clear block structure along the A/B/C axis.

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic base signals
python src/data/generator.py --config configs/var_base.yaml

# Inject anomalies (all types)
python src/taxonomy/injectors.py --base data/synthetic/base.npz --output data/synthetic/injected/

# Run a single model experiment
python src/experiments/run_experiment.py --model interfusion --config configs/exp_default.yaml

# Build model × type recall matrix
python src/eval/matrix.py --results results/

# Run tests
pytest tests/
```

---

## Coding Conventions

- **Python 3.10+**, type hints on all public functions
- Each injector must implement `inject(x: np.ndarray, window: tuple) -> np.ndarray` and `verify_marginal(x_orig, x_injected) -> bool`
- Experiment configs in YAML; no hardcoded hyperparameters in scripts
- All random seeds set via `utils.set_seed(seed)` — reproducibility is mandatory for a benchmark paper
- Results saved as JSON + CSV; never overwrite, always timestamp
- Notebook filenames: `NB{number}_{description}.ipynb` (e.g., `NB01_injection_sanity_check.ipynb`)

---

## What NOT to Do

- Do not use SMAP or MSL as a base for type-labeled data — their binary labels are known to be flawed (Wu & Keogh, IEEE TKDE 2021)
- Do not claim the point/contextual/collective taxonomy is novel — it is from Chandola et al. (2009)
- Do not use PA-F1 (point-adjusted F1) as an evaluation metric
- Do not reinvent A/B-type injectors — cite and reuse Lai et al. (NeurIPS 2021) synthetic criterion
- Do not hardcode API keys or data paths

---

## Key References

| Paper | Role in this project |
|-------|---------------------|
| Chandola et al. (2009) ACM CSUR | Origin of point/contextual/collective taxonomy |
| Lai et al. (NeurIPS 2021) | A/B-type injection baseline; 35 synthetic datasets |
| Wu & Keogh (IEEE TKDE 2021) | Critique of SMAP/MSL; motivation for new benchmark |
| InterFusion — Li et al. (KDD 2021) | Inter-metric modeling; baseline model |
| TSB-AD — Liu & Paparrizos (NeurIPS 2024) | VUS-PR metric; multivariate benchmark comparison |
| Wang et al. (Sensors 2025) | MTS anomaly survey; confirms inter-metric split exists at high level |

---

## Project Status

- [ ] VAR base signal generator
- [ ] C1 (Correlation-break) injector with IAAFT surrogate + marginal check
- [ ] C2–C5 injectors
- [ ] A/B-type injectors (port from Lai et al. criterion)
- [ ] Model wrappers: baseline → classic → deep
- [ ] VUS-PR implementation
- [ ] Pilot experiment: AE vs InterFusion on C1 (hypothesis validation)
- [ ] Full model × type recall matrix experiment
- [ ] Critique experiment on SMAP/MSL
