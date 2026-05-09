# Python Implementation

Self-contained Python implementation of the core credibility models from Morris (2026).

Covers the ML models (B-S standard, Joint-Decay scalar λ, Joint-Decay continuous λ).
The Bayesian tercile-λ model (the proposed model) requires brms/Stan and is only
available in the R implementation — see `../run_ca.R`.

---

## Quick Start

```bash
pip install -r requirements.txt
python example_cas_data.py          # downloads CAS data, fits models, prints results
python example_cas_data.py --plot   # also saves Z-curve plot to outputs/ca_python/
```

Expected output (matches Table 3 of the paper within rounding):

```
Model                       wMSE      Gini_pct  Slope   pct_vs_bs
Market Mean (baseline)      0.00391   ...
Last Year LR (naive)        0.00412   ...
Bühlmann-Straub (standard)  0.00335   ...       ...     0.0
Joint-Decay (scalar λ)      0.00255   ...       ...    -24.0
Joint-Decay (continuous λ)  0.00250   ...       ...    -25.5
```

---

## Module: `credibility.py`

| Function | Description |
|----------|-------------|
| `load_cas_data(filepath, lob)` | Load CAS Schedule P CSV; downloads if not present |
| `build_panel(df)` | Build 7-lag panel, standardise covariates, assign size terciles |
| `ewma_fbar(df, lam)` | Arithmetic EWMA (scalar lambda) |
| `ewma_fbar_vec(df, lam_vec)` | Arithmetic EWMA (per-row lambda vector) |
| `fit_bs_standard(df_train)` | MoM Bühlmann-Straub |
| `predict_bs_standard(fit, df)` | B-S predictions |
| `fit_jdecay_scalar(df_train)` | Joint-Decay scalar λ (ML via L-BFGS-B) |
| `predict_jdecay_scalar(fit, df)` | Predictions |
| `fit_jdecay_continuous(df_train)` | Joint-Decay continuous λ (ML) |
| `predict_jdecay_continuous(fit, df)` | Predictions |
| `eval_model(pred_rel, actual_rel, expo, mu_t, label)` | wMSE, Gini, slope |
| `eval_all_models(df_test, predictions)` | Evaluate dict of models |
| `plot_zcurves(fit_scalar, fit_bs, df_train)` | Z curve figure |

---

## Why no Bayesian model in Python?

The proposed tercile-λ model uses a non-linear brms formula with seven jointly
estimated parameters (α, β, az, bz, λ_Sm, λ_Md, λ_Lg) under a Gamma
likelihood. Replicating this in Python would require writing custom Stan code
or a Pyro/NumPyro equivalent — non-trivial and outside the scope of this
companion repo. The R version takes ~10 minutes on 4 chains × 2000 iterations
and caches the fitted model to `outputs/ca/models/ca_jdecay_tercile.rds`.

---

## Dependencies

```
numpy >= 1.23
pandas >= 1.5
scipy >= 1.9
matplotlib >= 3.6
statsmodels >= 0.13
```

Python 3.10+ recommended.
