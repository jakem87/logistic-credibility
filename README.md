# Logistic Credibility: Code Companion

Companion repository for:

> Morris, J. (2026). *Logistic Credibility with Temporal Decay: Extending Bühlmann-Straub for Commercial Lines*. [https://arxiv.org/abs/2606.08692]

This repository lets you reproduce the main empirical results (Table 3) and cross-LoB validation using publicly available CAS Schedule P data.

---

## Quick Start

```r
# Commercial Auto — reproduces Table 3
source("run_ca.R")

# Other Liability — cross-LoB validation
source("run_ol.R")

# Nesting demonstration (no Stan required, ~20 seconds)
source("run_nesting_demo.R")
```

---

## What This Repo Contains

| File | Purpose |
|------|---------|
| `run_ca.R` | One-click CA analysis → results + plots in `outputs/ca/` |
| `run_ol.R` | One-click OL analysis → results + plots in `outputs/ol/` |
| `run_nesting_demo.R` | Self-contained nesting illustration (no Bayesian models, ~20 seconds) |
| `R/01_data.R` | Data download, filtering, panel construction |
| `R/02_features.R` | EWMA helpers and multi-start optimiser |
| `R/03_models_ca.R` | All 8 models fitted on Commercial Auto |
| `R/04_models_ol.R` | Same models applied to Other Liability |
| `R/05_evaluate.R` | wMSE, Gini, calibration slope; bootstrap CIs |
| `R/06_plots.R` | Key figures |
| `R/07_nesting_demo.R` | S1 simulation showing logistic nests B-S |
| `data/README.md` | CAS data download instructions |

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| Market Mean | Baseline | Portfolio mean µ_t each year |
| Last Year LR | Baseline | Lag-1 loss ratio |
| Bühlmann-Straub (standard) | MoM | Classical B-S, single pooled K |
| B-S (best sequential patch) | MoM + manual | Stratified K + size complement + EWMA tercile λ |
| GLMM | REML | Random intercept per company + size fixed effect |
| Joint-Decay (scalar λ) | MLE | Logistic Z + single estimated decay parameter |
| Joint-Decay (continuous λ) | MLE | As above, λ varies continuously with account size |
| **Joint-Decay (tercile λ)** | **Bayesian** | **Proposed model: size-specific λ + continuous size-slope complement** |

The proposed model achieves ~40% wMSE reduction over standard B-S on the held-out test set (oracle complement: realised µ_t used). In a forecast setting (µ_t replaced by prior-year estimate) this reduces to 29–32% — see paper Section 5.2 and the Notes section below.

---

## Dependencies

```r
install.packages(c("dplyr", "tidyr", "ggplot2", "glmmTMB", "scales"))

# For Bayesian model (Joint-Decay tercile lambda):
install.packages("brms")
# brms requires Stan — install cmdstanr or rstan:
# https://mc-stan.org/cmdstanr/  or  https://mc-stan.org/rstan/
```

R version ≥ 4.2, brms ≥ 2.21 recommended.

The Bayesian model takes ~10 minutes to fit (4 chains × 2000 iterations). After the first run, the fitted model is cached to `outputs/ca/models/ca_jdecay_tercile.rds` and subsequent runs load from cache. Set `REFIT <- TRUE` at the top of `run_ca.R` to force a refit.

---

## Data

CAS Schedule P data is publicly available. See `data/README.md` for download links. The data files are not committed to this repository; the run scripts will download them automatically if not present.

---

## Output Files

After running `source("run_ca.R")`:

| File | Contents |
|------|----------|
| `outputs/ca/results_ca.csv` | wMSE, Gini, Slope for all 8 models |
| `outputs/ca/results_ca_tercile.csv` | Same broken down by size tercile |
| `outputs/ca/bootstrap_ci_ca.csv` | 90% bootstrap CI on wMSE improvement vs B-S |
| `outputs/ca/wmse_by_tercile.png` | wMSE improvement figure |
| `outputs/ca/zcurves.png` | Fitted Z vs log exposure |
| `outputs/ca/lambda_posteriors.png` | Lambda posterior by size tercile |

---

## Citation

```
Morris, J. (2026). Logistic Credibility with Temporal Decay: Extending Bühlmann-Straub for Commercial Lines. [https://arxiv.org/abs/2606.08692].
```

---

## Notes

- **Oracle complement**: predictions use the realised portfolio mean µ_t for the test year (AY 2006–2007). In deployment, µ_T must be forecast. Replacing the oracle with a prior-year or trend-based estimate increases wMSE by 7–11% while leaving model rankings unchanged (see paper Section 5.2).
- **Data proxy**: each "account" here is an entire insurance company. Company-aggregate loss ratios are smoother than individual-account loss ratios; Z and λ estimates from this study should be treated as illustrative rather than direct deployment defaults.
- **GLMM limitation**: the GLMM cannot score new accounts (random effect = 0 for unseen companies). In this balanced panel all test companies appear in training, so this limitation is latent.
- **MLE vs Bayesian discrepancy**: the paper fits Joint-Decay scalar λ and continuous λ via Bayesian inference (brms/Stan) with weakly-informative priors. This repo implements them as pure MLE (`optim`/`scipy.optimize`). Results will differ slightly — Bayesian priors regularise λ away from boundary values, which matters most when a tercile has few companies. The MAP variants (`fit_jdecay_scalar_map`, `fit_jdecay_tercile_map`) add equivalent log-prior penalties to the MLE objective and produce results much closer to the paper's Bayesian estimates without requiring Stan.
