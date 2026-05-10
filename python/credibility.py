"""
credibility.py — Core implementation of adaptive experience credibility.

Implements the models from Morris (2026) "Adaptive Experience Credibility:
A Logistic Extension to Bühlmann-Straub" using pure Python/NumPy/SciPy.

Covers: B-S standard (MoM), Joint-Decay scalar λ (ML), Joint-Decay
continuous λ (ML), and evaluation metrics (wMSE, Gini, calibration slope).

The Bayesian tercile-λ model (the proposed model) requires brms/Stan and is only
available in the R implementation (see R/03_models_ca.R).
"""

import urllib.request
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import gamma as gamma_dist
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data loading and panel construction
# ---------------------------------------------------------------------------

CAS_URLS = {
    "ca": "https://www.casact.org/sites/default/files/2026-03/comauto_pos_98-07.csv",
    "ol": "https://www.casact.org/sites/default/files/2026-03/othliab_pos_98-07.csv",
}

TRAIN_YEARS = list(range(2001, 2006))
TEST_YEARS  = list(range(2006, 2008))
W_MAX       = 7
MIN_NEP     = 100


def _weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    return float(np.average(df[value_col], weights=df[weight_col]))


def _portfolio_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exposure-weighted portfolio LR mean by accident year."""
    result = (
        df.assign(_lr_expo=df["lr"] * df["expo"])
        .groupby("AccidentYear")
        .agg(_lr_expo_sum=("_lr_expo", "sum"), _expo_sum=("expo", "sum"))
        .assign(mu_t=lambda d: d["_lr_expo_sum"] / d["_expo_sum"])
        [["mu_t"]]
        .reset_index()
    )
    return result


def load_cas_data(filepath: str, lob: str = "ca") -> pd.DataFrame:
    """Load CAS Schedule P data, downloading if not present.

    Args:
        filepath: Path to save/load the CSV.
        lob: Line of business — "ca" (Commercial Auto) or "ol" (Other Liability).

    Returns:
        DataFrame with columns: GRCODE, AccidentYear, expo, lr.
    """
    if not os.path.exists(filepath):
        url = CAS_URLS[lob]
        print(f"Downloading CAS data from: {url}")
        urllib.request.urlretrieve(url, filepath)

    raw = pd.read_csv(filepath)

    ult = (
        raw[raw["DevelopmentLag"] == 10]
        .assign(
            expo=lambda d: d["EarnedPremNet"],
            lr=lambda d: d["IncurredLosses"] / d["EarnedPremNet"],
        )
        .query("expo >= @MIN_NEP")[["GRCODE", "GRNAME", "AccidentYear", "expo", "lr"]]
        .copy()
    )

    # Keep companies with exactly 10 valid years and all positive LRs
    counts = ult.groupby("GRCODE").agg(n=("lr", "count"), any_bad=("lr", lambda x: (x < 0).any()))
    valid_cos = counts[(counts["n"] == 10) & (~counts["any_bad"])].index
    ult = ult[ult["GRCODE"].isin(valid_cos)].copy()

    if lob == "ol":
        yr_means = _portfolio_means(ult)
        ult = ult.merge(yr_means, on="AccidentYear")
        ult["lr_rel_tmp"] = ult["lr"] / ult["mu_t"]
        max_lr = ult.groupby("GRCODE")["lr_rel_tmp"].max()
        extreme = max_lr[max_lr > 5].index
        ult = ult[~ult["GRCODE"].isin(extreme)].drop(columns=["mu_t", "lr_rel_tmp"]).copy()

    print(f"Qualifying companies: {ult['GRCODE'].nunique()}")
    return ult.reset_index(drop=True)


def _rolling_cv(lrs: np.ndarray, w_max: int) -> list:
    """Compute rolling coefficient of variation of LRs."""
    cvs = []
    for j in range(len(lrs)):
        h = lrs[max(0, j - w_max):j]
        if len(h) < 2:
            cvs.append(np.nan)
        else:
            cvs.append(float(np.std(h, ddof=1) / max(float(np.mean(h)), 1e-8)))
    return cvs


def build_panel(df: pd.DataFrame, w_max: int = W_MAX) -> tuple:
    """Build lag panel, standardise covariates, assign size terciles.

    Returns:
        (df_train, df_test, tr_stats) where tr_stats is a dict of standardisation
        constants computed from training observations.
    """
    # Portfolio mean LR by year (Option B normalisation)
    yr_means = _portfolio_means(df)

    panel = df.merge(yr_means, on="AccidentYear").copy()
    panel["lr_rel"] = panel["lr"] / panel["mu_t"]
    panel = panel.sort_values(["GRCODE", "AccidentYear"]).reset_index(drop=True)

    # Build lag columns per company (pandas shift is group-key safe)
    for k in range(1, w_max + 1):
        panel[f"lr_lag{k}"]   = panel.groupby("GRCODE")["lr"].shift(k)
        panel[f"expo_lag{k}"] = panel.groupby("GRCODE")["expo"].shift(k)
        panel[f"mu_lag{k}"]   = panel.groupby("GRCODE")["mu_t"].shift(k)

    # Zero-fill pre-entry lags (k >= 4)
    for k in range(4, w_max + 1):
        panel[f"expo_lag{k}"] = panel[f"expo_lag{k}"].fillna(0.0)
        panel[f"mu_lag{k}"]   = panel[f"mu_lag{k}"].fillna(panel["mu_t"])
        panel[f"lr_lag{k}"]   = panel[f"lr_lag{k}"].fillna(panel["mu_t"])

    # Relative lags
    for k in range(1, w_max + 1):
        panel[f"lr_lag{k}_rel"] = panel[f"lr_lag{k}"] / panel[f"mu_lag{k}"]

    # expo_used, past_lr_rel, W_i
    panel["expo_used"] = sum(panel[f"expo_lag{k}"] for k in range(1, w_max + 1))
    panel["past_lr_rel"] = (
        sum(panel[f"lr_lag{k}_rel"] * panel[f"expo_lag{k}"] for k in range(1, w_max + 1))
        / panel["expo_used"].clip(lower=1e-8)
    )
    panel["W_i"] = sum((panel[f"expo_lag{k}"] > 0).astype(int) for k in range(1, w_max + 1))
    panel["n_years_hist"] = panel.groupby("GRCODE").cumcount()

    # CV of annual LRs — explicit loop avoids pandas 3.0 groupby.apply issues
    cv_series_parts = []
    for _, grp in panel.groupby("GRCODE", sort=False):
        lrs = grp["lr"].values
        cv_series_parts.append(pd.Series(_rolling_cv(lrs, w_max), index=grp.index))
    panel["cv_lr"] = pd.concat(cv_series_parts).reindex(panel.index)

    # Drop rows with missing required columns (lags 1–3 not yet filled)
    panel = panel.dropna(subset=["lr_rel", "lr_lag1_rel", "expo_lag1", "n_years_hist", "cv_lr", "mu_t"])
    panel = panel[panel["AccidentYear"].isin(TRAIN_YEARS + TEST_YEARS)].copy()

    # Standardise on training observations
    tr = panel[panel["AccidentYear"].isin(TRAIN_YEARS)]
    tr_stats = {
        "log_expo_mean":      float(np.log(tr["expo"]).mean()),
        "log_expo_sd":        float(np.log(tr["expo"]).std(ddof=1)),
        "log_expo_used_mean": float(np.log(tr["expo_used"]).mean()),
        "log_expo_used_sd":   float(np.log(tr["expo_used"]).std(ddof=1)),
        "ny_mean":            float(tr["n_years_hist"].mean()),
        "ny_sd":              float(tr["n_years_hist"].std(ddof=1)),
        "cv_mean":            float(tr["cv_lr"].mean()),
        "cv_sd":              float(tr["cv_lr"].std(ddof=1)),
        "expo_mean":          float(tr["expo"].mean()),
        "yr_mean":            float(tr["AccidentYear"].mean()),
        "yr_sd":              float(tr["AccidentYear"].std(ddof=1)),
    }

    panel = panel.copy()
    panel["log_expo_sc"]      = (np.log(panel["expo"])      - tr_stats["log_expo_mean"])  / tr_stats["log_expo_sd"]
    panel["log_expo_used_sc"] = (np.log(panel["expo_used"]) - tr_stats["log_expo_used_mean"]) / tr_stats["log_expo_used_sd"]
    panel["n_years_sc"]       = (panel["n_years_hist"]      - tr_stats["ny_mean"])        / tr_stats["ny_sd"]
    panel["cv_sc"]            = (panel["cv_lr"]             - tr_stats["cv_mean"])        / tr_stats["cv_sd"]
    panel["yr_sc"]            = (panel["AccidentYear"]      - tr_stats["yr_mean"])        / tr_stats["yr_sd"]
    panel["expo_wt"]          = panel["expo"] / tr_stats["expo_mean"]

    # Log mean expo and size terciles (based on mean training NEP per company)
    tr = panel[panel["AccidentYear"].isin(TRAIN_YEARS)]
    co_mean_expo = tr.groupby("GRCODE")["expo"].mean().rename("mean_expo").reset_index()
    lme_mean = float(np.log(co_mean_expo["mean_expo"]).mean())
    lme_sd   = float(np.log(co_mean_expo["mean_expo"]).std(ddof=1))
    co_mean_expo["log_mean_expo_sc"] = (np.log(co_mean_expo["mean_expo"]) - lme_mean) / lme_sd

    tercile_breaks = np.quantile(co_mean_expo["mean_expo"].values, [1/3, 2/3])
    co_mean_expo["size_tercile"] = pd.cut(
        co_mean_expo["mean_expo"],
        bins=[-np.inf] + list(tercile_breaks) + [np.inf],
        labels=["Small", "Mid", "Large"],
    ).astype(str)
    co_mean_expo["isSm"] = (co_mean_expo["size_tercile"] == "Small").astype(int)
    co_mean_expo["isMd"] = (co_mean_expo["size_tercile"] == "Mid").astype(int)
    co_mean_expo["isLg"] = (co_mean_expo["size_tercile"] == "Large").astype(int)

    panel = panel.merge(
        co_mean_expo[["GRCODE", "size_tercile", "isSm", "isMd", "isLg", "log_mean_expo_sc"]],
        on="GRCODE", how="left",
    )

    df_train = panel[panel["AccidentYear"].isin(TRAIN_YEARS)].copy()
    df_test  = panel[panel["AccidentYear"].isin(TEST_YEARS)].copy()

    print(f"Training obs: {len(df_train)}   Test obs: {len(df_test)}")
    print(f"Mean lr_rel in training: {df_train['lr_rel'].mean():.4f} (should be ~1.0)")
    return df_train, df_test, tr_stats


# ---------------------------------------------------------------------------
# EWMA helpers
# ---------------------------------------------------------------------------

def ewma_fbar(df: pd.DataFrame, lam: float) -> np.ndarray:
    """Arithmetic exposure-weighted EWMA of relative LRs (scalar lambda)."""
    num = df["lr_lag1_rel"].values * df["expo_lag1"].values
    den = df["expo_lag1"].values.copy().astype(float)
    for k in range(2, W_MAX + 1):
        w = lam ** (k - 1)
        num = num + df[f"lr_lag{k}_rel"].values * df[f"expo_lag{k}"].values * w
        den = den + df[f"expo_lag{k}"].values * w
    return num / np.maximum(den, 1e-8)


def ewma_fbar_vec(df: pd.DataFrame, lam_vec: np.ndarray) -> np.ndarray:
    """Arithmetic EWMA with per-row lambda vector (continuous-lambda model)."""
    num = df["lr_lag1_rel"].values * df["expo_lag1"].values
    den = df["expo_lag1"].values.copy().astype(float)
    for k in range(2, W_MAX + 1):
        w = lam_vec ** (k - 1)
        num = num + df[f"lr_lag{k}_rel"].values * df[f"expo_lag{k}"].values * w
        den = den + df[f"expo_lag{k}"].values * w
    return num / np.maximum(den, 1e-8)


# ---------------------------------------------------------------------------
# Bühlmann-Straub standard (MoM)
# ---------------------------------------------------------------------------

def fit_bs_standard(df_train: pd.DataFrame) -> dict:
    """Fit B-S standard model using method-of-moments K estimation."""
    # Company summaries: total exposure and exposure-weighted mean relative LR
    grp = df_train.groupby("GRCODE")
    w_i = grp["expo"].sum()
    f_i = (
        df_train.assign(_wlr=df_train["lr_rel"] * df_train["expo"])
        .groupby("GRCODE")["_wlr"].sum()
        / w_i
    )
    ins = pd.DataFrame({"w_i": w_i, "f_i": f_i}).reset_index()

    W_tot     = float(ins["w_i"].sum())
    n_i       = len(ins)
    f_bar_rel = float(np.average(ins["f_i"], weights=ins["w_i"]))
    c_w       = W_tot - float((ins["w_i"] ** 2).sum()) / W_tot

    # Within-company process variance
    s2_parts = []
    for grcode, g in df_train.groupby("GRCODE"):
        wg = g["expo"].values
        fg = g["lr_rel"].values
        f_ig = float(np.average(fg, weights=wg))
        n_g = len(g)
        # Matches R: sum(expo * (lr_rel - wmean)^2) / (n - 1)
        s2 = float(np.sum(wg * (fg - f_ig) ** 2)) / max(n_g - 1, 1)
        s2_parts.append((grcode, float(wg.sum()), s2))

    s2_df = pd.DataFrame(s2_parts, columns=["GRCODE", "w_i", "s2"])
    sigma2_hat = float(np.average(s2_df["s2"], weights=s2_df["w_i"]))

    ss_b  = float(np.average((ins["f_i"] - f_bar_rel) ** 2, weights=ins["w_i"])) * W_tot
    var_b = max((ss_b - (n_i - 1) * sigma2_hat) / c_w, 1e-8)
    K_hat = sigma2_hat / var_b

    print(f"  B-S standard: K = {K_hat:.2f}   f_bar_rel = {f_bar_rel:.4f}")
    return {
        "type": "bs_standard",
        "K_hat": K_hat,
        "f_bar_rel": f_bar_rel,
        "company_stats": ins.rename(columns={"w_i": "w_i_bs", "f_i": "f_i_bs"}).set_index("GRCODE"),
    }


def predict_bs_standard(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict from fitted B-S standard model."""
    stats = fit["company_stats"]
    K    = fit["K_hat"]
    fbar = fit["f_bar_rel"]
    w_i  = df["GRCODE"].map(stats["w_i_bs"]).values.astype(float)
    f_i  = df["GRCODE"].map(stats["f_i_bs"]).values.astype(float)
    Z    = w_i / (w_i + K)
    return (1 - Z) * fbar + Z * f_i


# ---------------------------------------------------------------------------
# Joint-Decay scalar λ (ML)
# ---------------------------------------------------------------------------

def _nll_gamma(params: np.ndarray, df: pd.DataFrame, mode: str = "scalar") -> float:
    """Negative log-likelihood for Joint-Decay model under Gamma(identity link)."""
    alpha, beta, az, bz, log_shape = params[:5]

    if mode == "scalar":
        lam  = float(expit(params[5]))
        fbar = ewma_fbar(df, lam)
    elif mode == "continuous":
        lam_vec = expit(params[5] + params[6] * df["log_mean_expo_sc"].values)
        fbar    = ewma_fbar_vec(df, lam_vec)
    else:  # tercile
        lam_sm, lam_md, lam_lg = expit(params[5]), expit(params[6]), expit(params[7])
        df_reset = df.reset_index(drop=True)
        fbar = np.empty(len(df_reset))
        for trc, lam_t in (("Small", lam_sm), ("Mid", lam_md), ("Large", lam_lg)):
            idx = df_reset.index[df_reset["size_tercile"] == trc].tolist()
            if idx:
                fbar[idx] = ewma_fbar(df_reset.loc[idx], lam_t)

    base  = np.exp(alpha + beta * df["log_expo_sc"].values)
    Z     = expit(az + bz * df["log_expo_used_sc"].values)
    mu    = np.maximum((1 - Z) * base + Z * fbar, 1e-8)
    shape = np.exp(log_shape)
    wt    = df["expo_wt"].values

    ll = gamma_dist.logpdf(df["lr_rel"].values, a=shape, scale=mu / shape)
    return float(-np.sum(ll * wt))


def _multi_optim(obj, x0: np.ndarray, bounds: list, n_starts: int = 5, seed: int = 48):
    rng = np.random.default_rng(seed)
    lo  = np.array([b[0] for b in bounds])
    hi  = np.array([b[1] for b in bounds])

    candidates = [x0]
    for _ in range(n_starts - 1):
        candidates.append(lo + rng.uniform(size=len(lo)) * (hi - lo))

    best = None
    for x in candidates:
        try:
            res = minimize(obj, x, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    return best


def fit_jdecay_scalar(df_train: pd.DataFrame) -> dict:
    """Fit Joint-Decay model with a single scalar decay parameter λ (ML)."""
    x0     = np.array([0.0, 0.0, -1.0, 0.5, 1.0, 0.0])
    bounds = [(-2, 2), (-1, 1), (-6, 3), (-3, 5), (-2, 5), (-6, 6)]

    res = _multi_optim(lambda p: _nll_gamma(p, df_train, "scalar"), x0, bounds)
    p   = res.x
    print(f"  Joint-Decay scalar: alpha={p[0]:.3f}  beta={p[1]:.3f}  "
          f"az={p[2]:.2f}  bz={p[3]:.2f}  lam={float(expit(p[5])):.3f}")
    return {"type": "jdecay_scalar", "par": p}


def predict_jdecay_scalar(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict from fitted scalar-λ Joint-Decay model."""
    p    = fit["par"]
    lam  = float(expit(p[5]))
    fbar = ewma_fbar(df, lam)
    base = np.exp(p[0] + p[1] * df["log_expo_sc"].values)
    Z    = expit(p[2] + p[3] * df["log_expo_used_sc"].values)
    return (1 - Z) * base + Z * fbar


# ---------------------------------------------------------------------------
# Joint-Decay continuous λ (ML)
# ---------------------------------------------------------------------------

def fit_jdecay_continuous(df_train: pd.DataFrame) -> dict:
    """Fit Joint-Decay model where λ varies continuously with account size (ML)."""
    x0     = np.array([0.0, 0.0, -1.0, 0.5, 1.0, 0.0, 0.0])
    bounds = [(-2, 2), (-1, 1), (-6, 3), (-3, 5), (-2, 5), (-6, 6), (-3, 3)]

    res = _multi_optim(lambda p: _nll_gamma(p, df_train, "continuous"), x0, bounds)
    p   = res.x
    print(f"  Joint-Decay continuous: az={p[2]:.2f}  bz={p[3]:.2f}  "
          f"lam0={p[5]:.3f}  lam1={p[6]:.3f}")
    return {"type": "jdecay_continuous", "par": p}


def predict_jdecay_continuous(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict from fitted continuous-λ Joint-Decay model."""
    p       = fit["par"]
    lam_vec = expit(p[5] + p[6] * df["log_mean_expo_sc"].values)
    fbar    = ewma_fbar_vec(df, lam_vec)
    base    = np.exp(p[0] + p[1] * df["log_expo_sc"].values)
    Z       = expit(p[2] + p[3] * df["log_expo_used_sc"].values)
    return (1 - Z) * base + Z * fbar


# ---------------------------------------------------------------------------
# Joint-Decay tercile λ (MLE)
#
# Three free decay parameters (lamSm, lamMd, lamLg) estimated jointly with
# the continuous complement exp(alpha + beta*log_expo_sc) via MLE.
#
# Note: without Bayesian priors, estimates can be noisy for terciles with
# few companies. The R/brms version regularises via Normal(0, 1.5) priors on
# each lam parameter. Results may therefore differ from Table 3.
# ---------------------------------------------------------------------------

def fit_jdecay_tercile(df_train: pd.DataFrame) -> dict:
    """Fit Joint-Decay model with tercile-specific decay parameters (MLE)."""
    x0     = np.array([0.0, 0.0, -1.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    bounds = [(-2, 2), (-1, 1), (-6, 3), (-3, 5), (-2, 5), (-6, 6), (-6, 6), (-6, 6)]

    res = _multi_optim(lambda p: _nll_gamma(p, df_train, "tercile"), x0, bounds)
    p   = res.x
    print(f"  Joint-Decay tercile (MLE): az={p[2]:.2f}  bz={p[3]:.2f}  "
          f"lam_Sm={float(expit(p[5])):.3f}  lam_Md={float(expit(p[6])):.3f}  "
          f"lam_Lg={float(expit(p[7])):.3f}")
    return {"type": "jdecay_tercile_mle", "par": p}


def predict_jdecay_tercile(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict from fitted tercile-λ Joint-Decay model."""
    p       = fit["par"]
    lam_sm  = float(expit(p[5]))
    lam_md  = float(expit(p[6]))
    lam_lg  = float(expit(p[7]))
    base    = np.exp(p[0] + p[1] * df["log_expo_sc"].values)
    Z       = expit(p[2] + p[3] * df["log_expo_used_sc"].values)

    df_r = df.reset_index(drop=True)
    fbar = np.empty(len(df_r))
    for trc, lam_t in (("Small", lam_sm), ("Mid", lam_md), ("Large", lam_lg)):
        idx = df_r.index[df_r["size_tercile"] == trc].tolist()
        if idx:
            fbar[idx] = ewma_fbar(df_r.loc[idx], lam_t)

    return (1 - Z) * base + Z * fbar


# ---------------------------------------------------------------------------
# MAP estimation — paper Appendix, Listing 4
#
# Adds weakly-informative log-prior penalties to the MLE objective.
# Useful when lambda is weakly identified (small portfolios or few companies
# per tercile). The tercile MAP is particularly helpful when lam_Md hits the
# boundary (~1.0) in the MLE fit.
#
# Priors:
#   Normal(0, 1)   on alpha, beta, az, bz (logit-scale parameters)
#   Normal(2, 1)   on log_shape
#   Normal(0, 1.5) on each lam_raw (matches brms prior in proposed model)
# ---------------------------------------------------------------------------

from scipy.stats import norm as _norm


def _log_prior_scalar(params: np.ndarray) -> float:
    """Log-prior for scalar-lambda MAP model (6 parameters)."""
    alpha, beta, az, bz, log_shape, lam_raw = params
    return (
        _norm.logpdf(alpha,    0, 1.0) +
        _norm.logpdf(beta,     0, 1.0) +
        _norm.logpdf(az,       0, 1.0) +
        _norm.logpdf(bz,       0, 1.0) +
        _norm.logpdf(log_shape, 2, 1.0) +
        _norm.logpdf(lam_raw,  0, 1.5)
    )


def _log_prior_tercile(params: np.ndarray) -> float:
    """Log-prior for tercile-lambda MAP model (8 parameters)."""
    alpha, beta, az, bz, log_shape = params[:5]
    lam_sm_r, lam_md_r, lam_lg_r  = params[5], params[6], params[7]
    return (
        _norm.logpdf(alpha,    0, 1.0) +
        _norm.logpdf(beta,     0, 1.0) +
        _norm.logpdf(az,       0, 1.0) +
        _norm.logpdf(bz,       0, 1.0) +
        _norm.logpdf(log_shape, 2, 1.0) +
        _norm.logpdf(lam_sm_r, 0, 1.5) +
        _norm.logpdf(lam_md_r, 0, 1.5) +
        _norm.logpdf(lam_lg_r, 0, 1.5)
    )


def fit_jdecay_scalar_map(df_train: pd.DataFrame) -> dict:
    """Fit Joint-Decay scalar-lambda model via MAP (MLE + weakly-informative priors)."""
    def nlp(params):
        return _nll_gamma(params, df_train, "scalar") - _log_prior_scalar(params)

    x0     = np.array([0.0, 0.0, -1.0, 0.5, 2.0, 0.0])
    bounds = [(-2, 2), (-1, 1), (-6, 3), (-3, 5), (-2, 5), (-6, 6)]
    res = _multi_optim(nlp, x0, bounds)
    p   = res.x
    print(f"  Joint-Decay scalar (MAP): az={p[2]:.2f}  bz={p[3]:.2f}  lam={float(expit(p[5])):.3f}")
    return {"type": "jdecay_scalar_map", "par": p}


# Predictions are identical structure to MLE scalar model
def predict_jdecay_scalar_map(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict from MAP scalar-lambda model."""
    p    = fit["par"]
    lam  = float(expit(p[5]))
    fbar = ewma_fbar(df, lam)
    base = np.exp(p[0] + p[1] * df["log_expo_sc"].values)
    Z    = expit(p[2] + p[3] * df["log_expo_used_sc"].values)
    return (1 - Z) * base + Z * fbar


def fit_jdecay_tercile_map(df_train: pd.DataFrame) -> dict:
    """Fit Joint-Decay tercile-lambda model via MAP (MLE + weakly-informative priors).

    Regularises lambda estimates away from the boundary — particularly useful
    when lam_Md hits ~1.0 in the MLE fit.
    """
    def nlp(params):
        return _nll_gamma(params, df_train, "tercile") - _log_prior_tercile(params)

    x0     = np.array([0.0, 0.0, -1.0, 0.5, 2.0, 0.0, 0.0, 0.0])
    bounds = [(-2, 2), (-1, 1), (-6, 3), (-3, 5), (-2, 5), (-6, 6), (-6, 6), (-6, 6)]
    res = _multi_optim(nlp, x0, bounds)
    p   = res.x
    print(f"  Joint-Decay tercile (MAP): az={p[2]:.2f}  bz={p[3]:.2f}  "
          f"lam_Sm={float(expit(p[5])):.3f}  lam_Md={float(expit(p[6])):.3f}  "
          f"lam_Lg={float(expit(p[7])):.3f}")
    return {"type": "jdecay_tercile_map", "par": p}


def predict_jdecay_tercile_map(fit: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict from MAP tercile-lambda model."""
    return predict_jdecay_tercile(fit, df)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def eval_model(pred_rel: np.ndarray, actual_rel: np.ndarray,
               expo: np.ndarray, mu_t: np.ndarray, label: str) -> dict:
    """Compute wMSE, normalised Gini, and calibration slope (absolute scale)."""
    pred_abs   = pred_rel * mu_t
    actual_abs = actual_rel * mu_t
    w          = expo

    # wMSE
    wmse = float(np.average((pred_abs - actual_abs) ** 2, weights=w))

    # Normalised Gini
    ord_pred = np.argsort(pred_abs)
    cum_w    = np.cumsum(w[ord_pred]) / w.sum()
    cum_loss = np.cumsum(w[ord_pred] * actual_abs[ord_pred]) / (w * actual_abs).sum()
    gini_raw = 2 * float(np.trapezoid(cum_loss, cum_w)) - 1

    ord_or      = np.argsort(actual_abs)
    cw_or       = np.cumsum(w[ord_or]) / w.sum()
    cl_or       = np.cumsum(w[ord_or] * actual_abs[ord_or]) / (w * actual_abs).sum()
    gini_oracle = 2 * float(np.trapezoid(cl_or, cw_or)) - 1
    gini_pct    = 100 * gini_raw / gini_oracle if abs(gini_oracle) > 1e-8 else float("nan")

    # Calibration slope (WLS: actual ~ pred, weights = expo)
    sw  = w.sum()
    sx  = (w * pred_abs).sum()
    sy  = (w * actual_abs).sum()
    sxx = (w * pred_abs ** 2).sum()
    sxy = (w * pred_abs * actual_abs).sum()
    denom = sw * sxx - sx ** 2
    slope = float((sw * sxy - sx * sy) / denom) if abs(denom) > 1e-16 else float("nan")

    return {
        "Model":    label,
        "wMSE":     round(wmse, 6),
        "Gini_pct": round(gini_pct, 1),
        "Slope":    round(slope, 3),
    }


def eval_all_models(df_test: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """Evaluate multiple models and return a comparison DataFrame.

    Args:
        df_test: Test set DataFrame (must contain lr_rel, expo, mu_t).
        predictions: Dict mapping model label -> pred_rel array.
    """
    rows = [
        eval_model(
            pred_rel   = pred_rel,
            actual_rel = df_test["lr_rel"].values,
            expo       = df_test["expo"].values,
            mu_t       = df_test["mu_t"].values,
            label      = label,
        )
        for label, pred_rel in predictions.items()
    ]
    results = pd.DataFrame(rows)

    bs_rows = results[results["Model"].str.contains("Bühlmann-Straub", regex=False)]
    if len(bs_rows) == 1:
        bs_wmse = float(bs_rows["wMSE"].iloc[0])
        results["pct_vs_bs"] = ((results["wMSE"] - bs_wmse) / bs_wmse * 100).round(1)

    return results


# ---------------------------------------------------------------------------
# Z-curve plot
# ---------------------------------------------------------------------------

def plot_zcurves(fit_scalar: dict, fit_bs: dict, df_train: pd.DataFrame,
                 lob_label: str = "", ax=None):
    """Plot logistic Z vs log exposure (standardised scale)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    grid = np.linspace(
        float(df_train["log_expo_used_sc"].min()),
        float(df_train["log_expo_used_sc"].max()),
        200,
    )

    # B-S: constant pooled Z
    K_bs = fit_bs["K_hat"]
    bs_z = float((df_train["expo_used"] / (df_train["expo_used"] + K_bs)).mean())
    ax.axhline(bs_z, linestyle="--", color="tab:blue", label="B-S (pooled Z)")

    # Joint-Decay scalar λ
    p = fit_scalar["par"]
    z_scalar = expit(p[2] + p[3] * grid)
    ax.plot(grid, z_scalar, color="tab:orange", label="Joint-Decay (scalar λ)")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Log exposure used (standardised)")
    ax.set_ylabel("Credibility weight Z")
    ax.set_title(f"Fitted Z curves — {lob_label}")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    return ax
