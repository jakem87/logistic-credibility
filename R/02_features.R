# =============================================================================
# 02_features.R  —  Feature engineering helpers
#
# Defines:
#   ewma_fbar(df, lam)       — arithmetic exposure-weighted EWMA of relative LR
#   ewma_log_fbar(df, lam)   — log-space version (for geometric blend)
#   multi_optim(obj, ...)    — multi-start L-BFGS-B wrapper
#
# These helpers are used by 03_models_ca.R and 04_models_ol.R.
# Source this file after 01_data.R has been run.
# =============================================================================

# -----------------------------------------------------------------------------
# EWMA helpers (7-lag, zero-fill safe)
#
# Pre-entry lags have expo = 0, so their terms vanish regardless of lr_rel value.
# lam is the decay weight: lag-k contribution is weighted by lam^(k-1).
# lam = 1 gives equal-weight averaging (equivalent to classical B-S).
# -----------------------------------------------------------------------------

ewma_fbar <- function(df, lam) {
  # Arithmetic blend: fbar = sum(expo_lag_k * lam^(k-1) * lr_rel_lag_k) / sum(expo_lag_k * lam^(k-1))
  num <- df$lr_lag1_rel * df$expo_lag1 +
         df$lr_lag2_rel * df$expo_lag2 * lam +
         df$lr_lag3_rel * df$expo_lag3 * lam^2 +
         df$lr_lag4_rel * df$expo_lag4 * lam^3 +
         df$lr_lag5_rel * df$expo_lag5 * lam^4 +
         df$lr_lag6_rel * df$expo_lag6 * lam^5 +
         df$lr_lag7_rel * df$expo_lag7 * lam^6
  den <- df$expo_lag1 +
         df$expo_lag2 * lam   + df$expo_lag3 * lam^2 +
         df$expo_lag4 * lam^3 + df$expo_lag5 * lam^4 +
         df$expo_lag6 * lam^5 + df$expo_lag7 * lam^6
  num / pmax(den, 1e-8)
}

ewma_log_fbar <- function(df, lam) {
  # Log-space EWMA: used in geometric-blend models (Gamma log link)
  num <- log(pmax(df$lr_lag1_rel, 1e-8)) * df$expo_lag1 +
         log(pmax(df$lr_lag2_rel, 1e-8)) * df$expo_lag2 * lam +
         log(pmax(df$lr_lag3_rel, 1e-8)) * df$expo_lag3 * lam^2 +
         log(pmax(df$lr_lag4_rel, 1e-8)) * df$expo_lag4 * lam^3 +
         log(pmax(df$lr_lag5_rel, 1e-8)) * df$expo_lag5 * lam^4 +
         log(pmax(df$lr_lag6_rel, 1e-8)) * df$expo_lag6 * lam^5 +
         log(pmax(df$lr_lag7_rel, 1e-8)) * df$expo_lag7 * lam^6
  den <- df$expo_lag1 +
         df$expo_lag2 * lam   + df$expo_lag3 * lam^2 +
         df$expo_lag4 * lam^3 + df$expo_lag5 * lam^4 +
         df$expo_lag6 * lam^5 + df$expo_lag7 * lam^6
  num / pmax(den, 1e-8)
}

# -----------------------------------------------------------------------------
# Vectorised EWMA (per-row lambda vector)
#
# Used when lambda varies by account (continuous lambda model).
# lam_vec is a vector of length nrow(df), one decay value per row.
# -----------------------------------------------------------------------------

ewma_fbar_vec <- function(df, lam_vec) {
  l1 <- lam_vec; l2 <- lam_vec^2; l3 <- lam_vec^3
  l4 <- lam_vec^4; l5 <- lam_vec^5; l6 <- lam_vec^6
  num <- df$lr_lag1_rel * df$expo_lag1 +
         df$lr_lag2_rel * df$expo_lag2 * l1 +
         df$lr_lag3_rel * df$expo_lag3 * l2 +
         df$lr_lag4_rel * df$expo_lag4 * l3 +
         df$lr_lag5_rel * df$expo_lag5 * l4 +
         df$lr_lag6_rel * df$expo_lag6 * l5 +
         df$lr_lag7_rel * df$expo_lag7 * l6
  den <- df$expo_lag1 +
         df$expo_lag2 * l1 + df$expo_lag3 * l2 +
         df$expo_lag4 * l3 + df$expo_lag5 * l4 +
         df$expo_lag6 * l5 + df$expo_lag7 * l6
  num / pmax(den, 1e-8)
}

# -----------------------------------------------------------------------------
# Multi-start optimiser
#
# Runs L-BFGS-B from N_STARTS random starting points and returns the best
# (lowest objective value). This reduces sensitivity to local optima in the
# non-linear credibility likelihoods.
# -----------------------------------------------------------------------------

N_STARTS <- 5

multi_optim <- function(obj, par_init, lower, upper, data, n_starts = N_STARTS) {
  best <- NULL
  # First try: use provided starting values
  starts <- vector("list", n_starts)
  starts[[1]] <- par_init
  # Remaining: random perturbations within bounds
  for (k in 2:n_starts) {
    starts[[k]] <- lower + runif(length(lower)) * (upper - lower)
  }
  for (s in starts) {
    res <- tryCatch(
      optim(s, obj, data = data, method = "L-BFGS-B",
            lower = lower, upper = upper,
            control = list(maxit = 2000)),
      error = function(e) list(value = Inf)
    )
    if (is.null(best) || res$value < best$value) best <- res
  }
  best
}
