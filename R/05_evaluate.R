# =============================================================================
# 05_evaluate.R  —  Evaluation metrics
#
# Defines:
#   eval_model(pred_rel, actual_rel, expo, label)  — wMSE, Gini, slope
#   eval_all_models(df_test, mu_t_col)             — run eval for all models
#   bootstrap_wmse_improvement(df_test, B)         — company-level pairs bootstrap CI
#
# All metrics are computed on absolute scale: pred_abs = pred_rel * mu_t.
# wMSE is NEP-weighted (weights accounts by premium volume, consistent with
# the B-S variance assumption Var(LR_it) proportional to 1/E_it).
# =============================================================================

library(dplyr)

# Model labels — order matches Table 3 in the paper
MODEL_LABELS <- c(
  "Market Mean (baseline)"                       = "pred_market_mean",
  "Last Year LR (naive)"                         = "pred_last_year",
  "Bühlmann-Straub (standard)"              = "pred_bs_standard",
  "B-S (best sequential patch)"                  = "pred_bs_best_patch",
  "GLMM (random intercept + size)"               = "pred_glmm",
  "Joint-Decay (scalar λ)"                  = "pred_jdecay_scalar",
  "Joint-Decay (continuous λ)"              = "pred_jdecay_cont",
  "Joint-Decay (tercile λ) [proposed]"      = "pred_jdecay_tercile"
)

# -----------------------------------------------------------------------------
# Single-model evaluation
# -----------------------------------------------------------------------------

eval_model <- function(pred_rel, actual_rel, expo, label, mu_t) {
  # Convert to absolute scale for wMSE (pred_rel already in relative space)
  pred_abs   <- pred_rel * mu_t
  actual_abs <- actual_rel * mu_t

  # wMSE: exposure-weighted mean squared error
  wmse <- sum(expo * (pred_abs - actual_abs)^2) / sum(expo)

  # Gini (normalised): ranks predictions and measures concentration of actual losses
  ord      <- order(pred_abs)
  cum_w    <- cumsum(expo[ord]) / sum(expo)
  cum_loss <- cumsum(expo[ord] * actual_abs[ord]) / sum(expo * actual_abs)
  gini_raw <- 2 * (sum(diff(cum_w) * (head(cum_loss, -1) + tail(cum_loss, -1)) / 2) - 0.5)

  ord_or   <- order(actual_abs)
  cw_or    <- cumsum(expo[ord_or]) / sum(expo)
  cl_or    <- cumsum(expo[ord_or] * actual_abs[ord_or]) / sum(expo * actual_abs)
  gini_oracle <- 2 * (sum(diff(cw_or) * (head(cl_or, -1) + tail(cl_or, -1)) / 2) - 0.5)
  gini_pct    <- ifelse(abs(gini_oracle) > 1e-8, 100 * gini_raw / gini_oracle, NA_real_)

  # Calibration slope: ideal = 1.0; < 1 = over-credentialled (predictions too spread)
  slope <- coef(lm(actual_abs ~ pred_abs, weights = expo))[2]

  tibble(
    Model    = label,
    wMSE     = round(wmse, 6),
    Gini_pct = round(gini_pct, 1),
    Slope    = round(slope, 3)
  )
}

# -----------------------------------------------------------------------------
# Evaluate all models on the test set
# -----------------------------------------------------------------------------

eval_all_models <- function(df_test) {
  results <- bind_rows(lapply(names(MODEL_LABELS), function(label) {
    col <- MODEL_LABELS[label]
    if (!col %in% names(df_test)) return(NULL)
    eval_model(
      pred_rel   = df_test[[col]],
      actual_rel = df_test$lr_rel,
      expo       = df_test$expo,
      label      = label,
      mu_t       = df_test$mu_t
    )
  }))

  # Compute % improvement vs B-S standard
  bs_wmse <- results$wMSE[results$Model == "Bühlmann-Straub (standard)"]
  if (length(bs_wmse) == 1) {
    results <- results %>%
      mutate(pct_vs_bs = round(100 * (wMSE - bs_wmse) / bs_wmse, 1))
  }

  results
}

# -----------------------------------------------------------------------------
# Evaluation by size tercile
# -----------------------------------------------------------------------------

eval_by_tercile <- function(df_test) {
  bind_rows(lapply(c("Small", "Mid", "Large"), function(trc) {
    df_t <- df_test %>% filter(size_tercile == trc)
    bind_rows(lapply(names(MODEL_LABELS), function(label) {
      col <- MODEL_LABELS[label]
      if (!col %in% names(df_t)) return(NULL)
      eval_model(df_t[[col]], df_t$lr_rel, df_t$expo, label, df_t$mu_t) %>%
        mutate(Tercile = trc, .before = 1)
    }))
  }))
}

# -----------------------------------------------------------------------------
# Bootstrap CI on wMSE improvement vs B-S standard
#
# Company-level pairs bootstrap: resample companies with replacement,
# preserving each company's two test-year observations.
# Returns 90% percentile interval for % improvement over B-S standard.
# -----------------------------------------------------------------------------

bootstrap_wmse_improvement <- function(df_test, B = 2000, seed = 48) {
  set.seed(seed)
  companies <- unique(df_test$GRCODE)
  n_co      <- length(companies)

  bs_col <- MODEL_LABELS["Bühlmann-Straub (standard)"]
  model_cols <- setdiff(MODEL_LABELS, bs_col)

  boot_mat <- matrix(NA_real_, nrow = B, ncol = length(model_cols),
                     dimnames = list(NULL, names(model_cols)))

  for (b in seq_len(B)) {
    co_sample <- sample(companies, n_co, replace = TRUE)
    df_b <- bind_rows(lapply(co_sample, function(co) df_test[df_test$GRCODE == co, ]))

    bs_wmse_b <- sum(df_b$expo * (df_b[[bs_col]] * df_b$mu_t - df_b$lr_rel * df_b$mu_t)^2) /
                 sum(df_b$expo)

    for (j in seq_along(model_cols)) {
      col <- model_cols[j]
      if (!col %in% names(df_b)) next
      mod_wmse_b <- sum(df_b$expo * (df_b[[col]] * df_b$mu_t - df_b$lr_rel * df_b$mu_t)^2) /
                    sum(df_b$expo)
      boot_mat[b, j] <- 100 * (mod_wmse_b - bs_wmse_b) / bs_wmse_b
    }
  }

  bind_rows(lapply(seq_along(model_cols), function(j) {
    ci <- quantile(boot_mat[, j], c(0.05, 0.95), na.rm = TRUE)
    tibble(
      Model     = names(model_cols)[j],
      ci_lo_pct = round(ci[1], 1),
      ci_hi_pct = round(ci[2], 1)
    )
  }))
}
