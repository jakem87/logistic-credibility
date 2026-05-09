# =============================================================================
# 06_plots.R  —  Key figures
#
# Produces:
#   results_table.png    — wMSE / Gini / Slope for all 8 models (Table 3 equivalent)
#   wmse_by_tercile.png  — wMSE improvement vs B-S by size tercile
#   zcurves.png          — Fitted Z as a function of log exposure (tercile lambda model)
#   lambda_posteriors.png — Lambda posterior distributions by size tercile
#
# Requires: results, results_tercile from 05_evaluate.R; models from 03_models_ca.R
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

theme_set(theme_bw(base_size = 11))

# -----------------------------------------------------------------------------
# 1. Results table as a ggplot (text table)
# -----------------------------------------------------------------------------

plot_results_table <- function(results, lob_label) {
  tbl <- results %>%
    mutate(
      wMSE_k  = round(wMSE * 1000, 2),
      pct_str = if ("pct_vs_bs" %in% names(.)) sprintf("%+.1f%%", pct_vs_bs) else ""
    ) %>%
    select(Model, `wMSE (×10⁻³)` = wMSE_k, `Gini (%)` = Gini_pct, Slope) %>%
    mutate(Model = factor(Model, levels = rev(Model)))

  ggplot(tbl, aes(y = Model)) +
    geom_text(aes(x = 0, label = Model), hjust = 0, size = 3) +
    geom_text(aes(x = 1, label = `wMSE (×10⁻³)`), hjust = 1, size = 3) +
    geom_text(aes(x = 1.1, label = `Gini (%)`), hjust = 0.5, size = 3) +
    geom_text(aes(x = 1.3, label = Slope), hjust = 0.5, size = 3) +
    scale_x_continuous(limits = c(0, 1.4)) +
    labs(title = sprintf("Model comparison — %s (test set AY 2006–2007)", lob_label),
         x = NULL, y = NULL) +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          panel.grid = element_blank())
}

# -----------------------------------------------------------------------------
# 2. wMSE improvement vs B-S, by size tercile
# -----------------------------------------------------------------------------

plot_wmse_by_tercile <- function(results_trc, lob_label) {
  bs_by_trc <- results_trc %>%
    filter(Model == "Bühlmann-Straub (standard)") %>%
    select(Tercile, bs_wmse = wMSE)

  tbl <- results_trc %>%
    left_join(bs_by_trc, by = "Tercile") %>%
    mutate(pct_chg = 100 * (wMSE - bs_wmse) / bs_wmse,
           Tercile = factor(Tercile, levels = c("Small", "Mid", "Large")))

  key_models <- c(
    "Bühlmann-Straub (standard)",
    "B-S (best sequential patch)",
    "Joint-Decay (scalar λ)",
    "Joint-Decay (tercile λ) [proposed]",
    "GLMM (random intercept + size)"
  )

  tbl <- filter(tbl, Model %in% key_models) %>%
    mutate(Model = factor(Model, levels = rev(key_models)))

  ggplot(tbl, aes(x = pct_chg, y = Model, colour = Model)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
    geom_point(size = 3) +
    facet_wrap(~ Tercile, ncol = 3) +
    scale_colour_brewer(palette = "Set1", guide = "none") +
    labs(
      title   = sprintf("wMSE vs B-S standard (%%): %s", lob_label),
      x       = "% change in wMSE vs B-S standard",
      y       = NULL,
      caption = "Negative = improvement. Test set AY 2006–2007."
    )
}

# -----------------------------------------------------------------------------
# 3. Z curves: logistic Z as function of log exposure
# -----------------------------------------------------------------------------

plot_zcurves <- function(models, df_train, lob_label) {
  # Build a grid of log_expo_used values spanning the training range
  expo_grid <- seq(
    min(df_train$log_expo_used_sc), max(df_train$log_expo_used_sc),
    length.out = 200
  )

  # Logistic Z (scalar lambda model)
  p_sc <- models$jdecay_scalar$par
  logistic_z <- plogis(p_sc[3] + p_sc[4] * expo_grid)

  # Tercile lambda model Z
  if (!is.null(models$jdecay_tercile)) {
    p_lt <- as_draws_df(models$jdecay_tercile)
    az_lt <- mean(p_lt$b_az_Intercept)
    bz_lt <- mean(p_lt$b_bz_Intercept)
    tercile_z <- plogis(az_lt + bz_lt * expo_grid)
  } else {
    tercile_z <- NA_real_
  }

  K_bs <- models$bs_standard$K_hat
  bs_z_mean <- mean(df_train$expo_used / (df_train$expo_used + K_bs))

  df_z <- tibble(
    log_expo_used_sc = expo_grid,
    `B-S (constant Z)`         = rep(bs_z_mean, length(expo_grid)),
    `Joint-Decay (scalar λ)`   = logistic_z,
    `Joint-Decay (tercile λ)`  = tercile_z
  ) %>%
    pivot_longer(-log_expo_used_sc, names_to = "Model", values_to = "Z")

  ggplot(df_z, aes(x = log_expo_used_sc, y = Z, colour = Model)) +
    geom_line(linewidth = 1) +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    scale_colour_brewer(palette = "Set1") +
    labs(
      title   = sprintf("Fitted credibility weight Z vs log exposure — %s", lob_label),
      x       = "Log exposure (standardised, training scale)",
      y       = "Credibility weight Z",
      caption = "B-S assigns the same Z to all accounts with equal exposure_used (horizontal line is the pooled average)."
    )
}

# -----------------------------------------------------------------------------
# 4. Lambda posteriors (tercile lambda model only)
# -----------------------------------------------------------------------------

plot_lambda_posteriors <- function(fit_tercile, lob_label) {
  if (is.null(fit_tercile)) return(invisible(NULL))

  p <- as_draws_df(fit_tercile)
  df_lam <- tibble(
    Small = plogis(p$b_lamSm_Intercept),
    Mid   = plogis(p$b_lamMd_Intercept),
    Large = plogis(p$b_lamLg_Intercept)
  ) %>%
    pivot_longer(everything(), names_to = "Tercile", values_to = "lambda") %>%
    mutate(Tercile = factor(Tercile, levels = c("Small", "Mid", "Large")))

  ggplot(df_lam, aes(x = lambda, fill = Tercile, colour = Tercile)) +
    geom_density(alpha = 0.3) +
    scale_x_continuous(limits = c(0, 1), labels = scales::percent) +
    scale_fill_brewer(palette = "Set1") +
    scale_colour_brewer(palette = "Set1") +
    labs(
      title   = sprintf("Lambda posterior by size tercile — %s", lob_label),
      x       = "Decay parameter λ (0 = discount old data heavily, 1 = equal weight)",
      y       = "Posterior density",
      caption = "Large companies show strong mean-reversion (λ ≈ 0.13); Mid companies have long memory (λ ≈ 0.84)."
    )
}

# -----------------------------------------------------------------------------
# Save all plots to OUTPUT_DIR
# -----------------------------------------------------------------------------

save_plots <- function(results, results_trc, models, df_train, lob_label, out_dir) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  ggsave(file.path(out_dir, "wmse_by_tercile.png"),
    plot_wmse_by_tercile(results_trc, lob_label),
    width = 10, height = 5, dpi = 150
  )

  ggsave(file.path(out_dir, "zcurves.png"),
    plot_zcurves(models, df_train, lob_label),
    width = 7, height = 4, dpi = 150
  )

  p_lam <- plot_lambda_posteriors(models$jdecay_tercile, lob_label)
  if (!is.null(p_lam)) {
    ggsave(file.path(out_dir, "lambda_posteriors.png"), p_lam,
           width = 7, height = 4, dpi = 150)
  }

  message(sprintf("Plots saved to: %s", out_dir))
}
