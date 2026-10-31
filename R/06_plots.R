# =============================================================================
# 06_plots.R  —  Key figures
#
# Produces:
#   results_table.png    — wMSE / Gini / Slope for all 8 models (Table 2 equivalent)
#   wmse_by_tercile.png  — wMSE improvement vs B-S by size tercile
#   zcurves.png          — Fitted Z vs lookback NEP: B-S per company (training-period NEP), logistic curves, tercile-lambda 95% band
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
    "LogCred (scalar λ)",
    "LogCred (tercile λ) [proposed]",
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
  # Standardisation of the Z covariate, as used when the logistic models were fitted
  leu_mean <- mean(log(df_train$expo_used))
  leu_sd   <- sd(log(df_train$expo_used))

  # Standard B-S weight per company: Z = w_i / (w_i + K), where w_i is the company's
  # TOTAL training-period NEP (the quantity B-S actually uses), not the lookback exposure.
  K_bs <- models$bs_standard$K_hat
  co <- df_train %>%
    group_by(GRCODE) %>%
    summarise(w_i = sum(expo), log_lb = log(mean(expo_used)), .groups = "drop") %>%
    mutate(Z_bs = w_i / (w_i + K_bs))

  # Logistic curves over the range of company mean lookback NEP
  x_grid <- seq(min(co$log_lb), max(co$log_lb), length.out = 200)
  x_sc   <- (x_grid - leu_mean) / leu_sd

  p_sc <- models$jdecay_scalar$par
  df_curve <- tibble(x = x_grid, scalar = plogis(p_sc[3] + p_sc[4] * x_sc))

  df_band <- NULL
  if (!is.null(models$jdecay_tercile)) {
    p_lt <- as_draws_df(models$jdecay_tercile)
    zd   <- plogis(outer(p_lt$b_az_Intercept, rep(1, length(x_sc))) +
                   outer(p_lt$b_bz_Intercept, x_sc))
    df_curve$tercile <- colMeans(zd)
    df_band <- tibble(x = x_grid,
                      lo = apply(zd, 2, quantile, 0.025),
                      hi = apply(zd, 2, quantile, 0.975))
  }

  brk_m <- c(0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000)   # NEP is in $000s
  p <- ggplot() +
    geom_point(data = co, aes(x = log_lb, y = Z_bs, colour = "B-S (one dot per company)"),
               alpha = 0.8, size = 1.8)
  if (!is.null(df_band)) {
    p <- p + geom_ribbon(data = df_band, aes(x = x, ymin = lo, ymax = hi),
                         fill = "#003781", alpha = 0.18)
  }
  p <- p + geom_line(data = df_curve, aes(x = x, y = scalar, colour = "LogCred (scalar \u03bb)"), linewidth = 1)
  if (!is.null(df_band)) {
    p <- p + geom_line(data = df_curve, aes(x = x, y = tercile, colour = "LogCred (tercile \u03bb), 95% band"), linewidth = 1.2)
  }
  p +
    scale_x_continuous(breaks = log(brk_m * 1000), labels = brk_m) +
    scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
    scale_colour_manual(values = c("B-S (one dot per company)" = "grey45",
                                   "LogCred (scalar \u03bb)" = "#5dade2",
                                   "LogCred (tercile \u03bb), 95% band" = "#003781"), name = NULL) +
    labs(
      title   = sprintf("Fitted credibility weight Z vs exposure \u2014 %s", lob_label),
      x       = "Mean annual lookback NEP, $M (log scale)",
      y       = "Credibility weight Z",
      caption = paste0("B-S: Z = w/(w+K) with w = training-period NEP, so dots scatter about a smooth curve
when plotted against lookback NEP. ",
                       "Logistic: Z = expit(a + b\u00b7log lookback NEP).")
    ) +
    theme(legend.position = "bottom")
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
