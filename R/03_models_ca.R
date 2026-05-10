# =============================================================================
# 03_models_ca.R  —  Model fitting: Commercial Auto
#
# Fits the 8 models from Table 3 of the paper. All models operate in relative
# space (response = lr_rel = lr / mu_t). Predictions are converted back to
# absolute scale at evaluation time via pred_abs = pred_rel * mu_t.
#
# Models:
#   fit_bs_standard()          B-S standard (MoM K, full history)
#   fit_bs_best_patch()        B-S stratified K + size comp + EWMA tercile lambda
#   fit_glmm()                 GLMM (random intercept + size fixed effect)
#   fit_jdecay_scalar()        Joint-Decay (scalar lambda, ML)
#   fit_jdecay_continuous()    Joint-Decay (continuous lambda, ML)
#   fit_jdecay_tercile()       Joint-Decay (tercile lambda, continuous comp, Bayesian)  <- proposed
#
# Requires: df_train, df_test from 01_data.R; helpers from 02_features.R.
# =============================================================================

library(glmmTMB)
library(brms)
options(mc.cores = parallel::detectCores())

# REFIT = FALSE: load cached Bayesian RDS files if they exist (saves ~10 min per model)
# REFIT = TRUE:  refit all Bayesian models from scratch
if (!exists("REFIT")) REFIT <- FALSE

# -----------------------------------------------------------------------------
# B-S standard: Bühlmann-Straub with MoM K (full training history)
#
# Z_i = w_i / (w_i + K)  where w_i = total NEP over all training years
# K = sigma2_within / var_between (method of moments)
# -----------------------------------------------------------------------------

fit_bs_standard <- function(df) {
  ins <- df %>%
    group_by(GRCODE) %>%
    summarise(
      w_i = sum(expo),
      f_i = sum(lr_rel * expo) / sum(expo),
      .groups = "drop"
    )
  W_tot     <- sum(ins$w_i)
  n_i       <- nrow(ins)
  f_bar_rel <- sum(ins$w_i * ins$f_i) / W_tot
  c_w       <- W_tot - sum(ins$w_i^2) / W_tot

  sigma2_hat <- df %>%
    group_by(GRCODE) %>%
    summarise(
      w_i  = sum(expo),
      s2_i = sum(expo * (lr_rel - sum(expo * lr_rel) / sum(expo))^2) / (n() - 1),
      .groups = "drop"
    ) %>%
    summarise(s2 = sum(w_i * s2_i) / sum(w_i)) %>%
    pull(s2)

  ss_b   <- sum(ins$w_i * (ins$f_i - f_bar_rel)^2)
  var_b  <- max((ss_b - (n_i - 1) * sigma2_hat) / c_w, 1e-8)
  K_hat  <- sigma2_hat / var_b

  message(sprintf("  B-S standard: K = %.2f   f_bar_rel = %.4f", K_hat, f_bar_rel))
  list(type = "bs_standard", K_hat = K_hat, f_bar_rel = f_bar_rel,
       company_stats = ins %>% rename(w_i_bs = w_i, f_i_bs = f_i))
}

# Predict from a fitted B-S standard model
predict_bs_standard <- function(fit, df_new) {
  df_new %>%
    left_join(fit$company_stats, by = "GRCODE") %>%
    mutate(
      Z_i      = w_i_bs / (w_i_bs + fit$K_hat),
      pred_rel = (1 - Z_i) * fit$f_bar_rel + Z_i * f_i_bs
    ) %>%
    pull(pred_rel)
}

# -----------------------------------------------------------------------------
# B-S stratified K: MoM K estimated separately within each size tercile,
# combined with size-varying complement and EWMA with tercile-specific lambda.
# This is the strongest sequential-patching competitor to the logistic model.
# -----------------------------------------------------------------------------

fit_bs_best_patch <- function(df) {
  # Step 1: stratified K by size tercile
  strat_stats <- df %>%
    group_by(size_tercile, GRCODE) %>%
    summarise(
      w_i = sum(expo),
      f_i = sum(lr_rel * expo) / sum(expo),
      .groups = "drop"
    )

  bs_strat <- strat_stats %>%
    group_by(size_tercile) %>%
    group_modify(function(ins, key) {
      W_tot     <- sum(ins$w_i)
      n_i       <- nrow(ins)
      f_bar_rel <- sum(ins$w_i * ins$f_i) / W_tot
      c_w       <- W_tot - sum(ins$w_i^2) / W_tot

      sigma2_hat <- df %>%
        filter(size_tercile == key$size_tercile) %>%
        group_by(GRCODE) %>%
        summarise(
          w_i  = sum(expo),
          s2_i = sum(expo * (lr_rel - sum(expo * lr_rel) / sum(expo))^2) / (n() - 1),
          .groups = "drop"
        ) %>%
        summarise(s2 = sum(w_i * s2_i) / sum(w_i)) %>%
        pull(s2)

      ss_b  <- sum(ins$w_i * (ins$f_i - f_bar_rel)^2)
      var_b <- max((ss_b - (n_i - 1) * sigma2_hat) / c_w, 1e-8)
      tibble(K_hat = sigma2_hat / var_b, f_bar_rel = f_bar_rel)
    }) %>%
    ungroup()

  # Step 2: implied-complement calibration (matches paper).
  # Compute Z_bs at lam=1 (flat weights), back-solve the implied complement for
  # each obs, then fit log-linear WLS: log(comp_implied) ~ log_expo_sc.
  # This ensures the complement is calibrated consistently with Z.
  df2 <- df %>%
    left_join(strat_stats %>% rename(w_i_strat = w_i, f_i_strat = f_i),
              by = c("GRCODE", "size_tercile")) %>%
    left_join(bs_strat %>% select(size_tercile, K_hat), by = "size_tercile") %>%
    mutate(Z_bs = w_i_strat / (w_i_strat + K_hat))

  df3 <- df2 %>%
    mutate(comp_implied = (lr_rel - Z_bs * past_lr_rel) / pmax(1 - Z_bs, 0.01)) %>%
    filter(comp_implied > 0)

  comp_fit <- lm(log(comp_implied) ~ log_expo_sc,
                 data    = df3,
                 weights = pmax(1 - df3$Z_bs, 0.01) * df3$expo)
  alpha_c <- coef(comp_fit)[1]
  beta_c  <- coef(comp_fit)[2]

  # Step 3: single scalar λ estimated by 1D search minimising training wMSE.
  # Matches paper's "B-S best sequential patch": strat K + size comp + EWMA,
  # scalar λ pooled across all terciles. This is the strongest sequential
  # competitor — parameters are estimated sequentially (K, then comp, then λ),
  # not jointly, and Z remains the B-S form rather than logistic.
  obj_lam <- function(lam_raw) {
    lam  <- plogis(lam_raw)
    fbar <- ewma_fbar(df2, lam)
    comp <- exp(alpha_c + beta_c * df2$log_expo_sc)
    pred <- (1 - df2$Z_bs) * comp + df2$Z_bs * fbar
    sum(df2$expo_wt * (df2$lr_rel - pred)^2)
  }

  opt <- optimize(obj_lam, interval = c(-6, 6))
  lam <- plogis(opt$minimum)

  message(sprintf(
    "  B-S best patch: K_Sm=%.1f K_Md=%.1f K_Lg=%.1f  lam=%.3f",
    bs_strat$K_hat[bs_strat$size_tercile == "Small"],
    bs_strat$K_hat[bs_strat$size_tercile == "Mid"],
    bs_strat$K_hat[bs_strat$size_tercile == "Large"],
    lam
  ))

  list(type = "bs_best_patch", bs_strat = bs_strat,
       alpha_c = alpha_c, beta_c = beta_c, lam = lam,
       company_stats = strat_stats %>% rename(w_i_strat = w_i, f_i_strat = f_i))
}

predict_bs_best_patch <- function(fit, df_new) {
  df_work <- df_new %>%
    left_join(fit$company_stats, by = c("GRCODE", "size_tercile")) %>%
    left_join(fit$bs_strat %>% select(size_tercile, K_hat), by = "size_tercile") %>%
    mutate(Z_bs = w_i_strat / (w_i_strat + K_hat))

  fbar <- ewma_fbar(df_work, fit$lam)
  comp <- exp(fit$alpha_c + fit$beta_c * df_work$log_expo_sc)
  (1 - df_work$Z_bs) * comp + df_work$Z_bs * fbar
}

# -----------------------------------------------------------------------------
# GLMM: random intercept per company + size fixed effect
#
# lr_rel ~ log_expo_sc + (1 | GRCODE)  [Gamma, log link, NEP weights]
#
# KEY LIMITATION (stated in paper): for an account not seen in training the
# random effect = 0 (falls back to fixed effects only). In this balanced panel
# all test companies appear in training so this is latent, not active.
# -----------------------------------------------------------------------------

fit_glmm <- function(df) {
  fit <- glmmTMB(
    lr_rel ~ log_expo_sc + (1 | GRCODE),
    weights = expo_wt,
    family  = Gamma(link = "log"),
    data    = df
  )
  message(sprintf("  GLMM: intercept=%.3f  beta_expo=%.3f  sigma_u=%.3f",
    fixef(fit)$cond[1],
    fixef(fit)$cond[2],
    sqrt(as.numeric(VarCorr(fit)$cond$GRCODE[1]))
  ))
  fit
}

# -----------------------------------------------------------------------------
# Joint-Decay (scalar lambda, ML)
#
# theta_hat_rel = (1-Z) * exp(alpha + beta*log_expo_sc) + Z * fbar_rel(lambda)
# Z = logistic(az + bz * log_expo_used_sc)
# lambda: single scalar (ML)
# -----------------------------------------------------------------------------

fit_jdecay_scalar <- function(df) {
  nll <- function(p, data) {
    lam    <- plogis(p[6])
    fbar   <- ewma_fbar(data, lam)
    base   <- exp(p[1] + p[2] * data$log_expo_sc)
    Z      <- plogis(p[3] + p[4] * data$log_expo_used_sc)
    mu     <- pmax((1 - Z) * base + Z * fbar, 1e-8)
    shape  <- exp(p[5])
    -sum(dgamma(data$lr_rel, shape = shape, rate = shape / mu, log = TRUE) * data$expo_wt)
  }
  par_init <- c(alpha = 0, beta = 0, az = -1, bz = 0.5, log_shape = 1, lam_raw = 0)
  lower    <- c(-2, -1, -6, -3, -2, -6)
  upper    <- c( 2,  1,  3,  5,  5,  6)
  fit <- multi_optim(nll, par_init, lower, upper, data = df)
  p   <- fit$par
  message(sprintf(
    "  Joint-Decay scalar: alpha=%.3f  beta=%.3f  az=%.2f  bz=%.2f  lam=%.3f",
    p[1], p[2], p[3], p[4], plogis(p[6])
  ))
  list(type = "jdecay_scalar", par = p)
}

predict_jdecay_scalar <- function(fit, df_new) {
  p      <- fit$par
  lam    <- plogis(p[6])
  fbar   <- ewma_fbar(df_new, lam)
  base   <- exp(p[1] + p[2] * df_new$log_expo_sc)
  Z      <- plogis(p[3] + p[4] * df_new$log_expo_used_sc)
  (1 - Z) * base + Z * fbar
}

# -----------------------------------------------------------------------------
# Joint-Decay (continuous lambda, ML)
#
# As above but lambda varies continuously with mean log NEP:
# lambda_i = logistic(lam0 + lam1 * log_mean_expo_sc)
# -----------------------------------------------------------------------------

fit_jdecay_continuous <- function(df) {
  nll <- function(p, data) {
    lam   <- plogis(p[6] + p[7] * data$log_mean_expo_sc)
    fbar  <- ewma_fbar_vec(data, lam)   # vectorised: one lambda per row
    base  <- exp(p[1] + p[2] * data$log_expo_sc)
    Z     <- plogis(p[3] + p[4] * data$log_expo_used_sc)
    mu    <- pmax((1 - Z) * base + Z * fbar, 1e-8)
    shape <- exp(p[5])
    -sum(dgamma(data$lr_rel, shape = shape, rate = shape / mu, log = TRUE) * data$expo_wt)
  }
  # Note: lam1 may hit its bound on CA data. The true lambda gradient is
  # non-monotone in size (Mid > Small > Large in the tercile model), so a
  # linear logit-scale model cannot fit all three terciles simultaneously.
  # The resulting wMSE is still valid as a comparison point.
  par_init <- c(alpha = 0, beta = 0, az = -1, bz = 0.5, log_shape = 1,
                lam0 = 0, lam1 = 0)
  lower    <- c(-2, -1, -6, -3, -2, -6, -5)
  upper    <- c( 2,  1,  3,  5,  5,  6,  5)
  fit <- multi_optim(nll, par_init, lower, upper, data = df)
  p   <- fit$par
  message(sprintf(
    "  Joint-Decay continuous: az=%.2f  bz=%.2f  lam0=%.3f  lam1=%.3f",
    p[3], p[4], p[6], p[7]
  ))
  list(type = "jdecay_continuous", par = p)
}

predict_jdecay_continuous <- function(fit, df_new) {
  p    <- fit$par
  base <- exp(p[1] + p[2] * df_new$log_expo_sc)
  Z    <- plogis(p[3] + p[4] * df_new$log_expo_used_sc)
  lam  <- plogis(p[6] + p[7] * df_new$log_mean_expo_sc)
  fbar <- ewma_fbar_vec(df_new, lam)
  (1 - Z) * base + Z * fbar
}

# -----------------------------------------------------------------------------
# Joint-Decay (tercile lambda, Bayesian) — PROPOSED MODEL
#
# The proposed model from the paper: continuous size-slope complement
# exp(alpha + beta*log_expo_sc) combined with three free decay parameters
# (lamSm, lamMd, lamLg) estimated jointly in a single Bayesian pass.
#
# Parameters: alpha, beta (complement), az, bz (Z logistic), lamSm, lamMd, lamLg (decay)
# brms draw columns: b_alpha_Intercept, b_beta_Intercept, b_az_Intercept,
#                    b_bz_Intercept, b_lamSm_Intercept, b_lamMd_Intercept, b_lamLg_Intercept
#
# brms / Stan required. First run: ~10 min (4 chains x 2000 iter).
# Subsequent runs load the cached RDS file from MODEL_DIR.
# -----------------------------------------------------------------------------

fit_jdecay_tercile <- function(df, rds_path) {
  if (!REFIT && file.exists(rds_path)) {
    message("  Joint-Decay (tercile lambda): loading cached model")
    return(readRDS(rds_path))
  }

  form <- bf(
    lr_rel | weights(expo_wt) ~
      (1 - inv_logit(az + bz * log_expo_used_sc)) *
        exp(alpha + beta * log_expo_sc) +
      inv_logit(az + bz * log_expo_used_sc) *
        (lr_lag1_rel * expo_lag1 +
           lr_lag2_rel * expo_lag2 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg)) +
           lr_lag3_rel * expo_lag3 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^2 +
           lr_lag4_rel * expo_lag4 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^3 +
           lr_lag5_rel * expo_lag5 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^4 +
           lr_lag6_rel * expo_lag6 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^5 +
           lr_lag7_rel * expo_lag7 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^6) /
        (expo_lag1 +
           expo_lag2 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg)) +
           expo_lag3 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^2 +
           expo_lag4 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^3 +
           expo_lag5 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^4 +
           expo_lag6 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^5 +
           expo_lag7 * (isSm * inv_logit(lamSm) + isMd * inv_logit(lamMd) + isLg * inv_logit(lamLg))^6),
    alpha ~ 1, beta ~ 1,
    az ~ 1, bz ~ 1,
    lamSm ~ 1, lamMd ~ 1, lamLg ~ 1,
    nl = TRUE
  )

  pri <- c(
    prior(normal(0.0, 0.3),  nlpar = "alpha"),   # exp(0) = 1 = market mean in relative space
    prior(normal(0.0, 0.3),  nlpar = "beta"),
    prior(normal(-0.5, 1.0), nlpar = "az"),
    prior(normal( 0.5, 0.5), nlpar = "bz"),
    prior(normal(0.0, 1.5),  nlpar = "lamSm"),
    prior(normal(0.0, 1.5),  nlpar = "lamMd"),
    prior(normal(0.0, 1.5),  nlpar = "lamLg")
  )

  while (sink.number() > 0) sink()  # clear any stuck output connections
  fit <- brm(
    form, data = df,
    family  = Gamma(link = "identity"),
    prior   = pri,
    chains  = 4, cores = parallel::detectCores(), iter = 2000,
    control = list(adapt_delta = 0.97),
    save_pars = save_pars(all = TRUE),
    seed = 48, refresh = 200
  )
  saveRDS(fit, rds_path)

  p <- as_draws_df(fit)
  message(sprintf(
    "  Joint-Decay (tercile lambda): lam_Sm=%.3f  lam_Md=%.3f  lam_Lg=%.3f  az=%.2f  bz=%.2f",
    plogis(mean(p$b_lamSm_Intercept)),
    plogis(mean(p$b_lamMd_Intercept)),
    plogis(mean(p$b_lamLg_Intercept)),
    mean(p$b_az_Intercept),
    mean(p$b_bz_Intercept)
  ))
  fit
}

predict_jdecay_tercile <- function(fit, df_new) {
  p      <- as_draws_df(fit)
  az     <- mean(p$b_az_Intercept)
  bz     <- mean(p$b_bz_Intercept)
  alpha  <- mean(p$b_alpha_Intercept)
  beta   <- mean(p$b_beta_Intercept)
  lam_sm <- plogis(mean(p$b_lamSm_Intercept))
  lam_md <- plogis(mean(p$b_lamMd_Intercept))
  lam_lg <- plogis(mean(p$b_lamLg_Intercept))

  Z    <- plogis(az + bz * df_new$log_expo_used_sc)
  comp <- exp(alpha + beta * df_new$log_expo_sc)
  fbar <- numeric(nrow(df_new))

  for (trc in c("Small", "Mid", "Large")) {
    lam_t <- switch(trc, Small = lam_sm, Mid = lam_md, Large = lam_lg)
    idx   <- which(df_new$size_tercile == trc)
    if (length(idx) > 0) fbar[idx] <- ewma_fbar(df_new[idx, ], lam_t)
  }

  (1 - Z) * comp + Z * fbar
}

# -----------------------------------------------------------------------------
# Joint-Decay (scalar lambda, MAP) — paper Appendix, Listing 4
#
# Adds weakly-informative log-prior penalties to the MLE objective.
# Priors: Normal(0, 1) on all logit-scale parameters (alpha, beta, az, bz,
# lam_raw); Normal(2, 1) on log_shape.
# Useful when lambda is weakly identified (small portfolios). MAP estimates
# lie close to MLE when data are informative but are pulled toward moderate
# values (less extreme lambda, more conservative Z) when data are sparse.
# -----------------------------------------------------------------------------

fit_jdecay_scalar_map <- function(df) {
  nlp <- function(p, data) {
    nll_val <- {
      lam   <- plogis(p[6])
      fbar  <- ewma_fbar(data, lam)
      base  <- exp(p[1] + p[2] * data$log_expo_sc)
      Z     <- plogis(p[3] + p[4] * data$log_expo_used_sc)
      mu    <- pmax((1 - Z) * base + Z * fbar, 1e-8)
      shape <- exp(p[5])
      -sum(dgamma(data$lr_rel, shape = shape, rate = shape / mu, log = TRUE) * data$expo_wt)
    }
    # Weakly-informative priors: Normal(0,1) on logit-scale params; Normal(2,1) on log_shape
    log_prior <- sum(dnorm(p[-5], mean = 0, sd = 1, log = TRUE)) +
                 dnorm(p[5],  mean = 2, sd = 1, log = TRUE)
    nll_val - log_prior
  }
  par_init <- c(alpha = 0, beta = 0, az = -1, bz = 0.5, log_shape = 2, lam_raw = 0)
  lower    <- c(-2, -1, -6, -3, -2, -6)
  upper    <- c( 2,  1,  3,  5,  5,  6)
  fit <- multi_optim(nlp, par_init, lower, upper, data = df)
  p   <- fit$par
  message(sprintf(
    "  Joint-Decay scalar (MAP): alpha=%.3f  beta=%.3f  az=%.2f  bz=%.2f  lam=%.3f",
    p[1], p[2], p[3], p[4], plogis(p[6])
  ))
  list(type = "jdecay_scalar_map", par = p)
}

# predict_jdecay_scalar_map is identical to the MLE version
predict_jdecay_scalar_map <- predict_jdecay_scalar

# -----------------------------------------------------------------------------
# Joint-Decay (tercile lambda, MAP)
#
# MAP version of the MLE tercile model. Priors pull each lambda away from the
# boundary — particularly helpful when lambda_Mid hits lam ~ 1 (all weight on
# the most recent year) as seen in the MLE fit. Normal(0, 1.5) on each
# lam_raw matches the brms prior used in the Bayesian (proposed) model.
# -----------------------------------------------------------------------------

fit_jdecay_tercile_map <- function(df) {
  nlp <- function(p, data) {
    nll_val <- {
      lam_sm <- plogis(p[6]); lam_md <- plogis(p[7]); lam_lg <- plogis(p[8])
      lam_i  <- ifelse(data$size_tercile == "Small", lam_sm,
                 ifelse(data$size_tercile == "Mid",  lam_md, lam_lg))
      fbar   <- ewma_fbar_vec(data, lam_i)
      base   <- exp(p[1] + p[2] * data$log_expo_sc)
      Z      <- plogis(p[3] + p[4] * data$log_expo_used_sc)
      mu     <- pmax((1 - Z) * base + Z * fbar, 1e-8)
      shape  <- exp(p[5])
      -sum(dgamma(data$lr_rel, shape = shape, rate = shape / mu, log = TRUE) * data$expo_wt)
    }
    # Normal(0,1) on alpha, beta, az, bz; Normal(2,1) on log_shape;
    # Normal(0,1.5) on lam_raw_Sm/Md/Lg — matches brms priors in proposed model
    log_prior <- sum(dnorm(p[c(1,2,3,4)], mean = 0, sd = 1,   log = TRUE)) +
                 dnorm(p[5],              mean = 2, sd = 1,   log = TRUE) +
                 sum(dnorm(p[c(6,7,8)],  mean = 0, sd = 1.5, log = TRUE))
    nll_val - log_prior
  }
  par_init <- c(alpha = 0, beta = 0, az = -1, bz = 0.5, log_shape = 2,
                lam_sm = 0, lam_md = 0, lam_lg = 0)
  lower    <- c(-2, -1, -6, -3, -2, -6, -6, -6)
  upper    <- c( 2,  1,  3,  5,  5,  6,  6,  6)
  fit <- multi_optim(nlp, par_init, lower, upper, data = df)
  p   <- fit$par
  message(sprintf(
    "  Joint-Decay tercile (MAP): az=%.2f  bz=%.2f  lam_Sm=%.3f  lam_Md=%.3f  lam_Lg=%.3f",
    p[3], p[4], plogis(p[6]), plogis(p[7]), plogis(p[8])
  ))
  list(type = "jdecay_tercile_map", par = p)
}

predict_jdecay_tercile_map <- function(fit, df_new) {
  p      <- fit$par
  az     <- p[3]; bz <- p[4]; alpha <- p[1]; beta <- p[2]
  lam_sm <- plogis(p[6]); lam_md <- plogis(p[7]); lam_lg <- plogis(p[8])
  Z    <- plogis(az + bz * df_new$log_expo_used_sc)
  comp <- exp(alpha + beta * df_new$log_expo_sc)
  fbar <- numeric(nrow(df_new))
  for (trc in c("Small", "Mid", "Large")) {
    lam_t <- switch(trc, Small = lam_sm, Mid = lam_md, Large = lam_lg)
    idx   <- which(df_new$size_tercile == trc)
    if (length(idx) > 0) fbar[idx] <- ewma_fbar(df_new[idx, ], lam_t)
  }
  (1 - Z) * comp + Z * fbar
}

# -----------------------------------------------------------------------------
# Fit all models on training data  (CA only — OL fitting is in 04_models_ol.R)
# -----------------------------------------------------------------------------

if (exists("LINE_OF_BUSINESS") && LINE_OF_BUSINESS == "ca") {

  message("\n--- Fitting models (Commercial Auto) ---")

  models <- list(
    bs_standard        = fit_bs_standard(df_train),
    bs_best_patch      = fit_bs_best_patch(df_train),
    glmm               = fit_glmm(df_train),
    jdecay_scalar      = fit_jdecay_scalar(df_train),
    jdecay_scalar_map  = fit_jdecay_scalar_map(df_train),
    jdecay_cont        = fit_jdecay_continuous(df_train),
    jdecay_tercile_map = fit_jdecay_tercile_map(df_train),
    jdecay_tercile     = fit_jdecay_tercile(
      df_train,
      rds_path = file.path(MODEL_DIR, "ca_jdecay_tercile.rds")
    )
  )

  df_test <- df_test %>%
    mutate(
      pred_market_mean        = 1,
      pred_last_year          = lr_lag1 / mu_t,
      pred_bs_standard        = predict_bs_standard(models$bs_standard,   .),
      pred_bs_best_patch      = predict_bs_best_patch(models$bs_best_patch, .),
      pred_glmm               = predict(models$glmm, newdata = ., type = "response",
                                         allow.new.levels = FALSE),
      pred_jdecay_scalar      = predict_jdecay_scalar(models$jdecay_scalar, .),
      pred_jdecay_scalar_map  = predict_jdecay_scalar_map(models$jdecay_scalar_map, .),
      pred_jdecay_cont        = predict_jdecay_continuous(models$jdecay_cont, .),
      pred_jdecay_tercile_map = predict_jdecay_tercile_map(models$jdecay_tercile_map, .),
      pred_jdecay_tercile     = predict_jdecay_tercile(models$jdecay_tercile, .)
    )

  message("Models fitted and test predictions generated.")

}
