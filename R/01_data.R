# =============================================================================
# 01_data.R  —  Data loading, filtering, and panel construction
#
# Produces: df_train, df_test, year_means, co_tercile_tbl, tr_stats
# These objects are used by 03_models_ca.R and 04_models_ol.R.
#
# Call this script via run_ca.R or run_ol.R, which set LINE_OF_BUSINESS
# before sourcing. Do not source this file directly.
# =============================================================================

# LINE_OF_BUSINESS must be set by the caller: "ca" or "ol"
stopifnot(exists("LINE_OF_BUSINESS"), LINE_OF_BUSINESS %in% c("ca", "ol"))

library(dplyr)
library(tidyr)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

W_MAX       <- 7          # maximum lag window
TRAIN_YEARS <- 2001:2005
TEST_YEARS  <- 2006:2007
MIN_NEP     <- 100        # minimum net earned premium ($000s)
USE_WEIGHTS <- TRUE       # weight Gamma likelihood by current-year NEP

if (LINE_OF_BUSINESS == "ca") {
  DATA_FILE <- file.path(DATA_DIR, "comauto_pos_98-07.csv")
  DATA_URL  <- "https://www.casact.org/sites/default/files/2026-03/comauto_pos_98-07.csv"
} else {
  DATA_FILE <- file.path(DATA_DIR, "othliab_pos_98-07.csv")
  DATA_URL  <- "https://www.casact.org/sites/default/files/2026-03/othliab_pos_98-07.csv"
}

# -----------------------------------------------------------------------------
# Download data if not present
# -----------------------------------------------------------------------------

if (!file.exists(DATA_FILE)) {
  message("Downloading CAS data from: ", DATA_URL)
  download.file(DATA_URL, DATA_FILE, mode = "wb")
}

raw <- read.csv(DATA_FILE)

# -----------------------------------------------------------------------------
# Filter to lag-10 ultimates, qualifying companies
# -----------------------------------------------------------------------------

ult <- raw %>%
  filter(DevelopmentLag == 10, EarnedPremNet >= MIN_NEP) %>%
  transmute(
    GRCODE, GRNAME, AccidentYear,
    expo = EarnedPremNet,
    lr   = IncurredLosses / EarnedPremNet
  )

valid_cos <- ult %>%
  group_by(GRCODE) %>%
  summarise(
    n        = n(),
    any_neg  = any(lr < 0),
    any_zero_mod = any(lr <= 0 & AccidentYear %in% c(TRAIN_YEARS, TEST_YEARS)),
    .groups  = "drop"
  ) %>%
  filter(n == 10, !any_neg, !any_zero_mod) %>%
  pull(GRCODE)

ult <- filter(ult, GRCODE %in% valid_cos)

# Other Liability: remove one company with a single extreme lr_rel > 5.
# (Dorinco Rein Co has lr_rel = 678 in one year; destabilises all MoM estimates.)
if (LINE_OF_BUSINESS == "ol") {
  yr_tmp <- ult %>%
    group_by(AccidentYear) %>%
    summarise(mu_yr = sum(lr * expo) / sum(expo), .groups = "drop")
  extreme_cos <- ult %>%
    left_join(yr_tmp, by = "AccidentYear") %>%
    mutate(lr_rel_tmp = lr / mu_yr) %>%
    group_by(GRCODE) %>%
    summarise(max_lr_rel = max(lr_rel_tmp), .groups = "drop") %>%
    filter(max_lr_rel > 5) %>%
    pull(GRCODE)
  ult <- filter(ult, !GRCODE %in% extreme_cos)
}

message(sprintf("Qualifying companies: %d", length(unique(ult$GRCODE))))

# -----------------------------------------------------------------------------
# Portfolio mean LR by year — used to normalise response and lags (Option B)
#
# Option B formulation:
#   theta_hat_rel = Z * fbar_rel(lambda) + (1-Z) * exp(alpha + beta * log_expo)
#   theta_hat     = theta_hat_rel * mu_T
#
# By normalising both response and lags by the annual portfolio mean, the
# calendar-year market cycle cancels out of both sides of the blend. No
# year-trend parameter is needed in the complement.
# -----------------------------------------------------------------------------

year_means <- ult %>%
  group_by(AccidentYear) %>%
  summarise(mu_yr = sum(lr * expo) / sum(expo), .groups = "drop")

# -----------------------------------------------------------------------------
# Build panel: relative LRs + lag structure
# -----------------------------------------------------------------------------

panel <- ult %>%
  left_join(year_means, by = "AccidentYear") %>%
  rename(mu_t = mu_yr) %>%
  arrange(GRCODE, AccidentYear) %>%
  group_by(GRCODE) %>%
  mutate(
    lr_rel = lr / mu_t,
    lr_lag1 = lag(lr, 1), lr_lag2 = lag(lr, 2), lr_lag3 = lag(lr, 3),
    lr_lag4 = lag(lr, 4), lr_lag5 = lag(lr, 5), lr_lag6 = lag(lr, 6), lr_lag7 = lag(lr, 7),
    expo_lag1 = lag(expo, 1), expo_lag2 = lag(expo, 2), expo_lag3 = lag(expo, 3),
    expo_lag4 = lag(expo, 4), expo_lag5 = lag(expo, 5), expo_lag6 = lag(expo, 6), expo_lag7 = lag(expo, 7),
    mu_lag1 = lag(mu_t, 1), mu_lag2 = lag(mu_t, 2), mu_lag3 = lag(mu_t, 3),
    mu_lag4 = lag(mu_t, 4), mu_lag5 = lag(mu_t, 5), mu_lag6 = lag(mu_t, 6), mu_lag7 = lag(mu_t, 7),
    lr_lag1_rel = lr_lag1 / mu_lag1,
    lr_lag2_rel = lr_lag2 / mu_lag2,
    lr_lag3_rel = lr_lag3 / mu_lag3,
    n_years_hist = row_number() - 1
  ) %>%
  ungroup() %>%
  # Zero-fill pre-entry lags: expo -> 0, lr_rel -> 1 (neutral; weight = 0 anyway)
  mutate(
    expo_lag4 = coalesce(expo_lag4, 0), expo_lag5 = coalesce(expo_lag5, 0),
    expo_lag6 = coalesce(expo_lag6, 0), expo_lag7 = coalesce(expo_lag7, 0),
    mu_lag4   = coalesce(mu_lag4, mu_t), mu_lag5 = coalesce(mu_lag5, mu_t),
    mu_lag6   = coalesce(mu_lag6, mu_t), mu_lag7 = coalesce(mu_lag7, mu_t),
    lr_lag4   = coalesce(lr_lag4, mu_t), lr_lag5 = coalesce(lr_lag5, mu_t),
    lr_lag6   = coalesce(lr_lag6, mu_t), lr_lag7 = coalesce(lr_lag7, mu_t),
    lr_lag4_rel = lr_lag4 / mu_lag4, lr_lag5_rel = lr_lag5 / mu_lag5,
    lr_lag6_rel = lr_lag6 / mu_lag6, lr_lag7_rel = lr_lag7 / mu_lag7,
    # expo_used: total exposure over the W_MAX lag window (= credibility mass)
    expo_used = expo_lag1 + expo_lag2 + expo_lag3 +
                expo_lag4 + expo_lag5 + expo_lag6 + expo_lag7,
    # past_lr_rel: exposure-weighted mean of lag relative LRs (B-S input)
    past_lr_rel = (lr_lag1_rel*expo_lag1 + lr_lag2_rel*expo_lag2 +
                   lr_lag3_rel*expo_lag3 + lr_lag4_rel*expo_lag4 +
                   lr_lag5_rel*expo_lag5 + lr_lag6_rel*expo_lag6 +
                   lr_lag7_rel*expo_lag7) / pmax(expo_used, 1e-8),
    # W_i: count of lags with positive exposure (effective history length)
    W_i = as.integer(expo_lag1 > 0) + as.integer(expo_lag2 > 0) +
          as.integer(expo_lag3 > 0) + as.integer(expo_lag4 > 0) +
          as.integer(expo_lag5 > 0) + as.integer(expo_lag6 > 0) +
          as.integer(expo_lag7 > 0),
    # cv_lr: coefficient of variation of annual LRs (process-variance proxy)
    cv_lr = sapply(seq_along(lr), function(j) {
      h <- lr[max(1, j - W_MAX):(j - 1)]
      if (length(h) < 2) NA_real_ else sd(h) / max(mean(h), 1e-8)
    })
  ) %>%
  filter(!is.na(lr_lag3))

# -----------------------------------------------------------------------------
# Train / test split
# -----------------------------------------------------------------------------

pred_panel <- panel %>%
  filter(AccidentYear %in% c(TRAIN_YEARS, TEST_YEARS)) %>%
  filter(complete.cases(lr_rel, lr_lag1_rel, expo_lag1, n_years_hist, cv_lr, mu_t))

# Standardise covariates on training observations only
tr_stats <- pred_panel %>%
  filter(AccidentYear %in% TRAIN_YEARS) %>%
  summarise(
    log_expo_mean      = mean(log(expo)),
    log_expo_sd        = sd(log(expo)),
    log_expo_used_mean = mean(log(expo_used)),
    log_expo_used_sd   = sd(log(expo_used)),
    ny_mean            = mean(n_years_hist),
    ny_sd              = sd(n_years_hist),
    cv_mean            = mean(cv_lr, na.rm = TRUE),
    cv_sd              = sd(cv_lr, na.rm = TRUE),
    expo_mean          = mean(expo),
    yr_mean            = mean(AccidentYear),
    yr_sd              = sd(AccidentYear)
  )

pred_panel <- pred_panel %>%
  mutate(
    log_expo_sc      = (log(expo)      - tr_stats$log_expo_mean)      / tr_stats$log_expo_sd,
    log_expo_used_sc = (log(expo_used) - tr_stats$log_expo_used_mean) / tr_stats$log_expo_used_sd,
    n_years_sc       = (n_years_hist   - tr_stats$ny_mean)            / tr_stats$ny_sd,
    cv_sc            = (cv_lr          - tr_stats$cv_mean)            / tr_stats$cv_sd,
    yr_sc            = (AccidentYear   - tr_stats$yr_mean)            / tr_stats$yr_sd,
    expo_wt          = if (USE_WEIGHTS) expo / tr_stats$expo_mean else 1
  )

# Size terciles: based on mean training NEP per company (stable across years)
co_mean_expo_trc <- pred_panel %>%
  filter(AccidentYear %in% TRAIN_YEARS) %>%
  group_by(GRCODE) %>%
  summarise(mean_expo = mean(expo), .groups = "drop")

expo_tercile_breaks <- quantile(co_mean_expo_trc$mean_expo, c(1/3, 2/3))

log_mean_expo_mean <- mean(log(co_mean_expo_trc$mean_expo))
log_mean_expo_sd   <- sd(log(co_mean_expo_trc$mean_expo))

co_tercile_tbl <- co_mean_expo_trc %>%
  mutate(
    size_tercile   = cut(mean_expo, breaks = c(-Inf, expo_tercile_breaks, Inf),
                         labels = c("Small", "Mid", "Large")),
    size_tercile_f = factor(size_tercile, levels = c("Small", "Mid", "Large")),
    isSm           = as.integer(size_tercile == "Small"),
    isMd           = as.integer(size_tercile == "Mid"),
    isLg           = as.integer(size_tercile == "Large"),
    log_mean_expo_sc = (log(mean_expo) - log_mean_expo_mean) / log_mean_expo_sd
  ) %>%
  select(GRCODE, size_tercile, size_tercile_f, isSm, isMd, isLg, log_mean_expo_sc, mean_expo)

pred_panel <- pred_panel %>% left_join(co_tercile_tbl, by = "GRCODE")

df_train <- filter(pred_panel, AccidentYear %in% TRAIN_YEARS)
df_test  <- filter(pred_panel, AccidentYear %in% TEST_YEARS)

message(sprintf("Training obs: %d   Test obs: %d", nrow(df_train), nrow(df_test)))
message(sprintf("Exposure-weighted mean lr_rel in training: %.4f (should be ~1.0)",
                sum(df_train$expo * df_train$lr_rel) / sum(df_train$expo)))
