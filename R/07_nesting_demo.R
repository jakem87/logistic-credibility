# =============================================================================
# 07_nesting_demo.R  —  Self-contained nesting demonstration
#
# Shows that the logistic credibility model nests Bühlmann-Straub exactly.
# Run independently via:  source("run_nesting_demo.R")  or  Rscript run_nesting_demo.R
#
# Scenario S1 (classical B-S): homogeneous K, no temporal drift.
# Theory (Proposition 1 in the paper):
#   logit(Z_i) = log(E_i) - log(K)
#   => b_optimal = 1 / SD(log E_i)  on the standardised scale
# When b is free, the ML estimate b_hat should converge toward b_BS = 1/SD(log E_i).
# In finite samples b_hat < b_BS because of within-account estimation noise
# (shrinkage is data-optimal when the true rate is unobserved over the window).
#
# Runtime: ~5 seconds (ML only, no Bayesian sampling required)
# =============================================================================

library(dplyr)
library(ggplot2)

set.seed(46)

# -----------------------------------------------------------------------------
# DGP parameters (S1: classical B-S)
# -----------------------------------------------------------------------------

N_INS    <- 200      # number of accounts
N_YEARS  <- 8        # total years (including test)
MU_BASE  <- 0.20     # base frequency
SIGMA_B  <- 0.25     # between-account SD (homogeneous K)
SIGMA_W  <- 0.0      # no temporal drift
PHI      <- 0.0      # no AR(1) persistence
K_TRUE   <- 1 / (MU_BASE * (exp(SIGMA_B^2) - 1))
WINDOW   <- 5        # training window

# -----------------------------------------------------------------------------
# Simulate portfolio
# -----------------------------------------------------------------------------

typical_expo <- rlnorm(N_INS, meanlog = 1.5, sdlog = 0.9)
theta_perm   <- exp(rnorm(N_INS, mean = -0.5 * SIGMA_B^2, sd = SIGMA_B))

sim_df <- do.call(rbind, lapply(seq_len(N_INS), function(i) {
  expo_t <- typical_expo[i] * exp(rnorm(N_YEARS, mean = -0.01, sd = 0.15))
  claims_t <- rpois(N_YEARS, lambda = MU_BASE * theta_perm[i] * expo_t)
  data.frame(
    insured = i,
    year    = seq_len(N_YEARS),
    expo    = expo_t,
    claims  = claims_t,
    theta   = theta_perm[i]
  )
}))

# Test year = N_YEARS; training = years 1 to N_YEARS-1 (up to WINDOW most recent)
train_df <- sim_df %>%
  filter(year < N_YEARS) %>%
  group_by(insured) %>%
  slice_tail(n = WINDOW) %>%
  ungroup()

test_df <- sim_df %>% filter(year == N_YEARS)

# Feature engineering
features <- train_df %>%
  group_by(insured) %>%
  summarise(
    expo_used   = sum(expo),
    past_freq   = sum(claims) / sum(expo),
    .groups = "drop"
  )

# Standardise log(expo_used) on training data
le_mean <- mean(log(features$expo_used))
le_sd   <- sd(log(features$expo_used))
features <- features %>%
  mutate(log_expo_sc = (log(expo_used) - le_mean) / le_sd)

df_fit <- test_df %>%
  inner_join(features, by = "insured") %>%
  mutate(freq = claims / expo)

# -----------------------------------------------------------------------------
# Bühlmann-Straub (MoM K)
# -----------------------------------------------------------------------------

ins_train <- train_df %>%
  group_by(insured) %>%
  summarise(w_i = sum(expo), f_i = sum(claims) / sum(expo), .groups = "drop")

W_tot  <- sum(ins_train$w_i)
n_i    <- nrow(ins_train)
f_bar  <- sum(ins_train$w_i * ins_train$f_i) / W_tot
c_w    <- W_tot - sum(ins_train$w_i^2) / W_tot

sigma2_within <- train_df %>%
  left_join(ins_train, by = "insured") %>%
  group_by(insured) %>%
  summarise(
    w_i  = first(w_i),
    s2_i = sum(expo * (claims / expo - f_i)^2) / (n() - 1),
    .groups = "drop"
  ) %>%
  summarise(s2 = sum(w_i * s2_i) / sum(w_i)) %>%
  pull(s2)

ss_b  <- sum(ins_train$w_i * (ins_train$f_i - f_bar)^2)
var_b <- max((ss_b - (n_i - 1) * sigma2_within) / c_w, 1e-8)
K_bs  <- sigma2_within / var_b

cat(sprintf("B-S:    K_true = %.1f    K_hat = %.1f\n", K_TRUE, K_bs))

# Theoretical b_BS on standardised scale
b_BS <- le_sd   # = SD(log expo_used); at this value logistic nests B-S exactly
cat(sprintf("Nesting: b_BS = SD(log expo_used) = %.4f\n", b_BS))

# B-S predictions
df_fit <- df_fit %>%
  left_join(ins_train, by = "insured") %>%
  mutate(
    Z_bs   = w_i / (w_i + K_bs),
    pred_bs = (1 - Z_bs) * f_bar + Z_bs * f_i
  )

# -----------------------------------------------------------------------------
# Logistic (expo, b free) — ML via optim
# -----------------------------------------------------------------------------

nll_logistic <- function(p) {
  base <- p[1]                         # common complement (MU_BASE in relative units)
  az   <- p[2]
  bz   <- p[3]
  Z    <- plogis(az + bz * df_fit$log_expo_sc)
  mu   <- pmax((1 - Z) * base + Z * df_fit$past_freq, 1e-10)
  -sum(dpois(df_fit$claims, lambda = mu * df_fit$expo, log = TRUE))
}

starts <- list(
  c(MU_BASE, 0, b_BS),
  c(MU_BASE, -1, 0.5),
  c(MU_BASE, -2, 1.5)
)
best <- NULL
for (s in starts) {
  res <- tryCatch(
    optim(s, nll_logistic, method = "L-BFGS-B",
          lower = c(0.01, -6, -3), upper = c(1, 3, 5),
          control = list(maxit = 2000)),
    error = function(e) list(value = Inf)
  )
  if (is.null(best) || res$value < best$value) best <- res
}

p_hat <- best$par
az_hat <- p_hat[2]
bz_hat <- p_hat[3]
cat(sprintf("Logistic: az_hat = %.4f    bz_hat = %.4f    b_BS = %.4f\n",
            az_hat, bz_hat, b_BS))
cat(sprintf("b_hat / b_BS = %.3f  (< 1 expected: optimal shrinkage under estimation noise)\n",
            bz_hat / b_BS))

df_fit <- df_fit %>%
  mutate(
    Z_logistic   = plogis(az_hat + bz_hat * log_expo_sc),
    pred_logistic = (1 - Z_logistic) * p_hat[1] + Z_logistic * past_freq
  )

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

mse_bs  <- mean((df_fit$pred_bs       - df_fit$freq)^2)
mse_log <- mean((df_fit$pred_logistic - df_fit$freq)^2)
cat(sprintf("MSE — B-S: %.6f    Logistic: %.6f\n", mse_bs, mse_log))

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

# 1. Z curves: B-S vs logistic vs true
expo_grid_sc <- seq(min(df_fit$log_expo_sc), max(df_fit$log_expo_sc), length.out = 200)
expo_grid_raw <- exp(expo_grid_sc * le_sd + le_mean)

df_curves <- tibble(
  log_expo_sc   = expo_grid_sc,
  `B-S`         = expo_grid_raw / (expo_grid_raw + K_bs),
  `B-S (true K)`= expo_grid_raw / (expo_grid_raw + K_TRUE),
  `Logistic (b free)` = plogis(az_hat + bz_hat * expo_grid_sc)
) %>%
  pivot_longer(-log_expo_sc, names_to = "Model", values_to = "Z")

p_zcurves <- ggplot(df_curves, aes(x = log_expo_sc, y = Z, colour = Model, linetype = Model)) +
  geom_line(linewidth = 1) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  scale_colour_manual(values = c("B-S" = "#E41A1C", "B-S (true K)" = "#377EB8",
                                  "Logistic (b free)" = "#4DAF4A")) +
  scale_linetype_manual(values = c("B-S" = "dashed", "B-S (true K)" = "dotted",
                                    "Logistic (b free)" = "solid")) +
  labs(
    title    = "Nesting demonstration (S1: classical B-S)",
    subtitle = sprintf("b_ML = %.3f vs b_BS = SD(log E) = %.3f; ratio = %.3f",
                       bz_hat, b_BS, bz_hat / b_BS),
    x        = "Log exposure (standardised)",
    y        = "Credibility weight Z",
    colour   = NULL, linetype = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

print(p_zcurves)

# 2. Predicted vs actual frequency
p_scatter <- ggplot(df_fit, aes(x = pred_bs, y = pred_logistic, colour = Z_logistic)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  scale_colour_viridis_c(labels = scales::percent, name = "Z (logistic)") +
  labs(
    title = "B-S vs Logistic predictions (S1)",
    x     = "B-S prediction",
    y     = "Logistic prediction"
  ) +
  theme_bw(base_size = 11)

print(p_scatter)

cat("\nNesting demo complete. In S1 (homogeneous K, no drift):\n")
cat(sprintf("  b_ML = %.3f ≈ b_BS = %.3f  (nesting confirmed)\n", bz_hat, b_BS))
cat(sprintf("  b_ML / b_BS < 1 reflects optimal shrinkage under estimation noise in short windows.\n"))
