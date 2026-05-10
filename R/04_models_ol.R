# =============================================================================
# 04_models_ol.R  —  Model fitting: Other Liability
#
# Reuses the same model functions as 03_models_ca.R (sourced before this file).
# The only differences are the data (df_train/df_test from the OL run of 01_data.R)
# and the Bayesian model RDS cache path.
#
# Produces: models list and df_test with prediction columns (same structure as CA).
# =============================================================================

message("\n--- Fitting models (Other Liability) ---")

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
    rds_path = file.path(MODEL_DIR, "ol_jdecay_tercile.rds")
  )
)

df_test <- df_test %>%
  mutate(
    pred_market_mean        = 1,
    pred_last_year          = lr_lag1 / mu_t,
    pred_bs_standard        = predict_bs_standard(models$bs_standard, .),
    pred_bs_best_patch      = predict_bs_best_patch(models$bs_best_patch, .),
    pred_glmm               = predict(models$glmm, newdata = ., type = "response",
                                       allow.new.levels = FALSE),
    pred_jdecay_scalar      = predict_jdecay_scalar(models$jdecay_scalar, .),
    pred_jdecay_scalar_map  = predict_jdecay_scalar_map(models$jdecay_scalar_map, .),
    pred_jdecay_cont        = predict_jdecay_continuous(models$jdecay_cont, .),
    pred_jdecay_tercile_map = predict_jdecay_tercile_map(models$jdecay_tercile_map, .),
    pred_jdecay_tercile     = predict_jdecay_tercile(models$jdecay_tercile, .)
  )

message("OL models fitted and test predictions generated.")
