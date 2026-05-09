# =============================================================================
# run_ca.R  —  Commercial Auto: reproduce Table 3 of Morris (2026)
#
# Usage:  source("run_ca.R")   or   Rscript run_ca.R
#
# Runtime:
#   ML models (B-S, GLMM, Joint-Decay scalar/continuous): < 2 min
#   Bayesian model (Joint-Decay tercile lambda): ~10 min first run;
#     cached to outputs/ca/models/ for subsequent runs
#
# Outputs (written to outputs/ca/):
#   results_ca.csv         — Table 3 equivalent (wMSE, Gini, Slope per model)
#   results_ca_tercile.csv — Same broken down by size tercile
#   bootstrap_ci_ca.csv    — 90% bootstrap CI on wMSE improvement vs B-S
#   wmse_by_tercile.png    — Figure: wMSE improvement by size tercile
#   lambda_posteriors.png  — Figure: lambda posterior by size tercile
# =============================================================================

# Paths — set to the directory containing this script
`%||%` <- function(a, b) if (!is.null(a)) a else b

REPO_DIR <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),   # when sourced
  error = function(e) getwd()                   # when run via Rscript
)
DATA_DIR   <- file.path(REPO_DIR, "data")
OUTPUT_DIR <- file.path(REPO_DIR, "outputs", "ca")
MODEL_DIR  <- file.path(REPO_DIR, "outputs", "ca", "models")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(MODEL_DIR,  showWarnings = FALSE, recursive = TRUE)

# Load helpers
source(file.path(REPO_DIR, "R", "02_features.R"))
source(file.path(REPO_DIR, "R", "05_evaluate.R"))
source(file.path(REPO_DIR, "R", "06_plots.R"))

# Data
LINE_OF_BUSINESS <- "ca"
source(file.path(REPO_DIR, "R", "01_data.R"))

# Models
source(file.path(REPO_DIR, "R", "03_models_ca.R"))

# Evaluate
results         <- eval_all_models(df_test)
results_tercile <- eval_by_tercile(df_test)

message("\n=== Commercial Auto — Test Set Results ===")
print(results, n = Inf)

message("\n--- By size tercile ---")
print(results_tercile %>% arrange(Tercile, Model), n = Inf)

# Bootstrap CI (takes ~1 min with B = 2000)
message("\nComputing bootstrap CIs...")
boot_ci <- bootstrap_wmse_improvement(df_test, B = 2000)
message("Bootstrap CIs (90%, % improvement vs B-S standard):")
print(boot_ci)

# Save outputs
write.csv(results,         file.path(OUTPUT_DIR, "results_ca.csv"),         row.names = FALSE)
write.csv(results_tercile, file.path(OUTPUT_DIR, "results_ca_tercile.csv"), row.names = FALSE)
write.csv(boot_ci,         file.path(OUTPUT_DIR, "bootstrap_ci_ca.csv"),    row.names = FALSE)

save_plots(results, results_tercile, models, df_train,
           lob_label = "Commercial Auto", out_dir = OUTPUT_DIR)

message("\nDone. Outputs written to: ", OUTPUT_DIR)
