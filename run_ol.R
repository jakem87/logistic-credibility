# =============================================================================
# run_ol.R  —  Other Liability: cross-LoB validation
#
# Usage:  source("run_ol.R")   or   Rscript run_ol.R
#
# Reuses all model functions from 03_models_ca.R.
# Outputs written to outputs/ol/.
# =============================================================================

`%||%` <- function(a, b) if (!is.null(a)) a else b

REPO_DIR <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
DATA_DIR   <- file.path(REPO_DIR, "data")
OUTPUT_DIR <- file.path(REPO_DIR, "outputs", "ol")
MODEL_DIR  <- file.path(REPO_DIR, "outputs", "ol", "models")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(MODEL_DIR,  showWarnings = FALSE, recursive = TRUE)

source(file.path(REPO_DIR, "R", "02_features.R"))
source(file.path(REPO_DIR, "R", "05_evaluate.R"))
source(file.path(REPO_DIR, "R", "06_plots.R"))

# Data (Other Liability)
LINE_OF_BUSINESS <- "ol"
source(file.path(REPO_DIR, "R", "01_data.R"))

# Model functions (same as CA — OL uses identical specifications)
source(file.path(REPO_DIR, "R", "03_models_ca.R"))   # loads fit_* functions
source(file.path(REPO_DIR, "R", "04_models_ol.R"))   # fits on OL data

# Evaluate
results         <- eval_all_models(df_test)
results_tercile <- eval_by_tercile(df_test)

message("\n=== Other Liability — Test Set Results ===")
print(results, n = Inf)

write.csv(results,         file.path(OUTPUT_DIR, "results_ol.csv"),         row.names = FALSE)
write.csv(results_tercile, file.path(OUTPUT_DIR, "results_ol_tercile.csv"), row.names = FALSE)

save_plots(results, results_tercile, models, df_train,
           lob_label = "Other Liability", out_dir = OUTPUT_DIR)

message("\nDone. Outputs written to: ", OUTPUT_DIR)
