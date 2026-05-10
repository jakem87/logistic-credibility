# =============================================================================
# run_nesting_demo.R  —  Nesting demonstration (no Stan required)
#
# Usage:  source("run_nesting_demo.R")   or   Rscript run_nesting_demo.R
#
# Runtime: ~20 seconds
# =============================================================================

library(scales)    # for percent labels in plots

`%||%` <- function(a, b) if (!is.null(a)) a else b
REPO_DIR <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile)),
  error = function(e) getwd()
)

source(file.path(REPO_DIR, "R", "07_nesting_demo.R"))
