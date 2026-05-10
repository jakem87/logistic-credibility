"""
example_cas_data.py — End-to-end example on CAS Commercial Auto data.

Reproduces Table 3 rows 3, 6, 7 (B-S standard, Joint-Decay scalar λ,
Joint-Decay continuous λ) using Python. Results should match the R output
within rounding (same optimisation problem, different implementation).

Expected wMSE values (CA test set AY 2006-2007, absolute scale):
  B-S standard             ≈ 0.01327  (reference)
  Joint-Decay scalar λ     ≈ 0.00872  (~34% improvement)
  Joint-Decay continuous λ ≈ 0.00838  (~37% improvement)

Note: The Bayesian tercile-λ model (proposed model, Table 3 row 8) achieves
~29-32% improvement but requires brms/Stan. Use the R implementation for it:
  source("run_ca.R")

Usage:
    python example_cas_data.py          # downloads data if not present
    python example_cas_data.py --plot   # also saves Z-curve plot
"""

import argparse
import os
import sys

# Ensure UTF-8 output on Windows (handles λ, ü in model names)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running from repo root or from python/ subdirectory
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from credibility import (
    load_cas_data,
    build_panel,
    fit_bs_standard,
    predict_bs_standard,
    fit_jdecay_scalar,
    predict_jdecay_scalar,
    fit_jdecay_scalar_map,
    predict_jdecay_scalar_map,
    fit_jdecay_continuous,
    predict_jdecay_continuous,
    fit_jdecay_tercile,
    predict_jdecay_tercile,
    fit_jdecay_tercile_map,
    predict_jdecay_tercile_map,
    eval_all_models,
    plot_zcurves,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_repo = os.path.dirname(_here)
DATA_DIR   = os.path.join(_repo, "data")
OUTPUT_DIR = os.path.join(_repo, "outputs", "ca_python")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "comauto_pos_98-07.csv")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(make_plot: bool = False) -> None:
    print("=" * 60)
    print("Logistic Credibility — Python Example (Commercial Auto)")
    print("=" * 60)

    # 1. Load and build panel
    print("\n--- Loading data ---")
    df_raw = load_cas_data(DATA_FILE, lob="ca")

    print("\n--- Building panel ---")
    df_train, df_test, tr_stats = build_panel(df_raw)

    # 2. Fit models
    print("\n--- Fitting models ---")
    fit_bs      = fit_bs_standard(df_train)
    fit_sc      = fit_jdecay_scalar(df_train)
    fit_sc_map  = fit_jdecay_scalar_map(df_train)
    fit_cont    = fit_jdecay_continuous(df_train)
    fit_trc     = fit_jdecay_tercile(df_train)
    fit_trc_map = fit_jdecay_tercile_map(df_train)

    # 3. Generate test predictions (relative space)
    preds = {
        "Bühlmann-Straub (standard)":         predict_bs_standard(fit_bs, df_test),
        "Joint-Decay (scalar λ)":             predict_jdecay_scalar(fit_sc, df_test),
        "Joint-Decay (scalar λ, MAP)":        predict_jdecay_scalar_map(fit_sc_map, df_test),
        "Joint-Decay (continuous λ)":         predict_jdecay_continuous(fit_cont, df_test),
        "Joint-Decay (tercile λ) [MLE]":      predict_jdecay_tercile(fit_trc, df_test),
        "Joint-Decay (tercile λ, MAP)":       predict_jdecay_tercile_map(fit_trc_map, df_test),
    }
    # Market mean and last-year baselines
    preds["Market Mean (baseline)"] = np.ones(len(df_test))
    preds["Last Year LR (naive)"]   = (df_test["lr_lag1"].values / df_test["mu_t"].values)

    # 4. Evaluate
    print("\n--- Evaluating ---")
    display_order = [
        "Market Mean (baseline)",
        "Last Year LR (naive)",
        "Bühlmann-Straub (standard)",
        "Joint-Decay (scalar λ)",
        "Joint-Decay (scalar λ, MAP)",
        "Joint-Decay (continuous λ)",
        "Joint-Decay (tercile λ) [MLE]",
        "Joint-Decay (tercile λ, MAP)",
    ]
    preds_ordered = {k: preds[k] for k in display_order if k in preds}
    results = eval_all_models(df_test, preds_ordered)

    print("\n=== Commercial Auto — Test Set Results (Python) ===")
    print(results.to_string(index=False))

    # Save
    out_csv = os.path.join(OUTPUT_DIR, "results_ca_python.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    # 5. Equivalence check
    # Python ML models give ~17-20% wMSE improvement over B-S.
    # The paper's 29-32% figure is from the Bayesian tercile-λ model (R only).
    print("\n--- Model comparison notes ---")
    bs_row  = results[results["Model"].str.contains("Bühlmann-Straub", regex=False)]
    sc_row  = results[results["Model"] == "Joint-Decay (scalar λ)"]
    if len(bs_row) == 1 and len(sc_row) == 1:
        bs_wmse = float(bs_row["wMSE"].iloc[0])
        sc_wmse = float(sc_row["wMSE"].iloc[0])
        pct_imp = 100 * (bs_wmse - sc_wmse) / bs_wmse
        print(f"  B-S wMSE:          {bs_wmse:.5f}")
        print(f"  Scalar-λ wMSE:     {sc_wmse:.5f}  ({pct_imp:.1f}% improvement)")
        print(f"  (Paper 29-32% improvement is from the Bayesian tercile-λ model — R only)")

    # 6. Optional Z-curve plot
    if make_plot:
        fig, ax = plt.subplots(figsize=(7, 4))
        plot_zcurves(fit_sc, fit_bs, df_train, lob_label="Commercial Auto", ax=ax)
        fig.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "zcurves_python.png")
        fig.savefig(plot_path, dpi=150)
        print(f"\nZ-curve plot saved to: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic credibility CA example")
    parser.add_argument("--plot", action="store_true", help="Save Z-curve plot")
    args = parser.parse_args()
    main(make_plot=args.plot)
