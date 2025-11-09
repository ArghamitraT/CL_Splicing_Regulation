import pandas as pd
import numpy as np
import os

def summarize_by_sample_size(sota_csv, pred_csv, out_path):
    """
    Compare SOTA and prediction metrics grouped by sample size quantiles.

    Args:
        sota_csv (str): Path to SOTA metrics CSV.
        pred_csv (str): Path to prediction metrics CSV.
        out_path (str): Output folder for summary CSVs.
        mode (str): The mode of classification ("binary" or "triclass").
    """

    os.makedirs(out_path, exist_ok=True)

    # --- Load data ---
    df_sota = pd.read_csv(sota_csv)
    df_pred = pd.read_csv(pred_csv)

    # Standardize column names
    df_sota.columns = df_sota.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    # Detect metric type

    if "f1" in df_sota.columns or "f1" in df_pred.columns:
        mode = "binary"
        metric_cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    else:
        mode = "triclass"
        metric_cols = ["accuracy", "macro_f1", "weighted_f1",
                       "auroc_1_vs_-1", "auprc_1_vs_-1",
                       "auroc_0_vs_-1", "auprc_0_vs_-1"]

    print(f"📊 Detected {mode}-classification metrics.")

    # --- Combine both models ---
    df_sota["Type"] = "SOTA"
    df_pred["Type"] = "Prediction"
    df = pd.concat([df_sota, df_pred], ignore_index=True)

    # --- Sort by sample size ---
    df = df.sort_values("n")

    # --- Compute quantile-based bins ---
    q1, q2 = df["n"].quantile([1/3, 2/3])
    bins = [df["n"].min()-1, q1, q2, df["n"].max()+1]
    labels = ["Low (few samples)", "Medium", "High (many samples)"]
    df["SampleGroup"] = pd.cut(df["n"], bins=bins, labels=labels)

    # --- Aggregate by sample group ---
    summary = (
        df.groupby(["Type", "SampleGroup"])[metric_cols + ["n"]]
          .agg(["mean", "count"])
    )

    # --- Compute sample range per group ---
    ranges = (
        df.groupby("SampleGroup")["n"]
          .agg(["min", "max"])
          .rename(columns={"min": "Sample_Min", "max": "Sample_Max"})
    )

    # --- Save results ---
    summary_path = os.path.join(out_path, f"sample_size_summary_{mode}.csv")
    range_path = os.path.join(out_path, f"sample_size_ranges_{mode}.csv")

    summary.to_csv(summary_path)
    ranges.to_csv(range_path)

    print(f"✅ Saved summary → {summary_path}")
    print(f"✅ Saved sample ranges → {range_path}")

    # --- Print quick view ---
    print("\n📈 Average metrics by sample-size group:")
    print(summary)
    print("\n📊 Sample ranges per group:")
    print(ranges)


# Example usage
if __name__ == "__main__":

    root_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/RECOMB_26/classification/"
    out_path = root_path

    # pred_csv = f"{root_path}TS_CLSwpd_300bp_10Aug_tissue_metrics_triclass_20251106-213156.csv"
    # sota_csv = f"{root_path}TS_noCL_300bp_rerun_codeChange_tissue_metrics_triclass_20251106-213156.csv"


    pred_csv = f"{root_path}TS_CLSwpd_300bp_10Aug_tissue_metrics_logitdelta_binary_20251106-213156.csv"
    sota_csv = f"{root_path}TS_noCL_300bp_rerun_codeChange_tissue_metrics_logitdelta_binary_20251106-213156.csv"


    summarize_by_sample_size(sota_csv, pred_csv, out_path)
