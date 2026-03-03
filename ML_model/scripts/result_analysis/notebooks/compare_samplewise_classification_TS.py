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


    # Path to your CSV file (adjust if needed)
    class_csv = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/tissue_counts_detailed_ExonBinPsi.csv"

    # Load the tissue class mapping
    class_df = pd.read_csv(class_csv)[["Tissue", "SampleClass"]]
    class_df.rename(columns={"Tissue": "tissue"}, inplace=True)

    # Merge with your main df (assuming both share a "tissue" column)
    df = df.merge(class_df, on="tissue", how="left")

    # === Step 5: Summarize each class ===
    # --- Compute groupwise mean/std for each metric by SampleClass and Type ---
    class_summary = (
        df.groupby(["SampleClass", "Type"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # --- Step 6: Display results ---
    print("\n📊 Example tissue-level rows:")
    print(df.head(10).to_string(index=False))

    print("\n📈 Summary of metrics by sample-size class and model type:")
    print(class_summary.to_string(index=False))
    # === Optional: save results ===
    # summary_df.to_csv(f"{root_path}/sample_size_summary_TS_{mode}.csv", index=False)
    class_summary.to_csv(f"{root_path}/sample_size_ranges_TS_{mode}.csv", index=False)


    # --- Save results ---
    # summary_path = os.path.join(out_path, f"sample_size_summary_TS_{mode}.csv")
    # range_path = os.path.join(out_path, f"sample_size_ranges_TS_{mode}.csv")

    # summary.to_csv(summary_path)
    # ranges.to_csv(range_path)

    # print(f"✅ Saved summary → {summary_path}")
    # print(f"✅ Saved sample ranges → {range_path}")

    # --- Print quick view ---
    # print("\n📈 Average metrics by sample-size group:")
    # print(summary)
    # print("\n📊 Sample ranges per group:")
    # print(ranges)


# Example usage
if __name__ == "__main__":

    root_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/RECOMB_26/classification/"
    out_path = root_path

    pred_csv = f"{root_path}TS_CLSwpd_300bp_10Aug_tissue_metrics_triclass_20251106-213156.csv"
    sota_csv = f"{root_path}TS_noCL_300bp_rerun_codeChange_tissue_metrics_triclass_20251106-213156.csv"


    # pred_csv = f"{root_path}TS_CLSwpd_300bp_10Aug_tissue_metrics_logitdelta_binary_20251106-213156.csv"
    # sota_csv = f"{root_path}TS_noCL_300bp_rerun_codeChange_tissue_metrics_logitdelta_binary_20251106-213156.csv"


    summarize_by_sample_size(sota_csv, pred_csv, out_path)
