import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
import time
import os
from scipy.special import logit

# Timestamp for output
timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

# ===============================
# Loaders
# ===============================
def load_predictions(path: str) -> pd.DataFrame:
    """Load predicted ΔPSI (or Δlogit) CSV/TSV file."""
    ext = os.path.splitext(path)[1]
    sep = "\t" if ext == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    df.rename(columns={df.columns[0]: "exon_id"}, inplace=True)
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth PSI CSV file."""
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "exon_id"}, inplace=True)
    return df


# ===============================
# ΔPSI computation
# ===============================
def compute_delta_psi(psi_percent: pd.Series, mean_psi: pd.Series) -> np.ndarray:
    """
    Compute ΔPSI = PSI - mean_psi (both in % units).
    """
    return psi_percent - mean_psi


# ===============================
# Merging + comparison
# ===============================
def merge_delta_psi(gt_df: pd.DataFrame, pred_df: pd.DataFrame, tissue: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge and extract ΔPSI (true vs predicted) for a tissue.
    Ground truth ΔPSI = (PSI_tissue - mean_psi)
    Predicted ΔPSI = directly from prediction CSV (already ΔPSI per tissue)
    """
    if tissue not in pred_df.columns:
        raise ValueError(f"Tissue '{tissue}' not found in predicted ΔPSI file.")
    if tissue not in gt_df.columns:
        raise ValueError(f"Tissue '{tissue}' not found in ground truth PSI file.")

    merged = pd.merge(
        gt_df[["exon_id", tissue, "mean_psi"]],
        pred_df[["exon_id", tissue]],
        on="exon_id",
        suffixes=("_gt", "_pred")
    )

    delta_true = compute_delta_psi(merged[f"{tissue}_gt"], merged["mean_psi"])
    delta_pred = merged[f"{tissue}_pred"].to_numpy()

    mask = ~pd.isna(delta_true) & ~pd.isna(delta_pred)
    return delta_true[mask], delta_pred[mask]


# ===============================
# Plotting
# ===============================
def plot_histogram(y_true, y_pred, tissue: str, value_type: str = "ΔPSI", save_path: str = None):
    """Plot histogram for ΔPSI comparison."""
    plt.figure(figsize=(12, 5))
    plt.hist(y_true, bins=50, alpha=0.5, label=f"{value_type}_true", density=True, color="skyblue")
    plt.hist(y_pred, bins=50, alpha=0.5, label=f"{value_type}_pred", density=True, color="orange")

    plt.xlabel(value_type)
    plt.ylabel("Density")
    plt.title(f"Histogram of {value_type} in {tissue}: True vs Predicted")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()


# ===============================
# Unified class
# ===============================
class DeltaPSIPlotter:
    def __init__(self, pred_path: str, gt_path: str):
        self.pred_df = load_predictions(pred_path)
        self.gt_df = load_ground_truth(gt_path)

    def plot_tissue(self, tissue: str, save_path: str = None):
        y_true, y_pred = merge_delta_psi(self.gt_df, self.pred_df, tissue)
        plot_histogram(y_true, y_pred, tissue, value_type="ΔPSI", save_path=save_path)

    def compute_spearman(self, tissue: str):
        """Compute Spearman correlation between predicted and true ΔPSI."""
        from scipy.stats import spearmanr
        y_true, y_pred = merge_delta_psi(self.gt_df, self.pred_df, tissue)
        rho, _ = spearmanr(y_true, y_pred)
        print(f"📈 Spearman (ΔPSI, {tissue}): {rho:.4f}")
        return rho

    def plot_for_multiple_tissues(self, tissues: list):
        for tissue in tissues:
            try:
                self.plot_tissue(tissue)
            except ValueError as e:
                print(f"⚠️ {e}")


# ===============================
# Example usage
# ===============================
running_platform = "NYGC"
tissue = "Retina - Eye"
run_name = "mtsplice_originalTFweight_results"
ground_truth_file = "variable_cassette_exons_with_logit_mean_psi.csv"

if running_platform == "NYGC":
    server_path = "/gpfs/commons/home/atalukder/"
else:
    server_path = "/mnt/home/at3836/"

# Paths
prediction_csv = f"{server_path}Contrastive_Learning/files/results/mtsplice_originalTFweight_results/variable_all_tissues_predicted_logit_delta.tsv"
ground_truth_csv = f"{server_path}Contrastive_Learning/data/final_data/ASCOT_finetuning/{ground_truth_file}"

save_path = f"{server_path}Contrastive_Learning/code/ML_model/scripts/figures/{tissue}_Delta_PSI_Histogram{timestamp}.png"

plotter = DeltaPSIPlotter(prediction_csv, ground_truth_csv)
plotter.plot_tissue(tissue, save_path)
plotter.compute_spearman(tissue)
