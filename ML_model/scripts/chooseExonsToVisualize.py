#!/usr/bin/env python3
"""
Filter exons by tissue expression level and overlap with Multiz dataset.
Author: Argha Talukder
"""

import pandas as pd
import pickle
import argparse
from pathlib import Path
import os
import sys

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

try:
    root_path = str(find_contrastive_root())
    os.environ["CONTRASTIVE_ROOT"] = root_path
    print(f"CONTRASTIVE_ROOT set to: {root_path}") 
except RuntimeError as e:
    print(f"Error finding CONTRASTIVE_ROOT: {e}.")
    sys.exit(1)


def load_data(multiz_path, psi_path, train_list_path):
    df_multiz = pd.read_csv(multiz_path)
    df_psi = pd.read_csv(psi_path)
    df_train = pd.read_csv(train_list_path)
    return df_multiz, df_psi, df_train



def filter_exons_by_tissue(
    df_psi: pd.DataFrame,
    tissue: str,
    top_frac: float = 0.20,
    bottom_frac: float = 0.20,
    tol: float = 10.0,  # within ±10 PSI of mean_psi
):
    """
    Select two classes for a tissue:
      - HIGH class: exons whose mean_psi is in the top `top_frac` and the tissue PSI is within ±tol of mean_psi
      - LOW class:  exons whose mean_psi is in the bottom `bottom_frac` and the tissue PSI is within ±tol of mean_psi
    """
    if tissue not in df_psi.columns:
        raise ValueError(f"Tissue '{tissue}' not found in PSI file columns.")
    if "mean_psi" not in df_psi.columns:
        raise ValueError("Column 'mean_psi' not found in PSI file.")

    # Drop rows with missing values needed for the logic
    df = df_psi.dropna(subset=[tissue, "mean_psi"]).copy()

    # Sort by mean_psi (global average expression)
    df_sorted = df.sort_values("mean_psi", ascending=False)
    n = len(df_sorted)
    n_top = max(1, int(top_frac * n))
    n_bot = max(1, int(bottom_frac * n))

    # Candidates by mean
    high_mean = df_sorted.head(n_top)
    low_mean  = df_sorted.tail(n_bot)

    # Keep exons whose tissue PSI is close to their mean_psi (not DE in that tissue)
    close_to_mean_high = high_mean[(high_mean[tissue] - high_mean["mean_psi"]).abs() <= tol]
    close_to_mean_low  = low_mean[(low_mean[tissue]  - low_mean["mean_psi"]).abs()  <= tol]

    # (Optional) if you want to enforce that tissue PSI is also high/low in absolute terms:
    # close_to_mean_high = close_to_mean_high[close_to_mean_high[tissue] >= close_to_mean_high["mean_psi"] - tol]
    # close_to_mean_low  = close_to_mean_low[close_to_mean_low[tissue]  <= close_to_mean_low["mean_psi"] + tol]

    return close_to_mean_high, close_to_mean_low

def filter_diff_exons_by_tissue(
    df_psi: pd.DataFrame,
    tissue: str,
    top_frac: float = 0.30,
    bottom_frac: float = 0.30,
    n_select: int = 100,
    delta_min: float = 20.0  # minimum difference between tissue and mean_psi
):
    """
    Select exons that differ strongly from their mean_psi in a tissue:
      - HIGH_DIFF_LOW: exons whose mean_psi is high but tissue PSI is much lower
      - LOW_DIFF_HIGH: exons whose mean_psi is low but tissue PSI is much higher
    Returns up to `n_select` exons per class.
    """
    if tissue not in df_psi.columns:
        raise ValueError(f"Tissue '{tissue}' not found in PSI file columns.")
    if "mean_psi" not in df_psi.columns:
        raise ValueError("Column 'mean_psi' not found in PSI file.")

    df = df_psi.dropna(subset=[tissue, "mean_psi"]).copy()

    # Sort by mean expression
    df_sorted = df.sort_values("mean_psi", ascending=False)
    n = len(df_sorted)
    n_top = max(1, int(top_frac * n))
    n_bot = max(1, int(bottom_frac * n))

    high_mean = df_sorted.head(n_top)
    low_mean = df_sorted.tail(n_bot)

    # 1️⃣ High mean, tissue much lower than average → "high_mean_low_tissue"
    high_mean_low_tissue = high_mean[(high_mean["mean_psi"] - high_mean[tissue]) >= delta_min]

    # 2️⃣ Low mean, tissue much higher than average → "low_mean_high_tissue"
    low_mean_high_tissue = low_mean[(low_mean[tissue] - low_mean["mean_psi"]) >= delta_min]

    # Limit to n_select each
    high_mean_low_tissue = high_mean_low_tissue.nlargest(n_select, "mean_psi", keep="all")
    low_mean_high_tissue = low_mean_high_tissue.nsmallest(n_select, "mean_psi", keep="all")

    return high_mean_low_tissue, low_mean_high_tissue

# def map_exons(df_multiz, df_psi_subset):
#     """
#     Map ascot_exon_id → ENST exon name using the multiz overlap file.
#     """
#     merged = pd.merge(
#         df_psi_subset,
#         df_multiz[["Exon Name", "ascot_exon_id"]],
#         how="left",
#         left_on="exon_id",
#         right_on="ascot_exon_id"
#     )
#     merged = merged.dropna(subset=["Exon Name"])
#     return merged["Exon Name"].unique().tolist()


def map_exons(df_multiz, df_psi_subset, df_train):
    """
    Map ascot_exon_id → ENST exon name using multiz overlaps
    and retain only those present in the training exon list.
    """
    merged = pd.merge(
        df_psi_subset,
        df_multiz[["Exon Name", "ascot_exon_id"]],
        how="left",
        left_on="exon_id",
        right_on="ascot_exon_id"
    )
    merged = merged.dropna(subset=["Exon Name"])

    # Keep only exons that are in the multiz training set
    valid_training_exons = set(df_train["exon_id"])
    merged = merged[merged["Exon Name"].isin(valid_training_exons)]

    return merged["Exon Name"].unique().tolist()



def save_as_pickle(result_dict, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(result_dict, f)
    print(f"✅ Saved pickle to {output_path}")


def main(multiz_path, psi_path, train_list_path, output_path):
    df_multiz, df_psi, df_train = load_data(multiz_path, psi_path, train_list_path)
    # Detect all tissue columns dynamically
    exclude_cols = {"exon_id", "cassette_exon", "alternative_splice_site_group", "linked_exons",
                    "mutually_exclusive_exons", "exon_strand", "exon_length", "gene_type",
                    "gene_id", "gene_symbol", "exon_location", "exon_boundary", "chromosome",
                    "mean_psi", "logit_mean_psi"}
    tissue_cols = [c for c in df_psi.columns if c not in exclude_cols]
    tissues = tissue_cols
    results = {}

    for tissue in tissues:
        print(f"\n🧬 Processing tissue: {tissue}")
        # high, low = filter_exons_by_tissue(df_psi, tissue)
        high, low = filter_diff_exons_by_tissue(df_psi, tissue)
        high_enst = map_exons(df_multiz, high, df_train)
        low_enst = map_exons(df_multiz, low, df_train)
        results[tissue] = {"high_expression_exons": high_enst, "low_expression_exons": low_enst}
        print(f"  → {len(high_enst)} high, {len(low_enst)} low (training overlap)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_as_pickle(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and map exons by tissue expression.")
    parser.add_argument(
        "--multiz_path",
        default=f"{root_path}/data/ASCOT/train_cassette_exons_multizOverlaps.csv",
        help="Path to train_cassette_exons_multizOverlaps.csv"
    )
    parser.add_argument(
        "--psi_path",
        default=f"{root_path}/data/final_data_old/ASCOT_finetuning/train_cassette_exons_with_logit_mean_psi.csv",
        help="Path to train_cassette_exons_with_logit_mean_psi.csv"
    )
    parser.add_argument(
        "--train_list_path",
        default=f"{root_path}/data/final_data_old/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_exon_list.csv",
        help="Path to training exon list"
    )
    # parser.add_argument(
    #     "--output",
    #     default=f"{root_path}/data/extra/filtered_HiLow_exons_by_tissue.pkl",
    #     help="Output pickle file path"
    # )
    parser.add_argument(
        "--output",
        default=f"{root_path}/data/extra/filtered_avgHexpLVV_exons_by_tissue.pkl",
        help="Output pickle file path"
    )

    args = parser.parse_args()
    main(args.multiz_path, args.psi_path, args.train_list_path, args.output)
