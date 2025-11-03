# helper.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import os
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# ---------------- root discovery ----------------
def find_contrastive_root(start: Path = Path.cwd()) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("❌ Could not find 'Contrastive_Learning' directory from current path.")

def get_prediction_file(ROOT_RESULTS, result_file_name):
    """
    Given a result folder (exprmnt_YYYY_MM_DD__), find its test_set prediction file.
    """
    eval_dir = f"{ROOT_RESULTS}/{result_file_name}/ensemble_evaluation_from_valdiation/test_set_evaluation/tsplice_final_predictions_all_tissues.tsv"
    return eval_dir


# ---------------- column cleanup & scaling ----------------
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.columns = [c.replace("--", " - ") for c in df.columns]
    return df

def _detect_and_scale_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    """Ensure predictions are PSI in [0,1]. If they look like %, divide by 100."""
    if pred.shape[1] <= 1:
        return pred
    numeric = pred.select_dtypes(include=[np.number])
    if numeric.empty:
        return pred
    vals = numeric.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size and np.median(vals) >= 1.5:
        pred.iloc[:, 1:] = pred.iloc[:, 1:] / 100.0
    return pred

# ---------------- logit & Δlogit ----------------
def logit_safe(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps).astype(float)
    return np.log(p) - np.log1p(-p)

def delta_logit_scores(psi_pred: np.ndarray, mean_logit_scalar_or_vec) -> np.ndarray:
    """Δlogit = logit(psi_pred) - logit(mean_psi). mean_logit can be scalar or 1D array."""
    return logit_safe(psi_pred) - np.asarray(mean_logit_scalar_or_vec, dtype=float)

# ---------------- alignment ----------------
def load_and_align_for_delta_logit(
    gt_file: str,
    pred_file: str,
    require_cols: List[str] = ["logit_mean_psi"]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns a merged DataFrame with:
      exon_id, logit_mean_psi, <tissues>_gt, <tissues>_pred
    and the list of tissue column base-names.
    """
    gt   = _standardize_columns(pd.read_csv(gt_file))
    pred = _standardize_columns(pd.read_csv(pred_file, sep="\t"))
    pred = _detect_and_scale_predictions(pred)  # PSI in [0,1]

    for col in require_cols:
        if col not in gt.columns:
            raise ValueError(f"Expected '{col}' in GT file.")

    # tissues = intersection
    tissue_cols = [c for c in pred.columns if c in gt.columns and c != "exon_id"]
    if not tissue_cols:
        raise ValueError("No overlapping tissue columns between GT and predictions.")

    gt_use   = gt[["exon_id"] + require_cols + tissue_cols].drop_duplicates(subset=["exon_id"]).copy()
    pred_use = pred[["exon_id"] + tissue_cols].drop_duplicates(subset=["exon_id"]).copy()

    merged = pd.merge(gt_use, pred_use, on="exon_id", how="inner", suffixes=("_gt", "_pred"))
    return merged, tissue_cols


def pull_pm1_vectors_from_row(row: pd.Series, tissue_cols: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a merged row (has <tissue>_gt and <tissue>_pred):
      - Keep only GT in {-1, +1} (drop 0 and NaN)
      - Return (y_true_bin, y_psi, mask_idx)
        where y_true_bin is 0/1 with (-1 -> 0, +1 -> 1)
    """
    g = pd.to_numeric(row[[f"{t}_gt" for t in tissue_cols]], errors="coerce").to_numpy(dtype="float64")
    p = pd.to_numeric(row[[f"{t}_pred" for t in tissue_cols]], errors="coerce").to_numpy(dtype="float64")

    # valid where GT is -1 or +1, pred is finite
    valid = np.isfinite(p) & np.isfinite(g) & ((g == -1) | (g == 1))
    if not np.any(valid):
        return np.array([], dtype=int), np.array([], dtype=float), valid

    y_true_bin = (g[valid] == 1).astype(int)  # +1 -> 1, -1 -> 0
    y_psi = p[valid].astype(float)
    return y_true_bin, y_psi, valid

# ---------------- vectorization / masking ----------------
def pull_vectors_from_row(
    row: pd.Series,
    tissue_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """From a merged *row*, return (y_true, y_psi) masked to finite."""
    y_true = pd.to_numeric(row[[f"{t}_gt" for t in tissue_cols]], errors="coerce").to_numpy(dtype="float64")
    y_psi  = pd.to_numeric(row[[f"{t}_pred" for t in tissue_cols]], errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_psi)
    return y_true[mask].astype(int), y_psi[mask]

def pull_vectors_per_tissue(
    merged: pd.DataFrame,
    tissue: str
) -> Tuple[np.ndarray, np.ndarray]:
    """From merged *column*, return (y_true, y_psi) masked to finite across exons."""
    g = pd.to_numeric(merged[f"{tissue}_gt"], errors="coerce")
    p = pd.to_numeric(merged[f"{tissue}_pred"], errors="coerce")
    mask = g.isin([0, 1]) & (~p.isna())
    return g[mask].to_numpy(dtype=int), p[mask].to_numpy(dtype="float64"), mask.to_numpy()

# ---------------- scoring & predictions ----------------
def curve_score_from_dlogit(y_score: np.ndarray, expression_type: str) -> np.ndarray:
    """
    For ROC/PR, score must be higher = more likely positive.
    HIGH: positive ⇔ Δlogit < 0  → invert
    LOW : positive ⇔ Δlogit > 0  → as-is
    """
    if expression_type.upper() == "HIGH":
        return -y_score
    return y_score

def hard_preds_from_dlogit(y_score: np.ndarray, expression_type: str) -> np.ndarray:
    """Hard labels at Δlogit = 0."""
    if expression_type.upper() == "HIGH":
        return (y_score < 0).astype(int)
    return (y_score > 0).astype(int)

# ---------------- metrics ----------------
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Guarded metrics: returns accuracy/precision/recall/F1 + AUROC/AUPRC (NaN if degenerate)."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": np.nan,
        "auprc": np.nan,
    }
    pos = int(y_true.sum()); neg = int(len(y_true) - pos)
    if pos > 0 and neg > 0 and np.isfinite(y_score).all():
        out["auroc"] = roc_auc_score(y_true, y_score)
        out["auprc"] = average_precision_score(y_true, y_score)
    return out







# # util.py

# import pandas as pd
# import numpy as np
# from typing import List, Tuple, Dict, Optional
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix







# # ---------------- prediction utilities ----------------
# def _detect_and_scale_predictions(pred: pd.DataFrame) -> pd.DataFrame:
#     """
#     Ensure prediction matrix is in PSI ∈ [0,1] (decimal).
#     If we detect values like 35, 62, ... we assume input is in % and divide by 100.
#     """
#     if pred.shape[1] <= 1:
#         return pred

#     # peek numeric part (excluding exon_id if present)
#     numeric = pred.select_dtypes(include=[np.number])
#     if numeric.empty:
#         return pred

#     med = numeric.values.flatten()
#     med = med[~np.isnan(med)]
#     if med.size == 0:
#         return pred

#     median_val = np.median(med)
#     # If median >= 1.5, assume percent (e.g., 35) → convert to decimals
#     if median_val >= 1.5:
#         pred.iloc[:, 1:] = pred.iloc[:, 1:] / 100.0
#     else:
#         # if already in [0,1], keep as is
#         pass
#     return pred

# def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Trim whitespace and normalize some oddities in column names."""
#     df = df.copy()
#     df.columns = df.columns.str.strip()
#     # normalize double hyphens often found in ASCOT tissue names
#     df.columns = [c.replace("--", " - ") for c in df.columns]
#     return df


# def logit_safe(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
#     p = np.clip(p, eps, 1 - eps).astype(float)
#     return np.log(p) - np.log1p(-p)  # log(p) - log(1-p)

# def delta_logit_scores(psi_pred: np.ndarray, mean_logit: np.ndarray) -> np.ndarray:
#     """
#     Δlogit = logit(psi_pred) - logit(mean_psi)
#     psi_pred in [0,1]; mean_logit already in logit space.
#     """
#     return logit_safe(psi_pred) - mean_logit


# def align_gt_and_pred_tissues(
#     gt_df: pd.DataFrame,
#     pred_df: pd.DataFrame,
#     meta_cols: Optional[set] = None
# ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
#     """
#     Align ground-truth PSI table and prediction PSI table on shared tissue columns.
#     Both should contain 'exon_id'. Returns (gt, pred, tissue_cols).
#     Assumes PSI values are in decimals [0,1] and that gt_df has a 'mean_psi' column (in decimals).
#     """
#     if meta_cols is None:
#         meta_cols = {
#             "exon_id","cassette_exon","alternative_splice_site_group","linked_exons",
#             "mutually_exclusive_exons","exon_strand","exon_length","gene_type",
#             "gene_id","gene_symbol","exon_location","exon_boundary",
#             "chromosome","chr","mean_psi","logit_mean_psi","chromosome.1"
#         }

#     gt_df = _standardize_columns(gt_df)
#     pred_df = _standardize_columns(pred_df)

#     # Coerce tissues to numeric (keep NaN)
#     gt_df = gt_df.copy()
#     pred_df = pred_df.copy()

#     # detect and scale predictions to decimals
#     pred_df = _detect_and_scale_predictions(pred_df)

#     # If 'mean_psi' is in percent in gt_df, convert to decimals
#     if "mean_psi" in gt_df.columns and gt_df["mean_psi"].dropna().median() > 1:
#         gt_df["mean_psi"] = gt_df["mean_psi"] / 100.0

#     # For tissue columns: present in both and not metadata
#     tissue_cols = [c for c in pred_df.columns if c in gt_df.columns and c != "exon_id" and c not in meta_cols]
#     if not tissue_cols:
#         raise ValueError("No overlapping tissue columns between GT and predictions.")

#     # Coerce these columns to numeric decimals
#     gt_df[tissue_cols] = gt_df[tissue_cols].apply(pd.to_numeric, errors="coerce")
#     # If GT looks like percent (>1), convert to decimals
#     if (gt_df[tissue_cols].max(numeric_only=True).max() or 0) > 1:
#         gt_df[tissue_cols] = gt_df[tissue_cols] / 100.0

#     pred_df[tissue_cols] = pred_df[tissue_cols].apply(pd.to_numeric, errors="coerce")

#     # Keep only needed columns
#     gt_use = gt_df[["exon_id", "mean_psi"] + tissue_cols].copy()
#     pred_use = pred_df[["exon_id"] + tissue_cols].copy()

#     # Inner join on exon_id to guarantee 1–1 alignment
#     gt_use = gt_use.drop_duplicates(subset=["exon_id"])
#     pred_use = pred_use.drop_duplicates(subset=["exon_id"])
#     merged_exons = pd.merge(gt_use[["exon_id"]], pred_use[["exon_id"]], on="exon_id", how="inner")

#     gt_use = pd.merge(merged_exons, gt_use, on="exon_id", how="left")
#     pred_use = pd.merge(merged_exons, pred_use, on="exon_id", how="left")

#     return gt_use, pred_use, tissue_cols

# # ---------------- class-3 labeling ----------------
# def class3_from_mean_and_psi(
#     mean_psi: pd.Series,
#     psi_df: pd.DataFrame,
#     margin: float = 0.10
# ) -> pd.DataFrame:
#     """
#     Compute 3-class labels (-1, 0, +1) per (exon, tissue) using:
#       0  if |psi - mean| ≤ margin
#       +1 if (psi - mean) >  margin
#       -1 if (mean - psi) >  margin
#     mean_psi and psi_df must both be in decimals [0,1].
#     Returns a DataFrame with same index/columns as psi_df, dtype Int8 (nullable), with pd.NA for missing PSI.
#     """
#     if not 0 <= margin <= 1:
#         raise ValueError("margin should be in [0,1] (decimal PSI units).")

#     # Broadcast mean to each tissue column
#     mu = mean_psi.values.reshape(-1, 1)
#     lower = np.clip(mu - margin, 0.0, 1.0)
#     upper = np.clip(mu + margin, 0.0, 1.0)

#     psi_vals = psi_df.values.astype(float)
#     is_nan = np.isnan(psi_vals)

#     cls = np.where(psi_vals > upper,  1,
#           np.where(psi_vals < lower, -1, 0))

#     out = pd.DataFrame(cls, index=psi_df.index, columns=psi_df.columns)
#     out = out.astype("Int8")
#     # restore missing PSI as missing labels
#     out = out.mask(is_nan, pd.NA)
#     return out

# # ---------------- evaluation ----------------
# def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     """
#     Multiclass metrics for labels in {-1,0,+1}. NaNs should already be removed.
#     Returns accuracy and macro/micro F1.
#     """
#     metrics = {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
#         "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
#     }
#     return metrics

# def evaluate_tissue_specific_exon_classification(
#     gt_df: pd.DataFrame,
#     pred_df: pd.DataFrame,
#     tissue_cols: List[str],
#     margin: float = 0.10
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Build class-3 labels for GT and PRED, then compute per-tissue metrics.
#     Returns:
#       - per_tissue_metrics: DataFrame of metrics per tissue
#       - long_gt: long table (exon_id, tissue, class3_gt)
#       - long_pred: long table (exon_id, tissue, class3_pred)
#     """
#     # Compute class-3
#     gt_cls = class3_from_mean_and_psi(gt_df["mean_psi"], gt_df[tissue_cols], margin=margin)
#     pr_cls = class3_from_mean_and_psi(gt_df["mean_psi"], pred_df[tissue_cols], margin=margin)  # use same mean_psi

#     # Long format
#     long_gt = gt_cls.copy()
#     long_gt.insert(0, "exon_id", gt_df["exon_id"].values)
#     long_gt = long_gt.melt(id_vars=["exon_id"], var_name="tissue", value_name="class3_gt")

#     long_pred = pr_cls.copy()
#     long_pred.insert(0, "exon_id", pred_df["exon_id"].values)
#     long_pred = long_pred.melt(id_vars=["exon_id"], var_name="tissue", value_name="class3_pred")

#     merged = pd.merge(long_gt, long_pred, on=["exon_id", "tissue"], how="inner")

#     # Per-tissue metrics with NaN handling: drop rows where either is NA
#     per_tissue_rows = []
#     for t in tissue_cols:
#         sub = merged[merged["tissue"] == t]
#         mask = (~sub["class3_gt"].isna()) & (~sub["class3_pred"].isna())
#         sub = sub[mask]
#         if sub.empty:
#             per_tissue_rows.append({"tissue": t, "n": 0, "accuracy": np.nan, "f1_macro": np.nan, "f1_micro": np.nan})
#             continue

#         y_true = sub["class3_gt"].astype("Int8").to_numpy()
#         y_pred = sub["class3_pred"].astype("Int8").to_numpy()

#         m = multiclass_metrics(y_true, y_pred)
#         m.update({"tissue": t, "n": int(len(sub))})
#         per_tissue_rows.append(m)

#     per_tissue_metrics = pd.DataFrame(per_tissue_rows).sort_values("tissue").reset_index(drop=True)
#     return per_tissue_metrics, long_gt, long_pred
