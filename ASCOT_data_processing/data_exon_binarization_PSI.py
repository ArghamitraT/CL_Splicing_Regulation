import pandas as pd
import numpy as np
import os
from pathlib import Path

# ---------------- root discovery ----------------
def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path
print(f"CONTRASTIVE_ROOT set to: {root_path}")

# === Config ===
division = 'variable'
MARGIN = 0.10  # your ±0.1 window around mean_psi (in PSI units, not %)

# === I/O ===
FILE     = f"{root_path}/data/ASCOT/{division}_cassette_exons_with_logit_mean_psi.csv"
OUT_WIDE = f"{root_path}/data/ASCOT/{division}_cassette_exons_with_binary_labels_ExonBinPsi.csv"
# OUT_LONG = f"{root_path}/data/ASCOT/{division}_cassette_exons_class3_by_tissue_LONG.csv"
OUT_SUMM = f"{root_path}/data/ASCOT/{division}_cassette_exons_class3_counts_by_tissue.csv"

# === Load data ===
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()
original_cols = list(df.columns)  # preserve exact order


# Ensure 'mean_psi' exists and convert percent→decimal
if "mean_psi" not in df.columns:
    raise ValueError("No 'mean_psi' column found in file.")
df["mean_psi"] = df["mean_psi"] / 100.0

# Identify tissue columns (everything not metadata)
meta_cols_set = {
    "exon_id","cassette_exon","alternative_splice_site_group","linked_exons",
    "mutually_exclusive_exons","exon_strand","exon_length","gene_type",
    "gene_id","gene_symbol","exon_location","exon_boundary",
    "chromosome","chr","mean_psi","logit_mean_psi","chromosome.1"
}
meta_cols = [c for c in df.columns if c in meta_cols_set]  # ORDERED list

tissue_cols = [c for c in df.columns if c not in meta_cols]
print(f"Detected {len(tissue_cols)} tissue columns → {tissue_cols[:5]}...")

# --- 3-class labeling per tissue (exon-centric) ---
mu = df["mean_psi"].values  # (N,)
lower = np.maximum(mu - MARGIN, 0.0)
upper = np.minimum(mu + MARGIN, 1.0)

class_wide = pd.DataFrame({"exon_id": df["exon_id"] if "exon_id" in df.columns else df.index})
for t in tissue_cols:
    psi_t = (df[t] / 100.0).astype(float).values  # percent → decimal
    is_nan = np.isnan(psi_t)

    cls = np.where(psi_t > upper,  1,
        np.where(psi_t < lower, 0, -1)).astype(np.int8)

    # Convert to pandas nullable Int8, then mark NaNs as missing (pd.NA)
    cls = pd.Series(cls, dtype="Int8")
    cls[is_nan] = pd.NA

    class_wide[t] = cls  #

# --- Save wide table ---
# --- Reorder (defensive; already the same) and save ---
# Guarantee metadata is untouched and column order identical
class_wide[meta_cols] = df[meta_cols]           # restore metadata verbatim
class_wide = class_wide.reindex(columns=df.columns)  # keep exact original order
class_wide.to_csv(OUT_WIDE, index=False)

# --- Build long table (exon × tissue × class) ---
class_long = class_wide.melt(
    id_vars=["exon_id"],
    value_vars=tissue_cols,
    var_name="tissue",
    value_name="class3"
)
# class_long.to_csv(OUT_LONG, index=False)

# --- Quick per-tissue class counts ---
counts = (
    class_long
    .groupby(["tissue", "class3"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={-1: "count_neg1", 0: "count_0", 1: "count_pos1"})
    .reset_index()
)
counts.to_csv(OUT_SUMM, index=False)

# --- Summary ---
total = len(class_long)
print("\n=== Summary ===")
print(f"Saved WIDE → {OUT_WIDE} (shape {class_wide.shape})")
# print(f"Saved LONG → {OUT_LONG} (rows {len(class_long)})")
print(f"Saved counts→ {OUT_SUMM}")
print(f"Class distribution overall: {class_long['class3'].value_counts(dropna=False).to_dict()}")
