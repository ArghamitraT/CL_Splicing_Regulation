import pandas as pd
import numpy as np
import os


# ---------------- root discovery ----------------
from pathlib import Path
def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path
print(f"CONTRASTIVE_ROOT set to: {root_path}")


# === Input and Output ===
division = 'variable'
FILE = f"{root_path}/data/ASCOT/{division}_cassette_exons_with_logit_mean_psi.csv"   # path to your file
OUT_HIGH = f"{root_path}/data/ASCOT/{division}_cassette_exons_with_binary_labels_HIGH.csv"
OUT_LOW  = f"{root_path}/data/ASCOT/{division}_cassette_exons_with_binary_labels_LOW.csv"

# === Load data ===
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

# Ensure 'mean_psi' exists
if "mean_psi" not in df.columns:
    raise ValueError("No 'mean_psi' column found in file.")

# Convert mean_psi from percent to decimal
df["mean_psi"] = df["mean_psi"] / 100.0

# Identify tissue columns (everything not metadata)
meta_cols = {
    "exon_id","cassette_exon","alternative_splice_site_group","linked_exons",
    "mutually_exclusive_exons","exon_strand","exon_length","gene_type",
    "gene_id","gene_symbol","exon_location","exon_boundary",
    "chromosome","chr","mean_psi","logit_mean_psi","chromosome.1"
}
tissue_cols = [c for c in df.columns if c not in meta_cols]

print(f"Detected {len(tissue_cols)} tissue columns → {tissue_cols[:5]}...")

# --- Divide into high vs low ---
high_df = df[df["mean_psi"] > 0.5].copy()
low_df  = df[df["mean_psi"] <= 0.5].copy()

# --- Binarize tissue PSI values ---
# High-expressed exons: tissue < 0.4 → 1 else 0
for t in tissue_cols:
    high_df[t] = (high_df[t] / 100.0 < 0.4).astype(int)

# Low-expressed exons: tissue > 0.6 → 1 else 0
for t in tissue_cols:
    low_df[t] = (low_df[t] / 100.0 > 0.6).astype(int)

# --- Save results separately ---
high_df.to_csv(OUT_HIGH, index=False)
low_df.to_csv(OUT_LOW, index=False)

print(f"✅ Saved high-expressed binary file → {OUT_HIGH} ({len(high_df)} rows)")
print(f"✅ Saved low-expressed binary file  → {OUT_LOW} ({len(low_df)} rows)")

# --- Optional sanity summary ---
print("\n=== Summary ===")
print(f"High group total 1s: {high_df[tissue_cols].sum().sum()}")
print(f"Low group total 1s:  {low_df[tissue_cols].sum().sum()}")