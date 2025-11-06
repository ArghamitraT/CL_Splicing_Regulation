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
division = 'test'
# === I/O ===
path = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_cassette_exons_with_binary_labels_ExonBinPsi.csv"

# ======= Load file =======
df = pd.read_csv(path)

# ======= Identify tissue columns =======
meta_cols = [
    "exon_id","cassette_exon","alternative_splice_site_group","linked_exons",
    "mutually_exclusive_exons","exon_strand","exon_length","gene_type",
    "gene_id","gene_symbol","exon_location","exon_boundary",
    "chromosome","mean_psi","logit_mean_psi"
]

tissue_cols = [c for c in df.columns if c not in meta_cols]

# ======= Extract numeric tissue matrix =======
vals = df[tissue_cols].apply(pd.to_numeric, errors="coerce").values.flatten()

# Drop NaN
vals = vals[~np.isnan(vals)].astype(int)

# ======= Count & percentage =======
unique, counts = np.unique(vals, return_counts=True)
total = counts.sum()
percent = counts / total * 100

summary = pd.DataFrame({
    "Label": unique,
    "Count": counts,
    "Percentage": np.round(percent, 2)
}).sort_values("Label").reset_index(drop=True)

print(f"🔍 Label distribution across all tissues in {path}:")
print(summary)
