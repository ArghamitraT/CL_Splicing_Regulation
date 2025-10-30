import pandas as pd
import pickle
import os
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

# --- Input files ---
GT_FILE = f"{root_path}/data/final_data/ASCOT_finetuning/variable_cassette_exons_with_logit_mean_psi.csv"
PRED_FILE = f"{root_path}/files/results/exprmnt_2025_10_28__20_12_58/weights/checkpoints/run_6/tsplice_final_predictions_all_tissues.tsv"
OUT_FILE = f"{root_path}/data/extra/exons_toshow_contrast.pkl"

# --- 1. Load ground truth and compute mean PSI ---
gt_df = pd.read_csv(GT_FILE)
# Filter only PSI columns (all columns after gene info)
psi_cols = gt_df.columns[gt_df.columns.str.contains(r'^\d|\.\d|^\w+-?\w*$', regex=True, case=False)]

gt_df = gt_df[["exon_id", "mean_psi"]]
gt_df["mean_psi"] = gt_df["mean_psi"]/100

# --- 2. Categorize exons ---
high_expr = gt_df.query("mean_psi >= 0.9 and mean_psi <= 1")["exon_id"].tolist()
low_expr = gt_df.query("mean_psi <= 0.05 and mean_psi >= 0")["exon_id"].tolist()

# --- 3. Load predictions ---
pred_df = pd.read_csv(PRED_FILE, sep="\t")
pred_df.columns = pred_df.columns.str.strip()  # Remove stray spaces in column names
tissue_cols = [c for c in pred_df.columns if c != "exon_id"]

# --- 4. Convert prediction values to PSI scale (divide by 100) ---
pred_df[tissue_cols] = pred_df[tissue_cols] / 100.0

# --- 5. Merge ground truth PSI for same tissues (if available) ---
merged_df = pred_df.merge(gt_df, on="exon_id", how="left")

# --- 6. Initialize results dictionary ---
results = {tissue: {"high": [], "low": []} for tissue in tissue_cols}

# --- 7. For each tissue, compare prediction vs mean PSI ---
for tissue in tissue_cols:
    for _, row in merged_df.iterrows():
        exon_id = row["exon_id"]
        pred_psi = row[tissue]
        gt_psi = row["mean_psi"]

        if pd.isna(gt_psi) or pd.isna(pred_psi):
            continue

        diff = abs(pred_psi - gt_psi)
        if diff < 0.05:
            if exon_id in high_expr:
                results[tissue]["high"].append(exon_id)
            elif exon_id in low_expr:
                results[tissue]["low"].append(exon_id)
    
    print("=== Accurate Exon Counts per Tissue ===")
    n_high = len(results[tissue]["high"])
    n_low = len(results[tissue]["low"])
    print(f"{tissue:<35} | High: {n_high:5d} | Low: {n_low:5d}")


# --- 8. Save as pickle ---
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"✅ Saved categorized results for {len(tissue_cols)} tissues → {OUT_FILE}")

from functools import reduce
# --- 9. Find common high/low exons across all tissues ---
all_high_sets = [set(results[t]["high"]) for t in tissue_cols if len(results[t]["high"]) > 0]
all_low_sets  = [set(results[t]["low"])  for t in tissue_cols if len(results[t]["low"])  > 0]

common_high = list(reduce(lambda a, b: a & b, all_high_sets)) if all_high_sets else []
common_low  = list(reduce(lambda a, b: a & b, all_low_sets))  if all_low_sets else []

common_dict = {
    "common_high_accuracy_high": common_high,
    "common_high_accuracy_low": common_low
}

# --- 10. Save common lists ---
COMMON_OUT_PKL = f"{root_path}/data/extra/common_high_low_exons.pkl"
COMMON_OUT_CSV = f"{root_path}/data/extra/common_high_low_exons.csv"

# with open(COMMON_OUT_PKL, "wb") as f:
#     pickle.dump(common_dict, f)

# --- 11. Save also as CSV ---
max_len = max(len(common_high), len(common_low))
df_common = pd.DataFrame({
    "common_high_accuracy_high": common_high + [""] * (max_len - len(common_high)),
    "common_high_accuracy_low": common_low + [""] * (max_len - len(common_low))
})
df_common.to_csv(COMMON_OUT_CSV, index=False)

# --- 12. Print summary ---
print("\n=== Common High/Low Accurate Exons Across All Tissues ===")
print(f"Common high-expressed accurately predicted exons: {len(common_high)}")
print(f"Common low-expressed accurately predicted exons:  {len(common_low)}")
print(f"✅ Saved → {COMMON_OUT_PKL}")
print(f"✅ Saved → {COMMON_OUT_CSV}")