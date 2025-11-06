import pandas as pd
import numpy as np
import os
from pathlib import Path

# ---------------- root discovery ----------------
def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    try:
        start = Path(__file__)
    except NameError:
        start = Path.cwd()  # Fallback for interactive environments
        
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path
print(f"CONTRASTIVE_ROOT set to: {root_path}")

# === Config ===
division = 'test'
STD_DEV_MULTIPLIER = 0.5 # Bins will be mean ± 0.5σ

# === I/O ===
# Input file with 1, 0, -1 labels
path = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_cassette_exons_with_binary_labels_ExonBinPsi.csv" 
# Output file for this analysis
OUT_CSV_PATH = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_upDown_imbalance_std_dev_bins.csv"

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

# ==============================================================================
# ======= Per-Tissue "Up" (1) vs. "Down" (0) Imbalance Analysis (Pass 1) =======
# ==============================================================================

print(f"\n🔬 Analyzing 'Up' (1) vs. 'Down' (0) imbalance for {len(tissue_cols)} tissues...")

analysis_results = []

for tissue in tissue_cols:
    series = pd.to_numeric(df[tissue], errors="coerce")
    count_1 = (series == 1).sum()
    count_0 = (series == 0).sum()
    total_change_events = count_1 + count_0
    
    percent_up = np.nan
    if total_change_events > 0:
        percent_up = count_1 / total_change_events
    
    analysis_results.append({
        "tissue": tissue,
        "count_1 (Up)": count_1,
        "count_0 (Down)": count_0,
        "total_change_events": total_change_events,
        "percent_up": np.round(percent_up, 4)
    })

# Convert results to a DataFrame
imbalance_df = pd.DataFrame(analysis_results)

# ==============================================================================
# ======= Statistical Binning (Pass 2) =======
# ==============================================================================

# --- Calculate Mean and Std. Dev. of the distribution ---
valid_percent_up = imbalance_df['percent_up'].dropna()
mean = np.mean(valid_percent_up)
std_dev = np.std(valid_percent_up)

lower_thresh = mean - (STD_DEV_MULTIPLIER * std_dev)
upper_thresh = mean + (STD_DEV_MULTIPLIER * std_dev)

print(f"\n--- Statistical Binning Info ---")
print(f"Distribution Mean (μ): {mean:.4f}")
print(f"Distribution Std. Dev. (σ): {std_dev:.4f}")
print(f"Using {STD_DEV_MULTIPLIER}σ multiplier:")
print(f"  Bin 1 (Low): < {lower_thresh:.4f}")
print(f"  Bin 2 (Mid): {lower_thresh:.4f} - {upper_thresh:.4f}")
print(f"  Bin 3 (High): > {upper_thresh:.4f}")


# --- Define Binning Logic ---
def categorize_imbalance(percent_up):
    if pd.isna(percent_up):
        return "N/A"
    
    if percent_up < lower_thresh:
        return "1. Low (μ - 0.5σ)"
    elif percent_up <= upper_thresh:
        return "2. Medium (μ ± 0.5σ)"
    else: # percent_up > upper_thresh
        return "3. High (μ + 0.5σ)"

# --- Apply the new categories ---
imbalance_df['category'] = imbalance_df['percent_up'].apply(categorize_imbalance)

# Sort by percent_up for clean viewing
imbalance_df = imbalance_df.sort_values("percent_up")

# --- Save the DataFrame to a CSV file ---
imbalance_df.to_csv(OUT_CSV_PATH, index=False)
print(f"\n✅ Successfully saved imbalance data to: {OUT_CSV_PATH}")

print("\n✅ Imbalance Analysis Complete. Per-tissue results:")
print(imbalance_df.to_string()) # .to_string() to print all rows

# ======= Print the final summary of categories =======
print("\n📊 Summary of Tissue Categories:")
print(imbalance_df['category'].value_counts())