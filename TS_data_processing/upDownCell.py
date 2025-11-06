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
# === I/O ===
path = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_cassette_exons_with_binary_labels_ExonBinPsi.csv"
OUT_CSV_PATH = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_upDown_imbalance.csv"

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
# ======= NEW: Per-Tissue "Up" (1) vs. "Down" (0) Imbalance Analysis =======
# ==============================================================================

print(f"\n🔬 Analyzing 'Up' (1) vs. 'Down' (0) imbalance for {len(tissue_cols)} tissues...")

analysis_results = []

for tissue in tissue_cols:
    # Extract the column as numeric, coercing errors
    series = pd.to_numeric(df[tissue], errors="coerce")

    # Count Class 1 (Up)
    count_1 = (series == 1).sum()
    
    # Count Class 0 (Down)
    count_0 = (series == 0).sum()
    
    # Total "Change" events (denominator for our metric)
    total_change_events = count_1 + count_0
    
    percent_up = np.nan
    category = "No 'Up' or 'Down' Events" # Default category
    
    if total_change_events > 0:
        # Calculate the metric: percent_up = count_1 / (count_1 + count_0)
        percent_up = count_1 / total_change_events
        
        # Categorize based on the thresholds
        if percent_up < 0.3:
            category = "Down-Heavy"
        elif 0.3 <= percent_up <= 0.7:
            category = "Balanced"
        else: # percent_up > 0.7
            category = "Up-Heavy"
    
    analysis_results.append({
        "tissue": tissue,
        "count_1 (Up)": count_1,
        "count_0 (Down)": count_0,
        "total_change_events": total_change_events,
        "percent_up": np.round(percent_up, 4), # Rounded for nice printing
        "category": category
    })

# Convert results to a DataFrame for easy viewing
imbalance_df = pd.DataFrame(analysis_results).sort_values("percent_up")

# --- CHANGED: Save the DataFrame to a CSV file ---
imbalance_df.to_csv(OUT_CSV_PATH, index=False)
print(f"\n✅ Successfully saved imbalance data to: {OUT_CSV_PATH}")
# --- End Change ---

print("\n✅ Imbalance Analysis Complete. Per-tissue results:")
print(imbalance_df.to_string()) # .to_string() to print all rows

# ======= Print the final summary of categories =======
print("\n📊 Summary of Tissue Categories:")
print(imbalance_df['category'].value_counts())