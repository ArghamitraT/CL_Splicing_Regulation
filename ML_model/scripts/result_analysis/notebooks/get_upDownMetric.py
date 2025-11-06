import pandas as pd
import numpy as np
import os
import os
from pathlib import Path
import time

timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"Script started at: {timestamp}")

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

# --- 1. SET YOUR FILE PATHS HERE ---

# Path to the imbalance CSV you created
imbalance_file = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_upDown_imbalance.csv"

# Path to your model's per-tissue metrics

# DO NOT Erase
# NOTE: 'CLSwpd' is your model, 'noCL' is SOTA/baseline.
model_metrics_file = f"{root_path}/files/classification_eval/TS_CLSwpd_300bp_10Aug_tissue_metrics_logitdelta_binary_20251105-180839.csv"
# Path to the SOTA/baseline model's per-tissue metrics
sota_metrics_file = f"{root_path}/files/classification_eval/TS_noCL_300bp_rerun_codeChange_tissue_metrics_logitdelta_binary_20251105-180839.csv"

low_lim = 0.15
high_lim = 0.25

# --- 2. LOAD ALL DATA FILES ---

try:
    # Load the imbalance data
    imbalance_df = pd.read_csv(imbalance_file)
    print(f"✅ Loaded imbalance data from: {imbalance_file}")

    # Load the per-tissue metrics for both models
    model_metrics_df = pd.read_csv(model_metrics_file)
    sota_metrics_df = pd.read_csv(sota_metrics_file)
    
    # Get model names for labels
    your_model_name = model_metrics_df['model'].iloc[0]
    sota_model_name = sota_metrics_df['model'].iloc[0]
    
    print(f"✅ Loaded metrics for '{your_model_name}' (Your Model)")
    print(f"✅ Loaded metrics for '{sota_model_name}' (SOTA/Baseline)")

    # Combine metrics into one DataFrame
    all_metrics_df = pd.concat([model_metrics_df, sota_metrics_df], ignore_index=True)

except FileNotFoundError as e:
    print(f"--- ERROR ---")
    print(f"File not found: {e.filename}")
    print("Please update the file paths at the top of the script and try again.")
    exit()


# --- 3. DEFINE NEW, TIGHTER BINS ---

def categorize_imbalance(percent_up):
    """
    Applies tighter bins to the 'percent_up' metric.
    """
    if pd.isna(percent_up):
        return "N/A"
    
    if percent_up < low_lim:
        return f"1. Slightly Down-Heavy (< {low_lim})"
    elif low_lim <= percent_up <= high_lim:
        return f"2. Highly Balanced ({low_lim} - {high_lim})"
    else: # percent_up > high_lim
        return f"3. Slightly Up-Heavy (> {high_lim})"
# Apply the new categories
imbalance_df['granular_category'] = imbalance_df['percent_up'].apply(categorize_imbalance)


# --- 4. MERGE, AGGREGATE, AND CREATE PIVOT TABLE ---

# Merge the metrics with the imbalance categories
merged_df = all_metrics_df.merge(imbalance_df, on="tissue")

# Group by the new category and the model, then calculate the mean
summary_df = merged_df.groupby(['granular_category', 'model']).agg(
    f1_mean=('f1', 'mean'),
    auprc_mean=('auprc', 'mean'),
    auroc_mean=('auroc', 'mean'),
    precision_mean=('precision', 'mean'),
    n_tissues=('tissue', 'size')
).reset_index()

# Pivot the table for easy comparison
pivot_table = summary_df.pivot_table(
    index='granular_category',
    columns='model',
    values=['f1_mean', 'auprc_mean', 'auroc_mean', 'precision_mean']
)

# Also show the tissue counts
tissue_counts = summary_df.pivot_table(
    index='granular_category',
    columns='model',
    values='n_tissues'
)

# --- 5. BUILD THE FORMATTED TEXT REPORT ---

# Helper function to find the winner
def get_winner(val1, val2):
    if np.isclose(val1, val2, atol=0.001):
        return "Tie"
    return "Your Model" if val1 > val2 else "SOTA (noCL)"

# Initialize an empty string for the report
report_string = ""
report_string += "="*80 + "\n"
report_string += "          Stratified Analysis Summary ('Up' vs. 'Down' Task)          \n"
report_string += "="*80 + "\n\n"

report_string += f"Your Model: {your_model_name}\n"
report_string += f"SOTA (noCL): {sota_model_name}\n\n"

report_string += "--- Tissue Counts Per Bin ---\n"
report_string += tissue_counts.to_string() + "\n\n"

# List of metrics to report
metrics = [
    ('f1_mean', 'F1-Score'),
    ('precision_mean', 'Precision'),
    ('auprc_mean', 'AUPRC'),
    ('auroc_mean', 'AUROC')
]

for metric_col, metric_name in metrics:
    report_string += "\n" + "-"*40 + "\n"
    report_string += f"📊 Metric: {metric_name} (Mean)\n"
    report_string += "-"*40 + "\n"
    
    # Create a formatted table
    report_string += f"| {'Category':<30} | {'Your Model':<12} | {'SOTA (noCL)':<12} | {'Winner':<12} |\n"
    report_string += "|" + "-"*32 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|\n"
    
    for category in pivot_table.index:
        your_val = pivot_table.loc[category, (metric_col, your_model_name)]
        sota_val = pivot_table.loc[category, (metric_col, sota_model_name)]
        winner = get_winner(your_val, sota_val)
        
        report_string += f"| {category:<30} | {your_val:<12.4f} | {sota_val:<12.4f} | {winner:<12} |\n"

report_string += "\n" + "="*80 + "\n"


# --- 6. SAVE THE REPORT TO A TEXT FILE ---
output_txt_path = f"{root_path}/files/classification_eval/nov5_files/stratified_analysis_summary_{timestamp}.txt"

try:
    with open(output_txt_path, 'w') as f:
        f.write(report_string)
    print(f"\n✅ Successfully saved text report to: {output_txt_path}")
except Exception as e:
    print(f"\n--- ERROR writing file ---")
    print(e)

# Also print the report to the console
print(report_string)