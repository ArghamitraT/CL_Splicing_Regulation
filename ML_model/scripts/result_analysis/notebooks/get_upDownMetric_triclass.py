import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

# ================================================================
# Tri-Class Stratified Analysis Summary Script
# ================================================================

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
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

# ================================================================
# Config
# ================================================================
division = 'test'
low_lim = 0.15
high_lim = 0.25

# --- 1. SET YOUR FILE PATHS HERE ---
imbalance_file = f"{root_path}/data/TS_data/tabula_sapiens/final_data/{division}_upDown_imbalance.csv"

# Tri-class metric CSVs (Your Model vs SOTA)
# model_metrics_file = f"{root_path}/files/classification_eval/TS_CLSwpd_300bp_10Aug_tissue_metrics_triclass_20251106-131411.csv"
# sota_metrics_file = f"{root_path}/files/classification_eval/TS_noCL_300bp_rerun_codeChange_tissue_metrics_triclass_20251106-131411.csv"


model_metrics_file = f"{root_path}/files/classification_eval/TS_CLSwpd_300bp_10Aug_tissue_metrics_triclass_20251106-134418.csv"
sota_metrics_file = f"{root_path}/files/classification_eval/TS_noCL_300bp_rerun_codeChange_tissue_metrics_triclass_20251106-134418.csv"

# ================================================================
# 2. LOAD ALL DATA FILES
# ================================================================
try:
    imbalance_df = pd.read_csv(imbalance_file)
    print(f"✅ Loaded imbalance data from: {imbalance_file}")

    model_metrics_df = pd.read_csv(model_metrics_file)
    sota_metrics_df = pd.read_csv(sota_metrics_file)

    your_model_name = model_metrics_df['model'].iloc[0]
    sota_model_name = sota_metrics_df['model'].iloc[0]

    print(f"✅ Loaded tri-class metrics for '{your_model_name}' (Your Model)")
    print(f"✅ Loaded tri-class metrics for '{sota_model_name}' (SOTA/Baseline)")

    all_metrics_df = pd.concat([model_metrics_df, sota_metrics_df], ignore_index=True)

except FileNotFoundError as e:
    print(f"--- ERROR ---\nFile not found: {e.filename}")
    print("Please update the file paths at the top of the script and try again.")
    exit()

# ================================================================
# 3. DEFINE NEW, TIGHTER BINS
# ================================================================
def categorize_imbalance(percent_up):
    """Categorize tissues based on percent_up imbalance."""
    if pd.isna(percent_up):
        return "N/A"
    if percent_up < low_lim:
        return f"1. Slightly Down-Heavy (< {low_lim})"
    elif low_lim <= percent_up <= high_lim:
        return f"2. Highly Balanced ({low_lim} - {high_lim})"
    else:
        return f"3. Slightly Up-Heavy (> {high_lim})"

imbalance_df['granular_category'] = imbalance_df['percent_up'].apply(categorize_imbalance)

# ================================================================
# 4. MERGE, AGGREGATE, AND CREATE PIVOT TABLE
# ================================================================
merged_df = all_metrics_df.merge(imbalance_df, on="tissue")

# summary_df = merged_df.groupby(['granular_category', 'model']).agg({
#     'accuracy': ('accuracy', 'mean'),
#     'macro_f1': ('macro_f1', 'mean'),
#     'weighted_f1': ('weighted_f1', 'mean'),
#     'auroc_1_vs_-1': ('auroc_1_vs_-1', 'mean'),
#     'auprc_1_vs_-1': ('auprc_1_vs_-1', 'mean'),
#     'auroc_0_vs_-1': ('auroc_0_vs_-1', 'mean'),
#     'auprc_0_vs_-1': ('auprc_0_vs_-1', 'mean'),
#     'tissue': ('tissue', 'size')
# }).reset_index()

# # Rename the aggregated columns for clarity
# summary_df = summary_df.rename(columns={
#     'accuracy': 'accuracy_mean',
#     'macro_f1': 'macro_f1_mean',
#     'weighted_f1': 'weighted_f1_mean',
#     'auroc_1_vs_-1': 'auroc_1_vs_-1_mean',
#     'auprc_1_vs_-1': 'auprc_1_vs_-1_mean',
#     'auroc_0_vs_-1': 'auroc_0_vs_-1_mean',
#     'auprc_0_vs_-1': 'auprc_0_vs_-1_mean',
#     'tissue': 'n_tissues'
# })

summary_df = merged_df.groupby(['granular_category', 'model']).agg(**{
    'accuracy_mean': ('accuracy', 'mean'),
    'macro_f1_mean': ('macro_f1', 'mean'),
    'weighted_f1_mean': ('weighted_f1', 'mean'),
    'auroc_1_vs_-1_mean': ('auroc_1_vs_-1', 'mean'),
    'auprc_1_vs_-1_mean': ('auprc_1_vs_-1', 'mean'),
    'auroc_0_vs_-1_mean': ('auroc_0_vs_-1', 'mean'),
    'auprc_0_vs_-1_mean': ('auprc_0_vs_-1', 'mean'),
    'n_tissues': ('tissue', 'size')
}).reset_index()


pivot_table = summary_df.pivot_table(
    index='granular_category',
    columns='model',
    values=[
        'accuracy_mean', 'macro_f1_mean', 'weighted_f1_mean',
        'auroc_1_vs_-1_mean', 'auprc_1_vs_-1_mean',
        'auroc_0_vs_-1_mean', 'auprc_0_vs_-1_mean'
    ]
)

tissue_counts = summary_df.pivot_table(
    index='granular_category',
    columns='model',
    values='n_tissues'
)

# ================================================================
# 5. BUILD THE FORMATTED TEXT REPORT
# ================================================================
def get_winner(val1, val2):
    """Determine the metric winner."""
    if np.isnan(val1) or np.isnan(val2):
        return "N/A"
    if np.isclose(val1, val2, atol=0.001):
        return "Tie"
    return "Your Model" if val1 > val2 else "SOTA (noCL)"

report_string = ""
report_string += "="*80 + "\n"
report_string += "              Stratified Tri-Class Analysis Summary              \n"
report_string += "="*80 + "\n\n"
report_string += f"Your Model: {your_model_name}\n"
report_string += f"SOTA (noCL): {sota_model_name}\n\n"
report_string += "--- Tissue Counts Per Bin ---\n"
report_string += tissue_counts.to_string() + "\n\n"

# Metric display names
metrics = [
    ('accuracy_mean', 'Accuracy'),
    ('macro_f1_mean', 'Macro F1'),
    ('weighted_f1_mean', 'Weighted F1'),
    ('auroc_1_vs_-1_mean', 'AUROC (1 vs -1)'),
    ('auprc_1_vs_-1_mean', 'AUPRC (1 vs -1)'),
    ('auroc_0_vs_-1_mean', 'AUROC (0 vs -1)'),
    ('auprc_0_vs_-1_mean', 'AUPRC (0 vs -1)')
]

for metric_col, metric_name in metrics:
    report_string += "\n" + "-"*40 + "\n"
    report_string += f"📊 Metric: {metric_name} (Mean)\n"
    report_string += "-"*40 + "\n"
    report_string += f"| {'Category':<34} | {'Your Model':<12} | {'SOTA (noCL)':<12} | {'Winner':<12} |\n"
    report_string += "|" + "-"*36 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|\n"
    
    for category in pivot_table.index:
        your_val = pivot_table.loc[category, (metric_col, your_model_name)]
        sota_val = pivot_table.loc[category, (metric_col, sota_model_name)]
        winner = get_winner(your_val, sota_val)
        report_string += f"| {category:<34} | {your_val:<12.4f} | {sota_val:<12.4f} | {winner:<12} |\n"

# ================================================================
# 6. SAVE THE REPORT TO A TEXT FILE
# ================================================================
output_dir = f"{root_path}/files/classification_eval/nov5_files"
os.makedirs(output_dir, exist_ok=True)
output_txt_path = f"{output_dir}/stratified_analysis_summary_triclass_{timestamp}.txt"

try:
    with open(output_txt_path, 'w') as f:
        f.write(report_string)
    print(f"\n✅ Successfully saved text report to: {output_txt_path}")
except Exception as e:
    print(f"\n--- ERROR writing file ---\n{e}")

# Also print to console
print(report_string)
