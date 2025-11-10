import pandas as pd
import numpy as np

# === Step 1: Load CSV ===
csv_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/test_cassette_exons_with_binary_labels_ExonBinPsi.csv"



import pandas as pd

# --- Load the file ---
path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/tissue_counts_detailed_ExonBinPsi.csv"
df = pd.read_csv(path)

# --- Clean column names (remove stray symbols/spaces) ---
df.columns = df.columns.str.strip()

# --- Compute metrics per SampleClass ---
summary = (
    df.groupby("SampleClass")
      .agg({
          "#1": "sum",
          "#0": "sum"
      })
      .assign(
          sum_1_0 = lambda x: x["#1"] + x["#0"],
          pos_pct = lambda x: 100 * x["#1"] / (x["#1"] + x["#0"])
      )
      .reset_index()
)

print(summary)
