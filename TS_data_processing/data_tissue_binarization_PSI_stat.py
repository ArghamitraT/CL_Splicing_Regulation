import pandas as pd

# --- Path to your CSV ---
path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/test_cassette_exons_with_binary_labels_HIGH_TissueBinPsi.csv"

# --- Read the file ---
df = pd.read_csv(path)

# --- Detect tissue columns (numeric 0/1 ones) ---
# Assuming first few columns are metadata like exon_id, gene_id, exon_strand, etc.
# So we can detect numeric ones automatically
num_cols = df.select_dtypes(include=["number"]).columns

# --- Flatten all numeric values and compute percentage of 1’s ---
total_values = df[num_cols].size
num_ones = (df[num_cols] == 1).sum().sum()
percent_ones = 100 * num_ones / total_values

print(f"✅ Total numeric entries: {total_values:,}")
print(f"✅ Number of 1's: {num_ones:,}")
print(f"✅ Percentage of 1's: {percent_ones:.2f}%")
