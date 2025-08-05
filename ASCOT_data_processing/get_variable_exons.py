import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv("/home/argha/Contrastive_Learning/data/ASCOT/gtex_psi.csv")

# Extract tissue PSI columns (from column L onward)
psi_df = df.iloc[:, 12:]  # Assuming first 11 columns are metadata

# Step 1: Filter exons on chr2, chr3, chr5
chr_mask = df['exon_location'].str.startswith(('chr2:', 'chr3:', 'chr5:'))

# Step 2: Replace -1 with NaN for proper numerical operations
psi_numeric = psi_df.replace(-1, np.nan)
# psi_numeric = psi_df


# Step 3: Compute tissue-average per exon
psi_mean = psi_numeric.mean(axis=1)

# Step 4: Find rows where at least one tissue has deviation ≥ 0.2
deviation_mask = (psi_numeric.sub(psi_mean, axis=0).abs() > 0.2*100).any(axis=1)

# Step 5: Filter exons expressed in at least 10 tissues (i.e., ≥10 non-NaN values)
tissue_expr_mask = psi_numeric.notna().sum(axis=1) >= 10


# ✅ Step 8: Cassette exons only
cassette_mask = df['cassette_exon'] == 'Yes'

# Final combined mask
final_mask = chr_mask & deviation_mask & tissue_expr_mask & cassette_mask

# Apply mask to full dataframe
filtered_df = df[final_mask]

# # Save the result
# filtered_df.to_excel("filtered_variable_exons.xlsx", index=False)

filtered_df.to_csv("/home/argha/Contrastive_Learning/data/ASCOT/filtered_variable_exons.csv", index=False)