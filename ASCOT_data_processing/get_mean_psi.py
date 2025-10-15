import pandas as pd
import numpy as np
from scipy.special import logit

# Load CSV
name = 'variable'  # Replace with your actual name
csv_path = f"/home/atalukder/Contrastive_Learning/data/ASCOT/{name}_cassette_exons.csv"
df = pd.read_csv(csv_path)
test_file = 1
# Identify tissue columns (after 'exon_boundary')

if test_file:
    boundary_idx = df.columns.get_loc('exon_boundary')
    chrom_idx = df.columns.get_loc('chromosome')
    tissue_cols = df.columns[boundary_idx+1:chrom_idx]

else:
    boundary_idx = df.columns.get_loc('exon_boundary')
    tissue_cols = df.columns[boundary_idx+1:]

# Convert -1.0 to NaN
df[tissue_cols] = df[tissue_cols].replace(-1.0, np.nan)

# Compute mean PSI (across tissues, per exon)
df['mean_psi'] = df[tissue_cols].mean(axis=1)

# Compute logit(mean_psi/100) for each exon
eps = 1e-6
df['logit_mean_psi'] = logit(np.clip(df['mean_psi'] / 100, eps, 1-eps))

df.to_csv(f"/home/atalukder/Contrastive_Learning/data/ASCOT/{name}_cassette_exons_with_logit_mean_psi.csv", index=False)
print(df[['exon_id', 'mean_psi', 'logit_mean_psi']].head())
