import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from scipy.cluster.hierarchy import linkage, leaves_list

ts_df = pd.read_csv("/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/full_cassette_exons_with_mean_psi.csv")
cell_types = ts_df.columns[12:-3].tolist()

psi_vals = ts_df[cell_types].apply(pd.to_numeric, errors="coerce")

# Calculate difference from mean
deviation = psi_vals.sub(ts_df["mean_psi"], axis=0).abs()
# Filter for deviations above 10 percentage points
deviation["variable_count"] = (deviation >= 10).sum(axis=1)

print(f"At least one cell type >=10% from mean:")
print((deviation["variable_count"] >= 1).sum())

print(f"At least 10 cell types >=10% from mean:")
print((deviation["variable_count"] >= 10).sum())

mask = deviation["variable_count"] >= 10
variable = ts_df[mask]
var_psi = variable[cell_types]
corr = var_psi.corr(method="pearson")
print(f"Pearson correlation min and max")
print(corr.min().min(), corr.max().max())

# Calculate Ward linkage for cell types
cell_linkage = linkage(corr.values, method='ward')   # corr: your correlation matrix

# Calculate heat map of tissue similarity/clustering
sns.clustermap(
    corr,
    cmap="RdBu_r",
    vmin=0,
    vmax=1,
    figsize=(20, 20),
    linewidths=0.5,
    row_cluster=True,
    col_cluster=True,
    row_linkage = cell_linkage,
    col_linkage = cell_linkage,
    xticklabels=True,
    yticklabels=True
)
plt.title("Tissue–Tissue Correlation (Variable Exons)")
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/cell_type_corr.png", dpi=300)

# # Plot PSI heat map for filtered exons
# # subset_exons = var_psi.sample(n=1000, random_state=1)
# subset_exons = var_psi
# plot_df = subset_exons.T
# # Fill NaNs for clustering only
# plot_df_filled = plot_df.fillna(50)

# # Cluster columns
# col_linkage = linkage(plot_df_filled.T, method="ward")


# cmap = sns.color_palette("RdBu_r", as_cmap=True)
# cmap.set_bad(color='lightgray')

# sns.clustermap(
#     plot_df,
#     row_linkage=cell_linkage,
#     col_linkage=col_linkage,
#     cmap=cmap,
#     vmin=0, vmax=100,
#     figsize=(20, 20),
#     xticklabels=False,
#     yticklabels=True
# )
# plt.xlabel("Exons")
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/psi_heat_map.png", dpi=300)

