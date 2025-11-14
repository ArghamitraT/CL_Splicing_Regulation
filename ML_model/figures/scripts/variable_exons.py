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

# agg_df = deviation["variable_count"].value_counts().reset_index()
# agg_df.columns = ["variable_count", "num_exons"]
# agg_df = agg_df.sort_values("variable_count")

# p_varcount = (
#     ggplot(agg_df, aes(x="variable_count", y="num_exons"))
#     + geom_col(fill="#de8f05")
#     + labs(
#         x="Number of cell types ≥10% from mean",
#         y="Number of exons",
#         title="Distribution of Variable Exons by Cell Type Count"
#     )
#     + theme_bw()
#     + theme(
#         figure_size=(10,6),
#         axis_text_x=element_text(size=12),
#         axis_text_y=element_text(size=12),
#         axis_title_x=element_text(size=14, weight="bold"),
#         axis_title_y=element_text(size=14, weight="bold"),
#         plot_title=element_text(size=16, weight="bold", ha="center")
#     )
# )

# # Save
# p_varcount.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/variable_count_hist.svg")

mask = deviation["variable_count"] >= 10
variable = ts_df[mask]
var_psi = variable[cell_types]
corr = var_psi.corr(method="pearson")
print(f"Pearson correlation min and max")
print(corr.min().min(), corr.max().max())

# Calculate Ward linkage for cell types
cell_linkage = linkage(corr.values, method='ward')   # corr: your correlation matrix

# Calculate heat map of tissue similarity/clustering
g = sns.clustermap(
    corr,
    cmap="RdBu_r",
    vmin=0,
    vmax=1,
    figsize=(30, 30),
    linewidths=0.5,
    row_cluster=True,
    col_cluster=True,
    row_linkage = cell_linkage,
    col_linkage = cell_linkage,
    xticklabels=True,
    yticklabels=True
)
g.cax.set_title("Pearson Correlation", fontsize=18)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/cell_type_corr.svg")

# Plot PSI heat map for filtered exons
subset_exons = var_psi
plot_df = subset_exons.T
# Fill NaNs for clustering only
plot_df_filled = plot_df.fillna(50)

# Cluster columns
col_linkage = linkage(plot_df_filled.T, method="ward")


cmap = sns.color_palette("RdBu_r", as_cmap=True)
cmap.set_bad(color='lightgray')

g2 = sns.clustermap(
    plot_df,
    row_linkage=cell_linkage,
    col_linkage=col_linkage,
    cmap=cmap,
    vmin=0, vmax=100,
    figsize=(30, 30),
    xticklabels=False,
    yticklabels=True
)
g2.ax_heatmap.set_xlabel("Exons", fontsize=18)
g2.cax.set_title("PSI", fontsize=18)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/psi_heat_map.png")

