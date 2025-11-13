import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

# Map Broad Cell Types
with open("/gpfs/commons/home/nkeung/tabula_sapiens/broad_celltype_mapping.json", "r") as f:
    broad_type_mapping = json.load(f)

ts_df = pd.read_csv("/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/full_cassette_exons_with_mean_psi.csv")
cell_types = ts_df.columns[12:-3].tolist()

psi_vals = ts_df[cell_types].apply(pd.to_numeric, errors="coerce")

orig_len = len(psi_vals)
# Filter out exons with only one cell type
psi_vals = psi_vals.dropna(thresh=1, axis=0)
new_len = len(psi_vals)
print(f"Kept {new_len} exons out of {orig_len}")
print(psi_vals)

# !!! Filling missing values with PSI = 0. Change if needed!!!
# psi_vals = psi_vals.fillna(0)

# !!! Filling missing values with mean PSI for that exon
psi_vals = psi_vals.apply(lambda row: row.fillna(row.mean()), axis=1)

X = psi_vals.T
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

palette = sns.color_palette("hls", 44)

# ========== PCA ==========

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("PCA explained variance:")
print(pca.explained_variance_ratio_)

# Save results in dataframe with broad cell type
pca_df = pd.DataFrame({
    "cell_type": X.index,
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1]
})
pca_df["broad_cell_type"] = pca_df["cell_type"].map(broad_type_mapping)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="broad_cell_type",
    palette=palette,
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of PSI Values Across Cell Types")
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),  # negative y moves legend below plot
    fontsize=6,
    markerscale=0.5,
    ncol=4
)
plt.tight_layout()
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/pca_ts_DRAFT.png", dpi=300)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/pca_ts_FILTERED_DRAFT.png", dpi=300)


# ========== t-SNE ==========
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            n_iter=1000, random_state=1, init="pca")
X_tsne = tsne.fit_transform(X_scaled)

# Save results in dataframe with broad cell type
tsne_df = pd.DataFrame({
    "cell_type": X.index,
    "t-SNE1": X_tsne[:, 0],
    "t-SNE2": X_tsne[:, 1]
})
tsne_df["broad_cell_type"] = tsne_df["cell_type"].map(broad_type_mapping)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=tsne_df,
    x="t-SNE1",
    y="t-SNE2",
    hue="broad_cell_type",
    palette=palette,
)

plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.title("t-SNE of PSI Values Across Cell Types")
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    fontsize=6,
    markerscale=0.5,
    ncol=4
)
plt.tight_layout()
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/tsne_ts_DRAFT.png", dpi=300)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/tsne_ts_FILTERED_DRAFT.png", dpi=300)


# ========== UMAP ==========
reducer = umap.UMAP(
    n_neighbors=5,
    min_dist=0.3,
    n_components=2,
    random_state=1
)
X_umap = reducer.fit_transform(X_scaled)

umap_df = pd.DataFrame({
    "cell_type": X.index,
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1]
})

# Map broad cell types
umap_df["broad_cell_type"] = umap_df["cell_type"].map(broad_type_mapping)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=umap_df,
    x="UMAP1",
    y="UMAP2",
    hue="broad_cell_type",
    palette=palette,
)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of PSI Values Across Cell Types")
plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    fontsize=6,
    markerscale=0.5,
    ncol=4
)
plt.tight_layout()
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/umap_ts_DRAFT.png", dpi=300)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/umap_ts_FILTERED_DRAFT.png", dpi=300)

