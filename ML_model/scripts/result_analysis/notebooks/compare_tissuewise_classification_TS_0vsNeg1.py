#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare cell-type–specific classification metrics between our model and SOTA baseline.
Computes Δ(AUROC) and Δ(AUPRC) (Our − SOTA), assigns each of 112 cell types to a curated
biological lineage, summarizes results by lineage, and generates boxplots and ranked barplots.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === 1️⃣ File paths ===
root_path = Path("/gpfs/commons/home/atalukder/Contrastive_Learning/files/RECOMB_26/classification/")
our_file  = root_path / "TS_CLSwpd_300bp_10Aug_tissue_metrics_triclass_20251106-213156.csv"
sota_file = root_path / "TS_noCL_300bp_rerun_codeChange_tissue_metrics_triclass_20251106-213156.csv"

# === 2️⃣ Load data ===
our_df  = pd.read_csv(our_file)
sota_df = pd.read_csv(sota_file)

# Normalize cell type names
def clean_name(x):
    return str(x).strip().lower().replace('"', "").replace("'", "")

our_df["tissue"] = our_df["tissue"].apply(clean_name)
sota_df["tissue"] = sota_df["tissue"].apply(clean_name)

# Merge both datasets
merged = our_df.merge(sota_df, on="tissue", suffixes=("_our", "_sota"))
print(f"✅ Merged {len(merged)} shared cell types between our model and SOTA baseline.\n")

# === 3️⃣ Compute Δ metrics (Our − SOTA) ===
metrics = [
    "accuracy", "macro_f1", "weighted_f1",
    "auroc_1_vs_-1", "auprc_1_vs_-1",
    "auroc_0_vs_-1", "auprc_0_vs_-1"
]
for m in metrics:
    merged[f"delta_{m}"] = merged[f"{m}_our"] - merged[f"{m}_sota"]

# === 4️⃣ Explicit mapping of 112 cell types ===
celltype_to_category = {
    "acinar cell of salivary gland": "Secretory/Glandular",
    "activated cd4-positive, alpha-beta t cell": "Immune/Hematopoietic",
    "adventitial fibroblast": "Fibroblast/Stromal",
    "airway smooth muscle cell": "Muscle/Cardiac",
    "alveolar fibroblast": "Fibroblast/Stromal",
    "arterial endothelial cell": "Endothelial/Vascular",
    "atrial cardiac muscle cell": "Muscle/Cardiac",
    "b cell": "Immune/Hematopoietic",
    "basal bladder urothelial cell": "Epithelial",
    "basal cell": "Epithelial",
    "basal epithelial cell": "Epithelial",
    "bladder urothelial cell": "Epithelial",
    "capillary aerocyte": "Endothelial/Vascular",
    "capillary endothelial cell": "Endothelial/Vascular",
    "cardiac endothelial cell": "Endothelial/Vascular",
    "cd24 neutrophil": "Immune/Hematopoietic",
    "cd34+ fibroblasts": "Fibroblast/Stromal",
    "cd4-positive helper t cell": "Immune/Hematopoietic",
    "cd4-positive, alpha-beta memory t cell": "Immune/Hematopoietic",
    "cd4-positive, alpha-beta t cell": "Immune/Hematopoietic",
    "cd4-positive, alpha-beta thymocyte": "Immune/Hematopoietic",
    "cd8-positive, alpha-beta memory t cell": "Immune/Hematopoietic",
    "cd8-positive, alpha-beta t cell": "Immune/Hematopoietic",
    "cd8-positive, alpha-beta thymocyte": "Immune/Hematopoietic",
    "cdc2": "Stem/Progenitor",
    "ciliated columnar cell of tracheobronchial tree": "Epithelial",
    "classical monocyte": "Immune/Hematopoietic",
    "colon endothelial cell": "Endothelial/Vascular",
    "common myeloid progenitor": "Immune/Hematopoietic",
    "connective tissue cell": "Fibroblast/Stromal",
    "dendritic cell": "Immune/Hematopoietic",
    "duct epithelial cell": "Epithelial",
    "endometrial stromal fibroblast": "Fibroblast/Stromal",
    "endothelial cell": "Endothelial/Vascular",
    "endothelial cell of lymphatic vessel": "Endothelial/Vascular",
    "endothelial cell of vascular tree": "Endothelial/Vascular",
    "enterocyte of epithelium of large intestine": "Epithelial",
    "enterocyte of epithelium proper of small intestine": "Epithelial",
    "enteroglial cell": "Neural/Retinal",
    "epithelial cell": "Epithelial",
    "epithelial cell of uterus": "Epithelial",
    "erythrocyte": "Immune/Hematopoietic",
    "erythroid progenitor cell": "Immune/Hematopoietic",
    "eye photoreceptor cell": "Neural/Retinal",
    "fast muscle cell": "Muscle/Cardiac",
    "fibroblast": "Fibroblast/Stromal",
    "fibroblast of breast": "Fibroblast/Stromal",
    "fibroblast of cardiac tissue": "Fibroblast/Stromal",
    "granulocyte": "Immune/Hematopoietic",
    "hematopoietic stem cell": "Immune/Hematopoietic",
    "hr positive luminal epithelial cell of mammary gland": "Epithelial",
    "innate lymphoid cell": "Immune/Hematopoietic",
    "intermediate bladder urothelial cell": "Epithelial",
    "intestinal crypt stem cell of small intestine": "Stem/Progenitor",
    "intrahepatic cholangiocyte": "Epithelial",
    "kidney epithelial cell": "Epithelial",
    "lacrimal gland functional unit cell": "Secretory/Glandular",
    "leukocyte": "Immune/Hematopoietic",
    "ltf+ epithelial cell": "Epithelial",
    "lung ciliated cell": "Epithelial",
    "macrophage": "Immune/Hematopoietic",
    "mast cell": "Immune/Hematopoietic",
    "mature enterocyte": "Epithelial",
    "mature nk t cell": "Immune/Hematopoietic",
    "memory b cell": "Immune/Hematopoietic",
    "mesenchymal stem cell": "Fibroblast/Stromal",
    "mesenchymal stem cell of adipose tissue": "Adipose",
    "monocyte": "Immune/Hematopoietic",
    "muscle cell": "Muscle/Cardiac",
    "myeloid cell": "Immune/Hematopoietic",
    "myeloid dendritic cell": "Immune/Hematopoietic",
    "myeloid progenitor": "Immune/Hematopoietic",
    "myofibroblast cell": "Fibroblast/Stromal",
    "myofibroblast cell and pericyte": "Fibroblast/Stromal",
    "naive b cell": "Immune/Hematopoietic",
    "naive cd8-positive t cell": "Immune/Hematopoietic",
    "naive thymus-derived cd4-positive, alpha-beta t cell": "Immune/Hematopoietic",
    "nampt neutrophil": "Immune/Hematopoietic",
    "natural killer cell": "Immune/Hematopoietic",
    "neutrophil": "Immune/Hematopoietic",
    "nk cell": "Immune/Hematopoietic",
    "nk t cell": "Immune/Hematopoietic",
    "non-classical monocyte": "Immune/Hematopoietic",
    "paneth cell of colon": "Secretory/Glandular",
    "paneth cell of epithelium of small intestine": "Secretory/Glandular",
    "pericyte": "Fibroblast/Stromal",
    "plasma cell": "Immune/Hematopoietic",
    "platelet": "Immune/Hematopoietic",
    "regulatory t cell": "Immune/Hematopoietic",
    "respiratory goblet cell": "Epithelial",
    "retina - microglia": "Neural/Retinal",
    "retinal blood vessel endothelial cell": "Endothelial/Vascular",
    "salivary gland cell": "Secretory/Glandular",
    "sebum secreting cell": "Secretory/Glandular",
    "secretory luminal epithelial cell of mammary gland": "Epithelial",
    "skeletal muscle satellite stem cell": "Muscle/Cardiac",
    "slow muscle cell": "Muscle/Cardiac",
    "small intestine goblet cell": "Epithelial",
    "smooth muscle cell": "Muscle/Cardiac",
    "stratified squamous epithelial cell": "Epithelial",
    "stromal cell": "Fibroblast/Stromal",
    "t cell": "Immune/Hematopoietic",
    "tendon cell": "Fibroblast/Stromal",
    "thymocyte": "Immune/Hematopoietic"
}

merged["Category"] = merged["tissue"].map(celltype_to_category).fillna("Other")

# === 5️⃣ Category summary ===
summary = (
    merged.groupby("Category")[["delta_auroc_0_vs_-1", "delta_auprc_0_vs_-1"]]
    .mean()
    .sort_values("delta_auprc_0_vs_-1", ascending=False)
)

counts = merged["Category"].value_counts().reindex(summary.index)
print("📊 Mean Δ(AUROC, AUPRC) by lineage (with counts):\n")
for cat in summary.index:
    auroc, auprc = summary.loc[cat]
    print(f"{cat:<25} n={counts[cat]:3d}   ΔAUROC={auroc:6.3f}   ΔAUPRC={auprc:6.3f}")

# === 6️⃣ Top 20 cell types by ΔAUPRC ===
top_ct = (
    merged.sort_values("delta_auprc_0_vs_-1", ascending=False)
    [["tissue", "Category", "delta_auroc_0_vs_-1", "delta_auprc_0_vs_-1"]]
    .head(20)
)
print("\n🔥 Top 20 cell types with strongest AUPRC (1 vs -1) improvement:\n")
print(top_ct.round(3).to_string(index=False))

# === 7️⃣ Boxplot per lineage ===
plt.figure(figsize=(10,6))
sns.boxplot(
    data=merged,
    x="Category",
    y="delta_auprc_0_vs_-1",
    order=summary.index,
    color="lightblue",
    fliersize=3
)
plt.axhline(0, color="gray", linestyle="--", lw=1)
plt.title("ΔAUPRC(1 vs -1) per Cell Lineage (Our − SOTA)", fontsize=13)
plt.ylabel("Improvement in AUPRC (Our − SOTA)")
plt.xlabel("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# === 8️⃣ Barplot: Top 20 improved cell types ===
plt.figure(figsize=(8,8))
top20 = merged.sort_values("delta_auprc_0_vs_-1", ascending=False).head(20)
sns.barplot(
    data=top20,
    y="tissue",
    x="delta_auprc_0_vs_-1",
    hue="Category",
    dodge=False,
    palette="tab10"
)
plt.axvline(0, color="gray", linestyle="--", lw=1)
plt.title("Top 20 Cell Types by ΔAUPRC(1 vs -1)", fontsize=13)
plt.xlabel("Improvement in AUPRC (Our − SOTA)")
plt.ylabel("")
plt.tight_layout()
plt.show()


# === 5️⃣ Collapse lineages into system-level supercategories ===
lineage_to_system = {
    # Each lineage from the 112-type mapping → higher-order group
    "Secretory/Glandular": "Endocrine",
    "Epithelial": "Digestive/Epithelial",
    "Neural/Retinal": "Brain",
    "Muscle/Cardiac": "Heart/Vascular",
    "Endothelial/Vascular": "Heart/Vascular",
    "Fibroblast/Stromal": "Other",        # connective & stromal tissues
    "Immune/Hematopoietic": "Immune/Cells",
    "Adipose": "Adipose",
    "Stem/Progenitor": "Reproductive",    # can also be "Other" if you prefer
    "Other": "Other"
}

merged["System"] = merged["Category"].map(lineage_to_system).fillna("Other")

# === 6️⃣ Compute mean improvements at system level ===
system_summary = (
    merged.groupby("System")[["delta_auroc_0_vs_-1", "delta_auprc_0_vs_-1"]]
    .mean()
    .sort_values("delta_auprc_0_vs_-1", ascending=False)
)
print("\n📊 Mean Δ(AUROC, AUPRC) by *System-level* category:\n")
print(system_summary.round(3).to_string())

# Optional: count cell types per group
system_counts = merged["System"].value_counts().reindex(system_summary.index)
print("\n🧩 Cell type counts per system:\n")
print(system_counts)