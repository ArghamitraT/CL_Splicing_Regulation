import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1️⃣ Load your model & SOTA results ---
root_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/RECOMB_26/classification/"
our_df = pd.read_csv(f"{root_path}300bp_MTCLSwept_10Aug_noExonPad_tissue_metrics_triclass_20251107-141104.csv")
sota_df = pd.read_csv(f"{root_path}MTSplice_original_SOTA_tissue_metrics_triclass_20251107-141104.csv")

# --- 2️⃣ Merge by tissue name ---
merged = our_df.merge(
    sota_df,
    on="tissue",
    suffixes=("_our", "_sota")
)

# --- 3️⃣ Compute deltas (Our − SOTA) ---
for metric in ["accuracy", "macro_f1", "weighted_f1",
               "auroc_1_vs_-1", "auprc_1_vs_-1",
               "auroc_0_vs_-1", "auprc_0_vs_-1"]:
    merged[f"delta_{metric}"] = merged[f"{metric}_our"] - merged[f"{metric}_sota"]

# --- 4️⃣ Define tissue categories manually ---
category_map = {
    # Brain & CNS
    "Brain": ["Amygdala", "Anterior cingulate", "Caudate", "Cerebellar", "Cerebellum", "Cortex",
              "Frontal Cortex", "Hippocampus", "Hypothalamus", "Nucleus accumbens",
              "Putamen", "Spinal cord", "Substantia nigra"],
    # Heart / Vascular / Muscle
    "Heart/Vascular": ["Aorta", "Coronary", "Atrial Appendage", "Left Ventricle",
                       "Tibial - Artery", "Skeletal", "Tibial - Nerve"],
    # Digestive / Epithelial
    "Digestive/Epithelial": ["Stomach", "Colon", "Gastroesoph", "Mucosa", "Muscularis",
                             "Ileum", "Esophagus"],
    # Eye
    "Eye": ["Retina", "RPE", "Choroid", "Sclera"],
    # Reproductive
    "Reproductive": ["Ovary", "Uterus", "Vagina", "Testis", "Cervix", "Ectocervix", "Endocervix", "Fallopian"],
    # Immune/Blood/Cells
    "Immune/Cells": ["Whole Blood", "Leukemia", "EBV", "Fibroblast", "Lymphocytes"],
    # Metabolic/Endocrine
    "Endocrine": ["Liver", "Pancreas", "Thyroid", "Adrenal"],
    # Adipose
    "Adipose": ["Subcutaneous", "Visceral", "Adipose"],
    # Miscellaneous
    "Other": []
}

def assign_category(tissue):
    for cat, terms in category_map.items():
        if any(term.lower() in tissue.lower() for term in terms):
            return cat
    return "Other"

merged["Category"] = merged["tissue"].apply(assign_category)

# --- 5️⃣ Compute groupwise mean improvements ---
summary = (
    merged.groupby("Category")[["delta_auroc_1_vs_-1", "delta_auprc_1_vs_-1"]]
    .mean()
    .sort_values("delta_auprc_1_vs_-1", ascending=False)
)
print("\n📊 Mean Δ(AUROC, AUPRC) by tissue category:\n")
print(summary.round(3))

# --- 6️⃣ Top tissues with strongest gains ---
top_tissues = merged.sort_values("delta_auprc_1_vs_-1", ascending=False)[
    ["tissue", "Category", "delta_auroc_1_vs_-1", "delta_auprc_1_vs_-1"]
].head(15)

print("\n🔥 Top tissues with strongest AUPRC (1 vs -1) improvement:\n")
print(top_tissues.round(3).to_string(index=False))

# --- 7️⃣ Visualization ---
plt.figure(figsize=(10,6))
sns.boxplot(
    data=merged,
    x="Category",
    y="delta_auprc_1_vs_-1",
    order=summary.index,
    color="lightblue"
)
plt.axhline(0, color="gray", linestyle="--")
plt.title("ΔAUPRC(1 vs -1) per Tissue Category (Our - MTSplice)")
plt.ylabel("Improvement in AUPRC")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
