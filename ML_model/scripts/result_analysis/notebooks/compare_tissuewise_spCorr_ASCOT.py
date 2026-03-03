import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1️⃣ Load CSVs ---
root_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/"
our = pd.read_csv(f"{root_dir}exprmnt_2025_11_01__12_32_21/ensemble_evaluation_from_valdiation/test_set_evaluation/tsplice_spearman_by_tissue.tsv", sep="\t")
sota = pd.read_csv(f"{root_dir}mtsplice_originalTFweight_results/intron_300bp_results/variable_all_tissues_spearman_correlations.tsv", sep="\t")

# --- 2️⃣ Standardize column names ---
our = our.rename(columns={
    "spearman_psi": "psi_our",
    "spearman_delta": "delta_our"
})
sota = sota.rename(columns={
    "spearman_rho_psi": "psi_sota",
    "spearman_rho_delta_psi": "delta_sota"
})

# --- 3️⃣ Merge ---
df = our.merge(sota, on="tissue", how="inner")

# --- 4️⃣ Compute differences ---
df["Δpsi"]   = df["psi_our"]   - df["psi_sota"]
df["Δdelta"] = df["delta_our"] - df["delta_sota"]

# --- 5️⃣ Define tissue categories ---
category_map = {
    "Brain": ["Amygdala", "Anterior cingulate", "Caudate", "Cerebellar",
              "Cerebellum", "Cortex", "Frontal", "Hippocampus", "Hypothalamus",
              "Nucleus", "Putamen", "Spinal", "Substantia"],
    "Heart/Vascular": ["Aorta", "Coronary", "Atrial", "Ventricle", "Artery", "Muscle", "Nerve"],
    "Digestive/Epithelial": ["Colon", "Stomach", "Esophagus", "Ileum", "Gastroesoph", "Mucosa", "Muscularis"],
    "Endocrine": ["Liver", "Pancreas", "Thyroid", "Adrenal"],
    "Reproductive": ["Testis", "Ovary", "Uterus", "Vagina", "Cervix", "Ectocervix", "Endocervix", "Fallopian"],
    "Immune/Cells": ["Blood", "Leukemia", "EBV", "Fibroblast", "Lymphocyte"],
    "Eye": ["Retina", "RPE", "Choroid", "Sclera"],
    "Adipose": ["Adipose", "Subcutaneous", "Visceral"],
    "Other": []
}

def assign_category(tissue):
    for cat, terms in category_map.items():
        if any(term.lower() in tissue.lower() for term in terms):
            return cat
    return "Other"

df["Category"] = df["tissue"].apply(assign_category)

# --- 6️⃣ Summary stats ---
summary = (
    df.groupby("Category")[["Δpsi", "Δdelta"]]
    .mean()
    .sort_values("Δdelta", ascending=False)
)
print("\n📊 Mean ΔSpearman by category:")
print(summary.round(3))

# --- 7️⃣ Top tissues by Δdelta (ΔPSI correlation improvement) ---
top = df.sort_values("Δdelta", ascending=False)[["tissue", "Category", "Δpsi", "Δdelta"]]
print("\n🔥 Top tissues with strongest ΔPSI correlation gains:")
print(top.head(15).round(3).to_string(index=False))

# --- 8️⃣ Visualization ---
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x="Category", y="Δdelta", order=summary.index, color="lightblue")
plt.axhline(0, color="gray", linestyle="--")
plt.title("ΔSpearman (ΔPSI) per Tissue Category (Our − MTSplice)")
plt.ylabel("ΔSpearman for ΔPSI")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
