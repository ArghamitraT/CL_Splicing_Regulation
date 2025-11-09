import pandas as pd
import numpy as np

# === Step 1: Load CSV ===
csv_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/test_cassette_exons_with_binary_labels_ExonBinPsi.csv"
df = pd.read_csv(csv_path)

# === Step 2: Identify tissue columns ===
meta_cols = [
    "exon_id","cassette_exon","alternative_splice_site_group","linked_exons",
    "mutually_exclusive_exons","exon_strand","exon_length","gene_type",
    "gene_id","gene_symbol","exon_location","exon_boundary",
    "chromosome","chr","mean_psi","logit_mean_psi"
]
tissue_cols = [c for c in df.columns if c not in meta_cols]

print(f"🧬 Detected {len(tissue_cols)} tissue columns.")

# === Step 3: Count 1, 0, -1 per tissue ===
summary = []
for tissue in tissue_cols:
    vals = df[tissue].dropna()
    n_1 = (vals == 1).sum()
    n_0 = (vals == 0).sum()
    n_m1 = (vals == -1).sum()
    N = n_1 + n_0 + n_m1
    summary.append({
        "Tissue": tissue,
        "#1": n_1,
        "#0": n_0,
        "#-1": n_m1,
        "N_total": N
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values("N_total", ascending=True)

# === Step 4: Divide tissues into 3 classes (Low / Medium / High) ===
q1, q2 = summary_df["N_total"].quantile([1/3, 2/3])
bins = [summary_df["N_total"].min()-1, q1, q2, summary_df["N_total"].max()+1]
labels = ["Low sample", "Medium sample", "High sample"]
summary_df["SampleClass"] = pd.cut(summary_df["N_total"], bins=bins, labels=labels)

# === Step 5: Summarize each class ===
class_summary = (
    summary_df.groupby("SampleClass")["N_total"]
    .agg(["count", "min", "max", "mean", "std", "median"])
    .reset_index()
)

# === Step 6: Display results ===
print("\n📊 Tissue-wise counts:")
print(summary_df.head(10).to_string(index=False))

print("\n📈 Summary by sample-size class:")
print(class_summary.to_string(index=False))

# === Optional: save results ===
summary_df.to_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/tissue_counts_detailed_ExonBinPsi.csv", index=False)
class_summary.to_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/TS_data/tabula_sapiens/final_data/tissue_class_summary_ExonBinPsi.csv", index=False)
