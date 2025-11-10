import pandas as pd
import pickle
from plotnine import *
import numpy as np

# names_df = pd.read_csv("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/species_name_codes.csv")

# # seqs_used_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"

# # # Load data
# # with open(f"{seqs_used_path}/test_merged_filtered_min30Views.pkl", "rb") as f:
# #     test_dict = pickle.load(f)
# # with open(f"{seqs_used_path}/train_merged_filtered_min30Views.pkl", "rb") as f:
# #     train_dict = pickle.load(f)
# # with open(f"{seqs_used_path}/val_merged_filtered_min30Views.pkl", "rb") as f:
# #     val_dict = pickle.load(f)

# # rows = []
# # for split, dictionary in {
# #     "test": test_dict,
# #     "train": train_dict,
# #     "val": val_dict
# # }.items():
# #     for exon_id, species_dict in dictionary.items():
# #         for species in species_dict.keys():
# #             rows.append({
# #                 "split": split,
# #                 "transcript_id": exon_id,
# #                 "species": species
# #             })

# # used_species = pd.DataFrame(rows)
# # exon_counts_df = used_species.groupby("species").size().reset_index(name="count")
# # print(exon_counts_df)

# merged_df = names_df.merge(
#     exon_counts_df,
#     left_on="ucsc_code",
#     right_on="species",
#     how="left"
# )
# merged_df = merged_df.drop("species", axis=1)

# # Add numbering to dataframe
# merged_df = merged_df.reset_index().rename(columns={"index":"num"})
# merged_df["num"] = merged_df["num"] + 1

# merged_df.to_csv("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/species_exon_counts.csv", index=False)

merged_df = pd.read_csv("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/species_exon_counts.csv")

subsets = [
    ("Primates", 0, 11),
    ("Euarchontoglires", 12, 25),
    ("Laurasiatheria", 26, 50),
    ("Afrotheria", 51, 56),
    ("Mammal", 57, 61),
    ("Aves", 62, 75),
    ("Sarcopterygii", 76, 83),
    ("Fish", 84, 99)
]

merged_df["subset_name"] = None
for (subset_name, start, end) in subsets:
    merged_df.loc[start:end, "subset_name"] = subset_name

filtered_df = merged_df.dropna(subset=["count"]).copy()

# Save figure
plt.tight_layout()
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts.pdf", dpi=300)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts.png", dpi=300)
