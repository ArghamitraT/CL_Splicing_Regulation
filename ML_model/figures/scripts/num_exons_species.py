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

# Sort into subsets for color coding
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

# Define a set order for each label (for plotting)
merged_df["subset_name"] = pd.Categorical(merged_df["subset_name"], categories=merged_df["subset_name"].unique(), ordered=True)
merged_df["common_name"] = pd.Categorical(merged_df["common_name"], categories=merged_df["common_name"], ordered=True)

# Remove species with no exons
filtered_df = merged_df.dropna(subset=["count"]).copy()

p = (
    ggplot(filtered_df, aes(x="common_name", y="count", fill="subset_name"))
    + geom_col()
    + labs(
        title="Number of Exons Per Species",
        x="Species",
        y="Exon Counts",
        fill="Multiz Subsets"
    )
    + scale_fill_brewer(type="qual", palette="Dark2")
    + theme_bw()
    + theme(
        figure_size=(18,9),
        plot_title=element_text(size=16),
        axis_text_x=element_text(rotation=90, hjust=1, size=12),
        axis_text_y=element_text(size=12),
        axis_title_x=element_text(size=14),
        axis_title_y=element_text(size=14),
        legend_background=element_rect(color="black", fill="white", size=0.5),
        legend_position=(0.99, 0.89),
        legend_justification="right",
        legend_title=element_text(size=14),
        legend_text=element_text(size=12)
    )
)

# Save figure
p.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts.pdf", dpi=300)
p.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts.png", dpi=300)
