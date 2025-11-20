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
    ("Other Mammals", 57, 61),
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
            legend_position=(0.99, 1.09),
            legend_justification=(1, 1),
                legend_background=element_blank(),
                legend_title=element_blank(),
                legend_text=element_text(size=15),
        axis_text_x=element_text(rotation=90, hjust=1, size=15, ha="center", va="top", color="#222222"),
        axis_text_y=element_text(size=15, color="#222222"),
        axis_title_x=element_text(margin={'t': 8}, size=24, weight="bold"),
        axis_title_y=element_text(size=24, weight="bold"),
        plot_title=element_text(size=24, weight="bold", ha="center"),
        panel_border=element_blank(),
        panel_grid_major_y=element_text(color="#dddddd"),
        panel_grid_minor_y=element_blank(),
        panel_grid_major_x=element_blank(),
        panel_grid_minor_x=element_blank(),
        axis_line_x=element_text(color="black"),
        axis_line_y=element_text(color="black"),
    )
)

# Save figure
p.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts_long.svg")

# # Plot with horizontal bars
# p_horizontal = (
#     ggplot(filtered_df, aes(x="common_name", y="count", fill="subset_name"))
#     + geom_col()
#     + coord_flip()
#     + labs(
#         title="Number of Exons Per Species",
#         x="Exon Counts",
#         y="Species",
#         fill="Multiz Subsets"
#     )
#     + scale_fill_brewer(type="qual", palette="Dark2")
#     + theme_bw()
#     + theme(
#         figure_size=(9,18),
#             legend_position=(1.09, 0.99),
#             legend_justification=(1, 1),
#                 legend_background=element_blank(),
#                 legend_title=element_blank(),
#                 legend_text=element_text(size=15),
#         axis_text_x=element_text(size=15, ha="center", va="top", color="#222222"),
#         axis_text_y=element_text(size=15, color="#222222"),
#         axis_title_x=element_text(margin={'t': 8}, size=18, weight="bold"),
#         axis_title_y=element_text(size=20, weight="bold"),
#         plot_title=element_text(size=15, weight="bold", ha="center"),
#         panel_border=element_blank(),
#         panel_grid_major_y=element_text(color="#dddddd"),
#         panel_grid_minor_y=element_blank(),
#         panel_grid_major_x=element_blank(),
#         panel_grid_minor_x=element_blank(),
#         axis_line_x=element_text(color="black"),
#         axis_line_y=element_text(color="black"),
#     )
# )
# p_horizontal.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts_tall.svg")


# # Save summary figure of just counts among subsets
# subset_df = filtered_df.groupby("subset_name", as_index=False)["count"].sum()

# p_subset = (
#     ggplot(subset_df, aes(x="subset_name", y="count", fill="subset_name"))
#     + geom_col()
#     + labs(
#         title="Number of Exons Per Species Subset",
#         x="Multiz Subset",
#         y="Exon Counts",
#         fill="Multiz Subsets"
#     )
#     + scale_fill_brewer(type="qual", palette="Dark2")
#     + theme_bw()
#     + theme(
#         figure_size=(6,9),
#         legend_position="none",
#         axis_text_x=element_text(rotation=90, hjust=1, size=15, ha="center", va="top", color="#222222"),
#         axis_text_y=element_text(size=15, color="#222222"),
#         axis_title_x=element_text(margin={'t': 8}, size=18, weight="bold"),
#         axis_title_y=element_text(size=20, weight="bold"),
#         plot_title=element_text(size=15, weight="bold", ha="center"),
#         panel_border=element_blank(),
#         panel_grid_major_y=element_text(color="#dddddd"),
#         panel_grid_minor_y=element_blank(),
#         panel_grid_major_x=element_blank(),
#         panel_grid_minor_x=element_blank(),
#         axis_line_x=element_text(color="black"),
#         axis_line_y=element_text(color="black"),
#     )
# )

# p_subset.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts_subset.svg")

