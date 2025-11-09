import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np

df_known = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv")
names_df = pd.read_csv("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/species_name_codes.csv")


exon_counts_df = df_known.groupby("Species Name").size().reset_index(name="count")

merged_df = names_df.merge(
    exon_counts_df,
    left_on="ucsc_code",
    right_on="Species Name",
    how="left"
)
merged_df = merged_df.drop("Species Name", axis=1)

# Add numbering to dataframe for grap
merged_df = merged_df.reset_index().rename(columns={"index":"num"})
merged_df["num"] = merged_df["num"] + 1

merged_df.to_csv("/gpfs/commons/home/nkeung/Contrastive_Learning/code/figures/species_exon_counts.csv", index=False)

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
palette = sns.color_palette("tab20", n_colors=len(subsets))
colors = np.empty(len(merged_df), dtype=object)
for i, (clade, start, end) in enumerate(subsets):
    colors[start:end+1] = [palette[i]] * (end - start + 1)

merged_df["color"] = colors


plt.figure(figsize=(12, 30))

# Horizontal bar plot
plt.barh(
    y=merged_df["common_name"][::-1],  # species on y-axis
    width=merged_df["count"][::-1],    # exon counts
    color=merged_df["color"][::-1]     # precomputed clade color
)
plt.ylim(-0.5, len(merged_df) - 0.5)
plt.xlabel("Exon Counts")
plt.ylabel("Species")
plt.title("Number of Exons per Species")

# Legend for clades
legend_elements = [
    Patch(facecolor=color, label=clade_name)
    for (clade_name, _, _), color in zip(subsets, palette)
]
plt.legend(
    handles=legend_elements,
    title="Multiz Subsets",
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0
)

# Save figure
plt.tight_layout()
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts.pdf", dpi=300)
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/multiz_exon_counts.png", dpi=300)
