import pandas as pd
from plotnine import *

ts_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/TSCelltype_finetuning/"
test_df = pd.read_csv(ts_dir + "test_cassette_exons_with_logit_mean_psi.csv")
train_df = pd.read_csv(ts_dir + "train_cassette_exons_with_logit_mean_psi.csv")
val_df = pd.read_csv(ts_dir + "val_cassette_exons_with_logit_mean_psi.csv")

ids_test = set(test_df["exon_id"])
ids_train = set(train_df["exon_id"])
ids_val = set(val_df["exon_id"])

print("Test ∩ Train:", len(ids_test & ids_train))
print("Test ∩ Val:", len(ids_test & ids_val))
print("Train ∩ Val:", len(ids_train & ids_val))

full_df = pd.concat([test_df, train_df, val_df], ignore_index=True)

all_cells = full_df.columns[12:-3].tolist()
print(len(all_cells))
ts_counts = full_df[all_cells].notna().sum()
ts_counts_df = ts_counts.reset_index()
ts_counts_df.columns = ["Cell Type", "Count"]

max_exons = ts_counts_df["Count"].max()
min_exons = ts_counts_df["Count"].min()
med_num_exons = ts_counts_df["Count"].median()
print(f"Max: {max_exons}")
print(f"Min: {min_exons}")
print(f"Median: {med_num_exons}")

ts_counts_df = ts_counts_df.sort_values("Count", ascending=False).reset_index(drop=True)
ts_counts_df["Group"] = "Low"
ts_counts_df.loc[:17, "Group"] = "High"
ts_counts_df.loc[18:93, "Group"] = "Medium"
color_map = {
    "Low": "#2384c9",
    "Medium": "#ffb42a",
    "High": "#d62728",
}
ts_counts_df["Color"] = ts_counts_df["Group"].map(color_map)

# Order bars in descending order
ts_counts_df["Cell Type"] = ts_counts_df["Cell Type"].str.title()
ts_counts_df["Cell Type"] = pd.Categorical(
    ts_counts_df["Cell Type"],
    categories=ts_counts_df.sort_values("Count", ascending=False)["Cell Type"],
    ordered=True
)

# Plot with horizontal bars
p_ts = (
    ggplot(ts_counts_df, aes(x="Cell Type", y="Count", fill="Color"))
    + geom_col()
    + scale_fill_identity(guide="legend", labels=["Low", "Medium", "High"], breaks=["#2384c9","#ffb42a","#d62728"])
    + coord_flip()
    + labs(
        title="Tabula Sapiens Data Cell-wise Exon Counts by Category",
        x="Cell Type",
        y="Total Exon Count",
    )
    + theme_bw()
    + theme(
        figure_size=(14,20),
        legend_position=(1.00, 0.99),
        legend_justification=(1, 1),
        legend_background=element_blank(),
        legend_title=element_blank(),
        legend_text=element_text(size=15),
        axis_text_x=element_text(size=15, ha="center", va="top", color="#222222"),
        axis_text_y=element_text(size=15, color="#222222"),
        axis_title_x=element_text(margin={'t': 8}, size=18, weight="bold"),
        axis_title_y=element_text(size=20, weight="bold"),
        plot_title=element_text(size=15, weight="bold", ha="center"),
        panel_border=element_blank(),
        panel_grid_major_y=element_text(color="#dddddd"),
        panel_grid_minor_y=element_blank(),
        panel_grid_major_x=element_blank(),
        panel_grid_minor_x=element_blank(),
        axis_line_x=element_text(color="black"),
        axis_line_y=element_text(color="black"),
    )
)
p_ts.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/ts_exon_counts.svg")