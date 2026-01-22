import pandas as pd
import numpy as np
from scipy.special import logit
import matplotlib.pyplot as plt
# import seaborn as sns
from plotnine import ggplot, aes, geom_histogram, labs, theme_bw, theme, element_text, element_blank


ts_df = pd.read_csv("/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/full_cassette_exons_with_mean_psi.csv")
ascot_df = pd.read_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons_with_logit_mean_psi.csv")

all_cell_types = ts_df.columns[12:-3].tolist()
all_tissues = ascot_df.columns[12:-4].str.replace("\n", "").str.strip().tolist()

# Plot ASCOT Distribution

def ascot_compute_delta_logit_psi(psi_col):
    psi_col = pd.to_numeric(psi_col, errors="coerce")
    valid_entries = psi_col.dropna()
    eps = 1e-6
    computed_logit = logit(np.clip(valid_entries / 100, eps, 1-eps))
    return computed_logit - ascot_df.loc[valid_entries.index, "logit_mean_psi"]

ascot_logit_df = ascot_df[all_tissues].apply(ascot_compute_delta_logit_psi, axis=0)
# ascot_logit_df = ascot_compute_delta_logit_psi(ascot_df["Retina - Eye"])      Sanity check, compare to provided figure
ascot_delta_logit = ascot_logit_df.values.flatten()
ascot_delta_logit = ascot_delta_logit[~np.isnan(ascot_delta_logit)]
ascot_delta_logit_series = pd.Series(ascot_delta_logit, name="Δ Logit PSI")
ascot_delta_logit_df = ascot_delta_logit_series.to_frame()

# sns.set_theme(style="whitegrid", context="paper")

# plt.figure(figsize=(8, 5))
# sns.histplot(
#     data=ascot_delta_logit_series,
#     bins=150,
#     stat="density",
#     color="#0173b2",
#     alpha=0.6,
#     edgecolor=None
# )
# plt.xlabel("Δ Logit PSI (All Tissues)")
# plt.ylabel("Density")
# plt.title("ASCOT Distribution of Δ Logit PSI Across All Tissues")
# plt.tight_layout()
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/ascot_delta_logit.png", dpi=300, bbox_inches="tight")

p = (
    ggplot(ascot_delta_logit_df, aes(x="Δ Logit PSI")) +
    geom_histogram(
        aes(y="..density.."),
        bins=150,
        fill="#af58ba",
        color="black",
        size=0.5
    ) + 
    labs(
        x="Δ Logit PSI (All Tissues)",
        y="Density",
        title="ASCOT Distribution of Δ Logit PSI Across All Tissues"
    ) +
    theme_bw() +
    theme(
        figure_size=(8, 5),
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
p.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/ascot_delta_logit.svg")


# Plot TS Distribution

def ts_compute_delta_logit_psi(psi_col):
    psi_col = pd.to_numeric(psi_col, errors="coerce")
    valid_entries = psi_col.dropna()
    eps = 1e-6
    computed_logit = logit(np.clip(valid_entries / 100, eps, 1-eps))
    return computed_logit - ts_df.loc[valid_entries.index, "logit_mean_psi"]

ts_logit_df = ts_df[all_cell_types].apply(ts_compute_delta_logit_psi, axis=0)
ts_delta_logit = ts_logit_df.values.flatten()
ts_delta_logit = ts_delta_logit[~np.isnan(ts_delta_logit)]
ts_delta_logit_series = pd.Series(ts_delta_logit, name="Δ Logit PSI")
ts_delta_logit_df = ts_delta_logit_series.to_frame()

# sns.set_theme(style="whitegrid", context="paper")

# plt.figure(figsize=(8, 5))
# sns.histplot(
#     data=ts_delta_logit_series,
#     bins=150,
#     stat="density",
#     color="#de8f05",
#     alpha=0.6,
#     edgecolor=None
# )
# plt.xlabel("Δ Logit PSI (All Cell Types)")
# plt.ylabel("Density")
# plt.title("Tabula Sapiens Distribution of Δ Logit PSI Across All Cell Types")
# plt.tight_layout()
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/ts_delta_logit.png", dpi=300, bbox_inches="tight")

p = (
    ggplot(ts_delta_logit_df, aes(x="Δ Logit PSI")) +
    geom_histogram(
        aes(y="..density.."),
        bins=150,
        fill="#de8f05",
        color="black",
        size=0.5
    ) + 
    labs(
        x="Δ Logit PSI (All Cell Types)",
        y="Density",
        title="TS Distribution of Δ Logit PSI Across All Cell Types"
    ) +
    theme_bw() +
    theme(
        figure_size=(8, 5),
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
p.save("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/ts_delta_logit.svg")
