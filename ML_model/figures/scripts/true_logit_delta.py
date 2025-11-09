import pandas as pd
import numpy as np
from scipy.special import logit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


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

sns.set_theme(style="whitegrid", context="paper")

plt.figure(figsize=(8, 5))
sns.histplot(
    data=ascot_delta_logit_series,
    bins=50,
    stat="density",
    color="#0173b2",
    alpha=0.6,
    edgecolor=None
)
plt.xlabel("Δ Logit PSI (All Cell Tissues)")
plt.ylabel("Frequency (Normalized)")
plt.title("ASCOT Distribution of Δ Logit PSI Across All Tissues")
plt.tight_layout()
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/figures/recomb_2026/ascot_delta_logit.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/figures/recomb_2026/ascot_delta_logit.png", dpi=300, bbox_inches="tight")


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

sns.set_theme(style="whitegrid", context="paper")

plt.figure(figsize=(8, 5))
sns.histplot(
    data=ts_delta_logit_series,
    bins=50,
    stat="density",
    color="#de8f05",
    alpha=0.6,
    edgecolor=None
)
plt.xlabel("Δ Logit PSI (All Cell Types)")
plt.ylabel("Frequency (Normalized)")
plt.title("Tabula Sapiens Distribution of Δ Logit PSI Across All Cell Types")
plt.tight_layout()
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/figures/recomb_2026/ts_delta_logit.pdf", dpi=300, bbox_inches="tight")
plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/figures/recomb_2026/ts_delta_logit.png", dpi=300, bbox_inches="tight")
