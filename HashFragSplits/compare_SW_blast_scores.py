import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PURE_PATH = "/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work/pure/candidate_scores.tsv"
LIGHT_PATH = "/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work/lightning/candidate_scores.tsv"

THRESH = 62

col_names = ["test", "train", "score"]

pure = pd.read_csv(PURE_PATH, sep="\t", names=col_names)
light = pd.read_csv(LIGHT_PATH, sep="\t", names=col_names)

merged = light.merge(
    right=pure,
    how="outer",
    validate="1:1",
    left_on=["test", "train"],
    right_on=["test", "train"],
    suffixes=["_blast", "_sw"]
)

# Verify no missing score values
print(len(merged), merged[["score_blast", "score_sw"]].isna().sum().to_dict())

# Plot 2D hist comparing scores
bins = np.arange(0, 810, 5)
fig, ax = plt.subplots(figsize=(6, 5))
h = ax.hist2d(merged["score_blast"], merged["score_sw"], bins=bins, cmin=1, norm=LogNorm())
fig.colorbar(h[3], ax=ax, label="pairs")
ax.axvline(THRESH, ls="--", c="k", lw=1)
ax.axhline(THRESH, ls="--", c="k", lw=1)
ax.set_xlabel("BLAST score")
ax.set_ylabel("SW score")
ax.set_aspect("equal")
ax.set_box_aspect(1)
fig.tight_layout()
fig.savefig("blast_vs_sw.png", dpi=300)
