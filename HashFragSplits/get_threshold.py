import os
import glob
import pandas as pd

DIR = "/gpfs/commons/home/nkeung/hashFrag_CL/calibration_blastn_array_job.work/blast_results"

files = glob.glob(f"{DIR}/*.tsv")
dfs = [
    pd.read_csv(f, sep="\t", header=None,
                names=["query", "target", "score"])
    for f in files
]
calibration_df = pd.concat(dfs, ignore_index=True)

# Get Max BLAST Score
max_scores = calibration_df.groupby("query")["score"].max()
print(max_scores.describe())
print("Overall maximum:", max_scores.max())

# max_scores:
# count    10414.000000
# mean        23.924429
# std          3.206781
# min         19.000000
# 25%         22.000000
# 50%         23.000000
# 75%         25.000000
# max         61.000000
# Name: score, dtype: float64
# Overall maximum: 61
