# import pickle

# file = 'filtered_HiLow_exons_by_tissue'
# path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/{file}.pkl"
# with open(path, "rb") as f:
#     data = pickle.load(f)

# print(data)

import pandas as pd

in_csv = "/mnt/home/at3836/Contrastive_Learning/data/final_data/TSCelltype_finetuning/test_cassette_exons_with_logit_mean_psi.csv"  # update
out_csv = "/mnt/home/at3836/Contrastive_Learning/data/final_data/TSCelltype_finetuning/test_cassette_exons_with_logit_mean_psi_dedup.csv"  # update

df = pd.read_csv(in_csv)
df.columns = df.columns.str.strip()
df["exon_id"] = df["exon_id"].astype(str).str.strip()  # just in case

before = len(df)
df_dedup = df.drop_duplicates(subset=["exon_id"], keep="first")
after = len(df_dedup)

df_dedup.to_csv(out_csv, index=False)
print(f"Removed {before - after} duplicate rows; saved → {out_csv}")
