"""
Checks coordinates of each exon associated with each species.
Adds up the total number of nucleotides. Then takes mod 3 to get
frame shift information. Then, compare values to hg38
"""

import pandas as pd

all_species = ["hg38", "mm10", "panTro4", "ponAbe2", "gorGor3", "rheMac3"]
species_mod = {}

df = pd.read_csv("/gpfs/commons/home/nkeung/data/processed_data/foxp2-aa-seqs.csv")

df["Nuc Length"] = (df["End Coord"] - df["Start Coord"]).abs() + 1
total_len = df.groupby(["Species"])["Nuc Length"].sum()
print(total_len.head(10))
print()

for spc, len in total_len.items():
    species_mod[spc] = (len, len % 3)

print(species_mod)
