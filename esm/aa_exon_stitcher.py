"""
Function searches input file and stitches together all exons
in order of the same species. 
"""

import pandas as pd

def build_full_seq(input_file):
    species_seq = []

    df = pd.read_csv(input_file)

    df = df.sort_values(by=["Species", "Number"])
    grouped_df = df.groupby("Species")

    for species, groups in grouped_df:
        sequence = ""
        for exon in groups["Seq"]:
            exon = exon.replace("-", "")
            exon = exon.replace("\n", "")
            sequence += exon
        species_seq.append((species, sequence))
    
    for species in species_seq:
        print(species)

    return species_seq

seqs = build_full_seq("/gpfs/commons/home/nkeung/data/processed_data/foxp2-aa-seqs.csv")
