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

    return species_seq

def single_exons(input_file, exon_num):
    exon_seqs = []

    df = pd.read_csv(input_file)

    df = df.sort_values(by=["Species", "Number"])
    grouped_df = df.groupby("Species")
    for species, group in grouped_df:
        match = group[group["Number"] == exon_num]
        if not match.empty:
            exon = match["Seq"]
            exon = exon.replace("-", "")
            exon = exon.replace("\n", "")
            exon_seqs.append(((species, exon_num), exon))

    return exon_seqs
