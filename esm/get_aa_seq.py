"""
Function searches input file and stitches together all exons
in order of the same species. 
"""

import pandas as pd
import argparse
import json

class Config:
    """
    Configure gene to study and input/output file
    """
    num_exons_map = {"foxp2": 16, "brca2": 26, "hla-a": 8, "tp53": 10}
    
    def __init__(self, gene):
        self.gene = gene
        self.exon_num = Config.num_exons_map[gene]
        self.input_file = f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}-all-seqs.csv"
        self.output_file = f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}"


def build_full_seq(cfg: Config):
    species_seq = []

    df = pd.read_csv(cfg.input_file)

    df = df.sort_values(by=["Species", "Number"])
    grouped_df = df.groupby("Species")

    for species, groups in grouped_df:
        sequence = ""
        for exon in groups["Seq"]:
            exon = exon.replace("-", "")
            exon = exon.replace("\n", "")
            sequence += exon
        species_seq.append((species, sequence))

    with open(cfg.output_file+"-full-stitched.json", "w") as f:
        json.dump(species_seq, f)
    
    print(f"Saved full sequence as {cfg.output_file}-full-stitched.json")


### NOTE: NOT NEEDED FOR ESM ###
def single_exons(cfg: Config):
    exon_seqs = []

    df = pd.read_csv(cfg.input_file)

    df = df.sort_values(by=["Species", "Number"])
    grouped_df = df.groupby("Species")
    for i in range(cfg.exon_num):
        for species, group in grouped_df:
            match = group[group["Number"] == 1+i]
            if not match.empty:
                exon = match["Seq"].iloc[0]
                exon = exon.replace("-", "")
                exon = exon.replace("\n", "")
                exon_seqs.append(((species, i+1), exon))

    with open(cfg.output_file+f"-exons.json", "w") as f:
        json.dump(exon_seqs, f)

    print(f"Saved exon sequences as {cfg.output_file}-exons.json")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store full amino acid sequences in .json")
    parser.add_argument(
        "--gene",
        type=str,
        choices=["foxp2", "brca2", "hla-a", "tp53"],
        required=True,
        help="Gene to perform similarity testing on"
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        choices=["full", "exon"],
        default="full",
        help="Mean pooling embedding over exon or entire protein"
    )
    args = parser.parse_args()
    cfg = Config(args.gene)
    if args.pool_type == "full":
        build_full_seq(cfg)
    else:
        single_exons(cfg)
