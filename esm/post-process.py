"""
Merge individual embeddings from ESM into a single dictionary and delete individual files.
Full sequence embeddings are saved in a dictionary with keys as species names.
Exon embeddings are saved in a dictionary as dictionary[exon_number][species_name] = vector.
"""

import os
import torch
import argparse

input_dir = "/gpfs/commons/home/nkeung/data/embeddings/"
output_dir = "/gpfs/commons/home/nkeung/data/embeddings"

# Example names: "foxp2_hg38_full.pt" and "foxp2_hg38_exon_1.pt"
gene = "brca2"

def process_full_seq():
    full_dir = os.path.join(input_dir, "full-seq/")
    full_seq_dict = {}
    for file in os.listdir(full_dir):
        if not file.endswith(".pt"):
            continue
        name_parts = file.replace(".pt", "").split("_")
        species = name_parts[1]
        if species not in full_seq_dict:
            vector = torch.load(os.path.join(full_dir, file))
            full_seq_dict[species] = vector
            os.remove(os.path.join(full_dir, file))

    torch.save(full_seq_dict, os.path.join(output_dir, f"{gene}_full.pt"))

def process_exon():
    full_dir = os.path.join(input_dir, "exon-seq/")
    exon_dict = {}
    for file in os.listdir(full_dir):
        if not file.endswith(".pt"):
            continue
        name_parts = file.replace(".pt", "").split("_")
        species = name_parts[1]
        exon_num = int(name_parts[3])
        
        if exon_num not in exon_dict:
            exon_dict[exon_num] = {}
        
        vector = torch.load(os.path.join(full_dir, file))
        exon_dict[exon_num][species] = vector
        os.remove(os.path.join(full_dir, file))

    torch.save(exon_dict, os.path.join(output_dir, f"{gene}_exons.pt"))

def main(pool_type):
    if pool_type == "full":
        process_full_seq()
    elif pool_type == "exon":
        process_exon()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ESM embeddings into a single dictionary")
    parser.add_argument(
        "--pool_type",
        type=str,
        choices=["full", "exon"],
        default="full",
        help="Pooling type for sequence representation (full sequence or exon)"
    )
    args = parser.parse_args()
    main(args.pool_type)
