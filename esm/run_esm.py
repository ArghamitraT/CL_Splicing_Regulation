"""
Loads ESM model and calculates embeddings. Protein sequences comes from
aa_exon_stitcher.py. Saves embeddings in "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt"
"""

import argparse
import torch
import esm
from get_aa_seq import build_full_seq
import pandas as pd

input_file = "/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/foxp2-aa-seqs.csv"
output_dir = "/gpfs/commons/home/nkeung/data/processed_data/"
num_exons = 16      # Number of exons in the foxp2 gene

def main(pool_type):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    data = build_full_seq(input_file)

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]      # results dictionary stores dictionary "representations" with keys for each layer
    print(f"Shape of token_representations: {token_representations.shape}")

    if (pool_type == "full"):
        # Generate full SEQUENCE representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = {}
        for i, tokens_len in enumerate(batch_lens):
            name = batch_labels[i]
            vector = token_representations[i, 1 : tokens_len - 1].mean(0)
            sequence_representations[name] = vector
            print(f"Shape of {name}: {vector.shape}")

        torch.save(sequence_representations, output_dir + "foxp2_full.pt")
    
    elif (pool_type == "exon"):
            # Generate per-EXON representations via averaging
            # Get exon length from input file
            df = pd.read_csv(input_file)
            df = df.sort_values(by=["Species", "Number"])
            df = df.groupby("Species")
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            exon_representations = {}
            for i, tokens_len in enumerate(batch_lens):
                name = batch_labels[i]
                for j in range(1, num_exons + 1):
                    # Get exon length
                    start_index = 1
                    match = df[(df["Species"] == name) & (df["Number"] == j)]
                    if match.empty:
                        print(f"Exon {j} not found for species {name}")
                        continue
                    exon_len = match["AA Len"].values[0]

                    end_index = start_index + exon_len
                    vector = token_representations[i, start_index : end_index].mean(0)
                    exon_representations[j][name] = vector
                    start_index = end_index

            torch.save(exon_representations, output_dir+"foxp2_exons.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESM-2 model on protein sequences")
    parser.add_argument(
        "--pool_type",
        type=str,
        choices=["full", "exon"],
        default="full",
        help="Pooling type for sequence representation (full sequence or exon)"
    )
    args = parser.parse_args()
    main(args.pool_type)
