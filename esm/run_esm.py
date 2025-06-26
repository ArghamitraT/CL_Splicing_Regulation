"""
Loads ESM model and calculates embeddings. Protein sequences come from 
.json files saved by save_aa_seq.py. Embeddings for each protein/exon 
are saved individually. Afterwards, run post-process.py to group embeddings
together.
"""

import argparse
import torch
import esm
import pandas as pd
import json

gene = None
num_exon_map = {"foxp2": 16, "brca2": 26, "hla-a":8, "tp53":10, "test": 2}
num_exons = None      # Number of exons in the gene

input_file = None       # See below
input_csv = None
output_dir = None

def initialize_globals(args):
    global gene, num_exons, input_file, input_csv, output_dir
    gene = args.gene
    num_exons = num_exon_map[gene]
    # input_file = f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}-full-stitched.json"
    input_file = f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/{gene}-full-stitched.json"
    # input_csv = f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}-all-seqs.csv"
    input_csv = f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/{gene}-all-seqs.csv"
    output_dir = "/gpfs/commons/home/nkeung/data/embeddings/"


"""
For genes with varying amino acid lengths, the batch size should be dynamic to avoid 
out-of-memory errors while still batching smaller exons together.
For example, BRCA2 has exons that are 142, 372, 203, 1644 amino acids long, but it
also hads exons that are 14 amino acids long.
"""
def dynamic_batcher(data, max_tokens=3500):
    batch = []
    tokens = 0
    for entry in sorted(data, key=lambda x:len(x[1])):
        seq_len = len(entry[1])
        if tokens + seq_len > max_tokens and batch:
            yield batch

            batch = [entry]
            tokens = seq_len
        else:
            batch.append(entry)
            tokens += seq_len
    if batch:
        yield batch

def main(pool_type):
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    with open(input_file, "r") as file:
        data = json.load(file)

    for batch_data in dynamic_batcher(data):
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # Get lengths of sequences without padding

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]      # results dictionary stores dictionary "representations" with keys for each layer
        # print(f"Shape of token_representations: {token_representations.shape}")     # (batch_size, seq_len, 1280)

        if (pool_type == "full"):
            # Generate full SEQUENCE representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            # sequence_representations = {}         # Not used, but can be used to save representations in a dictionary
            for i, tokens_len in enumerate(batch_lens):
                name = batch_labels[i]
                vector = token_representations[i, 1 : tokens_len - 1].mean(0)
                # sequence_representations[name] = vector
                # print(f"Shape of {name}: {vector.shape}")       # (1280)
                torch.save(vector, output_dir + f"full-seq/{gene}_{name}_full.pt")         # Save individual sequence representations

        
        elif (pool_type == "exon"):
                # Generate per-EXON representations via averaging
                # Get exon length from input file
                df = pd.read_csv(input_csv, dtype={"Seq": str})
                df["Seq"] = df["Seq"].fillna('')        # Turn nan sequences to an empty string
                df = df.sort_values(by=["Species", "Number"])
                # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
                # exon_representations = {}         # Not used, but can be used to save representations in a dictionary
                for i, tokens_len in enumerate(batch_lens):
                    name = batch_labels[i]
                    start_index = 1                 # Initialize index for first amino acid
                    for j in range(1, num_exons + 1):
                        # Get exon length
                        match = df[(df["Species"] == name) & (df["Number"] == j)]
                        # Check for missing exon
                        if match.empty:
                            print(f"Exon {j} not found for species {name}")
                            continue
                        exon_seq = match["Seq"].iloc[0]
                        exon_len = len(exon_seq.strip().replace("-", ""))
                        # Check for deleted exon
                        if exon_len == 0:
                            print(f"Exon missing. Skipped exon {j} for species {name}")
                            continue

                        end_index = start_index + exon_len
                        if end_index > token_representations.shape[1]:
                            print(f"Exon length exceeds tokens represented. Skipping exon {j} for {name}")
                            continue
                        vector = token_representations[i, start_index : end_index].mean(0)
                        start_index = end_index
                        # if j not in exon_representations:
                        #     exon_representations[j] = {}
                        # exon_representations[j][name] = vector
                        torch.save(vector, output_dir + f"exon-seq/{gene}_{name}_exon_{j}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESM-2 model on protein sequences")
    parser.add_argument(
        "--gene",
        type=str,
        choices=["foxp2", "brca2", "hla-a", "tp53", "test"],
        required=True,
        help="Gene for ESM input"
    )
    parser.add_argument(
        "--pool_type",
        type=str,
        choices=["full", "exon"],
        default="full",
        help="Pooling type for sequence representation (full sequence or exon)"
    )
    args = parser.parse_args()
    initialize_globals(args)
    main(args.pool_type)
