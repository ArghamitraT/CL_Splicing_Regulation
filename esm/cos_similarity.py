"""
Calculate cosine similarity from embeddings saved in "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt"
"""

import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

input_dir = "/gpfs/commons/home/nkeung/data/embeddings/"
num_exons = 16      # Number of exons in the foxp2 gene

def calculate_cosine_similarity(sequence_representations):
    # --- COSINE SIMILARITY ---
    # Store list of species names for later
    species_list = list(sequence_representations.keys())
    # Create numpy array of all embeddings
    embeddings = np.array([sequence_representations[species] for species in species_list])

    # Refernce vector (hg38)
    hg38_vector = embeddings[species_list.index("hg38")].reshape(1, -1)  # reshape to 2D array of dimensions (1 x #)

    # Find cosine similarities in comparison to "hg38" only
    cos_sim_matrix = cosine_similarity(hg38_vector, embeddings)

    similarities = dict(zip(species_list, cos_sim_matrix.flatten()))
    for species, score in similarities.items():
        print(f"{species}: {score:.4f}")


def main(pool_type):
    if pool_type == "full":
        sequence_representations = torch.load(input_dir+"foxp2_full.pt")
        calculate_cosine_similarity(sequence_representations)
    elif pool_type == "exon":
        sequence_representations = torch.load(input_dir+f"foxp2_exons.pt")
        for i in range(1, num_exons + 1):
            print(f"Exon {i} cosine similarity:")
            calculate_cosine_similarity(sequence_representations[i])
            print("---------------------\n")
 
    
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
