"""
Calculate cosine similarity from embeddings saved in "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt"
"""

import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

input_dir = "/gpfs/commons/home/nkeung/data/embeddings/"
output_dir = "/gpfs/commons/home/nkeung/data/figures/"
num_exons = 16      # Number of exons in the foxp2 gene
epsilon = 1e-6  # Small value to avoid log(0) issues

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
    # for species, score in similarities.items():
    #     print(f"{species}: {score:.4f}")
    return similarities


def main(pool_type):
    if pool_type == "full":
        sequence_representations = torch.load(input_dir+"foxp2_full.pt")
        similarity = calculate_cosine_similarity(sequence_representations)

        species = list(similarity.keys())
        values = np.array(list(similarity.values()))
        log_vals = -np.log(1 - np.clip(values, 0, 1 - epsilon))
        log_sim = {}
        for s, v in zip(species, log_vals):
            log_sim[s] = v

        print("Smallest cosine similarity:", min(similarity.values()))
        print("Smallest log cosine similarity:", min(log_sim.values()))
        print()

        # Calculate and sort log cosine similarity
        sorted_vals = sorted(log_sim.items(), key=lambda s: s[1], reverse=True)
        sorted_species, sorted_scores = zip(*sorted_vals)

        plt.figure(figsize=(18, 6))
        plt.bar(sorted_species, sorted_scores)
        plt.xticks(rotation=90)
        plt.xlabel("Species")
        plt.ylabel("Log Cosine Similarity to hg38")
        plt.title("Cosine Similarity of FoxP2 Full Sequence Representations")
        plt.savefig(output_dir+"foxp2_full_cosine_similarity.png", dpi=300, bbox_inches='tight')

    elif pool_type == "exon":
        sequence_representations = torch.load(input_dir+f"foxp2_exons.pt")
        similarity = {}
        for i in range(1, num_exons + 1):
            # print(f"Exon {i} cosine similarity:")
            similarity[i] = calculate_cosine_similarity(sequence_representations[i])
            # print("---------------------\n")
        
        # Store values in matrix of shape (num_exons, num_species)
        sim_matrix = np.array([[similarity[exon][species] 
                                for species in similarity[1].keys()
                                for exon in range(1, num_exons + 1)]])
        
        log_matrix = -np.log(1 - np.clip(sim_matrix, 0, 1 - epsilon))
        
        plt.figure(figsize=(18, 6))
        plt.imshow(log_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Log Cosine Similarity to hg38')
        plt.xticks(ticks=np.arange(len(similarity[1].keys())), 
                   labels=list(similarity[1].keys()), rotation=90)
        plt.yticks(ticks=np.arange(num_exons), labels=[f"Exon {i}" for i in range(1, num_exons + 1)])
        plt.title("Cosine Similarity of FoxP2 Exon Representations")
        plt.savefig(input_dir+"foxp2_exon_cosine_similarity.png", dpi=300, bbox_inches='tight')
 
    
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
