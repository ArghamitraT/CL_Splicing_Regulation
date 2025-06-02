"""
Calculate cosine similarity from embeddings saved in "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt"
"""

import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from ete3 import Tree

input_dir = "/gpfs/commons/home/nkeung/data/"
output_dir = "/gpfs/commons/home/nkeung/data/figures/"
num_exons = 16      # Number of exons in the foxp2 gene
epsilon = 1e-6  # Small value to avoid log(0) issues

species_colors = {}

common_tree = Tree(input_dir + "hg38.100way.commonNames.nh")
common_names = [leaf.name for leaf in common_tree.iter_leaves()]
ucsc_tree = Tree(input_dir + "hg38.100way.nh")
ucsc_codes = [leaf.name for leaf in ucsc_tree.iter_leaves()]

def calculate_cosine_similarity(representations):
    # --- COSINE SIMILARITY ---
    # Store list of species names for later
    species_list = list(representations.keys())
    # Create numpy array of all embeddings
    embeddings = np.array([representations[species] for species in species_list])

    # Refernce vector (hg38)
    hg38_vector = embeddings[species_list.index("hg38")].reshape(1, -1)  # reshape to 2D array of dimensions (1 x #)

    # Find cosine similarities in comparison to "hg38" only
    cos_sim_matrix = cosine_similarity(hg38_vector, embeddings)

    similarity = dict(zip(species_list, cos_sim_matrix.flatten()))
    for species in ucsc_codes:
        if species not in similarity:
            similarity[species] = 0.0

    # Sort embeddings in tree order, following ucsc_codes
    in_order = {species: similarity[species] for species in ucsc_codes}
    return in_order

def get_common_name(embeddings: dict):
    global ucsc_codes, common_names
    new_names = dict(zip(ucsc_codes, common_names))
    new_dict = {new_names[species]: embeddings[species] for species in embeddings if species in new_names}
    return new_dict

def plot_log_scale(cos_sim: dict):
    species = list(cos_sim.keys())
    values = np.array(list(cos_sim.values()))
    log_vals = -np.log(1 - np.clip(values, 0, 1 - epsilon))
    log_sim = {}
    for s, v in zip(species, log_vals):
        log_sim[s] = v

    print("Smallest cosine similarity:", min(cos_sim.values()))
    print("Smallest log cosine similarity:", min(log_sim.values()), "\n")
    print("Largest cosine similarity:", max(value for key, value in cos_sim.items() if key != "Human"))
    print("Largest log cosine similarity:", max(value for key, value in log_sim.items() if key != "Human"))

    # Calculate and sort log cosine similarity
    sorted_vals = sorted(log_sim.items(), key=lambda s: s[1], reverse=True)
    sorted_species, sorted_scores = zip(*sorted_vals)

    global species_colors
    species_colors = {"Human": "tab:blue"}
    for i in range(1, 100):
        if i <= 33:
            species_colors[sorted_species[i]] = "tab:orange"
        elif i <= 66:
            species_colors[sorted_species[i]] = "tab:green"
        else:
            species_colors[sorted_species[i]] = "tab:red"

    plt.figure(figsize=(18, 6))
    plt.bar(sorted_species, sorted_scores, color=[species_colors[s] for s in sorted_species])
    plt.xticks(rotation=90)
    plt.xlabel("Species")
    plt.ylabel("Log Cosine Similarity to hg38")
    plt.title("Cosine Similarity of FoxP2 Full Sequence Representations")
    plt.savefig(output_dir+"foxp2_full_cosine_similarity.png", dpi=300, bbox_inches='tight')

def plot_log_heatmap(similarity: dict):
    # Store values in matrix of shape (num_exons, num_species)
    sim_matrix_rows = []
    for exon in range(1, num_exons + 1):
        new_row = list(similarity[exon].values())
        sim_matrix_rows.append(new_row)
    sim_matrix = np.array(sim_matrix_rows)

    log_matrix = -np.log(1 - np.clip(sim_matrix, epsilon, 1 - epsilon))
    
    plt.figure(figsize=(18, 6))
    plt.imshow(log_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Log Cosine Similarity to hg38')
    plt.xticks(ticks=np.arange(len(similarity[1].keys())), 
                labels=list(similarity[1].keys()), rotation=90)
    plt.yticks(ticks=np.arange(num_exons), labels=[f"Exon {i}" for i in range(1, num_exons + 1)])
    plt.title("Cosine Similarity of FoxP2 Exon Representations")
    plt.savefig(input_dir+"figures/foxp2_exon_cosine_similarity.png", dpi=300, bbox_inches='tight')

def main(pool_type):
    if pool_type == "full":
        sequence_representations = torch.load(input_dir+"embeddings/foxp2_full.pt")
        similarity = calculate_cosine_similarity(sequence_representations)
        similarity = get_common_name(similarity)

        plot_log_scale(similarity)

    elif pool_type == "exon":
        sequence_representations = torch.load(input_dir+"embeddings/foxp2_exons.pt")
        similarity = {}
        for i in range(1, num_exons + 1):
            # print(f"Exon {i} cosine similarity:")
            exon_sim = calculate_cosine_similarity(sequence_representations[i])
            similarity[i] = get_common_name(exon_sim)
            # print("---------------------\n")
        
        plot_log_heatmap(similarity)
    
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
