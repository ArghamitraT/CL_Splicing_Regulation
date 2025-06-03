"""
Calculate cosine similarity from embeddings saved in "/gpfs/commons/home/nkeung/data/processed_data/{gene}-representations.pt"
"""

import argparse
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from ete3 import Tree, TreeStyle, NodeStyle, TextFace

# Set offscreen rendering for ete3
os.environ["QT_QPA_PLATFORM"] = "offscreen"

input_dir = "/gpfs/commons/home/nkeung/data/"
output_dir = "/gpfs/commons/home/nkeung/data/figures/"
gene = "brca2"
num_exons = 26      # Number of exons in the gene
epsilon = 1e-6      # Small value to avoid log(0) issues

species_colors = {}

common_tree = Tree(input_dir + "hg38.100way.commonNames.nh")
common_names = [leaf.name for leaf in common_tree.iter_leaves()]
ucsc_tree = Tree(input_dir + "hg38.100way.nh")
ucsc_codes = [leaf.name for leaf in ucsc_tree.iter_leaves()]


def calculate_cosine_similarity(representations):
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


def embedding_l1_dist(representations):
    species_list = list(representations.keys())
    embeddings = np.array([representations[species] for species in representations])
    hg38_vector = embeddings[species_list.index("hg38")].reshape(1,-1)

    dist_matrix = manhattan_distances(hg38_vector, embeddings)
    diff = dict(zip(species_list, dist_matrix.flatten()))
    for species in ucsc_codes:
        if species not in diff:
            diff[species] = np.nan
    
    # Sort embeddings in order of ucsc_codes
    in_order = {sp: diff[sp] for sp in ucsc_codes}
    return in_order


def embedding_l2_dist(representations):
    species_list = list(representations.keys())
    embeddings = np.array([representations[species] for species in representations])
    hg38_vector = embeddings[species_list.index("hg38")].reshape(1,-1)

    dist_matrix = euclidean_distances(hg38_vector, embeddings)
    diff = dict(zip(species_list, dist_matrix.flatten()))
    for species in ucsc_codes:
        if species not in diff:
            diff[species] = np.nan
    
    # Sort embeddings in order of ucsc_codes
    in_order = {sp: diff[sp] for sp in ucsc_codes}
    return in_order


def get_common_name(embeddings: dict):
    global ucsc_codes, common_names
    new_names = dict(zip(ucsc_codes, common_names))
    new_dict = {new_names[species]: embeddings[species] for species in embeddings if species in new_names}
    return new_dict


def plot_bar(similarities: dict):
    species = similarities.keys()
    scores = similarities.values()

    plt.figure(figsize=(18, 6))
    plt.bar(species, scores, color=[species_colors[s] for s in species])
    plt.xticks(rotation=90)
    plt.xlabel("Species")
    plt.ylabel("Log Cosine Similarity to hg38")
    plt.title(f"Cosine Similarity of {gene} Full Sequence Representations")
    plt.savefig(output_dir+f"{gene}_full_cosine_similarity.png", dpi=300, bbox_inches='tight')


def draw_tree(similarities: dict):
    # Set all four node styles
    styles = {}
    for color in ["tab:blue", "tab:orange", "tab:green", "tab:red"]:
        hex_color = to_hex(color)
        style = NodeStyle()
        style["fgcolor"] = hex_color
        style["size"] = 0
        style["vt_line_color"] = hex_color
        style["hz_line_color"] = hex_color
        style["hz_line_width"] = 2
        styles[color] = style


    for leaf in common_tree.iter_leaves():
        # Display similarity score for each leaf
        leaf_face = TextFace(similarities[leaf.name], fgcolor=to_hex(species_colors[leaf.name]))
        leaf.add_face(leaf_face, column=0, position="aligned")
        leaf.set_style(styles[species_colors[leaf.name]])
    
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.mode = "c"
    common_tree.render(output_dir+f"{gene}_tree.png", tree_style=ts, dpi=300)


def plot_heat_map(matrix, species):
    plt.figure(figsize=(18, 6))
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Log Cosine Similarity to hg38')
    plt.xticks(ticks=np.arange(len(species)), 
                labels=list(species), rotation=90)
    plt.yticks(ticks=np.arange(num_exons), labels=[f"Exon {i}" for i in range(1, num_exons + 1)])
    plt.title(f"Cosine Similarity of {gene} Exon Representations")
    plt.savefig(input_dir+f"figures/{gene}_exon_cosine_similarity.png", dpi=300, bbox_inches='tight')


def main(pool_type):
    if pool_type == "full":
        # --- COSINE SIMILARITY ---
        sequence_representations = torch.load(input_dir+f"embeddings/{gene}_full.pt")
        similarity = calculate_cosine_similarity(sequence_representations)
        similarity = get_common_name(similarity)

        # Calculate similarities in log scale
        species = list(similarity.keys())
        values = np.array(list(similarity.values()))
        log_vals = -np.log(1 - np.clip(values, 0, 1 - epsilon))
        log_sim = {}
        for s, v in zip(species, log_vals):
            log_sim[s] = v

        print("Smallest cosine similarity:", min(similarity.values()))
        print("Smallest log cosine similarity:", min(log_sim.values()), "\n")
        print("Largest cosine similarity:", max(value for key, value in similarity.items() if key != "Human"))
        print("Largest log cosine similarity:", max(value for key, value in log_sim.items() if key != "Human"))

        # Sort by descending similarity
        sorted_vals = sorted(log_sim.items(), key=lambda s: s[1], reverse=True)
        sorted_species, desc_log_cos = zip(*sorted_vals)
        log_sim = dict(sorted_vals)

        # Set colors after sorting
        global species_colors
        species_colors = {"Human": "tab:blue"}
        for i in range(1, 100):
            if i <= 33:
                species_colors[sorted_species[i]] = "tab:orange"
            elif i <= 66:
                species_colors[sorted_species[i]] = "tab:green"
            else:
                species_colors[sorted_species[i]] = "tab:red"

        # plot_bar(log_sim)
        # draw_tree(log_sim)

        # --- HIGH DIMENSIONAL DIFFERENCES ---
        # MANHATTAN DISTANCE (L1)
        l1_diff = get_common_name(embedding_l1_dist(sequence_representations))
        sorted_l1 = {sp: l1_diff[sp] for sp in sorted_species}
        
        # Manhattan Distance Across Species
        plt.figure(figsize=(18, 6))
        plt.bar(sorted_species, sorted_l1.values(), color=[species_colors[s] for s in sorted_species])
        plt.xticks(rotation=90)
        plt.xlabel("Species")
        plt.ylabel("L1 Distance from hg38")
        plt.title(f"Cosine Similarity of {gene} Full Sequence Representations")
        plt.savefig(output_dir+f"{gene}_full_manhattan.png", dpi=300, bbox_inches='tight')

        # Cos Similarity vs Manhattan Distance
        plt.figure(figsize=(18, 6))
        plt.scatter(desc_log_cos, sorted_l1.values())
        plt.xlabel("-Log (1 - Cosine Similarity)")
        plt.ylabel("L1 Distance from hg38")
        plt.title("Cosine Similarity vs Manhattan Distance")
        plt.savefig(output_dir+f"{gene}_full_cos_vs_manhattan.png", dpi=300, bbox_inches='tight')

        # EUCLIDEAN DISTANCE (L2)
        l2_diff = get_common_name(embedding_l1_dist(sequence_representations))
        sorted_l2 = {sp: l2_diff[sp] for sp in sorted_species}
        
        # Euclidean Distance Across Species
        plt.figure(figsize=(18, 6))
        plt.bar(sorted_species, sorted_l2.values(), color=[species_colors[s] for s in sorted_species])
        plt.xticks(rotation=90)
        plt.xlabel("Species")
        plt.ylabel("L2 Distance from hg38")
        plt.title(f"Cosine Similarity of {gene} Full Sequence Representations")
        plt.savefig(output_dir+f"{gene}_full_euclidean.png", dpi=300, bbox_inches='tight')

        # Cos Similarity vs Euclidean Distance
        plt.figure(figsize=(18, 6))
        plt.scatter(desc_log_cos, sorted_l2.values())
        plt.xlabel("-Log (1 - Cosine Similarity)")
        plt.ylabel("L2 Distance from hg38")
        plt.title("Cosine Similarity vs Euclidean Distance")
        plt.savefig(output_dir+f"{gene}_full_cos_vs_euclidean.png", dpi=300, bbox_inches='tight')

    elif pool_type == "exon":
        sequence_representations = torch.load(input_dir+f"embeddings/{gene}_exons.pt")
        similarity = {}
        for i in range(1, num_exons + 1):
            # print(f"Exon {i} cosine similarity:")
            exon_sim = calculate_cosine_similarity(sequence_representations[i])
            similarity[i] = get_common_name(exon_sim)
            # print("---------------------\n")
        
        # Store values in matrix of shape (num_exons, num_species)
        sim_matrix_rows = []
        for exon in range(1, num_exons + 1):
            new_row = list(similarity[exon].values())
            sim_matrix_rows.append(new_row)
        sim_matrix = np.array(sim_matrix_rows)
        log_matrix = -np.log(1 - np.clip(sim_matrix, epsilon, 1 - epsilon))
        plot_heat_map(log_matrix, similarity[1].keys())
    
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
