"""
Calculate similarities between amino acid embeddings for genes
FOXP2, HLA-A, and TP53. This script will generate plots showing the 
cosine similarity, color-coded phylogenetic tree, L1 and L2 distances,
and edit distance.
"""

import argparse
import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from ete3 import Tree, TreeStyle, NodeStyle, TextFace
import json
import pandas as pd

# Set offscreen rendering for ete3
os.environ["QT_QPA_PLATFORM"] = "offscreen"

input_dir = "/gpfs/commons/home/nkeung/data/"
output_dir = "/gpfs/commons/home/nkeung/data/figures/"

gene = None
num_exon_map = {"foxp2": 16, "brca2": 26, "hla-a":8, "tp53": 10}
num_exons = None      # Number of exons in the gene
epsilon = 1e-6      # Small value to aid log(0) issues

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


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def edit_distance(embeddings: dict):
    hg38_vector = embeddings["Human"]
    edit_dist = {sp:levenshteinDistance(hg38_vector, embeddings[sp]) for sp in embeddings.keys()}
    return edit_dist


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
    plt.savefig(output_dir+f"{gene}/{gene}_full_cosine_similarity.png", dpi=300, bbox_inches='tight')


def draw_tree(similarities: dict):
    # Set all four node styles
    styles = {}
    for color in ["tab:blue", "tab:green", "tab:orange", "tab:red"]:
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
    common_tree.render(output_dir+f"{gene}/{gene}_tree.png", tree_style=ts, dpi=300)


def plot_heat_map(matrix, species):
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(to_hex("tab:red"))
    plt.figure(figsize=(18, 6))
    plt.imshow(matrix, cmap, aspect='auto')
    plt.colorbar(label='Log Cosine Similarity to hg38')
    plt.xticks(ticks=np.arange(len(species)), 
                labels=list(species), rotation=90)
    plt.yticks(ticks=np.arange(num_exons), labels=[f"Exon {i}" for i in range(1, num_exons + 1)])
    plt.title(f"Cosine Similarity of {gene} Exon Representations")
    plt.savefig(output_dir+f"{gene}/{gene}_exon_cosine_similarity.png", dpi=300, bbox_inches='tight')


def main(pool_type, exon_to_compare=None):
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
        sorted_cos = {sp: similarity[sp] for sp in sorted_species}

        # Set colors after sorting
        global species_colors
        species_colors = {"Human": "tab:blue"}
        for i in range(1, 100):
            if i <= 33:
                species_colors[sorted_species[i]] = "tab:green"
            elif i <= 66:
                species_colors[sorted_species[i]] = "tab:orange"
            else:
                species_colors[sorted_species[i]] = "tab:red"

        plot_bar(log_sim)
        draw_tree(log_sim)

        # --- SEQUENCE SIMILARITY ---
        with open(f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}-full-stitched.json", "r") as file:
            aa_seqs = dict(json.load(file))
        aa_seqs = get_common_name(aa_seqs)
        for sp_name in common_names:
            if sp_name not in aa_seqs: 
                aa_seqs[sp_name] = ""       # Account for missing species
        sorted_aa = {sp: aa_seqs[sp] for sp in sorted_species}
        edit_dist = edit_distance(sorted_aa)
        fig = plt.figure(figsize=(18,6))
        ax = fig.add_subplot(111)
        plt.scatter(edit_dist.values(), desc_log_cos, color=[species_colors[s] for s in sorted_species])
        plt.xlabel("Levenshtein Distance")
        plt.ylabel("Log Cosine Similarity")
        plt.title(f"Edit Distance vs Cosine Similarity for {gene}")
        plt.savefig(output_dir+f"{gene}/{gene}_edit_dist.png", dpi=300, bbox_inches='tight')
        
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
        plt.title(f"L1 Distance of {gene} Full Sequence Representations")
        plt.savefig(output_dir+f"{gene}/{gene}_full_manhattan.png", dpi=300, bbox_inches='tight')

        # Cos Similarity vs Manhattan Distance
        plt.figure(figsize=(18, 6))
        plt.scatter(desc_log_cos, sorted_l1.values())
        plt.xlabel("-Log (1 - Cosine Similarity)")
        plt.ylabel("L1 Distance from hg38")
        plt.title("Cosine Similarity vs Manhattan Distance")
        plt.savefig(output_dir+f"{gene}/{gene}_logcos_vs_manhattan.png", dpi=300, bbox_inches='tight')

        # EUCLIDEAN DISTANCE (L2)
        l2_diff = get_common_name(embedding_l1_dist(sequence_representations))
        sorted_l2 = {sp: l2_diff[sp] for sp in sorted_species}
        
        # Euclidean Distance Across Species
        plt.figure(figsize=(18, 6))
        plt.bar(sorted_species, sorted_l2.values(), color=[species_colors[s] for s in sorted_species])
        plt.xticks(rotation=90)
        plt.xlabel("Species")
        plt.ylabel("L2 Distance from hg38")
        plt.title(f"L2 Distance of {gene} Full Sequence Representations")
        plt.savefig(output_dir+f"{gene}/{gene}_full_euclidean.png", dpi=300, bbox_inches='tight')

        # Cos Similarity vs Euclidean Distance
        plt.figure(figsize=(18, 6))
        plt.scatter(desc_log_cos, sorted_l2.values())
        plt.xlabel("-Log (1 - Cosine Similarity)")
        plt.ylabel("L2 Distance from hg38")
        plt.title("Cosine Similarity vs Euclidean Distance")
        plt.savefig(output_dir+f"{gene}/{gene}_logcos_vs_euclidean.png", dpi=300, bbox_inches='tight')

    elif pool_type == "exon":
        sequence_representations = torch.load(input_dir+f"embeddings/{gene}_exons.pt")      # dictionary: seq_representations[exon][species]
        
        # Check for any all-zero embeddings
        for i in range(1, num_exons + 1):
            exon_repr = sequence_representations[i]
            for sp, vec in exon_repr.items():
                if torch.all(vec == 0):
                    print(f"[Warning] Exon {i}, Species {sp}: All-zero embedding")
                elif torch.any(torch.isnan(vec)):
                    print(f"[Warning] Exon {i}, Species {sp}: Embedding has NaNs")

        # # Take weighted average of each exon and compare it to the sequence representation
        # full_seq = torch.load(input_dir + f"embeddings/{gene}_full.pt")
        # df = pd.read_csv(f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}-all-seqs.csv")
        # df = df.sort_values(by=["Species","Number"])

        # for sp in ucsc_codes:
        #     if sp not in full_seq.keys():
        #         continue
        #     total_len = 0
        #     tensor_sum = torch.zeros(1280)
        #     for exon_num in range(1, num_exons + 1):
        #         match = df[(df["Species"] == sp) & (df["Number"] == exon_num)]
        #         exon_len = 0
        #         if not match.empty:
        #             sequence = match["Seq"].iloc[0]
        #             exon_len = len(sequence.strip().replace("-",""))
        #         total_len += exon_len
        #         # Take weighted sum
        #         if exon_num in sequence_representations and sp in sequence_representations[exon_num]:
        #             tensor_sum += exon_len * (sequence_representations[exon_num][sp])
        #     avg_tensor = tensor_sum / total_len
        #     if torch.all(torch.isclose(avg_tensor, full_seq[sp])):
        #         print(f"{sp} averages match")
        #     else:
        #         print(f"*** {sp} average embeddings do not match! ***")
                

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
        masked_matrix = sim_matrix.copy()
        masked_matrix[masked_matrix==0.0] = np.nan
        log_matrix = -np.log(1 - np.clip(masked_matrix, epsilon, 1 - epsilon))
        plot_heat_map(log_matrix, similarity[1].keys())

        # --- EDIT DISTANCE ---
        if exon_to_compare:
            if exon_to_compare > num_exons or exon_to_compare <= 0:
                print("Exon does not exist")
                return
            
            with open(f"/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{gene}-exons.json", "r") as file:
                raw = json.load(file)
                all_seqs = {tuple(k): v for k, v in raw}        # Convert list [spec, #] to a tuple
            exon_seqs = {sp: all_seqs[(sp, exon_to_compare)] 
                        for (sp, ex) in all_seqs.keys()
                        if ex == exon_to_compare}
            exon_seqs = get_common_name(exon_seqs)

            found_species = list(exon_seqs.keys())
            sorted_aa = {sp: exon_seqs[sp] for sp in similarity[1].keys() if sp in found_species}
            edit_dist = edit_distance(sorted_aa)

            filtered_log = [val for sp, val in zip(similarity[1].keys(), log_matrix[exon_to_compare].tolist()) if sp in found_species]
            fig = plt.figure(figsize=(18,6))
            ax = fig.add_subplot(111)
            plt.scatter(edit_dist.values(), filtered_log)
            plt.xlabel("Levenshtein Distance")
            plt.ylabel("Log Cosine Similarity")
            plt.title(f"Edit Distance vs Cosine Similarity for {gene} Exon {exon_to_compare} ")
            plt.savefig(output_dir+f"{gene}/{gene}_{exon_to_compare}_edit_dist.png", dpi=300, bbox_inches='tight')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESM-2 model on protein sequences")
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
        help="Pooling type for sequence representation (full sequence or exon)"
    )
    parser.add_argument(
        "--exon",
        type=int,
        default=None,
        help="Specific exon to compare cosine similarity and edit distance"
    )
    args = parser.parse_args()
    gene = args.gene
    num_exons = num_exon_map[gene]
    main(args.pool_type, args.exon)
