"""
Calculate similarities between amino acid embeddings for genes
FOXP2, HLA-A, and TP53. This script will generate plots showing the 
cosine similarity, color-coded phylogenetic tree, L1 and L2 distances,
and edit distance.
"""

import os
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Set offscreen rendering for ete3
os.environ["QT_QPA_PLATFORM"] = "offscreen"

input_dir = "/gpfs/commons/home/nkeung/data/"
output_dir = "/gpfs/commons/home/nkeung/data/figures/"

epsilon = 1e-6      # Small value to aid log(0) issues


def main():
    # flat_embeddings = torch.load(input_dir+f"embeddings/hg38_exons.pt")
    flat_embeddings = torch.load(input_dir+f"embeddings/hg38_empire.pt")
    embeddings = {}
    for (gene, exon), vector in flat_embeddings.items():
        embeddings.setdefault(gene, {})[exon] = vector
    # dictionary: seq_representations[gene_code][exon_num]
    with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/rand_pairs.json", "r") as f:
        pairs = json.load(f)
    
    cos_sims = []
    exon_pairs = []
    for ex1, ex2 in pairs:
        gene1, exon1 = ex1
        gene2, exon2 = ex2

        # Account for missing exons or genes
        if gene1 not in embeddings or exon1 not in embeddings[gene1]:
            print(f"Gene {gene1}, exon {exon1} is missing!")
            continue
        elif gene2 not in embeddings or exon2 not in embeddings[gene2]:
            print(f"Gene {gene2}, exon {exon2} is missing!")
            continue
        vec1 = embeddings[gene1][exon1].reshape(1,-1)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = embeddings[gene2][exon2].reshape(1,-1)
        vec2 = vec2 / np.linalg.norm(vec2)

        cos_sims.append(cosine_similarity(vec1, vec2)[0][0])
        exon_pairs.append((f"{gene1}_{exon1}", f"{gene2}_{exon2}"))
    
    avg = sum(cos_sims) / len(cos_sims)
    print(f"Number of pairs: {len(cos_sims)}")
    print(f"Average cosine similarity: {avg}")
    print(f"Range: {min(cos_sims)} to {max(cos_sims)}")

    plt.figure(figsize=(18,6))
    plt.scatter(range(len(cos_sims)), cos_sims, c='tab:blue')
    ax = plt.gca()
    ax.set_ylim([-1, 1.2])
    plt.hlines(y=avg, xmin=0, xmax=len(cos_sims), colors='tab:red', linestyles='--')
    plt.xlabel("Comparison Number")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity Between Random Exons")
    # plt.savefig(output_dir+f"hg38_random_scatter.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir+f"hg38_empire_scatter.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(18,6))
    plt.boxplot(cos_sims)
    plt.ylabel("Cosine Similarities")
    plt.title("Distribution of Cos Similarities")
    # plt.savefig(output_dir+f"hg38_random_box.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir+f"hg38_empire_box.png", dpi=300, bbox_inches='tight')

    # Plot random exon similarities (your current baseline)
    sns.histplot(cos_sims, color='tab:blue', bins=50, kde=True)
    plt.xlabel("Cosine Similarity")
    xticks = np.arange(-0.5, 1.2, 0.1)
    xticks_labels = [f"{x:.1f}" for x in xticks]
    plt.xticks(xticks, xticks_labels)
    plt.ylabel("Count")
    plt.title("ESM Embedding Similarity Distributions")
    # plt.savefig(output_dir+f"hg38_histplot.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir+f"hg38_empire_histplot.png", dpi=300, bbox_inches='tight')


    
if __name__ == "__main__":
    main()
