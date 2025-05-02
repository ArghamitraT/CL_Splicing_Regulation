"""
Calculate cosine similarity from embeddings saved in "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt"
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sequence_representations = torch.load("/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt")

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
