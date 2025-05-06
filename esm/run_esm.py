"""
Loads ESM model and calculates embeddings. Protein sequences comes from
aa_exon_stitcher.py. Saves embeddings in "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt"
"""

import torch
import esm
from aa_exon_stitcher import build_full_seq

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

data = build_full_seq("/gpfs/commons/home/nkeung/data/processed_data/foxp2-aa-seqs.csv")

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = {}
for i, tokens_len in enumerate(batch_lens):
    name = batch_labels[i]
    vector = token_representations[i, 1 : tokens_len - 1].mean(0)
    sequence_representations[name] = vector

torch.save(sequence_representations, "/gpfs/commons/home/nkeung/data/processed_data/foxp2-representations.pt")
# OR convert pytorch to numpy, then save with pickle
