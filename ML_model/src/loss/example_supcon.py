import hier_supcon
import torch

# model = hier_supcon.weightedSupConLoss(correlation_dir="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data")

# print(f"Num exons: {len(model.name_to_global_idx)}")
# print(f"Shape mad_matrix_tensors: {model.mad_matrix_tensors.shape}")
# print(f"Shape global_d_scores: {model.global_d_scores.shape}")
# print(f"Shape global_to_alt_idx: {model.global_to_alt_idx.shape}")

model = hier_supcon.weightedSupConLoss(debug_mode=True)

# Input must be size [bsz, n_views, ...]
features = torch.tensor([
    # Exon_0
    [
        [1, 1, 1, 1, 1],    # view 1
        [2, 2, 2, 2, 2]     # view 2
    ],
    # Exon_1
    [
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]
    ]
]).float()

names = ["exon_0", "exon_1"]

model.forward_debug(features, names, "train")
