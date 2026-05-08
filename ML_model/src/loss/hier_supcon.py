"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import pickle
import pandas as pd

class FIREDistanceBias(nn.Module):
    """
    Add adaptive bias term to phylogenetic distance calculation. Based on code from
    GPN-Star: https://www.biorxiv.org/content/10.1101/2025.09.21.677619v1
    """
    def __init__(self, max_dist, fire_hidden_size=32):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(1, fire_hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(fire_hidden_size, 1, bias=False),
        )
        self.max_dist = max_dist

    def forward(self, dist: torch.tensor) -> torch.tensor:
        """
        dist: [bsz, bsz]
        """
        c = self.c.clamp(min=0)
        dist_norm = torch.log(c * dist + 1) / torch.log(c * self.max_dist + 1)
        return self.mlp(dist_norm.unsqueeze(-1)).squeeze(-1)


class hierSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07, correlation_dir=None, 
                 weight_mode='fixed'):
        super(hierSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.correlation_dir = correlation_dir
        self.weight_mode = weight_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load species-distance weights
        species_dist_path = f"{correlation_dir}/species_branch_dist.pkl"
        with open (species_dist_path, "rb") as f:
            species_dist_df = pickle.load(f)
        
        # Map species names to an index value
        self.species_list = species_dist_df.index.tolist()
        self.species_to_idx = {sp: i for i, sp in enumerate(self.species_list)}

        # Construct tensor using species_dist_df
        matrix = torch.tensor(
            species_dist_df.loc[self.species_list, self.species_list].values
        ).float()
        # Send tensor to device upon initialization, no need in forward()
        # Save self.dist_matrix
        self.register_buffer('dist_matrix', matrix)

        if self.weight_mode == "adaptive":
            max_dist = float(matrix.max().item())
            self.fire_bias = FIREDistanceBias(max_dist)

    def compute_weights(self, anchor_species, contrast_species):
        """
        Helper function to grab necessary weights from the dist_matrix tensor.
        Modify this function for dynamic attention-based weights
        """
        anchor_idx = torch.tensor(
            [self.species_to_idx[sp] for sp in anchor_species], 
            device=self.dist_matrix.device
        )
        contrast_idx = torch.tensor(
            [self.species_to_idx[sp] for sp in contrast_species],
            device=self.dist_matrix.device
        )
        # Select rows for all anchors
        # Then all columns for the contrastive species
        dist = self.dist_matrix[anchor_idx[:, None], contrast_idx[None, :]]
        
        if self.weight_mode == "fixed":
            return dist
        elif self.weight_mode == "adaptive":
            dist = self.dist_matrix[anchor_idx[:, None], contrast_idx[None, :]]
            # Apply MLP
            return self.fire_bias(dist)
        else:
            raise ValueError(f"Unknown weight mode: {self.weight_mode}")
    
    def forward(self, features, exon_name, division, species_list, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device) # identity matrix batch_size x batch_size
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (batchsize . view) x dimension
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        """
        (AT)
        When you use torch.matmul(anchor_feature, contrast_feature.T) without normalizing the rows of anchor_feature and contrast_feature, 
        The dot product will be proportional to the magnitude (norm) of each vector. If your features are not L2 normalized, you can easily get huge numbers (hundreds or thousands).
        """
        anchor_feature = nn.functional.normalize(anchor_feature, dim=1)
        contrast_feature = nn.functional.normalize(contrast_feature, dim=1)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        ################################################################
        # MODIFICATION FOR HIERARCHICAL WEIGHTED LOSS STARTS HERE
        ################################################################

        # species_list is a list of lists
        # For each batch, there are groupings of views (pairs)
        # Ex. [ ["hg38", "mm10"], ["hg38", "panTro4"], ["hg38", "oryLat2"] ]
        
        # Collect contrasitve species
        # For each grouping (sp_list), take each species to serve as a contrastive sample
        contrast_species = [sp_list[i] for sp_list in species_list for i in range(contrast_count)]
        
        # Collect list of anchor species
        if self.contrast_mode == 'one':
            # Always just view 0
            anchor_species = [sp_list[0] for sp_list in species_list]
        elif self.contrast_mode == 'all':
            anchor_species = [sp for sp_list in species_list for sp in sp_list]
        
        weights = self.compute_weights(anchor_species, contrast_species)


        # Compute log_prob with weights
        weights = torch.nan_to_num(weights, nan=0.0)
        exp_logits = torch.exp(logits) * logits_mask
        weighted_sum_exp_logits = (weights * exp_logits).sum(1, keepdim=True)
        log_prob = logits - torch.log(weighted_sum_exp_logits + 1e-9)
        
        ################################################################
        # MODIFICATION ENDS HERE
        ################################################################

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. S
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print(f"🦀 weightedSupConLoss: {loss.item():.4f}")

        return loss