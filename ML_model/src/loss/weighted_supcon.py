"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import pickle


class weightedSupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07, correlation_dir=None):
        super(weightedSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.correlation_dir = correlation_dir

    def forward(self, features, exon_name, division, labels=None, mask=None):
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
        # MODIFICATION FOR WEIGHTED LOSS STARTS HERE
        ################################################################

        corr_file_path = f"{self.correlation_dir}/ASCOT_data/{division}_ExonExon_meanAbsDist_ASCOTname.pkl"
        with open(corr_file_path, "rb") as f:
            self.correlation_df = pickle.load(f)

        # 2. For the current batch, get the corresponding correlation weights
        # Expand exon names to match the anchor and contrast dimensions
        # e.g., if batch_size=4, anchor_count=2 -> ['e1','e2','e3','e4','e1','e2','e3','e4']
        anchor_names = exon_name * anchor_count
        contrast_names = exon_name * contrast_count

        # Efficiently grab the sub-matrix of correlations using pandas
        # This creates a matrix where W[i, j] is the correlation between
        # anchor_name[i] and contrast_name[j].
        correlation_submatrix = self.correlation_df.loc[anchor_names, contrast_names]
        
        # Convert to a PyTorch tensor and move to the correct device
        weights = torch.from_numpy(correlation_submatrix.values).float().to(device)

        # Compute log_prob with the weights applied to the denominator
        exp_logits = torch.exp(logits) * logits_mask
        
        # 3. Apply the weights before summing to get the new denominator
        weighted_sum_exp_logits = (weights * exp_logits).sum(1, keepdim=True)
        
        # Add a small epsilon for numerical stability to avoid log(0)
        log_prob = logits - torch.log(weighted_sum_exp_logits + 1e-9)
        
        ################################################################
        # MODIFICATION ENDS HERE
        ################################################################

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
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
        print(f"🦀 weightedSupConLoss: {loss.item():.4f}")

        return loss