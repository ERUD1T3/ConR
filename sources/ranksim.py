# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple
import torch
import random
import torch.nn.functional as F

from ranking import TrueRanker, rank_normalised


def batchwise_ranking_regularizer(
    features: torch.Tensor, 
    targets: torch.Tensor, 
    lambda_val: float
) -> torch.Tensor:
    """
    Compute a ranking-based regularization loss to align feature similarities with label similarities.
    
    This function encourages the learned feature representations to have similar ranking structures
    as the label space. It computes pairwise similarities in both feature and label spaces, converts
    them to ranks, and minimizes the MSE between these rank structures.
    
    The regularizer helps ensure that samples with similar labels have similar feature representations
    and vice versa, which can improve the quality of learned embeddings.
    
    Args:
        features (torch.Tensor): Feature representations of shape (batch_size, feature_dim).
            These are typically the output of a neural network encoder.
        targets (torch.Tensor): Target labels of shape (batch_size,). Can be class indices,
            continuous values, or any comparable targets.
        lambda_val (float): Perturbation parameter for the differentiable ranking operation.
            Controls the smoothness of gradient estimation in TrueRanker.
    
    Returns:
        torch.Tensor: Scalar ranking regularization loss. Lower values indicate better
            alignment between feature and label ranking structures.
    
    Example:
        >>> features = torch.randn(8, 128)  # 8 samples, 128-dim features
        >>> targets = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1])  # Class labels
        >>> loss = batchwise_ranking_regularizer(features, targets, lambda_val=1.0)
        >>> print(f"Ranking regularization loss: {loss.item():.4f}")
    
    Note:
        - The function handles duplicate labels by sampling at most one instance per unique label
        - This reduces ties in ranking and boosts representation of infrequent labels
        - Feature similarities are computed using normalized cosine similarity
        - Label similarities are based on absolute differences (closer labels = higher similarity)
    """
    loss = torch.tensor(0.0, device=features.device, requires_grad=True)

    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)
    
    if len(batch_unique_targets) < len(targets):
        # Sample one instance per unique label to reduce ties and balance representation
        sampled_indices = []
        for target in batch_unique_targets:
            # Find all indices with this target label and randomly sample one
            target_indices = (targets == target).nonzero()[:, 0]
            sampled_indices.append(random.choice(target_indices).item())
        
        # Use only the sampled subset for regularization computation
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        # All targets are unique, use the full batch
        x = features
        y = targets

    # Compute pairwise feature similarities using normalized cosine similarity
    # Shape: (num_samples, num_samples)
    x_normalized = F.normalize(x.view(x.size(0), -1), dim=1)  # L2 normalize features
    xxt = torch.matmul(x_normalized, x_normalized.permute(1, 0))  # Cosine similarity matrix

    # Compute ranking similarity loss by comparing feature and label rank structures
    for i in range(len(y)):
        # Compute label-based ranking: samples with similar labels should have higher ranks
        # Use negative absolute difference so that smaller differences = higher similarities
        label_similarities = -torch.abs(y[i] - y).transpose(0,1)  # Shape: (num_samples,)
        label_ranks = rank_normalised(label_similarities)  # Normalize to [0,1]
        
        # Compute feature-based ranking using differentiable ranking
        feature_similarities = xxt[i].unsqueeze(dim=0)  # Shape: (1, num_samples)
        feature_ranks = TrueRanker.apply(feature_similarities, lambda_val)
        
        # Minimize MSE between feature ranks and label ranks
        # This encourages feature similarities to match label similarities
        loss += F.mse_loss(feature_ranks, label_ranks)
    
    return loss