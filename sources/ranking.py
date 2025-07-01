# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Optional
import torch


def rank(seq: torch.Tensor) -> torch.Tensor:
    """
    Compute the rank of elements in a sequence.
    
    This function computes the rank of each element in the input sequence,
    where the highest value gets rank 0, second highest gets rank 1, etc.
    
    Args:
        seq (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            containing the values to be ranked.
    
    Returns:
        torch.Tensor: Tensor of the same shape as input containing ranks.
            Higher input values correspond to lower rank numbers (0-indexed).
    
    Example:
        >>> seq = torch.tensor([[3.0, 1.0, 2.0]])
        >>> rank(seq)
        tensor([[0, 2, 1]])  # 3.0 gets rank 0, 2.0 gets rank 1, 1.0 gets rank 2
    """
    # First argsort gives indices that would sort the sequence in ascending order
    # Second argsort gives the rank positions
    # flip(1) reverses the order to get descending ranks (highest value = rank 0)
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized ranks in the range [0, 1].
    
    This function computes ranks using the rank() function and then normalizes
    them to the range [0, 1], where 1/n corresponds to the highest value and
    1.0 corresponds to the lowest value (n = sequence length).
    
    Args:
        seq (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            containing the values to be ranked.
    
    Returns:
        torch.Tensor: Tensor of the same shape as input containing normalized ranks
            in the range [0, 1]. Higher input values get lower normalized ranks.
    
    Example:
        >>> seq = torch.tensor([[3.0, 1.0, 2.0]])
        >>> rank_normalised(seq)
        tensor([[0.3333, 1.0000, 0.6667]])  # Normalized to [1/3, 2/3, 1.0]
    """
    # Add 1 to convert from 0-indexed to 1-indexed ranks, then normalize by sequence length
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    """
    A differentiable ranking function using the straight-through estimator approach.
    
    This class implements a custom autograd function that allows gradients to flow
    through ranking operations. The forward pass computes normalized ranks, while
    the backward pass uses a smooth approximation to estimate gradients.
    
    The gradient estimation is based on perturbing the input sequence and computing
    the difference in ranks, scaled by the perturbation magnitude (lambda_val).
    """
    
    @staticmethod
    def forward(ctx, sequence: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        Forward pass: compute normalized ranks.
        
        Args:
            ctx: PyTorch context object for saving tensors for backward pass
            sequence (torch.Tensor): Input tensor to be ranked, shape (batch_size, seq_len)
            lambda_val (float): Perturbation magnitude for gradient estimation in backward pass
        
        Returns:
            torch.Tensor: Normalized ranks of the input sequence, same shape as input
        """
        # Compute normalized ranks for the input sequence
        rank = rank_normalised(sequence)
        
        # Save lambda value and tensors needed for backward pass
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        
        return rank

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: estimate gradients using finite differences.
        
        Since ranking is a discrete operation, we approximate gradients by:
        1. Perturbing the input sequence by lambda_val * grad_output
        2. Computing ranks for the perturbed sequence
        3. Using the difference in ranks to estimate gradients
        
        Args:
            ctx: PyTorch context object containing saved tensors and lambda_val
            grad_output (torch.Tensor): Gradient of loss w.r.t. the ranking output
        
        Returns:
            Tuple[torch.Tensor, None]: Gradient w.r.t. sequence (first element),
                                     None for lambda_val (second element, not differentiable)
        """
        # Retrieve saved tensors from forward pass
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape, "Gradient shape must match rank shape"
        
        # Perturb the input sequence using the gradient and lambda value
        sequence_prime = sequence + ctx.lambda_val * grad_output
        
        # Compute ranks for the perturbed sequence
        rank_prime = rank_normalised(sequence_prime)
        
        # Estimate gradient using finite differences
        # Negative sign because we want to move in direction that increases desired ranks
        # Small epsilon (1e-8) prevents division by zero
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        
        return gradient, None  # Return None for lambda_val as it's not differentiable
