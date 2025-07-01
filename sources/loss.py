import torch
import torch.nn.functional as F
from typing import Optional, Literal


def weighted_mse_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute weighted Mean Squared Error loss.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
            
    Returns:
        torch.Tensor: Weighted MSE loss value.
    """
    # Calculate squared difference between inputs and targets
    loss = (inputs - targets) ** 2
    # Apply weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute weighted L1 (Mean Absolute Error) loss.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
            
    Returns:
        torch.Tensor: Weighted L1 loss value.
    """
    # Calculate absolute difference between inputs and targets
    loss = F.l1_loss(inputs, targets, reduction='none')
    # Apply weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: Optional[torch.Tensor] = None, 
    activate: Literal['sigmoid', 'tanh'] = 'sigmoid', 
    beta: float = 0.2, 
    gamma: float = 1
) -> torch.Tensor:
    """
    Compute weighted Focal MSE loss that focuses more on difficult examples.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
        activate (Literal['sigmoid', 'tanh']): Activation function to use for focusing.
        beta (float): Scaling factor for the focusing term.
        gamma (float): Power factor for the focusing term.
            
    Returns:
        torch.Tensor: Weighted focal MSE loss value.
    """
    # Calculate squared difference
    loss = (inputs - targets) ** 2
    
    # Apply focal weighting based on error magnitude
    error_abs = torch.abs(inputs - targets)
    if activate == 'tanh':
        # tanh-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (torch.tanh(beta * error_abs)) ** gamma
    else:  # sigmoid
        # sigmoid-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (2 * torch.sigmoid(beta * error_abs) - 1) ** gamma
    
    loss *= focal_weight
    
    # Apply sample weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: Optional[torch.Tensor] = None, 
    activate: Literal['sigmoid', 'tanh'] = 'sigmoid', 
    beta: float = 0.2, 
    gamma: float = 1
) -> torch.Tensor:
    """
    Compute weighted Focal L1 loss that focuses more on difficult examples.
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
        activate (Literal['sigmoid', 'tanh']): Activation function to use for focusing.
        beta (float): Scaling factor for the focusing term.
        gamma (float): Power factor for the focusing term.
            
    Returns:
        torch.Tensor: Weighted focal L1 loss value.
    """
    # Calculate absolute difference
    loss = F.l1_loss(inputs, targets, reduction='none')
    
    # Apply focal weighting based on error magnitude
    error_abs = torch.abs(inputs - targets)
    if activate == 'tanh':
        # tanh-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (torch.tanh(beta * error_abs)) ** gamma
    else:  # sigmoid
        # sigmoid-based focusing: scales from 0 to 1 based on error magnitude
        focal_weight = (2 * torch.sigmoid(beta * error_abs) - 1) ** gamma
    
    loss *= focal_weight
    
    # Apply sample weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: Optional[torch.Tensor] = None, 
    beta: float = 1.0
) -> torch.Tensor:
    """
    Compute weighted Huber loss (smooth L1 loss).
    
    Huber loss is less sensitive to outliers than MSE:
    - For |x| < beta, it behaves like MSE: 0.5 * x^2 / beta
    - For |x| >= beta, it behaves like L1: |x| - 0.5 * beta
    
    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element.
        beta (float): Threshold parameter that determines the transition point
                     between L1 and L2 behavior.
            
    Returns:
        torch.Tensor: Weighted Huber loss value.
    """
    # Calculate absolute difference
    l1_loss = torch.abs(inputs - targets)
    
    # Apply Huber loss formula: quadratic for small errors, linear for large errors
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    
    # Apply weights if provided
    if weights is not None:
        loss *= weights.expand_as(loss)
    
    # Return mean of all elements
    loss = torch.mean(loss)
    return loss


def weighted_coreg_loss(inputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute 1 minus the weighted Pearson Correlation Coefficient (PCC) between inputs and targets.

    Args:
        inputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.
        weights (Optional[torch.Tensor]): Optional weights for each element. If None, uniform weights are used.

    Returns:
        torch.Tensor: Scalar tensor representing 1 - PCC. Returns 1.0 if variance is zero.
    """
    if weights is None:
        weights = torch.ones_like(inputs)

    # Ensure weights have the same shape as inputs/targets for broadcasting
    weights = weights.expand_as(inputs)

    # Calculate weighted means
    # Add epsilon to avoid division by zero if sum of weights is zero (e.g., empty batch)
    epsilon = torch.finfo(weights.dtype).eps
    sum_weights = torch.sum(weights) + epsilon
    mean_inputs = torch.sum(weights * inputs) / sum_weights
    mean_targets = torch.sum(weights * targets) / sum_weights

    # Center the data
    inputs_centered = inputs - mean_inputs
    targets_centered = targets - mean_targets

    # Compute weighted covariance
    cov = torch.sum(weights * inputs_centered * targets_centered)

    # Compute weighted variances
    var_inputs = torch.sum(weights * torch.square(inputs_centered))
    var_targets = torch.sum(weights * torch.square(targets_centered))

    # Compute PCC
    std_dev_product = torch.sqrt(var_inputs * var_targets)
    
    # Handle potential zero standard deviation
    if std_dev_product < epsilon:
        # If std dev product is close to zero, correlation is undefined or meaningless.
        # Return 1.0 loss (representing zero correlation).
        return torch.tensor(1.0, device=inputs.device, dtype=inputs.dtype) 
        
    pcc = cov / (std_dev_product + epsilon) # Add epsilon for numerical stability

    # Clamp PCC to avoid potential numerical issues leading to values slightly outside [-1, 1]
    pcc = torch.clamp(pcc, -1.0, 1.0)

    return 1.0 - pcc


# ConR loss function
def ConR(
    features: torch.Tensor,
    targets: torch.Tensor, 
    preds: torch.Tensor,
    w: float = 1,
    weights: float = 1,
    t: float = 0.07,
    e: float = 0.01
) -> torch.Tensor:
    """
    Compute the ConR (Contrastive Regression) loss for learning feature representations.
    
    ConR loss encourages similar feature representations for samples with similar target values
    while pushing apart samples with dissimilar targets but similar predictions. This helps
    improve the quality of learned embeddings for regression tasks.
    
    Args:
        features (torch.Tensor): Feature embeddings of shape (batch_size, feature_dim).
            These are typically the output of a neural network encoder.
        targets (torch.Tensor): Ground truth target values of shape (batch_size,).
            Can be continuous regression targets or discrete class labels.
        preds (torch.Tensor): Model predictions of shape (batch_size,).
            These are the predicted values corresponding to the targets.
        w (float, optional): Distance threshold for determining positive/negative pairs.
            Samples with target distance <= w are considered positive pairs. Defaults to 1.
        weights (float, optional): Global weighting factor for the loss. Defaults to 1.
        t (float, optional): Temperature parameter for scaling similarity scores.
            Lower values make the softmax sharper. Defaults to 0.07.
        e (float, optional): Exponential scaling factor for pushing weights.
            Controls how much to push apart negative pairs. Defaults to 0.01.
    
    Returns:
        torch.Tensor: Scalar ConR loss value. Lower values indicate better alignment
            between feature similarities and target relationships.
    
    Note:
        - Positive pairs: Samples with similar targets (|target_i - target_j| <= w)
        - Negative pairs: Samples with dissimilar targets but similar predictions
        - The loss uses a contrastive formulation with exponential pushing weights
        - Self-pairs (i==j) are explicitly excluded from positive pairs
    """
    
    
    # Normalize feature vectors to unit length for cosine similarity computation
    # Both q (query) and k (key) are the same normalized features
    q = torch.nn.functional.normalize(features, dim=1)  # Shape: (batch_size, feature_dim)
    k = torch.nn.functional.normalize(features, dim=1)  # Shape: (batch_size, feature_dim)

    # Reshape targets and predictions for pairwise distance computation
    # l_k: targets reshaped to (1, batch_size) for broadcasting
    # l_q: targets kept as (batch_size,) for element-wise operations
    l_k = targets.flatten()[None,:]  # Shape: (1, batch_size)
    l_q = targets                    # Shape: (batch_size,)

    # p_k: predictions reshaped to (1, batch_size) for broadcasting  
    # p_q: predictions kept as (batch_size,) for element-wise operations
    p_k = preds.flatten()[None,:]    # Shape: (1, batch_size)
    p_q = preds                      # Shape: (batch_size,)
    

    # Compute pairwise distances between all samples
    # l_dist: absolute differences between target values, shape (batch_size, batch_size)
    # p_dist: absolute differences between predicted values, shape (batch_size, batch_size)
    l_dist= torch.abs(l_q - l_k)     # |target_i - target_j| for all pairs (i,j)
    p_dist= torch.abs(p_q - p_k)     # |pred_i - pred_j| for all pairs (i,j)

    
    # Define positive and negative pairs based on distance thresholds
    # Positive pairs: samples with similar targets (distance <= w)
    pos_i = l_dist.le(w)  # Shape: (batch_size, batch_size), True where |target_i - target_j| <= w
    
    # Negative pairs: samples with dissimilar targets BUT similar predictions
    # (~(l_dist.le(w))): targets are dissimilar (distance > w)
    # (p_dist.le(w)): predictions are similar (distance <= w)
    neg_i = ((~ (l_dist.le(w)))*(p_dist.le(w)))  # Shape: (batch_size, batch_size)

    # Remove self-pairs from positive pairs (diagonal elements)
    # A sample should not be considered similar to itself for contrastive learning
    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    # Compute pairwise feature similarities scaled by temperature
    # Einstein summation: "nc,kc->nk" computes dot products between all feature pairs
    prod = torch.einsum("nc,kc->nk", [q, k])/t  # Shape: (batch_size, batch_size)
    
    # Extract positive and negative similarities using the pair masks
    pos = prod * pos_i  # Positive pair similarities, zeros elsewhere
    neg = prod * neg_i  # Negative pair similarities, zeros elsewhere
    
    # Compute exponential pushing weights based on target distances
    # Larger target distances get stronger pushing (higher weights)
    pushing_w = weights*torch.exp(l_dist*e)  # Shape: (batch_size, batch_size)
    
    # Compute weighted sum of negative exponentials for each query sample
    # This represents the denominator contribution from negative pairs
    neg_exp_dot=(pushing_w*(torch.exp(neg))*neg_i).sum(1)  # Shape: (batch_size,)

    # For each query sample, if there is no negative pair, zero-out the loss.
    # This prevents division by zero and ensures meaningful contrastive learning
    no_neg_flag = (neg_i).sum(1).bool()  # Shape: (batch_size,), True if sample has negatives

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    # Count positive pairs for each query sample for normalization
    denom=pos_i.sum(1)  # Shape: (batch_size,), number of positive pairs per sample

    # Compute contrastive loss using log-softmax formulation
    # For each positive pair, compute: -log(exp(pos_sim) / (sum_pos_exp + sum_neg_exp))
    loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    
    # Apply final weighting and mask out samples without negative pairs
    # Take mean across batch for final scalar loss
    loss = (weights*(loss*no_neg_flag).unsqueeze(-1)).mean() 
    
    
    
    return loss

