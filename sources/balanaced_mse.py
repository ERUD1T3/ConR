import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import joblib
from typing import Dict, List


class GAILoss(_Loss):
    """
    Gaussian Adaptive Importance Loss.
    
    This loss combines MSE with a balancing term based on a Gaussian Mixture Model (GMM).
    
    Args:
        init_noise_sigma (float): Initial value for the noise standard deviation.
        gmm (str): Path to the joblib file containing the GMM parameters.
        device (torch.device or int, optional): Device to use for tensors. If None, uses the default device.
    """
    def __init__(self, init_noise_sigma: float, gmm: str, device=None) -> None:
        super(GAILoss, self).__init__()
        # Determine device
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
            
        # Load GMM from file and convert to PyTorch tensors
        self.gmm = joblib.load(gmm)
        self.gmm = {k: torch.tensor(self.gmm[k], device=device) for k in self.gmm}
        # Learnable noise parameter
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the GAI loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        noise_var = self.noise_sigma ** 2
        loss = gai_loss(pred, target, self.gmm, noise_var)
        return loss


def gai_loss(
        pred: torch.Tensor, 
        target: torch.Tensor, 
        gmm: Dict[str, torch.Tensor], 
        noise_var: torch.Tensor
    ) -> torch.Tensor:
    """
    Gaussian Adaptive Importance loss function.
    
    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.
        gmm (Dict[str, torch.Tensor]): Gaussian Mixture Model parameters.
        noise_var (torch.Tensor): Noise variance.
        
    Returns:
        torch.Tensor: Computed loss value.
    """
    # Expand GMM parameters to match batch size
    gmm = {k: gmm[k].reshape(1, -1).expand(pred.shape[0], -1) for k in gmm}
    
    # MSE term with noise normalization
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var + 0.5 * noise_var.log()
    
    # Balancing term based on GMM
    sum_var = gmm['variances'] + noise_var
    
    # FIX: Reshape pred to [batch_size, 1] for proper broadcasting with gmm['means']
    pred_reshaped = pred.view(-1, 1)
    
    balancing_term = - 0.5 * sum_var.log() - 0.5 * (pred_reshaped - gmm['means']).pow(2) / sum_var + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    
    # Combine terms and apply scaling
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()  # Scale by noise variance but detach to avoid affecting gradients

    return loss.mean()


class BMCLoss(_Loss):
    """
    Balanced MSE via Contrastive learning (BMC) Loss.
    
    Args:
        init_noise_sigma (float): Initial value for the noise standard deviation.
    """
    def __init__(self, init_noise_sigma: float) -> None:
        super(BMCLoss, self).__init__()
        # Learnable noise parameter
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the BMC loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred: torch.Tensor, target: torch.Tensor, noise_var: torch.Tensor) -> torch.Tensor:
    """
    Balanced MSE via Contrastive learning loss function.
    
    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.
        noise_var (torch.Tensor): Noise variance.
        
    Returns:
        torch.Tensor: Computed loss value.
    """
    # Compute pairwise distances as logits
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    
    # Cross entropy loss with identity matrix as targets (diagonal elements should match)
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    
    # Scale by noise variance but detach to avoid affecting gradients
    loss = loss * (2 * noise_var).detach()

    return loss


class BNILoss(_Loss):
    """
    Balanced Noise Injection (BNI) Loss.
    
    Args:
        init_noise_sigma (float): Initial value for the noise standard deviation.
        bucket_centers (List[float]): Centers of the buckets for balancing.
        bucket_weights (List[float]): Weights of the buckets for balancing.
    """
    def __init__(self, init_noise_sigma: float, bucket_centers: List[float], 
                 bucket_weights: List[float]) -> None:
        super(BNILoss, self).__init__()
        # Learnable noise parameter
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))
        # Bucket parameters for balancing
        self.bucket_centers = torch.tensor(bucket_centers).cuda()
        self.bucket_weights = torch.tensor(bucket_weights).cuda()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the BNI loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        noise_var = self.noise_sigma ** 2
        loss = bni_loss(pred, target, noise_var, self.bucket_centers, self.bucket_weights)
        return loss


def bni_loss(pred: torch.Tensor, target: torch.Tensor, noise_var: torch.Tensor,
             bucket_centers: torch.Tensor, bucket_weights: torch.Tensor) -> torch.Tensor:
    """
    Balanced Noise Injection loss function.
    
    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Target values.
        noise_var (torch.Tensor): Noise variance.
        bucket_centers (torch.Tensor): Centers of the buckets for balancing.
        bucket_weights (torch.Tensor): Weights of the buckets for balancing.
        
    Returns:
        torch.Tensor: Computed loss value.
    """
    # MSE term with noise normalization
    mse_term = F.mse_loss(pred, target, reduction='none') / 2 / noise_var

    # Prepare bucket parameters
    num_bucket = bucket_centers.shape[0]
    bucket_center = bucket_centers.unsqueeze(0).repeat(pred.shape[0], 1)
    bucket_weights = bucket_weights.unsqueeze(0).repeat(pred.shape[0], 1)

    # Balancing term based on buckets
    balancing_term = - 0.5 * (pred.expand(-1, num_bucket) - bucket_center).pow(2) / noise_var + bucket_weights.log()
    balancing_term = torch.logsumexp(balancing_term, dim=-1, keepdim=True)
    
    # Combine terms and apply scaling
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()  # Scale by noise variance but detach to avoid affecting gradients
    
    return loss.mean()