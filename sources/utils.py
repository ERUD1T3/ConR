import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from typing import List, Dict, Any


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    
    Args:
        name (str): Name of the metric to display.
        fmt (str): Format string for displaying the metric. Default: ':f'.
    """
    def __init__(self, name: str, fmt: str = ':f') -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics with new value.
        
        Args:
            val (float): Value to update with.
            n (int): Number of items this value represents. Default: 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """String representation of the meter."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Display training progress in a readable format.
    
    Args:
        num_batches (int): Total number of batches.
        meters (List[AverageMeter]): List of meters to display.
        prefix (str): Prefix string for the output. Default: "".
    """
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = "") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        """
        Display current batch progress.
        
        Args:
            batch (int): Current batch index.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        """
        Get the format string for batch numbers.
        
        Args:
            num_batches (int): Total number of batches.
            
        Returns:
            str: Format string for displaying batch numbers.
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def query_yes_no(question: str) -> bool:
    """ 
    Ask a yes/no question via input() and return their answer.
    
    Args:
        question (str): Question to ask the user.
        
    Returns:
        bool: True for yes, False for no.
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=':')
        choice = input().lower()
        if choice == '':
            return valid['y']
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args: Any) -> None:
    """
    Prepare output folders for the experiment.
    
    Args:
        args: Arguments containing store_root, store_name, resume, pretrained, and evaluate.
    """
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    # Check if output folder exists and handle overwriting
    if os.path.exists(folders_util[-1]) and not args.resume and not args.pretrained and not args.evaluate:
        if query_yes_no('overwrite previous folder: {} ?'.format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + ' removed.')
        else:
            raise RuntimeError('Output folder {} already exists'.format(folders_util[-1]))
    # Create folders if they don't exist
    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, args: Any) -> None:
    """
    Adjust learning rate based on schedule.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer to adjust.
        epoch (int): Current epoch.
        args: Arguments containing lr and schedule.
    """
    lr = args.lr
    # Apply learning rate decay at milestones
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    # Update learning rate for all parameter groups except noise_sigma
    for param_group in optimizer.param_groups:
        if 'name' in param_group and param_group['name'] == 'noise_sigma':
            continue
        param_group['lr'] = lr


def save_checkpoint(args: Any, state: Dict[str, Any], is_best: bool, prefix: str = '') -> None:
    """
    Save model checkpoint.
    
    Args:
        args: Arguments containing store_root and store_name.
        state (Dict[str, Any]): State dictionary to save.
        is_best (bool): Whether this is the best model so far.
        prefix (str): Prefix for the checkpoint filename. Default: ''.
    """
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def calibrate_mean_var(matrix: torch.Tensor, m1: torch.Tensor, v1: torch.Tensor, 
                       m2: torch.Tensor, v2: torch.Tensor, 
                       clip_min: float = 0.1, clip_max: float = 10) -> torch.Tensor:
    """
    Calibrate the mean and variance of a matrix.
    
    Args:
        matrix (torch.Tensor): Matrix to calibrate.
        m1 (torch.Tensor): Source mean.
        v1 (torch.Tensor): Source variance.
        m2 (torch.Tensor): Target mean.
        v2 (torch.Tensor): Target variance.
        clip_min (float): Minimum scaling factor. Default: 0.1.
        clip_max (float): Maximum scaling factor. Default: 10.
        
    Returns:
        torch.Tensor: Calibrated matrix.
    """
    # Handle edge case where source variance is too small
    if torch.sum(v1) < 1e-10:
        return matrix
    
    # Handle edge case where some source variances are zero
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    # Normal case: apply calibration to all dimensions
    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel: str, ks: int, sigma: float) -> np.ndarray:
    """
    Get kernel window for Label Distribution Smoothing (LDS).
    
    Args:
        kernel (str): Kernel type, one of ['gaussian', 'triang', 'laplace'].
        ks (int): Kernel size (should be odd).
        sigma (float): Sigma parameter for gaussian and laplace kernels.
        
    Returns:
        np.ndarray: Normalized kernel window.
    """
    assert kernel in ['gaussian', 'triang', 'laplace'], "Kernel must be one of ['gaussian', 'triang', 'laplace']"
    half_ks = (ks - 1) // 2
    
    if kernel == 'gaussian':
        # Create base kernel with a single peak in the middle
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        # Apply gaussian filter and normalize
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        # Triangular window
        kernel_window = triang(ks)
    else:  # laplace
        # Laplace distribution
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window
