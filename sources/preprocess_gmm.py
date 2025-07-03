from torch.utils.data import DataLoader
import argparse
import os
import time
import joblib
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import numpy as np

# Import TabDS and data loading functions from your module
from tab_ds import (TabDS, 
                          build_ed_ds,
                          build_sep_ds, 
                          build_sarcos_ds, 
                          build_onp_ds, 
                          build_bf_ds, 
                          build_asc_ds)

# Set up argument parser
parser = argparse.ArgumentParser(description='Preprocess tabular data to fit a Gaussian Mixture Model for balanced training')

# Dataset and Data Loading Args
parser.add_argument('--dataset', type=str, required=True, 
                    choices=['sep', 'sarcos', 'onp', 'bf', 'asc', 'ed'],
                    help='Name of the tabular dataset to preprocess.')
parser.add_argument('--data_dir', type=str, default='/home1/jmoukpe2016/BalancedMSE/neurips2025/data', help='Root directory containing dataset subfolders.')
parser.add_argument('--train_split_name', type=str, default='training', 
                    help='Name convention for the training data file (e.g., {dataset}_{split_name}.csv)')

# Data Loading Process Args
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for loading data to collect labels.')
parser.add_argument('--workers', type=int, default=4, help='Number of workers used in data loading.')
parser.add_argument('--print_freq', type=int, default=100, help='Logging frequency.')

# Reweighting and LDS Args (passed to TabDS)
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'inverse', 'sqrt_inv'],
                    help='Cost-sensitive reweighting scheme (used by TabDS).')
parser.add_argument('--lds', action='store_true', default=False, help='Whether to enable LDS (used by TabDS).')
parser.add_argument('--lds_kernel', type=str, default='gaussian',
                    choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type.')
parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number.')
parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma.')

# GMM Specific Args
parser.add_argument('--K', type=int, default=8, help='Number of components for the Gaussian Mixture Model.')
parser.add_argument('--gmm_save_path', type=str, default='gmm_params.pkl', help='Path to save the fitted GMM parameters.')
parser.add_argument('--store_root', type=str, default='checkpoint', help='Root path for storing GMM parameters')
parser.add_argument('--store_name', type=str, default='', help='Experiment store name')


def load_tabular_data(dataset_name: str, data_dir: str, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the specified tabular dataset's training split."""
    
    # Map dataset names to their loading functions
    build_func_map = {
        'sep': build_sep_ds,
        'sarcos': build_sarcos_ds,
        'onp': build_onp_ds,
        'bf': build_bf_ds,
        'asc': build_asc_ds,
        'ed': build_ed_ds
        # Add other datasets here
    }

    # Map dataset name to its specific subfolder within data_dir
    dataset_folder_map = { 
        'sep': 'sep', # Corrected based on previous context
        'sarcos': 'sarcos',
        'onp': 'onp', # Corrected based on previous context
        'bf': 'bf', # Corrected based on previous context
        'asc': 'asc', # Corrected based on previous context
        'ed': 'ed' # Specific folder for electron delta dataset
    }

    if dataset_name not in build_func_map:
        raise ValueError(f"Dataset '{dataset_name}' loading function is not configured in build_func_map.")
        
    if dataset_name not in dataset_folder_map:
         raise ValueError(f"Dataset '{dataset_name}' folder is not configured in dataset_folder_map.")

    dataset_folder = dataset_folder_map[dataset_name]
    build_func = build_func_map[dataset_name]

    if dataset_name == 'ed':
        # Special handling for 'ed' dataset which expects a directory path
        # Assumes structure like: data_dir / dataset_folder / split_name /
        data_path = os.path.join(data_dir, dataset_folder, split_name)
        if not os.path.isdir(data_path):
             raise FileNotFoundError(f"Training data directory not found for 'ed' dataset at: {data_path}")
        print(f"Loading 'ed' data from directory: {data_path}")
        # build_ed_ds returns X, y, logI, logI_prev. We only need X and y for TabDS.
        # For GMM fitting, we actually only need y.
        X_train, y_train, _, _ = build_func(data_path, shuffle_data=False) 
        # Note: build_ed_ds parameters like apply_log, inputs_to_use etc. use defaults here.
        # If specific parameters are needed for GMM preprocessing, pass them here.
        
    else:
        # Standard handling for datasets expecting a single file path
        # Define dataset file naming conventions
        file_patterns = {
            'sep': f"sep_{split_name}.csv", # Example pattern
            'sarcos': f"sarcos_{split_name}.csv", # Example pattern
            'onp': f"onp_{split_name}.csv", # Example pattern
            'bf': f"bf_{split_name}.csv", # Example pattern
            'asc': f"asc_{split_name}.csv" # Example pattern
        }
        if dataset_name not in file_patterns:
             raise ValueError(f"Dataset '{dataset_name}' file pattern is not configured.")
             
        file_name = file_patterns[dataset_name]
        data_path = os.path.join(data_dir, dataset_folder, file_name)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data file not found at: {data_path}")

        print(f"Loading data from: {data_path}")
        # Other build_*_ds functions return X, y
        X_train, y_train = build_func(data_path, shuffle_data=False) 
    
    return X_train, y_train


def preprocess_gmm() -> None:
    """
    Preprocess a tabular dataset to fit a Gaussian Mixture Model (GMM) on the training labels.
    
    This function:
    1. Loads the specified tabular training dataset (X_train, y_train)
    2. Creates a TabDS dataset instance
    3. Uses a DataLoader to efficiently iterate and collect all labels
    4. Fits a GMM to the distribution of these labels
    5. Saves the GMM parameters (means, weights, variances) to a file
    
    The GMM can be used later for balanced training approaches.
    """
    args = parser.parse_args()
    
    # --- 1. Load Tabular Training Data ---
    print(f"Loading training data for dataset: {args.dataset}...")
    start_time = time.time()
    try:
        X_train, y_train = load_tabular_data(args.dataset, args.data_dir, args.train_split_name)
        print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f'Data loading time: {time.time() - start_time:.2f} seconds')
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return
        
    # --- 2. Create TabDS Dataset ---
    print("Creating TabDS dataset instance...")
    train_dataset = TabDS(
        X=X_train,
        y=y_train,
        reweight=args.reweight, 
        lds=args.lds, 
        lds_kernel=args.lds_kernel, 
        lds_ks=args.lds_ks, 
        lds_sigma=args.lds_sigma
        # max_target can be inferred by TabDS __init__
    )
    print("TabDS dataset created.")

    # --- 3. Collect Labels using DataLoader ---
    print('Setting up DataLoader to collect labels...')
    start_time = time.time()
    # Use DataLoader for efficient batching, even if shuffle=False is okay here
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # Shuffling is not necessary just to collect labels
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False
    )
    print(f'DataLoader setup time: {time.time() - start_time:.2f} seconds')

    print('Collecting all training labels...')
    start_time = time.time()
    all_labels: List[torch.Tensor] = []
    for _, (_, targets, _) in tqdm(enumerate(train_loader), desc="Collecting labels", total=len(train_loader)):
        all_labels.append(targets) # targets should already be tensors from TabDS.__getitem__
    
    if not all_labels:
        print("Error: No labels were collected. Check dataset and dataloader.")
        return
        
    # Concatenate all batches of labels into a single tensor
    all_labels_tensor = torch.cat(all_labels).flatten() # Flatten to 1D tensor
    print(f'Collected labels shape: {all_labels_tensor.shape}')
    print(f'Label collection time: {time.time() - start_time:.2f} seconds')
    
    # --- 4. Fit GMM ---
    print(f'Fitting GMM with {args.K} components...')
    start_time = time.time()
    gmm = GaussianMixture(
        n_components=args.K, 
        random_state=0, 
        verbose=1, # Reduced verbosity
        reg_covar=1e-6 # Add regularization for stability
    ).fit(all_labels_tensor.reshape(-1, 1).cpu().numpy()) # Reshape to (n_samples, 1)
    
    print(f'GMM fitting time: {time.time() - start_time:.2f} seconds')
    
    # --- 5. Save GMM Parameters ---
    print("Saving GMM parameters...")
    start_time = time.time()
    gmm_dict: Dict[str, Any] = {
        'means': gmm.means_,       # Component means (K, 1)
        'weights': gmm.weights_,   # Component weights (K,)
        'variances': gmm.covariances_  # Component variances (K, 1, 1)
    }
    
    # Ensure the directory exists if store_root and store_name are used
    save_dir = os.path.join(args.store_root, args.store_name) if args.store_name else args.store_root
    os.makedirs(save_dir, exist_ok=True)
    
    # Update GMM filename to include dataset information
    gmm_filename = f"{args.dataset}_gmm_K{args.K}.pkl"
    if args.gmm_save_path != 'gmm_params.pkl':  # If user provided a custom name, preserve it but add dataset info
        base_name, ext = os.path.splitext(args.gmm_save_path)
        gmm_filename = f"{base_name}_{args.dataset}_K{args.K}{ext}"
    
    full_gmm_path = os.path.join(save_dir, gmm_filename)

    joblib.dump(gmm_dict, full_gmm_path)
    
    print(f'GMM parameter saving time: {time.time() - start_time:.2f} seconds')
    print(f'GMM parameters saved to: {full_gmm_path}')


if __name__ == '__main__':
    preprocess_gmm()
