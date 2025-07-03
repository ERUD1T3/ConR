import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Dict, Any, Tuple
from mlp import MLP, create_mlp
from collections import OrderedDict
from tab_ds import load_tabular_splits


def load_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a checkpoint file and return relevant components.
    
    Args:
        checkpoint_path: Path to the checkpoint (.pth.tar file)
        device: Device to load the model to ('cuda', 'cpu')
        
    Returns:
        Dictionary containing model state_dict and other checkpoint info
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove 'module.' prefix from state_dict keys if present (from DataParallel)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix
            else:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict
    
    return checkpoint


def create_model_from_checkpoint(checkpoint: Dict[str, Any], input_dim: int, dataset_name: str = None) -> torch.nn.Module:
    """
    Create and initialize a model from a checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        input_dim: Input dimension for the model
        dataset_name: Optional dataset name to use specific architecture
        
    Returns:
        Initialized PyTorch model
    """
    # For ED dataset, use specific architecture from the .sh file
    if dataset_name == "ed":
        model = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            hiddens=[2048, 128, 1024, 128, 512, 128, 256, 128],  # From .sh file
            skipped_layers=1,
            embed_dim=128,
            skip_repr=True,
            dropout=0.2,  # From .sh file
            fds=False
        )
    # Try to extract from checkpoint args, otherwise use defaults
    elif 'args' in checkpoint:
        args = checkpoint['args']
        model = create_mlp(
            input_dim=input_dim,
            output_dim=1,
            hiddens=args.get('mlp_hiddens', [100, 100, 100]),
            skipped_layers=args.get('mlp_skip_layers', 1),
            embed_dim=args.get('mlp_embed_dim', 128),
            skip_repr=args.get('mlp_skip_repr', True),
            dropout=args.get('mlp_dropout', 0.1),
            fds=args.get('fds', False),
            bucket_num=args.get('bucket_num', 100),
            bucket_start=args.get('bucket_start', 0),
            start_update=args.get('start_update', 0),
            start_smooth=args.get('start_smooth', 1),
            kernel=args.get('fds_kernel', 'gaussian'),
            ks=args.get('fds_ks', 5),
            sigma=args.get('fds_sigma', 1),
            momentum=args.get('fds_mmt', 0.9)
        )
    else:
        # Use default MLP if no configuration is available
        model = create_mlp(input_dim=input_dim, output_dim=1)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def get_predictions(model: torch.nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """
    Get predictions from a model for input data.
    
    Args:
        model: PyTorch model
        X: Input features
        device: Device to run inference on
        
    Returns:
        NumPy array of predictions
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        
        # Handle different return formats
        if isinstance(outputs, tuple):
            predictions = outputs[0]  # First element is predictions
        else:
            predictions = outputs
    
    # Convert to numpy and flatten if needed
    predictions = predictions.cpu().numpy()
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    
    return predictions


def plot_actual_vs_predicted(
    checkpoint_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    y_label: str = "Delta",
    output_dir: str = "./plots",
    filename_prefix: str = "mlp",
    device: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    dataset_name: Optional[str] = None,
    enable_fds: bool = False  # New parameter to enable FDS
) -> str:
    """
    Create and save a scatter plot of actual vs predicted values.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        X_test: Test features
        y_test: True target values
        title: Plot title
        lower_threshold: Lower threshold for highlighting rare values
        upper_threshold: Upper threshold for highlighting rare values
        y_label: Label for the y-axis
        output_dir: Directory to save the plot
        filename_prefix: Prefix for the saved file
        device: Device for inference ('cuda' or 'cpu')
        figsize: Figure size (width, height) in inches
        dataset_name: Optional dataset name to use specific architecture
        enable_fds: Whether to enable FDS in the model (for models trained with FDS)
        
    Returns:
        Path to the saved plot
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure y_test is flattened
    y_test = np.array(y_test).flatten()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Check if we need to enable FDS by looking for FDS-related keys in state_dict
    if not enable_fds:
        fds_keys = [k for k in checkpoint['state_dict'].keys() if 'fds_module' in k]
        if fds_keys:
            print(f"FDS-related keys found in checkpoint. Setting enable_fds=True.")
            enable_fds = True
    
    # Create model with appropriate FDS setting
    if dataset_name == "ed":
        model = create_mlp(
            input_dim=X_test.shape[1],
            output_dim=1,
            hiddens=[2048, 128, 1024, 128, 512, 128, 256, 128],  # From .sh file
            skipped_layers=1,
            embed_dim=128,
            skip_repr=True,
            dropout=0.2,  # From .sh file
            fds=enable_fds,  # Enable FDS if needed
            bucket_num=100,
            bucket_start=0,
            start_update=0,
            start_smooth=1,
            kernel='gaussian',
            ks=5,
            sigma=1.0,
            momentum=0.9
        )
    else:
        # Use default or args-based configuration
        if 'args' in checkpoint:
            args = checkpoint['args']
            model = create_mlp(
                input_dim=X_test.shape[1],
                output_dim=1,
                hiddens=args.get('mlp_hiddens', [100, 100, 100]),
                skipped_layers=args.get('mlp_skip_layers', 1),
                embed_dim=args.get('mlp_embed_dim', 128),
                skip_repr=args.get('mlp_skip_repr', True),
                dropout=args.get('mlp_dropout', 0.1),
                fds=enable_fds,  # Enable FDS if needed
                bucket_num=args.get('bucket_num', 100),
                bucket_start=args.get('bucket_start', 0),
                start_update=args.get('start_update', 0),
                start_smooth=args.get('start_smooth', 1),
                kernel=args.get('fds_kernel', 'gaussian'),
                ks=args.get('fds_ks', 5),
                sigma=args.get('fds_sigma', 1),
                momentum=args.get('fds_mmt', 0.9)
            )
        else:
            # Use default MLP
            model = create_mlp(input_dim=X_test.shape[1], output_dim=1, fds=enable_fds)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    
    # Get predictions
    model.eval()
    predictions = get_predictions(model, X_test, device)
    
    # Calculate metrics
    mse = np.mean((y_test - predictions) ** 2)
    pcc = np.corrcoef(y_test, predictions)[0, 1]  # Pearson correlation coefficient
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create grid
    plt.grid(True, alpha=0.5)
    
    # Fixed limits for both axes and colorbar
    min_val = -2.5
    max_val = 2.5
    
    # Create scatter plot with colormap based on actual values (not error)
    # Set explicit color limits to match axis limits
    scatter = plt.scatter(y_test, predictions, 
                         c=y_test,  # Color by actual value
                         cmap='viridis',
                         alpha=0.7, 
                         s=12,
                         vmin=min_val,  # Set minimum color value
                         vmax=max_val)  # Set maximum color value
    
    # Add perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    # Set plot title with metrics and fds info
    if title is None:
        density_info = "denseloss" if enable_fds else "denseless"
        title = f"{filename_prefix} amse{mse:.2f} apcc{pcc:.2f} {density_info} {y_label}"
    plt.title(f"{title}\ntesting_{dataset_name}_Actual_vs_Predicted_Changes")
    
    # Set axis labels
    plt.xlabel("Actual Changes")
    plt.ylabel("Predicted Changes")
    
    # Set fixed axis limits from -2.5 to 2.5
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Add colorbar with pointy ends and fixed limits
    cbar = plt.colorbar(scatter, extend='both')  # 'both' creates pointy ends at both extremes
    cbar.set_label('Actual Value')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot with FDS info in filename
    fds_tag = "_fds" if enable_fds else ""
    plot_filename = f"{dataset_name}{fds_tag}_actual_vs_predicted.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_sarcos_actual_vs_predicted(
    checkpoint_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "(a) Balanced-MSE",
    lower_threshold: float = -0.5,
    upper_threshold: float = 0.5,
    output_dir: str = "./plots",
    filename_prefix: str = "sarcos",
    device: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    enable_fds: bool = False
) -> str:
    """
    Specialized function for plotting actual vs predicted Torque_1 values for the SARCOS dataset.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        X_test: Test features
        y_test: True target values
        title: Plot title (should be simple like "(a) Balanced-MSE")
        lower_threshold: Lower threshold for rare values (-0.5 is default for SARCOS)
        upper_threshold: Upper threshold for rare values (0.5 is default for SARCOS)
        output_dir: Directory to save the plot
        filename_prefix: Prefix for the saved file
        device: Device for inference ('cuda' or 'cpu')
        figsize: Figure size (width, height) in inches
        enable_fds: Whether to enable FDS in the model
        
    Returns:
        Path to the saved plot
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure y_test is flattened
    y_test = np.array(y_test).flatten()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Check if we need to enable FDS by looking for FDS-related keys in state_dict
    if not enable_fds:
        fds_keys = [k for k in checkpoint['state_dict'].keys() if 'fds_module' in k]
        if fds_keys:
            print(f"FDS-related keys found in checkpoint. Setting enable_fds=True.")
            enable_fds = True
    
    # Create model with appropriate configuration for SARCOS
    if 'args' in checkpoint:
        args = checkpoint['args']
        # Use SARCOS-specific parameters
        hiddens = [512, 32, 256, 32, 128, 32, 64, 32]  # From MLP_HIDDENS
        embed_dim = 32  # From MLP_EMBED_DIM
        dropout = 0.2   # From MLP_DROPOUT
        
        model = create_mlp(
            input_dim=X_test.shape[1],
            output_dim=1,
            hiddens=hiddens,
            skipped_layers=args.get('mlp_skip_layers', 1),
            embed_dim=embed_dim,
            skip_repr=args.get('mlp_skip_repr', True),
            dropout=dropout,
            fds=enable_fds,
            bucket_num=args.get('bucket_num', 100),
            bucket_start=args.get('bucket_start', 0),
            start_update=args.get('start_update', 0),
            start_smooth=args.get('start_smooth', 1),
            kernel=args.get('fds_kernel', 'gaussian'),
            ks=args.get('fds_ks', 5),
            sigma=args.get('fds_sigma', 1),
            momentum=args.get('fds_mmt', 0.9)
        )
    else:
        # Use default SARCOS MLP if no configuration is available
        hiddens = [512, 32, 256, 32, 128, 32, 64, 32]
        model = create_mlp(
            input_dim=X_test.shape[1], 
            output_dim=1, 
            hiddens=hiddens,
            embed_dim=32,
            dropout=0.2,
            fds=enable_fds
        )
    
    # Load state dictionary
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    
    # Get predictions
    model.eval()
    predictions = get_predictions(model, X_test, device)
    
    # Calculate metrics
    mse = np.mean((y_test - predictions) ** 2)
    pcc = np.corrcoef(y_test, predictions)[0, 1]  # Pearson correlation coefficient
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot without error-based coloring
    scatter = ax.scatter(y_test, predictions, 
                        alpha=0.5, 
                        s=30,
                        color='blue')  # Use a single color instead of error-based coloring
    
    # Add perfect prediction line
    min_intensity = min(np.min(y_test), np.min(predictions))
    max_intensity = max(np.max(y_test), np.max(predictions))
    ax.plot([min_intensity, max_intensity], [min_intensity, max_intensity], 'k--', label='Perfect Prediction')
    
    # Add threshold lines (both red now)
    ax.axvline(lower_threshold, color='brown', linestyle='--', label='Rare Thresholds')
    ax.axhline(lower_threshold, color='brown', linestyle='--')
    
    ax.axvline(upper_threshold, color='brown', linestyle='--',)
    ax.axhline(upper_threshold, color='brown', linestyle='--')
    
    # Add labels and title specific to SARCOS with larger font size
    ax.set_xlabel('Actual Torque_1', fontsize=18)
    ax.set_ylabel('Predicted Torque_1', fontsize=18)
    
    # Set simplified title 
    # If title is provided, use it, otherwise use a default based on FDS
    plt.title(title, fontsize=18)
    
    # Add grid and legend with larger font
    ax.grid(True)
    ax.legend(fontsize=18)
    
    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    fds_tag = "_fds" if enable_fds else ""
    plot_filename = f"{filename_prefix}{fds_tag}_actual_vs_predicted.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path



# NOTE: does not work yet.
# def plot_tsne(
#     checkpoint_path: str,
#     X: np.ndarray,
#     y: np.ndarray,
#     title: str = None,
#     output_dir: str = "./plots",
#     filename_prefix: str = "mlp",
#     device: Optional[str] = None,
#     figsize: Tuple[int, int] = (18, 16),
#     dataset_name: Optional[str] = None,
#     enable_fds: bool = False,
#     seed: int = 42
# ) -> str:
#     """
#     Creates a t-SNE visualization of the model's learned feature representations.
    
#     Args:
#         checkpoint_path: Path to the model checkpoint
#         X: Input features
#         y: Target values (used for coloring points)
#         title: Plot title
#         output_dir: Directory to save the plot
#         filename_prefix: Prefix for the saved file
#         device: Device for inference ('cuda' or 'cpu')
#         figsize: Figure size (width, height) in inches
#         dataset_name: Optional dataset name to use specific architecture
#         enable_fds: Whether to enable FDS in the model
#         seed: Random seed for t-SNE initialization
        
#     Returns:
#         Path to the saved plot
#     """
#     # Import necessary libraries
#     from sklearn.manifold import TSNE
#     from scipy.spatial.distance import pdist
#     from sklearn.preprocessing import MinMaxScaler
#     from scipy.stats import pearsonr
    
#     # Set device
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Ensure y is flattened
#     y = np.array(y).flatten()
    
#     # Load checkpoint
#     checkpoint = load_checkpoint(checkpoint_path, device)
    
#     # Check if we need to enable FDS
#     if not enable_fds:
#         fds_keys = [k for k in checkpoint['state_dict'].keys() if 'fds_module' in k]
#         if fds_keys:
#             print(f"FDS-related keys found in checkpoint. Setting enable_fds=True.")
#             enable_fds = True
    
#     # Create model with appropriate FDS setting
#     if dataset_name == "ed":
#         model = create_mlp(
#             input_dim=X.shape[1],
#             output_dim=1,
#             hiddens=[2048, 128, 1024, 128, 512, 128, 256, 128],
#             skipped_layers=1,
#             embed_dim=128,
#             skip_repr=True,
#             dropout=0.2,
#             fds=enable_fds,
#             bucket_num=100,
#             bucket_start=0,
#             start_update=0,
#             start_smooth=1,
#             kernel='gaussian',
#             ks=5,
#             sigma=1.0,
#             momentum=0.9
#         )
#     else:
#         # Use default or args-based configuration
#         if 'args' in checkpoint:
#             args = checkpoint['args']
#             model = create_mlp(
#                 input_dim=X.shape[1],
#                 output_dim=1,
#                 hiddens=args.get('mlp_hiddens', [100, 100, 100]),
#                 skipped_layers=args.get('mlp_skip_layers', 1),
#                 embed_dim=args.get('mlp_embed_dim', 128),
#                 skip_repr=args.get('mlp_skip_repr', True),
#                 dropout=args.get('mlp_dropout', 0.1),
#                 fds=enable_fds,
#                 bucket_num=args.get('bucket_num', 100),
#                 bucket_start=args.get('bucket_start', 0),
#                 start_update=args.get('start_update', 0),
#                 start_smooth=args.get('start_smooth', 1),
#                 kernel=args.get('fds_kernel', 'gaussian'),
#                 ks=args.get('fds_ks', 5),
#                 sigma=args.get('fds_sigma', 1),
#                 momentum=args.get('fds_mmt', 0.9)
#             )
#         else:
#             # Use default MLP
#             model = create_mlp(input_dim=X.shape[1], output_dim=1, fds=enable_fds)
    
#     # Load state dictionary
#     model.load_state_dict(checkpoint['state_dict'], strict=True)
#     model.to(device)
#     model.eval()
    
#     # Extract features from the model
#     print("Extracting features for t-SNE...")
#     features = []
#     batch_size = 500  # Process in batches to avoid memory issues
    
#     with torch.no_grad():
#         for i in range(0, len(X), batch_size):
#             batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
#             # Get embeddings (second return value) from model
#             _, batch_features = model(batch_X)
#             features.append(batch_features.cpu().numpy())
    
#     # Combine batches
#     features = np.vstack(features)
#     print(f"Extracted features shape: {features.shape}")
    
#     # Apply t-SNE
#     print("Applying t-SNE dimensionality reduction...")
#     tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
#     tsne_result = tsne.fit_transform(features)
    
#     # Create figure with two subplots (t-SNE plot and Shepard plot)
#     fig, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})
    
#     # Plot t-SNE on the first subplot
#     plt.sca(axs[0])
    
#     # Fixed color range for Delta values
#     norm = plt.Normalize(-2.5, 2.5)
#     cmap = plt.cm.viridis  # Use viridis to be consistent with other plots
    
#     # Define thresholds for rare values
#     lower_thr, upper_thr = -0.5, 0.5
    
#     # Determine size and alpha based on value rarity
#     sizes = np.where((y > upper_thr) | (y < lower_thr), 50, 12)
#     alphas = np.where((y > upper_thr) | (y < lower_thr), 1.0, 0.3)
#     sizes = sizes.ravel()
#     alphas = alphas.ravel()
    
#     # Sort points so rare values appear on top
#     sort_order = np.argsort(sizes)
#     common_points_mask = sizes[sort_order] == 12
#     rare_points_mask = sizes[sort_order] == 50
#     common_points = sort_order[common_points_mask]
#     rare_points = sort_order[rare_points_mask]
    
#     # Plot common points first
#     sc = plt.scatter(
#         tsne_result[common_points, 0],
#         tsne_result[common_points, 1],
#         c=y[common_points],
#         cmap=cmap,
#         norm=norm,
#         s=sizes[common_points],
#         alpha=alphas[common_points]
#     )
    
#     # Plot rare points on top
#     plt.scatter(
#         tsne_result[rare_points, 0],
#         tsne_result[rare_points, 1],
#         c=y[rare_points],
#         cmap=cmap,
#         norm=norm,
#         s=sizes[rare_points],
#         alpha=alphas[rare_points]
#     )
    
#     # Add colorbar
#     cbar = plt.colorbar(sc, ax=axs[0], label='Delta Values', extend='both')
    
#     # Set title
#     if title is None:
#         density_info = "denseloss" if enable_fds else "denseless"
#         title = f"{filename_prefix} {density_info} {dataset_name}"
#     plt.title(f"{title}\nt-SNE Visualization of Feature Embeddings")
    
#     # Add Shepard plot on the second subplot
#     plt.sca(axs[1])
#     try:
#         # Calculate pairwise distances
#         print("Calculating pairwise distances for Shepard plot...")
#         distances_original = pdist(features, 'euclidean')
#         distances_tsne = pdist(tsne_result, 'euclidean')
        
#         # Normalize distances
#         scaler = MinMaxScaler()
#         distances_original_norm = scaler.fit_transform(distances_original[:, np.newaxis]).flatten()
#         distances_tsne_norm = scaler.fit_transform(distances_tsne[:, np.newaxis]).flatten()
        
#         # Calculate Pearson correlation
#         r, _ = pearsonr(distances_original_norm, distances_tsne_norm)
        
#         # Plot normalized distances
#         plt.scatter(distances_original_norm, distances_tsne_norm, alpha=0.5, s=1)
#         plt.plot([0, 1], [0, 1], 'k--')  # Perfect fit diagonal
#         plt.xlabel('Normalized Original Distances')
#         plt.ylabel('Normalized t-SNE Distances')
#         plt.title(f'Shepard Plot (correlation = {r:.2f})')
#         plt.grid(True)
#     except Exception as e:
#         print(f"Error creating Shepard plot: {str(e)}")
#         plt.title("Shepard Plot (failed to generate)")
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save the plot
#     fds_tag = "_fds" if enable_fds else ""
#     plot_filename = f"{dataset_name}{fds_tag}_tsne.png"
#     plot_path = os.path.join(output_dir, plot_filename)
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"t-SNE plot saved to: {plot_path}")
#     return plot_path


# def plot_repr_corr_dist(
#     checkpoint_path: str,
#     X: np.ndarray,
#     y: np.ndarray,
#     title: str = None,
#     output_dir: str = "./plots",
#     filename_prefix: str = "mlp",
#     device: Optional[str] = None,
#     figsize: Tuple[int, int] = (8, 6),
#     dataset_name: Optional[str] = None,
#     enable_fds: bool = False
# ) -> str:
#     """
#     Plots the correlation between distances in target values and distances in the representation space,
#     with each point colored based on the pair of labels.
    
#     Args:
#         checkpoint_path: Path to the model checkpoint
#         X: Input features 
#         y: Target values
#         title: Plot title
#         output_dir: Directory to save the plot
#         filename_prefix: Prefix for the saved file
#         device: Device for inference ('cuda' or 'cpu')
#         figsize: Figure size (width, height) in inches
#         dataset_name: Optional dataset name to use specific architecture
#         enable_fds: Whether to enable FDS in the model
        
#     Returns:
#         Path to the saved plot
#     """
#     from scipy.spatial.distance import pdist
#     from sklearn.preprocessing import MinMaxScaler
#     from scipy.stats import pearsonr
#     from matplotlib.patches import Wedge
    
#     # Set device
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Ensure y is flattened
#     y = np.array(y).flatten()
    
#     # Load checkpoint
#     checkpoint = load_checkpoint(checkpoint_path, device)
    
#     # Check if we need to enable FDS
#     if not enable_fds:
#         fds_keys = [k for k in checkpoint['state_dict'].keys() if 'fds_module' in k]
#         if fds_keys:
#             print(f"FDS-related keys found in checkpoint. Setting enable_fds=True.")
#             enable_fds = True
    
#     # Create model with appropriate FDS setting
#     if dataset_name == "ed":
#         model = create_mlp(
#             input_dim=X.shape[1],
#             output_dim=1,
#             hiddens=[2048, 128, 1024, 128, 512, 128, 256, 128],
#             skipped_layers=1,
#             embed_dim=128,
#             skip_repr=True,
#             dropout=0.2,
#             fds=enable_fds,
#             bucket_num=100,
#             bucket_start=0,
#             start_update=0,
#             start_smooth=1,
#             kernel='gaussian',
#             ks=5,
#             sigma=1.0,
#             momentum=0.9
#         )
#     else:
#         # Use default or args-based configuration
#         if 'args' in checkpoint:
#             args = checkpoint['args']
#             model = create_mlp(
#                 input_dim=X.shape[1],
#                 output_dim=1,
#                 hiddens=args.get('mlp_hiddens', [100, 100, 100]),
#                 skipped_layers=args.get('mlp_skip_layers', 1),
#                 embed_dim=args.get('mlp_embed_dim', 128),
#                 skip_repr=args.get('mlp_skip_repr', True),
#                 dropout=args.get('mlp_dropout', 0.1),
#                 fds=enable_fds,
#                 bucket_num=args.get('bucket_num', 100),
#                 bucket_start=args.get('bucket_start', 0),
#                 start_update=args.get('start_update', 0),
#                 start_smooth=args.get('start_smooth', 1),
#                 kernel=args.get('fds_kernel', 'gaussian'),
#                 ks=args.get('fds_ks', 5),
#                 sigma=args.get('fds_sigma', 1),
#                 momentum=args.get('fds_mmt', 0.9)
#             )
#         else:
#             # Use default MLP
#             model = create_mlp(input_dim=X.shape[1], output_dim=1, fds=enable_fds)
    
#     # Load state dictionary
#     model.load_state_dict(checkpoint['state_dict'], strict=True)
#     model.to(device)
#     model.eval()
    
#     # Extract representations
#     print("Extracting feature representations...")
#     representations = []
#     batch_size = 500  # Process in batches to avoid memory issues
    
#     with torch.no_grad():
#         for i in range(0, len(X), batch_size):
#             batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
#             # Get embeddings (second return value) from model
#             _, batch_features = model(batch_X)
#             representations.append(batch_features.cpu().numpy())
    
#     # Combine batches
#     representations = np.vstack(representations)
#     print(f"Extracted representations shape: {representations.shape}")
    
#     # Calculate pairwise distances
#     print("Calculating pairwise distances...")
#     distances_target = pdist(y.reshape(-1, 1), 'euclidean')
#     distances_repr = pdist(representations, 'euclidean')
    
#     # Normalize distances
#     scaler = MinMaxScaler()
#     distances_target_norm = scaler.fit_transform(distances_target.reshape(-1, 1)).flatten()
#     distances_repr_norm = scaler.fit_transform(distances_repr.reshape(-1, 1)).flatten()
    
#     # Calculate Pearson correlation
#     print("Calculating Pearson correlation...")
#     r, _ = pearsonr(distances_target_norm, distances_repr_norm)
    
#     # Define color assignment function
#     def get_color(label):
#         if label < -0.5:
#             return 'blue'
#         elif label > 0.5:
#             return 'red'
#         else:
#             return 'gray'
    
#     # Generate color pairs for all point pairs
#     print("Assigning colors based on label pairs...")
#     num_samples = len(y)
#     label_pairs = [(y[i], y[j]) for i in range(num_samples) for j in range(i + 1, num_samples)]
#     colors = [(get_color(label1), get_color(label2)) for label1, label2 in label_pairs]
    
#     # Create plot
#     print("Creating plot with half-colored dots...")
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Sample down if we have too many points (for performance)
#     max_points = 5000
#     if len(distances_target_norm) > max_points:
#         print(f"Sampling {max_points} points from {len(distances_target_norm)} total pairs...")
#         indices = np.random.choice(len(distances_target_norm), max_points, replace=False)
#         distances_target_norm = distances_target_norm[indices]
#         distances_repr_norm = distances_repr_norm[indices]
#         colors = [colors[i] for i in indices]
    
#     # Function to draw a half-colored dot
#     def draw_half_colored_dot(ax, x, y, color1, color2, size=0.01):
#         wedge1 = Wedge((x, y), size, 0, 180, color=color1)
#         wedge2 = Wedge((x, y), size, 180, 360, color=color2)
#         ax.add_patch(wedge1)
#         ax.add_patch(wedge2)
    
#     # Draw half-colored dots
#     dot_size = 0.005  # Adjust size as needed
#     for i, (x, y) in enumerate(zip(distances_target_norm, distances_repr_norm)):
#         color1, color2 = colors[i]
#         draw_half_colored_dot(ax, x, y, color1, color2, size=dot_size)
    
#     # Add perfect diagonal line
#     ax.plot([0, 1], [0, 1], 'k--')
    
#     # Add labels and title
#     ax.set_xlabel('Normalized Distance in Target Space')
#     ax.set_ylabel('Normalized Distance in Representation Space')
#     if title is None:
#         density_info = "denseloss" if enable_fds else "denseless"
#         title = f"{filename_prefix} {density_info} {dataset_name}"
#     ax.set_title(f'{title}\nRepresentation Space Correlation (pearson r= {r:.2f})')
    
#     # Add grid
#     ax.grid(True)
    
#     # Add legend to explain the colors
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='blue', edgecolor='blue', label='Label < -0.5'),
#         Patch(facecolor='gray', edgecolor='gray', label='-0.5 ≤ Label ≤ 0.5'),
#         Patch(facecolor='red', edgecolor='red', label='Label > 0.5')
#     ]
#     ax.legend(handles=legend_elements, loc='upper right')
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save the plot
#     fds_tag = "_fds" if enable_fds else ""
#     plot_filename = f"{dataset_name}{fds_tag}_repr_corr.png"
#     plot_path = os.path.join(output_dir, plot_filename)
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Representation correlation plot saved to: {plot_path}")
#     return plot_path


# 

if __name__ == "__main__":
    # Using values from the run_ed_bmse_chan.sh file for ED dataset
    try:
        # Dataset name
        dataset_name = "sarcos"
        data_dir = "./data"  # Update to your actual data path
        
        print(f"Loading {dataset_name.upper()} dataset...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_tabular_splits(
            dataset_name=dataset_name, 
            data_dir=data_dir
        )
        print(f"Data loaded. Test set shape: X={X_test.shape}, y={y_test.shape}")

        # Checkpoint path with FDS
        # C:\Users\the_3\Documents\github\BalancedMSE\neurips2025\checkpoint_rocklins\ed_mlp_gai_1.0_0.01_ed_gmm_K8_adam_mse_lr0.0001_bs2400_wd0.1_epoch6000_seed456789
        # checkpoint_path = "./checkpoint_rocklins/sarcos_mlp_gai_1.0_0.01_sarcos_gmm_K8_adam_mse_lr0.0005_bs14800_wd0.1_epoch6000_seed0/ckpt.final.pth.tar"
        checkpoint_path = "./checkpoint_rocklins/sarcos_mlp_lds_gau_5_1_fds_gau_5_1_0_1_0.9_adam_mse_lr0.0005_bs18000_wd0.1_epoch4001_seed456789/ckpt.best.pth.tar"
        print(f"Using model from: {checkpoint_path}")

        print(f"Generating plots for {dataset_name.upper()} dataset...")
        if dataset_name == "sarcos":
            print("Generating SARCOS-specific plot...")
            plot_path = plot_sarcos_actual_vs_predicted(
                checkpoint_path=checkpoint_path,
                X_test=X_test,
                y_test=y_test,
                lower_threshold=-0.5,  # SARCOS default thresholds
                upper_threshold=0.5,
                output_dir=f"./plots/{dataset_name}",
                filename_prefix="sarcos",
                title="(a) SQINV+LDS+FDS",
                enable_fds=True
            )
            print(f"SARCOS plot saved to: {plot_path}")
        else:
            plot_path = plot_actual_vs_predicted(
                checkpoint_path=checkpoint_path,
                X_test=X_test,
                y_test=y_test,
                y_label="Delta",
                output_dir=f"./plots/{dataset_name}",
                filename_prefix="mlp",
                dataset_name=dataset_name,
                enable_fds=False  # Enable FDS for this model
            )
        
        print(f"Plot saved to: {plot_path}")
        
        # # TODO: fix tsne plot colors
        # print(f"Generating t-SNE visualization for {dataset_name.upper()} dataset...")
        # tsne_path = plot_tsne(
        #     checkpoint_path=checkpoint_path,
        #     X=X_test,
        #     y=y_test,
        #     output_dir=f"./plots/{dataset_name}",
        #     filename_prefix="mlp",
        #     dataset_name=dataset_name,
        #     enable_fds=False
        # )
        # print(f"t-SNE plot saved to: {tsne_path}")
        
        # # TODO: fix crash
        # print(f"Generating representation correlation plot for {dataset_name.upper()} dataset...")
        # corr_path = plot_repr_corr_dist(
        #     checkpoint_path=checkpoint_path,
        #     X=X_test,
        #     y=y_test,
        #     output_dir=f"./plots/{dataset_name}",
        #     filename_prefix="mlp",
        #     dataset_name=dataset_name,
        #     enable_fds=False
        # )
        # print(f"Representation correlation plot saved to: {corr_path}")
        

        
    except Exception as e:
        print(f"Error: {str(e)}")
