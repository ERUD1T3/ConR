from fds import FDS

import torch
import torch.nn as nn
from typing import Union


class MLP(nn.Module):
    """
    A PyTorch MLP that:
      - Supports skip connections every 'skipped_layers' blocks
      - Applies batch normalization
      - Optionally applies dropout
      - Always produces a linear output for regression
      - Optionally applies FDS smoothing to the final representation
    """

    def __init__(
        self,
        input_dim: int = 100,
        output_dim: int = 1,
        hiddens: Union[list[int], None] = None,
        skipped_layers: int = 1,
        embed_dim: int = 128,
        skip_repr: bool = True,
        activation: Union[nn.Module, None] = None,
        dropout: float = 0.2,
        name: str = 'mlp',
        fds: bool = False,
        bucket_num: int = 50,
        bucket_start: int = 0,
        start_update: int = 0,
        start_smooth: int = 1,
        kernel: str = 'gaussian',
        ks: int = 5,
        sigma: float = 2.0,
        momentum: float = 0.9
    ) -> None:
        """
        Creates an MLP with optional skip connections, batch normalization, dropout, and FDS smoothing.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features (always linear regression).
            hiddens (list[int]): Sizes of hidden layers. Defaults to [50, 50] if None.
            skipped_layers (int): Frequency of residual skip connections.
            embed_dim (int): Size of the final embedding layer.
            skip_repr (bool): If True, merges a skip into the final embedding block.
            activation (nn.Module): Activation function to use, defaults to LeakyReLU if None.
            dropout (float): Dropout probability. No dropout if 0.
            name (str): Name of the model (not used internally, just for reference).
            fds (bool): If True, enables FDS smoothing on the final representation.
            bucket_num, bucket_start, start_update, start_smooth, kernel, ks, sigma, momentum:
                Parameters for FDS if enabled.
        """
        super().__init__()

        if hiddens is None:
            hiddens = [50, 50]
        if skipped_layers >= len(hiddens):
            raise ValueError(
                f"skipped_layers ({skipped_layers}) must be < number of hidden layers ({len(hiddens)})"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens
        self.skipped_layers = skipped_layers
        self.embed_dim = embed_dim
        self.skip_repr = skip_repr
        self.dropout_rate = dropout
        self.name = name
        self.activation_fn = activation if (activation is not None) else nn.LeakyReLU()

        # Set up FDS if requested
        self.fds_enabled = fds
        self.start_smooth = start_smooth
        if self.fds_enabled:
            self.fds_module = FDS(
                feature_dim=embed_dim,
                bucket_num=bucket_num,
                bucket_start=bucket_start,
                start_update=start_update,
                start_smooth=start_smooth,
                kernel=kernel,
                ks=ks,
                sigma=sigma,
                momentum=momentum
            )
        else:
            self.fds_module = None

        # Define hidden blocks as nn.Sequential modules stored in a ModuleList
        self.layers = nn.ModuleList()

        # 1) First block
        block0 = []
        block0.append(nn.Linear(input_dim, hiddens[0], bias=True))
        block0.append(nn.BatchNorm1d(hiddens[0]))
        self.layers.append(nn.Sequential(*block0))

        # 2) Additional hidden blocks
        for idx, units in enumerate(hiddens[1:], start=1):
            block = []
            block.append(nn.Linear(hiddens[idx - 1], units, bias=True))
            block.append(nn.BatchNorm1d(units))
            self.layers.append(nn.Sequential(*block))

        # 3) Final embedding block
        final_block = []
        final_block.append(nn.Linear(hiddens[-1], embed_dim, bias=True))
        final_block.append(nn.BatchNorm1d(embed_dim))
        self.final_embed = nn.Sequential(*final_block)

        # 4) Output layer (always linear)
        self.output_layer = nn.Linear(embed_dim, output_dim, bias=True)


        # Single dropout module (applied manually in forward)
        self.dropout_module = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        labels: Union[torch.Tensor, None] = None,
        epoch: Union[int, None] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            labels (torch.Tensor | None): Target labels for FDS. May be None if FDS is off.
            epoch (int | None): Current epoch for enabling FDS smoothing. May be None if FDS is off.

        Returns:
            If output_layer is defined:
                (prediction, final_repr)
            Otherwise:
                final_repr
        """
        # First hidden block
        out = self.layers[0](x)       # linear + BN
        out = self.activation_fn(out) # activation

        # Possibly skip from input
        if self.skipped_layers > 0:
            if out.shape[1] != x.shape[1]:
                # For dimension mismatch, define a small projection or do partial slice
                # Here, we do a direct linear for clarity:
                projection = nn.Linear(x.shape[1], out.shape[1], bias=False).to(x.device)
                skip_out = projection(x)
            else:
                skip_out = x

            out = out + skip_out
            # Apply dropout *after* first skip connection + activation
            if self.dropout_module is not None:
                out = self.dropout_module(out)
        else:
            # If no skip, just dropout after activation
            if self.dropout_module is not None:
                out = self.dropout_module(out)

        residual = out

        # Middle hidden blocks
        for idx in range(1, len(self.layers)):
            out = self.layers[idx](out)
            out = self.activation_fn(out)

            if self.skipped_layers > 0 and idx % self.skipped_layers == 0:
                if out.shape[1] != residual.shape[1]:
                    # Keep the original dynamic projection method
                    projection = nn.Linear(residual.shape[1], out.shape[1], bias=False).to(x.device)
                    skip_out = projection(residual)
                else:
                    skip_out = residual

                out = out + skip_out
                # Apply dropout *after* skip connection + activation
                if self.dropout_module is not None:
                    out = self.dropout_module(out)
                residual = out # Update residual state *after* dropout
            else:
                # If no skip, just dropout after activation
                if self.dropout_module is not None:
                    out = self.dropout_module(out)
                # Note: residual does not update here if no skip

        # Final embedding block
        out = self.final_embed(out) # linear + BN

        # --- Activation and Dropout Placement Logic Changed ---
        if self.skip_repr:
            # Apply activation *before* potential skip connection addition
            activated_out = self.activation_fn(out)

            if self.skipped_layers > 0:
                # Prepare skip connection from the last residual state
                if out.shape[1] != residual.shape[1]: # Compare shape *before* activation
                    projection = nn.Linear(residual.shape[1], out.shape[1], bias=False).to(x.device)
                    skip_out = projection(residual)
                else:
                    skip_out = residual

                # Apply dropout to the activated output *before* adding the skip
                if self.dropout_module is not None:
                    activated_out = self.dropout_module(activated_out)

                # Add skip connection to the *activated* output
                final_repr = activated_out + skip_out
                # NO activation after the add
            else:
                # Case: skip_repr is True, but no skip connection (skipped_layers=0)
                # Apply dropout if needed to the activated output
                if self.dropout_module is not None:
                    activated_out = self.dropout_module(activated_out)
                # Final repr is just the activated output (potentially dropout-applied)
                final_repr = activated_out
        else:
            # Case: skip_repr is False. Representation is pre-activation.
            # Apply dropout if needed to the output of final_embed
            if self.dropout_module is not None:
                out = self.dropout_module(out)
            final_repr = out # No activation applied here when skip_repr is False

        # Apply FDS smoothing if conditions are met
        if self.fds_enabled and self.training and epoch is not None and epoch >= self.start_smooth:
            if labels is None:
                raise ValueError("Labels must be provided for FDS smoothing.")
            # Apply FDS to the final representation *after* activation/addition logic
            final_repr = self.fds_module.smooth(final_repr, labels, epoch)

        # Output layer (always linear for regression)
        if self.output_layer is not None:
            preds = self.output_layer(final_repr)
            return preds, final_repr
        else:
            return final_repr


def create_mlp(
    input_dim: int = 100,
    output_dim: int = 1,
    hiddens: Union[list[int], None] = None,
    skipped_layers: int = 1,
    embed_dim: int = 128,
    skip_repr: bool = True,
    activation: Union[nn.Module, None] = None,
    dropout: float = 0.2,
    name: str = 'mlp',
    fds: bool = False,
    bucket_num: int = 50,
    bucket_start: int = 0,
    start_update: int = 0,
    start_smooth: int = 1,
    kernel: str = 'gaussian',
    ks: int = 5,
    sigma: float = 2.0,
    momentum: float = 0.9
) -> MLP:
    """
    Creates an MLP instance for regression, supporting optional skip connections,
    batch normalization, dropout, and FDS smoothing.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features (always linear).
        hiddens (list[int]): Hidden layer sizes (defaults to [50, 50] if None).
        skipped_layers (int): Skip connection frequency.
        embed_dim (int): Size of the final embedding.
        skip_repr (bool): If True, merges skip into the final embedding block.
        activation (nn.Module): Activation function, defaults to LeakyReLU if None.
        dropout (float): Dropout probability.
        name (str): Model name, unused in PyTorch but kept for reference.
        fds (bool): If True, enable FDS smoothing.
        bucket_num, bucket_start, start_update, start_smooth, kernel, ks, sigma, momentum:
            Hyperparameters for FDS.

    Returns:
        MLP: A PyTorch MLP module with the specified configuration.
    """
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hiddens=hiddens,
        skipped_layers=skipped_layers,
        embed_dim=embed_dim,
        skip_repr=skip_repr,
        activation=activation,
        dropout=dropout,
        name=name,
        fds=fds,
        bucket_num=bucket_num,
        bucket_start=bucket_start,
        start_update=start_update,
        start_smooth=start_smooth,
        kernel=kernel,
        ks=ks,
        sigma=sigma,
        momentum=momentum
    )
    return model
