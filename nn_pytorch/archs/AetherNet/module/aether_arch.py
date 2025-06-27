# This file defines the pure PyTorch AetherNet super-resolution architecture.
# It is designed to be a framework-agnostic "single source of truth" for the model,
# ensuring consistency and maintainability across different training and inference pipelines.

import math
import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_
from typing import Any

# This module contains the core architectural components of AetherNet.
# It is free of framework-specific imports like `neosr`, `traiNNer`, or `spandrel`.
# Quantization-Aware Training (QAT) utilities from PyTorch are included as they are
# part of the model's functionality and not specific to any external training framework.
import torch.ao.quantization as tq
from torch.ao.quantization.observer import MovingAverageMinMaxObserver


# --- Core Utility Functions and Modules ---

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Applies Stochastic Depth (DropPath) to the input tensor.
    This regularization technique randomly sets paths to zero during training,
    enhancing model robustness and preventing overfitting in deep networks.

    Args:
        x (Tensor): Input tensor. Shape (N, C, H, W) or (N, L, C) for sequence models.
        drop_prob (float): Probability of dropping a path (between 0.0 and 1.0).
        training (bool): If True, apply DropPath; otherwise, return input as is.
                         This ensures DropPath is only active during training.

    Returns:
        Tensor: Tensor with dropped paths. Elements are scaled by `1 / (1 - drop_prob)`
                      to maintain the expected sum of activations.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # Create a random tensor with shape (batch_size, 1, 1, 1) to apply independently per sample.
    # The `(1,) * (x.ndim - 1)` creates a tuple of ones for the non-batch dimensions,
    # ensuring broadcasting applies the same mask across features and spatial dims for each sample.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize to 0 or 1 (mask: 0 for dropped, 1 for kept)

    # Scale by `keep_prob` to compensate for dropped paths and maintain expected output mean.
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    A PyTorch module wrapper for the `drop_path` function.
    This allows `Stochastic Depth` to be integrated seamlessly into `nn.Sequential`
    or other `nn.ModuleList` structures within the network.
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the DropPath module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with stochastic depth applied (if in training mode).
        """
        return drop_path(x, self.drop_prob, self.training)


class ReparamLargeKernelConv(nn.Module):
    """
    A structural re-parameterization block for efficient large kernel convolutions.
    This module is designed to be trained with two parallel convolutional branches
    (a large kernel and a parallel small kernel) and then "fused" into a single,
    equivalent large kernel convolution for faster inference without performance loss.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the large kernel for the primary convolution.
        stride (int): Stride for the convolution operations.
        groups (int): Number of blocked connections from input to output channels.
                      When `groups` equals `in_channels` and `out_channels`, it performs depthwise convolution.
        small_kernel (int): Size of the small kernel for the parallel branch (e.g., 3x3).
        fused_init (bool): If True, initializes the module directly in its fused (inference-optimized) state.
                           This is typically set to `True` when loading a pre-fused model for deployment.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2 # Standard padding to maintain spatial dimensions
        self.small_kernel = small_kernel

        self.fused = fused_init # Internal flag to track the module's fusion state

        if self.fused:
            # In the fused state, only a single convolutional layer exists.
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=True
            )
            # When loading a fused model, its weights will directly populate this `fused_conv`.
        else:
            # In the unfused (training) state, two parallel convolutional branches and their biases are defined.
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups, bias=False
            )
            # Biases are handled separately and added after the convolutions in unfused mode.
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the ReparamLargeKernelConv module.

        If the module is in its `fused` state, it performs a single convolution.
        Otherwise (during training), it computes outputs from both large and small kernel branches
        and sums them along with their respective biases.

        Args:
            x (Tensor): Input feature map tensor.

        Returns:
            Tensor: Output feature map tensor.
        """
        if self.fused:
            return self.fused_conv(x)

        # Training-time forward pass: sum of large kernel, small kernel, and their biases.
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        # Biases are expanded to match the feature map dimensions for element-wise addition.
        return lk_out + self.lk_bias.view(1, -1, 1, 1) + sk_out + self.sk_bias.view(1, -1, 1, 1)

    def _fuse_kernel(self) -> tuple[Tensor, Tensor]:
        """
        Internal helper method to mathematically compute the combined kernel and bias
        from the individual large and small kernel branches. This is the core of re-parameterization.

        Raises:
            RuntimeError: If called on a module that is already in a fused state.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the `fused_kernel` and `fused_bias` tensors.
        """
        if self.fused:
            # This check acts as an internal safeguard, though `fuse()` method handles the primary check.
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")

        # Calculate padding needed to make the small kernel's size match the large kernel's size.
        pad = (self.kernel_size - self.small_kernel) // 2
        # Pad the small kernel's weights with zeros to match the dimensions of the large kernel.
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4) # [left, right, top, bottom]

        # The fused kernel is the sum of the large kernel's weights and the padded small kernel's weights.
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        # The fused bias is simply the sum of the two biases.
        fused_bias = self.lk_bias + self.sk_bias

        return fused_kernel, fused_bias

    def fuse(self):
        """
        Switches the `ReparamLargeKernelConv` module to its inference-optimized state
        by fusing the large and small kernel branches into a single convolutional layer.
        This operation modifies the module in-place. After fusion, the original unfused
        parameters (`lk_conv`, `sk_conv`, `lk_bias`, `sk_bias`) are removed to save memory
        and simplify the computational graph.
        """
        if self.fused:
            # If the module is already fused, there's no action needed.
            return

        fused_kernel, fused_bias = self._fuse_kernel() # Compute the fused parameters

        # Create the new `fused_conv` layer that replaces the two branches.
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True
        )

        # Assign the computed fused weights and biases to the new `fused_conv` layer.
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Remove the original unfused parameters from the module.
        # This is crucial for memory efficiency and to signify the fused state.
        self.__delattr__('lk_conv')
        self.__delattr__('sk_conv')
        self.__delattr__('lk_bias')
        self.__delattr__('sk_bias')

        self.fused = True # Update the internal flag to indicate the module is now fused.


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network (FFN) with GELU activation.
    This module enhances non-linear feature transformation and feature mixing
    within the AetherBlocks by introducing a gating mechanism.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to `in_features`.
        out_features (int, optional): Number of output features. Defaults to `in_features`.
        drop (float): Dropout rate applied after the first and second linear layers.
    """
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features # Default output features to input features
        hidden_features = hidden_features or in_features # Default hidden features to input features

        # First linear layer for the gating path.
        self.fc1_gate = nn.Linear(in_features, hidden_features)
        # First linear layer for the main path.
        self.fc1_main = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # GELU activation function, applied to the gate.
        self.drop1 = nn.Dropout(drop) # Dropout after the gating operation.
        self.fc2 = nn.Linear(hidden_features, out_features) # Second linear layer.
        self.drop2 = nn.Dropout(drop) # Dropout after the second linear layer.

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for GatedFFN.

        Args:
            x (Tensor): Input tensor. Expected shape (B, ..., in_features).

        Returns:
            Tensor: Output tensor after gated feed-forward operations.
                          Shape (B, ..., out_features).
        """
        gate = self.fc1_gate(x) # Compute the gate values.
        main = self.fc1_main(x) # Compute the main path values.

        # The gating mechanism: activated gate values element-wise multiply the main path values.
        x = self.act(gate) * main
        x = self.drop1(x) # Apply dropout.
        x = self.fc2(x) # Pass through the second linear layer.
        x = self.drop2(x) # Apply final dropout.
        return x


class AetherBlock(nn.Module):
    """
    The core building block of AetherNet. Each AetherBlock encapsulates:
    1. A `ReparamLargeKernelConv` for efficient global context modeling.
    2. A `LayerNorm` layer for stable training and feature normalization.
    3. A `GatedFFN` for powerful non-linear feature transformation and mixing.
    4. A residual connection with `Stochastic Depth (DropPath)` for improved training stability and generalization.

    Args:
        dim (int): Feature dimension of the block (input and output channels of the convolutional path).
        mlp_ratio (float): Ratio to determine the hidden dimension of the GatedFFN (hidden_features = dim * mlp_ratio).
        drop (float): Dropout rate for the GatedFFN.
        drop_path (float): Stochastic depth rate for the residual connection of this specific block.
        lk_kernel (int): Large kernel size for `ReparamLargeKernelConv`.
        sk_kernel (int): Small kernel size for `ReparamLargeKernelConv`.
        fused_init (bool): If True, initializes internal `ReparamLargeKernelConv` in its fused state.
    """
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3, fused_init: bool = False):
        super().__init__()

        # The re-parameterized large kernel convolution for local and global context.
        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )

        # Layer Normalization, applied before the FFN for stabilization.
        # eps is a small value added to the variance for numerical stability.
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Calculate the hidden dimension for the GatedFFN.
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GatedFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Stochastic Depth (DropPath) for the residual connection.
        # If drop_path rate is 0, it defaults to an Identity operation.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for an AetherBlock.

        Args:
            x (Tensor): Input tensor to the block. Expected shape (B, C, H, W).

        Returns:
            Tensor: Output tensor after passing through the block. Shape (B, C, H, W).
        """
        shortcut = x # Store input for the residual connection later.

        # Apply the re-parameterized convolution.
        x = self.conv(x)

        # Permute dimensions to (B, H, W, C) for Layer Normalization and FFN.
        # LayerNorm and Linear layers typically operate on the last dimension.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        # Apply the Gated Feed-Forward Network.
        x = self.ffn(x)

        # Permute dimensions back to (B, C, H, W) for subsequent convolutional layers.
        x = x.permute(0, 3, 1, 2)

        # Add the residual connection, applying stochastic depth if in training.
        x = shortcut + self.drop_path(x)

        return x


class Upsample(nn.Sequential):
    """
    Upsampling module for increasing image resolution using PixelShuffle.
    This module performs sub-pixel convolution, which is efficient and
    helps avoid checkerboard artifacts often seen with transposed convolutions.

    Args:
        scale (int): The upscale factor (e.g., 2, 3, 4).
        num_feat (int): Number of feature channels before upsampling.

    Raises:
        ValueError: If an unsupported upscale factor is provided (currently only 2, 3, 4, or powers of 2 are supported).
    """
    def __init__(self, scale: int, num_feat: int):
        m = [] # List to hold the sequential layers

        # Handle upscale factors that are powers of 2 (e.g., 2, 4, 8)
        if (scale & (scale - 1)) == 0:  # Check if scale is a power of 2
            # For each factor of 2 in the scale, add a Conv2d and PixelShuffle(2) layer.
            # E.g., for scale=4, it will do 2x upsample twice.
            for _ in range(int(math.log2(scale))):
                # Conv2d output channels are 4 * num_feat because PixelShuffle(2) divides channels by 4.
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        # Handle 3x upscale factor
        elif scale == 3:
            # For 3x, Conv2d output channels are 9 * num_feat because PixelShuffle(3) divides channels by 9.
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            # Raise an error for unsupported scales.
            raise ValueError(f"Upscale factor {scale} is not supported. Only 2, 3, 4, or powers of 2 are supported.")

        super(Upsample, self).__init__(*m) # Initialize as an nn.Sequential module


class aether(nn.Module):
    r"""
    AetherNet: A high-performance Single Image Super-Resolution (SISR) network.

    This architecture integrates structural re-parameterization (ReparamLargeKernelConv)
    and a Gated Feed-Forward Network (GatedFFN) for efficient and robust feature extraction.
    This is the core, framework-agnostic implementation of the model.

    Args:
        in_chans (int): Number of input image channels (e.g., 3 for RGB, 1 for grayscale).
        embed_dim (int): Feature dimension used throughout the network's main body.
                         This determines the width of the network.
        depths (tuple[int]): A tuple specifying the number of AetherBlocks in each
                              sequential layer group. This controls the depth of the network.
        mlp_ratio (float): Ratio to determine the hidden dimension of the FFNs.
                           `hidden_features = embed_dim * mlp_ratio`.
        drop_rate (float): Dropout rate applied within FFNs for regularization.
        drop_path_rate (float): Overall stochastic depth rate, linearly distributed across all AetherBlocks.
                                Helps prevent overfitting.
        lk_kernel (int): Large kernel size for `ReparamLargeKernelConv` (e.g., 11 or 13).
        sk_kernel (int): Small kernel size for `ReparamLargeKernelConv` (typically 3).
        scale (int): The super-resolution upscale factor (e.g., 2, 3, 4). This is a required argument.
        img_range (float): The maximum pixel value range. Images are typically normalized to [0,1]
                           or [0,255]. `img_range` is used for internal normalization/denormalization.
        fused_init (bool): If True, initializes the network with all `ReparamLargeKernelConv`
                           modules directly in their fused (inference-optimized) state.
                           This is typically used when loading a model that has already been fused.
    """
    def _init_weights(self, m: nn.Module):
        """
        Initializes weights for various module types (`nn.Linear`, `nn.LayerNorm`, `nn.Conv2d`).
        Applies truncated normal initialization to weights and sets biases to zero.
        This method is called during model initialization unless `fused_init` is True.

        Args:
            m (nn.Module): The module (layer) to initialize.
        """
        if isinstance(m, nn.Linear):
            # Truncated normal distribution is commonly used for weight initialization in Transformers/MLPs.
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # Initialize bias to zeros.
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # Bias for LayerNorm is typically zero.
            nn.init.constant_(m.weight, 1.0) # Weight for LayerNorm is typically one.
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        scale: int = 4, # This is now a required argument, with a default for easy use.
        img_range: float = 1.0,
        fused_init: bool = False, # Controls whether to initialize in fused state
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = scale # Store the effective upscale factor internally.
        self.fused_init = fused_init # Retain fused initialization state for internal logic.

        # Mean tensor for pixel normalization/denormalization.
        # This tensor will be moved to the correct device during the forward pass.
        self.mean = Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)

        # 1. Shallow feature extraction: Initial convolutional layer to embed input channels
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction: Stack of AetherBlocks
        self.num_layers = len(depths)
        self.layers = nn.ModuleList() # Use ModuleList to hold sequential groups of AetherBlocks.

        # Calculate individual stochastic depth probabilities for each block.
        # This creates a linear decay from 0 to `drop_path_rate` over all blocks,
        # distributing the regularization effect.
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        cur_drop_path_idx = 0 # Index to pick the correct drop_path rate for each block.
        for i_layer in range(self.num_layers): # Iterate through each layer group (defined by depths)
            layer_blocks = []
            for i in range(depths[i_layer]): # Iterate through blocks within the current layer group
                layer_blocks.append(AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur_drop_path_idx + i], # Assign specific drop_path rate to this block.
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=self.fused_init # Propagate `fused_init` flag down to `ReparamLargeKernelConv` within blocks.
                ))
            self.layers.append(nn.Sequential(*layer_blocks)) # Group blocks into an `nn.Sequential` module.
            cur_drop_path_idx += depths[i_layer] # Advance index for the next layer group.

        # Normalization (LayerNorm applied on the channel dimension after spatial collapse/re-expansion).
        self.norm = nn.LayerNorm(embed_dim)
        # Final convolution after the main body of AetherBlocks, before reconstruction.
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. High-quality image reconstruction
        num_feat_upsample = 64 # Fixed feature dimension before the upsampling layers.
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1),
            nn.LeakyReLU(inplace=True) # Activation before upsampling.
        )
        # Instantiate `Upsample` module using the determined upscale factor.
        self.upsample = Upsample(self.upscale, num_feat_upsample)
        # Final convolution to map features back to image channels.
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        # Apply initial weights only if the model is not initialized in a fused state.
        # If `fused_init` is True, it's assumed weights are loaded from a pre-fused checkpoint.
        if not self.fused_init:
            self.apply(self._init_weights)
        else:
            print("AetherNet initialized in fused state; skipping default weight initialization as weights are expected to be loaded.")

        # --- Quantization-Aware Training (QAT) Stubs ---
        # These stubs are inserted at the beginning and end of the quantizable part of the network.
        # They are replaced by `FakeQuantize` modules during `prepare_qat` and by
        # actual quantization operations during `convert_to_quantized`.
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the AetherNet model.

        Args:
            x (Tensor): Input low-resolution image tensor. Expected shape (B, C, H_lr, W_lr).

        Returns:
            Tensor: Output high-resolution image tensor. Expected shape (B, C, H_hr, W_hr).
        """
        # Ensure the mean tensor is on the same device and has the same data type as the input.
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        # Normalize input pixel values.
        x = (x - self.mean) * self.img_range

        # Apply QuantStub. During QAT, this will insert FakeQuantize for the input.
        x = self.quant(x)

        # Shallow feature extraction.
        x_first = self.conv_first(x)

        # Deep feature extraction through stacked AetherBlocks.
        res = x_first # Start residual connection from shallow features.
        for layer_group in self.layers: # Iterate through each layer group (Sequential of AetherBlocks).
            res = layer_group(res)

        # Permute for LayerNorm and then permute back for convolutions.
        res = res.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        res = self.norm(res)
        res = res.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

        # Apply final convolution after the main body and add residual connection.
        res = self.conv_after_body(res)
        res += x_first # Add shallow features back to deep features.

        # Upsampling and final reconstruction layers.
        x = self.conv_before_upsample(res)
        x = self.conv_last(self.upsample(x))

        # Apply DeQuantStub. During QAT, this will insert FakeQuantize for the output.
        x = self.dequant(x)

        # Denormalize output pixel values to their original range.
        return x / self.img_range + self.mean

    def fuse_model(self):
        """
        Fuses all `ReparamLargeKernelConv` modules in the network into single
        convolutional layers. This process is irreversible and is typically
        called after loading a trained unfused model to optimize it for faster
        inference. It modifies the model's structure in-place.
        """
        if self.fused_init:
            # If the model was already initialized in a fused state, no action is needed.
            print("Model already initialized in a fused state. Skipping fuse_model().")
            return

        print("Performing in-place fusion of ReparamLargeKernelConv modules...")
        for module in self.modules(): # Iterate through all sub-modules of the network.
            if isinstance(module, ReparamLargeKernelConv):
                if not module.fused: # Only fuse modules that are not already fused.
                    module.fuse() # Call the individual `fuse` method on the block.
        self.fused_init = True # Update the model's internal state to indicate fusion.
        print("Fusion complete.")

    def prepare_qat(self, opt: dict[str, Any]):
        """
        Prepares the model for Quantization-Aware Training (QAT).
        This method instruments the model with `FakeQuantize` modules at strategic points,
        allowing the model's weights and activations to adapt to quantization during training.
        It dynamically selects the `QConfig` (quantization configuration) based on the
        provided `opt` dictionary, specifically checking for `use_amp` and `bfloat16` flags.

        Args:
            opt (dict[str, Any]): A dictionary containing training options, typically from neosr's config.
                                  Expected to contain 'use_amp' and 'bfloat16' flags if QAT is conditional.

        Raises:
            RuntimeError: If `torch.ao.quantization` is not available, though this should be caught by imports.
        """
        if not hasattr(tq, 'prepare_qat'):
            raise RuntimeError("PyTorch quantization utilities (torch.ao.quantization) not available or not properly imported.")

        # Determine QConfig based on whether bfloat16 (with AMP) was enabled during training.
        # This impacts the type of observer used for activations, crucial for compatibility.
        if opt.get('use_amp', False) and opt.get('bfloat16', False):
            print("Setting QConfig for bfloat16-compatible QAT (MovingAverageMinMaxObserver for activations).")
            # For bfloat16, MovingAverageMinMaxObserver is generally preferred for activations.
            # `torch.quint8` is for 8-bit unsigned integer, `per_tensor_affine` for symmetric quantization.
            self.qconfig = tq.QConfig(
                activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
                weight=tq.default_per_channel_weight_fake_quant # Per-channel fake quantization for weights.
            )
        else:
            print("Setting QConfig for default FBGEMM QConfig (HistogramObserver for activations).")
            # `fbgemm` (Facebook Gloo Backend for Machine Learning) is a common backend for CPU/CUDA quantization.
            # Its default QConfig typically uses `HistogramObserver` for activations for better distribution modeling.
            self.qconfig = tq.get_default_qconfig("fbgemm")

        self.train() # Model must be in training mode for QAT preparation.

        # Apply `prepare_qat` to the model. This inserts the `FakeQuantize` modules
        # based on the configured `self.qconfig` and the inserted `QuantStub`/`DeQuantStub`.
        tq.prepare_qat(self, inplace=True)

        print("AetherNet has been prepared for Quantization-Aware Training (QAT).")

    def convert_to_quantized(self):
        """
        Converts the QAT-prepared (and trained) model into a truly quantized model (e.g., INT8).
        This method replaces the `FakeQuantize` modules with actual quantized operations (e.g., `quantized::linear`).
        It should be called *after* QAT training is complete, when the model weights have adapted.

        Returns:
            torch.nn.Module: The fully quantized version of the AetherNet model. This is a new model instance.

        Raises:
            RuntimeError: If `torch.ao.quantization` is not available or if `qconfig` was not set.
        """
        if not hasattr(tq, 'convert'):
            raise RuntimeError("PyTorch quantization utilities (torch.ao.quantization) not available for conversion.")
        if not hasattr(self, 'qconfig') or self.qconfig is None:
            raise RuntimeError("QConfig is not set. Call model.prepare_qat() before attempting conversion to quantized model.")

        self.eval() # Important: Switch to evaluation mode before conversion.
                     # This ensures batch norm and dropout layers behave correctly during conversion.

        # Perform the actual conversion. `inplace=False` ensures a new model is returned,
        # leaving the original (QAT-prepared) model untouched.
        quantized_model = tq.convert(self, inplace=False)

        print("AetherNet has been converted to a truly quantized model (e.g., INT8).")
        return quantized_model