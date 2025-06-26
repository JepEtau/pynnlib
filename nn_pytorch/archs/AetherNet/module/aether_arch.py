# aether_arch.py
# This file defines the AetherNet super-resolution architecture
# for integration with the neosr training framework.

import math
import torch
from torch import nn
from torch.nn.init import trunc_normal_

# neosr-specific imports for model registration and global options.
# These imports are essential for neosr to discover and configure the model.
# from neosr.archs.arch_util import net_opt, to_2tuple
# from neosr.utils.registry import ARCH_REGISTRY

# Retrieve the global upscale factor from neosr's configuration.
# This ensures consistency between the dataset's scale and the model's output.
# upscale_opt, __ = net_opt()


# --- Core Utility Functions and Modules ---

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Applies Stochastic Depth (DropPath) to the input tensor.
    This regularization technique randomly sets paths to zero during training,
    enhancing model robustness.
    Args:
        x (torch.Tensor): Input tensor.
        drop_prob (float): Probability of dropping a path.
        training (bool): If True, apply dropout; otherwise, return input as is.
    Returns:
        torch.Tensor: Tensor with dropped paths.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # Create a random tensor with shape (batch_size, 1, 1, 1) to apply independently per sample.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # Binarize to 0 or 1
    output = x.div(keep_prob) * random_tensor  # Scale by keep_prob to maintain expectation
    return output


class DropPath(nn.Module):
    """
    A PyTorch module wrapper for the drop_path function.
    Integrates Stochastic Depth into nn.Sequential or other module lists.
    """
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ReparamLargeKernelConv(nn.Module):
    """
    A structural re-parameterization block for efficient large kernel convolutions.
    It comprises a large kernel convolution and a parallel small kernel convolution.
    During inference, these two branches are fused into a single large convolution
    to accelerate computation without losing performance.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the large kernel.
        stride (int): Stride for the convolution.
        groups (int): Number of blocked connections from input to output channels.
                      Typically equals `in_channels` for depthwise convolution.
        small_kernel (int): Size of the small kernel for the parallel branch.
        fused_init (bool): If True, initializes the module directly in its fused state.
                           Useful when loading pre-fused models.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, groups: int, small_kernel: int, fused_init: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        self.small_kernel = small_kernel

        self.fused = fused_init

        if self.fused:
            # In fused mode, only the combined convolution is present.
            self.fused_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=True
            )
        else:
            # In unfused (training) mode, both large and small kernel branches exist.
            self.lk_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False
            )
            self.sk_conv = nn.Conv2d(
                in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups, bias=False
            )
            self.lk_bias = nn.Parameter(torch.zeros(out_channels))
            self.sk_bias = nn.Parameter(torch.zeros(out_channels))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ReparamLargeKernelConv module.
        If fused, uses the single fused convolution; otherwise, computes
        outputs from both branches and sums them.
        """
        if self.fused:
            return self.fused_conv(x)

        # Training-time forward pass: sum of large kernel, small kernel, and biases
        lk_out = self.lk_conv(x)
        sk_out = self.sk_conv(x)
        return lk_out + self.lk_bias.view(1, -1, 1, 1) + sk_out + self.sk_bias.view(1, -1, 1, 1)

    def _fuse_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method to compute the fused kernel and bias from the unfused branches.
        Pads the small kernel to match the large kernel size before summing.
        """
        if self.fused:
            raise RuntimeError("Cannot fuse an already fused ReparamLargeKernelConv module.")

        pad = (self.kernel_size - self.small_kernel) // 2
        # Pad the small kernel's weights to the size of the large kernel
        sk_kernel_padded = torch.nn.functional.pad(self.sk_conv.weight, [pad] * 4)

        # Sum the kernels and biases
        fused_kernel = self.lk_conv.weight + sk_kernel_padded
        fused_bias = self.lk_bias + self.sk_bias

        return fused_kernel, fused_bias

    def fuse(self):
        """
        Switches the module to inference mode by fusing the large and small kernel branches
        into a single convolutional layer. This operation modifies the module in-place.
        It removes the original unfused parameters to save memory and improve speed.
        """
        if self.fused:
            # If already fused, do nothing
            return

        fused_kernel, fused_bias = self._fuse_kernel()

        # Create the fused convolution layer
        self.fused_conv = nn.Conv2d(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, groups=self.groups, bias=True
        )

        # Assign the fused weights and biases
        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Remove the unfused components to free up memory
        self.__delattr__('lk_conv')
        self.__delattr__('sk_conv')
        self.__delattr__('lk_bias')
        self.__delattr__('sk_bias')

        self.fused = True


class GatedFFN(nn.Module):
    """
    Gated Feed-Forward Network (FFN) with GELU activation.
    This module enhances feature mixing within the AetherBlocks.
    """
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Two linear layers, one for gating and one for the main path
        self.fc1_gate = nn.Linear(in_features, hidden_features)
        self.fc1_main = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # GELU activation
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for GatedFFN."""
        gate = self.fc1_gate(x)
        main = self.fc1_main(x)
        # Element-wise multiplication after activation creates the gating mechanism
        x = self.act(gate) * main
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AetherBlock(nn.Module):
    """
    The core building block of AetherNet. Each block combines:
    1. A ReparamLargeKernelConv for efficient global context modeling.
    2. A Layer Normalization for stable training.
    3. A Gated FFN for non-linear feature transformation.
    4. A residual connection with stochastic depth.

    Args:
        dim (int): Feature dimension of the block.
        mlp_ratio (float): Ratio to determine hidden dimension for the FFN.
        drop (float): Dropout rate for FFN.
        drop_path (float): Stochastic depth rate for the residual connection.
        lk_kernel (int): Large kernel size for ReparamLargeKernelConv.
        sk_kernel (int): Small kernel size for ReparamLargeKernelConv.
        fused_init (bool): If True, initializes internal convolutions in fused state.
    """
    def __init__(self, dim: int, mlp_ratio: float = 2.0, drop: float = 0.,
                 drop_path: float = 0., lk_kernel: int = 11, sk_kernel: int = 3, fused_init: bool = False):
        super().__init__()

        self.conv = ReparamLargeKernelConv(
            in_channels=dim, out_channels=dim, kernel_size=lk_kernel,
            stride=1, groups=dim, small_kernel=sk_kernel, fused_init=fused_init
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GatedFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Apply DropPath if drop_path rate is positive, otherwise use identity
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for AetherBlock."""
        shortcut = x # Store input for residual connection

        # Apply convolution
        x = self.conv(x)

        # Permute for LayerNorm and FFN (which operate on last dimension)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)

        # Apply FFN
        x = self.ffn(x)

        # Permute back to (B, C, H, W) for subsequent convolutional layers
        x = x.permute(0, 3, 1, 2)

        # Add residual connection with stochastic depth
        x = shortcut + self.drop_path(x)

        return x


class Upsample(nn.Sequential):
    """
    Upsampling module for increasing image resolution using PixelShuffle.
    Supports 2x, 3x, and 4x upsampling.

    Args:
        scale (int): The upscale factor (e.g., 2, 3, 4).
        num_feat (int): Number of feature channels to upsample.
    Raises:
        ValueError: If an unsupported upscale factor is provided.
    """
    def __init__(self, scale: int, num_feat: int):
        m = []
        if (scale & (scale - 1)) == 0:  # Check if scale is a power of 2 (2, 4, 8...)
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2)) # Each PixelShuffle doubles resolution (2x)
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3)) # PixelShuffle(3) performs 3x upsampling
        else:
            raise ValueError(f"Upscale factor {scale} is not supported. Only 2, 3, 4, 8... are supported.")
        super(Upsample, self).__init__(*m)


class AetherNet(nn.Module):
    r"""
    AetherNet: A high-performance Single Image Super-Resolution (SISR) network.

    This architecture features:
    - Shallow feature extraction using an initial convolution.
    - Deep feature extraction via a stack of AetherBlocks, which leverage
      structural re-parameterization for efficient large kernel convolutions
      and Gated Feed-Forward Networks for robust feature mixing.
    - High-quality image reconstruction using PixelShuffle for upsampling.

    Args:
        in_chans (int): Number of input image channels (e.g., 3 for RGB).
        embed_dim (int): Feature dimension used throughout the network's main body.
        depths (tuple[int]): A tuple specifying the number of AetherBlocks in each
                              sequential layer group. Controls model depth.
        mlp_ratio (float): Ratio to determine the hidden dimension of the FFNs.
        drop_rate (float): Dropout rate applied within FFNs.
        drop_path_rate (float): Overall stochastic depth rate, distributed across blocks.
        lk_kernel (int): Kernel size for the large kernel convolution in ReparamLargeKernelConv.
        sk_kernel (int): Kernel size for the small kernel convolution in ReparamLargeKernelConv.
        scale (int): The super-resolution upscale factor (e.g., 2, 3, 4). This parameter name
                     is crucial for neosr to correctly pass the global 'scale' from its config.
                     Defaults to the value provided by `neosr.archs.arch_util.net_opt()`.
        img_range (float): The maximum pixel value range (e.g., 1.0 for [0,1] normalization).
        fused_init (bool): If True, initializes the network with fused convolutions directly,
                           suitable for loading pre-fused models for inference.
        **kwargs: Catches any additional keyword arguments passed from the neosr config
                  that are not explicitly defined here, ensuring compatibility.
    """
    def _init_weights(self, m: nn.Module):
        """
        Initializes weights for various module types (Linear, LayerNorm, Conv2d).
        Applies truncated normal initialization to weights and sets biases to zero.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 128,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6), # Default to medium variant depths
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        lk_kernel: int = 11,
        sk_kernel: int = 3,
        scale: int = 4, # Uses the global neosr 'scale' from net_opt() as default
        img_range: float = 1.0,
        fused_init: bool = False, # Controls whether to initialize in fused state
        # **kwargs, # Catch-all for extra config parameters
    ):
        super().__init__()

        self.img_range = img_range
        self.upscale = scale # Store the effective upscale factor internally
        self.fused_init = fused_init # Retain fused initialization state

        # Mean tensor for pixel normalization/denormalization.
        # It's broadcastable across spatial dimensions.
        self.mean = torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) if in_chans == 3 else torch.zeros(1, 1, 1, 1)

        # 1. Shallow feature extraction: Initial convolution to embed input channels
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction: Stack of AetherBlocks
        self.num_layers = len(depths)
        self.layers = nn.ModuleList()
        # Calculate individual stochastic depth probabilities for each block.
        # This creates a linear decay from 0 to drop_path_rate over all blocks.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur_drop_path_idx = 0
        for i_layer in range(self.num_layers):
            layer_blocks = []
            for i in range(depths[i_layer]):
                layer_blocks.append(AetherBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[cur_drop_path_idx + i], # Assign specific drop_path rate
                    lk_kernel=lk_kernel,
                    sk_kernel=sk_kernel,
                    fused_init=self.fused_init # Propagate fused_init to sub-modules
                ))
            self.layers.append(nn.Sequential(*layer_blocks))
            cur_drop_path_idx += depths[i_layer]

        # Normalization (LayerNorm on the channel dimension after spatial collapse)
        self.norm = nn.LayerNorm(embed_dim)
        # Final convolution before the reconstruction head
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. High-quality image reconstruction
        num_feat_upsample = 64 # Fixed feature dimension before upsampling
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat_upsample, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        # Instantiate Upsample module using the determined upscale factor
        self.upsample = Upsample(self.upscale, num_feat_upsample)
        self.conv_last = nn.Conv2d(num_feat_upsample, in_chans, 3, 1, 1)

        # Apply initial weights only if the model is not initialized in a fused state.
        # Fused models are typically loaded from pre-trained checkpoints.
        if not self.fused_init:
            self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AetherNet model.
        Args:
            x (torch.Tensor): Input low-resolution image tensor (B, C, H_lr, W_lr).
        Returns:
            torch.Tensor: Output high-resolution image tensor (B, C, H_hr, W_hr).
        """
        # Ensure the mean tensor is on the same device and has the same data type as the input.
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
        self.mean = self.mean.type_as(x)

        # Pixel normalization
        x = (x - self.mean) * self.img_range

        # Shallow feature extraction
        x_first = self.conv_first(x)

        # Deep feature extraction through stacked AetherBlocks
        res = x_first
        for layer in self.layers:
            res = layer(res)

        # Apply LayerNorm and permute back
        # Permute to (B, H, W, C) for LayerNorm, then back to (B, C, H, W)
        res = res.permute(0, 2, 3, 1)
        res = self.norm(res)
        res = res.permute(0, 3, 1, 2)

        # Residual connection from shallow features
        res = self.conv_after_body(res)
        res += x_first

        # Upsampling and final reconstruction
        x = self.conv_before_upsample(res)
        x = self.conv_last(self.upsample(x))

        # Denormalize output pixels
        return x / self.img_range + self.mean

    def fuse_model(self):
        """
        Fuses all `ReparamLargeKernelConv` modules in the network into single
        convolutional layers. This process is irreversible and is typically
        called after loading a trained unfused model to optimize it for faster
        inference. It modifies the model's structure in-place.
        """
        if self.fused_init:
            print("Model already initialized in a fused state. Skipping fuse_model().")
            return

        print("Performing in-place fusion of ReparamLargeKernelConv modules...")
        for module in self.modules():
            if isinstance(module, ReparamLargeKernelConv):
                if not module.fused: # Only fuse if not already fused
                    module.fuse()
        self.fused_init = True # Update the model's internal state to indicate fusion
        print("Fusion complete.")

