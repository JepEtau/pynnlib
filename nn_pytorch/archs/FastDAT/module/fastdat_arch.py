import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math




# --- Helper Functions ---
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization (from timm)"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills tensor with truncated normal distribution"""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# --- Core Components ---
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class LayerNorm(nn.Module):
    """Optimized LayerNorm for different input formats"""
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.dim = (dim,)

    def forward(self, x):
        if x.dim() == 4:  # B C H W format
            if x.is_contiguous(memory_format=torch.channels_last):
                return F.layer_norm(
                    x.permute(0, 2, 3, 1), self.dim, self.weight, self.bias, self.eps
                ).permute(0, 3, 1, 2)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        else:  # Sequence format
            return F.layer_norm(x, self.dim, self.weight, self.bias, self.eps)


class DySample(nn.Module):
    """DySample upsampling module"""
    def __init__(self, in_ch, out_ch, scale_factor, style='lp', groups=4, end_convolution=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.style = style
        self.groups = groups

        if style == 'lp':
            self.offset_gen = nn.Sequential(
                nn.Conv2d(in_ch, 2 * scale_factor**2, 1),
                nn.PixelShuffle(scale_factor)
            )
        else:  # 'pl'
            self.offset_gen = nn.Sequential(
                nn.Conv2d(in_ch, in_ch * scale_factor**2, 1),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(in_ch, 2, 1)
            )

        self.scope = 1.0

        if end_convolution:
            self.end_conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        else:
            self.end_conv = None

    def forward(self, x):
        B, C, H, W = x.shape

        # Generate offsets
        offset = self.offset_gen(x) * self.scope

        # Create base grid
        h_target, w_target = H * self.scale_factor, W * self.scale_factor
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h_target, device=x.device),
            torch.linspace(-1, 1, w_target, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # Add offset to grid
        grid = grid + offset
        grid = grid.permute(0, 2, 3, 1)

        # Upsample input
        x_up = F.interpolate(x, size=(h_target, w_target), mode='bilinear', align_corners=False)

        # Apply dynamic sampling
        out = F.grid_sample(x_up, grid, mode='bilinear', padding_mode='border', align_corners=False)

        if self.end_conv is not None:
            out = self.end_conv(out)

        return out


# --- FastDAT Core Components ---
class FastSpatialWindowAttention(nn.Module):
    """Efficient Spatial Window Attention"""
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def _window_partition(self, x, H_pad, W_pad):
        """Partition into windows."""
        B, L_pad, C = x.shape
        x = x.view(B, H_pad, W_pad, C)
        x = x.view(B, H_pad // self.window_size, self.window_size,
                  W_pad // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def _window_reverse(self, windows, H_pad, W_pad, B):
        """Reverse window partition."""
        x = windows.view(B, H_pad // self.window_size, W_pad // self.window_size,
                        self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
        return x

    def forward(self, x, H, W):
        B, L, C = x.shape

        # Padding for windowing
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        H_pad, W_pad = H + pad_b, W + pad_r

        if pad_r > 0 or pad_b > 0:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            x = x.view(B, H_pad * W_pad, C)

        # Window partition
        windows = self._window_partition(x, H_pad, W_pad)

        # QKV
        qkv = self.qkv(windows).reshape(-1, self.window_size * self.window_size, 3,
                                       self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)

        # Apply attention
        attended_windows = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        attended_windows = self.proj(attended_windows)

        # Reverse window partition
        x = self._window_reverse(attended_windows, H_pad, W_pad, B)

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        return x


class FastChannelAttention(nn.Module):
    """Channel-wise Self-Attention"""
    def __init__(self, dim, num_heads=4, temperature_init=1.0, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * temperature_init)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Transpose for channel attention
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # Normalize for stable attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Channel attention
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        # Apply attention and transpose back
        out = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class AdaptiveInteraction(nn.Module):
    """Adaptive Interaction Module (AIM)"""
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()

        # Spatial interaction
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # Channel interaction
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, attn_feat, conv_feat, H, W, interaction_type):
        B, L, C = attn_feat.shape

        if interaction_type == 'spatial_modulates_channel':
            # Use attention features to generate spatial map for conv features
            attn_2d = attn_feat.transpose(1, 2).contiguous().view(B, C, H, W)
            spatial_map = self.spatial_interaction(attn_2d)
            spatial_map = spatial_map.view(B, L, 1).transpose(1, 2)

            modulated_conv = conv_feat * spatial_map.transpose(1, 2)
            return attn_feat + modulated_conv

        else:  # 'channel_modulates_spatial'
            # Use conv features to generate channel map for attention features
            conv_2d = conv_feat.transpose(1, 2).contiguous().view(B, C, H, W)
            channel_map = self.channel_interaction(conv_2d)
            channel_map = channel_map.view(B, 1, C)

            modulated_attn = attn_feat * channel_map
            return modulated_attn + conv_feat


class FastFFN(nn.Module):
    """Efficient Feed-Forward Network with Depth-wise Convolution"""
    def __init__(self, dim, expansion_ratio=2.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                               padding=1, groups=hidden_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, L, C_in = x.shape

        # First linear projection
        x = self.fc1(x)
        B, L, C_hidden = x.shape

        # Reshape for depthwise conv
        x_conv = x.transpose(1, 2).contiguous().view(B, C_hidden, H, W)
        x_conv = self.dwconv(x_conv)
        x_conv = x_conv.view(B, C_hidden, L).transpose(1, 2).contiguous()

        # Activation and second linear projection
        x = self.act(x_conv)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class FastDATBlock(nn.Module):
    """Fast DAT Block with alternating spatial/channel attention and AIM"""
    def __init__(self, dim, num_heads, window_size=8,
                 ffn_expansion_ratio=2.0, aim_reduction_ratio=8,
                 block_type='spatial', drop_path=0.0, qkv_bias=False):
        super().__init__()
        self.block_type = block_type

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Main attention branch
        if self.block_type == 'spatial':
            self.attn = FastSpatialWindowAttention(dim, window_size, num_heads, qkv_bias)
        else:  # 'channel'
            self.attn = FastChannelAttention(dim, num_heads, qkv_bias=qkv_bias)

        # Parallel convolution branch
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False),
            nn.GELU()
        )

        # Adaptive interaction module
        self.interaction = AdaptiveInteraction(dim, reduction_ratio=aim_reduction_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # FFN
        self.ffn = FastFFN(dim, expansion_ratio=ffn_expansion_ratio)

    def _conv_branch_forward(self, x, H, W):
        """Forward pass for convolution branch."""
        B, L, C = x.shape
        x_conv = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x_conv = self.conv_branch(x_conv)
        x_conv = x_conv.view(B, C, L).transpose(1, 2).contiguous()
        return x_conv

    def forward(self, x, H, W):
        # First residual connection: attention + interaction
        residual = x
        x_norm1 = self.norm1(x)

        # Main attention branch
        attn_out = self.attn(x_norm1, H, W)

        # Parallel convolution branch
        conv_out = self._conv_branch_forward(x_norm1, H, W)

        # Adaptive interaction
        if self.block_type == 'spatial':
            fused_out = self.interaction(attn_out, conv_out, H, W, 'channel_modulates_spatial')
        else:  # 'channel' block
            fused_out = self.interaction(attn_out, conv_out, H, W, 'spatial_modulates_channel')

        x = residual + self.drop_path(fused_out)

        # Second residual connection: FFN
        ffn_residual = x
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2, H, W)
        x = ffn_residual + self.drop_path(ffn_out)

        return x


class FastResidualGroup(nn.Module):
    """Residual Group containing multiple DAT blocks"""
    def __init__(self, dim, depth, num_heads, window_size, ffn_expansion_ratio,
                 aim_reduction_ratio, group_block_pattern, drop_path_rates, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_type = group_block_pattern[i % len(group_block_pattern)]
            self.blocks.append(
                FastDATBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    aim_reduction_ratio=aim_reduction_ratio,
                    block_type=block_type, drop_path=drop_path_rates[i]
                )
            )

        # Refinement convolution
        self.conv_after_group = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

    def _forward_impl(self, x, H, W):
        B, C, H_in, W_in = x.shape

        # Convert to sequence format
        x_seq = x.view(B, C, H * W).transpose(1, 2).contiguous()

        # Pass through blocks
        for block in self.blocks:
            x_seq = block(x_seq, H, W)

        # Convert back to spatial format
        x_spatial = x_seq.transpose(1, 2).contiguous().view(B, C, H, W)

        # Refinement convolution
        x_spatial = self.conv_after_group(x_spatial)
        return x_spatial

    def forward(self, x, H, W):
        residual_skip = x

        if self.use_checkpoint and self.training:
            x_out = torch.utils.checkpoint.checkpoint(self._forward_impl, x, H, W, use_reentrant=False)
        else:
            x_out = self._forward_impl(x, H, W)

        return x_out + residual_skip


class FastDAT(nn.Module):
    """Fast Dual Aggregation Transformer for Efficient Super-Resolution"""

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        embed_dim=120,
        num_groups=4,
        depth_per_group=3,
        num_heads=4,
        window_size=8,
        ffn_expansion_ratio=2.0,
        aim_reduction_ratio=8,
        group_block_pattern=['spatial', 'channel'],
        drop_path_rate=0.1,
        upscale=2,
        upsampler_type='pixelshuffle',
        img_range=1.0,
        use_checkpoint=False,
        dysample_groups=4,
        **kwargs
    ):
        super().__init__()
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.use_checkpoint = use_checkpoint

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, padding=1, bias=True)

        # 2. Deep Feature Extraction
        self.num_groups = num_groups
        actual_depth_per_group = depth_per_group * len(group_block_pattern)

        # Stochastic depth decay rule
        total_depth = num_groups * actual_depth_per_group
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.groups = nn.ModuleList()
        for i in range(num_groups):
            group_dpr = dpr[i * actual_depth_per_group:(i + 1) * actual_depth_per_group]
            self.groups.append(
                FastResidualGroup(
                    dim=embed_dim,
                    depth=actual_depth_per_group,
                    num_heads=num_heads,
                    window_size=window_size,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    aim_reduction_ratio=aim_reduction_ratio,
                    group_block_pattern=group_block_pattern,
                    drop_path_rates=group_dpr,
                    use_checkpoint=use_checkpoint
                )
            )

        # Final processing
        self.norm_after_body = LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False)

        # 3. Reconstruction
        if upsampler_type == 'dysample':
            self.upsampler = DySample(
                embed_dim, num_out_ch, upscale,
                groups=dysample_groups,
                end_convolution=True
            )
        else:  # pixelshuffle
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_ch * (upscale ** 2), 3, padding=1),
                nn.PixelShuffle(upscale)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize input
        self.mean = self.mean.type_as(x)
        x_norm = (x - self.mean) * self.img_range

        # Shallow feature extraction
        x_shallow = self.conv_first(x_norm)

        # Deep feature extraction through residual groups
        x_deep = x_shallow
        for group in self.groups:
            x_deep = group(x_deep, H, W)

        # Final processing
        x_deep = self.conv_after_body(x_deep)
        x_deep = self.norm_after_body(x_deep)

        # Global residual connection
        x_body = x_deep + x_shallow

        # Upsampling
        x_out = self.upsampler(x_body)

        # Denormalize output
        x_out = x_out / self.img_range + self.mean
        return x_out

