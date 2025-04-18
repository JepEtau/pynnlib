from typing import Literal
import torch
import torch.nn as nn
from torch import einsum
from torch import Tensor
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from ..._shared.pad import pad, unpad

LayerNormType = Literal['BiasFree', 'WithBias']


def to(x: Tensor) -> dict[str, str | torch.dtype]:
    return {"device": x.device, "dtype": x.dtype}


def pair(x: tuple | int | float):
    return (x, x) if not isinstance(x, tuple) else x



def expand_dim(t: Tensor, dim, k) -> Tensor:
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)



def rel_to_abs(x: Tensor) -> Tensor:
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, "b l c -> b (l c)")
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x



def relative_logits_1d(q, rel_k: Tensor) -> Tensor:
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum("b x y d, r d -> b x y r", q, rel_k)
    logits = rearrange(logits, "b x y r -> (b x) y r")
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits



class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(
            torch.randn(height * 2 - 1, dim_head) * scale
        )
        self.rel_width = nn.Parameter(
            torch.randn(width * 2 - 1, dim_head) * scale
        )

    def forward(self, q: Tensor) -> Tensor:
        block = self.block_size

        q = rearrange(q, "b (x y) c -> b x y c", x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, "b x i y j-> b (x y) (i j)")

        q = rearrange(q, "b x y d -> b y x d")
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, "b x i y j -> b (y x) (j i)")
        return rel_logits_w + rel_logits_h



# Layer Norm
def to_3d(x: Tensor) -> Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x: Tensor, h: int, w: int) -> Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x: Tensor) -> Tensor:
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape


    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias



class LayerNorm(nn.Module):
    def __init__(self, dim: int, layer_norm_type: LayerNormType):
        super().__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        return to_4d(self.body(to_3d(x)), h, w)



# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float,
        bias: bool
    ):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features*2,
            hidden_features*2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features*2,
            bias=bias
        )
        self.project_out = nn.Conv2d(
            hidden_features,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)



# Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3,
            dim*3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim*3,
            bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        qkv: Tensor = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(
            out,
            "b head c (h w) -> b (head c) h w",
            head=self.num_heads,
            h=h,
            w=w
        )
        return self.project_out(out)



# Overlapping Cross-Attention (OCA)

class OCAB(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        dim_head: int,
        bias: bool
    ):
        super().__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size-window_size)//2
        )
        self.qkv = nn.Conv2d(
            self.dim, self.inner_dim*3, kernel_size=1, bias=bias
        )
        self.project_out = nn.Conv2d(
            self.inner_dim, dim, kernel_size=1, bias=bias
        )
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )


    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv: Tensor = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(
            qs,
            "b c (h p1) (w p2) -> (b h w) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size
        )
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(
            lambda t: rearrange(t, "b (c j) i -> (b i) j c", c=self.inner_dim),
            (ks, vs)
        )

        # print(f"qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}")
        #split heads
        qs, ks, vs = map(
            lambda t: rearrange(
                t, "b n (head c) -> (b head) n c", head=self.num_spatial_heads
            ),
            (qs, ks, vs)
        )

        # attention
        qs = qs * self.scale
        spatial_attn: Tensor = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)
        out = (spatial_attn @ vs)

        out = rearrange(
            out,
            "(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)",
            head=self.num_spatial_heads,
            h=h // self.window_size,
            w=w // self.window_size,
            p1=self.window_size,
            p2=self.window_size
        )

        # merge spatial and channel
        return self.project_out(out)



class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio,
        num_channel_heads: int,
        num_spatial_heads: int,
        spatial_dim_head: int,
        ffn_expansion_factor: float,
        bias: bool,
        layer_norm_type: LayerNormType,
    ):
        super().__init__()

        self.spatial_attn = OCAB(
            dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias
        )
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, layer_norm_type)
        self.norm2 = LayerNorm(dim, layer_norm_type)
        self.norm3 = LayerNorm(dim, layer_norm_type)
        self.norm4 = LayerNorm(dim, layer_norm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x: Tensor) -> Tensor:
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x



# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)



# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)



class SR_Upsample(nn.Sequential):
    """SR_Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of features.
    """

    def __init__(self, scale: int, num_feat: int):
        m: list = []
        if scale & (scale - 1) == 0:
            # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.extend([
                    nn.Conv2d(num_feat, 4 * num_feat, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2)
                ])

        elif scale == 3:
            m = [
                nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1),
                nn.PixelShuffle(3)
            ]

        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )
        super().__init__(*m)



class XRestormer(nn.Module):
    def __init__(self,
        scale: int = 1,
        in_nc: int = 3,
        out_nc: int = 3,
        dim: int = 48,
        num_blocks: list[int] = [4, 6, 6, 8],
        num_refinement_blocks = 4,
        channel_heads: list[int] = [1, 2, 4, 8],
        spatial_heads: list[int] = [2, 2, 3, 4],
        overlap_ratio: list[int] = [0.5, 0.5, 0.5, 0.5],
        window_size: int = 8,
        spatial_dim_head: int = 16,
        bias: bool = False,
        ffn_expansion_factor: float = 2.66,
        layer_norm_type: LayerNormType = 'WithBias',
        # dual_pixel_task = False # True for dual-pixel defocus deblurring only. Also set in_nc=6
    ):
        super().__init__()
        self.scale = scale

        self.patch_embed = OverlapPatchEmbed(in_nc, dim)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[0],
                    num_channel_heads=channel_heads[0],
                    num_spatial_heads=spatial_heads[0],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim) # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[1],
                    num_channel_heads=channel_heads[1],
                    num_spatial_heads=spatial_heads[1],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim*2**1)) # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**2),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[2],
                    num_channel_heads=channel_heads[2],
                    num_spatial_heads=spatial_heads[2],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim*2**2)) # From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**3),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[3],
                    num_channel_heads=channel_heads[3],
                    num_spatial_heads=spatial_heads[3],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim*2**3)) # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**2),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[2],
                    num_channel_heads=channel_heads[2],
                    num_spatial_heads=spatial_heads[2],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[2])
            ]
        )


        self.up3_2 = Upsample(int(dim*2**2)) # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[1],
                    num_channel_heads=channel_heads[1],
                    num_spatial_heads=spatial_heads[1],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim*2**1))  # From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[0],
                    num_channel_heads=channel_heads[0],
                    num_spatial_heads=spatial_heads[0],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim*2**1),
                    window_size=window_size,
                    overlap_ratio=overlap_ratio[0],
                    num_channel_heads=channel_heads[0],
                    num_spatial_heads=spatial_heads[0],
                    spatial_dim_head=spatial_dim_head,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    layer_norm_type=layer_norm_type
                )
                for _ in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            int(dim*2**1), out_nc, kernel_size=3, stride=1, padding=1, bias=bias
        )


    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[2:]
        x0 = pad(x0, modulo=64, mode='reflect')

        if self.scale > 1:
            x = F.interpolate(
                x, scale_factor=self.scale, mode="bilinear", align_corners=False
            )

        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + x

        out_dec_level1 = unpad(out_dec_level1, size, scale=self.scale)

        return out_dec_level1
