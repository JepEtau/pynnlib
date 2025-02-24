"""
author: Min Seok Lee and Wooseok Shin
Github repo: https://github.com/Karel911/TRACER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .EfficientNet import EfficientNet
from .effi_utils import get_model_shape
from .att_modules import RFB_Block, aggregation, ObjectAttention


class TRACER(nn.Module):
    def __init__(
        self,
        arch: int = 7,
        RFB_aggregated_channel: list[int] = [32, 64, 128],
        denoise: float = 0.93,
        gamma: float = 0.1,
    ):
        """
            arch                default='0'   Backbone Architecture'
            channels            type=list, default=[24, 40, 112, 320]
            RFB_aggregated_channel  default=[32, 64, 128]
            frequency_radius    type=int, default=16 Frequency radius r in FFT'
            denoise             type=float, default=0.93  Denoising background ratio'
            gamma               type=float, default=0.1 Confidence ratio')
        """

        super().__init__()
        self.model = EfficientNet.from_pretrained(
            f'efficientnet-b{arch}',
            advprop=True
        )
        self.block_idx, self.channels = get_model_shape(arch)

        # Receptive Field Blocks
        channels = [int(arg_c) for arg_c in RFB_aggregated_channel]
        self.rfb2 = RFB_Block(self.channels[1], channels[0])
        self.rfb3 = RFB_Block(self.channels[2], channels[1])
        self.rfb4 = RFB_Block(self.channels[3], channels[2])

        # Multi-level aggregation
        self.agg = aggregation(channels, gamma)

        # Object Attention
        self.ObjectAttention2 = ObjectAttention(channel=self.channels[1], kernel_size=3, denoise=denoise)
        self.ObjectAttention1 = ObjectAttention(channel=self.channels[0], kernel_size=3, denoise=denoise)

    def forward(self, inputs: Tensor):
        B, C, H, W = inputs.size()

        # EfficientNet backbone Encoder
        x = self.model.initial_conv(inputs)
        features, edge = self.model.get_blocks(x, H, W)

        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)

        ds_map0 = F.interpolate(D_0, scale_factor=8, mode='bilinear')

        D_1 = self.ObjectAttention2(D_0, features[1])
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode='bilinear')

        ds_map = F.interpolate(D_1, scale_factor=2, mode='bilinear')
        D_2 = self.ObjectAttention1(ds_map, features[0])
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode='bilinear')

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3

        return torch.sigmoid(final_map), torch.sigmoid(edge), \
               (torch.sigmoid(ds_map0), torch.sigmoid(ds_map1), torch.sigmoid(ds_map2))
