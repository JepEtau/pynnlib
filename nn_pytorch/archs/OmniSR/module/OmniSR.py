#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .OSAG import OSAG
from .pixelshuffle import pixelshuffle_block
from ..._shared.pad import pad, unpad



class OmniSR(nn.Module):

    def __init__(
        self,
        *,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        block_num=1,
        pe=True,
        window_size=8,
        res_num=1,
        scale=4,
        bias=True,
    ):
        super().__init__()

        residual_layer = []
        self.res_num = res_num

        self.scale = scale
        self.window_size = window_size

        for _ in range(res_num):
            temp_res = OSAG(
                channel_num=num_feat,
                bias=bias,
                block_num=block_num,
                window_size=self.window_size,
                pe=pe,
            )
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(
            in_channels=num_in_ch,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.output = nn.Conv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.up = pixelshuffle_block(num_feat, num_out_ch, scale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))


    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[2:]
        x = pad(x, modulo=self.window_size, mode='constant')

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = unpad(out, size, scale=self.scale)
        return out
