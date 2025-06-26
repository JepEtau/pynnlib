from typing import Literal
from pynnlib.utils.p_print import *
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 0
    in_nc: int = 3
    out_nc: int = 3

    upsample = Literal[
        'conv',
        'pixelshuffledirect',
        'pixelshuffle',
        'nearest+conv',
        'dysample'
    ]
    dim, in_nc = state_dict['in_to_dim.weight'].shape[:2]
    n_blocks = get_nsequences(state_dict, 'blocks')
    n_block = get_nsequences(state_dict, 'blocks.0.blocks')
    expansion_factor_shape = state_dict['blocks.0.blocks.0.fc1.weight'].shape
    expansion_factor = (expansion_factor_shape[0] / expansion_factor_shape[1]) / 2
    expansion_msg_shape = state_dict['blocks.0.msg.gated.0.fc1.weight'].shape
    expansion_msg = (expansion_msg_shape[0] / expansion_msg_shape[1]) / 2
    _, index, scale, _, out_nc, upsample_dim, _ = state_dict['upscale.MetaUpsample']
    upsampler = upsample[int(index)]

    # Update model parameters
    # from .module.MoESR import MoESR
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        n_blocks=n_blocks,
        n_block=n_block,
        dim=dim,
        expansion_factor=expansion_factor,
        expansion_msg=expansion_msg,
        upsampler=upsampler,
        upsample_dim=int(upsample_dim),
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="MoESR",
        detection_keys=(
            'in_to_dim.weight',
            'in_to_dim.bias',
            'blocks.0.blocks.0.gamma',
            'blocks.0.blocks.0.norm.weight',
            'blocks.0.blocks.0.fc1.weight',
            'blocks.0.blocks.0.conv.dwconv_hw.weight',
            'blocks.0.blocks.0.conv.dwconv_w.weight',
            'blocks.0.blocks.0.conv.dwconv_h.weight',
            'blocks.0.blocks.0.fc2.weight',
            'upscale.MetaUpsample',
        ),
        module=Module(file="MoESR", class_name="MoESR"),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16']),
    ),
)
