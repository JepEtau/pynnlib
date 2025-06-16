

import math
import re
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PyTorchModel
from pynnlib.nn_pytorch.torch_types import StateDict
from ..helpers import (
    get_scale_and_out_nc,
    get_max_indice,
    get_nsequences,
)
from .module.esc_arch import ESC
from ..torch_to_onnx import to_onnx


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    # max_indice = get_max_indice(model.state_dict, "body")

    # num_feat, in_nc = model.state_dict["body.0.weight"].shape[:2]
    # num_conv: int = (max_indice - 2) // 2
    # pixelshuffle_shape: int = model.state_dict[f"body.{max_indice}.bias"].shape[0]
    # scale, out_nc = get_scale_and_out_nc(pixelshuffle_shape, in_nc)
    dim, in_nc = state_dict["proj.weight"].shape[:2]
    pdim: int = 16
    window_size: int = 32
    # out channel is fixed in the code
    out_nc: int = 3


    arch_name: str = model.arch.name

    if "to_img.weight" in state_dict:
        scale = int(math.sqrt(state_dict["to_img.weight"].shape[0] // 3))

    elif "to_img.1.weight" in state_dict and "skip.0.weight" in state_dict:
        arch_name = f"{arch_name} (Real)"
        layer_norm = True

        scale: int = 0
        for k in state_dict.keys():
            if (result := re.search(re.compile(r"to_img.\d.weight"), k)) is not None:
                scale += 1

    kernel_size: int = 13
    fp: bool = False
    if "plk_filter" in state_dict:
        pdim = state_dict["plk_filter"].shape[0]
        kernel_size = state_dict["plk_filter"].shape[2]
        arch_name = f"{arch_name}"

    elif "lk_channel" in state_dict and "lk_spatial" in state_dict:
        pdim = state_dict["lk_spatial"].shape[0]
        kernel_size = state_dict["lk_spatial"].shape[2]
        arch_name = f"{arch_name} (FP)"
        fp = True

    else:
        raise ValueError("[E] Missing key to detect pdim")

    n_blocks: int = get_nsequences(state_dict=state_dict, seq_key="blocks")
    if n_blocks < 5:
        arch_name = f"{arch_name} (light)"
    conv_blocks: int = get_nsequences(
        state_dict=state_dict, seq_key="blocks.0.pconvs"
    )

    num_heads: int = state_dict["blocks.0.attn.relative_position_bias"].shape[0]

    exp_ratio: float = float(state_dict["blocks.0.convffns.0.proj.weight"].shape[0]) / dim

    layer_norm: bool = False
    if "blocks.0.lns.0.weight" in state_dict and not fp:
        layer_norm = True


    use_dysample: bool = bool("to_img.init_pos" in state_dict)
    if use_dysample:
        groups = 4
        scale = int(math.sqrt(state_dict["to_img.offset.weight"].shape[1] / (2 * groups)))
        if ")" in arch_name:
            arch_name = f"{arch_name[:-1]}, DySample)"
        else:
            arch_name = f"{arch_name} (DySample)"

    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=ESC,
        dim=dim,
        pdim=pdim,
        kernel_size=kernel_size,
        n_blocks=n_blocks,
        conv_blocks=conv_blocks,
        window_size=window_size,
        num_heads=num_heads,
        upscaling_factor=scale,
        exp_ratio=exp_ratio,
        attn_type='sdpa',
        fp=fp,
        use_dysample=use_dysample,
        realsr=layer_norm,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="ESC",
        detection_keys=(
            "proj.weight",
            # "blocks.0.pconvs",
            "blocks.0.attn.relative_position_bias",
            "blocks.2.ln_out.weight",
        ),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        )
    ),
)



# Base
# ESC_DFLIP_xxx
# ESC_DIV2K
# model_kwargs = {
#     "dim": 64,
#     "pdim": 16,
#     "kernel_size": 13,
#     "n_blocks": 5,
#     "conv_blocks": 5,
#     "window_size": 32,
#     "num_heads": 4,
#     "upscaling_factor": upsampling_factor,
#     "exp_ratio": 1.25,
#       attn_type: Flex

# FP
# ESC_FP
# model_kwargs = {
#     'dim': 48,
#     'pdim': 16,
#     'kernel_size': 13,
#     'n_blocks': 5,
#     'conv_blocks': 5,
#     'window_size': 32,
#     'num_heads': 3,
#     'upscaling_factor': upsampling_factor,
#     'exp_ratio': 1.25,
#       attn_type: Flex
# }

# Light
# model_kwargs = {
#     "dim": 64,
#     "pdim": 16,
#     "kernel_size": 13,
#     "n_blocks": 3,
#     "conv_blocks": 5,
#     "window_size": 32,
#     "num_heads": 4,
#     "upscaling_factor": upsampling_factor,
#     "exp_ratio": 1.25,
#       attn_type: Flex
# }

# ESC-Real
# model_kwargs = {
#     'dim': 64,
#     'pdim': 16,
#     'kernel_size': 13,
#     'n_blocks': 10,
#     'conv_blocks': 5,
#     'window_size': 32,
#     'num_heads': 4,
#     'upscaling_factor': upsampling_factor,
#     'exp_ratio': 2,
#       attn_type: Flex
# }
# ESC-Real-M
# model_kwargs = {
#     'dim': 64,
#     'pdim': 16,
#     'kernel_size': 13,
#     'n_blocks': 10,
#     'conv_blocks': 5,
#     'window_size': 32,
#     'num_heads': 4,
#     'upscaling_factor': upsampling_factor,
#     'exp_ratio': 2,
#     'upsampler': 'pixelshuffle',
# }
# shape = (batch_size, 3, height // upsampling_factor, width // upsampling_factor)
# model = ESCRealM(**model_kwargs)

