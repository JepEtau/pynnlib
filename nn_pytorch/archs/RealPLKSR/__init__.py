import math
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.utils.p_print import *
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 0
    in_nc: int = 3
    out_nc: int = 3

    dim: int = state_dict["feats.0.weight"].shape[0]
    max_block_indice: int = get_max_indice(state_dict, "feats")
    n_blocks: int = max_block_indice - 2
    scale = math.isqrt(
        state_dict[f"feats.{max_block_indice}.weight"].shape[0] // 3
    )

    shape: str = state_dict["feats.1.lk.conv.weight"].shape
    kernel_size, split_ratio = shape[2], shape[0] / dim

    use_ea: bool = bool("feats.1.attn.f.0.weight" in state_dict)

    use_dysample = "to_img.init_pos" in state_dict

    variant: str = ""
    if dim == 96:
        variant = " Large"
    elif (n_blocks == 12 and kernel_size == 13 and not use_ea):
        variant = " Small"

    arch_name = f"{model.arch_name}{variant}"
    if use_dysample:
        arch_name = f"{arch_name} DySample"

    # from .module.realplksr_arch import RealPLKSR
    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        dim=dim,
        n_blocks=n_blocks,
        upscaling_factor=scale,
        kernel_size=kernel_size,
        split_ratio=split_ratio,
        use_ea=use_ea,
        # norm_groups: int = 4,
        # dropout: float = 0,
        dysample=use_dysample,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RealPLKSR",
        detection_keys=(
            "feats.0.weight",
            "feats.1.channel_mixer.0.weight",
            "feats.1.lk.conv.weight"
        ),
        module=Module(file="realplksr_arch", class_name="RealPLKSR"),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'fp16', 'bf16']),
    ),
)
