import math
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.logger import is_debugging
from pynnlib.utils.p_print import *
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 2
    in_nc: int = 3
    out_nc: int = 3

    dim = state_dict["body.0.norm.scale"].shape[0]
    ffn_expansion: float = (
        state_dict["body.0.fc1.conv_3x3_rep.weight"].shape[0] / dim / 2.
    )
    n_blocks: int = get_nsequences(state_dict, "body")
    unshuffle_mod: bool = "to_feat.1.alpha" in state_dict
    dccm: bool = "body.0.fc2.alpha" in state_dict
    se = "body.0.conv.2.squeezing.0.weight" in state_dict

    if unshuffle_mod:
        w = state_dict["to_feat.1.conv1.k0"].shape[1]
        if w == 48:
            scale = 1
        else:
            scale = math.isqrt(state_dict["to_feat.1.weight"].shape[1] // 3)

    arch_name: str = model.arch.name
    if ffn_expansion == 1.5 and not dccm and unshuffle_mod:
        arch_name = f"{arch_name} (ul)"

    elif ffn_expansion == 2. and dccm and unshuffle_mod:
        arch_name = f"{arch_name} (l)"

    if is_debugging():
        from .module.RTMoSR import RTMoSR
        model.update(ModuleClass=RTMoSR)

    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        dim=dim,
        ffn_expansion=ffn_expansion,
        n_blocks=n_blocks,
        unshuffle_mod=unshuffle_mod,
        dccm=dccm,
        se=se,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RTMoSR",
        detection_keys=(
            "body.0.norm.scale",
            "to_img.0.alpha",
            "to_img.0.conv1.k0",
            "to_img.0.conv1.b1",
        ),
        module=Module(file="RTMoSR", class_name="RTMoSR"),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=set(['fp32', 'bf16', 'fp16']),
    ),
)
