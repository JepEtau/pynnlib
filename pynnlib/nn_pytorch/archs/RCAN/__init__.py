import math
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from ..helpers import (
    get_max_indice,
    get_pixelshuffle_params
)
from ..torch_to_onnx import to_onnx
from ...torch_types import StateDict


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict

    in_nc: int = state_dict["tail.1.weight"].shape[0]
    out_nc: int = in_nc

    n_resgroups = get_max_indice(state_dict, "body")
    n_resblocks = get_max_indice(state_dict, "body.0.body")

    max_block_indice: int = get_max_indice(state_dict, "tail.0")
    shape = state_dict[f"tail.0.{max_block_indice}.weight"].shape
    scale = math.isqrt(shape[0] // 3)
    num_feat = shape[1]

    scale, num_feat = get_pixelshuffle_params(state_dict, "tail.0")
    # n_colors = state_dict["tail.1.weight"].shape[0]
    # max_block_indice: int = get_max_indice(state_dict, "feats")
    # scale = math.isqrt(
    #     state_dict[f"tail.0.{max_block_indice}.weight"].shape[0] // 3
    # )

    rgb_range = 255
    kernel_size = state_dict[f"head.0.weight"].shape[-1]
    norm = "sub_mean.weight" in state_dict
    reduction = (
        num_feat // state_dict["body.0.body.0.body.3.conv_du.0.weight"].shape[0]
    )
    res_scale: int = 1

    # from .module.rcan import RCAN
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        n_resgroups=n_resgroups,
        n_resblocks=n_resblocks,
        n_feats=num_feat,
        reduction=reduction,
        n_colors=in_nc,
        kernel_size=kernel_size,
        res_scale=res_scale,
        rgb_range=rgb_range,
        norm=norm,

        num_feat=num_feat,
        # num_conv=num_conv,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RCAN",
        detection_keys=(
            "tail.1.weight",
            "body.0.body.0.body.0.weight",
            "body.0.body.0.body.3.conv_du.0.weight",
        ),
        module=Module(file="rcan", class_name="RCAN"),
        parse=parse,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        ),
        to_onnx=to_onnx,
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
        ),
    ),
)
