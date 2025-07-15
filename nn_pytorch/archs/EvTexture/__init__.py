import math
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_nsequences,
)
from ..torch_to_onnx import to_onnx


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict

    in_nc: int = state_dict["cnet.main.0.weight"].shape[1]
    out_nc, num_feat = state_dict["conv_last.weight"].shape[:2]
    num_block = get_nsequences(state_dict, "backward_trunk.main.2")
    # scale is fixed
    scale: int = 4

    # from .module.evtexture import EvTexture
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_feat=num_feat,
        num_block=num_block,
        spynet_path=""
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="EvTexture",
        detection_keys=(
            "cnet.main.0.weight",
            "enet.conv1.weight",
            "update_block.gru.convz.weight",
            "backward_trunk.main.0.weight",
            "spynet.basic_module.0.basic_module.0.weight",
            "conv_last.weight",
        ),
        module=Module(file="evtexture", class_name="EvTexture"),
        parse=parse,
        dtypes=('fp32'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        ),
        to_onnx=to_onnx,
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
        ),
    ),
)
