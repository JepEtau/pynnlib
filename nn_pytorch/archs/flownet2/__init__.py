from typing import Literal
from warnings import warn
from pynnlib.architecture import (
    NnPytorchArchitecture,
    SizeConstraint,
)
from pynnlib.model import PyTorchModel
from pynnlib.utils.p_print import *
# from ...torch_types import StateDict
# from ..torch_to_onnx import to_onnx
# from .module.flownet2 import FlowNet2



# def parse(model: PyTorchModel) -> None:
#     state_dict: StateDict = model.state_dict
#     scale: int = 1
#     in_nc: int = 3
#     out_nc: int = 3

#     model.update(
#         scale=scale,
#         in_nc=in_nc,
#         out_nc=out_nc,

#         ModuleClass=FlowNet2,
#     )



# MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
#     NnPytorchArchitecture(
#         name="flownet2",
#         detection_keys=(
#             "flownetc.conv1.0.weight",
#             "flownetc.upsampled_flow3_to_2.weight",
#             "flownets_1.upsampled_flow3_to_2.weight",
#             "flownets_2.upsampled_flow3_to_2.weight",
#             "flownets_d.upsampled_flow3_to_2.weigh",
#             "flownetfusion.upsampled_flow1_to_0.weight",
#         ),
#         parse=parse,
#         to_onnx=to_onnx,
#         dtypes=('fp32'),
#         size_constraint=SizeConstraint(
#             min=(8, 8)
#         )
#     ),
# )
