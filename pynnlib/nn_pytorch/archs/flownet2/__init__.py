from typing import Literal
from warnings import warn
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
# from ...torch_types import StateDict
#



# def parse(model: PyTorchModel) -> None:
#     state_dict: StateDict = model.state_dict
#     scale: int = 1
#     in_nc: int = 3
#     out_nc: int = 3

#     # from .module.flownet2 import FlowNet2
#     model.update(
#         scale=scale,
#         in_nc=in_nc,
#         out_nc=out_nc,
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
#         module=Module(file="flownet2", class_name="FlowNet2"),
#         parse=parse,
#         to_onnx = OnnxConv(
#             dtypes=set(['fp32']),
#             shape_strategy_types=set(['dynamic', 'static']),
#         ),
#         dtypes=('fp32'),
#         size_constraint=SizeConstraint(
#             min=(8, 8)
#         )
#     ),
# )
