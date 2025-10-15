from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    OnnxConv,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict

from ..helpers import get_max_indice, get_nsequences



def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    dim, in_nc = state_dict["m_head.0.weight"].shape[:2]
    config = (
        get_max_indice(state_dict, "m_down1"),
        get_max_indice(state_dict, "m_down2"),
        get_max_indice(state_dict, "m_down3"),
        get_nsequences(state_dict, "m_body"),
        get_max_indice(state_dict, "m_up3"),
        get_max_indice(state_dict, "m_up2"),
        get_max_indice(state_dict, "m_up1")
    )

    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=in_nc,

        dim=dim,
        config=config,
        drop_path_rate=0.,
        input_resolution=256
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="SCUNet",
        detection_keys=(
            "m_head.0.weight",
            "m_tail.0.weight"
        ),
        module=Module(file="network_scunet", class_name="SCUNet"),
        parse=parse,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(64, 64),
            modulo=1
        ),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
        ),
    ),
)
