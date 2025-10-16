from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    OnnxConv,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_scale_and_out_nc,
    get_max_indice,
)



def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict

    max_indice = get_max_indice(state_dict, "body")

    num_feat, in_nc = state_dict["body.0.weight"].shape[:2]
    num_conv: int = (max_indice - 2) // 2
    pixelshuffle_shape: int = state_dict[f"body.{max_indice}.bias"].shape[0]
    scale, out_nc = get_scale_and_out_nc(pixelshuffle_shape, in_nc)

    # from .module.SRVGG import SRVGGNetCompact
    model.update(
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_feat=num_feat,
        num_conv=num_conv,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="RealESRGAN (Compact)",
        detection_keys=(
            "body.0.weight",
            "body.1.weight"
        ),
        module=Module(file="SRVGG", class_name="SRVGGNetCompact"),
        parse=parse,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        ),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
            shape_strategy_types=set(['dynamic', 'fixed', 'static']),
        ),
    ),
)
