from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel


def parse(model: PyTorchModel) -> None:
    # state_dict: StateDict = model.state_dict
    scale: int = 1
    in_nc: int = 3
    out_nc: int = 3

    # from .module.SPyNet import SpyNet
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        # return_levels=[5],
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="SPyNet",
        detection_keys=(
            "basic_module.0.basic_module.0.weight",
            "basic_module.3.basic_module.0.weight",
            "basic_module.5.basic_module.0.weight",
        ),
        module=Module(file="SPyNet", class_name="SpyNet"),
        parse=parse,
        # to_onnx=to_onnx,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
)
