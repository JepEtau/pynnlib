from hutils import red, yellow
import math
from pynnlib.architecture import (
    InferType,
    Module,
    NnPytorchArchitecture,
    OnnxConv,
)
from pynnlib.model import PyTorchModel



def parse(model: PyTorchModel) -> None:
    recurrent_cell_shape = model.state_dict[f"recurrent_cell.conv_s1_first.0.weight"].shape
    in_nc: int = recurrent_cell_shape[3]
    num_feat: int = recurrent_cell_shape[0]
    scale: int = int(
        math.sqrt((recurrent_cell_shape[1] - num_feat - in_nc * 3) // in_nc)
    )
    num_block: tuple[int, int, int] = (5, 3, 2)
    out_nc: int = in_nc

    # from .module.vsr_arch import MSRSWVSR
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,
        num_feat=num_feat,
        num_block=num_block,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="AnimeSR",
        detection_keys=(
            "recurrent_cell.conv_s1_first.0.weight",
            "recurrent_cell.fusion.0.weight",
        ),
        module=Module(file="vsr_arch", class_name="MSRSWVSR"),
        parse=parse,
        dtypes=('fp32', 'fp16'),
        # size_constraint=SizeConstraint(
        #     min=(64, 64)
        # )
        infer_type=InferType(
            type='temporal',
            inputs=3,
            outputs=1,
        ),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
    ),
)
