from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
)
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx


def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 4
    in_nc: int = 3
    out_nc: int = 3
    num_feat: int = 64
    num_block: int = 25

    num_feat = state_dict["backward_trunk.main.0.weight"].shape[0]
    out_nc = in_nc
    # scale is fixed
    scale: int = 4
    num_block = get_nsequences(state_dict, "backward_trunk.main.2")

    encoder_fp: str = "ranker.pth"

    # from .module.CAVSR import CAVSR
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_feat=num_feat,
        num_block=num_block,
        encoder_fp=encoder_fp
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="CAVSR",
        detection_keys=(
            "encoder.E.E.0.weight",
            "backward_trunk.main.0.weight",
            "modulate_f.conv2.weight",
            "fusion.0.weight",
            "backward_trunk.main.2.0.conv1.weight"
        ),
        module=Module(file="CAVSR", class_name="CAVSR"),
        parse=parse,
        dtypes=('fp32'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        ),
        to_onnx=to_onnx,
    ),
)
