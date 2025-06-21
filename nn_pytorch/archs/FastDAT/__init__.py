from typing import Literal
import onnx
from pynnlib.architecture import NnPytorchArchitecture,SizeConstraint
from pynnlib.model import PyTorchModel
from pynnlib.nn_pytorch.archs.FastDAT.module.fdat import SampleMods3
from pynnlib.nn_types import Idtype, ShapeStrategy
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx


# copy from module to avoid importing module right now
SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv"
]


def parse(model: PyTorchModel) -> None:
    in_nc: int
    state_dict: StateDict = model.state_dict
    scale: int = 2
    in_nc = out_nc = 3

    # Detect the parameters from the state_dict
    embed_dim, in_nc = state_dict["conv_first.weight"].shape[:2]
    out_nc: int = in_nc

    # Currently use fixed value until end of PoC
    # from module.fdat import FDAT

    ffn_expansion_ratio: float = 2.
    window_size: int = 8
    aim_reduction_ratio: int = 8
    group_block_pattern: list[str] | None = None
    mid_dim: int = 64
    upsampler_type: SampleMods3 = 'transpose+conv'
    img_range: float = 1.0

    # Tiny
    if embed_dim == 96:
        arch_name = f"{model.arch.name} (tiny)"
        num_groups = 2
        depth_per_group = 2
        num_heads = 3
        ffn_expansion_ratio = 1.5
        drop_path_rate: float = 0.05
        upsampler_type = "pixelshuffle"

    # Light
    elif embed_dim == 108:
        arch_name = f"{model.arch.name} (light)"
        num_groups = 3
        depth_per_group = 2
        num_heads = 4
        drop_path_rate: float = 0.08

    # Medium
    elif embed_dim == 120:
        arch_name = f"{model.arch.name} (medium)"
        num_groups = 4
        depth_per_group = 3
        num_heads = 4
        drop_path_rate: float = 0.1


    # Large, XL
    elif embed_dim == 180:
        drop_path_rate: float = 0.1

        # use num_groups
        if ...:
            arch_name = f"{model.arch.name} (large)"
            num_groups = 4
            depth_per_group = 4
            num_heads = 6

        else:
            arch_name = f"{model.arch.name} (xl)"
            num_groups = 6
            depth_per_group = 6
            num_heads = 6


    else:
        raise ValueError("Variant not found")



    # Populate the model with arch's args
    # from .module.lhan_arch import Lhan
    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_in_ch=in_nc,
        num_out=out_nc,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        drop_path_rate=drop_path_rate,
        mid_dim=mid_dim,
        upsampler_type=upsampler_type,
        img_range=img_range,
    )




MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="FastDAT",
        detection_keys=(
            # TODO
            "fastdat",
        ),
        module_file="fdat",
        module_class_name="FDAT",
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8))
    ),
)
