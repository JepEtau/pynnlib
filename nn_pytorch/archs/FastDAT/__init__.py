import math
from typing import Literal
import onnx
from pynnlib.architecture import NnPytorchArchitecture,SizeConstraint
from pynnlib.model import PyTorchModel
from pynnlib.nn_pytorch.archs.FastDAT.module.fdat import SampleMods3
from pynnlib.nn_types import Idtype, ShapeStrategy
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx


# copy from module to avoid importing module right now
sample_types: list[str] = [
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
    from .module.fdat import FDAT
    (
        block_version,
        sample_type_index,
        scale,
        in_dim,
        out_nc,
        mid_dim,
        group
    ) = state_dict["upsampler.MetaUpsample"].tolist()
    if sample_type_index >= len(sample_types):
        raise ValueError(
            f"[E] \'sample_type_index\' {sample_type_index} is not supported"
        )

    num_groups: int = get_nsequences(state_dict, "groups")
    group_block_pattern: list[str] = ["spatial", "channel"]
    depth_per_group: int = int(
        get_nsequences(state_dict, "groups.0.blocks")
        / len(group_block_pattern)
    )
    num_heads, _, window_size_sq = (
        state_dict["groups.0.blocks.0.attn.bias"].shape[:3]
    )
    window_size = math.isqrt(window_size_sq)
    ffn_expansion_ratio: float = (
        float(state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0])
        / embed_dim
    )
    aim_reduction_ratio: int = int(
        embed_dim / state_dict["groups.0.blocks.0.inter.cg.1.weight"].shape[0]
    )
    img_range: float = 1.0

    variants: dict[int, str] = {
        96: " (tiny)",
        108: " (light)",
        120: " (medium)",
        180: " (large)"
    }
    arch_name = f"{model.arch.name}{variants.get(embed_dim, '')}"
    if embed_dim == 180 and num_groups == 6:
        arch_name = f"{model.arch.name} (xl)"

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
        mid_dim=mid_dim,
        upsampler_type=sample_types[sample_type_index],
        img_range=img_range,
    )




MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="FastDAT",
        detection_keys=(
            "groups.0.blocks.0.n1.weight",
            "groups.0.blocks.0.attn.bias",
            "groups.0.blocks.0.ffn.fc1.weight",
            "groups.0.blocks.0.inter.cg.1.weight",
            "upsampler.MetaUpsample",
        ),
        module_file="fdat",
        module_class_name="FDAT",
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8))
    ),
)
