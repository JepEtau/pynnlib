import math
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.logger import is_debugging
from pynnlib.model import PyTorchModel
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

    variants: dict[int, tuple[str, float]] = {
        96: (" (tiny)", 0.5),
        108: (" (light)", 0.08),
        120: (" (medium)", 0.1),
        180: (" (large)", 0.1),
    }
    variant_name, drop_path_rate = variants.get(embed_dim, ('', 0.1))
    arch_name = f"{model.arch.name}{variant_name}"
    if embed_dim == 180 and num_groups == 6:
        arch_name = f"{model.arch.name} (xl)"
        drop_path_rate = 0.1

    if is_debugging():
        from .module.fdat import FDAT
        model.update(ModuleClass=FDAT)

    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_in_ch=in_nc,
        num_out_ch=out_nc,
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
        module=Module(file="fdat", class_name="FDAT"),
        parse=parse,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8)),
        to_onnx=to_onnx,
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
        ),
    ),
)
