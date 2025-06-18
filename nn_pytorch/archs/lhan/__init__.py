import onnx
from pynnlib.architecture import NnPytorchArchitecture,SizeConstraint
from pynnlib.model import PyTorchModel
from pynnlib.nn_types import Idtype, ShapeStrategy
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx



# def _to_onnx(
#     model: PyTorchModel,
#     dtype: Idtype,
#     opset: int,
#     static: bool = False,
#     shape_strategy: ShapeStrategy | None = None,
#     device: str = 'cpu',
#     batch: int = 1,
# ) -> onnx.ModelProto | None:
#     model.dtypes = ('fp32')
#     model.to_onnx = True
#     return to_onnx(
#         model=model,
#         dtype='fp32',
#         opset=opset,
#         static=False,
#         shape_strategy=shape_strategy,
#         device=device,
#         batch=batch,
#     )


def parse(model: PyTorchModel) -> None:
    in_nc: int
    state_dict: StateDict = model.state_dict
    scale: int = 2
    in_nc = out_nc = 3

    # Detect the parameters from the state_dict
    embed_dim, in_nc = state_dict["conv_first.weight"].shape[:2]

    # Currently use fixed value until end of PoC
    # Tiny
    if embed_dim == 96:
        arch_name = f"{model.arch.name} (tiny)"
        num_groups=2
        depth_per_group=2
        num_heads=3
        ffn_expansion_ratio=1.5
        drop_path_rate=0.05
        upsampler_type='pixelshuffle'

    # Light
    elif embed_dim == 108:
        arch_name = f"{model.arch.name} (light)"
        num_groups=3
        depth_per_group=2
        num_heads=4
        ffn_expansion_ratio=2.0
        drop_path_rate=0.08
        upsampler_type='nearest_conv'

    # Medium
    elif embed_dim == 120:
        arch_name = f"{model.arch.name} (medium)"
        num_groups=4
        depth_per_group=3
        num_heads=4
        ffn_expansion_ratio=2.0
        drop_path_rate=0.1
        upsampler_type='transpose_conv'

    else:
        raise ValueError("Variant not found")

    window_size: int = 8
    aim_reduction_ratio: int = 8

    # Populate the model with arch's args
    # from .module.lhan_arch import Lhan
    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_in_ch=in_nc,
        num_out_ch=out_nc,
        upscaling_factor=scale,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        # group_block_pattern=['spatial', 'channel'],
        drop_path_rate=drop_path_rate,
        upsampler_type=upsampler_type,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="lhan",
        # Choice: use common keys for all variants,
        #   and use the parser to detect it until end of PoC
        detection_keys=(
            "groups.0.blocks.3.ffn.smix.weight",
            "groups.1.blocks.3.ffn.smix.weight",
        ),
        module_file="lhan_arch",
        module_class_name="Lhan",
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8))
    ),
)




