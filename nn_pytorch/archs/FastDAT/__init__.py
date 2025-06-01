import onnx
from pynnlib.architecture import NnPytorchArchitecture,SizeConstraint
from pynnlib.model import PyTorchModel
from pynnlib.nn_types import Idtype, ShapeStrategy
from ...torch_types import StateDict
from ..helpers import get_max_indice
from ..torch_to_onnx import to_onnx

from .module.fastdat_arch import FastDAT



def _to_onnx(
    model: PyTorchModel,
    dtype: Idtype,
    opset: int,
    static: bool = False,
    shape_strategy: ShapeStrategy | None = None,
    device: str = 'cpu',
    batch: int = 1,
) -> onnx.ModelProto | None:
    # Whatever the dtype, convert to fp32 because others are not supported
    model.dtypes = ('fp32',)
    return to_onnx(
        model=model,
        dtype='fp32',
        opset=opset,
        static=False,
        shape_strategy=shape_strategy,
        device=device,
        batch=batch,
    )



def parse(model: PyTorchModel) -> None:
    in_nc: int
    state_dict: StateDict = model.state_dict
    scale: int = 2

    in_nc = out_nc = 3

    # Default values from "large" for testing purpose
    embed_dim = 144
    num_groups = 6
    depth_per_group = 3
    num_heads = 6
    window_size = 8
    ffn_expansion_ratio = 2.5
    aim_reduction_ratio = 8
    drop_path_rate = 0.15
    upsampler_type = 'dysample'

    # Detect the parameters from the state_dict
    embed_dim, in_nc = state_dict["conv_first.weight"].shape[:2]

    # Populate the model with arch's args
    model.update(
        arch_name=model.arch.name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        # Args of the arch's module
        num_in_ch=in_nc,
        num_out_ch=out_nc,
        upscale=scale,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        drop_path_rate=drop_path_rate,
        upsampler_type=upsampler_type,
        use_checkpoint=False,
        # Module class
        ModuleClass=FastDAT,
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="FastDAT",
        detection_keys=(
            "groups.5.blocks.2.interaction.spatial_interaction.0.weight",
        ),
        parse=parse,
        to_onnx=_to_onnx,
        dtypes=('fp32', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8))
    ),
)



# Model Variants
# def fastdat_tiny(**kwargs):
#     """Tiny variant for maximum speed."""
#     return FastDAT(
#         embed_dim=96,
#         num_groups=2,
#         depth_per_group=2,
#         num_heads=3,
#         window_size=8,
#         ffn_expansion_ratio=1.5,
#         drop_path_rate=0.05,
#         upsampler_type='pixelshuffle',
#         **kwargs
#     )


# def fastdat_light(**kwargs):
#     """Light variant balancing speed and quality."""
#     return FastDAT(
#         embed_dim=108,
#         num_groups=3,
#         depth_per_group=2,
#         num_heads=4,
#         window_size=8,
#         ffn_expansion_ratio=2.0,
#         drop_path_rate=0.08,
#         upsampler_type='dysample',
#         **kwargs
#     )


# def fastdat_medium(**kwargs):
#     """Medium variant - recommended balance."""
#     return FastDAT(
#         embed_dim=120,
#         num_groups=4,
#         depth_per_group=3,
#         num_heads=4,
#         window_size=8,
#         ffn_expansion_ratio=2.0,
#         drop_path_rate=0.1,
#         use_checkpoint=True,
#         upsampler_type='dysample',
#         **kwargs
#     )


# def fastdat_large(**kwargs):
#     """Large variant for maximum quality."""
#     return FastDAT(
#         embed_dim=144,
#         num_groups=6,
#         depth_per_group=3,
#         num_heads=6,
#         window_size=8,
#         ffn_expansion_ratio=2.5,
#         drop_path_rate=0.15,
#         use_checkpoint=True,
#         upsampler_type='dysample',
#         **kwargs
#     )

