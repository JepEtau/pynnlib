from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    OnnxConv,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.logger import is_debugging
from pynnlib.model import PyTorchModel
from pynnlib.nn_pytorch.archs import contains_all_keys
from ...torch_types import StateDict



def lhan_detect(state_dict: StateDict, keys: tuple[str | tuple[str]]) -> bool:
    if not contains_all_keys(state_dict=state_dict, keys=keys):
        return False
    # differentiate from FastDAT
    if "upsampler.MetaUpsample" in state_dict:
        return False
    return True



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
        num_groups = 2
        depth_per_group = 2
        num_heads = 3
        ffn_expansion_ratio = 1.5
        drop_path_rate = 0.05
        upsampler_type = 'pixelshuffle'

    # Light
    elif embed_dim == 108:
        arch_name = f"{model.arch.name} (light)"
        num_groups = 3
        depth_per_group = 2
        num_heads = 4
        ffn_expansion_ratio = 2.0
        drop_path_rate = 0.08
        upsampler_type = 'nearest_conv'

    # Medium
    elif embed_dim == 120:
        arch_name = f"{model.arch.name} (medium)"
        num_groups = 4
        depth_per_group = 3
        num_heads = 4
        ffn_expansion_ratio = 2.0
        drop_path_rate = 0.1
        upsampler_type = 'transpose_conv'

    else:
        raise ValueError("Variant not found")

    window_size: int = 8
    aim_reduction_ratio: int = 8

    if is_debugging():
        from .module.lhan_arch import Lhan
        model.update(ModuleClass=Lhan)

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
        detect=lhan_detect,
        detection_keys=(
            "groups.0.blocks.3.ffn.smix.weight",
            "groups.1.blocks.3.ffn.smix.weight",
        ),
        module=Module(file="lhan_arch", class_name="Lhan"),
        parse=parse,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8)),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16', 'bf16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16', 'bf16']),
        ),
    ),
)




