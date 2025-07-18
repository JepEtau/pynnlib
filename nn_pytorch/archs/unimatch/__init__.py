from typing import Literal
from warnings import warn
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from pynnlib.utils.p_print import *
from ...torch_types import StateDict
from ..helpers import get_nsequences
from ..torch_to_onnx import to_onnx

# Impossible to differentiate unimatch from GMFlow

def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 8
    in_nc: int = 3
    out_nc: int = 3

    is_refine: bool = bool("refine_proj.weight" in state_dict)

    is_stereo: bool = False
    if "refine.flow_head.conv2.weight" in state_dict:
        is_stereo = bool(
            state_dict["refine.flow_head.conv2.weight"].shape[0] == 1
            and is_refine
        )

    num_scales: int = 1
    if "backbone.trident_conv.weight" in state_dict:
        num_scales = 2
        scale = 4

    if (
        (is_stereo and not "stereo" in model.filepath)
        or (not is_stereo and "stereo" in model.filepath)
    ):
        warn(yellow(f"  Error: wrong task name, detected: {'stereo' if is_stereo else 'not stereo'}"))

    # Cannot determinate if flow, depth or stereo
    task: Literal['flow', 'depth', 'stereo'] = 'flow'
    if 'flow' in model.filepath:
        task = 'flow'
    if 'depth' in model.filepath:
        task = 'depth'
    if 'stereo' in model.filepath:
        task = 'stereo'

    num_transformer_layers = get_nsequences(
        state_dict=state_dict,
        seq_key="transformer.layers"
    )

    # always 128
    feature_channels = state_dict["backbone.conv2.weight"].shape[0]
    if feature_channels != 128:
        raise ValueError(red(f"Wrong detected feature_channels: {feature_channels}, should be 128"))

    arch_name = f"{model.arch.name} ({task})"

    # from .module.unimatch import UniMatch
    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        task=task,
        reg_refine=is_refine,
        num_scales=num_scales,
        upsample_factor=scale,
        num_transformer_layers=num_transformer_layers,
        feature_channels=feature_channels
    )
    # padding_factor = 32


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="unimatch",
        detection_keys=(
            # "refine.encoder.convc1.weight",
            "feature_flow_attn.q_proj.weight",
            "backbone.conv1.weight",
            "backbone.conv2.weight",
        ),
        module=Module(file="unimatch", class_name="UniMatch"),
        parse=parse,
        dtypes=('fp32', 'fp16'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        ),
        # to_onnx=to_onnx,
    ),
)
