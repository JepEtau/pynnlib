from __future__ import annotations
import os
from pprint import pprint
import re

from .nn_onnx.archs.save import generate_onnx_basename

from .architecture import NnTensorrtArchitecture
from .nn_types import Hdtype, ShapeStrategy
from .core import nn_lib as nnlib

from .model import NnModel, OnnxModel, PyTorchModel, TrtModel
from .framework import NnFrameworkType
from hytils import (
    get_extension,
    path_split,
    path_basename,
    red,
)
try:
    from .nn_tensor_rt.archs.save import generate_tensorrt_basename
except:
    # nnlogger.debug("[W] TensorRT is not supported: model cannot be converted")
    def generate_tensorrt_basename(*args) -> str:
        raise RuntimeError("TensorRT is not supported")


def save_as(
    model_fp: str,
    model: NnModel,
    autonaming: bool = False,
) -> None:
    directory, basename, ext = path_split(model_fp)
    out_model_fp = model_fp if not autonaming else None

    if not directory:
        directory="./"

    if model.framework.type == NnFrameworkType.PYTORCH:
        # PyTorch, SafeTensors
        model.framework.save(
            filepath=out_model_fp,
            model=model,
            directory=directory,
            basename=basename,
            ext=ext
        )

    elif model.framework.type == NnFrameworkType.ONNX:
        model.framework.save(
            model=model,
            filepath=out_model_fp,
            directory=directory,
            basename=basename,
            suffix="",
        )

    elif model.framework.type == NnFrameworkType.TENSORRT:
        model.framework.save(
            model=model,
            filepath=out_model_fp,
            directory=directory,
            basename=basename,
            suffix="",
        )

    else:
        raise ValueError(get_extension(f"Unknown framework: {model.framework.type}"))



def generate_out_model_fp(
    model: NnModel,
    to: NnFrameworkType,
    opset: int = 21,
    dtype: Hdtype = 'fp32',
    force_weak_typing: bool = False,
    device: str = 'cpu',
    out_dir: str = "",
    shape_strategy: ShapeStrategy = ShapeStrategy(type='dynamic'),
    suffix: str = "",
) -> str:
    """Generate the output filepath using a dummy model.
    returns an empty str if not supported or erroneous arg
    """
    out_model_fp: str = ""
    basename = path_basename(model.filepath)
    suffix = f"_{suffix}" if suffix else ""

    dummy_model: NnModel
    if to == NnFrameworkType.ONNX:
        if model.fwk_type != NnFrameworkType.PYTORCH:
            return ""

        dummy_model: OnnxModel = OnnxModel(
            filepath="",
            framework=nnlib.frameworks[NnFrameworkType.ONNX],
            arch=model.arch,
            model_proto=None,

            opset=opset,
            torch_arch=model.arch,
            force_weak_typing=force_weak_typing,
            dtypes=set([dtype]),
            shape_strategy=shape_strategy,
        )
        onnx_basename: str = generate_onnx_basename(dummy_model, basename)
        out_model_fp = os.path.join(out_dir, f"{onnx_basename}{suffix}.onnx")

    elif to == NnFrameworkType.TENSORRT:
        torch_arch: PyTorchModel = model.arch

        if model.fwk_type == NnFrameworkType.PYTORCH:
            if torch_arch.to_tensorrt is None:
                raise ValueError(red(f"[E]: Conversion to tensorRT: arch \'{torch_arch.name}\' is not supported"))
            if dtype not in torch_arch.to_tensorrt.dtypes:
                raise ValueError(red(f"[E]: Conversion to tensorRT: dtype \'{dtype}\' is not supported"))

        elif model.fwk_type == NnFrameworkType.ONNX:
            opset = model.opset
            basename = re.sub(r"_op\d{1,2}", '', basename)
            for dt in ('_fp32', '_fp16', '_bf16', '_int8'):
                basename = basename.replace(dt, '')
        else:
            raise ValueError(f"Cannot convert to {to.value}")

        if NnFrameworkType.TENSORRT not in nnlib.frameworks:
            raise ValueError("TensorRT framework is not supported")
        dummy_model: TrtModel = TrtModel(
            framework=nnlib.frameworks[NnFrameworkType.TENSORRT],
            arch=NnTensorrtArchitecture,
            torch_arch=torch_arch,
            filepath="",
            device=device,
            dtypes=set([dtype]),
            force_weak_typing=force_weak_typing,
            engine=None,
            shape_strategy=shape_strategy,
            opset=opset,
        )
        trt_basename: str = generate_tensorrt_basename(dummy_model, basename)
        out_model_fp = os.path.join(out_dir, f"{trt_basename}{suffix}.trtzip")

    else:
        raise ValueError(f"Cannot convert to {to.value}")

    return out_model_fp
