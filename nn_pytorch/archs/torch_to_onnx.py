from __future__ import annotations
from io import BytesIO
from pprint import pprint
from typing import TYPE_CHECKING
from contextlib import nullcontext

import onnx
import torch

from pynnlib.architecture import SizeConstraint
from pynnlib.import_libs import is_cuda_available
from pynnlib.nn_types import Idtype, ShapeStrategy
from pynnlib.utils.torch_tensor import IdtypeToTorch
from ..inference.session import PyTorchSession
if TYPE_CHECKING:
    from pynnlib.model import PyTorchModel
from pynnlib.utils.p_print import *


def to_onnx(
    model: PyTorchModel,
    dtype: Idtype,
    opset: int,
    shape_strategy: ShapeStrategy | None = None,
    device: str = 'cpu',
    batch: int = 1,
) -> onnx.ModelProto | None:
    """Returns an Onnx model as a byte buffer
        if size is not None, use it to convert to a static shape
    """
    # TODO use model.arch to determine if static is possible or not
    print(f"[V] PyTorch to ONNX")

    try:
        session: PyTorchSession = model.arch.create_session(model)
    except:
        raise NotImplementedError(red(f"{model.arch.name} is not supported"))
        return None

    # if dtype == 'bf16':
    #     raise NotImplementedError("Conversion to bf16 is not supported yet.")

    fp16: bool = bool(dtype == 'fp16')
    if not is_cuda_available():
        print("[W] cuda not available, fallback to cpu")
        device = 'cpu'
        fp16 = False

    # https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-convert-model
    print(f"[V] PyTorch to ONNX: initialize a session, device={device}, dtype={dtype}, fp16={fp16}, opset={opset}")
    session.initialize(device=device, dtype=dtype)

    print(f"[V]  shape strategy: {shape_strategy}")
    dynamic_axes: dict[str, str] | None = None
    if shape_strategy is not None:
        if shape_strategy.type == 'static':
            print("[V]  shape strategy is static")
            w, h = shape_strategy.opt_size
            # TODO: verify against constraint

        else:
            print("[V]  shape strategy is dynamic or fixed")
            if batch == 1:
                dynamic_axes = {
                    'input': {2: "height", 3: "width"},
                    'output': {2: "height", 3: "width"},
                }

            else:
                dynamic_axes = {
                    'input': {0: "batch", 2: "height", 3: "width"},
                    'output': {0: "batch", 2: "height", 3: "width"},
                }

            size: SizeConstraint | None = model.size_constraint
            w, h = size.min if size is not None and size.min is not None else (32, 32)

    else:
        print("[V]  shape strategy is None")
        dynamic_axes = {
            'input': {0: "batch", 2: "height", 3: "width"},
            'output': {0: "batch", 2: "height", 3: "width"},
        }
        size: SizeConstraint | None = model.size_constraint
        w, h = size.min if size is not None and size.min is not None else (32, 32)

    dummy_input = torch.rand(
        batch,
        model.in_nc,
        h,
        w,
        device=device,
        requires_grad=True
    )
    tensor_dtype: torch.dtype = IdtypeToTorch[dtype]
    dummy_input = dummy_input.to(dtype=tensor_dtype)
    # dummy_input = dummy_input.half() if fp16 else dummy_input.float()
    # dummy_input = dummy_input.float()
    print(f"[V]   use a dummy input: {dummy_input.shape}, {dummy_input.dtype}, {dummy_input.device}")

    model_proto: onnx.ModelProto
    if tensor_dtype == torch.bfloat16:
        context = torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=tensor_dtype
        )
    else:
        context = nullcontext()

    with context:
        with BytesIO() as bytes_io:
            torch.onnx.export(
                session.module,
                dummy_input,
                bytes_io,
                export_params=True,
                input_names=['input'],
                output_names=['output'],
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
            bytes_io.seek(0)
            model_proto = onnx.load(bytes_io)

        try:
            onnx.checker.check_model(model_proto)
        except Exception as e:
            print(f"[E] Converted Onnx model is not valid: {type(e)}")
            return None

    print(f"[V] ONNX model proto generated")

    return model_proto

