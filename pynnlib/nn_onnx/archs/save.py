from __future__ import annotations
from hutils import (
    is_access_granted,
    parent_directory,
)
from warnings import warn
import onnx
import os
from pathlib import Path
from pynnlib.model import OnnxModel



def generate_onnx_basename(
    model: OnnxModel,
    basename: str
) -> str:
    dtypes = '_'.join(
        [fp for fp in ('fp32', 'fp16', 'bf16', 'int8') if fp in model.dtypes]
    )
    shape: str = ""
    if model.shape_strategy is None:
        raise ValueError(f"the shape strategy must be specified")

    if model.shape_strategy.type == 'static':
        # verify that the shape is valid
        if not model.shape_strategy.is_valid():
            raise ValueError(f"the shape strategy is not valid for strategy=\'{model.shape_strategy.type}\'")
        shape = "_static_" + 'x'.join([str(x) for x in model.shape_strategy.opt_size])

    return f"{basename}_op{model.opset}_{dtypes}{shape}"


def save_as(
    model: OnnxModel,
    filepath: str | Path | None = None,
    directory: str | Path | None = None,
    basename: str | None = None,
    suffix: str = "",
) -> str:

    model_proto: onnx.ModelProto = model.model_proto
    if model_proto is None:
        return ""

    if filepath is not None:
        directory = parent_directory(filepath)

    else:
        # Generate filepath based on args
        directory = str(directory) if isinstance(directory, Path) else directory
        if not basename:
            return ""
        basename = generate_onnx_basename(model, basename)
        filepath = os.path.join(directory, f"{basename}{suffix}.onnx")

    if not is_access_granted(directory, 'w'):
        raise PermissionError(f"{directory} is read only")

    # Add metadata to the proto
    del model_proto.metadata_props[:]
    for k, v in model.metadata.items():
        if not isinstance(v, str):
            try:
                v = str(v)
            except:
                warn(f"Value for key \'{k}\' cannot be converted to a valid string")
                continue
        entry = model_proto.metadata_props.add()
        entry.key = k
        entry.value = v

    try:
        onnx.checker.check_model(model=model_proto)
        onnx.save(model_proto, filepath)
    except Exception as e:
        print(f"[E] Failed to save onnx model: {e}")
        return ""

    return filepath
