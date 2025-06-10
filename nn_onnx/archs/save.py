from __future__ import annotations
from datetime import datetime
import json
import onnx
from onnx import helper
import os
from pathlib import Path
import time
from pynnlib.utils import is_access_granted
from pynnlib.model import OnnxModel


def generate_onnx_basename(
    model: OnnxModel,
    basename: str
) -> str:
    dtypes = '_'.join(
        [fp for fp in ('fp32', 'fp16', 'bf16') if fp in model.dtypes]
    )
    shape: str = ""
    if model.shape_strategy.static or "static" in model.shape_strategy.type:
        shape = "_static_" + 'x'.join([str(x) for x in model.shape_strategy.opt_size])

    return f"{basename}_op{model.opset}_{dtypes}{shape}"


def save(
    model: OnnxModel,
    directory: str | Path,
    basename: str,
    suffix: str | None = None
) -> bool:
    model_proto: onnx.ModelProto = model.model_proto
    if model_proto is None:
        return False

    directory = str(directory) if isinstance(directory, Path) else directory
    if not is_access_granted(directory, 'w'):
        raise PermissionError(f"{directory} is read only")

    if basename == '':
        return False
    basename = generate_onnx_basename(model, basename)
    suffix = suffix if suffix is not None else ''
    filepath = os.path.join(directory, f"{basename}.onnx")
    model.filepath = filepath

    # Append metadata
    metadata = {
        'application': 'pynnlib',
        'arch_name': model.arch_name,
        'name': basename,
        'scale': f"{model.scale}",
        'datetime': datetime.strptime(
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            '%Y-%m-%dT%H:%M:%S%z').isoformat(),
        'license': model.metadata.get('license', ''),
        'author': model.metadata.get('author', ''),
        'version': model.metadata.get('version', ''),
    }
    # entry = model_proto.metadata_props.add()
    # entry.key = 'info'
    # entry.value = json.dumps(metadata).encode('utf-8')
    for k, v in metadata.items():
        if not isinstance(v, str):
            v = str(v)
        entry = model_proto.metadata_props.add()
        entry.key = k
        entry.value = v


    try:
        onnx.checker.check_model(model=model_proto)
        onnx.save(model_proto, filepath)
    except Exception as e:
        print(f"[E] Failed to save onnx model: {e}")
        return False

    return True
