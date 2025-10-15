
from copy import deepcopy
from hutils import is_access_granted
import json
import os
from pathlib import Path
from pprint import pprint
from warnings import warn
import zipfile

from pynnlib.metadata import generate_metadata

from pynnlib.model import TrtModel
from pynnlib.nn_types import ShapeStrategy
from pynnlib.import_libs import trt
_has_herlegon_system_ = False
try:
    from system import (
        GpuDevices,
        NvidiaGpu
    )
    _has_herlegon_system_ = True
except:
    pass


def generate_tensorrt_basename(
    model: TrtModel,
    basename: str
) -> str:
    dtypes = '_'.join([fp for fp in ('fp32', 'fp16', 'bf16') if fp in model.dtypes])
    opset = f"op{model.opset}"
    shape: str
    if model.shape_strategy.type in ('static', 'fixed'):
        shape = (
            f"{model.shape_strategy.type}_"
            + 'x'.join([str(x) for x in model.shape_strategy.opt_size])
        )

    else:
        shape_strategy = deepcopy(model.shape_strategy)
        if shape_strategy.min_size == (0, 0):
            shape_strategy.min_size = shape_strategy.opt_size
        if shape_strategy.max_size == (0, 0):
            shape_strategy.max_size = shape_strategy.opt_size
        shape = 'x'.join([str(x) for x in shape_strategy.min_size])
        shape += '_' + 'x'.join([str(x) for x in shape_strategy.opt_size])
        shape += '_' + 'x'.join([str(x) for x in shape_strategy.max_size])
    tensorrt_version = trt.__version__

    if _has_herlegon_system_:
        # TODO: correct this to use another GPU than first one
        nvidia_gpus: list[NvidiaGpu] = GpuDevices().get_nvidia_gpus()
        cc = nvidia_gpus[0].compute_capability
    else:
        import torch
        cc = '.'.join(map(str, torch.cuda.get_device_capability()))

    # Use the torch arch
    weak_typing: bool = False
    if model.torch_arch is not None:
        weak_typing = model.torch_arch.to_tensorrt.weak_typing
    else:
        weak_typing = model.arch.to_tensorrt.weak_typing

    weak_typed: str = (
        "_weak" if model.force_weak_typing or weak_typing else ""
    )

    return f"{basename}_cc{cc}_{opset}_{dtypes}_{shape}{weak_typed}_{tensorrt_version}"



def save_as(
    model: TrtModel,
    directory: str | Path,
    basename: str,
    suffix: str = "",
) -> bool:
    try:
        trt_engine = model.engine
    except:
        return False

    directory = str(directory) if isinstance(directory, Path) else directory
    if not is_access_granted(directory, 'w'):
        raise PermissionError(red(f"[E] \'{directory}\' is read only"))

    if basename == '':
        return False
    basename = generate_tensorrt_basename(model, basename)
    ext = '.trtzip'
    model.filepath = os.path.join(directory, f"{basename}{suffix}{ext}")

    generate_metadata(model, {})
    try:
        with zipfile.ZipFile(
            model.filepath, "w", compression=zipfile.ZIP_DEFLATED
        ) as trtzip_file:
            trtzip_file.writestr(
                f"{basename}{suffix}.engine",
                bytes(trt_engine.serialize())
            )
            trtzip_file.writestr(
                "metadata.json",
                json.dumps(model.metadata, indent=2)
            )
    except Exception as e:
        raise ValueError(red(f"[E] Failed to save engine as \'{model.filepath}\': {e}"))

    return True
