
from copy import deepcopy
import json
import os
from pathlib import Path
from pprint import pprint
import re
from warnings import warn
import zipfile

from pynnlib.metadata import set_metadata_

from ...utils import is_access_granted
from ...utils.p_print import *
from pynnlib.model import TrtModel
from pynnlib.nn_types import ShapeStrategy
import tensorrt as trt
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
    if model.shape_strategy.static:
        shape = "static_" + 'x'.join([str(x) for x in model.shape_strategy.opt_size])
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

    return f"{basename}_cc{cc}_{opset}_{dtypes}_{shape}_{tensorrt_version}"


def basename_to_config(
    basename: str
) -> dict[str, str | int | tuple[int,int] | ShapeStrategy] | None:
    config = None

    if (
        match := re.match(
            re.compile(
                r".*_cc(\d+.\d+)_([bfp1236_]+)_op(\d{1,2})_(\d+x\d+)_(\d+x\d+)_(\d+x\d+)_(\d+\.\d+\.\d+)$"
            ),
            basename
        )
    ):
        #.......
        dtypes = match.group(1)
        fp16: bool = 'fp16' in dtypes
        fp32: bool = 'fp32' in dtypes
        bf16: bool = 'bf16' in dtypes
        opset = int(match.group(2))
        shape_strategy: ShapeStrategy = ShapeStrategy(
            static=False,
            min_size=[int(x) for x in match.group(3).split('x')],
            opt_size=[int(x) for x in match.group(4).split('x')],
            max_size=[int(x) for x in match.group(5).split('x')],
        )
        tensorrt_version: str = match.group(6)

    # elif (match := re.match(re.compile("....."))):
    else:
        return None

    config = dict(
        fp16=fp16,
        fp32=fp32,
        bf16=bf16,
        opset=opset,
        shape_strategy=shape_strategy,
        tensorrt_version=tensorrt_version
    )
    return config



def save(
    model: TrtModel,
    directory: str | Path,
    basename: str,
    suffix: str | None = None,
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
    suffix = suffix if suffix is not None else ''
    ext = '.trtzip'
    model.filepath = os.path.join(directory, f"{basename}{suffix}{ext}")

    set_metadata_(model, {})

    print(model)

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
