from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypeAlias

from ..import_libs import is_tensorrt_available
from ..logger import nnlogger
if is_tensorrt_available():
    import tensorrt_rtx as trt
    # from tensorrt import ICudaEngine
    from trt import ICudaEngine
    TrtEngine: TypeAlias = ICudaEngine
else:
    nnlogger.debug("[W] Importing ICudaEngine failed")
    TrtEngine: TypeAlias = Any

@dataclass
class TensorrtModel:
    engine: TrtEngine
