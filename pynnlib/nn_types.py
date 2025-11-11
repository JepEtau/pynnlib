from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import os
import onnx
from pathlib import PurePath
from typing import Literal, TypeAlias

from .nn_pytorch.torch_types import StateDict
from .nn_tensor_rt.trt_types import TrtEngine


NnArchitectureType: TypeAlias = str

class NnFrameworkType(Enum):
    ONNX = 'ONNX'
    PYTORCH = 'PyTorch'
    TENSORRT = 'TensorRT'


def filepath_to_fwk(filepath: str) -> NnFrameworkType:
    # Cannot use _value2member_map_ because of lower/upper case
    value = PurePath(os.path.dirname(filepath)).parts[-1].replace('nn_', '')
    enum_dict = {t.value.lower(): t.value for t in NnFrameworkType}
    try:
        return NnFrameworkType._value2member_map_[enum_dict[value]]
    except:
        raise ValueError(f"[E] {value} is not a valid framework key")


NnModelObject = onnx.ModelProto | StateDict | TrtEngine

Hdtype = Literal['fp32', 'fp16', 'bf16', 'int8']

ShapeStrategyType = Literal[
    # Conversion to:
    #   ONNX: static, opt size must be specified
    #   TensorRT: Static or dynamic Onnx, depends on ONNX strategy, fixed TensorRT shapes
    'static',

    # Only for TensorRT: static or dynamic Onnx, fixed TensorRT shapes
    #   (if used with conversion to ONNX -> static ONNX strategy)
    'fixed',

    # Use dynamic shapes for both ONNX and tensorRT
    'dynamic'
]


@dataclass
class ShapeStrategy:
    """Shapes: (width, height)
    """
    type: ShapeStrategyType = 'dynamic'
    min_size: tuple[int, int] = (0, 0)
    opt_size: tuple[int, int] = (0, 0)
    max_size: tuple[int, int] = (0, 0)


    def __post_init__(self):
        self._modulo: int = 1


    def is_valid(self) -> bool:
        if self.type == 'static':
            if any(x == 0 for x in self.opt_size):
                return False
        else:
            for d in range(2):
                values = [x[d] for x in (self.min_size, self.opt_size, self.max_size)]
                if min([values[i+1] - values[i] for i in range(len(values)-1)]) < 0:
                    return False
        return True


    def is_fixed(self):
        if self.type != 'dynamic':
            return True

        w, h = self.opt_size
        for size in (self.min_size, self.max_size):
            if size[0] != w or size[1] != h:
                return False
        return True


    def __str__(self) -> str:
        class_str = "{\n"
        indent: str = "    "
        for k, v in self.__dict__.items():
            v_str = f"\'{v}\'" if isinstance(v, str) else f"{v}"
            class_str += f"{indent}{indent}{k}: {type(v).__name__} = {v_str}\n"
        class_str += f"{indent}{'}'}\n"
        return class_str
