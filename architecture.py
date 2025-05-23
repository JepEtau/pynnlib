from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from pprint import pprint
import onnx
from pathlib import Path

try:
    import torch
    torch_device = torch.device
except:
    torch_device = str
    pass
from typing import Any, Literal, Optional, overload
from .model import (
    OnnxModel,
    NnModel,
    PyTorchModel,
    SizeConstraint,
)
from .nn_types import (
    Idtype,
    NnArchitectureType,
    ShapeStrategy,
)
from .session import NnModelSession


ParseFunction = Callable[[NnModel], None]


@dataclass(slots=True)
class InferType:
    type: Literal[
        'simple',
        'inpaint',
        'temporal',
        'optical_flow'
    ] = 'simple'
    inputs: int = 1
    outputs: int = 1



@dataclass
class NnGenericArchitecture:
    name: str = 'unknown'
    type: NnArchitectureType = NnArchitectureType()
    category: str = 'Generic'

    # Function used to detect arch,
    # can be customized for more scalability
    detect: tuple[Callable] | Callable = None

    """Parse a model object and update the model"""
    parse: ParseFunction | None = None

    create_session: Callable[[NnModel], NnModelSession] | None = None

    # TODO: remove this
    default: bool = False

    # Supported datatypes
    dtypes: list[Idtype] = field(default_factory=list)

    size_constraint: SizeConstraint | None = None

    # Simple, temporal, nb of inputs/outputs, etc.
    infer_type: InferType = field(default_factory=InferType)

    @overload
    def update(self, d: dict[Any, Any]) -> None: ...
    @overload
    def update(self, k: str, v:Any) -> None: ...
    def update(self,
        d: Optional[dict[Any, Any]] = None,
        k: Optional[str] = None,
        v: Optional[Any] = None):
        """Only existing attributes will be updated"""
        if d is not None:
            for k, v in d.items():
                if k is not None and hasattr(self, k):
                    setattr(self, k, v)

        elif k is not None and hasattr(self, k):
            setattr(self, k, v)

    def __str__(self) -> str:
        class_str = f"{self.__class__}: {'{'}\n"
        for k, v in self.__dict__.items():
            # if k in ['model_proto', 'state_dict', 'engine', 'arch', 'framework']:
            #     class_str += (
            #         f"\t{k}: {f'{type(v).__name__} ...' if v is not None else 'None'}\n"
            #     )
            #     continue
            class_str += f"\t{k}: {type(v).__name__} = {v}\n"
        class_str += "}\n"
        return class_str



ConvertToOnnxFct = Callable[
    [
        PyTorchModel,
        Idtype,
        int,
        bool,
        torch_device | str,
        ShapeStrategy,
        int
    ],
    onnx.ModelProto
]
"""Convert a Pytorch model to an Onnx model.

    Arguments:
        model: PytorchModel
        dtype: Idtype
        opset: int
        static: bool = False
        device: str = 'cpu'
        batch: int = 1
"""


ConvertToTensorrtFct = Callable[
    [PyTorchModel | OnnxModel, str, bool, Any, bool],
    bytes
]


@dataclass
class NnPytorchArchitecture(NnGenericArchitecture):
    detection_keys: tuple[str | tuple[str]] | dict = ()
    """Convert a model from pytorch to onnx"""
    to_onnx: ConvertToOnnxFct | None = None
    to_tensorrt: ConvertToTensorrtFct | None = None
    # TODO add dtypes supported for conversion to tensorRT ???
    #   example: SCUNET does not support conversion to fp16


@dataclass
class NnOnnxArchitecture(NnGenericArchitecture):
    scale: int | None = None
    to_tensorrt: Callable | None = None


@dataclass
class NnTensorrtArchitecture(NnGenericArchitecture):
    version: str = ''


NnArchitecture = (
    NnGenericArchitecture
    | NnOnnxArchitecture
    | NnPytorchArchitecture
    | NnTensorrtArchitecture
)


# GetModelArchFct = Callable[
#     [str | Path, dict[str, dict], str | Any],
#     tuple[NnArchitecture | None, NnModelObject | None]
# ]

GetModelArchFct = Callable[[str | Path, dict[str, Any]], tuple[str, Any]]

def detect_model_arch(
    model: Any,
    architectures: dict[str, NnArchitecture]
) -> NnArchitecture:
    """Detect the model architecture and returns it"""
    for arch in architectures.values():
        if (detect_fct := arch.detect) is None:
            raise NotImplementedError(f"Detection function is not implemented for {arch.name}")

        if isinstance(detect_fct, tuple | list):
            # List of functions for detection
            for func in detect_fct:
                if func(model):
                    return arch
        else:
            # Use a customized function for detection
            if detect_fct(model):
                return arch

    return None
