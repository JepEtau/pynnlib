from __future__ import annotations
from collections.abc import Callable, Set
from dataclasses import dataclass, field
from importlib import util as importlib_util
import inspect
import os
import sys
import onnx
from pathlib import Path

try:
    import torch
    torch_device = torch.device
except:
    torch_device = str
from torch import nn as nn
from typing import Any, Literal, Optional, overload
from .model import (
    OnnxModel,
    NnModel,
    PyTorchModel,
    SizeConstraint,
)
from .logger import nnlogger
from .nn_types import (
    Idtype,
    NnArchitectureType,
    ShapeStrategy,
    ShapeStrategyType,
)
from .session import NnModelSession
from .utils import absolute_path, path_split


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
    category: str = 'unknown'

    # Function used to detect arch,
    # can be customized for more scalability
    detect: tuple[Callable] | Callable = None

    """Parse a model object and update the model"""
    parse: ParseFunction | None = None

    create_session: Callable[[NnModel], NnModelSession] | None = None

    # Supported datatypes for inference and conversion
    #   TODO: as it may differ, use a dataclass
    dtypes: list[Idtype] = field(default_factory=list)

    size_constraint: SizeConstraint | None = None

    # Simple, temporal, nb of inputs/outputs, etc.
    infer_type: InferType = field(default_factory=InferType)

    _locked: bool = field(default=False, init=False, repr=False)

    def lock(self):
        self._locked = True

    def __setattr__(self, key, value):
        if getattr(self, '_locked', False) and key != '_locked':
            raise AttributeError(f"Cannot modify '{key}'; object is locked.")
        super().__setattr__(key, value)

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


@dataclass(slots=True)
class TensorRTConv:
    # Some archs don't support strong typing,
    #   caution: conversion might fail or slower inference
    dtypes: Set[Idtype] = field(
        default_factory=lambda: {'fp32', 'fp16', 'bf16'}
    )
    weak_typing: bool = False
    shape_strategy_types: Set[ShapeStrategyType] = field(
        default_factory=lambda: {'dynamic', 'fixed', 'static'}
    )


@dataclass
class Module:
    file: str = ""
    class_name: str = ""
    module_class: nn.Module | None = None


@dataclass
class NnPytorchArchitecture(NnGenericArchitecture):
    detection_keys: tuple[str | tuple[str]] | dict = field(default_factory=tuple)
    module: Module = field(default_factory=Module)

    to_onnx: ConvertToOnnxFct | None = None
    to_tensorrt: TensorRTConv | None = None

    __caller_dir: str = ""


    def __post_init__(self):
        frame = inspect.currentframe()
        try:
            # Go up two levels: one for the __post_init__ call,
            #   and one for the instantiation
            caller_file = frame.f_back.f_back.f_globals.get('__file__', 'unknown')
        finally:
            del frame
        self.__caller_dir= absolute_path(path_split(caller_file)[0])


    def import_module(self) -> None:
        """Dynamic import the nn.module
        """
        if self.module.module_class is not None:
            return

        pynnlib_dir = absolute_path(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
        )
        arch_module_key: str = (
            self.__caller_dir[len(pynnlib_dir) + 1:].replace(os.sep, ".")
        )

        arch_module: Module = self.module
        if any(not x for x in (arch_module.file, arch_module.class_name)):
            print("[W] Missing module package or class_name")

        nn_module_filepath = os.path.join(
            self.__caller_dir, "module", arch_module.file.replace(".", os.sep)
        )
        if not nn_module_filepath.endswith(".py"):
            nn_module_filepath += ".py"

        sys_module_key = f"{arch_module_key}.module.{arch_module.file}"
        module_spec = importlib_util.spec_from_file_location(
            name=sys_module_key,
            location=nn_module_filepath
        )
        module = importlib_util.module_from_spec(module_spec)
        sys.modules[sys_module_key] = module
        module_spec.loader.exec_module(module)
        self.module.module_class = getattr(module, arch_module.class_name)



@dataclass
class NnOnnxArchitecture(NnGenericArchitecture):
    scale: int | None = None
    to_tensorrt: TensorRTConv | None = None


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
                    nnlogger.debug(f"[V] detected arch: {arch.name}")
                    return arch
        else:
            # Use a customized function for detection
            if detect_fct(model):
                nnlogger.debug(f"[V] detected arch: {arch.name}")
                return arch

    return None
