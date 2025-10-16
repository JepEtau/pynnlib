from __future__ import annotations
from dataclasses import dataclass, field
import onnx
from typing import TYPE_CHECKING, Literal
from hutils import arg_list

from torch import nn
from .nn_pytorch.torch_types import StateDict
from .nn_tensor_rt.trt_types import TrtEngine
from .nn_types import Idtype, NnFrameworkType, ShapeStrategy
if TYPE_CHECKING:
    from .architecture import (
        NnArchitecture,
        SizeConstraint,
        NnPytorchArchitecture,
    )
    from .nn_types import NnModelDtype, NnFrameworkType
    from .framework import NnFramework



@dataclass(slots=True)
class SizeConstraint:
    min: tuple[int, int] = None
    max: tuple[int, int] = None
    modulo: int = 1

    def is_size_valid(
        self,
        size_or_shape: tuple[int, int, int] | tuple[int, int],
        is_shape: bool = True
    ) -> bool:
        """Return True if the size is valid.
        The size can be provided as a np.shape (h,w,c) or as a tuple of dims (w, h)
        TODO: verify modulo
        """
        if is_shape:
            h, w = size_or_shape[:2]
        else:
            w, h = size_or_shape[:2]
        if self.min is not None:
            if w < self.min[0] or h < self.min[1]:
                return False
        if self.max is not None:
            if w > self.max[0] or h > self.max[1]:
                return False
        return True



@dataclass(slots=True)
class ModelExecutor:
    """Used for inference in a mp"""
    device: str
    dtype: Idtype
    pinned_mem: bool
    in_max_shape: tuple[int, int, int]



@dataclass(slots=True)
class GenericModel:
    framework: NnFramework
    arch: NnArchitecture

    scale: int = 0
    in_nc: int = 0
    out_nc: int = 0
    # list IO dtypes, consider a single input/output
    io_dtypes: dict[Literal['input', 'output'], NnModelDtype] = (
        field(default_factory=dict)
    )

    filepath: str = None

    # Use this device to parse a TensorRT model
    # TBD: for inference?
    device: str = 'cpu'

    # tensorRT and Onnx only: indicates which dtype
    # is supported by this model.
    # PyTorch: use the dtypes specified by each arch
    # TODO: move this as a property like size_constraint
    dtypes: list[NnModelDtype] = field(default_factory=list)
    force_weak_typing: bool = False

    # Object used to initialize an executor.
    # when in multiprocess. Useless otherwise
    executor: ModelExecutor = None

    metadata: dict[str, str] = field(default_factory=dict)

    # Shape strategy when converting to onnx/tensorrt
    shape_strategy: ShapeStrategy = field(default_factory=ShapeStrategy)

   # Private fields
    _arch_name: str = field(default="", init=False, repr=False)
    _size_constraint: SizeConstraint | None = field(default=None, init=False, repr=False)


    @property
    def fwk_type(self) -> NnFrameworkType:
        return self.framework.type


    @property
    def arch_name(self) -> str:
        if self.arch is None and not self._arch_name:
            return "unknown"
        elif self._arch_name:
            return self._arch_name
        return self.arch.name


    @arch_name.setter
    def arch_name(self, name: str = "") -> None:
        """Set to an empty string so that the property returns the arch name
            and not the overwritten one
        """
        self._arch_name = name


    def update(self, **kwargs):
        """Update multiple fields of this class.
            \nRaises an exception if a key is not defined.
        """
        # Get all slot names (field names) from the class hierarchy
        defined_keys: list[str] = []
        for cls in type(self).__mro__:
            if hasattr(cls, '__slots__'):
                defined_keys.extend(cls.__slots__)

        # Append the list of arguments used by a PyTorch nn.Module
        module_key: str = 'ModuleClass'
        module = None
        if module_key in defined_keys:
            module = (
                kwargs[module_key]
                if module_key in kwargs
                else getattr(self, module_key, None)
            )
            # Append args of the module
            if module is not None:
                defined_keys.extend(arg_list(module))

        for key, value in kwargs.items():
            if module is not None and key not in defined_keys:
                raise KeyError(f"Undefined key '{key}' in {self.__class__.__name__}")
            setattr(self, key, value)


    def __str__(self) -> str:
        class_str = f"{self.__class__}: {'{'}\n"
        indent: str = "    "

        for cls in type(self).__mro__:
            if hasattr(cls, '__slots__'):
                for k in cls.__slots__:
                    if not hasattr(self, k):
                        continue

                    v = getattr(self, k)

                    # Do not print content if too complex
                    if k in ['model_proto', 'state_dict', 'engine', 'arch', 'framework']:
                        class_str += (
                            f"{indent}{k}: {f'{type(v).__name__} ...' if v is not None else 'None'}\n"
                        )
                        continue

                    # dict
                    if isinstance(v, dict):
                        class_str += f"{indent}{k}: {type(v).__name__} = {{\n{indent}{indent}"
                        items = []
                        for key, value in v.items():
                            items.append(
                                f"'{key}': '{value}'"
                                if isinstance(value, str)
                                else f"'{key}': {value}"
                            )
                        class_str += f",\n{indent}{indent}".join(items)
                        class_str += f"\n{indent}}}\n"
                    else:
                        v_str = f"'{v}'" if isinstance(v, str) else f"{v}"
                        class_str += f"{indent}{k}: {type(v).__name__} = {v_str}\n"

        class_str += "}\n"
        return class_str


    def supported_dtypes(self) -> set[NnModelDtype]:
        """Returns supported dtypes by this model
        """
        return set(
            self.arch.dtypes
            if self.arch.type != NnFrameworkType.TENSORRT
            else self.dtypes
        )

    @property
    def size_constraint(self) -> SizeConstraint:
        return (
            self.arch.size_constraint
            if self._size_constraint is None
            else self._size_constraint
        )


    @size_constraint.setter
    def size_constraint(self, size_constraint: SizeConstraint) -> None:
        self._size_constraint = size_constraint


    def is_size_valid(
        self,
        size_or_shape: tuple[int, int, int] | tuple[int, int],
        is_shape: bool = True
    ) -> bool:
        """Return True if the size is valid.
        The size can be provided as a np.shape (h,w,c) or as a tuple of dims (w, h)
        TODO: verify modulo
        """
        return (
            self.size_constraint.is_size_valid(size_or_shape, is_shape)
            if self.size_constraint is not None
            else True
        )



@dataclass(slots=True)
class OnnxModel(GenericModel):
    model_proto: onnx.ModelProto = None
    opset: int = 21
    alt_arch_name: str = ''
    in_shape_order: str = 'NCHW'
    torch_arch: NnPytorchArchitecture = None



@dataclass(slots=True)
class PyTorchModel(GenericModel):
    state_dict: StateDict = field(default_factory=StateDict)
    num_feat: int = 0
    num_conv: int = 0
    ModuleClass: nn.Module = None
    module: nn.Module = None



@dataclass(slots=True)
class TrtModel(GenericModel):
    engine: TrtEngine = None
    engine_version: int = 0
    opset: int = 21
    # Put here the device id? no? used for conversion load
    device: str = ""
    torch_arch: NnPytorchArchitecture = None

    # For information only
    typing: Literal['', 'weak', 'strong'] = ''


NnModel = OnnxModel | PyTorchModel | TrtModel
