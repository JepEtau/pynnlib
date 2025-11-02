from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
import logging
import onnx
from typing import TYPE_CHECKING, Literal
from hutils import arg_list

from torch import nn
from .logger import nnlogger
from .nn_pytorch.torch_types import StateDict
from .nn_tensor_rt.trt_types import TrtEngine
from .nn_types import Idtype, NnFrameworkType, ShapeStrategy
if TYPE_CHECKING:
    from .architecture import (
        NnArchitecture,
        SizeConstraint,
        NnPytorchArchitecture,
    )
    from .nn_types import NnModelDtype, NnFrameworkType, NnModelObject
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


@dataclass
class GenericModel:
    framework: NnFramework
    arch: NnArchitecture
    alt_arch_name: str = ''

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
        if self.alt_arch_name:
            return self.alt_arch_name
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
        defined_keys: list[str] = list(self.__dict__.keys())

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
                raise KeyError(f"Undefined key \'{key}\' in {self.__class__.__name__}")
            setattr(self, key, value)


    def __str__(self) -> str:
        class_str = f"{self.__class__}: {'{'}\n"
        indent: str = "    "
        for k, v in self.__dict__.items():
            # Do not print content if too complex
            if k in ['model_proto', 'state_dict', 'engine', 'arch', 'framework']:
                class_str += (
                    f"{indent}{k}: {f'{type(v).__name__} ...' if v is not None else 'None'}\n"
                )
                continue
            # dict
            if isinstance(v, dict):
                class_str += f"{indent}{k}: {type(v).__name__} = {'{'}\n{indent}{indent}"
                items = []
                for key, value in v.items():
                    items.append(
                        f"\'{key}\': \'{value}\'"
                        if isinstance(value, str)
                        else f"\'{key}\': {value}"
                    )
                class_str += f",\n{indent}{indent}".join(items)
                class_str += f"\n{indent}{'}'}\n"

            else:
                v_str = f"\'{v}\'" if isinstance(v, str) else f"{v}"
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



@dataclass
class OnnxModel(GenericModel):
    model_proto: onnx.ModelProto = None
    opset: int = 21
    alt_arch_name: str = ''
    in_shape_order: str = 'NCHW'
    torch_arch: NnPytorchArchitecture = None



@dataclass
class PyTorchModel(GenericModel):
    state_dict: StateDict = field(default_factory=StateDict)
    num_feat: int = 0
    num_conv: int = 0
    ModuleClass: nn.Module = None
    module: nn.Module = None



@dataclass
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




def create_model(
    nn_model_path:str,
    framework: NnFramework,
    model_arch: NnArchitecture,
    model_obj: NnModelObject,
    metadata: dict[str, str] = {},
    device: str = 'cpu',
) -> NnModel:

    if framework.type == NnFrameworkType.PYTORCH:
        model = PyTorchModel(
            filepath=nn_model_path,
            framework=framework,
            arch=model_arch,
            state_dict=model_obj,
            metadata=metadata,
            device=device,
        )

    elif framework.type == NnFrameworkType.ONNX:
        model = OnnxModel(
            filepath=nn_model_path,
            framework=framework,
            arch=model_arch,
            model_proto=model_obj,
            metadata=metadata,
            device=device,
        )

    elif framework.type == NnFrameworkType.TENSORRT:
        if not device.startswith("cuda"):
            nnlogger.debug("[W] wrong device to load a tensorRT model, use default cuda device")
            device = "cuda:0"
        model = TrtModel(
            filepath=nn_model_path,
            framework=framework,
            arch=model_arch,
            engine=model_obj,
            metadata=metadata,
            device=device,
        )

    else:
        raise ValueError("[E] Unknown framework")

    # Parse a model object to detect the model info: scale, dtype, ...
    if logging.getLevelName(nnlogger.getEffectiveLevel()) == "DEBUG":
        # Don't catch the exception when in development
        model_arch.parse(model)
    else:
        try:
            model_arch.parse(model)
        except Exception as e:
            nnlogger.error(str(e))
            raise ValueError(f"Exception while parsing the model: {str(e)}")

    return model
