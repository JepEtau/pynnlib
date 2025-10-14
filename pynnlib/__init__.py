from .logger import nnlogger
from .import_libs import *
if not is_cuda_available():
    print("Error: torch and torchvision must be installed and a CUDA device is mandatory.")
    # raise SystemError("Error: torch and torchvision must be installed and a CUDA device is mandatory.")

if not is_tensorrt_available():
    print("Warning: no CUDA device detected, tensorRT is not available")

from .core import nn_lib as nnlib

from .model import (
    OnnxModel,
    TrtModel,
    PyTorchModel,
    ModelExecutor,
    NnModel,
    TrtEngine,
    SizeConstraint,
)

from .nn_onnx.inference.session import OnnxSession
from .nn_pytorch.inference.session import PyTorchSession
from .session import NnModelSession

try:
    from .nn_tensor_rt.inference.session import TensorRtSession
except:
    TensorRtSession = None

from .nn_types import (
    NnFrameworkType,
    NnModelObject,
    Idtype,
    ShapeStrategy,
    ShapeStrategyType,
)
from .architecture import (
    NnPytorchArchitecture,
    NnOnnxArchitecture,
    NnTensorrtArchitecture,
)
from .framework import (
    get_supported_model_extensions,
)

from .utils.torch_tensor import (
    torch_dtype_to_np,
    tensor_to_img,
    img_to_tensor,
)

from .utils.torch_tensor import (
    flip_r_b_channels,
    IdtypeToTorch,
    np_dtype_to_torch,
    to_nchw,
    to_hwc,
)

from .save import save_as

__all__ = [
    "nnlogger",
    "nnlib",
    "NnFrameworkType",

    "NnPytorchArchitecture",
    "NnOnnxArchitecture",
    "NnTensorrtArchitecture",

    "get_supported_model_extensions",

    "NnModel",
    "OnnxModel",
    "PyTorchModel",
    "TrtModel",

    "NnModelObject",
    "Idtype",
    "IdtypeToTorch",

    "NnModelSession",
    "OnnxSession",
    "PyTorchSession",
    "TensorRtSession",

    "ModelExecutor",
    "ShapeStrategy",
    "ShapeStrategyType",

    "is_cuda_available",
    "is_tensorrt_available",

    "torch_dtype_to_np",
    "np_dtype_to_torch",
    "flip_r_b_channels",
    "to_nchw",
    "to_hwc",
    "tensor_to_img",
    "img_to_tensor",

    "TrtEngine",

    "SizeConstraint",

    "save_as",
]
