import os
from pprint import pprint

from pynnlib.utils import absolute_path
from .logger import nnlogger

__is_tensorrt_available__: bool = False

def import_trt():
    trt = None
    try:
        import ctypes
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tensor_rtx_libs = os.path.join(base_dir, "libs")
        tensor_rtx_dll = os.path.join(tensor_rtx_libs, "tensorrt_rtx_1_0.dll")

        current_path = os.environ.get("PATH", "").split(os.pathsep)
        if tensor_rtx_libs not in current_path:
            current_path.insert(0, tensor_rtx_libs)
            os.environ["PATH"] = os.pathsep.join(current_path)

        ctypes.WinDLL(tensor_rtx_dll)

        import tensorrt_rtx as trt
        print("Imported tensorrt_rtx")

    except (FileNotFoundError, OSError, ImportError):
        try:
            import tensorrt as trt
            print("Imported tensorrt")
        except:
            nnlogger.debug("[W] Tensorrt package not found")
            print("[W] Tensorrt package not found")

    import sys
    modules = set(sys.modules) & set(globals())
    module_names = [sys.modules[m] for m in modules]
    if 'tensorrt' in module_names:
        print(f"in modules: [{module_names}]")
        __is_tensorrt_available__ = True

    if 'trt' in sys.modules or 'tensorrt' in sys.modules:
        print(trt.__version__)
        __is_tensorrt_available__ = True
    print(f"[I] Tensorrt package loaded (version {trt.__version__})")

    return trt

trt = import_trt()


try:
    nnlogger.debug(f"[V] TensorRT version: {trt.__version__}")
    __is_tensorrt_available__ = True
except:
    __is_tensorrt_available__ = False


HAS_PYTORCH_PACKAGE: bool = False
__is_cuda_available__: bool = False
try:
    import torch
    HAS_PYTORCH_PACKAGE = True
    nnlogger.debug(f"[I] PyTorch package loaded (version {torch.__version__})")
    __is_cuda_available__ = torch.cuda.is_available()

except ModuleNotFoundError:
    nnlogger.debug("[W] Failed to load PyTorch package")



def is_tensorrt_available() -> bool:
    return __is_tensorrt_available__


def is_cuda_available() -> bool:
    return __is_cuda_available__
