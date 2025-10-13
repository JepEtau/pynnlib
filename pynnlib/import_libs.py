import os
from pprint import pprint
from .logger import nnlogger

__is_tensorrt_available__: bool = False

def import_trt():
    trt = None
    import sys
    try:
        import ctypes
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tensor_rtx_libs = os.path.join(base_dir, "libs")

        # Determine platform-specific library name and loader
        if sys.platform == "win32":
            tensor_rtx_dll = os.path.join(tensor_rtx_libs, "tensorrt_rtx_1_0.dll")

            current_path = os.environ.get("PATH", "").split(os.pathsep)
            if tensor_rtx_libs not in current_path:
                current_path.insert(0, tensor_rtx_libs)
                os.environ["PATH"] = os.pathsep.join(current_path)

            ctypes.WinDLL(tensor_rtx_dll)

            import tensorrt_rtx as trt
            nnlogger.info("Imported tensorrt_rtx")

        elif sys.platform == "linux":
            # Untested
            lib_name = "libtensorrt_rtx_1_0.so"
            full_lib_path = os.path.join(tensor_rtx_libs, lib_name)

            # Add to LD_LIBRARY_PATH if not already there
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if tensor_rtx_libs not in current_ld_path:
                current_ld_path.insert(0, tensor_rtx_libs)
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(current_ld_path)

            ctypes.CDLL(full_lib_path)
        else:
            raise NotImplementedError("Not supported platform")

    except (FileNotFoundError, OSError, ImportError):
        try:
            import tensorrt as trt
            nnlogger.info("Imported tensorrt")
        except:
            nnlogger.warning("[W] Tensorrt package not found")

    modules = set(sys.modules) & set(globals())
    module_names = [sys.modules[m] for m in modules]
    if 'tensorrt' in module_names:
        __is_tensorrt_available__ = True

    if 'trt' in sys.modules or 'tensorrt' in sys.modules:
        __is_tensorrt_available__ = True
        nnlogger.info(f"[I] Tensorrt package loaded (version {trt.__version__})")

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
