
from __future__ import annotations
from functools import partial
from importlib import util as importlib_util
import os
from pathlib import Path, PurePath
import sys
from typing import Type

from torch import nn

from ..torch_types import StateDict
from pynnlib.logger import is_debugging, nnlogger
from pynnlib.architecture import NnPytorchArchitecture
from pynnlib.utils.p_print import *
from pynnlib.model import PyTorchModel
from pynnlib.import_libs import is_tensorrt_available
from .helpers import parameters_to_args



def import_model_architectures() -> list[NnPytorchArchitecture]:
    imported_archs: list[NnPytorchArchitecture] = []

    # Current module name
    dir_parts: tuple[str] = PurePath(
        os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    ).parts
    model_module_base: str = '.'.join(dir_parts[-3:])

    # Get all subfolders
    model_directory: Path = Path(os.path.dirname(__file__)).absolute()
    arch_dirs: set = set([d for d in os.listdir(model_directory) \
        if os.path.isdir(os.path.join(model_directory, d))])
    arch_dirs.discard('__pycache__')

    # Walk trough subdirectories
    detected_archs: list[tuple[str, str, str]] = []
    for arch in arch_dirs:
        # Check structure
        init_file: str = os.path.join(model_directory, arch, '__init__.py')
        if (not os.path.isfile(init_file)
            or not os.path.isdir(os.path.join(model_directory, arch, 'module'))):
            continue

        # Append to detected
        arch_module_name: str = f"{model_module_base}.{arch}"
        detected_archs.append((arch, arch_module_name, init_file))

    # Import modules
    if is_debugging():
        from pprint import pprint
        pprint(detected_archs)

    for arch, arch_module_name, init_file in detected_archs:
        if arch_module_name in sys.modules:
            nnlogger.debug(f"architecture {arch_module_name!r} already in sys.modules")
            module = sys.modules[arch_module_name]

        elif (module_spec := importlib_util.spec_from_file_location(arch_module_name, init_file)) is not None:
            module = importlib_util.module_from_spec(module_spec)
            sys.modules[arch_module_name] = module
            nnlogger.debug(f"load {arch_module_name!r} from {module}")
            module_spec.loader.exec_module(module)
            nnlogger.debug(f"{arch_module_name!r} has been imported")
        # else:
        #     nnlogger.debug(f"can't find the {arch_module_name!r} module")

        # if (hasattr(module, "MODEL_ARCHITECTURES")
        #     and getattr(module, "MODEL_ARCHITECTURES") is not None):
        try:
            if getattr(module, "MODEL_ARCHITECTURES") is not None:
                imported_archs.extend(module.MODEL_ARCHITECTURES)
        except:
            pass
    nnlogger.debug("[V] Imported PyTorch archs")
    for arch in imported_archs:
        nnlogger.debug(f"[V]\t{arch.name}")

    return imported_archs



def contains_all_keys(state_dict: StateDict, keys: tuple[str | tuple[str]]) -> bool:
    """Define a simple detection function to detect if all keys
        are in a state_dict
    """
    return all(key in state_dict for key in keys)



def contains_any_keys(state_dict: StateDict, keys: tuple[str | tuple[str]]) -> bool:
    return any(key in state_dict for key in keys)


def create_session(model: Type[PyTorchModel]) -> Type[PyTorchModel]:
    TorchModule: type[nn.Module] = None
    if model.ModuleClass is not None:
        TorchModule = model.ModuleClass

    elif model.arch.module.module_class is not None:
        TorchModule = model.arch.module.module_class

    else:
        model.arch.import_module()
        TorchModule = model.arch.module.module_class

    if TorchModule is None:
        raise ValueError(f"No nn.module imported for arch {model.arch}")

    args = parameters_to_args(model, TorchModule)
    model.module = TorchModule(**args)
    return model.framework.Session(model)



# Detect and list model architectures
MODEL_ARCHITECTURES: list[NnPytorchArchitecture] = import_model_architectures()


# Append common variables for all architectures
_is_debug: bool = is_debugging()
for arch in MODEL_ARCHITECTURES:
    arch.type = arch.name

    # By default use a simple detection function which checks
    # that the state_dict contains all detection_keys
    arch.detect = (
        partial(contains_all_keys, keys=arch.detection_keys)
        if arch.detect is None
        else partial(arch.detect, keys=arch.detection_keys)
    )

    if arch.create_session is None:
        arch.create_session = create_session

    if _is_debug:
        print(f"[V] {arch.name}, detection keys:\n  {'\n  '.join(arch.detection_keys)}")

    # if is_tensorrt_available():
    #     if arch.to_tensorrt is not None:
            # Use default onnx to trt conversion function


        # else:
        #   conversion to tensorrt is not supported

    arch.lock()

