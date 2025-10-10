from __future__ import annotations
import logging
import os
from pathlib import Path
from pprint import pprint
import re
from warnings import warn


from .import_libs import is_tensorrt_available
from .logger import is_debugging, nnlogger
from .metadata import generate_metadata
from .nn_types import ShapeStrategy

import onnx
nnlogger.debug(f"[I] ONNX package loaded (version {onnx.__version__})")

try:
    from .nn_tensor_rt.archs.save import generate_tensorrt_basename
except:
    # nnlogger.debug("[W] TensorRT is not supported: model cannot be converted")
    def generate_tensorrt_basename(*args) -> str:
        raise RuntimeError("TensorRT is not supported")

import torch
from .utils import (
    get_extension,
    os_path_basename,
)
from .utils.p_print import *

from .architecture import (
    NnArchitecture,
    NnTensorrtArchitecture
)
from .framework import (
    NnFramework,
    import_frameworks,
    extensions_to_framework,
)
from .model import (
    NnModel,
    OnnxModel,
    PyTorchModel,
    TrtModel,
)
from .nn_types import (
    Idtype,
    NnModelObject,
    NnFrameworkType
)
from .session import NnModelSession



class NnLib:

    def __init__(self) -> None:
        self.frameworks: dict[NnFrameworkType, NnFramework] = import_frameworks()
        nnlogger.debug(
            f"[I] Available frameworks: {', '.join(list([fwk.type.value for fwk in self.frameworks.values()]))}"
        )


    def get_framework_from_extension(self, nn_model_path: str | Path) -> NnFramework:
        extension = get_extension(nn_model_path)
        try:
            return self.frameworks[extensions_to_framework[extension]]
        except KeyError:
            nnlogger.debug(f"[E] No framwework found for model {nn_model_path}, unrecognized extension")
        except:
            raise ValueError(f"No framwework found for model {nn_model_path}")

        return None


    def open(
        self,
        model_path: str,
        device: str = 'cpu',
    ) -> NnModel | None:
        """Open and parse a model and returns its parameters"""
        if not os.path.exists(model_path):
            warn(red(f"[E] {model_path} does not exist"))
            return None

        # Detect the framework to use to load, parse and detect arch
        fwk: NnFramework = self.get_framework_from_extension(model_path)
        if fwk is None:
            warn(f"[E] No framework found for model {model_path}")
            return None

        # Load the model into a device and get metadata
        model_obj, metadata = fwk.load(model_path, device)
        if model_arch is None:
            warn(f"{red("[E] Failed to load model")} {model_path}")
            return None

        # Get the model arch
        model_arch, model_obj = fwk.detect_arch(model_obj, device)
        nnlogger.debug(yellow(f"fwk={fwk.type.value}, arch={model_arch.name}"))

        model = self._create_model(
            nn_model_path=model_path,
            framework=fwk,
            model_arch=model_arch,
            model_obj=model_obj,
            metadata=metadata,
            device=device,
        )
        if model is None:
            warn(f"{red("[E] Erroneous model or unsupported architecture:")}: {model_path}")
            return None

        # Parse metadata
        if model.framework.type == NnFrameworkType.PYTORCH:
            print(red("LD metadata"))
            if 'metadata' in model.state_dict and isinstance(model.state_dict['metadata'], dict):
                model.metadata = model.state_dict['metadata']
        elif model.framework.type == NnFrameworkType.ONNX:
            pass
        elif model.framework.type == NnFrameworkType.TENSORRT:
            pass

        if (
            model is not None
            and any(x <= 0 for x in (model.scale, model.in_nc, model.out_nc))
        ):
            nnlogger.debug("warning: at least a property has not been found, unsupported model")
            # return None

        return model


    def session(self, model: NnModel) -> NnModelSession:
        """Returns an inference session for a model"""
        create_session_fct = model.arch.create_session
        if create_session_fct is not None:
            session: NnModelSession = create_session_fct(model)
        else:
            raise NotImplementedError(f"Cannot create session for {model.fwk_type.value}")

        return session


    @staticmethod
    def _create_model(
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
            )
        elif framework.type == NnFrameworkType.ONNX:
            model = OnnxModel(
                filepath=nn_model_path,
                framework=framework,
                arch=model_arch,
                model_proto=model_obj,
                metadata=metadata,
            )
        elif framework.type == NnFrameworkType.TENSORRT:
            model = TrtModel(
                filepath=nn_model_path,
                framework=framework,
                arch=model_arch,
                engine=model_obj,
                metadata=metadata,
            )
            if not device.startswith("cuda"):
                nnlogger.debug("[W] wrong device to load a tensorRT model, use default cuda device")
                device = "cuda:0"
            model.device = device
        else:
            raise ValueError("[E] Unknown framework")

        # Parse a model object to detect the model info: scale, dtype, ...
        model.arch_name = model_arch.name

        if logging.getLevelName(nnlogger.getEffectiveLevel()) == "DEBUG":
            # Don't catch the exception when in development
            model_arch.parse(model)
        else:
            try:
                model_arch.parse(model)
            except Exception as e:
                return None

        return model


    def convert_to_onnx(
        self,
        model: NnModel,
        opset: int = 20,
        dtype: Idtype = 'fp32',
        device: str = 'cpu',
        shape_strategy: ShapeStrategy | None = None,
        out_dir: str | Path = "",
        suffix: str = "",
    ) -> str | OnnxModel:
        """Convert a model into an onnx model.

        Args:
            model: input model
            opset: onnx opset version
            dtype: Idtype
            device: device used for this conversion. the converted model will not use fp16
                    if this device does not support it.
            outdir: directory to save the onnx model. If not set, the model is not saved
        """
        if model.fwk_type == NnFrameworkType.ONNX:
            nnlogger.debug(f"This model is already an ONNX model")
            return model

        onnx_model: OnnxModel = None
        static: bool = bool(
            shape_strategy is not None and shape_strategy.type == 'static'
        )
        nnlogger.debug(yellow(f"[I] Convert to onnx model: ")
             + f"device={device}, dtype={dtype}, opset={opset}, static={static}"
             + f", shape={'x'.join([str(x) for x in shape_strategy.opt_size]) if static else ''}"
        )
        # TODO: put the following code in an Try-Except block
        if (
            model.arch is not None
            and (convert_fct := model.arch.to_onnx) is not None
        ):
            onnx_model_object: onnx.ModelProto = convert_fct(
                model=model,
                dtype=dtype,
                opset=opset,
                shape_strategy=shape_strategy,
                device=device,
            )
        else:
            raise RuntimeError(f"Cannot convert from {model.arch_name} to ONNX (not supported)")

        # Instantiate a new model
        onnx_fwk = self.frameworks[NnFrameworkType.ONNX]
        model_arch, _ = onnx_fwk.detect_arch(onnx_model_object)
        onnx_model = self._create_model(
            nn_model_path='',
            framework=onnx_fwk,
            model_arch=model_arch,
            model_obj=onnx_model_object,
        )
        onnx_model.opset = opset
        onnx_model.arch_name = model.arch.name
        onnx_model.torch_arch = model.arch
        onnx_model.force_weak_typing = model.force_weak_typing
        onnx_model.dtypes = set([dtype])

        # Add shape strategy, it will use
        if shape_strategy is not None:
            onnx_model.shape_strategy = shape_strategy
        else:
            onnx_model.shape_strategy = ShapeStrategy(type='dynamic')

        # Copy info
        onnx_model.alt_arch_name = model.arch_name
        if onnx_model.scale == 0:
            onnx_model.scale = model.scale

        # Remove metadata from proto and generate new ones in the model
        del onnx_model.model_proto.metadata_props[:]
        onnx_model.metadata = generate_metadata(model, {})

        # Save this model
        filepath: str = ""
        if out_dir:
            basename = os_path_basename(model.filepath)
            filepath = onnx_fwk.save(onnx_model, out_dir, basename, suffix)
            if filepath:
                nnlogger.debug(f"[I] Onnx model saved as {filepath}")
                onnx_model.filepath = filepath
            else:
                nnlogger.debug(f"[E] Failed to save the Onnx model")

        return onnx_model


    def convert_to_tensorrt(self,
        model: NnModel,
        shape_strategy: ShapeStrategy,
        dtype: Idtype = 'fp32',
        force_weak_typing: bool = False,
        optimization_level: int = 3,
        opset: int = 20,
        device: str = "cuda:0",
        out_dir: str | Path | None = None,
        suffix: str = "",
        overwrite: bool = False,
    ) -> TrtModel:
        """Convert a model into a tensorrt model.
        Returns a new instance of model.
        Refer to https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec

        Args:
            model: input model
            shape_strategy: specify the min/opt/max shapes.
                when static flag is set to True, the converter uses the opt shape
            dtype: datatype of the tensorRT engine, the onnx model will always be in fp32
            optimization_level: (not supported) Set the builder optimization level to build the engine with.
            opset: onnx opset version if the input model is NOT an onnx model.
            device: GPU device used for this conversion: This model shall run on the same
                device.
            out_dir: directory to save the onnx/tensorrt model. Not saved if set to None
            suffix: a suffix added to the model filename
        """

        if model.fwk_type == NnFrameworkType.TENSORRT:
            raise ValueError(f"This model is already a TensorRT model")

        model.shape_strategy = shape_strategy
        model.force_weak_typing = force_weak_typing

        trt_dtypes = set([dtype])

        # Remove suffixes from ONNX basename
        basename = os_path_basename(model.filepath)
        if model.fwk_type == NnFrameworkType.ONNX:
            opset = model.opset
            basename = re.sub(r"_op\d{1,2}", '', basename)
            for dt in ('_fp32', '_fp16', '_bf16'):
                basename = basename.replace(dt, '')

        # Check if conversion is possible: PyTorch
        # TODO: add ONNX check:
        #       - if onnx is generated by pynnlib: use metadata
        #       - else warning
        torch_arch: NnArchitecture | None = None
        if model.fwk_type == NnFrameworkType.PYTORCH:
            torch_arch = model.arch
            if torch_arch.to_tensorrt is None:
                raise ValueError(red(f"[E]: Conversion to tensorRT: arch \'{torch_arch.name}\' is not supported"))
            if dtype not in torch_arch.to_tensorrt.dtypes:
                raise ValueError(red(f"[E]: Conversion to tensorRT: dtype \'{dtype}\' is not supported"))

        # Verify if tensor engine already exists, create a fake model
        if out_dir is not None:
            _fake_model: TrtModel = TrtModel(
                framework=self.frameworks[NnFrameworkType.TENSORRT],
                arch=NnTensorrtArchitecture,
                torch_arch=torch_arch,
                arch_name='generic',
                filepath='',
                device=device,
                dtypes=trt_dtypes,
                force_weak_typing=force_weak_typing,
                engine=None,
                shape_strategy=shape_strategy,
                opset=opset,
            )
            trt_basename: str = generate_tensorrt_basename(_fake_model, basename)
            suffix = suffix if suffix is not None else ''
            filepath = os.path.join(out_dir, f"{trt_basename}{suffix}.trtzip")
            if os.path.exists(filepath):
                if not overwrite:
                    nnlogger.debug(f"[I] Engine {filepath} already exists, do not convert")
                    return self.open(filepath, device)
                else:
                    os.remove(filepath)
            else:
                nnlogger.debug(f"[I] Engine {filepath} does not exist")
            del _fake_model

        # Convert to Onnx
        # Always use fp32 when converting to onnx
        if model.fwk_type == NnFrameworkType.ONNX:
            nnlogger.debug(f"This model is already an ONNX model")
        else:
            nnlogger.debug(yellow(f"[I] Convert to onnx ({dtype}), use {device}"))
            onnx_model: OnnxModel = self.convert_to_onnx(
                model=model,
                opset=opset,
                dtype=dtype,
                device=device,
                shape_strategy=shape_strategy,
                out_dir=out_dir,
            )
        if is_debugging():
            print(onnx_model)
        torch.cuda.empty_cache()

        # TODO: put the following code in an Try-Except block
        trt_engine = None
        if onnx_model.torch_arch.to_tensorrt is not None:
            from pynnlib.nn_onnx.archs.onnx_to_tensorrt import to_tensorrt
            trt_engine = to_tensorrt(
                model=onnx_model,
                device=device,
                dtypes=trt_dtypes,
                shape_strategy=shape_strategy
            )
        else:
            raise NotImplementedError("Conversion to TensorRT is not supported")

        if trt_engine is None:
            nnlogger.debug(f"Error while converting {model.fwk_type} to TensorRT")
            return None

        # Instantiate a new model
        trt_fwk = self.frameworks[NnFrameworkType.TENSORRT]
        model_arch, _ = trt_fwk.detect_arch(trt_engine)
        trt_model = self._create_model(
            nn_model_path='',
            framework=trt_fwk,
            model_arch=model_arch,
            model_obj=trt_engine,
        )

        # Update with onnx params
        trt_model.torch_arch = onnx_model.torch_arch
        trt_model.arch_name = model.arch_name
        trt_model.opset = opset
        trt_model.shape_strategy = shape_strategy
        trt_model.scale = model.scale
        trt_model.dtypes = trt_dtypes.copy()
        trt_model.force_weak_typing = model.force_weak_typing
        trt_model.metadata = generate_metadata(trt_model)

        # Save this engine as a model
        if out_dir is not None:
            nnlogger.debug(f"[V] save tensort RT engine to {out_dir}")
            success = trt_fwk.save(trt_model, out_dir, basename, suffix)
            if success:
                nnlogger.debug(f"[I] TRT engine saved as {trt_model.filepath}")
            else:
                nnlogger.debug(f"[E] Failed to save the TRT engine as {trt_model.filepath}")

        return trt_model


    def set_session_constructor(
        self,
        framework: NnFrameworkType,
        ModelSession: NnModelSession
    ):
        """Set a custom session contructor ffor a framework"""
        if framework == NnFrameworkType.TENSORRT and not is_tensorrt_available():
            raise ValueError("[E] Framework not supported: cannot set a custom session function")
        self.frameworks[framework].Session = ModelSession


    def load_model(
        self,
        model_name: str = 'efficienttam_s_512x512',
        device: str = 'cuda:0',
    ) -> NnModel:
        # PoC
        from pynnlib.nn_pytorch.archs.EfficientTAM import PREDEFINED_MODEL_ARCHITECTURES
        predefined_models: dict[str, dict] = {
            'efficienttam_s_512x512': PREDEFINED_MODEL_ARCHITECTURES['efficienttam_s_512x512'],
        }

        model = self._create_model(
            nn_model_path="",
            framework=self.frameworks[NnFrameworkType.PYTORCH],
            model_arch=predefined_models[model_name],
            model_obj=None,
            device=device,
        )
        return model


nn_lib: NnLib = NnLib()

