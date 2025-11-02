from argparse import (
    ArgumentParser,
    RawTextHelpFormatter,
)
from hutils import (
    absolute_path,
    path_split,
    red,
    yellow,
    lightgreen,
)
import logging
import os
import re
import signal
import sys
import time
from typing import Any, Optional

if not os.path.exists("pynnlib"):
    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if os.path.exists(os.path.join(root_path, "pynnlib")):
        sys.path.append(root_path)

from pynnlib import (
    Idtype,
    nnlib,
    NnModel,
    TrtModel,
    ShapeStrategy,
    is_cuda_available,
    is_tensorrt_available,
    nnlogger,
)


def _str_to_size(size_str: str) -> tuple[int, int] | None:
    if (match := re.match(re.compile(r"^(\d+)x(\d+)$"), size_str)):
        return (int(match.group(1)), int(match.group(2)))
    return None


def convert_to_tensorrt(
    arguments: Any,
    model: NnModel,
    device: str,
    dtype: Idtype,
    force_weak_typing: bool = False,
    force: bool = False,
    debug: Optional[bool] = False
) -> TrtModel | None:
    trt_model: TrtModel | None = None

    # Shape strategy
    shape_strategy: ShapeStrategy = ShapeStrategy()

    opt_size = _str_to_size(arguments.opt_size)
    if opt_size is None:
        sys.exit(red(f"[E] Erroneous option: {arguments.opt_size}"))
    shape_strategy.opt_size = opt_size

    if arguments.static:
        shape_strategy.type = 'static'
        shape_strategy.opt_size = _str_to_size(arguments.size)
        print(red(shape_strategy))

    else:
        if not arguments.fixed_size:
            min_size = _str_to_size(arguments.min_size)
            if min_size is None:
                sys.exit(red(f"[E] Erroneous option: {arguments.min_size}"))
            shape_strategy.min_size = min_size

            max_size = _str_to_size(arguments.max_size)
            if max_size is None:
                sys.exit(red(f"[E] Erroneous option: {arguments.max_size}"))
            shape_strategy.max_size = max_size

        else:
            shape_strategy.type = 'fixed'
            shape_strategy.min_size = shape_strategy.opt_size
            shape_strategy.max_size = shape_strategy.opt_size

    if not shape_strategy.is_valid():
        sys.exit(red(f"[E] Erroneous sizes"))

    # Optimization
    opt_level = arguments.opt_level
    opt_level = None if not 1 <= opt_level <= 5 else opt_level

    trt_model: TrtModel = nnlib.convert_to_tensorrt(
        model=model,
        shape_strategy=shape_strategy,
        dtype=dtype,
        force_weak_typing=force_weak_typing,
        optimization_level=opt_level,
        opset=arguments.opset,
        device=device,
        out_dir=path_split(model.filepath)[0],
        overwrite=force,
    )

    return trt_model



def main():
    parser = ArgumentParser(
        description="Convert a model to ONNX format or a TensorRT engine.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='',
        required=True,
        help="""Path to the input model file (PyTorch or ONNX).
\n"""
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        required=False,
        default=False,
        help="""Force regeneration of the model (overwrite existing output).
\n"""
    )

    parser.add_argument(
        "-overwrite",
        "--overwrite",
        action="store_true",
        required=False,
        default=False,
        help="""Force regeneration of the model (same as --force).
\n"""
    )


    parser.add_argument(
        "-onnx",
        "--onnx",
        action="store_true",
        required=False,
        default=False,
        help="""Export the model to ONNX format.
\n"""
    )

    parser.add_argument(
        "-trt",
        "--tensorrt",
        action="store_true",
        required=False,
        default=False,
        help="""Convert the model to a TensorRT engine.
\n"""
    )

    parser.add_argument(
        "-fp32",
        "--fp32",
        action="store_true",
        required=False,
        default=True,
        help="""Use full precision (FP32).
\n"""
    )

    parser.add_argument(
        "-fp16",
        "--fp16",
        action="store_true",
        required=False,
        default=False,
        help="""Use half precision (FP16).
\n"""
    )

    parser.add_argument(
        "-bf16",
        "--bf16",
        action="store_true",
        required=False,
        default=False,
        help="""Use mixed precision (BF16).
\n"""
    )

    parser.add_argument(
        "-opset",
        "--opset",
        type=int,
        required=False,
        default=20,
        help="""ONNX opset version to use when exporting from PyTorch.
\n"""
    )
    parser.add_argument(
        "--static",
        "-static",
        action="store_true",
        required=False,
        default=False,
        help="""(TensorRT) Use a static shape (based on opt_size) for engine generation.
\n"""
    )
    parser.add_argument(
        "-size",
        "--size",
        type=str,
        default='0x0',
        required=False,
        help="""(ONNX) Static input size (required if --static is set). Format: WxH.
\n"""
    )
    parser.add_argument(
        "-min",
        "--min_size",
        type=str,
        default='64x64',
        required=False,
        help="""(TensorRT) Minimum input size. Format: WxH.
\n"""
    )
    parser.add_argument(
        "-opt",
        "--opt_size",
        type=str,
        default='768x576',
        required=False,
        help="""(TensorRT) Optimal input size. Format: WxH. Use 'input' to match the input video dimensions.
\n"""
    )
    parser.add_argument(
        "-max",
        "--max_size",
        type=str,
        default='1920x1080',
        required=False,
        help="""(TensorRT) Maximum input size. Format: WxH.
\n"""
    )
    parser.add_argument(
        "-fixed",
        "--fixed_size",
        action="store_true",
        required=False,
        default=False,
        help="""(TensorRT) Use the same dimensions for min_size, opt_size, and max_size.
\n"""
    )
    parser.add_argument(
        "--opt_level",
        type=int,
        default=3,
        required=False,
        help="""(TensorRT) Optimization level (1–5). Currently not supported.
\n"""
    )
    parser.add_argument(
        "--weak",
        "-weak",
        action="store_true",
        required=False,
        default=False,
        help="""(TensorRT) Use weak typing during engine conversion.
\n"""
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="""Enable verbose output.
\n"""
    )

    arguments = parser.parse_args()
    force: bool = arguments.force or arguments.overwrite

    if arguments.verbose:
        nnlogger.addHandler(logging.StreamHandler(sys.stdout))
        nnlogger.setLevel("DEBUG")

    if not arguments.tensorrt and not arguments.onnx:
        sys.exit(red(f"[E] at least --onnx or --tensorrt must be specified"))

    if arguments.fp16 and not is_cuda_available():
        sys.exit(red(f"[E] No CUDA device found, cannot convert with fp16 support"))

    if arguments.bf16 and not is_cuda_available():
        sys.exit(red(f"[E] No CUDA device found, cannot convert with bf16 support"))

    # device and datatype
    device = "cuda" if is_cuda_available() else 'cpu'

    # Open model file
    model_filepath: str = absolute_path(arguments.model)
    if not os.path.isfile(model_filepath):
        raise ValueError(red(f"[E] {model_filepath} is not a valid file"))
    model: NnModel = nnlib.open(model_filepath, device)
    print(lightgreen(f"[I] arch: {model.arch_name}"))
    print(model)

    fp16: bool = arguments.fp16
    # if arguments.fp16 and not 'fp16' in model.arch.dtypes:
    #     sys.exit(red(f"[E] This arch does not support conversion with fp16 support"))
    # fp16: bool = arguments.fp16 and 'fp16' in model.arch.dtypes

    bf16: bool = arguments.bf16
    # if arguments.bf16 and not 'bf16' in model.arch.dtypes:
    #     sys.exit(red(f"[E] This arch does not support conversion with bf16 support, supported dtypes: {model.arch.dtypes}"))
    # bf16: bool = arguments.bf16 and 'bf16' in model.arch.dtypes

    # bf16 has the priority if multiple types are provided
    c_dtype: Idtype = 'fp32'
    if fp16:
        c_dtype = 'fp16'
    if bf16:
        c_dtype = 'bf16'

    # Model conversion
    static: bool = arguments.static
    shape_strategy: ShapeStrategy | None = None
    if static and arguments.size == "0x0":
        sys.exit(red(f"Error: requires --size when converting to a static ONNX model."))
    elif static:
        print(f"Static strategy: {arguments.size}")
        shape_strategy: ShapeStrategy = ShapeStrategy(
            type='static',
            opt_size=_str_to_size(arguments.size)
        )
        model.shape_strategy = shape_strategy

    model = model
    if arguments.onnx:
        print(f"[V] Convert {model.filepath} to ONNX (dtype={c_dtype}): ")
        start_time = time.time()
        onnx_model = nnlib.convert_to_onnx(
            model=model,
            opset=arguments.opset,
            dtype=c_dtype,
            shape_strategy=shape_strategy,
            device=device,
            out_dir=path_split(model.filepath)[0],
        )
        elapsed_time = time.time() - start_time
        print(lightgreen(f"[I] Onnx model saved as {onnx_model.filepath}"))
        print(f"[V] Converted in {elapsed_time:.2f}s")
        print(onnx_model)

    if arguments.tensorrt:
        if is_tensorrt_available():
            print(f"[V] Convert {model.filepath} to TensorRT (c_dtype={c_dtype}): ")
            start_time = time.time()
            trt_model = convert_to_tensorrt(
                arguments,
                model=model,
                device=device,
                dtype=c_dtype,
                force_weak_typing=arguments.weak,
                force=force,
                debug=True
            )
            if trt_model is None:
                print(red("Error: Failed to convert to a TensorRT engine."))
            else:
                elapsed_time = time.time() - start_time
                print(lightgreen(f"[I] TensorRT engine saved as {trt_model.filepath}"))
                print(f"[V] Converted in {elapsed_time:.2f}s")
        else:
            print(red("Error: No compatible device detected — TensorRT engine conversion requires a supported NVIDIA GPU."))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
