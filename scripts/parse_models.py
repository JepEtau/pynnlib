from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
)
from hutils import (
    absolute_path,
    get_extension,
    lightcyan,
    lightgreen,
    red,
    yellow,
)
import logging
import logging.config
import os
from pprint import pprint
import signal
import sys
import textwrap
import time

# logging.config.fileConfig('config.ini')
# logger = logging.getLogger("pynnlib")
# logging.basicConfig(filename="logs.log", filemode="w", format="%(name)s → %(levelname)s: %(message)s")
# logging.config.fileConfig('config.ini')
start_time = time.time()

if not os.path.exists("pynnlib"):
    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if os.path.exists(os.path.join(root_path, "pynnlib")):
        sys.path.append(root_path)

from pynnlib import (
    nnlogger,
    nnlib,
    NnModel,
    is_tensorrt_available,
    get_supported_model_extensions,
    NnFrameworkType,
    print_metadata,
)
print(f"pynnlib loaded in {(time.time() - start_time):.02f}s")


def main():
    parser = ArgumentParser(
        description="Walk through a directory and parse models",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        required=False,
        help="Path to a specific model file."
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="",
        required=False,
        help="Directory containing model files."
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        required=False,
        help="Recursively search through subdirectories."
    )

    parser.add_argument(
        "-f",
        "--filter",
        choices=[
            *get_supported_model_extensions(),
            'pytorch',
            'onnx',
            'trt',
            'trtzip',
        ],
        default="",
        required=False,
        help="Filter models by file extension or framework (e.g., pytorch, onnx, trtzip)."
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        required=False,
        help="Display model metadata."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Display detailed model information."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enable debug mode for additional logging."
    )

    arguments = parser.parse_args()

    debug: bool = arguments.debug
    if debug:
        # FileOutputHandler = logging.FileHandler('logs.log', mode='w')
        # nnlogger.addHandler(FileOutputHandler)
        nnlogger.addHandler(logging.StreamHandler(sys.stdout))
        nnlogger.setLevel("DEBUG")


    filtered_exts: tuple[str] = get_supported_model_extensions()
    if arguments.filter != '':
        if arguments.filter == 'onnx':
            filtered_exts = get_supported_model_extensions(NnFrameworkType.ONNX)
        elif arguments.filter == 'pytorch':
            filtered_exts = get_supported_model_extensions(NnFrameworkType.PYTORCH)
        elif arguments.filter in  ('trt', 'trtzip'):
            filtered_exts = get_supported_model_extensions(NnFrameworkType.TENSORRT)
        else:
            filtered_exts = (arguments.filter)
    trt_extensions: tuple[int] = get_supported_model_extensions(NnFrameworkType.TENSORRT)

    model_fp: str = arguments.model
    if model_fp:
        model_fp = absolute_path(model_fp)
        ext = get_extension(model_fp)
        if ext not in filtered_exts:
            print(f"[E] Unsupported model: {arguments.model} cannot be loaded")
        if ext in trt_extensions and not is_tensorrt_available():
            print(f"[E] Unsupported hardware: {arguments.model} cannot be loaded")
        device = 'cuda' if ext in trt_extensions else 'cpu'

        try:
            start_time= time.time()
            model: NnModel = nnlib.open(model_fp, device)
            elapsed = time.time() - start_time
            print(
                f"\tarch:", lightcyan(model.arch_name),
                f"scale:", lightcyan(model.scale),
                f"\t\t({1000 * elapsed:.1f}ms)"
            )
            # Display metadata
            if arguments.metadata:
                print_metadata(model.metadata)


            if arguments.verbose:
                print("Model:")
                print(model)
                print("\nArchitecture:")
                print(model.arch)
        except Exception as e:
            # For debug:
            # model: NnModel = nnlib.open(model_fp, device)
            print(red(e))
        return

    ml_models_path: str = absolute_path(arguments.dir)
    for root, _, files in os.walk(ml_models_path):
        for f in sorted(files):
            ext = get_extension(f)
            if ext not in filtered_exts:
                continue
            if ext in trt_extensions and not is_tensorrt_available():
                print(f"[E] Unsupported hardware: {f} cannot be loaded")

            filepath = os.path.join(root, f)
            device = 'cuda' if ext in trt_extensions else 'cpu'

            print(lightgreen(f"{f}"))
            try:
                start_time= time.time()
                model: NnModel = nnlib.open(filepath, device)
                if model is None:
                    continue
                elapsed = time.time() - start_time
                dtype: str = (
                    ", ".join(model.arch.dtypes)
                    if model.framework.type == NnFrameworkType.PYTORCH
                    else model.io_dtypes['input']
                )
                print(
                    f"    arch:", lightcyan(model.arch_name),
                    f"scale:", lightcyan(model.scale),
                    f"\n    in_dtype:", lightcyan(dtype),
                    f"\n    ({1000 * elapsed:.1f}ms)\n"
                )

                # Display metadata
                if arguments.metadata:
                    print_metadata(model.metadata)

                if arguments.verbose or arguments.debug:
                    print("Model:")
                    print(model)
                if arguments.debug:
                    print("Arch:")
                    print(model.arch)
            except Exception as e:
                model: NnModel = nnlib.open(filepath, device)
                print(e)

        if not arguments.recursive:
            break

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
