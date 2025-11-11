from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
)
from hytils import (
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
import time
from warnings import warn


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
    get_supported_model_extensions,
    get_out_model_fp,
    nnlogger,
    nnlib,
    NnModel,
    NnFrameworkType,
    ShapeStrategy,
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
        help="model"
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="",
        required=False,
        help="Directory"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Print the model info"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Used to debug"
    )

    arguments = parser.parse_args()

    debug: bool = arguments.debug
    if debug:
        # FileOutputHandler = logging.FileHandler('logs.log', mode='w')
        # nnlogger.addHandler(FileOutputHandler)
        nnlogger.addHandler(logging.StreamHandler(sys.stdout))
        nnlogger.setLevel("DEBUG")


    trt_extensions: tuple[int] = get_supported_model_extensions(NnFrameworkType.TENSORRT)

    model_fp: str = arguments.model
    if model_fp:
        device = 'cuda' if get_extension(model_fp) in trt_extensions else 'cpu'

        try:
            start_time= time.time()
            model: NnModel = nnlib.open(model_fp, device)
            elapsed = time.time() - start_time
            print(
                f"\tarch:", lightcyan(model.arch_name),
                f"scale:", lightcyan(model.scale),
                f"\t\t({1000 * elapsed:.1f}ms)"
            )
            if arguments.verbose:
                print("Model:")
                print(model)
                print("\nArchitecture:")
                print(model.arch)

            out_fp: str = ""


            # PyTorch -> ONNX
            print(lightcyan("PyTorch -> ONNX"))
            to: NnFrameworkType = NnFrameworkType.ONNX
            print("  Opset:")
            for opset in range(20, 21):
                out_fp = get_out_model_fp(model=model, to=to, opset=opset)
                print(f"    opset={opset}: {out_fp}")

            print("  Dtype:")
            for dtype in ('fp32', 'fp16', 'bf16'):
                out_fp = get_out_model_fp(model=model, to=to, dtype=dtype)
                print(f"    dtype={dtype}: {out_fp}")

            print("  Strategy:")
            for strategy in ('static', 'fixed', 'dynamic'):
                out_fp = get_out_model_fp(
                    model=model,
                    to=to,
                    shape_strategy=ShapeStrategy(type=strategy, opt_size=(720, 540)),
                )
                print(f"  strategy={strategy}: {out_fp}")

            print("  Suffix:")
            for suffix in ("", "custom-suffix"):
                out_fp = get_out_model_fp(model=model, to=to, suffix=suffix)
                print(f"    suffix={suffix}: {out_fp}")


            # PyTorch -> TensorRT
            print(lightcyan("PyTorch -> tensorRT"))
            to: NnFrameworkType = NnFrameworkType.TENSORRT
            if to not in nnlib.frameworks:
                warn("TensorRT framework is not supported. Skipping tests")

            else:
                print("  Opset:")
                for opset in range(20, 21):
                    out_fp = get_out_model_fp(model=model, to=to, opset=opset)
                    print(f"    opset={opset}: {out_fp}")

                print("  Dtype:")
                for dtype in ('fp32', 'fp16', 'bf16'):
                    out_fp = get_out_model_fp(model=model, to=to, dtype=dtype)
                    print(f"    dtype={dtype}: {out_fp}")

                print("  Strategy:")
                for strategy in ('static', 'fixed', 'dynamic'):
                    out_fp = get_out_model_fp(
                        model=model,
                        to=to,
                        shape_strategy=ShapeStrategy(type=strategy, opt_size=(720, 540)),
                    )
                    print(f"  strategy={strategy}: {out_fp}")

                print("  Suffix:")
                for suffix in ("", "custom-suffix"):
                    out_fp = get_out_model_fp(model=model, to=to, suffix=suffix)
                    print(f"    suffix={suffix}: {out_fp}")


            # ONNX to TensoRT
            print(lightcyan("ONNX -> tensorRT"))
            to: NnFrameworkType = NnFrameworkType.TENSORRT
            if to not in nnlib.frameworks:
                warn("TensorRT framework is not supported. Skipping tests")

            else:
                opset = 20
                dtype: Hdtype = 'fp32'
                shape_strategy: ShapeStrategy = ShapeStrategy(type=strategy, opt_size=(720, 540))
                suffix: str = ""

                onnx_model: OnnxModel = nnlib.convert_to_onnx(
                    model: NnModel,
                    opset=opset,
                    dtype=dtype,
                    device='cpu' if dtype == 'fp32' else 'cuda',
                    shape_strategy=shape_strategy,
                    suffix=suffix,
                )

                out_fp = get_out_model_fp(
                    model=model,
                    to=to,
                    opset=opset,
                    dtype=dtype,
                    shape_strategy=shape_strategy,
                    suffix=suffix,
                )
                print(f"    opset={opset}: {out_fp}")


        except Exception as e:
            print(red(e))

        print("Ended.")


    # Directory:
    # ml_models_path: str = absolute_path(arguments.dir)
    # for root, _, files in os.walk(ml_models_path):
    #     for f in sorted(files):
    #         ext = get_extension(f)
    #         if ext in trt_extensions and not is_tensorrt_available():
    #             print(f"[E] Unsupported hardware: {f} cannot be loaded")

    #         filepath = os.path.join(root, f)
    #         device = 'cuda' if ext in trt_extensions else 'cpu'

    #         print(lightgreen(f"{f}"))
    #         try:
    #             start_time= time.time()
    #             model: NnModel = nnlib.open(filepath, device)
    #             if model is None:
    #                 continue
    #             elapsed = time.time() - start_time
    #             dtype: str = (
    #                 ", ".join(model.arch.dtypes)
    #                 if model.framework.type == NnFrameworkType.PYTORCH
    #                 else model.io_dtypes['input']
    #             )
    #             print(
    #                 f"    arch:", lightcyan(model.arch_name),
    #                 f"scale:", lightcyan(model.scale),
    #                 f"\n    in_dtype:", lightcyan(dtype),
    #                 f"\n    ({1000 * elapsed:.1f}ms)\n"
    #             )
    #             if arguments.verbose or arguments.debug:
    #                 print("Model:")
    #                 print(model)
    #             if arguments.debug:
    #                 print("Arch:")
    #                 print(model.arch)
    #         except Exception as e:
    #             model: NnModel = nnlib.open(filepath, device)
    #             print(e)

    #     if not arguments.recursive:
    #         break

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
