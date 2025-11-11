from argparse import (
    ArgumentParser,
    RawTextHelpFormatter,
)
from hytils import (
    absolute_path,
    lightcyan,
    path_split,
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
    save_as,
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
        help="model"
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default="",
        help="Path to save the output model file (ignored if --overwrite is used)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help="Overwrite the input model instead of saving to a new file",
    )
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="Print the model info")
    parser.add_argument("--debug", action="store_true", required=False, help="Used to debug")
    parser.add_argument("--author", type=str, default="", help="Author name")
    parser.add_argument("--purpose", type=str, default="", help="Model purpose (can include \\n)")
    parser.add_argument("--purpose-file", type=str, default="", help="Path to a text file containing purpose")
    parser.add_argument("--license", type=str, default="", help="License (e.g. MIT)")
    parser.add_argument("--name", type=str, default="", help="Model name")

    args = parser.parse_args()

    # Determine output model path
    model_fp: str = absolute_path(args.model)
    if args.overwrite:
        out_model_fp = args.model
    elif args.out:
        out_model_fp = args.out
    else:
        directory, basename, extension = path_split(args.model)
        out_model_fp = os.path.join(directory, f"{basename}_metadata{extension}")

    # Debug
    debug: bool = args.debug
    if debug:
        # FileOutputHandler = logging.FileHandler('logs.log', mode='w')
        # nnlogger.addHandler(FileOutputHandler)
        nnlogger.addHandler(logging.StreamHandler(sys.stdout))
        nnlogger.setLevel("DEBUG")


    device: str = 'cpu'

    try:
        start_time= time.time()
        model: NnModel = nnlib.open(model_fp, device)
        elapsed = time.time() - start_time
        print(
            f"\tarch:", lightcyan(model.arch_name),
            f"scale:", lightcyan(model.scale),
            f"\t\t({1000 * elapsed:.1f}ms)"
        )
        if args.verbose:
            print("Model:")
            print(model)
            print("\nArchitecture:")
            print(model.arch)
    except Exception as e:
        # For debug:
        print(red("Error"))
        model: NnModel = nnlib.open(model_fp, device)
        print(e)

    # List of standard metadata keys
    metadata_keys = ["name", "author", "license"]
    # Start with normal keys
    metadata = {
        key: getattr(args, key)
        for key in metadata_keys
        if getattr(args, key).strip()
    }
    # Handle purpose separately, supporting --purpose-file > --purpose
    purpose_text = ""
    if args.purpose_file:
        with open(args.purpose_file, "r", encoding="utf-8") as f:
            purpose_text = f.read().strip()
    elif args.purpose.strip():
        purpose_text = args.purpose.encode("utf-8").decode("unicode_escape")

    if purpose_text:
        metadata["purpose"] = purpose_text

    # Inject
    print("Model path:", args.model)
    print("Metadata to inject:")
    print_metadata(metadata)

    for k, v in metadata.items():
        print(f"  {k}: {v}")

    model.metadata = metadata
    save_as(out_model_fp, model)

    # Verify
    new_model: NnModel = nnlib.open(out_model_fp)
    print("Injected metadata:")
    print_metadata(new_model.metadata)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
