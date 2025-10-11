from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
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
)
from pynnlib.utils import absolute_path, get_extension
from pynnlib.utils.p_print import *
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

    model_fp: str = absolute_path(arguments.model)
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
        if arguments.verbose:
            print("Model:")
            print(model)
            print("\nArchitecture:")
            print(model.arch)
    except Exception as e:
        # For debug:
        print(red("Error"))
        model: NnModel = nnlib.open(model_fp, device)
        print(e)

    metadata: dict[str, str] = {
        'name': "my_small_model",
        'date': "2025-10-10",
        'version': "1.0",
        'license': "Common",
        'author': "john doe",
        'comment': "a comment",
    }

    out_model_fp: str = "test.pth"
    if absolute_path(out_model_fp) == absolute_path(model_fp):
        print("metadata")
        pprint(metadata)
    else:
        model.metadata = metadata
        save_as("test.pth", model)

        new_model: NnModel = nnlib.open("test.pth")
        print("metadata to inject:")
        pprint(metadata)
        print("injected metadata")
        pprint(new_model.metadata)



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
