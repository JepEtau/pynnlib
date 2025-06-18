import logging

nnlogger: logging.Logger = logging.getLogger("pynnlib")

# Enable this to debug imports
import sys
# nnlogger.addHandler(logging.StreamHandler(sys.stdout))
nnlogger.setLevel("DEBUG")


def is_debugging() -> bool:
    return bool(logging.getLevelName(nnlogger.getEffectiveLevel()) == "DEBUG")

