from __future__ import annotations
from pprint import pprint
import torch
from pynnlib.architecture import (
    detect_model_arch,
    NnPytorchArchitecture,
)
from pynnlib.model import StateDict



def get_torch_model_arch(
    state_dict: StateDict,
    architectures: dict[str, NnPytorchArchitecture],
) -> NnPytorchArchitecture | None:
    return detect_model_arch(state_dict, architectures)
