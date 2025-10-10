from __future__ import annotations
from pprint import pprint
import torch
from pynnlib.architecture import (
    detect_model_arch,
    NnArchitecture,
)
from pynnlib.model import StateDict



def get_torch_model_arch(
    state_dict: StateDict,
    architectures: dict[str, dict],
    device: str | torch.device = 'cpu'
) -> tuple[NnArchitecture, StateDict | None]:

    arch: NnArchitecture | None = detect_model_arch(state_dict, architectures)

    return arch
