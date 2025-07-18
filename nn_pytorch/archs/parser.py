from __future__ import annotations
from pathlib import Path
from pprint import pprint
from warnings import warn
from safetensors.torch import load_file
import torch
from .unpickler import RestrictedUnpickle
from pynnlib.architecture import detect_model_arch
from pynnlib.model import StateDict
from pynnlib.utils import get_extension
from pynnlib.logger import nnlogger

# https://github.com/chaiNNer-org/spandrel/blob/main/src/spandrel/__helpers/canonicalize.py
def remove_common_prefix(state_dict: StateDict) -> StateDict:
    prefixes = ("module.", "netG.")
    try:
        for prefix in prefixes:
            if all(i.startswith(prefix) for i in state_dict.keys()):
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    except:
        pass
    return state_dict


def standardize_state_dict(state_dict: StateDict) -> StateDict:
    unwrap_keys = (
        "state_dict",
        "params_ema",
        "params",
        "model",
        "net"
    )
    for unwrap_key in unwrap_keys:
        try:
            if isinstance(state_dict[unwrap_key], dict):
                state_dict = state_dict[unwrap_key]
                break
        except:
            pass
    return state_dict


def get_model_arch(
    model_path: str | Path,
    architectures: dict[str, dict],
    device: str | torch.device = 'cpu'
) -> tuple[str, StateDict | None]:
    state_dict = None
    arch_name: str = ''
    ext = get_extension(model_path)

    # Overwrite device because faster to parse model
    device: str = 'cpu'

    state_dict = None
    try:
        if ext == ".pt":
            try:
                # Try loading as a ScriptModule
                script_module: torch.jit.ScriptModule = torch.jit.load(
                    model_path,
                    map_location=device
                )
                state_dict = script_module.state_dict()

            except RuntimeError:
                # Fall back to regular load
                state_dict = torch.load(
                    model_path,
                    map_location=device,
                    pickle_module=RestrictedUnpickle,
                )

        elif ext in (".pth", ".ckpt"):
            state_dict = torch.load(
                model_path,
                map_location=device,
                pickle_module=RestrictedUnpickle,
            )

        elif ext == ".safetensors":
            state_dict = load_file(
                model_path,
                device=device
            )

        else:
            raise ValueError(
                f"Unsupported model file extension \'{ext}\'. Please try a supported model type."
            )

    except:
        state_dict = None

    if state_dict is None:
        warn(
            f"Failed to load model {model_path}."
        )
        return None, None

    # Standardize
    state_dict = standardize_state_dict(state_dict=state_dict)

    # Remove known common prefixes
    state_dict = remove_common_prefix(state_dict)

    # Find the architecture
    arch_name = detect_model_arch(state_dict, architectures)

    return arch_name, state_dict
