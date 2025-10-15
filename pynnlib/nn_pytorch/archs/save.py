from __future__ import annotations
from copy import deepcopy
from hutils import (
    absolute_path,
    is_access_granted,
)
import json
import os
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Literal

import torch
from safetensors.torch import save_file

from .load import load_state_dict
from pynnlib.metadata import generate_metadata
if TYPE_CHECKING:
    from pynnlib.model import PyTorchModel


def save_as(
    model: PyTorchModel,
    directory: str | Path,
    basename: str,
    ext: Literal['.pth', '.safetensors'],
) -> PyTorchModel:
    directory = absolute_path(directory)
    if not is_access_granted(directory, 'w'):
        raise PermissionError(f"{directory} is not writable")

    filepath: str = os.path.join(directory, f"{basename}{ext}")
    metadata: dict[str, str] = generate_metadata(model, model.metadata)

    state_dict, _ = load_state_dict(model.filepath, device='cpu')
    if state_dict is None:
        raise ValueError(f"{model.filepath} is not a supported model")

    if 'metadata' in state_dict:
        del state_dict['metadata']

    if ext == '.pth':
        state_dict[f'metadata'] = json.dumps(metadata)
        torch.save(state_dict, filepath)

    elif ext == '.safetensors':
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                state_dict[k] = v.contiguous()

        save_file(state_dict, filepath, metadata=metadata)

    else:
        raise ValueError(f"Not supported: {ext}")

