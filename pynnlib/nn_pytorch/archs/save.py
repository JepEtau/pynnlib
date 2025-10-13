from __future__ import annotations
from copy import deepcopy
import json
import os
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Literal

import torch

from .load import load_state_dict
from pynnlib.metadata import generate_metadata
from pynnlib.utils import absolute_path, is_access_granted
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
        raise PermissionError(f"{directory} is read only")

    filepath: str = os.path.join(directory, f"{basename}{ext}")
    metadata = generate_metadata(model, model.metadata)

    state_dict, _ = load_state_dict(model.filepath)
    if state_dict is None:
        raise ValueError(f"{model.filepath} is not a supported model")

    state_dict[f'metadata'] = json.dumps(metadata)
    if ext == '.pth':
        torch.save(state_dict, filepath)

