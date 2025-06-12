from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from pynnlib.metadata import generate_metadata
from pynnlib.utils import is_access_granted, path_split
if TYPE_CHECKING:
    from pynnlib.model import PyTorchModel


def save_as(
    model: PyTorchModel,
    ext: Literal['.pth', '.safetensors'],
    filepath: str,
) -> PyTorchModel:
    # Don't use directory/suffix because it can be saved from a torch model only

    directory, basename, ext = path_split(filepath)

    if not is_access_granted(directory, 'w'):
        raise PermissionError(f"{directory} is read only")

    filepath = os.path.join(directory, f"{basename}{ext}")
    if filepath == model.filepath:
        model_to_save = model
    else:
        model_to_save: PyTorchModel = deepcopy(model)
    model_to_save.filepath = filepath

    metadata = generate_metadata(model)
    model_to_save.metadata = metadata

    # model_to_save.state_dict[f'metadata'] = json.dumps(metadata)
    if ext == '.pth':
        torch.save(
            model_to_save.state_dict,
            model_to_save.filepath
        )

    return model_to_save



