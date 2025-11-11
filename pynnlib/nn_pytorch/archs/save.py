from __future__ import annotations
import copy
import gc
from hytils import (
    absolute_path,
    is_access_granted,
    parent_directory,
)
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from safetensors.torch import save_file

from .load import load_state_dict
from pynnlib.metadata import generate_metadata
if TYPE_CHECKING:
    from pynnlib.model import PyTorchModel


def save_as(
    model: PyTorchModel,
    filepath: str | Path | None = None,
    directory: str | Path | None = None,
    basename: str | None = None,
    ext: Literal['.pth', '.safetensors'] = '.pth',
) -> str:

    if filepath is not None:
        directory = parent_directory(filepath)

    else:
        filepath: str = os.path.join(directory, f"{basename}{ext}")

    directory = absolute_path(directory)
    if not is_access_granted(directory, 'w'):
        raise PermissionError(f"{directory} is not writable")

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

        tmp_fp = filepath + ".tmp"
        save_file(state_dict, tmp_fp, metadata=metadata)
        os.replace(tmp_fp, filepath)

    else:
        raise ValueError(f"Not supported: {ext}")

    return filepath
