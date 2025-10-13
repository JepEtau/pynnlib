from __future__ import annotations

from .model import NnModel
from .framework import NnFrameworkType
from .utils import path_split


def save_as(model_fp: str, model: NnModel) -> None:
    directory, basename, ext = path_split(model_fp)
    if not directory:
        directory="./"

    if model.framework.type == NnFrameworkType.PYTORCH:
        # PyTorch, SafeTensors
        model.framework.save(
            model=model,
            directory=directory,
            basename=basename,
            ext=ext
        )

    elif model.framework.type == NnFrameworkType.ONNX:
        model.framework.save(
            model=model,
            directory=directory,
            basename=basename,
            suffix="",
        )

    elif model.framework.type == NnFrameworkType.TENSORRT:
        model.framework.save(
            model=model,
            directory=directory,
            basename=basename,
            suffix="",
        )

    else:
        raise ValueError(f"Unknown framework: {model.framework.type}")
