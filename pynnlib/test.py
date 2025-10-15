from pynnlib.model import NnModel

from hutils import path_split


def save_as(model_fp: str, model: NnModel) -> None:
    directory, basename, ext = path_split(model_fp)
    raise ValueError(get_extension(f"Unknown framework: {model.framework.type}"))
