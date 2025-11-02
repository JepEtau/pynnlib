from datetime import datetime
import textwrap
from hutils import get_extension, lightcyan
import time
from typing import Any
from warnings import warn

from .nn_types import NnFrameworkType
from .model import NnModel


def parse_metadata_(model: NnModel) -> None:

    # torch, ckpt
    # state_dict.get('metadata', ""))

    # safetensors
    # # Load existing metadata
    # with safe_open("model.safetensors", framework="pt") as f:
    #     old_metadata = f.metadata()

    # onnx
    # model.metadata_props

    pass


def generate_metadata(
    model: NnModel,
    metadata: dict[str, Any] = {}
) -> dict[str, str]:

    builtin_metadata = {
        'date_modified': datetime.strptime(
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            '%Y-%m-%dT%H:%M:%S%z'
        ).isoformat(),
        'generated_by': 'pynnlib'
    }

    # Overwrite metadata by builtin
    if metadata:
        metadata.update(builtin_metadata)
    else:
        metadata = builtin_metadata

    if model.framework.type == NnFrameworkType.ONNX:
        if model.arch_name.lower() not in ('unknown', 'generic'):
            metadata['arch_name'] = model.arch_name

        if model.force_weak_typing:
            metadata['typing'] = "weak"
        elif model.arch.to_tensorrt is not None:
            metadata['typing'] = (
                "weak"
                if model.arch.to_tensorrt.weak_typing
                else "strong"
            )

    elif model.framework.type == NnFrameworkType.PYTORCH:
        ext = get_extension(model.filepath)
        if ext in (".pth", ".ckpt"):
            model.state_dict.pop('metadata', None)
            model.state_dict['metadata'] = metadata

        elif ext == ".safetensors":
            # use model.metadata when saving file
            # state_dict = load_file(filepath)
            # save_file(state_dict, filepath, metadata=new_metadata)
            pass

        elif ext == '.pt':
            # it may be a torch.jit.ScriptModule or a regular torch model
            # not supported.
            warn(f"Adding metadata to a \'{ext}\' file is not supported")

    elif model.framework.type == NnFrameworkType.TENSORRT:
        metadata['trtzip_version'] = "1.0"
        if model.arch_name.lower() not in ('unknown', 'generic'):
            metadata['arch_name'] = model.arch_name

        metadata['opset'] = f"{model.opset}"

        metadata['shapes'] = model.shape_strategy.type
        if model.force_weak_typing:
            # weak typing may be forced for testing puprose, use it
            metadata['typing'] = "weak"

        elif (
            model.torch_arch is not None
            and model.torch_arch.to_tensorrt is not None
        ):
            metadata['typing'] = (
                "weak" if model.torch_arch.to_tensorrt.weak_typing else "strong"
            )
        elif model.typing:
            metadata['typing'] = model.typing

    return metadata



def print_metadata(metadata: dict[str, str]) -> None:
    indent = " " * 8
    print( f"{indent}metadata:")
    indent += " " * 4
    # aligns wrapped comment lines
    line_width = 80
    subindent = " " * 18
    for key, value in metadata.items():
        key_fmt = lightcyan(f"{key.title():<15} : ")

        if key.lower() == "comment" and isinstance(value, str):
            # Split paragraphs while preserving blank lines
            paragraphs = [p.strip() for p in value.split("\n") if p.strip()]

            first_para = True
            for para in paragraphs:
                wrapped = textwrap.fill(
                    para,
                    width=line_width,
                    initial_indent=indent + (key_fmt if first_para else subindent),
                    subsequent_indent=indent + subindent,
                )
                print(wrapped)
                first_para = False
                # blank line between paragraphs
                print()
        else:
            print(f"{indent}{key_fmt}{value}")
