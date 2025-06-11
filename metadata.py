from datetime import datetime
import time
from typing import Any
from warnings import warn

import onnx


from .nn_types import NnFrameworkType
from .model import NnModel
from .utils import get_extension


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


def set_metadata_(model: NnModel, metadata: dict[str, Any] = {}) -> None:

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
        model_proto: onnx.ModelProto = model.model_proto
        del model_proto.metadata_props[:]
        for k, v in metadata.items():
            if not isinstance(v, str):
                try:
                    v = str(v)
                except:
                    warn(f"Value for key \'{k}\' cannot be converted to a valid string")
                    continue
            entry = model_proto.metadata_props.add()
            entry.key = k
            entry.value = v

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





    model.metadata = metadata
