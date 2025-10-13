from __future__ import annotations
from collections import OrderedDict

from .archs import MODEL_ARCHITECTURES
from .archs.load import load_state_dict
from .archs.parser import get_torch_model_arch
from .archs.save import save_as
from .inference.session import PyTorchSession
from pynnlib.framework import (
    NnFramework,
    NnFrameworkType,
)


FRAMEWORK: NnFramework = NnFramework(
    type=NnFrameworkType.PYTORCH,
    architectures=OrderedDict((a.name, a) for a in MODEL_ARCHITECTURES),
    load=load_state_dict,
    detect_arch=get_torch_model_arch,
    Session=PyTorchSession,
    save=save_as,
)
