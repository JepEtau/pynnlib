from collections import OrderedDict
from .archs.arch import MODEL_ARCHITECTURES
from .archs.parser import get_model_arch
from .archs.save import save_as
from .inference.session import OnnxSession
from pynnlib.framework import (
    NnFramework,
    NnFrameworkType,
)


FRAMEWORK: NnFramework = NnFramework(
    type=NnFrameworkType.ONNX,
    architectures=OrderedDict((a.name, a) for a in MODEL_ARCHITECTURES),
    get_arch=get_model_arch,
    save=save_as,
    Session=OnnxSession,
)
