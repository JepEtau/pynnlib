from collections import OrderedDict
from .archs.arch import MODEL_ARCHITECTURES
from .archs.parser import get_tensorrt_model_arch
from .archs.save import save_as
from .inference.session import TensorRtSession
from pynnlib.framework import (
    NnFramework,
    NnFrameworkType,
)

FRAMEWORK: NnFramework = NnFramework(
    type=NnFrameworkType.TENSORRT,
    architectures=OrderedDict((a.name, a) for a in MODEL_ARCHITECTURES),
    # load=,
    get_arch=get_tensorrt_model_arch,
    save=save_as,
    Session=TensorRtSession,
)
