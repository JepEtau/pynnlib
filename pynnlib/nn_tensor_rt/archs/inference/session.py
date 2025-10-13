from pynnlib.session import TensorRtSession
from pynnlib.model import TrtModel


def create_session(model: TrtModel) -> TensorRtSession:
    return model.framework.Session(model)

