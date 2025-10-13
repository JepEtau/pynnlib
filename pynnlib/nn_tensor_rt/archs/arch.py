from .parser import parse
from .inference.session import create_session
from ...architecture import NnTensorrtArchitecture


def is_model_generic(model) -> bool:
    return True


MODEL_ARCHITECTURES: tuple[NnTensorrtArchitecture] = (
    NnTensorrtArchitecture(
        name='unknown',
        detect=is_model_generic,
        parse=parse,
        create_session=create_session,
        dtypes=('fp32', 'fp16', 'bf16'),
    ),
)
