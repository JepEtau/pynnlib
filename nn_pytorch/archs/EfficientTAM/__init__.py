import sys
from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PyTorchModel
import os

# models: https://huggingface.co/yunyangx/efficient-track-anything/tree/main
# TODO: define a default location for models
ml_models_dir: str = "A:\\ml_models"
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def parse(model: PyTorchModel) -> None:
    in_nc: int = 3
    out_nc: int = in_nc

    model_dir: str = os.path.abspath(
        os.path.expanduser(os.path.join(ml_models_dir, "EfficientTAM"))
    )
    # ti=tiny, s=small
    checkpoint_fp = os.path.join(model_dir, "efficienttam_s_512x512.pt")
    config_fp =  os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "efficient_track_anything",
            "configs",
            "efficienttam",
            "efficienttam_s_512x512.yaml"
        )
    )
    if not os.path.isfile(config_fp):
        raise FileExistsError(f"Missing file: {config_fp}")
    if not os.path.isfile(checkpoint_fp):
        raise FileExistsError(f"Missing file: {checkpoint_fp}")

    del model.state_dict

    from .module.etam import _VirtualEfficientTAM
    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=out_nc,

        ModuleClass=_VirtualEfficientTAM,
        filepath=checkpoint_fp,
        config_fp=config_fp,
    )


efficient_tam_arch = NnPytorchArchitecture(
    name="EfficientTAM",
    category="segmentation",
    detection_keys=(
        "efficienttam",
    ),
    parse=parse,
    dtypes=('fp32'),
    size_constraint=SizeConstraint(
        min=(8, 8)
    )
)

MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    efficient_tam_arch,
)

PREDEFINED_MODEL_ARCHITECTURES: dict[str, NnPytorchArchitecture] = {
    "efficienttam_s_512x512": efficient_tam_arch,
}
