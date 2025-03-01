from pynnlib.architecture import NnPytorchArchitecture, SizeConstraint
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
import os


ml_models_dir: str = "A:\\ml_models"

def parse(model: PyTorchModel) -> None:
    from module.build_efficienttam import build_efficienttam_video_predictor

    in_nc: int = 3
    out_nc: int = in_nc

    model_dir: str = os.path.join(ml_models_dir, "EfficientTAM")
    # ti=tiny, s=small
    checkpoint_fp = os.path.join(model_dir, "efficienttam_s_512x512.pt")
    config_fp = os.path.join(model_dir, "efficienttam_s_512x512.yaml")

    device = "cuda:0"

    del model.state_dict

    module = build_efficienttam_video_predictor(
        config_fp,
        ckpt_path=checkpoint_fp,
        device=device,
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
        vos_optimized=False,
    )

    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=out_nc,

        module=module
    )


MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="EfficientTAM",
        category="segmentation",
        detection_keys=("efficienttam"),
        parse=parse,
        dtypes=('fp32'),
        size_constraint=SizeConstraint(
            min=(8, 8)
        )
    ),
)
