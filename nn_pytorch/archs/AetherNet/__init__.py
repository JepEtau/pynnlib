from warnings import warn
from pynnlib.architecture import (
    Module,
    NnPytorchArchitecture,
    SizeConstraint,
    TensorRTConv,
)
from pynnlib.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import (
    get_max_indice,
    get_nsequences,
)
from ..torch_to_onnx import to_onnx
from pynnlib.logger import is_debugging


# (state_dict: StateDict, keys: tuple[str | tuple[str]]) -> bool:
def has_fused_keys(state_dict: StateDict, **kwargs) -> bool:
    """
    Checks if the state_dict contains keys typical of a fused AetherNet model.
    A fused model has 'fused_conv.weight' and 'fused_conv.bias' keys within its
    ReparamLargeKernelConv layers, but *not* the 'lk_conv', 'sk_conv', 'lk_bias', 'sk_bias' keys.
    """

    # fused: bool = False
    # unfused: bool = False
    # for key in state_dict:
    #     if '.conv.fused_conv.weight' in key:
    #         fused = True
    #     if '.conv.lk_conv.weight' in key or '.conv.sk_conv.weight' in key:
    #         unfused = True
    #         print("unfused detected")

    #     # If both fused and unfused keys exist, it's a problematic state_dict,
    #     # likely not a fully fused or unfused model.
    #     if fused and unfused:
    #         return False

    # # A model is considered fully fused if it has fused keys and no unfused keys.
    # return fused and not unfused

    def _is_unfused_key(key) -> bool:
        return (
            ".conv.lk_conv.weight" in key
            or ".conv.sk_conv.weight" in key
        )

    keys = list(state_dict)
    for i, key in enumerate(keys):
        if ".conv.fused_conv.weight" in key:
            # Fused
            for other_key in keys[i+1:]:
                if _is_unfused_key(other_key):
                    # If both fused and unfused keys exist,
                    # it's a problematic state_dict,
                    # likely not a fully fused or unfused model.
                    return False

            # A model is considered fully fused
            # if it has fused keys and no unfused keys.
            return True

        elif _is_unfused_key(key):
            warn("unfused")
            return False

    return False




def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict


    # num_feat, in_nc = state_dict["body.0.weight"].shape[:2]
    # num_conv: int = (max_indice - 2) // 2
    scale: int = 4

    in_nc: int = state_dict['conv_first.weight'].shape[1]
    out_nc: int = state_dict['conv_last.weight'].shape[0]

    embed_dim: int = state_dict['conv_first.weight'].shape[0]



    # Deduce depths by counting blocks in the 'layers' ModuleList.
    # It assumes the 'layers' structure is consistent across the model,
    # and ReparamLargeKernelConv (or its fused version) are inside.
    depths: list[int] = []
    # Check for both fused and unfused keys
    max_layer_group_idx = get_nsequences(state_dict=state_dict, seq_key="layers")
    if max_layer_group_idx:
        for i in range(max_layer_group_idx + 1):
            block_count_in_group = 0
            for key in state_dict.keys():
                if key.startswith(f'layers.{i}.') and ('.conv.lk_conv.weight' in key or '.conv.fused_conv.weight' in key):
                    try:
                        block_idx = int(key.split('.')[2])
                        block_count_in_group = max(block_count_in_group, block_idx + 1)
                    except ValueError:
                        continue
            if block_count_in_group > 0:
                depths.append(block_count_in_group)
    depths = tuple(depths)

    # Fallback for depths if deduction from state_dict keys is difficult or initial
    # This assumes standard AetherNet variants based on embed_dim
    if not depths and embed_dim > 0:
        if embed_dim == 96:
            depths = (4, 4, 4, 4)

        elif embed_dim == 128:
            depths = (6, 6, 6, 6, 6, 6)

        elif embed_dim == 180:
            depths = (8, 8, 8, 8, 8, 8, 8, 8)

        else:
            # Default to a safe tuple or raise error if specific depth is crucial.
            # This means model might not load correctly if arbitrary depths are used.
            warn(
                f"Warning: Could not deduce 'depths' for embed_dim={embed_dim}. Using default (4,4,4,4).")
            depths = (4, 4, 4, 4)

    # Deduce upscale factor from the Upsample layer's first conv output
    # `upsample.0.weight` is the weight for the first conv in Upsample module
    # Its output channels are num_feat * scale^2 (for PixelShuffle)
    # The `conv_before_upsample.0.weight` output channels provide `num_feat_upsample`
    num_feat_upsample = state_dict['conv_before_upsample.0.weight'].shape[0]
    upsample_first_conv_out_channels = state_dict['upsample.0.weight'].shape[0]

    scale = 4
    if num_feat_upsample > 0 and upsample_first_conv_out_channels > 0:
        ratio = upsample_first_conv_out_channels / num_feat_upsample
        # Check for 2x (4 = 2^2)
        if abs(ratio - 4.0) < 1e-6:
            scale = 2

        # Check for 3x (9 = 3^2)
        elif abs(ratio - 9.0) < 1e-6:
            scale = 3

        # For 4x, AetherNet uses two 2x PixelShuffles,
        # so `upsample` is `nn.Sequential(Conv2d, PixelShuffle, Conv2d, PixelShuffle)`
        # This means `upsample.2.weight` would exist.
        # Checks for the second PixelShuffle's preceding conv
        elif 'upsample.2.weight' in state_dict:
            scale = 4

        else:
            warn(f"Warning: Could not precisely deduce upscale from upsample layer. Defaulting to 4. Ratio: {ratio}")
    else:
        warn("Warning: Could not deduce upscale due to missing feature counts. Defaulting to 4.")


    # Deduce mlp_ratio from GatedFFN's first linear layer (fc1_gate)
    mlp_ratio: float = 2.
    # Check if a GatedFFN layer exists and deduce from its dimensions
    # Assuming at least one block in the first layer group has an FFN
    if depths and 'layers.0.0.ffn.fc1_gate.weight' in state_dict:
        # fc1_gate.weight.shape[0] is hidden_features, embed_dim is input_features
        hidden_features = state_dict['layers.0.0.ffn.fc1_gate.weight'].shape[0]
        if embed_dim > 0:
            # Round for typical ratios like 2.0, 2.5
            mlp_ratio = round(hidden_features / embed_dim, 1)

    # Usually fixed, not in state_dict
    drop_rate: float = 0
    drop_path_rate: float = 0.1


    # lk_kernel and sk_kernel are usually fixed per variant.
    # We can infer them if aether_small/medium/large follow a consistent pattern.
    lk_kernel: int = 11
    sk_kernel: int = 3
    # Aether Large
    if embed_dim == 180 and depths == (8, 8, 8, 8, 8, 8, 8, 8):
        lk_kernel = 13

    # img_range is often a fixed hyperparameter, not always directly deducible from weights.
    # Assuming common value 1.0 for [0,1] or 255.0 for [0,255]
    # If the model explicitly stores `img_range` in its state_dict (e.g., as `model.img_range`),
    # you could retrieve it, but it's not standard for generic checkpoints.
    img_range: float = 1.

    fused_init: bool = True

    # Detect variant name
    arch_name = model.arch.name
    if embed_dim == 96 and depths == (4, 4, 4, 4):
        arch_name = f"{arch_name} (small)"

    elif embed_dim == 128 and depths == (6, 6, 6, 6, 6, 6):
        arch_name = f"{arch_name} (medium)"

    elif embed_dim == 180 and depths == (8, 8, 8, 8, 8, 8, 8, 8):
        arch_name = f"{arch_name} (large)"

    if is_debugging():
        from .module.aether_arch import AetherNet
        model.update(ModuleClass=AetherNet)

    model.update(
        arch_name=arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        in_chans=in_nc,
        embed_dim=embed_dim,
        depths=depths,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        lk_kernel=lk_kernel,
        sk_kernel=sk_kernel,
        img_range=img_range,
        fused_init=fused_init,
    )




MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="AetherNet",
        detect=has_fused_keys,
        module=Module(file="aether_arch", class_name="AetherNet"),
        parse=parse,
        to_onnx=to_onnx,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(
            min=(64, 64)
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
        ),

    ),
)

