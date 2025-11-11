import sys
import numpy as np
import torch
from torch import Tensor

from ..nn_types import Hdtype


# https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_utils.py
# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
np_dtype_to_torch = {
    np.bool_      : torch.bool,
    np.uint8      : torch.uint8,
    np.uint16     : torch.uint16,
    np.uint32     : torch.uint32,
    np.uint64     : torch.uint64,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128,
    np.dtype('float32'): torch.float32,
    np.dtype('float16'): torch.float16,
    np.dtype('uint8'): torch.uint8,
    np.dtype('uint16'): torch.uint16,
}

_torch_max_value: dict[torch.dtype, float] = {
    torch.uint8: 255.,
    torch.uint16: 65535.,
    torch.bfloat16: 1.,
    torch.float16: 1.,
    torch.float32: 1.,
}

if sys.platform == "win32":
    # Size of `np.intc` is platform defined.
    # It is returned by functions like `bitwise_not`.
    # On Windows `int` is 32-bit
    # https://docs.microsoft.com/en-us/cpp/cpp/data-type-ranges?view=msvc-160
    np_dtype_to_torch[np.intc] = torch.int

# Dict of torch dtype -> NumPy dtype
torch_dtype_to_np: dict[torch.dtype, np.dtype] = {
    value: key
    for (key, value) in np_dtype_to_torch.items()
}
# np_to_torch_dtype_dict.update({
#     torch.bfloat16: np.float32,
#     torch.complex32: np.complex64
# })


HdtypeToTorch: dict[Hdtype, torch.dtype] = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}
HdtypeToNumpy: dict[Hdtype, torch.dtype] = {
    'fp32': np.float32,
    'fp16': np.float16,
}
TorchDtypeToHdtype: dict[torch.dtype, Hdtype] = {
    value: key
    for (key, value) in HdtypeToTorch.items()
}

def to_nchw(t: Tensor) -> Tensor:
    shape_size = len(t.shape)
    if shape_size == 3:
        # (H, W, C) -> (1, C, H, W)
        return t.permute(2, 0, 1).unsqueeze(0)

    elif shape_size == 2:
        # (H, W) -> (1, 1, H, W)
        return t.unsqueeze(0).unsqueeze(0)

    else:
        raise ValueError("Unsupported input tensor shape")


def to_hwc(t: Tensor) -> Tensor:
    if len(t.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return t.squeeze(0).permute(1, 2, 0)

    else:
        raise ValueError("Unsupported output tensor shape")


def flip_r_b_channels(t: Tensor) -> Tensor:
    if t.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return t[:, :, [2, 1, 0]].contiguous()

    elif t.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return t[:, :, [2, 1, 0, 3]].contiguous()

    return t


def img_to_tensor(
    d_img: Tensor,
    tensor_dtype: torch.dtype,
    flip_r_b: bool = False,
) -> Tensor:
    """ Create a 4D tensor from a 3D image (tensor type), reshaped and normalized
    """
    d_tensor: Tensor = d_img
    img_dtype: torch.dtype = d_img.dtype
    if flip_r_b:
        d_tensor = flip_r_b_channels(d_tensor)
    d_tensor = to_nchw(d_tensor)

    if not isinstance(img_dtype, torch.dtype):
        raise TypeError(f"img_to_tensor: erroneous image dtype: {img_dtype}")
    if not isinstance(tensor_dtype, torch.dtype):
        raise TypeError(f"img_to_tensor: erroneous tensor dtype: {tensor_dtype}")
    divisor: float = (
        _torch_max_value[img_dtype]
        if tensor_dtype != d_img.dtype
        else 1.
    )
    if divisor != 1.:
        d_tensor = d_tensor.to(dtype=torch.float32) / divisor

    d_tensor = d_tensor.to(dtype=tensor_dtype)
    return d_tensor.contiguous()



def tensor_to_img(
    tensor: Tensor,
    img_dtype: np.dtype | torch.dtype,
    flip_r_b: bool = False,
) -> Tensor:
    """Convert a 4D tensor to an image (h,w,c)
    - dtype: image dtype
    - flip_r_b: flip red and blue channels
    """
    d_img: Tensor = to_hwc(tensor)
    if flip_r_b:
        d_img = flip_r_b_channels(d_img)

    tensor_dtype: torch.dtype = tensor.dtype
    img_dtype: torch.dtype = (
        img_dtype
        if isinstance(img_dtype, torch.dtype)
        else np_dtype_to_torch[img_dtype]
    )

    multiplier: float = 1.
    if tensor_dtype != img_dtype:
        num = _torch_max_value[img_dtype]
        denum = _torch_max_value[tensor_dtype]
        multiplier: float = num / denum

    if multiplier != 1.:
        d_img = d_img.to(dtype=torch.float32)
        d_img = d_img * multiplier

    return d_img.to(dtype=img_dtype).contiguous()
