from __future__ import annotations
from collections.abc import Callable
from contextlib import nullcontext
from pprint import pprint
from warnings import warn
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, autocast
from typing import TYPE_CHECKING
from pynnlib import is_cuda_available
from pynnlib.logger import nnlogger
from pynnlib.model import PyTorchModel
from pynnlib.architecture import SizeConstraint
from pynnlib.nn_types import Idtype
from pynnlib.session import GenericSession
from hutils import (
    yellow,
    red,
)
from pynnlib.utils.torch_tensor import (
    img_to_tensor,
    tensor_to_img,
    to_nchw,
    flip_r_b_channels,
    to_hwc
)
if TYPE_CHECKING:
    from pynnlib.architecture import InferType
from torch import Tensor


class PyTorchSession(GenericSession):
    """An example of session used to perform the inference using a PyTorch model.
    There is a reason why the initialization is not done in the __init__: mt/mp
    It maybe be overloaded by a customized session
    """

    def __init__(self, model: PyTorchModel):
        super().__init__()
        self.model: PyTorchModel = model
        self.device: torch.device = torch.device('cpu')

        try:
            self.model.module.load_state_dict(self.model.state_dict, strict=True)
        except:
            print(red(f"Try loading {self.model.arch.name} with `strict=False`, correct this!"))
            self.model.module.load_state_dict(self.model.state_dict, strict=False)

        for _, v in self.model.module.named_parameters():
            v.requires_grad = False

        self._process_fct: Callable[[np.ndarray], np.ndarray] = self._torch_process
        infer_type: InferType = model.arch.infer_type
        if (
            infer_type.type == 'inpaint'
            and infer_type.inputs == 2
            and infer_type.outputs == 1
        ):
            self._process_fct = self._torch_process_inpaint

        elif (
            infer_type.type != 'temporal'
            and (infer_type.inputs != 1 or infer_type.outputs != 1)
        ):
            raise NotImplementedError(f"Cannot create a session for arch {model.arch_name} ")


    @property
    def module(self) -> nn.Module:
        return self.model.module

    @module.setter
    def module(self, module: nn.Module) -> None:
        self.model.module = module

    @property
    def infer_stream(self) -> torch.cuda.Stream:
        return self._infer_stream


    @infer_stream.setter
    def infer_stream(self, stream: torch.cuda.Stream) -> None:
        self._infer_stream = stream


    def initialize(
        self,
        device: str,
        dtype: Idtype | torch.dtype = 'fp32',
        warmup: bool = False,
        **kwargs,
    ) -> None:
        # module: nn.Module = self.module
        super().initialize(device=device, dtype=dtype)

        if not is_cuda_available() or dtype not in self.model.arch.dtypes:
            self.dtype = 'fp32'
            self.device = 'cpu'
            print(yellow(f"Initialize session, fallback to {self.dtype}"))

        nnlogger.debug(f"[V] Initialize a PyTorch inference session ({self.dtype})")
        torch.backends.cudnn.enabled = True

        self.module = self.module.eval()
        for param in self.module.parameters():
            param.requires_grad = False

        nnlogger.debug(f"[V] load model to {self.device}, {dtype}, dtype={self.dtype}")
        self.module = self.module.to(self.device)
        self.module = self.module.to(dtype=self.dtype)

        if warmup and 'cuda' in device:
            self.warmup(3)


    def warmup(self, count: int = 1):
        size_constraint: SizeConstraint = self.model.size_constraint
        if size_constraint is not None and size_constraint.min is not None:
            shape = (*reversed(size_constraint.min), self.model.in_nc)
        else:
            shape = (32, 32, self.model.in_nc)
        nnlogger.debug(f"[V] warmup ({count}x) with a random img ({shape})")

        # TODO: specific warmup
        imgs: list[np.ndarray] = list([
            np.random.random(shape).astype(np.float32)
            for _ in range(self.model.arch.infer_type.inputs)
        ])
        for _ in range(count):
            self._process_fct(*imgs)

    @torch.inference_mode()
    def infer(self, in_img: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self._process_fct(in_img, *args, **kwargs)


    @torch.inference_mode()
    def _torch_process(self, in_img: np.ndarray) -> np.ndarray:
        """Example of how to perform an inference session.
        This is an unoptimized function
        """
        nnlogger.debug(f"[V] inference: img: {in_img.shape}, {in_img.dtype}")
        if self.dtype == torch.bfloat16:
            context = torch.autocast(
                device_type="cuda" if "cuda" in self.device else "cpu",
                dtype=self.dtype
            )
        else:
            context = nullcontext()

        in_dtype: np.dtype = in_img.dtype
        in_img: Tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        d_in_img: Tensor = in_img.to(device=self.device)
        d_in_tensor: Tensor = img_to_tensor(
            d_img=d_in_img,
            tensor_dtype=self.dtype,
            flip_r_b=True,
        )
        nnlogger.debug(f"[V]   in tensor: {d_in_tensor.shape}, {d_in_tensor.dtype}")

        with context:
            d_out_tensor: Tensor = self.module(d_in_tensor)
        # d_out_tensor = d_in_tensor
        d_out_tensor = torch.clamp(d_out_tensor, 0., 1.)

        nnlogger.debug(f"[V]   out tensor: {d_out_tensor.shape}, {d_out_tensor.dtype}")
        d_out_img: Tensor = tensor_to_img(
            tensor=d_out_tensor,
            img_dtype=in_dtype,
            flip_r_b=True,
        )
        out_img: np.ndarray = d_out_img.contiguous().cpu().numpy()
        nnlogger.debug(f"[V]   out img: {out_img.shape}, {out_img.dtype}")

        return out_img



    @torch.inference_mode()
    def _torch_process_inpaint(
        self,
        in_img: np.ndarray,
        in_mask: np.ndarray
    ) -> np.ndarray:
        in_tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        in_tensor = in_tensor.to(self.device)
        in_tensor = in_tensor.to(dtype=self.dtype)
        in_tensor = flip_r_b_channels(in_tensor)
        in_tensor = to_nchw(in_tensor).contiguous()

        if len(in_mask.shape) > 2:
            gray = cv2.cvtColor(in_mask, cv2.COLOR_BGR2GRAY)
            _, in_mask = cv2.threshold(gray, 0.5, 1., cv2.THRESH_BINARY)

        in_tensor_mask = torch.from_numpy(np.ascontiguousarray(in_mask))
        in_tensor_mask = in_tensor_mask.to(self.device)
        in_tensor_mask = in_tensor_mask.half() if self.fp16 else in_tensor_mask.float()
        in_tensor_mask = in_tensor_mask / 255.
        in_tensor_mask = to_nchw(in_tensor_mask).contiguous()

        out_tensor: Tensor = self.module(in_tensor, in_tensor_mask)
        out_tensor = torch.clamp_(out_tensor, 0, 1)

        out_tensor = to_hwc(out_tensor)
        out_tensor = flip_r_b_channels(out_tensor)
        out_tensor = out_tensor.float()
        out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return out_img


