from __future__ import annotations
from pprint import pprint
import re
from typing import Literal
import onnxruntime as ort
# ort.set_default_logger_severity(0)
import numpy as np
import torch
import torch.nn.functional as F

from pynnlib.import_libs import is_cuda_available
from pynnlib.model import OnnxModel
from pynnlib.nn_types import Idtype
from pynnlib.session import GenericSession
from pynnlib.utils.torch_tensor import (
    IdtypeToNumpy,
    IdtypeToTorch,
    flip_r_b_channels,
    to_nchw,
    to_hwc,
    torch_dtype_to_np,
)


class OnnxSession(GenericSession):
    """An example of session used to perform the inference using an Onnx model.
    """

    def __init__(self, model: OnnxModel):
        super().__init__()
        self.model: OnnxModel = model
        self.execution_providers: list[str | tuple[str, int]] = [
            "CPUExecutionProvider"
        ]
        self.device_type: str = 'cpu'
        self.cuda_device_id: int = 0


    def initialize(
        self,
        device: Literal['cpu', 'dml'] = 'cpu',
        dtype: Idtype | torch.dtype = 'fp32',
        **kwargs,
    ):
        super().initialize(device=device, dtype=dtype)
        self.execution_providers = [
            "CPUExecutionProvider",
        ]

        self.cuda_device_id: int = 0
        if 'cuda' in device:
            if (
                is_cuda_available()
                and 'CUDAExecutionProvider' in ort.get_available_providers()
            ):
                if (match := re.match(re.compile(r"^cuda[:]?(\d)?$"), device)):
                    self.cuda_device_id = (
                        int(match.group(1))
                        if match.group(1) is not None
                        else 0
                    )
                    self.execution_providers.insert(
                        0, ('CUDAExecutionProvider', {"device_id": self.cuda_device_id})
                    )
                    self.device_type = 'cuda'
            else:
                raise ValueError("Unsupported hardware: cuda")

        elif device == 'dml':
            self.device_type = 'dml'
            if 'DmlExecutionProvider' in ort.get_available_providers():
                self.execution_providers.insert(
                    0, ('DmlExecutionProvider', {"device_id": 0})
                )
            else:
                raise ValueError("Unsupported hardware: DirectML")

        model_proto = self.model.model_proto
        byte_model: bytes = model_proto.SerializeToString()

        session_options = ort.SessionOptions()
        # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if device == 'dml':
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.add_session_config_entry("session.use_dml", "1")
            session_options.add_session_config_entry("session.use_arena_allocator", "1")
            session_options.add_session_config_entry("session.use_arena", "1")
            # https://onnxruntime.ai/docs/performance/tune-performance/threading.html
            # session_options.intra_op_num_threads = 4
        try:
            self.session = ort.InferenceSession(
                byte_model,
                sess_options=session_options,
                providers=self.execution_providers
            )
        except:
            raise RuntimeError("Cannot create an Onnx session")

        fp16: bool = bool(dtype == 'fp16' and 'fp16' in self.model.dtypes)
        if fp16 and 'fp16' not in self.model.dtypes:
            raise ValueError("Half datatype (fp16) is not supported by this model")

        if (
            device == 'cpu'
            and 'fp32' not in self.model.dtypes
        ):
            raise ValueError(f"The execution provider ({device}) does not support the datatype of this model (fp16)")

        self.input_name: str = self.session.get_inputs()[0].name
        self.output_name: str = self.session.get_outputs()[0].name





    def run_np(self, in_img: np.ndarray) -> np.ndarray:
        # Unused: first PoC, use np for image to tensor
        # keep for historical reason
        if in_img.dtype != np.float32:
            raise NotImplementedError("Only float32 input image is supported")

        if self.dtype == torch.float16:
            in_img = in_img.astype(np.float16, copy=False)
        in_img = flip_r_b_channels(in_img)
        in_img = to_nchw(in_img)
        in_img = np.ascontiguousarray(in_img)

        out_img: np.ndarray = self.session.run(
            [self.output_name],
            {self.input_name: in_img}
        )[0]

        out_img = to_hwc(out_img)
        out_img = flip_r_b_channels(out_img)
        out_img = out_img.clip(0, 1., out=out_img)
        if self.dtype == torch.float16:
            out_img = out_img.astype(np.float32)

        return np.ascontiguousarray(out_img)


    def infer(self, in_img: np.ndarray, *args, **kwargs) -> np.ndarray:
        if in_img.dtype != np.float32:
            raise ValueError("np.float32 img only")

        in_h, in_w, c = in_img.shape
        in_tensor_dtype: torch.dtype = IdtypeToTorch[self.model.io_dtypes['input']]

        out_shape = (in_h * self.model.scale, in_w * self.model.scale, c)
        out_tensor_shape = (1, self.model.out_nc, *out_shape[:2])
        out_tensor_dtype: torch.dtype = IdtypeToTorch[self.model.io_dtypes['output']]

        device: str = 'cpu'
        if self.device_type == 'cuda':
            device = f"cuda:{self.cuda_device_id}"

        # Image to tensor
        in_tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(in_img))
        in_tensor = in_tensor.to(device=device, dtype=in_tensor_dtype)
        in_tensor = flip_r_b_channels(in_tensor)
        in_tensor = to_nchw(in_tensor)
        if c == 4:
            alpha = in_tensor[:, 3:4, :, :]
            in_tensor = in_tensor[:, :3, :, :]
        in_tensor = in_tensor.contiguous()

        # directml uses 'cuda' even if not a cuda device
        device_type: str = 'cuda' if self.device_type == 'dml' else self.device_type
        device_type = 'cpu'
        # Move input to CPU
        if 'cuda' in in_tensor.device.type:
            in_tensor = in_tensor.cpu()

        in_tensor = in_tensor.numpy()

        session: ort.InferenceSession = self.session

        # Access provider options and device info
        # providers = session.get_providers()
        # print("Execution Providers:", providers)

        # provider_options = session.get_provider_options()
        # print("Provider Options:", provider_options)


        output_name = session.get_outputs()[0].name
        outputs = session.run([output_name], {self.input_name: in_tensor})
        out_tensor = outputs[0]
        # print(out_tensor.shape)
        # print(out_tensor.dtype)
        # print(type(out_tensor))
        out_tensor = torch.from_numpy(out_tensor)

        # # Allocate memory
        # out_tensor: torch.Tensor = torch.empty(
        #     out_tensor_shape, dtype=out_tensor_dtype, device=device,
        # ).contiguous()
        # out_tensor = out_tensor.numpy()


        # print(self.input_name)
        # print(in_tensor.shape)
        # print(self.output_name)
        # print(out_tensor.shape)
        # print(device_type)

        # # IO bindings
        # session: ort.InferenceSession = self.session
        # io_binding = session.io_binding()
        # in_ort_tensor = ort.OrtValue.ortvalue_from_numpy(in_tensor, 'cpu', 0)
        # io_binding.bind_input(
        #     name=self.input_name,
        #     device_type=device_type,
        #     device_id=0 if device in ('cpu', 'dml') else self.cuda_device_id,
        #     element_type=IdtypeToNumpy[self.model.io_dtypes['input']],
        #     shape=in_tensor.shape,
        #     buffer_ptr=in_ort_tensor.data_ptr(),
        # )
        # print("binded input")
        # out_ort_tensor = ort.OrtValue.ortvalue_from_numpy(out_tensor, 'cpu', 0)

        # io_binding.bind_output(
        #     name=self.output_name,
        #     device_type='cuda',
        #     device_id=0 if device in ('cpu', 'dml') else self.cuda_device_id,
        #     element_type=IdtypeToNumpy[self.model.io_dtypes['output']],
        #     shape=out_tensor.shape,
        #     buffer_ptr=out_ort_tensor.data_ptr(),
        # )

        # print("Output names:", session.get_outputs())
        # print("Bindings:", io_binding.get_outputs())
        # print(ort.get_device())

        # # Inference
        # print("Session provider:", session.get_providers())
        # print("run")
        # print(self.device_type)
        # pprint(io_binding)
        # print(IdtypeToNumpy[self.model.io_dtypes['input']])
        # print(IdtypeToNumpy[self.model.io_dtypes['output']])
        # print(in_tensor.device)
        # print(in_tensor.dtype)
        # try:
        #     session.run_with_iobinding(io_binding)
        # except Exception as e:
        #     print(f"Error during inference: {e}")
        # print("done")

        out_tensor = torch.clamp(out_tensor, 0, 1)
        if c == 4:
            scale = out_tensor.shape[-1] // alpha.shape[-1]
            alpha = F.interpolate(
                alpha,
                scale_factor=scale,
                mode='bilinear',
                align_corners=False
            )
            out_tensor = torch.cat([out_tensor, alpha], dim=1)
            out_tensor = out_tensor.contiguous()

        out_tensor = to_hwc(out_tensor)
        out_tensor = flip_r_b_channels(out_tensor)
        out_tensor = out_tensor.float()
        out_img: np.ndarray = out_tensor.detach().cpu().numpy()

        return np.ascontiguousarray(out_img)

