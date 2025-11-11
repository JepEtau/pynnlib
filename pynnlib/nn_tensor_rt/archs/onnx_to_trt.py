from copy import deepcopy
from hytils import red, yellow
import numpy as np
from pprint import pprint

from pynnlib.logger import nnlogger
from pynnlib.nn_types import Hdtype, ShapeStrategy
from pynnlib.import_libs import trt
TrtDType = trt.DataType

from pynnlib.model import OnnxModel
from ..inference.session import TRT_LOGGER
from ..trt_types import TrtEngine



# https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/builder.py#L405

def onnx_to_trt_engine(
    model: OnnxModel,
    device: str,
    dtypes: set[Hdtype],
    shape_strategy: ShapeStrategy,
) -> TrtEngine:
    """
    Convert an ONNX model to a serialized TensorRT engine.

    Restrictions:
        support only a single input tensor

    """
    # Typing is strong by default, it can be forced to weak
    # if the initial arch forces it or if forced by the user.
    tensorrt_version = list(map(int, trt.__version__.split(".")))
    is_tensorrt_min_10_12 = bool(tensorrt_version[0] * 100 + tensorrt_version[1] >= 1012)
    use_strongly_typed: bool = False
    if is_tensorrt_min_10_12 and model.torch_arch is not None:
        use_strongly_typed = not model.torch_arch.to_tensorrt.weak_typing
        if model.force_weak_typing:
            use_strongly_typed = False
    print(red(f"use_strongly_typed: {use_strongly_typed}"))

    has_fp16: bool = bool('fp16' in dtypes)
    has_bf16: bool = bool('bf16' in dtypes)

    print(f"[V] Start converting to TRT, request fp16={has_fp16}, bf16={has_bf16}")
    network_flags = 0
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if use_strongly_typed:
        network_flags += 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(flags=network_flags) as network,
        trt.OnnxParser(network, TRT_LOGGER) as onnx_parser
    ):
        runtime = trt.Runtime(TRT_LOGGER)

        # Parse the serializedonnx model
        if model.model_proto is not None:
            # Serialize the onnx model. Don't use filepath as it may
            # not an optimized onnx model
            serialized_onnx_model = model.model_proto.SerializeToString()
            success = onnx_parser.parse(serialized_onnx_model)
        if not success:
            for idx in range(onnx_parser.num_errors):
                print(f"Error: {onnx_parser.get_error(idx)}")
            raise ValueError("TensorRT: failed to parse the ONNX model")

        # Input tensors
        onnx_in_tensor = None
        for i in range(network.num_inputs):
            onnx_in_tensor = network.get_input(i)
            break
        if onnx_in_tensor is None:
            raise ValueError("Missing input tensor in model")
        input_name = onnx_in_tensor.name

        if onnx_in_tensor.dtype == TrtDType.BF16:
            is_onnx_fp16 = False
        else:
            is_onnx_fp16 = bool(trt.nptype(onnx_in_tensor.dtype) == np.float16)
            nnlogger.debug(f"is_onnx_fp16: {is_onnx_fp16}, to_fp16: {has_fp16}")
            print(trt.nptype(onnx_in_tensor.dtype))
            print(f"is_onnx_fp16: {is_onnx_fp16}, to_fp16: {has_fp16}")

        print(f"[V]   is_onnx_fp16={is_onnx_fp16}, bf16={has_bf16}")
        print(f"[V]   onnx input shape: {onnx_in_tensor.shape}")

        # builder.max_batch_size = 1

        # Create a build configuration specifying how TensorRT should optimize the model
        builder_config = builder.create_builder_config()

        # for debug
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # 1GB
        # 2025-03-15: don"t limit the memory pool bc of DAT2 (bf16)
        # builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # optimization level: default=3
        # builder.builder_optimization_level = 5

        if is_onnx_fp16 or has_fp16:
            has_fp16_support: bool = bool(
                getattr(builder, "platform_has_fast_fp16", None) is not None
            )
            print(f"has_fp16_support: {has_fp16_support}")
            if has_fp16_support:
                if builder.platform_has_fast_fp16:
                    if not use_strongly_typed:
                        builder_config.set_flag(trt.BuilderFlag.FP16)
                    print(f"[V]   set fp16 flag")
                else:
                    raise RuntimeError("Error: fp16 is requested but this platform does not support it")
            # else:
            #     builder_config.set_flag(trt.BuilderFlag.FP16)

        if not use_strongly_typed and has_bf16:
            print(f"use_strongly_typed: {use_strongly_typed}, has_bf16: {has_bf16}")
            builder_config.set_flag(trt.BuilderFlag.BF16)

        onnx_h, onnx_w = onnx_in_tensor.shape[2:]
        static_onnx: bool = bool(onnx_h != -1 and onnx_w != -1)
        strategy: ShapeStrategy = deepcopy(shape_strategy)
        fixed_trt: bool = shape_strategy.is_fixed()
        batch_opt = 1

        if static_onnx:
            print(yellow("onnx_to_trt_engine: static onnx model"))
        else:
            profile = builder.create_optimization_profile()
            if fixed_trt:
                shape = (batch_opt, model.in_nc, *reversed(strategy.opt_size))
                print(yellow("onnx_to_trt_engine: fixed trt"), f"{shape}")
                profile.set_shape(
                    input=input_name, min=shape, opt=shape, max=shape,
                )

            else:
                print(yellow("onnx_to_trt_engine: dynamic onnx model"))
                if strategy.min_size == (0, 0):
                    strategy.min_size = strategy.opt_size
                if strategy.max_size == (0, 0):
                    strategy.max_size = strategy.opt_size

                profile.set_shape(
                    input=input_name,
                    min=(batch_opt, model.in_nc, *reversed(strategy.min_size)),
                    opt=(batch_opt, model.in_nc, *reversed(strategy.opt_size)),
                    max=(batch_opt, model.in_nc, *reversed(strategy.max_size)),
                )

            builder_config.add_optimization_profile(profile)

        nnlogger.info("[I] Building a TensortRT engine; this may take a while...")
        engine_bytes = builder.build_serialized_network(network, builder_config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create Tensor RT engine")
        trt_engine = runtime.deserialize_cuda_engine(engine_bytes)

    return trt_engine

