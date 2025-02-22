from copy import deepcopy
from pprint import pprint
import numpy as np
from pynnlib.logger import nnlogger
from pynnlib.nn_types import Idtype
from pynnlib.utils.p_print import *
import tensorrt as trt
from pynnlib.model import OnnxModel
from ..inference.session import TRT_LOGGER
from ..trt_types import ShapeStrategy, TrtEngine



# https://github.com/NVIDIA/TensorRT/blob/main/demo/BERT/builder.py#L405

def onnx_to_trt_engine(
    model: OnnxModel,
    device: str,
    dtypes: set[Idtype],
    shape_strategy: ShapeStrategy,
) -> TrtEngine:
    """
    Convert an ONNX model to a serialized TensorRT engine.

    Restrictions:
        support only a single input tensor

    """

    has_fp16: bool = bool('fp16' in dtypes)
    has_bf16: bool = bool('bf16' in dtypes)

    print(f"[V] Start converting to TRT, request fp16={has_fp16}, bf16={has_bf16}")

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
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
            raise ValueError("Failed to parse the onnx model")

        # Input tensors
        input_tensor = None
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            break
        if input_tensor is None:
            raise ValueError("Missing input tensor in model")
        input_name = input_tensor.name
        is_fp16 = True if trt.nptype(input_tensor.dtype) == np.float16 else has_fp16

        nnlogger.debug(f"is_fp16: {is_fp16}, to_fp16: {has_fp16}")
        print(f"[V]   fp16={is_fp16}, bf16={has_bf16}")
        print(f"[V]   input shape: {input_tensor.shape}")

        # builder.max_batch_size = 1

        # Create a build configuration specifying how TensorRT should optimize the model
        builder_config = builder.create_builder_config()

        # builder_config.set_flag(trt.BuilderFlag.REFIT)
        # 1GB
        # builder_config.max_workspace_size = 1 << 30
        # builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        # builder_config.avg_timing_iterations = 10

        # tactic_source = builder_config.get_tactic_sources() & ~(1 << int(trt.TacticSource.CUDNN))
        # builder_config.set_tactic_sources(tactic_source)

        # builder_config.set_preview_feature(
        #     trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, False)

        # Set config:
        # config.
        #   num_optimization_profiles
        #   profiling_verbosity
        #   engine_capability
        #   builder_optimization_level
        #   hardware_compatibility_level
        #   flags
        # config.flag

        # optimization level: default=3
        # builder.builder_optimization_level = 5

        if is_fp16 or has_fp16:
            if builder.platform_has_fast_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
            else:
                raise RuntimeError("Error: fp16 is requested but this platform does not support it")

        if has_bf16:
            builder_config.set_flag(trt.BuilderFlag.BF16)

        onnx_h, onnx_w = input_tensor.shape[2:]
        static_onnx: bool = bool(onnx_h != -1 and onnx_w != -1)

        if static_onnx:
            print(yellow("onnx_to_trt_engine: static onnx model"))
            profile = builder.create_optimization_profile()
            profile.set_shape(
                input=input_name,
                min=input_tensor.shape,
                opt=input_tensor.shape,
                max=input_tensor.shape,
            )
            builder_config.add_optimization_profile(profile)

        elif not shape_strategy.static:
            print(yellow("onnx_to_trt_engine: dynamic onnx model"))
            profile = builder.create_optimization_profile()
            batch_opt = 1

            strategy = deepcopy(shape_strategy)
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

