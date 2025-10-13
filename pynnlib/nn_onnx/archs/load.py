import onnx


def load_onnx_model(
    model_path: str,
    device: str = 'cpu'
) -> tuple[onnx.ModelProto | None, dict[str, str]]:
    onnx_model: onnx.ModelProto = onnx.load_model(model_path)
    metadata: dict[str, str] = {}

    return onnx_model, metadata






