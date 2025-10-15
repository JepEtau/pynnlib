from hutils import get_extension
import json
import os
from pprint import pprint
import zipfile



def load_tensorrt_engine(
    model_path: str,
    device: str = 'cpu'
) -> tuple[bytes, dict[str, str]]:
    # We don't load engine in the cuda device, only in RAM,
    # # loading in device will be done later....

    engine_bytes: bytes | None = None
    metadata: dict[str, str] = {}

    # Load an engine from storage
    if not os.path.exists(model_path):
        return None, metadata

    ext = get_extension(model_path)
    if ext == '.trtzip':
        with zipfile.ZipFile(model_path, 'r') as trtzip_file:
            engine_filename: str = next(
                (s for s in trtzip_file.namelist() if s.endswith('.engine')),
                None
            )
            if engine_filename is None:
                raise ValueError("[E] Not a valid trtzip file")
            engine_bytes = trtzip_file.read(engine_filename)
            metadata = json.loads(trtzip_file.read("metadata.json"))


    elif ext == '.engine':
        with open(model_path, 'rb') as f:
            engine_bytes = f.read()

    return engine_bytes, metadata
