# pynnlib: Python Neural Network Library

A library I use as a **submodule** for my other projects.


> [!IMPORTANT]
> Developed in free time, **my current** choices:
> - Can be integrated as a submodule only
> - Not stable API: this library must not constraint the different applications that use it
> - New functionalities are integrated step by step
> - Only basic comments. Documentation won't be published (handwritten)
> - Only basic and non optimized inference sessions (slow) are integrated into this open-source project
> - Execution providers for pytorch and tensorRT only: Nvidia, cpu (partial)
> - Coding rules are simplified a lot
> - No systematic validation tests and not in this repo

<br/>

# Install this library

## As a library in an untracked project
Unzip the code in a directory name `pynnlib`
or clone at the root of a project
```
git clone https://github.com/JepEtau/pynnlib.git
```
## As a submodule in a git project
```
git submodule init
git submodule add https://github.com/JepEtau/pynnlib.git
```

<br/>

## Parse models
example:
`python -m scripts.parse_models -d ~/ml_models --filter pytorch`


## Convert a model
example, fixed shape
`python -m scripts.convert_model -trt -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k.pth -opt 640x480 -fixed -bf16`
append `-f` or `--force` to overwrite existing engine

## Single image inference
- TensorRT
`python -m scripts.img_infer -i .\img_640x480.png  -m  A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.9.0.34.engine -bf16`


- CUDA
`python -m scripts.img_infer --img img.png --model 2x-LD-Anime-Compact.pth --cuda`


<br/>

# How I use it
> [!CAUTION]
> The following examples are not validated and use a default session for the inference. Do not expect fast inference.


## Open a model

```python
from pynnlib import (
    nnlib,
    NnModel,
)

model: NnModel = nnlib.open(model_filepath)
```

## Dev: convert a model to a TensorRT model
...and save the engine

```python
from pynnlib import (
    nnlib,
    NnModel,
    TrtModel,
    ShapeStrategy,
)

model: NnModel = nnlib.open(model_filepath)
trt_model: TrtModel = nnlib.convert_to_tensorrt(
    model=model,
    shape_strategy=shape_strategy,
    dtype='fp16',
    # optimization_level=opt_level, # Not yet supported
    opset=opset,
    device=device,
    out_dir="output_dir",
)
```

## Dev: perform an inference
with a default inference session.<br/>
PyTorch to transfer an image (bgr, np.float32) to the execution provider (device) using pageable memory.


```python
from pynnlib import (
    nnlib,
    NnModel,
    NnModelSession,
)
# ...

# Open an image, datatype must be np.float32
in_img: np.ndarray = load_image(img_filepaths)

# Open a model
model: NnModel = nnlib.open(model_filepath)

# Create a session
session: NnModelSession = nnlib.session(model)
session.initialize(
    device=device,
    dtype='fp16',
)

# Perform inference
out_img: np.ndarray = session.process(in_img)

# ...

```

## Use a custom inference session

```python
from pynnlib import (
    nnlib,
    NnFrameworkType,
)

# PyTorch
nnlib.set_session_constructor(
    NnFrameworkType.PYTORCH, CustomPyTorchSession
)

# TensorRT
nnlib.set_session_constructor(
    NnFrameworkType.TENSORRT, CustomPyTensorRtSession
)

```
