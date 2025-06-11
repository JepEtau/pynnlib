# pynnlib: Python Neural Network Library

A library I use as a **submodule** in my other projects.


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

