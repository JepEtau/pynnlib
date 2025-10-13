# pynnlib: Python Neural Network Library


## Parse models
example:
`python scripts/parse_models -d ~/ml_models --filter pytorch`


## Convert a model
example, fixed shape
`python scripts.convert_model -trt -m A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k.pth -opt 640x480 -fixed -bf16`
append `-f` or `--force` to overwrite existing engine

## Single image inference
- TensorRT
`python scripts.img_infer -i .\img_640x480.png  -m  A:\ml_models\2x_Pooh_DAT-2_Candidate_1_305k_cc8.9_op20_fp32_bf16_640x480_640x480_640x480_10.9.0.34.engine -bf16`


- CUDA
`python scripts.img_infer --img img.png --model 2x-LD-Anime-Compact.pth --cuda`

