:: python -m scripts.convert_model ^
::     --onnx ^
::     -m A:\ml_models\fastDAT\2x_animefilm_light_161k.pth ^
::     --static ^
::     --size 720x540 ^
::     --fp16 ^
::     --overwrite


:: python -m scripts.convert_model ^
::     --tensorrt ^
::     -m A:\ml_models\FDAT-Models-1.0.0-AF\2x_animefilm_light_161k_fp16_static_576x432.onnx ^
::     --fp16 ^
::     --overwrite ^
::     --verbose


:: python -m scripts.convert_model ^
::     --onnx ^
::     -m A:\ml_models\FDAT-Models-1.0.0-AF\2x_animefilm_light_161k.pth ^
::     --fp16 ^
::     --static ^
::     --size 720x540 ^
::     --overwrite ^
::     --verbose

python -m scripts.convert_model ^
    --tensorrt ^
    -m A:\ml_models\FDAT-Models-1.0.0-AF\2x_animefilm_light_161k.pth ^
    --fp16 ^
    --static ^
    --size 720x540 ^
    --weak ^
    --overwrite ^
    --verbose

