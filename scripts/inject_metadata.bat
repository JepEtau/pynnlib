
::python -m scripts.inject_metadata ^
::    --model A:\ml_models\fastDAT\2x_animefilm_light_161k.pth ^
::    --name "2x AnimeFilm FDAT" ^
::    --author "SharekhaN" ^
::    --comment "Compression Removal, Halo Removal, Bleed\nRemoval, Ringing Removal, pre 2000's Anime\nRestoration \nThe dataset seems be able to support creation of a\nwell rounded, anime focused  model which can\ntarget a lot of film based DVD sources. Here the\nchoice of arch allows inference to be multiple\ntimes faster ~ maybe by 10 times as fast as DAT-2\ni.e. 0.5 fps vs 6.3 fps on a similar input. I\nthink the performance is very close and evenly\nmatched to the DAT-2 model." ^
::    --license "MIT"

:: or use a text file with args `--comment_file`

:: python -m scripts.inject_metadata ^
::     --model A:\ml_models\fastDAT\2x_animefilm_light_161k.pth ^
::     --name "2x AnimeFilm FDAT"

:: python -m scripts.inject_metadata ^
::     --model A:\ml_models\fastDAT\2x_animefilm_light_161k.safetensors ^
::     --name "2x AnimeFilm FDAT" ^
::     --overwrite


python -m scripts.inject_metadata ^
    --model A:\ml_models\fastDAT\2x_animefilm_light_161k_op21_fp16_static_720x540.onnx ^
    --name "2x AnimeFilm FDAT" ^
    --debug ^
    --verbose


